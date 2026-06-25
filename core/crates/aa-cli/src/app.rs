use aa_framework::{Application, ApplicationError, Runtime, Transition};
use client_evm::{
    ARBITRUM_USDC_TOKEN_ADDRESS, AnyIssuedRequest, AnyRequestId, BlockHash, ChainEndpoints,
    ChainKey, ClientEvent, ClientEvmError, ClientHead, ETHEREUM_USDC_TOKEN_ADDRESS, GraphEndpoints,
    MetadataCache, PoolRef, ProtocolPoolKey, PoolDataResult, PoolLog, PoolMetadataResult, RequestId,
    RpcConfig, TokenAddress, TokenMetadataResult, bootstrap, fetch_block_header, fetch_block_logs,
    fetch_finalized_block_header, fetch_pool_candidates_in_range, fetch_pool_data,
    fetch_pool_metadata, fetch_token_metadata, kernel,
    multi_chain_kernel::{
        Effect, Event, OptimizationPoolReserves, State, Subscription, SubscriptionData, transition,
    },
    subscribe_new_heads, subscribe_pool_events,
};
use optimization::{
    OptimizationBackendSelection, OptimizationSessionConfig, OptimizationStepConfig,
};
use std::{
    collections::{HashMap, HashSet},
    sync::{OnceLock, mpsc::Sender},
    thread::{self, JoinHandle},
    time,
};

use crate::{
    latest_slot::{LatestReceiveError, LatestReceiver, LatestSender, latest_slot},
    logger::Logger,
    optimization::{RunOptimizationError, run_optimization},
    view::View,
};

pub(crate) struct ClientEvmApp {}

pub(crate) struct ClientEvmRuntime {
    agent: ureq::Agent,
    rpc_config: RpcConfig,
    endpoints: ChainEndpoints,
    // Consumed in Increment 4 (the v4 metadata resolver); plumbed now so the config/endpoint wiring
    // lands in one reviewable change. Empty when The Graph is unconfigured.
    #[allow(dead_code)]
    graph_endpoints: GraphEndpoints,
    metadata_cache: MetadataCache,
    optimization_sender: OnceLock<LatestSender<OptimizationPoolReserves>>,
    view: View,
    logger: Logger,
}

impl ClientEvmRuntime {
    pub(crate) fn new(
        rpc_config: RpcConfig,
        endpoints: ChainEndpoints,
        graph_endpoints: GraphEndpoints,
        metadata_cache: MetadataCache,
        logger: Logger,
        view: View,
    ) -> ClientEvmRuntime {
        ClientEvmRuntime {
            agent: ureq::Agent::new_with_defaults(),
            rpc_config,
            endpoints,
            graph_endpoints,
            metadata_cache,
            optimization_sender: OnceLock::new(),
            view,
            logger,
        }
    }

    /// dRPC/websocket config used by the subscription channel (still single-provider).
    fn rpc_config(&self) -> &RpcConfig {
        &self.rpc_config
    }

    /// Per-chain HTTP endpoint pools used by every RPC fetch (multi-provider, with failover).
    fn endpoints(&self) -> &ChainEndpoints {
        &self.endpoints
    }

    /// Per-chain Uniswap v4 subgraph pools (empty when unconfigured). Consumed by the v4 metadata
    /// resolver in Increment 4; exposed here for the constructor smoke test until then.
    #[cfg(test)]
    fn graph_endpoints(&self) -> &GraphEndpoints {
        &self.graph_endpoints
    }

    /// Resolves pool metadata through the persistent cache: known pools are served from disk, only
    /// the misses hit RPC, and freshly validated metadata is written back. A cache fault degrades to
    /// a plain RPC fetch, so the cache can never make a request fail that would otherwise succeed.
    fn cached_pool_metadata(
        &self,
        chain: ChainKey,
        at: BlockHash,
        candidates: HashSet<ProtocolPoolKey>,
    ) -> Result<HashMap<ProtocolPoolKey, PoolMetadataResult>, ClientEvmError> {
        let cached = match self.metadata_cache.load_pool_metadata(chain, &candidates) {
            Ok(cached) => cached,
            Err(error) => {
                self.logger.log(&format!(
                    "error chain={chain:?} metadata_cache_load_failed kind=pool error={error}"
                ));
                HashMap::new()
            }
        };

        let misses = candidates
            .into_iter()
            .filter(|candidate| !cached.contains_key(candidate))
            .collect::<HashSet<_>>();

        let mut metadata = fetch_pool_metadata(&self.agent, self.endpoints(), chain, at, misses)?;

        if let Err(error) = self.metadata_cache.store_pool_metadata(chain, &metadata) {
            self.logger.log(&format!(
                "error chain={chain:?} metadata_cache_store_failed kind=pool error={error}"
            ));
        }

        metadata.extend(
            cached
                .into_iter()
                .map(|(candidate, value)| (candidate, Ok(value))),
        );

        Ok(metadata)
    }

    /// Token-metadata counterpart of [`ClientEvmRuntime::cached_pool_metadata`].
    fn cached_token_metadata(
        &self,
        chain: ChainKey,
        at: BlockHash,
        tokens: HashSet<TokenAddress>,
    ) -> Result<HashMap<TokenAddress, TokenMetadataResult>, ClientEvmError> {
        let cached = match self.metadata_cache.load_token_metadata(&tokens) {
            Ok(cached) => cached,
            Err(error) => {
                self.logger.log(&format!(
                    "error chain={chain:?} metadata_cache_load_failed kind=token error={error}"
                ));
                HashMap::new()
            }
        };

        let misses = tokens
            .into_iter()
            .filter(|token| !cached.contains_key(token))
            .collect::<HashSet<_>>();

        let mut metadata = fetch_token_metadata(&self.agent, self.endpoints(), chain, at, misses)?;

        if let Err(error) = self.metadata_cache.store_token_metadata(&metadata) {
            self.logger.log(&format!(
                "error chain={chain:?} metadata_cache_store_failed kind=token error={error}"
            ));
        }

        metadata.extend(cached.into_iter().map(|(token, value)| (token, Ok(value))));

        Ok(metadata)
    }
}

impl Application for ClientEvmApp {
    type State = State;
    type Input = Event;
    type Effect = Effect;
    type Subscription = Subscription;

    fn init() -> Transition<Self::State, Self::Effect> {
        let (state, effects) = State::init(client_evm::ACTIVE_CHAINS);
        Transition { state, effects }
    }

    fn transition(state: Self::State, input: Self::Input) -> Transition<Self::State, Self::Effect> {
        let (new_state, effects) = transition(state, input);
        Transition {
            state: new_state,
            effects,
        }
    }

    fn subscriptions() -> Vec<Self::Subscription> {
        let mut subscriptions: Vec<Subscription> = client_evm::ACTIVE_CHAINS
            .iter()
            .flat_map(|&chain| {
                [
                    Subscription::NewHeadsSubscription(chain),
                    Subscription::PoolEventsSubscription(chain),
                ]
            })
            .collect();
        subscriptions.push(Subscription::TickSubscription(time::Duration::from_millis(
            1000,
        )));
        subscriptions.push(Subscription::OptimizationSubscription);
        subscriptions
    }
}

impl Runtime<ClientEvmApp> for ClientEvmRuntime {
    fn execute_effect(
        &self,
        effect: <ClientEvmApp as Application>::Effect,
    ) -> Vec<<ClientEvmApp as Application>::Input> {
        match effect {
            Effect::FetchFinalizedHeader { chain } => {
                let r = fetch_finalized_block_header(&self.agent, self.endpoints(), chain);
                match r {
                    Ok(Some(header)) => vec![Event::FinalizedHeaderReceived {
                        chain,
                        block_hash: header.inner.hash,
                    }],
                    Ok(None) => vec![Event::FinalizedHeaderUnavailable { chain }],
                    Err(_) => vec![Event::FinalizedHeaderUnavailable { chain }],
                }
            }
            Effect::ChainEffect { chain, effect } => self.execute_chain_effect(chain, effect),
            Effect::BootstrapEffect { chain, effect } => {
                self.execute_bootstrap_effect(chain, effect)
            }
            Effect::RunOptimization { input } => {
                self.logger.log(&format_run_optimization_effect_log(&input));
                self.send_optimization_input(input);
                Vec::new()
            }
        }
    }

    fn spawn_subscription(
        &self,
        sender: &Sender<<ClientEvmApp as Application>::Input>,
        subscription: <ClientEvmApp as Application>::Subscription,
    ) {
        match subscription {
            Subscription::NewHeadsSubscription(chain) => {
                let map_client_event = |client_event: ClientEvent| {
                    map_client_subscription_data(client_event)
                        .map(|data| Event::SubscriptionData { chain, data })
                };
                let _ = subscribe_new_heads(self.rpc_config(), chain, sender, map_client_event);
            }
            Subscription::PoolEventsSubscription(chain) => {
                let map_client_event = |client_event: ClientEvent| {
                    map_client_subscription_data(client_event)
                        .map(|data| Event::SubscriptionData { chain, data })
                };
                let _ = subscribe_pool_events(self.rpc_config(), chain, sender, map_client_event);
            }
            Subscription::TickSubscription(interval) => {
                drop(spawn_tick_subscription(sender.clone(), interval));
            }
            Subscription::OptimizationSubscription => {
                let (slot_sender, slot_receiver) = latest_slot();

                if self.optimization_sender.set(slot_sender).is_ok() {
                    drop(spawn_optimization_subscription(
                        slot_receiver,
                        sender.clone(),
                        self.logger.clone(),
                    ));
                } else {
                    self.logger
                        .log("error optimization_subscription_already_started");
                }
            }
        }
    }

    fn log_input(&self, input: &<ClientEvmApp as Application>::Input) {
        self.logger.log(&format_input_log(input));
    }

    fn log_error(&self, error: ApplicationError<<ClientEvmApp as Application>::Input>) {
        match error {
            ApplicationError::SendError(error) => {
                self.logger.log(&format!(
                    "error send_failed input={}",
                    format_input_log(&error.0)
                ));
            }
        }
    }

    fn observe_state(&self, state: &<ClientEvmApp as Application>::State) {
        self.view.render(state);
    }
}

impl ClientEvmRuntime {
    fn send_optimization_input(&self, input: OptimizationPoolReserves) {
        if input.reserves.is_empty() {
            return;
        }

        match self.optimization_sender.get() {
            Some(sender) => {
                if let Err(error) = sender.send(input) {
                    self.logger
                        .log(&format!("error optimization_send_failed error={error:?}"));
                }
            }
            None => self.logger.log("error optimization_sender_uninitialized"),
        }
    }
}

pub(crate) fn start_runtime(
    config: RpcConfig,
    endpoints: ChainEndpoints,
    graph_endpoints: GraphEndpoints,
    metadata_cache: MetadataCache,
    logger: Logger,
    view: View,
) -> JoinHandle<()> {
    let (_sender, handle) = <ClientEvmRuntime as Runtime<ClientEvmApp>>::run(
        ClientEvmRuntime::new(config, endpoints, graph_endpoints, metadata_cache, logger, view),
    );

    handle
}

fn format_input_log(input: &Event) -> String {
    match input {
        Event::FinalizedHeaderReceived { chain, block_hash } => {
            format!("input finalized_header_received chain={chain:?} block={block_hash}")
        }
        Event::FinalizedHeaderUnavailable { chain } => {
            format!("input finalized_header_unavailable chain={chain:?}")
        }
        Event::SubscriptionData { chain, data } => format_subscription_data_log(*chain, data),
        Event::ChainEvent { chain, event } => format_chain_event_log(*chain, event),
        Event::BootstrapEvent { chain, event } => format_bootstrap_event_log(*chain, event),
        Event::OptimizationStepCompleted { result } => format!(
            "input optimization_step_completed status={:?} profit={} reserves={} iterations={}",
            result.status, result.profit_amount, result.reserves_count, result.iterations_completed,
        ),
        Event::Tick => "input tick".to_owned(),
    }
}

fn format_run_optimization_effect_log(input: &OptimizationPoolReserves) -> String {
    let blocks = input
        .block_hashes
        .iter()
        .map(|(chain, hash)| format!("{chain:?}:{hash}"))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "effect run_optimization blocks={blocks} reserves={}",
        input.reserves.len()
    )
}

fn format_subscription_data_log(chain: ChainKey, data: &SubscriptionData) -> String {
    match data {
        SubscriptionData::NewHead {
            hash, parent_hash, ..
        } => format!("input chain={chain:?} head_observed hash={hash} parent={parent_hash}"),
        SubscriptionData::PoolLog { block_hash, .. } => {
            format!("input chain={chain:?} log_observed block={block_hash} pools=1")
        }
    }
}

fn format_chain_event_log(chain: ChainKey, event: &kernel::Event) -> String {
    match event {
        kernel::Event::HeadObserved {
            hash, parent_hash, ..
        } => {
            format!("input chain={chain:?} head_observed hash={hash} parent={parent_hash}")
        }
        kernel::Event::FinalizedBlockObserved { block_hash } => {
            format!("input chain={chain:?} finalized_block_observed hash={block_hash}")
        }
        kernel::Event::BlockHeaderReceived {
            request_id,
            hash,
            parent_hash,
            ..
        } => format!(
            "input chain={chain:?} block_header_received request={} hash={hash} parent={parent_hash}",
            format_typed_request_id_log(request_id),
        ),
        kernel::Event::BlockHeaderNotFound { request_id } => format!(
            "input chain={chain:?} block_header_not_found request={}",
            format_typed_request_id_log(request_id),
        ),
        kernel::Event::BlockLogsReceived { request_id, logs } => format!(
            "input chain={chain:?} block_logs_received request={} pools={}",
            format_typed_request_id_log(request_id),
            logs.len(),
        ),
        kernel::Event::LogObserved { block_hash, logs } => format!(
            "input chain={chain:?} log_observed block={block_hash} pools={}",
            logs.len(),
        ),
        kernel::Event::PoolDataReceived { request_id, pools } => format!(
            "input chain={chain:?} pool_data_received request={} pools={}",
            format_typed_request_id_log(request_id),
            pools.len(),
        ),
        kernel::Event::PoolMetadataReceived {
            request_id,
            metadata,
        } => format!(
            "input chain={chain:?} pool_metadata_received request={} candidates={}",
            format_typed_request_id_log(request_id),
            metadata.len(),
        ),
        kernel::Event::TokenMetadataReceived {
            request_id,
            metadata,
        } => format!(
            "input chain={chain:?} token_metadata_received request={} tokens={}",
            format_typed_request_id_log(request_id),
            metadata.len(),
        ),
        kernel::Event::RequestFailed { request_id } => format!(
            "input chain={chain:?} request_failed request={}",
            format_request_id_log(request_id),
        ),
        kernel::Event::Tick => format!("input chain={chain:?} tick"),
    }
}

fn format_bootstrap_event_log(chain: ChainKey, event: &bootstrap::Event) -> String {
    match event {
        bootstrap::Event::FinalizedHeaderReceived { anchor, .. } => format!(
            "input chain={chain:?} bootstrap_finalized_header_received hash={} number={}",
            anchor.hash, anchor.number,
        ),
        bootstrap::Event::PoolCandidatesReceived { blocks, .. } => format!(
            "input chain={chain:?} bootstrap_pool_candidates_received blocks={}",
            blocks.len(),
        ),
        bootstrap::Event::PoolMetadataReceived { metadata, .. } => format!(
            "input chain={chain:?} bootstrap_pool_metadata_received candidates={}",
            metadata.len(),
        ),
        bootstrap::Event::TokenMetadataReceived { metadata, .. } => format!(
            "input chain={chain:?} bootstrap_token_metadata_received tokens={}",
            metadata.len(),
        ),
        bootstrap::Event::RequestFailed { request_id } => {
            format!("input chain={chain:?} bootstrap_request_failed request={request_id:?}")
        }
        bootstrap::Event::Tick => format!("input chain={chain:?} bootstrap_tick"),
    }
}

fn format_typed_request_id_log<R>(request_id: &RequestId<R>) -> String {
    format!("{request_id:?}")
}

fn format_request_id_log(request_id: &AnyRequestId) -> String {
    format!("{request_id:?}")
}

impl ClientEvmRuntime {
    fn execute_bootstrap_effect(&self, chain: ChainKey, effect: bootstrap::Effect) -> Vec<Event> {
        let endpoints = self.endpoints();
        let bootstrap::Effect::Request(request) = effect;

        let event = match request {
            bootstrap::AnyIssuedRequest::FinalizedHeader(request) => {
                let request_id = request.request_id;

                match fetch_finalized_block_header(&self.agent, endpoints, chain) {
                    Ok(Some(header)) => bootstrap::Event::FinalizedHeaderReceived {
                        request_id,
                        anchor: bootstrap::FinalizedAnchor {
                            hash: header.inner.hash,
                            number: header.inner.inner.number,
                        },
                    },
                    Ok(None) => bootstrap::Event::RequestFailed {
                        request_id: bootstrap::AnyRequestId::FinalizedHeader(request_id),
                    },
                    Err(error) => {
                        let request_id = bootstrap::AnyRequestId::FinalizedHeader(request_id);
                        self.logger.log(&format!(
                            "error chain={chain:?} bootstrap_request_failed request={request_id:?} error={error}"
                        ));
                        bootstrap::Event::RequestFailed { request_id }
                    }
                }
            }
            bootstrap::AnyIssuedRequest::PoolCandidates(request) => {
                let request_id = request.request_id;
                let from_block = request.request_payload.from_block;

                match fetch_pool_candidates_in_range(&self.agent, endpoints, chain, from_block) {
                    Ok(blocks) => bootstrap::Event::PoolCandidatesReceived { request_id, blocks },
                    Err(error) => {
                        let request_id = bootstrap::AnyRequestId::PoolCandidates(request_id);
                        self.logger.log(&format!(
                            "error chain={chain:?} bootstrap_request_failed request={request_id:?} error={error}"
                        ));
                        bootstrap::Event::RequestFailed { request_id }
                    }
                }
            }
            bootstrap::AnyIssuedRequest::PoolMetadata(request) => {
                let request_id = request.request_id;
                let at = request.request_payload.at;
                let mut candidates = request.request_payload.candidates;

                // Widen the finalized..tip discovery with the full known pool set from the cache so a
                // narrowed scan still re-activates every previously-validated pool: cached addresses
                // resolve as hits in `cached_pool_metadata` (no RPC); only genuinely new pools are
                // fetched. A cache fault degrades to the scan-only set rather than failing bootstrap.
                // This is bootstrap-only — the live closures stay scoped to their specific candidate.
                match self.metadata_cache.load_pool_candidates(chain) {
                    Ok(known) => candidates.extend(known),
                    Err(error) => self.logger.log(&format!(
                        "error chain={chain:?} metadata_cache_load_failed kind=pool_addresses error={error}"
                    )),
                }

                match self.cached_pool_metadata(chain, at, candidates) {
                    Ok(metadata) => bootstrap::Event::PoolMetadataReceived {
                        request_id,
                        metadata,
                    },
                    Err(error) => {
                        let request_id = bootstrap::AnyRequestId::PoolMetadata(request_id);
                        self.logger.log(&format!(
                            "error chain={chain:?} bootstrap_request_failed request={request_id:?} error={error}"
                        ));
                        bootstrap::Event::RequestFailed { request_id }
                    }
                }
            }
            bootstrap::AnyIssuedRequest::TokenMetadata(request) => {
                let request_id = request.request_id;
                let at = request.request_payload.at;
                let tokens = request.request_payload.tokens;

                match self.cached_token_metadata(chain, at, tokens) {
                    Ok(metadata) => bootstrap::Event::TokenMetadataReceived {
                        request_id,
                        metadata,
                    },
                    Err(error) => {
                        let request_id = bootstrap::AnyRequestId::TokenMetadata(request_id);
                        self.logger.log(&format!(
                            "error chain={chain:?} bootstrap_request_failed request={request_id:?} error={error}"
                        ));
                        bootstrap::Event::RequestFailed { request_id }
                    }
                }
            }
        };

        vec![Event::BootstrapEvent { chain, event }]
    }

    fn execute_chain_effect(&self, chain: ChainKey, effect: kernel::Effect) -> Vec<Event> {
        execute_chain_effect_with(
            chain,
            effect,
            &self.logger,
            |block_hash| fetch_block_header(&self.agent, self.endpoints(), chain, block_hash),
            |block_hash| fetch_block_logs(&self.agent, self.endpoints(), chain, block_hash),
            |at, pools| fetch_pool_data(&self.agent, self.endpoints(), chain, at, pools),
            |at, candidates| self.cached_pool_metadata(chain, at, candidates),
            |at, tokens| self.cached_token_metadata(chain, at, tokens),
        )
    }
}

fn map_client_subscription_data(client_chain_event: ClientEvent) -> Option<SubscriptionData> {
    match client_chain_event {
        ClientEvent::NewHead { header, .. } => Some(SubscriptionData::NewHead {
            hash: header.inner.hash,
            parent_hash: header.inner.inner.parent_hash,
            logs_bloom: header.inner.inner.logs_bloom,
        }),
        ClientEvent::PoolLogObserved {
            block_hash, log, ..
        } => Some(SubscriptionData::PoolLog { block_hash, log }),
        ClientEvent::Subscribed { .. } => None,
        ClientEvent::Closed { .. } => None,
    }
}

fn spawn_tick_subscription(sender: Sender<Event>, interval: time::Duration) -> JoinHandle<()> {
    thread::spawn(move || {
        loop {
            thread::sleep(interval);

            if sender.send(Event::Tick).is_err() {
                break;
            }
        }
    })
}

fn spawn_optimization_subscription(
    receiver: LatestReceiver<OptimizationPoolReserves>,
    sender: Sender<Event>,
    logger: Logger,
) -> JoinHandle<()> {
    thread::spawn(move || {
        let result = run_optimization(
            receiver,
            default_optimization_backend(),
            default_optimization_session_config(),
            default_optimization_step_config(),
            sender,
            |result| Event::OptimizationStepCompleted { result },
        );

        match result {
            Ok(()) => {}
            Err(RunOptimizationError::Receive(LatestReceiveError::Closed)) => {}
            Err(error) => logger.log(&format!("error optimization_worker_failed error={error:?}")),
        }
    })
}

fn default_optimization_backend() -> OptimizationBackendSelection {
    OptimizationBackendSelection::Auto
}

fn default_optimization_session_config() -> OptimizationSessionConfig<TokenAddress> {
    OptimizationSessionConfig {
        init_asset: ETHEREUM_USDC_TOKEN_ADDRESS,
        bridges: default_optimization_bridges(),
    }
}

/// Synthetic 1:1 connection between each chain's USDC, treating them as fungible so the single
/// `init_asset` (Ethereum USDC) can traverse Arbitrum pools and close cross-chain cycles. Bridges are
/// directional in the optimizer, so both orderings are registered; the optimizer ignores a bridge
/// whose endpoints aren't yet present, so an Arbitrum entry is harmless before Arbitrum has reported.
fn default_optimization_bridges() -> HashSet<(TokenAddress, TokenAddress)> {
    HashSet::from([
        (ETHEREUM_USDC_TOKEN_ADDRESS, ARBITRUM_USDC_TOKEN_ADDRESS),
        (ARBITRUM_USDC_TOKEN_ADDRESS, ETHEREUM_USDC_TOKEN_ADDRESS),
    ])
}

fn default_optimization_step_config() -> OptimizationStepConfig {
    OptimizationStepConfig {
        input_amount: 1000.0,
        iterations: 10,
    }
}

fn execute_chain_effect_with<
    FetchBlockHeader,
    FetchBlockLogs,
    FetchPoolData,
    FetchPoolMetadata,
    FetchTokenMetadata,
>(
    chain: ChainKey,
    effect: kernel::Effect,
    logger: &Logger,
    fetch_block_header: FetchBlockHeader,
    fetch_block_logs: FetchBlockLogs,
    fetch_pool_data: FetchPoolData,
    fetch_pool_metadata: FetchPoolMetadata,
    fetch_token_metadata: FetchTokenMetadata,
) -> Vec<Event>
where
    FetchBlockHeader: FnOnce(BlockHash) -> Result<Option<ClientHead>, ClientEvmError>,
    FetchBlockLogs: FnOnce(BlockHash) -> Result<Vec<PoolLog>, ClientEvmError>,
    FetchPoolData: FnOnce(
        BlockHash,
        HashSet<PoolRef>,
    ) -> Result<HashMap<PoolRef, PoolDataResult>, ClientEvmError>,
    FetchPoolMetadata:
        FnOnce(
            BlockHash,
            HashSet<ProtocolPoolKey>,
        )
            -> Result<HashMap<ProtocolPoolKey, PoolMetadataResult>, ClientEvmError>,
    FetchTokenMetadata:
        FnOnce(
            BlockHash,
            HashSet<TokenAddress>,
        ) -> Result<HashMap<TokenAddress, TokenMetadataResult>, ClientEvmError>,
{
    match effect {
        kernel::Effect::Request(request) => match request {
            AnyIssuedRequest::BlockHeader(request) => {
                let request_id = request.request_id;
                let block_hash = request.request_payload.block_hash;

                match fetch_block_header(block_hash) {
                    Ok(Some(header)) => vec![Event::ChainEvent {
                        chain,
                        event: kernel::Event::BlockHeaderReceived {
                            request_id,
                            hash: header.inner.hash,
                            parent_hash: header.inner.inner.parent_hash,
                            logs_bloom: header.inner.inner.logs_bloom,
                        },
                    }],
                    Ok(None) => vec![Event::ChainEvent {
                        chain,
                        event: kernel::Event::BlockHeaderNotFound { request_id },
                    }],
                    Err(error) => {
                        let request_id = AnyRequestId::BlockHeader(request_id);
                        logger.log(&format!(
                            "error chain={chain:?} request_failed request={request_id:?} error={error}"
                        ));
                        vec![Event::ChainEvent {
                            chain,
                            event: kernel::Event::RequestFailed { request_id },
                        }]
                    }
                }
            }
            AnyIssuedRequest::BlockLogs(request) => {
                let request_id = request.request_id;
                let block_hash = request.request_payload.block_hash;

                match fetch_block_logs(block_hash) {
                    Ok(logs) => vec![Event::ChainEvent {
                        chain,
                        event: kernel::Event::BlockLogsReceived { request_id, logs },
                    }],
                    Err(error) => {
                        let request_id = AnyRequestId::BlockLogs(request_id);
                        logger.log(&format!(
                            "error chain={chain:?} request_failed request={request_id:?} error={error}"
                        ));
                        vec![Event::ChainEvent {
                            chain,
                            event: kernel::Event::RequestFailed { request_id },
                        }]
                    }
                }
            }
            AnyIssuedRequest::PoolData(request) => {
                let request_id = request.request_id;
                let at = request.request_payload.at;
                let pools = request.request_payload.pools;

                match fetch_pool_data(at, pools) {
                    Ok(pools) => vec![Event::ChainEvent {
                        chain,
                        event: kernel::Event::PoolDataReceived { request_id, pools },
                    }],
                    Err(error) => {
                        let request_id = AnyRequestId::PoolData(request_id);
                        logger.log(&format!(
                            "error chain={chain:?} request_failed request={request_id:?} error={error}"
                        ));
                        vec![Event::ChainEvent {
                            chain,
                            event: kernel::Event::RequestFailed { request_id },
                        }]
                    }
                }
            }
            AnyIssuedRequest::PoolMetadata(request) => {
                let request_id = request.request_id;
                let at = request.request_payload.at;
                let candidates = request.request_payload.candidates;

                match fetch_pool_metadata(at, candidates) {
                    Ok(metadata) => vec![Event::ChainEvent {
                        chain,
                        event: kernel::Event::PoolMetadataReceived {
                            request_id,
                            metadata,
                        },
                    }],
                    Err(error) => {
                        let request_id = AnyRequestId::PoolMetadata(request_id);
                        logger.log(&format!(
                            "error chain={chain:?} request_failed request={request_id:?} error={error}"
                        ));
                        vec![Event::ChainEvent {
                            chain,
                            event: kernel::Event::RequestFailed { request_id },
                        }]
                    }
                }
            }
            AnyIssuedRequest::TokenMetadata(request) => {
                let request_id = request.request_id;
                let at = request.request_payload.at;
                let tokens = request.request_payload.tokens;

                match fetch_token_metadata(at, tokens) {
                    Ok(metadata) => vec![Event::ChainEvent {
                        chain,
                        event: kernel::Event::TokenMetadataReceived {
                            request_id,
                            metadata,
                        },
                    }],
                    Err(error) => {
                        let request_id = AnyRequestId::TokenMetadata(request_id);
                        logger.log(&format!(
                            "error chain={chain:?} request_failed request={request_id:?} error={error}"
                        ));
                        vec![Event::ChainEvent {
                            chain,
                            event: kernel::Event::RequestFailed { request_id },
                        }]
                    }
                }
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use client_evm::{
        Bloom, ConfigScope, GetBlockHeader, GetBlockLogs, GetPoolData, GetPoolMetadata,
        GetTokenMetadata, IssuedRequest, PoolRef, ProtocolPoolKey, PoolDataResult,
        PoolFee, PoolLog, PoolMetadata, PoolMetadataResult, RequestId, TokenAddress,
        TokenMetadataResult, UniswapV3Fee,
    };
    use serde_json::json;
    use std::{sync::mpsc, time::Duration};

    use super::*;

    #[test]
    fn runtime_constructor_stores_rpc_config() {
        let config = rpc_config();
        let runtime = ClientEvmRuntime::new(
            config.clone(),
            test_endpoints(),
            GraphEndpoints::empty(),
            in_memory_metadata_cache(),
            Logger::sink(),
            View::sink(),
        );

        assert_eq!(runtime.rpc_config(), &config);
    }

    #[test]
    fn runtime_constructor_stores_graph_endpoints() {
        let runtime = ClientEvmRuntime::new(
            rpc_config(),
            test_endpoints(),
            GraphEndpoints::single(ChainKey::Ethereum, "thegraph", "http://graph.invalid"),
            in_memory_metadata_cache(),
            Logger::sink(),
            View::sink(),
        );

        assert!(runtime.graph_endpoints().pool(ChainKey::Ethereum).is_some());
        assert!(runtime.graph_endpoints().pool(ChainKey::Arbitrum).is_none());
    }

    #[test]
    fn subscriptions_include_optimization_worker() {
        let subscriptions = ClientEvmApp::subscriptions();

        assert!(subscriptions.iter().any(|subscription| {
            matches!(
                subscription,
                Subscription::NewHeadsSubscription(ChainKey::Ethereum)
            )
        }));
        assert!(
            subscriptions
                .iter()
                .any(|subscription| matches!(subscription, Subscription::TickSubscription(_)))
        );
        assert!(
            subscriptions
                .iter()
                .any(|subscription| matches!(subscription, Subscription::OptimizationSubscription))
        );
    }

    #[test]
    fn subscriptions_open_new_heads_for_every_active_chain() {
        let subscriptions = ClientEvmApp::subscriptions();

        let new_heads_chains = subscriptions
            .iter()
            .filter_map(|subscription| match subscription {
                Subscription::NewHeadsSubscription(chain) => Some(*chain),
                _ => None,
            })
            .collect::<Vec<_>>();

        assert_eq!(new_heads_chains.len(), client_evm::ACTIVE_CHAINS.len());
        for chain in client_evm::ACTIVE_CHAINS {
            assert!(
                new_heads_chains.contains(chain),
                "expected a new-heads subscription for active chain {chain:?}"
            );
        }
    }

    #[test]
    fn subscriptions_open_pool_events_for_every_active_chain() {
        let subscriptions = ClientEvmApp::subscriptions();

        let pool_event_chains = subscriptions
            .iter()
            .filter_map(|subscription| match subscription {
                Subscription::PoolEventsSubscription(chain) => Some(*chain),
                _ => None,
            })
            .collect::<Vec<_>>();

        assert_eq!(pool_event_chains.len(), client_evm::ACTIVE_CHAINS.len());
        for chain in client_evm::ACTIVE_CHAINS {
            assert!(
                pool_event_chains.contains(chain),
                "expected a pool-events subscription for active chain {chain:?}"
            );
        }
    }

    #[test]
    fn runtime_starts_with_uninitialized_optimization_sender() {
        let runtime = ClientEvmRuntime::new(
            rpc_config(),
            test_endpoints(),
            GraphEndpoints::empty(),
            in_memory_metadata_cache(),
            Logger::sink(),
            View::sink(),
        );

        assert!(runtime.optimization_sender.get().is_none());
    }

    #[test]
    fn input_log_formats_global_multi_chain_events() {
        let block_hash = hash(1);

        assert_eq!(format_input_log(&Event::Tick), "input tick");
        assert_eq!(
            format_input_log(&Event::FinalizedHeaderReceived {
                chain: ChainKey::Ethereum,
                block_hash,
            }),
            format!("input finalized_header_received chain=Ethereum block={block_hash}")
        );
        assert_eq!(
            format_input_log(&Event::FinalizedHeaderUnavailable {
                chain: ChainKey::Ethereum,
            }),
            "input finalized_header_unavailable chain=Ethereum"
        );
    }

    #[test]
    fn input_log_formats_optimization_step_completed() {
        let result = optimization::OptimizationStepResult {
            status: optimization::OptimizationStepStatus::Updated,
            input_amount: 1_000.0,
            output_amount: 1_012.5,
            profit_amount: 12.5,
            reserves_count: 4,
            iterations_completed: 10,
        };

        assert_eq!(
            format_input_log(&Event::OptimizationStepCompleted { result }),
            "input optimization_step_completed status=Updated profit=12.5 reserves=4 iterations=10"
        );
    }

    #[test]
    fn input_log_formats_chain_events() {
        let block_hash = hash(1);
        let parent_hash = hash(2);
        let request_id = RequestId::<GetBlockHeader>::from_raw_for_test(7);

        assert_eq!(
            format_input_log(&Event::ChainEvent {
                chain: ChainKey::Ethereum,
                event: kernel::Event::HeadObserved {
                    hash: block_hash,
                    parent_hash,
                    logs_bloom: Bloom::default(),
                },
            }),
            format!("input chain=Ethereum head_observed hash={block_hash} parent={parent_hash}")
        );
        assert_eq!(
            format_input_log(&Event::ChainEvent {
                chain: ChainKey::Ethereum,
                event: kernel::Event::BlockHeaderReceived {
                    request_id,
                    hash: block_hash,
                    parent_hash,
                    logs_bloom: Bloom::default(),
                },
            }),
            format!(
                "input chain=Ethereum block_header_received request=7 hash={block_hash} parent={parent_hash}"
            )
        );
        assert_eq!(
            format_input_log(&Event::ChainEvent {
                chain: ChainKey::Ethereum,
                event: kernel::Event::Tick,
            }),
            "input chain=Ethereum tick"
        );
    }

    #[test]
    fn input_log_formats_request_result_counts() {
        let logs_request_id = RequestId::<GetBlockLogs>::from_raw_for_test(8);
        let pool_request_id = RequestId::<GetPoolData>::from_raw_for_test(9);
        let metadata_request_id = RequestId::<GetPoolMetadata>::from_raw_for_test(10);
        let token_metadata_request_id = RequestId::<GetTokenMetadata>::from_raw_for_test(11);

        assert_eq!(
            format_input_log(&Event::ChainEvent {
                chain: ChainKey::Ethereum,
                event: kernel::Event::BlockLogsReceived {
                    request_id: logs_request_id,
                    logs: Vec::new(),
                },
            }),
            "input chain=Ethereum block_logs_received request=8 pools=0"
        );
        assert_eq!(
            format_input_log(&Event::ChainEvent {
                chain: ChainKey::Ethereum,
                event: kernel::Event::PoolDataReceived {
                    request_id: pool_request_id,
                    pools: HashMap::new(),
                },
            }),
            "input chain=Ethereum pool_data_received request=9 pools=0"
        );
        assert_eq!(
            format_input_log(&Event::ChainEvent {
                chain: ChainKey::Ethereum,
                event: kernel::Event::PoolMetadataReceived {
                    request_id: metadata_request_id,
                    metadata: HashMap::new(),
                },
            }),
            "input chain=Ethereum pool_metadata_received request=10 candidates=0"
        );
        assert_eq!(
            format_input_log(&Event::ChainEvent {
                chain: ChainKey::Ethereum,
                event: kernel::Event::TokenMetadataReceived {
                    request_id: token_metadata_request_id,
                    metadata: HashMap::new(),
                },
            }),
            "input chain=Ethereum token_metadata_received request=11 tokens=0"
        );
    }

    #[test]
    fn input_log_formats_request_failures_and_not_found() {
        let header_request_id = RequestId::<GetBlockHeader>::from_raw_for_test(7);

        assert_eq!(
            format_input_log(&Event::ChainEvent {
                chain: ChainKey::Ethereum,
                event: kernel::Event::BlockHeaderNotFound {
                    request_id: header_request_id,
                },
            }),
            "input chain=Ethereum block_header_not_found request=7"
        );
        assert_eq!(
            format_input_log(&Event::ChainEvent {
                chain: ChainKey::Ethereum,
                event: kernel::Event::RequestFailed {
                    request_id: AnyRequestId::BlockHeader(header_request_id),
                },
            }),
            "input chain=Ethereum request_failed request=block_header#7"
        );
    }

    #[test]
    fn request_id_log_formats_request_kind_and_id() {
        assert_eq!(
            format_request_id_log(&AnyRequestId::BlockHeader(
                RequestId::<GetBlockHeader>::from_raw_for_test(7),
            )),
            "block_header#7"
        );
        assert_eq!(
            format_request_id_log(&AnyRequestId::BlockLogs(
                RequestId::<GetBlockLogs>::from_raw_for_test(8),
            )),
            "block_logs#8"
        );
        assert_eq!(
            format_request_id_log(&AnyRequestId::PoolData(
                RequestId::<GetPoolData>::from_raw_for_test(9),
            )),
            "pool_data#9"
        );
        assert_eq!(
            format_request_id_log(&AnyRequestId::PoolMetadata(
                RequestId::<GetPoolMetadata>::from_raw_for_test(10),
            )),
            "pool_metadata#10"
        );
    }

    #[test]
    fn optimization_effect_log_formats_block_and_reserve_count() {
        let input = optimization_input(hash(7));

        assert_eq!(
            format_run_optimization_effect_log(&input),
            format!(
                "effect run_optimization blocks=Ethereum:{} reserves=0",
                hash(7)
            )
        );
    }

    #[test]
    fn bootstrap_pool_metadata_unions_the_cached_pool_set_as_hits_without_rpc() {
        // Prime the cache with a pool that is absent from the (narrowed) discovery scan.
        let cache = in_memory_metadata_cache();
        let cached = pool_candidate_address(7);
        cache
            .store_pool_metadata(
                ChainKey::Ethereum,
                &HashMap::from([(cached, Ok(pool_metadata(7)))]),
            )
            .expect("store cached pool");

        let runtime = ClientEvmRuntime::new(
            rpc_config(),
            test_endpoints(),
            GraphEndpoints::empty(),
            cache,
            Logger::sink(),
            View::sink(),
        );

        // The scan yielded no candidates; the union must still surface the cached pool, resolved as
        // a hit. `rpc_config()` points nowhere, so any RPC (a cache miss) would error/panic instead.
        let request = bootstrap::AnyIssuedRequest::PoolMetadata(IssuedRequest {
            request_id: RequestId::from_raw_for_test(1),
            request_payload: GetPoolMetadata {
                at: hash(1),
                candidates: HashSet::new(),
            },
        });
        let events = runtime
            .execute_bootstrap_effect(ChainKey::Ethereum, bootstrap::Effect::Request(request));

        let event = match events.as_slice() {
            [event] => event,
            _ => panic!("expected exactly one bootstrap event"),
        };
        let metadata = match event {
            Event::BootstrapEvent {
                event: bootstrap::Event::PoolMetadataReceived { metadata, .. },
                ..
            } => metadata,
            _ => panic!("expected a PoolMetadataReceived bootstrap event"),
        };
        assert!(metadata.get(&cached).is_some_and(|result| result.is_ok()));
    }

    #[test]
    fn run_optimization_effect_returns_no_events() {
        let runtime = ClientEvmRuntime::new(
            rpc_config(),
            test_endpoints(),
            GraphEndpoints::empty(),
            in_memory_metadata_cache(),
            Logger::sink(),
            View::sink(),
        );
        let events = runtime.execute_effect(Effect::RunOptimization {
            input: optimization_input(hash(7)),
        });

        assert!(events.is_empty());
    }

    #[test]
    fn default_session_config_bridges_the_chain_usdc_quote_tokens() {
        let config = default_optimization_session_config();

        // Ethereum USDC is the single global quote/init asset; the bidirectional 1:1 bridge to
        // Arbitrum USDC lets the solver close cross-chain cycles back to it.
        assert_eq!(config.init_asset, ETHEREUM_USDC_TOKEN_ADDRESS);
        assert!(
            config
                .bridges
                .contains(&(ETHEREUM_USDC_TOKEN_ADDRESS, ARBITRUM_USDC_TOKEN_ADDRESS))
        );
        assert!(
            config
                .bridges
                .contains(&(ARBITRUM_USDC_TOKEN_ADDRESS, ETHEREUM_USDC_TOKEN_ADDRESS))
        );
    }

    #[test]
    fn empty_optimization_snapshot_is_dropped() {
        let runtime = ClientEvmRuntime::new(
            rpc_config(),
            test_endpoints(),
            GraphEndpoints::empty(),
            in_memory_metadata_cache(),
            Logger::sink(),
            View::sink(),
        );
        let (slot_sender, slot_receiver) = crate::latest_slot::latest_slot();
        assert!(runtime.optimization_sender.set(slot_sender).is_ok());

        let events = runtime.execute_effect(Effect::RunOptimization {
            input: optimization_input(hash(7)),
        });

        assert!(events.is_empty());
        assert_eq!(slot_receiver.try_take(), Ok(None));
    }

    #[test]
    fn non_empty_optimization_snapshot_is_forwarded_when_sender_is_initialized() {
        let runtime = ClientEvmRuntime::new(
            rpc_config(),
            test_endpoints(),
            GraphEndpoints::empty(),
            in_memory_metadata_cache(),
            Logger::sink(),
            View::sink(),
        );
        let (slot_sender, slot_receiver) = crate::latest_slot::latest_slot();
        assert!(runtime.optimization_sender.set(slot_sender).is_ok());
        let input = optimization_input_with_reserves(hash(7));

        let events = runtime.execute_effect(Effect::RunOptimization {
            input: input.clone(),
        });

        assert!(events.is_empty());
        assert_eq!(slot_receiver.try_take(), Ok(Some(input)));
    }

    #[test]
    fn non_empty_optimization_snapshot_is_dropped_when_sender_is_uninitialized() {
        let runtime = ClientEvmRuntime::new(
            rpc_config(),
            test_endpoints(),
            GraphEndpoints::empty(),
            in_memory_metadata_cache(),
            Logger::sink(),
            View::sink(),
        );

        let events = runtime.execute_effect(Effect::RunOptimization {
            input: optimization_input_with_reserves(hash(7)),
        });

        assert!(events.is_empty());
        assert!(runtime.optimization_sender.get().is_none());
    }

    #[test]
    fn optimization_subscription_initializes_sender() {
        let runtime = ClientEvmRuntime::new(
            rpc_config(),
            test_endpoints(),
            GraphEndpoints::empty(),
            in_memory_metadata_cache(),
            Logger::sink(),
            View::sink(),
        );
        let (sender, _receiver) = mpsc::channel();

        runtime.spawn_subscription(&sender, Subscription::OptimizationSubscription);

        assert!(runtime.optimization_sender.get().is_some());
    }

    #[test]
    fn optimization_subscription_starts_only_once() {
        let runtime = ClientEvmRuntime::new(
            rpc_config(),
            test_endpoints(),
            GraphEndpoints::empty(),
            in_memory_metadata_cache(),
            Logger::sink(),
            View::sink(),
        );
        let (sender, _receiver) = mpsc::channel();

        runtime.spawn_subscription(&sender, Subscription::OptimizationSubscription);
        let first_sender = runtime
            .optimization_sender
            .get()
            .map(|sender| sender as *const _);
        runtime.spawn_subscription(&sender, Subscription::OptimizationSubscription);

        assert_eq!(
            runtime
                .optimization_sender
                .get()
                .map(|sender| sender as *const _),
            first_sender
        );
    }

    #[test]
    fn optimization_worker_exits_cleanly_when_slot_closes_before_initialization() {
        let (slot_sender, slot_receiver) = crate::latest_slot::latest_slot();
        let (sender, _receiver) = mpsc::channel();
        let handle = spawn_optimization_subscription(slot_receiver, sender, Logger::sink());

        drop(slot_sender);

        assert!(handle.join().is_ok());
    }

    #[test]
    fn tick_subscription_worker_sends_tick_event() {
        let (sender, receiver) = mpsc::channel();
        let handle = spawn_tick_subscription(sender, Duration::from_millis(1));

        assert!(matches!(
            receiver.recv_timeout(Duration::from_millis(250)),
            Ok(Event::Tick)
        ));

        drop(receiver);
        assert!(handle.join().is_ok());
    }

    #[test]
    fn tick_subscription_worker_exits_when_receiver_is_dropped() {
        let (sender, receiver) = mpsc::channel();
        drop(receiver);

        let handle = spawn_tick_subscription(sender, Duration::from_millis(1));

        assert!(handle.join().is_ok());
    }

    #[test]
    fn block_header_request_success_maps_to_chain_event() -> Result<(), serde_json::Error> {
        let chain = ChainKey::Ethereum;
        let requested_hash = hash(2);
        let parent_hash = hash(4);
        let (effect, expected_request_id) = block_header_request_effect(requested_hash);
        let header = block_header(requested_hash, parent_hash)?;

        let events = execute_chain_effect_with(
            chain,
            effect,
            &Logger::sink(),
            |block_hash| {
                assert_eq!(block_hash, requested_hash);
                Ok(Some(header))
            },
            unexpected_block_logs_fetch,
            unexpected_pool_data_fetch,
            unexpected_pool_metadata_fetch,
            unexpected_token_metadata_fetch,
        );

        assert!(matches!(
            events.as_slice(),
            [Event::ChainEvent {
                chain: event_chain,
                event:
                    kernel::Event::BlockHeaderReceived {
                        request_id,
                        hash: event_hash,
                        parent_hash: event_parent_hash,
                        ..
                    },
            }] if *event_chain == chain
                && *request_id == expected_request_id
                && *event_hash == requested_hash
                && *event_parent_hash == parent_hash
        ));

        Ok(())
    }

    #[test]
    fn block_header_request_not_found_maps_to_chain_event() {
        let chain = ChainKey::Ethereum;
        let requested_hash = hash(2);
        let (effect, expected_request_id) = block_header_request_effect(requested_hash);

        let events = execute_chain_effect_with(
            chain,
            effect,
            &Logger::sink(),
            |block_hash| {
                assert_eq!(block_hash, requested_hash);
                Ok(None)
            },
            unexpected_block_logs_fetch,
            unexpected_pool_data_fetch,
            unexpected_pool_metadata_fetch,
            unexpected_token_metadata_fetch,
        );

        assert!(matches!(
            events.as_slice(),
            [Event::ChainEvent {
                chain: event_chain,
                event: kernel::Event::BlockHeaderNotFound { request_id },
            }] if *event_chain == chain && *request_id == expected_request_id
        ));
    }

    #[test]
    fn block_header_request_failure_maps_to_request_failed_event() {
        let chain = ChainKey::Ethereum;
        let requested_hash = hash(2);
        let (effect, expected_request_id) = block_header_request_effect(requested_hash);

        let events = execute_chain_effect_with(
            chain,
            effect,
            &Logger::sink(),
            |block_hash| {
                assert_eq!(block_hash, requested_hash);
                Err(ClientEvmError::InvalidConfig {
                    scope: ConfigScope::Http,
                    reason: "bad config".to_owned(),
                })
            },
            unexpected_block_logs_fetch,
            unexpected_pool_data_fetch,
            unexpected_pool_metadata_fetch,
            unexpected_token_metadata_fetch,
        );

        assert!(matches!(
            events.as_slice(),
            [Event::ChainEvent {
                chain: event_chain,
                event:
                    kernel::Event::RequestFailed {
                        request_id: client_evm::AnyRequestId::BlockHeader(request_id),
                    },
            }] if *event_chain == chain && *request_id == expected_request_id
        ));
    }

    #[test]
    fn block_logs_request_success_maps_to_chain_event() {
        let chain = ChainKey::Ethereum;
        let block_hash = hash(2);
        let (effect, expected_request_id) = block_logs_request_effect(block_hash);

        let events = execute_chain_effect_with(
            chain,
            effect,
            &Logger::sink(),
            unexpected_block_header_fetch,
            |requested_hash| {
                assert_eq!(requested_hash, block_hash);
                Ok(Vec::new())
            },
            unexpected_pool_data_fetch,
            unexpected_pool_metadata_fetch,
            unexpected_token_metadata_fetch,
        );

        assert!(matches!(
            events.as_slice(),
            [Event::ChainEvent {
                chain: event_chain,
                event:
                    kernel::Event::BlockLogsReceived {
                        request_id,
                        logs,
                    },
            }] if *event_chain == chain
                && *request_id == expected_request_id
                && logs.is_empty()
        ));
    }

    #[test]
    fn block_logs_request_failure_maps_to_request_failed_event() {
        let chain = ChainKey::Ethereum;
        let block_hash = hash(2);
        let (effect, expected_request_id) = block_logs_request_effect(block_hash);

        let events = execute_chain_effect_with(
            chain,
            effect,
            &Logger::sink(),
            unexpected_block_header_fetch,
            |requested_hash| {
                assert_eq!(requested_hash, block_hash);
                Err(ClientEvmError::InvalidConfig {
                    scope: ConfigScope::Http,
                    reason: "bad config".to_owned(),
                })
            },
            unexpected_pool_data_fetch,
            unexpected_pool_metadata_fetch,
            unexpected_token_metadata_fetch,
        );

        assert!(matches!(
            events.as_slice(),
            [Event::ChainEvent {
                chain: event_chain,
                event:
                    kernel::Event::RequestFailed {
                        request_id: client_evm::AnyRequestId::BlockLogs(request_id),
                    },
            }] if *event_chain == chain && *request_id == expected_request_id
        ));
    }

    #[test]
    fn pool_data_request_success_maps_to_chain_event() {
        let chain = ChainKey::Ethereum;
        let at = hash(2);
        let (effect, expected_request_id) = pool_data_request_effect(at);

        let events = execute_chain_effect_with(
            chain,
            effect,
            &Logger::sink(),
            unexpected_block_header_fetch,
            unexpected_block_logs_fetch,
            |requested_at, requested_pools| {
                assert_eq!(requested_at, at);
                assert!(requested_pools.is_empty());
                Ok(HashMap::new())
            },
            unexpected_pool_metadata_fetch,
            unexpected_token_metadata_fetch,
        );

        assert!(matches!(
            events.as_slice(),
            [Event::ChainEvent {
                chain: event_chain,
                event:
                    kernel::Event::PoolDataReceived {
                        request_id,
                        pools,
                    },
            }] if *event_chain == chain
                && *request_id == expected_request_id
                && pools.is_empty()
        ));
    }

    #[test]
    fn pool_data_request_failure_maps_to_request_failed_event() {
        let chain = ChainKey::Ethereum;
        let at = hash(2);
        let (effect, expected_request_id) = pool_data_request_effect(at);

        let events = execute_chain_effect_with(
            chain,
            effect,
            &Logger::sink(),
            unexpected_block_header_fetch,
            unexpected_block_logs_fetch,
            |requested_at, requested_pools| {
                assert_eq!(requested_at, at);
                assert!(requested_pools.is_empty());
                Err(ClientEvmError::InvalidConfig {
                    scope: ConfigScope::Http,
                    reason: "bad config".to_owned(),
                })
            },
            unexpected_pool_metadata_fetch,
            unexpected_token_metadata_fetch,
        );

        assert!(matches!(
            events.as_slice(),
            [Event::ChainEvent {
                chain: event_chain,
                event:
                    kernel::Event::RequestFailed {
                        request_id: client_evm::AnyRequestId::PoolData(request_id),
                    },
            }] if *event_chain == chain && *request_id == expected_request_id
        ));
    }

    #[test]
    fn pool_metadata_request_success_maps_to_chain_event() {
        let chain = ChainKey::Ethereum;
        let at = hash(2);
        let candidate = pool_candidate_address(3);
        let (effect, expected_request_id) =
            pool_metadata_request_effect(at, HashSet::from([candidate]));

        let events = execute_chain_effect_with(
            chain,
            effect,
            &Logger::sink(),
            unexpected_block_header_fetch,
            unexpected_block_logs_fetch,
            unexpected_pool_data_fetch,
            |requested_at, requested_candidates| {
                assert_eq!(requested_at, at);
                assert_eq!(requested_candidates, HashSet::from([candidate]));
                Ok(HashMap::new())
            },
            unexpected_token_metadata_fetch,
        );

        assert!(matches!(
            events.as_slice(),
            [Event::ChainEvent {
                chain: event_chain,
                event:
                    kernel::Event::PoolMetadataReceived {
                        request_id,
                        metadata,
                    },
            }] if *event_chain == chain
                && *request_id == expected_request_id
                && metadata.is_empty()
        ));
    }

    #[test]
    fn pool_metadata_request_failure_maps_to_request_failed_event() {
        let chain = ChainKey::Ethereum;
        let at = hash(2);
        let candidates = HashSet::from([pool_candidate_address(3)]);
        let (effect, expected_request_id) = pool_metadata_request_effect(at, candidates.clone());

        let events = execute_chain_effect_with(
            chain,
            effect,
            &Logger::sink(),
            unexpected_block_header_fetch,
            unexpected_block_logs_fetch,
            unexpected_pool_data_fetch,
            |requested_at, requested_candidates| {
                assert_eq!(requested_at, at);
                assert_eq!(requested_candidates, candidates);
                Err(ClientEvmError::InvalidConfig {
                    scope: ConfigScope::Http,
                    reason: "bad config".to_owned(),
                })
            },
            unexpected_token_metadata_fetch,
        );

        assert!(matches!(
            events.as_slice(),
            [Event::ChainEvent {
                chain: event_chain,
                event:
                    kernel::Event::RequestFailed {
                        request_id: client_evm::AnyRequestId::PoolMetadata(request_id),
                    },
            }] if *event_chain == chain && *request_id == expected_request_id
        ));
    }

    #[test]
    fn token_metadata_request_success_maps_to_chain_event() {
        let chain = ChainKey::Ethereum;
        let at = hash(2);
        let token = token_address(3);
        let (effect, expected_request_id) =
            token_metadata_request_effect(at, HashSet::from([token]));

        let events = execute_chain_effect_with(
            chain,
            effect,
            &Logger::sink(),
            unexpected_block_header_fetch,
            unexpected_block_logs_fetch,
            unexpected_pool_data_fetch,
            unexpected_pool_metadata_fetch,
            |requested_at, requested_tokens| {
                assert_eq!(requested_at, at);
                assert_eq!(requested_tokens, HashSet::from([token]));
                Ok(HashMap::new())
            },
        );

        assert!(matches!(
            events.as_slice(),
            [Event::ChainEvent {
                chain: event_chain,
                event:
                    kernel::Event::TokenMetadataReceived {
                        request_id,
                        metadata,
                    },
            }] if *event_chain == chain
                && *request_id == expected_request_id
                && metadata.is_empty()
        ));
    }

    #[test]
    fn token_metadata_request_failure_maps_to_request_failed_event() {
        let chain = ChainKey::Ethereum;
        let at = hash(2);
        let tokens = HashSet::from([token_address(3)]);
        let (effect, expected_request_id) = token_metadata_request_effect(at, tokens.clone());

        let events = execute_chain_effect_with(
            chain,
            effect,
            &Logger::sink(),
            unexpected_block_header_fetch,
            unexpected_block_logs_fetch,
            unexpected_pool_data_fetch,
            unexpected_pool_metadata_fetch,
            |requested_at, requested_tokens| {
                assert_eq!(requested_at, at);
                assert_eq!(requested_tokens, tokens);
                Err(ClientEvmError::InvalidConfig {
                    scope: ConfigScope::Http,
                    reason: "bad config".to_owned(),
                })
            },
        );

        assert!(matches!(
            events.as_slice(),
            [Event::ChainEvent {
                chain: event_chain,
                event:
                    kernel::Event::RequestFailed {
                        request_id: client_evm::AnyRequestId::TokenMetadata(request_id),
                    },
            }] if *event_chain == chain && *request_id == expected_request_id
        ));
    }

    fn block_header_request_effect(
        block_hash: BlockHash,
    ) -> (kernel::Effect, RequestId<GetBlockHeader>) {
        let finalized_hash = hash(1);
        let observed_hash = hash(3);
        let state = kernel::State::init(kernel::FinalizedState::empty_at(finalized_hash));

        let (_state, effects) = kernel::transition(
            ChainKey::Ethereum,
            state,
            kernel::Event::HeadObserved {
                hash: observed_hash,
                parent_hash: block_hash,
                logs_bloom: Bloom::default(),
            },
        );

        let effect = effects
            .into_iter()
            .next()
            .expect("missing parent should request block header");
        let request_id = match &effect {
            kernel::Effect::Request(AnyIssuedRequest::BlockHeader(request)) => request.request_id,
            _ => panic!("expected block header request"),
        };

        (effect, request_id)
    }

    fn block_logs_request_effect(
        block_hash: BlockHash,
    ) -> (kernel::Effect, RequestId<GetBlockLogs>) {
        let request_id = RequestId::from_raw_for_test(7);
        (
            kernel::Effect::Request(AnyIssuedRequest::BlockLogs(IssuedRequest {
                request_id,
                request_payload: GetBlockLogs { block_hash },
            })),
            request_id,
        )
    }

    fn pool_data_request_effect(at: BlockHash) -> (kernel::Effect, RequestId<GetPoolData>) {
        let request_id = RequestId::from_raw_for_test(8);
        (
            kernel::Effect::Request(AnyIssuedRequest::PoolData(IssuedRequest {
                request_id,
                request_payload: GetPoolData {
                    at,
                    pools: HashSet::new(),
                },
            })),
            request_id,
        )
    }

    fn pool_metadata_request_effect(
        at: BlockHash,
        candidates: HashSet<ProtocolPoolKey>,
    ) -> (kernel::Effect, RequestId<GetPoolMetadata>) {
        let request_id = RequestId::from_raw_for_test(9);
        (
            kernel::Effect::Request(AnyIssuedRequest::PoolMetadata(IssuedRequest {
                request_id,
                request_payload: GetPoolMetadata { at, candidates },
            })),
            request_id,
        )
    }

    fn token_metadata_request_effect(
        at: BlockHash,
        tokens: HashSet<TokenAddress>,
    ) -> (kernel::Effect, RequestId<GetTokenMetadata>) {
        let request_id = RequestId::from_raw_for_test(10);
        (
            kernel::Effect::Request(AnyIssuedRequest::TokenMetadata(IssuedRequest {
                request_id,
                request_payload: GetTokenMetadata { at, tokens },
            })),
            request_id,
        )
    }

    fn unexpected_block_header_fetch(
        _block_hash: BlockHash,
    ) -> Result<Option<ClientHead>, ClientEvmError> {
        panic!("block header fetch must not be called")
    }

    fn unexpected_block_logs_fetch(_block_hash: BlockHash) -> Result<Vec<PoolLog>, ClientEvmError> {
        panic!("block logs fetch must not be called")
    }

    fn unexpected_pool_data_fetch(
        _at: BlockHash,
        _pools: HashSet<PoolRef>,
    ) -> Result<HashMap<PoolRef, PoolDataResult>, ClientEvmError> {
        panic!("pool data fetch must not be called")
    }

    fn unexpected_pool_metadata_fetch(
        _at: BlockHash,
        _candidates: HashSet<ProtocolPoolKey>,
    ) -> Result<HashMap<ProtocolPoolKey, PoolMetadataResult>, ClientEvmError> {
        panic!("pool metadata fetch must not be called")
    }

    fn unexpected_token_metadata_fetch(
        _at: BlockHash,
        _tokens: HashSet<TokenAddress>,
    ) -> Result<HashMap<TokenAddress, TokenMetadataResult>, ClientEvmError> {
        panic!("token metadata fetch must not be called")
    }

    fn pool_candidate_address(last_byte: u8) -> ProtocolPoolKey {
        let address = format!("0x{}", format!("{last_byte:040x}"))
            .parse()
            .expect("test address must parse");

        ProtocolPoolKey::UniswapV3(address)
    }

    fn pool_metadata(last_byte: u8) -> PoolMetadata {
        PoolMetadata {
            token0: pool_candidate_address(last_byte)
                .uniswap_v3_address()
                .expect("v3 pool"),
            token1: pool_candidate_address(last_byte.wrapping_add(1))
                .uniswap_v3_address()
                .expect("v3 pool"),
            fee: PoolFee::Tiered(UniswapV3Fee::Fee3000),
        }
    }

    fn token_address(last_byte: u8) -> TokenAddress {
        let address = format!("0x{}", format!("{last_byte:040x}"))
            .parse()
            .expect("test address must parse");

        TokenAddress(address, ChainKey::Ethereum)
    }

    fn block_header(
        block_hash: BlockHash,
        parent_hash: BlockHash,
    ) -> Result<ClientHead, serde_json::Error> {
        serde_json::from_value(json!({
            "hash": block_hash,
            "parentHash": parent_hash,
            "sha3Uncles": hash(5),
            "miner": "0x0000000000000000000000000000000000000006",
            "stateRoot": hash(7),
            "transactionsRoot": hash(8),
            "receiptsRoot": hash(9),
            "logsBloom": zero_logs_bloom(),
            "difficulty": "0xd",
            "number": "0x9",
            "gasLimit": "0xb",
            "gasUsed": "0xa",
            "timestamp": "0xc",
            "extraData": "0x010203",
            "mixHash": hash(10),
            "nonce": "0x000000000000000f"
        }))
    }

    fn hash(value: u8) -> BlockHash {
        BlockHash::with_last_byte(value)
    }

    fn optimization_input(
        block_hash: BlockHash,
    ) -> client_evm::multi_chain_kernel::OptimizationPoolReserves {
        client_evm::multi_chain_kernel::OptimizationPoolReserves {
            block_hashes: std::collections::BTreeMap::from([(ChainKey::Ethereum, block_hash)]),
            reserves: Vec::new(),
        }
    }

    fn optimization_input_with_reserves(
        block_hash: BlockHash,
    ) -> client_evm::multi_chain_kernel::OptimizationPoolReserves {
        client_evm::multi_chain_kernel::OptimizationPoolReserves {
            block_hashes: std::collections::BTreeMap::from([(ChainKey::Ethereum, block_hash)]),
            reserves: vec![optimization_pool_reserves()],
        }
    }

    fn optimization_pool_reserves() -> optimization::PoolReserves<PoolRef, TokenAddress> {
        optimization::PoolReserves {
            pool_id: pool_address(1),
            token0: token_address(1),
            token1: token_address(2),
            value: optimization::VirtualReserveValues {
                token_0: 1_000.0,
                token_1: 2_000.0,
                fee_multiplier: 0.997,
                max_swap_0: 100.0,
                max_swap_1: 200.0,
            },
        }
    }

    fn pool_address(last_byte: u8) -> PoolRef {
        let address = format!("0x{}", format!("{last_byte:040x}"))
            .parse()
            .expect("test address must parse");

        PoolRef::uniswap_v3(address, ChainKey::Ethereum)
    }

    fn zero_logs_bloom() -> String {
        format!("0x{}", "00".repeat(256))
    }

    fn rpc_config() -> RpcConfig {
        RpcConfig {
            http_url: "https://example.invalid/http".to_owned(),
            ws_url: "wss://example.invalid/ws".to_owned(),
            api_key: "api-key".to_owned(),
        }
    }

    // Endpoint pools pointing at the (unreachable) test dRPC URL, public fallbacks disabled — so any
    // RPC a test triggers errors instead of escaping to a real provider.
    fn test_endpoints() -> ChainEndpoints {
        client_evm::assemble_chain_endpoints(
            &rpc_config(),
            &std::collections::BTreeMap::new(),
            false,
        )
        .expect("test endpoint assembly")
    }

    fn in_memory_metadata_cache() -> MetadataCache {
        MetadataCache::in_memory().expect("in-memory metadata cache")
    }
}
