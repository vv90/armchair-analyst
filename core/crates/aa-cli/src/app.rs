use aa_framework::{Application, ApplicationError, Runtime, Transition};
use client_evm::{
    ARBITRUM_NATIVE_TOKEN_ADDRESS, ARBITRUM_USDC_TOKEN_ADDRESS, ARBITRUM_USDT_TOKEN_ADDRESS,
    ARBITRUM_WBTC_TOKEN_ADDRESS, ARBITRUM_WETH_TOKEN_ADDRESS, AVALANCHE_USDC_TOKEN_ADDRESS,
    AVALANCHE_USDT_TOKEN_ADDRESS, AVALANCHE_WBTC_TOKEN_ADDRESS, AVALANCHE_WETH_TOKEN_ADDRESS,
    AnyIssuedRequest, AnyRequestId, BASE_CBBTC_TOKEN_ADDRESS, BASE_NATIVE_TOKEN_ADDRESS,
    BASE_USDC_TOKEN_ADDRESS, BASE_WETH_TOKEN_ADDRESS, BNB_BTCB_TOKEN_ADDRESS,
    BNB_USDC_TOKEN_ADDRESS, BNB_USDT_TOKEN_ADDRESS, BNB_WETH_TOKEN_ADDRESS, BlockHash,
    ChainEndpoints, ChainKey, ChainSubscriptions, ClientEvent, ClientEvmError, ClientHead,
    ETHEREUM_DAI_TOKEN_ADDRESS, ETHEREUM_NATIVE_TOKEN_ADDRESS, ETHEREUM_USDC_TOKEN_ADDRESS,
    ETHEREUM_USDT_TOKEN_ADDRESS, ETHEREUM_WBTC_TOKEN_ADDRESS, ETHEREUM_WETH_TOKEN_ADDRESS,
    GetLogsRange, GraphEndpoints, MetadataCache, OPTIMISM_DAI_TOKEN_ADDRESS,
    OPTIMISM_NATIVE_TOKEN_ADDRESS, OPTIMISM_USDC_TOKEN_ADDRESS, OPTIMISM_USDT_TOKEN_ADDRESS,
    OPTIMISM_WBTC_TOKEN_ADDRESS, OPTIMISM_WETH_TOKEN_ADDRESS, POLYGON_USDC_TOKEN_ADDRESS,
    POLYGON_USDT_TOKEN_ADDRESS, POLYGON_WBTC_TOKEN_ADDRESS, POLYGON_WETH_TOKEN_ADDRESS,
    POOL_LOG_BATCH_WINDOW, PoolDataResult, PoolLog, PoolMetadata, PoolMetadataResult, PoolRef,
    ProtocolPoolKey, RangeLogBlock, RequestId, TokenAddress, TokenMetadataResult, TokenWhitelist,
    WsSubscriptionEndpoint, bootstrap, consolidate_pool_logs, fetch_block_header,
    fetch_canonical_block_header_at, fetch_block_logs,
    fetch_finalized_block_header, fetch_pool_candidates_window, fetch_pool_data,
    fetch_pool_logs_in_range, fetch_pool_metadata, fetch_token_metadata, fetch_v4_pool_metadata,
    kernel,
    multi_chain_kernel::{
        ChainObservation, ChainProgress, Effect, Event, OptimizationPoolReserves, PlanVerification,
        State, Subscription, SubscriptionData, transition,
    },
    plan_ws_subscriptions, subscribe_new_heads, subscribe_pool_events,
};
use optimization::{
    OptimizationBackendSelection, OptimizationSessionConfig, OptimizationStepConfig,
    OptimizationStepResult,
};
use std::{
    collections::{HashMap, HashSet},
    fmt,
    sync::{
        Mutex, OnceLock,
        atomic::{AtomicU64, Ordering},
        mpsc::{Sender, channel},
    },
    thread::{self, JoinHandle},
    time,
};

use crate::{
    latest_slot::{LatestReceiveError, LatestReceiver, LatestSender, latest_slot},
    logger::Logger,
    optimization::{RunOptimizationError, run_optimization},
    utils::CliError,
    view::View,
};

pub(crate) struct ClientEvmApp {}

pub(crate) struct ClientEvmRuntime {
    agent: ureq::Agent,
    subscriptions: ChainSubscriptions,
    endpoints: ChainEndpoints,
    // Per-chain Uniswap v4 subgraph pools used by the v4 metadata resolver. Empty when The Graph is
    // unconfigured, in which case v4 metadata resolution is skipped.
    graph_endpoints: GraphEndpoints,
    metadata_cache: MetadataCache,
    optimization_sender: OnceLock<LatestSender<OptimizationPoolReserves>>,
    /// The optimizer's session config, built at startup (with any token whitelist already applied
    /// to its `whitelist` field and bridges) and handed to the optimization worker on spawn.
    optimization_session_config: OptimizationSessionConfig<TokenAddress>,
    /// The token whitelist, when configured: pools over non-whitelisted tokens are gated at
    /// metadata resolution ([`ClientEvmRuntime::cached_pool_metadata`]) so the kernel rejects
    /// them before they are ever tracked. `None` admits every pool (whitelisting disabled).
    token_whitelist: Option<TokenWhitelist>,
    view: View,
    logger: Logger,
    /// Wall-clock millis of the last emitted gauge line, throttling it to roughly once per second so
    /// the per-chain backlog snapshot lands in the log without per-transition spam.
    last_gauge_millis: AtomicU64,
    /// Monotonic base for the transition timing fields below.
    started: time::Instant,
    /// Micros since `started` when the current input was dequeued (`log_input` runs just before the
    /// transition, `observe_state` just after, both on the single transition thread — so their
    /// difference is the pure transition duration, and plain interior mutability suffices).
    last_input_micros: AtomicU64,
    /// The current input's formatted log line, echoed into the slow-transition warning so the
    /// culprit is on the warning line itself (adjacent lines can interleave with effect threads).
    last_input_line: Mutex<String>,
    /// Transitions processed since the last gauge line (`events=` on the gauge): with the 1/s gauge
    /// cadence this is the event-loop throughput, the saturation gauge for the single fold thread.
    transitions_since_gauge: AtomicU64,
    /// Slowest transition since the last gauge line, in micros (`max_ms=` on the gauge).
    max_transition_micros: AtomicU64,
    /// The last plan verification logged, so `observe_state` emits a verification line only when
    /// the kernel's verdict changes rather than on every transition.
    last_verification: Mutex<Option<PlanVerification>>,
}

impl ClientEvmRuntime {
    pub(crate) fn new(
        subscriptions: ChainSubscriptions,
        endpoints: ChainEndpoints,
        graph_endpoints: GraphEndpoints,
        metadata_cache: MetadataCache,
        optimization_session_config: OptimizationSessionConfig<TokenAddress>,
        token_whitelist: Option<TokenWhitelist>,
        logger: Logger,
        view: View,
    ) -> ClientEvmRuntime {
        ClientEvmRuntime {
            agent: ureq::Agent::new_with_defaults(),
            subscriptions,
            endpoints,
            graph_endpoints,
            metadata_cache,
            optimization_sender: OnceLock::new(),
            optimization_session_config,
            token_whitelist,
            view,
            logger,
            last_gauge_millis: AtomicU64::new(0),
            started: time::Instant::now(),
            last_input_micros: AtomicU64::new(0),
            last_input_line: Mutex::new(String::new()),
            transitions_since_gauge: AtomicU64::new(0),
            max_transition_micros: AtomicU64::new(0),
            last_verification: Mutex::new(None),
        }
    }

    /// Micros since runtime construction: the monotonic clock the transition timing reads.
    fn elapsed_micros(&self) -> u64 {
        self.started.elapsed().as_micros() as u64
    }

    /// Per-chain WebSocket endpoints used by the new-heads / pool-events subscription channel (single
    /// connection per chain).
    fn subscriptions(&self) -> &ChainSubscriptions {
        &self.subscriptions
    }

    /// Per-chain HTTP endpoint pools used by every RPC fetch (multi-provider, with failover).
    fn endpoints(&self) -> &ChainEndpoints {
        &self.endpoints
    }

    /// Per-chain Uniswap v4 subgraph pools (empty when unconfigured), used by the v4 metadata resolver.
    fn graph_endpoints(&self) -> &GraphEndpoints {
        &self.graph_endpoints
    }

    /// Resolves pool metadata through the persistent cache: known pools are served from disk, only the
    /// misses are fetched, and freshly validated metadata is written back. Misses are split by protocol
    /// — v3 pools go to RPC, v4 pools to the subgraph aggregator (a `PoolId` is a one-way hash, so v4
    /// metadata cannot be read from chain) — and merged. A cache fault degrades to a plain fetch, so the
    /// cache can never make a request fail that would otherwise succeed.
    fn cached_pool_metadata(
        &self,
        chain: ChainKey,
        at: BlockHash,
        candidates: HashSet<ProtocolPoolKey>,
    ) -> Result<HashMap<ProtocolPoolKey, PoolMetadataResult>, ClientEvmError> {
        // The whitelist gate runs on the way out — after the cache stored the true `Ok` results —
        // so gating decisions never persist: widening the whitelist on a later run re-admits
        // previously gated pools as cache hits.
        resolve_pool_metadata_with(
            chain,
            candidates,
            &self.logger,
            |candidates| self.metadata_cache.load_pool_metadata(chain, candidates),
            |metadata| self.metadata_cache.store_pool_metadata(chain, metadata),
            |v3_misses| fetch_pool_metadata(&self.agent, self.endpoints(), chain, at, v3_misses),
            |v4_misses| {
                fetch_v4_pool_metadata(&self.agent, self.graph_endpoints(), chain, v4_misses)
            },
        )
        .map(|results| match &self.token_whitelist {
            Some(whitelist) => whitelist.gate_pool_metadata_results(chain, results),
            None => results,
        })
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
                        number: header.inner.number,
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

    fn effect_pool_size(&self) -> usize {
        EFFECT_POOL_SIZE
    }

    fn spawn_subscription(
        &self,
        sender: &Sender<<ClientEvmApp as Application>::Input>,
        subscription: <ClientEvmApp as Application>::Subscription,
    ) {
        match subscription {
            Subscription::NewHeadsSubscription(chain) => {
                self.spawn_new_heads_fan_out(sender, chain);
            }
            Subscription::PoolEventsSubscription(chain) => {
                self.spawn_pool_events_fan_out(sender, chain);
            }
            Subscription::TickSubscription(interval) => {
                drop(spawn_tick_subscription(sender.clone(), interval));
            }
            Subscription::OptimizationSubscription => {
                let (slot_sender, slot_receiver) = latest_slot();

                if self.optimization_sender.set(slot_sender).is_ok() {
                    drop(spawn_optimization_subscription(
                        slot_receiver,
                        self.optimization_session_config.clone(),
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
        let line = format_input_log(input);
        self.logger.log(&line);
        self.last_input_micros
            .store(self.elapsed_micros(), Ordering::Relaxed);
        if let Ok(mut last_line) = self.last_input_line.lock() {
            *last_line = line;
        }
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
        self.record_transition_duration();
        self.view.render(state);
        self.log_verification(state);
        self.log_gauge(state);
    }
}

/// Width of the dedicated blocking-I/O pool that runs effects, replacing rayon's CPU-sized global
/// pool. It is the runtime's global cap on concurrent in-flight RPC effects, so a chain running a
/// deep ancestry walk can no longer occupy every worker and starve the others. Sized for I/O, not
/// CPUs: with 7 chains, headroom for an ancestry-walk burst, and each effect's `aggregate3_batched`
/// fan-out of up to `MULTICALL_MAX_CONCURRENT_BATCHES`, this caps peak sockets well under the level
/// that trips free-tier provider rate limits. The single tuning knob — raise it if chains can't keep
/// up, lower it if `json-rpc error 15: too many requests` climbs.
const EFFECT_POOL_SIZE: usize = 64;

/// Minimum gap between gauge lines. `observe_state` runs on every transition, so without this the
/// gauge would be emitted thousands of times a second.
const GAUGE_INTERVAL_MILLIS: u64 = 1000;

/// A single transition at or above this earns its own `warn slow_transition` line. Sized against
/// the fastest tracked block cadence (Arbitrum ~250ms): one event routinely eating a tenth of
/// that budget on the single transition thread deserves a trace before the backlog does.
const SLOW_TRANSITION_WARN_MILLIS: u64 = 25;

/// The `warn slow_transition` line for a transition that ran `elapsed_micros`, or `None` below the
/// [`SLOW_TRANSITION_WARN_MILLIS`] threshold. Echoes the input's own log line so the culprit event
/// is on the warning itself. Pure so the threshold gate is unit-tested without a runtime.
fn slow_transition_log(elapsed_micros: u64, input_line: &str) -> Option<String> {
    let millis = elapsed_micros / 1000;
    (millis >= SLOW_TRANSITION_WARN_MILLIS)
        .then(|| format!("warn slow_transition ms={millis} {input_line}"))
}

/// First wait after a subscription drops, and the value the backoff resets to after a healthy run.
const RECONNECT_BASE: time::Duration = time::Duration::from_millis(250);
/// Ceiling for the exponential backoff so a persistently-failing endpoint is retried steadily.
const RECONNECT_CAP: time::Duration = time::Duration::from_secs(30);
/// A subscription that ran at least this long counts as healthy, so its next drop reconnects fast
/// instead of inheriting a grown backoff.
const RECONNECT_STABILITY_WINDOW: time::Duration = time::Duration::from_secs(10);

/// Backoff after a subscription attempt returns. A run that lasted at least the stability window is
/// treated as healthy and resets to [`RECONNECT_BASE`]; otherwise the previous delay doubles up to
/// [`RECONNECT_CAP`] (or starts at the base on the first drop). Pure so the policy is unit-tested
/// without opening a socket.
fn next_reconnect_delay(previous: Option<time::Duration>, ran_for: time::Duration) -> time::Duration {
    if ran_for >= RECONNECT_STABILITY_WINDOW {
        return RECONNECT_BASE;
    }
    match previous {
        None => RECONNECT_BASE,
        Some(previous) => (previous * 2).min(RECONNECT_CAP),
    }
}

/// Runs one WebSocket subscription forever, reconnecting with [`next_reconnect_delay`] backoff when it
/// drops. `attempt` performs one connect-and-stream cycle (one of `subscribe_new_heads` /
/// `subscribe_pool_events`) and returns when the socket closes (`Ok`) or fails (`Err`) — either way we
/// reconnect, so a free-tier idle-timeout no longer kills a feed for good. Free of `self` so each of a
/// chain's fanned-out provider threads owns its own loop with a cloned [`Logger`]. Never returns.
fn reconnect_loop(
    logger: &Logger,
    chain: ChainKey,
    kind: &str,
    mut attempt: impl FnMut() -> Result<(), ClientEvmError>,
) {
    let mut previous_delay: Option<time::Duration> = None;
    let mut connects: u64 = 0;
    loop {
        connects += 1;
        logger.log(&format!(
            "subscription chain={chain:?} kind={kind} connecting attempt={connects}"
        ));

        let started = time::Instant::now();
        let outcome = attempt();
        let ran_for = started.elapsed();

        let reason = match &outcome {
            Ok(()) => "closed".to_owned(),
            Err(error) => format!("error: {error}"),
        };
        let delay = next_reconnect_delay(previous_delay, ran_for);
        logger.log(&format!(
            "subscription chain={chain:?} kind={kind} disconnected ran_ms={} reason={reason} reconnect_in_ms={}",
            ran_for.as_millis(),
            delay.as_millis()
        ));

        thread::sleep(delay);
        previous_delay = Some(delay);
    }
}

/// Blocks on every fanned-out subscription thread. Each [`reconnect_loop`] never returns, so this
/// parks the owning subscription thread for the life of the process (the threads run concurrently).
fn join_all(handles: Vec<JoinHandle<()>>) {
    for handle in handles {
        drop(handle.join());
    }
}

impl ClientEvmRuntime {
    /// Fans the new-heads feed out across every configured WS provider for `chain`: one independent
    /// reconnecting connection each, all delivering straight into the kernel channel. Duplicate heads
    /// across providers are absorbed by the kernel (`DuplicateBlock`); heads are latency-sensitive, so
    /// there is no debounce here. Blocks for the life of the process (each connection loops forever).
    fn spawn_new_heads_fan_out(&self, sender: &Sender<Event>, chain: ChainKey) {
        let endpoints = self.chain_ws_endpoints(chain, "new_heads");
        let handles = endpoints
            .into_iter()
            .map(|endpoint| {
                let logger = self.logger.clone();
                let sender = sender.clone();
                thread::spawn(move || {
                    let kind = format!("new_heads provider={}", endpoint.label);
                    reconnect_loop(&logger, chain, &kind, || {
                        subscribe_new_heads(&endpoint.url, &sender, |client_event| {
                            map_new_head_data(client_event)
                                .map(|data| Event::SubscriptionData { chain, data })
                        })
                    });
                })
            })
            .collect();
        join_all(handles);
    }

    /// Fans the pool-events feed out across every configured WS provider for `chain`: each provider's
    /// per-log stream is funnelled through one mpsc into a single consolidator, which debounces and
    /// dedups the burst (pure [`consolidate_pool_logs`]) into one batched `LogObserved` per block
    /// before it reaches the kernel channel. Blocks for the life of the process.
    fn spawn_pool_events_fan_out(&self, sender: &Sender<Event>, chain: ChainKey) {
        let endpoints = self.chain_ws_endpoints(chain, "pool_events");
        if endpoints.is_empty() {
            return;
        }

        let (raw_sender, raw_receiver) = channel::<(BlockHash, PoolLog)>();
        let mut handles: Vec<JoinHandle<()>> = endpoints
            .into_iter()
            .map(|endpoint| {
                let logger = self.logger.clone();
                let raw_sender = raw_sender.clone();
                thread::spawn(move || {
                    let kind = format!("pool_events provider={}", endpoint.label);
                    reconnect_loop(&logger, chain, &kind, || {
                        subscribe_pool_events(&endpoint.url, &raw_sender, map_pool_log_data)
                    });
                })
            })
            .collect();
        // Drop the original handle so the consolidator only sees `Disconnected` once every provider
        // thread is gone (never, in practice — each reconnects forever).
        drop(raw_sender);

        let sender = sender.clone();
        handles.push(thread::spawn(move || {
            consolidate_pool_logs(&raw_receiver, POOL_LOG_BATCH_WINDOW, |block_hash, logs| {
                let event = Event::SubscriptionData {
                    chain,
                    data: SubscriptionData::PoolLog { block_hash, logs },
                };
                drop(sender.send(event));
            });
        }));
        join_all(handles);
    }

    /// The configured WS endpoints for one chain, taken from the pure fan-out plan. Logs (once) and
    /// yields empty when the chain has none, so a missing WS config surfaces rather than silently
    /// opening no feed.
    fn chain_ws_endpoints(&self, chain: ChainKey, kind: &str) -> Vec<WsSubscriptionEndpoint> {
        let endpoints: Vec<WsSubscriptionEndpoint> = plan_ws_subscriptions(self.subscriptions())
            .into_iter()
            .filter(|endpoint| endpoint.chain == chain)
            .collect();
        if endpoints.is_empty() {
            self.logger.log(&format!(
                "error chain={chain:?} subscription_ws_unavailable kind={kind}"
            ));
        }
        endpoints
    }

    /// Measures the transition that just ran (`log_input` stamped its start on this same thread),
    /// feeds the per-gauge-interval counters, and warns on a single transition slow enough to
    /// threaten the event loop's cadence — the direct probe for the per-event overlay-fold cost.
    fn record_transition_duration(&self) {
        let elapsed_micros = self
            .elapsed_micros()
            .saturating_sub(self.last_input_micros.load(Ordering::Relaxed));
        self.transitions_since_gauge.fetch_add(1, Ordering::Relaxed);
        self.max_transition_micros
            .fetch_max(elapsed_micros, Ordering::Relaxed);

        if let Ok(last_line) = self.last_input_line.lock() {
            if let Some(warning) = slow_transition_log(elapsed_micros, &last_line) {
                self.logger.log(&warning);
            }
        }
    }

    /// Emits a single per-chain backlog snapshot (`behind` / `window` / `pools` / `inflight`) plus
    /// the interval's event-loop counters to the log, at most once per [`GAUGE_INTERVAL_MILLIS`].
    /// Runs on the single transition thread, so the throttle needs no synchronization beyond
    /// interior mutability.
    fn log_gauge(&self, state: &State) {
        let now = now_millis();
        if now.saturating_sub(self.last_gauge_millis.load(Ordering::Relaxed)) < GAUGE_INTERVAL_MILLIS
        {
            return;
        }
        self.last_gauge_millis.store(now, Ordering::Relaxed);
        let events = self.transitions_since_gauge.swap(0, Ordering::Relaxed);
        let max_transition_millis = self.max_transition_micros.swap(0, Ordering::Relaxed) / 1000;
        self.logger
            .log(&format_gauge_log(&state.observe(), events, max_transition_millis));
    }

    /// Logs the kernel's latest plan-verification verdict next to the claimed profit, once per
    /// change (verdicts repeat across transitions until the next completed optimization step).
    /// Runs on the single transition thread, so the cell needs only interior mutability.
    fn log_verification(&self, state: &State) {
        let verification = state.latest_plan_verification();
        let Ok(mut last) = self.last_verification.lock() else {
            return;
        };
        if *last == verification {
            return;
        }
        *last = verification;
        if let (Some(result), Some(verification)) =
            (state.latest_optimization_result(), verification)
        {
            self.logger
                .log(&format_verification_log(result, verification));
        }
    }

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
    subscriptions: ChainSubscriptions,
    endpoints: ChainEndpoints,
    graph_endpoints: GraphEndpoints,
    metadata_cache: MetadataCache,
    optimization_session_config: OptimizationSessionConfig<TokenAddress>,
    token_whitelist: Option<TokenWhitelist>,
    logger: Logger,
    view: View,
) -> JoinHandle<()> {
    let (_sender, handle) = <ClientEvmRuntime as Runtime<ClientEvmApp>>::run(
        ClientEvmRuntime::new(
            subscriptions,
            endpoints,
            graph_endpoints,
            metadata_cache,
            optimization_session_config,
            token_whitelist,
            logger,
            view,
        ),
    );

    handle
}

fn format_input_log(input: &Event) -> String {
    match input {
        Event::FinalizedHeaderReceived {
            chain,
            block_hash,
            number,
        } => {
            format!(
                "input finalized_header_received chain={chain:?} block={block_hash} number={number}"
            )
        }
        Event::FinalizedHeaderUnavailable { chain } => {
            format!("input finalized_header_unavailable chain={chain:?}")
        }
        Event::SubscriptionData { chain, data } => format_subscription_data_log(*chain, data),
        Event::ChainEvent { chain, event } => format_chain_event_log(*chain, event),
        Event::BootstrapEvent { chain, event } => format_bootstrap_event_log(*chain, event),
        Event::OptimizationStepCompleted { result, plan } => format!(
            "input optimization_step_completed status={:?} profit={} reserves={} routed={} entropy={:.2} eff_pools={:.2} iterations={} plan_steps={}",
            result.status,
            result.profit_amount,
            result.reserves_count,
            result.routed_pool_count,
            result.route_entropy,
            result.effective_pools,
            result.iterations_completed,
            plan.as_ref().map_or(0, |plan| plan.steps.len()),
        ),
        Event::Tick => "input tick".to_owned(),
    }
}

/// One log line putting the lossless-replay verdict next to the optimizer's claimed profit for the
/// same step (and entry size). Emitted once per verdict change by `log_verification`.
fn format_verification_log(
    result: OptimizationStepResult,
    verification: PlanVerification,
) -> String {
    match verification {
        PlanVerification::Verified {
            profit,
            hit_tick_limit,
        } => format!(
            "optimization plan_verification claimed={} verified={}{}",
            result.profit_amount,
            profit,
            if hit_tick_limit { " tick_limited" } else { "" },
        ),
        PlanVerification::Unverifiable(failure) => format!(
            "optimization plan_verification claimed={} unverifiable={failure:?}",
            result.profit_amount,
        ),
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

/// Wall-clock millis since the Unix epoch, used only to throttle the gauge line.
fn now_millis() -> u64 {
    time::SystemTime::now()
        .duration_since(time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis() as u64)
        .unwrap_or(0)
}

/// Renders one `gauge` line: a per-chain backlog snapshot plus the interval's event-loop counters.
/// Active chains carry `behind`/`window`/`pools`/`inflight`/`ws_miss`; bootstrapping chains render
/// as `init` with their replay-buffer depth. `events` is the transitions processed since the last
/// gauge line and `max_transition_millis` the slowest of them — together the saturation gauge for
/// the single transition thread. Pure so it can be unit-tested off a synthetic observation set.
fn format_gauge_log(
    observations: &[(ChainKey, ChainObservation)],
    events: u64,
    max_transition_millis: u64,
) -> String {
    let segments = observations
        .iter()
        .map(|(chain, observation)| match observation {
            ChainObservation::Initializing { buffered_events } => {
                format!("{chain:?}=init buffered={buffered_events}")
            }
            ChainObservation::Active(ChainProgress {
                verified_pools,
                blocks_behind_tip,
                canonical_window,
                in_flight_requests,
                ws_misses,
            }) => {
                let behind = format_block_count(*blocks_behind_tip);
                let window = format_block_count(*canonical_window);
                format!(
                    "{chain:?}=active behind={behind} window={window} pools={verified_pools} inflight={in_flight_requests} ws_miss={ws_misses}"
                )
            }
        })
        .collect::<Vec<_>>()
        .join("  ");

    format!("gauge {segments}  events={events} max_ms={max_transition_millis}")
}

/// Renders an optional block count with `?` for the not-yet-measurable case (no dispatch reference
/// yet, or a tip not connected to the anchor).
fn format_block_count(count: Option<usize>) -> String {
    match count {
        Some(count) => count.to_string(),
        None => "?".to_owned(),
    }
}

fn format_subscription_data_log(chain: ChainKey, data: &SubscriptionData) -> String {
    match data {
        SubscriptionData::NewHead {
            hash,
            parent_hash,
            number,
            ..
        } => format!(
            "input chain={chain:?} head_observed hash={hash} parent={parent_hash} number={number}"
        ),
        SubscriptionData::PoolLog { block_hash, logs } => {
            format!(
                "input chain={chain:?} log_observed block={block_hash} pools={}",
                logs.len(),
            )
        }
    }
}

fn format_chain_event_log(chain: ChainKey, event: &kernel::Event) -> String {
    match event {
        kernel::Event::HeadObserved {
            hash,
            parent_hash,
            number,
            ..
        } => format!(
            "input chain={chain:?} head_observed hash={hash} parent={parent_hash} number={number}"
        ),
        kernel::Event::FinalizedBlockObserved { block_hash, number } => {
            format!(
                "input chain={chain:?} finalized_block_observed hash={block_hash} number={number}"
            )
        }
        kernel::Event::BlockHeaderReceived {
            request_id,
            hash,
            parent_hash,
            number,
            ..
        } => format!(
            "input chain={chain:?} block_header_received request={} hash={hash} parent={parent_hash} number={number}",
            format_typed_request_id_log(request_id),
        ),
        kernel::Event::BlockHeaderNotFound { request_id } => format!(
            "input chain={chain:?} block_header_not_found request={}",
            format_typed_request_id_log(request_id),
        ),
        kernel::Event::CanonicalHeaderAtHeightReceived {
            request_id,
            hash,
            number,
        } => format!(
            "input chain={chain:?} canonical_header_at_height_received request={} hash={hash} number={number}",
            format_typed_request_id_log(request_id),
        ),
        kernel::Event::BlockLogsReceived { request_id, logs } => format!(
            "input chain={chain:?} block_logs_received request={} pools={}",
            format_typed_request_id_log(request_id),
            logs.len(),
        ),
        kernel::Event::BlockLogsRangeReceived { request_id, blocks } => format!(
            "input chain={chain:?} block_logs_range_received request={} blocks={}",
            format_typed_request_id_log(request_id),
            blocks.len(),
        ),
        kernel::Event::LogObserved { block_hash, logs } => format!(
            "input chain={chain:?} log_observed block={block_hash} pools={}",
            logs.len(),
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
        kernel::Event::PoolDataReceived { request_id, pools } => format!(
            "input chain={chain:?} pool_data_received request={} pools={}",
            format_typed_request_id_log(request_id),
            pools.len(),
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

/// One line per issued ranged-getLogs fetch, mirroring the response line's request id so
/// issue → response/failure pairs up in the log.
fn format_logs_range_effect_log(
    chain: ChainKey,
    request_id: &RequestId<GetLogsRange>,
    from: u64,
    to: u64,
    covered: usize,
) -> String {
    format!(
        "effect logs_range chain={chain:?} request={} from={from} to={to} covered={covered}",
        format_typed_request_id_log(request_id),
    )
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
                let scan_tip = request.request_payload.scan_tip;

                match fetch_pool_candidates_window(
                    &self.agent,
                    endpoints,
                    chain,
                    from_block,
                    scan_tip,
                ) {
                    Ok((blocks, scan_tip, next_from)) => {
                        bootstrap::Event::PoolCandidatesReceived {
                            request_id,
                            blocks,
                            scan_tip,
                            next_from,
                        }
                    }
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
            |at, candidates| self.cached_pool_metadata(chain, at, candidates),
            |at, tokens| self.cached_token_metadata(chain, at, tokens),
            // Pool data is anchor-specific and the anchor moves, so — unlike immutable metadata —
            // it is read live with no cache wrapper.
            |at, pools| fetch_pool_data(&self.agent, self.endpoints(), chain, at, pools),
            |from, to| fetch_pool_logs_in_range(&self.agent, self.endpoints(), chain, from, to),
            |number| fetch_canonical_block_header_at(&self.agent, self.endpoints(), chain, number),
        )
    }
}

fn map_new_head_data(client_chain_event: ClientEvent) -> Option<SubscriptionData> {
    match client_chain_event {
        ClientEvent::NewHead { header, .. } => Some(SubscriptionData::NewHead {
            hash: header.inner.hash,
            parent_hash: header.inner.inner.parent_hash,
            logs_bloom: header.inner.inner.logs_bloom,
            number: header.inner.inner.number,
        }),
        ClientEvent::PoolLogObserved { .. }
        | ClientEvent::Subscribed { .. }
        | ClientEvent::Closed { .. } => None,
    }
}

/// The raw per-log projection for the pool-events feed. Each provider connection maps into this;
/// the consolidator ([`consolidate_pool_logs`]) then dedups and batches the burst before it reaches
/// the kernel channel — so the debounce, not this mapper, decides delivery.
fn map_pool_log_data(client_chain_event: ClientEvent) -> Option<(BlockHash, PoolLog)> {
    match client_chain_event {
        ClientEvent::PoolLogObserved {
            block_hash, log, ..
        } => Some((block_hash, log)),
        ClientEvent::NewHead { .. }
        | ClientEvent::Subscribed { .. }
        | ClientEvent::Closed { .. } => None,
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
    session_config: OptimizationSessionConfig<TokenAddress>,
    sender: Sender<Event>,
    logger: Logger,
) -> JoinHandle<()> {
    thread::spawn(move || {
        let result = run_optimization(
            receiver,
            default_optimization_backend(),
            session_config,
            default_optimization_step_config(),
            sender,
            |result, plan| Event::OptimizationStepCompleted { result, plan },
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

/// Builds the optimizer's session config, applying the token whitelist when one is configured:
/// the init asset must be whitelisted (hard error otherwise — nothing could route), and bridges
/// with a non-whitelisted endpoint are dropped rather than rejected, so a deliberately narrow
/// whitelist (e.g. a single chain) still starts. Returns the dropped pairs for startup logging.
pub(crate) fn optimization_session_config(
    whitelist: Option<&TokenWhitelist>,
) -> Result<
    (
        OptimizationSessionConfig<TokenAddress>,
        Vec<(TokenAddress, TokenAddress)>,
    ),
    CliError,
> {
    let bridges = default_optimization_bridges();

    let Some(whitelist) = whitelist else {
        return Ok((
            OptimizationSessionConfig {
                source_asset: ETHEREUM_USDC_TOKEN_ADDRESS,
                output_asset: ETHEREUM_USDC_TOKEN_ADDRESS,
                bridges,
                whitelist: None,
            },
            Vec::new(),
        ));
    };

    if !whitelist.allows(ETHEREUM_USDC_TOKEN_ADDRESS) {
        return Err(CliError::InitAssetNotWhitelisted {
            init_asset: format!("{ETHEREUM_USDC_TOKEN_ADDRESS:?}"),
        });
    }

    let (bridges, dropped): (HashSet<_>, HashSet<_>) = bridges
        .into_iter()
        .partition(|&(a, b)| whitelist.allows(a) && whitelist.allows(b));

    Ok((
        OptimizationSessionConfig {
            source_asset: ETHEREUM_USDC_TOKEN_ADDRESS,
            output_asset: ETHEREUM_USDC_TOKEN_ADDRESS,
            bridges,
            whitelist: Some(whitelist.token_set().clone()),
        },
        dropped.into_iter().collect(),
    ))
}

/// Synthetic 1:1 connections the optimizer treats as fungible passthroughs. Bridges are directional,
/// so both orderings of each pair are registered; the optimizer ignores a bridge whose endpoints
/// aren't yet present, so an entry is harmless before its tokens have reported.
///
/// * Cross-chain USDC: lets the single `init_asset` (Ethereum USDC) traverse every other chain's pools
///   and close cross-chain cycles back to it. Ethereum USDC is the hub; each chain's native USDC bridges
///   to and from it, so all chains' USDC are mutually reachable.
/// * Cross-chain exposure equivalents: Ethereum USDT, WBTC, and WETH are hubs for the vetted
///   chain-specific variants; Ethereum DAI connects to Optimism DAI. BTC variants include cbBTC on
///   Base and BTCB on BNB Chain.
/// * Native ETH ↔ WETH: wrapping is 1:1, so this unifies v4 native-ETH pools (`token0 = address(0)`)
///   with v3 WETH liquidity; without it, native-ETH pools would be an isolated island in the graph.
///   Registered per chain (native ETH and WETH are distinct `(Address, ChainKey)` tokens on each chain),
///   and registered ONLY for chains whose native gas token is ETH (Ethereum, Arbitrum, Base, Optimism).
///   Polygon (POL), BNB (BNB) and Avalanche (AVAX) have a non-ETH native token, so their `WETH` is an
///   ordinary bridged ERC20 that is NOT 1:1 with the native token — bridging it to native would be wrong,
///   so it is omitted and that liquidity connects through pools instead.
fn default_optimization_bridges() -> HashSet<(TokenAddress, TokenAddress)> {
    HashSet::from([
        // Cross-chain USDC hub (Ethereum USDC ⇄ each chain).
        (ETHEREUM_USDC_TOKEN_ADDRESS, ARBITRUM_USDC_TOKEN_ADDRESS),
        (ARBITRUM_USDC_TOKEN_ADDRESS, ETHEREUM_USDC_TOKEN_ADDRESS),
        (ETHEREUM_USDC_TOKEN_ADDRESS, BASE_USDC_TOKEN_ADDRESS),
        (BASE_USDC_TOKEN_ADDRESS, ETHEREUM_USDC_TOKEN_ADDRESS),
        (ETHEREUM_USDC_TOKEN_ADDRESS, OPTIMISM_USDC_TOKEN_ADDRESS),
        (OPTIMISM_USDC_TOKEN_ADDRESS, ETHEREUM_USDC_TOKEN_ADDRESS),
        (ETHEREUM_USDC_TOKEN_ADDRESS, POLYGON_USDC_TOKEN_ADDRESS),
        (POLYGON_USDC_TOKEN_ADDRESS, ETHEREUM_USDC_TOKEN_ADDRESS),
        (ETHEREUM_USDC_TOKEN_ADDRESS, BNB_USDC_TOKEN_ADDRESS),
        (BNB_USDC_TOKEN_ADDRESS, ETHEREUM_USDC_TOKEN_ADDRESS),
        (ETHEREUM_USDC_TOKEN_ADDRESS, AVALANCHE_USDC_TOKEN_ADDRESS),
        (AVALANCHE_USDC_TOKEN_ADDRESS, ETHEREUM_USDC_TOKEN_ADDRESS),
        // Cross-chain USDT exposure hub.
        (ETHEREUM_USDT_TOKEN_ADDRESS, ARBITRUM_USDT_TOKEN_ADDRESS),
        (ARBITRUM_USDT_TOKEN_ADDRESS, ETHEREUM_USDT_TOKEN_ADDRESS),
        (ETHEREUM_USDT_TOKEN_ADDRESS, OPTIMISM_USDT_TOKEN_ADDRESS),
        (OPTIMISM_USDT_TOKEN_ADDRESS, ETHEREUM_USDT_TOKEN_ADDRESS),
        (ETHEREUM_USDT_TOKEN_ADDRESS, POLYGON_USDT_TOKEN_ADDRESS),
        (POLYGON_USDT_TOKEN_ADDRESS, ETHEREUM_USDT_TOKEN_ADDRESS),
        (ETHEREUM_USDT_TOKEN_ADDRESS, BNB_USDT_TOKEN_ADDRESS),
        (BNB_USDT_TOKEN_ADDRESS, ETHEREUM_USDT_TOKEN_ADDRESS),
        (ETHEREUM_USDT_TOKEN_ADDRESS, AVALANCHE_USDT_TOKEN_ADDRESS),
        (AVALANCHE_USDT_TOKEN_ADDRESS, ETHEREUM_USDT_TOKEN_ADDRESS),
        // Cross-chain BTC exposure hub.
        (ETHEREUM_WBTC_TOKEN_ADDRESS, ARBITRUM_WBTC_TOKEN_ADDRESS),
        (ARBITRUM_WBTC_TOKEN_ADDRESS, ETHEREUM_WBTC_TOKEN_ADDRESS),
        (ETHEREUM_WBTC_TOKEN_ADDRESS, BASE_CBBTC_TOKEN_ADDRESS),
        (BASE_CBBTC_TOKEN_ADDRESS, ETHEREUM_WBTC_TOKEN_ADDRESS),
        (ETHEREUM_WBTC_TOKEN_ADDRESS, OPTIMISM_WBTC_TOKEN_ADDRESS),
        (OPTIMISM_WBTC_TOKEN_ADDRESS, ETHEREUM_WBTC_TOKEN_ADDRESS),
        (ETHEREUM_WBTC_TOKEN_ADDRESS, POLYGON_WBTC_TOKEN_ADDRESS),
        (POLYGON_WBTC_TOKEN_ADDRESS, ETHEREUM_WBTC_TOKEN_ADDRESS),
        (ETHEREUM_WBTC_TOKEN_ADDRESS, BNB_BTCB_TOKEN_ADDRESS),
        (BNB_BTCB_TOKEN_ADDRESS, ETHEREUM_WBTC_TOKEN_ADDRESS),
        (ETHEREUM_WBTC_TOKEN_ADDRESS, AVALANCHE_WBTC_TOKEN_ADDRESS),
        (AVALANCHE_WBTC_TOKEN_ADDRESS, ETHEREUM_WBTC_TOKEN_ADDRESS),
        // Cross-chain ETH exposure hub.
        (ETHEREUM_WETH_TOKEN_ADDRESS, ARBITRUM_WETH_TOKEN_ADDRESS),
        (ARBITRUM_WETH_TOKEN_ADDRESS, ETHEREUM_WETH_TOKEN_ADDRESS),
        (ETHEREUM_WETH_TOKEN_ADDRESS, BASE_WETH_TOKEN_ADDRESS),
        (BASE_WETH_TOKEN_ADDRESS, ETHEREUM_WETH_TOKEN_ADDRESS),
        (ETHEREUM_WETH_TOKEN_ADDRESS, OPTIMISM_WETH_TOKEN_ADDRESS),
        (OPTIMISM_WETH_TOKEN_ADDRESS, ETHEREUM_WETH_TOKEN_ADDRESS),
        (ETHEREUM_WETH_TOKEN_ADDRESS, POLYGON_WETH_TOKEN_ADDRESS),
        (POLYGON_WETH_TOKEN_ADDRESS, ETHEREUM_WETH_TOKEN_ADDRESS),
        (ETHEREUM_WETH_TOKEN_ADDRESS, BNB_WETH_TOKEN_ADDRESS),
        (BNB_WETH_TOKEN_ADDRESS, ETHEREUM_WETH_TOKEN_ADDRESS),
        (ETHEREUM_WETH_TOKEN_ADDRESS, AVALANCHE_WETH_TOKEN_ADDRESS),
        (AVALANCHE_WETH_TOKEN_ADDRESS, ETHEREUM_WETH_TOKEN_ADDRESS),
        // Cross-chain DAI exposure.
        (ETHEREUM_DAI_TOKEN_ADDRESS, OPTIMISM_DAI_TOKEN_ADDRESS),
        (OPTIMISM_DAI_TOKEN_ADDRESS, ETHEREUM_DAI_TOKEN_ADDRESS),
        // Native ETH ↔ WETH, only on ETH-native chains.
        (ETHEREUM_NATIVE_TOKEN_ADDRESS, ETHEREUM_WETH_TOKEN_ADDRESS),
        (ETHEREUM_WETH_TOKEN_ADDRESS, ETHEREUM_NATIVE_TOKEN_ADDRESS),
        (ARBITRUM_NATIVE_TOKEN_ADDRESS, ARBITRUM_WETH_TOKEN_ADDRESS),
        (ARBITRUM_WETH_TOKEN_ADDRESS, ARBITRUM_NATIVE_TOKEN_ADDRESS),
        (BASE_NATIVE_TOKEN_ADDRESS, BASE_WETH_TOKEN_ADDRESS),
        (BASE_WETH_TOKEN_ADDRESS, BASE_NATIVE_TOKEN_ADDRESS),
        (OPTIMISM_NATIVE_TOKEN_ADDRESS, OPTIMISM_WETH_TOKEN_ADDRESS),
        (OPTIMISM_WETH_TOKEN_ADDRESS, OPTIMISM_NATIVE_TOKEN_ADDRESS),
    ])
}

fn default_optimization_step_config() -> OptimizationStepConfig {
    OptimizationStepConfig {
        input_amount: 1000.0,
        iterations: 10,
    }
}

/// Resolves pool metadata from cache + fetch, with every effectful dependency injected so the
/// cache-degradation, protocol routing, and merge logic are unit-testable without IO. Cache hits are
/// served as `Ok`; the remaining misses are partitioned (v4 by [`ProtocolPoolKey::uniswap_v4_pool_id`])
/// and routed to `fetch_v3` / `fetch_v4`; the merged results are written back (a store fault is logged,
/// not fatal) and unioned with the hits. A cache *load* fault degrades to "no hits" so the fetch still
/// runs.
fn resolve_pool_metadata_with<Load, Store, FetchV3, FetchV4, LoadErr, StoreErr>(
    chain: ChainKey,
    candidates: HashSet<ProtocolPoolKey>,
    logger: &Logger,
    load_cache: Load,
    store_cache: Store,
    fetch_v3: FetchV3,
    fetch_v4: FetchV4,
) -> Result<HashMap<ProtocolPoolKey, PoolMetadataResult>, ClientEvmError>
where
    Load: FnOnce(
        &HashSet<ProtocolPoolKey>,
    ) -> Result<HashMap<ProtocolPoolKey, PoolMetadata>, LoadErr>,
    Store: FnOnce(&HashMap<ProtocolPoolKey, PoolMetadataResult>) -> Result<(), StoreErr>,
    FetchV3: FnOnce(
        HashSet<ProtocolPoolKey>,
    ) -> Result<HashMap<ProtocolPoolKey, PoolMetadataResult>, ClientEvmError>,
    FetchV4: FnOnce(
        HashSet<ProtocolPoolKey>,
    ) -> Result<HashMap<ProtocolPoolKey, PoolMetadataResult>, ClientEvmError>,
    LoadErr: fmt::Display,
    StoreErr: fmt::Display,
{
    let cached = match load_cache(&candidates) {
        Ok(cached) => cached,
        Err(error) => {
            logger.log(&format!(
                "error chain={chain:?} metadata_cache_load_failed kind=pool error={error}"
            ));
            HashMap::new()
        }
    };

    // Partition the misses: v4 pools resolve from the subgraph aggregator, v3 from RPC.
    let (v4_misses, v3_misses): (HashSet<_>, HashSet<_>) = candidates
        .into_iter()
        .filter(|candidate| !cached.contains_key(candidate))
        .partition(|candidate| candidate.uniswap_v4_pool_id().is_some());

    let mut metadata = fetch_v3(v3_misses)?;
    metadata.extend(fetch_v4(v4_misses)?);

    if let Err(error) = store_cache(&metadata) {
        logger.log(&format!(
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

fn execute_chain_effect_with<
    FetchBlockHeader,
    FetchBlockLogs,
    FetchPoolMetadata,
    FetchTokenMetadata,
    FetchPoolData,
    FetchLogsRange,
    FetchCanonicalHeader,
>(
    chain: ChainKey,
    effect: kernel::Effect,
    logger: &Logger,
    fetch_block_header: FetchBlockHeader,
    fetch_block_logs: FetchBlockLogs,
    fetch_pool_metadata: FetchPoolMetadata,
    fetch_token_metadata: FetchTokenMetadata,
    fetch_pool_data: FetchPoolData,
    fetch_logs_range: FetchLogsRange,
    fetch_canonical_header: FetchCanonicalHeader,
) -> Vec<Event>
where
    FetchBlockHeader: FnOnce(BlockHash) -> Result<Option<ClientHead>, ClientEvmError>,
    FetchCanonicalHeader: FnOnce(u64) -> Result<Option<ClientHead>, ClientEvmError>,
    FetchBlockLogs: FnOnce(BlockHash) -> Result<Vec<PoolLog>, ClientEvmError>,
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
    FetchPoolData:
        FnOnce(
            BlockHash,
            HashSet<PoolRef>,
        ) -> Result<HashMap<PoolRef, PoolDataResult>, ClientEvmError>,
    FetchLogsRange: FnOnce(u64, u64) -> Result<Vec<RangeLogBlock>, ClientEvmError>,
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
                            number: header.inner.inner.number,
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
            AnyIssuedRequest::LogsRange(request) => {
                let request_id = request.request_id;
                let from = request.request_payload.from_block();
                let to = request.request_payload.to_block();
                // The rare authoritative path (finalization verification / hole backstop) is worth
                // a line at issuance — otherwise it is only visible when it fails.
                logger.log(&format_logs_range_effect_log(
                    chain,
                    &request_id,
                    from,
                    to,
                    request.request_payload.covered().len(),
                ));

                match fetch_logs_range(from, to) {
                    Ok(blocks) => vec![Event::ChainEvent {
                        chain,
                        event: kernel::Event::BlockLogsRangeReceived {
                            request_id,
                            blocks: blocks
                                .into_iter()
                                .map(|block| (block.hash, block.logs))
                                .collect(),
                        },
                    }],
                    Err(error) => {
                        let request_id = AnyRequestId::LogsRange(request_id);
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
            AnyIssuedRequest::CanonicalHeader(request) => {
                let request_id = request.request_id;
                let number = request.request_payload.number;

                match fetch_canonical_header(number) {
                    Ok(Some(header)) => vec![Event::ChainEvent {
                        chain,
                        event: kernel::Event::CanonicalHeaderAtHeightReceived {
                            request_id,
                            hash: header.inner.hash,
                            number: header.inner.inner.number,
                        },
                    }],
                    // The height is not yet available on the answering endpoint: retry (the anchor's
                    // height is below the tip, so this is transient provider lag, not a real absence).
                    Ok(None) | Err(_) => {
                        let request_id = AnyRequestId::CanonicalHeader(request_id);
                        logger.log(&format!(
                            "error chain={chain:?} request_failed request={request_id:?} finality_probe number={number}"
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
        Bloom, ConfigScope, GetBlockHeader, GetBlockLogs, GetCanonicalHeaderAtHeight, GetLogsRange,
        GetPoolData, GetPoolMetadata, GetTokenMetadata, IssuedRequest, PoolRef, ProtocolPoolKey,
        PoolFee, PoolLog, PoolMetadata, PoolMetadataResult, RequestId, TokenAddress,
        TokenMetadataResult, UniswapV3Fee,
    };
    use serde_json::json;
    use std::{sync::mpsc, time::Duration};

    use super::*;

    #[test]
    fn next_reconnect_delay_starts_at_base_then_doubles_to_the_cap() {
        let short = Duration::from_millis(0);

        // First drop: base, no prior delay.
        let first = next_reconnect_delay(None, short);
        assert_eq!(first, RECONNECT_BASE);

        // Consecutive short-lived drops double the delay until the cap, then stay there.
        let mut delay = first;
        for _ in 0..20 {
            delay = next_reconnect_delay(Some(delay), short);
            assert!(delay <= RECONNECT_CAP, "delay exceeded cap: {delay:?}");
        }
        assert_eq!(delay, RECONNECT_CAP);
    }

    #[test]
    fn next_reconnect_delay_resets_to_base_after_a_healthy_run() {
        let grown = RECONNECT_CAP;
        let healthy = RECONNECT_STABILITY_WINDOW;

        assert_eq!(next_reconnect_delay(Some(grown), healthy), RECONNECT_BASE);
    }

    #[test]
    fn format_gauge_log_renders_active_and_initializing_chains() {
        let observations = vec![
            (
                ChainKey::Ethereum,
                ChainObservation::Active(ChainProgress {
                    verified_pools: 37,
                    blocks_behind_tip: Some(4),
                    canonical_window: Some(70),
                    in_flight_requests: 12,
                    ws_misses: 2,
                }),
            ),
            (
                ChainKey::Base,
                ChainObservation::Initializing { buffered_events: 9 },
            ),
            (
                ChainKey::Arbitrum,
                ChainObservation::Active(ChainProgress {
                    verified_pools: 0,
                    blocks_behind_tip: None,
                    canonical_window: None,
                    in_flight_requests: 0,
                    ws_misses: 0,
                }),
            ),
        ];

        assert_eq!(
            format_gauge_log(&observations, 153, 31),
            "gauge Ethereum=active behind=4 window=70 pools=37 inflight=12 ws_miss=2  \
             Base=init buffered=9  \
             Arbitrum=active behind=? window=? pools=0 inflight=0 ws_miss=0  \
             events=153 max_ms=31"
        );
    }

    #[test]
    fn slow_transition_log_fires_at_the_threshold_and_echoes_the_input_line() {
        let threshold_micros = SLOW_TRANSITION_WARN_MILLIS * 1000;

        assert_eq!(
            slow_transition_log(threshold_micros, "input chain=Arbitrum tick"),
            Some(format!(
                "warn slow_transition ms={SLOW_TRANSITION_WARN_MILLIS} input chain=Arbitrum tick"
            ))
        );
        assert_eq!(
            slow_transition_log(threshold_micros - 1, "input chain=Arbitrum tick"),
            None
        );
    }

    #[test]
    fn runtime_constructor_stores_subscriptions() {
        let runtime = ClientEvmRuntime::new(
            test_subscriptions(),
            test_endpoints(),
            GraphEndpoints::empty(),
            in_memory_metadata_cache(),
            test_session_config(),
            None,
            Logger::sink(),
            View::sink(),
        );

        assert_eq!(
            runtime
                .subscriptions()
                .ws_endpoints(ChainKey::Ethereum)
                .expect("ethereum ws")
                .iter()
                .map(|spec| spec.url.as_str())
                .collect::<Vec<_>>(),
            vec!["wss://example.invalid/ws"]
        );
    }

    #[test]
    fn runtime_sizes_the_effect_pool_for_io() {
        let runtime = ClientEvmRuntime::new(
            test_subscriptions(),
            test_endpoints(),
            GraphEndpoints::empty(),
            in_memory_metadata_cache(),
            test_session_config(),
            None,
            Logger::sink(),
            View::sink(),
        );

        // The blocking-I/O pool must be sized well above the CPU-bound default so a behind chain
        // cannot pin every worker; guards against a silent regression to the trait default.
        assert_eq!(runtime.effect_pool_size(), EFFECT_POOL_SIZE);
        assert!(runtime.effect_pool_size() > 8);
    }

    #[test]
    fn runtime_constructor_stores_graph_endpoints() {
        let runtime = ClientEvmRuntime::new(
            test_subscriptions(),
            test_endpoints(),
            GraphEndpoints::single(ChainKey::Ethereum, "thegraph", "http://graph.invalid"),
            in_memory_metadata_cache(),
            test_session_config(),
            None,
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
            test_subscriptions(),
            test_endpoints(),
            GraphEndpoints::empty(),
            in_memory_metadata_cache(),
            test_session_config(),
            None,
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
                number: 42,
            }),
            format!("input finalized_header_received chain=Ethereum block={block_hash} number=42")
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
            disabled_count: 0,
            pool_slots: 4,
            route_entropy: 1.1,
            effective_pools: 3.0,
            routed_pool_count: 3,
            iterations_completed: 10,
        };

        assert_eq!(
            format_input_log(&Event::OptimizationStepCompleted { result, plan: None }),
            "input optimization_step_completed status=Updated profit=12.5 reserves=4 routed=3 entropy=1.10 eff_pools=3.00 iterations=10 plan_steps=0"
        );
    }

    #[test]
    fn verification_log_puts_verified_next_to_claimed() {
        let result = optimization::OptimizationStepResult {
            status: optimization::OptimizationStepStatus::Updated,
            input_amount: 1_000.0,
            output_amount: 1_012.5,
            profit_amount: 12.5,
            reserves_count: 4,
            disabled_count: 0,
            pool_slots: 4,
            route_entropy: 1.1,
            effective_pools: 3.0,
            routed_pool_count: 3,
            iterations_completed: 10,
        };

        assert_eq!(
            format_verification_log(
                result,
                PlanVerification::Verified {
                    profit: 11.25,
                    hit_tick_limit: false,
                },
            ),
            "optimization plan_verification claimed=12.5 verified=11.25"
        );
        assert_eq!(
            format_verification_log(
                result,
                PlanVerification::Verified {
                    profit: -0.5,
                    hit_tick_limit: true,
                },
            ),
            "optimization plan_verification claimed=12.5 verified=-0.5 tick_limited"
        );
        assert_eq!(
            format_verification_log(
                result,
                PlanVerification::Unverifiable(
                    client_evm::multi_chain_kernel::PlanVerificationFailure::InitAssetUnknown,
                ),
            ),
            "optimization plan_verification claimed=12.5 unverifiable=InitAssetUnknown"
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
                    number: 1,
                },
            }),
            format!(
                "input chain=Ethereum head_observed hash={block_hash} parent={parent_hash} number=1"
            )
        );
        assert_eq!(
            format_input_log(&Event::ChainEvent {
                chain: ChainKey::Ethereum,
                event: kernel::Event::BlockHeaderReceived {
                    request_id,
                    hash: block_hash,
                    parent_hash,
                    logs_bloom: Bloom::default(),
                    number: 1,
                },
            }),
            format!(
                "input chain=Ethereum block_header_received request=7 hash={block_hash} parent={parent_hash} number=1"
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
    fn logs_range_effect_log_carries_the_request_id_and_window() {
        let request_id = RequestId::<GetLogsRange>::from_raw_for_test(9);

        assert_eq!(
            format_logs_range_effect_log(ChainKey::Arbitrum, &request_id, 100, 164, 3),
            "effect logs_range chain=Arbitrum request=9 from=100 to=164 covered=3"
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
            test_subscriptions(),
            test_endpoints(),
            GraphEndpoints::empty(),
            cache,
            test_session_config(),
            None,
            Logger::sink(),
            View::sink(),
        );

        // The scan yielded no candidates; the union must still surface the cached pool, resolved as
        // a hit. `test_endpoints()` points nowhere, so any RPC (a cache miss) would error instead.
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
    fn resolve_pool_metadata_routes_v3_to_rpc_and_v4_to_aggregator_and_merges() {
        let v3 = pool_candidate_address(2);
        let v4 = v4_pool_candidate(3);

        let result = resolve_pool_metadata_with(
            ChainKey::Ethereum,
            HashSet::from([v3, v4]),
            &Logger::sink(),
            |_candidates| Ok::<_, &str>(HashMap::new()),
            |_metadata| Ok::<(), &str>(()),
            |v3_misses| {
                assert_eq!(v3_misses, HashSet::from([v3]), "only v3 misses go to RPC");
                Ok(HashMap::from([(v3, Ok(pool_metadata(2)))]))
            },
            |v4_misses| {
                assert_eq!(
                    v4_misses,
                    HashSet::from([v4]),
                    "only v4 misses go to the aggregator"
                );
                Ok(HashMap::from([(v4, Ok(pool_metadata(3)))]))
            },
        )
        .expect("resolve succeeds");

        assert_eq!(result.get(&v3), Some(&Ok(pool_metadata(2))));
        assert_eq!(result.get(&v4), Some(&Ok(pool_metadata(3))));
    }

    #[test]
    fn resolve_pool_metadata_serves_cache_hits_without_fetching() {
        let v4 = v4_pool_candidate(3);

        let result = resolve_pool_metadata_with(
            ChainKey::Ethereum,
            HashSet::from([v4]),
            &Logger::sink(),
            |_candidates| Ok::<_, &str>(HashMap::from([(v4, pool_metadata(3))])),
            |_metadata| Ok::<(), &str>(()),
            |v3_misses| {
                assert!(v3_misses.is_empty());
                Ok(HashMap::new())
            },
            |v4_misses| {
                assert!(v4_misses.is_empty(), "a cached pool must not be re-fetched");
                Ok(HashMap::new())
            },
        )
        .expect("resolve succeeds");

        assert_eq!(result.get(&v4), Some(&Ok(pool_metadata(3))));
    }

    #[test]
    fn resolve_pool_metadata_writes_fetched_results_to_cache() {
        let v4 = v4_pool_candidate(3);
        let stored = std::cell::RefCell::new(None);

        let _ = resolve_pool_metadata_with(
            ChainKey::Ethereum,
            HashSet::from([v4]),
            &Logger::sink(),
            |_candidates| Ok::<_, &str>(HashMap::new()),
            |metadata| {
                *stored.borrow_mut() = Some(metadata.clone());
                Ok::<(), &str>(())
            },
            |_v3_misses| Ok(HashMap::new()),
            |_v4_misses| Ok(HashMap::from([(v4, Ok(pool_metadata(3)))])),
        )
        .expect("resolve succeeds");

        let stored = stored.borrow();
        let stored = stored.as_ref().expect("store must be called with the fetched results");
        assert_eq!(stored.get(&v4), Some(&Ok(pool_metadata(3))));
    }

    #[test]
    fn cached_pool_metadata_serves_a_cached_v4_pool_without_network() {
        let cache = in_memory_metadata_cache();
        let v4 = v4_pool_candidate(7);
        cache
            .store_pool_metadata(
                ChainKey::Ethereum,
                &HashMap::from([(v4, Ok(pool_metadata(7)))]),
            )
            .expect("store cached v4 pool");

        let runtime = ClientEvmRuntime::new(
            test_subscriptions(),
            test_endpoints(),
            GraphEndpoints::empty(),
            cache,
            test_session_config(),
            None,
            Logger::sink(),
            View::sink(),
        );

        // The pool is a cache hit, so neither RPC (which points nowhere) nor the aggregator (the graph
        // is unconfigured) is contacted — proving v4 metadata round-trips through the persistent cache.
        let result = runtime
            .cached_pool_metadata(ChainKey::Ethereum, hash(1), HashSet::from([v4]))
            .expect("resolve succeeds");

        assert_eq!(result.get(&v4), Some(&Ok(pool_metadata(7))));
    }

    #[test]
    fn cached_pool_metadata_gates_pools_over_non_whitelisted_tokens() {
        let cache = in_memory_metadata_cache();
        let v4 = v4_pool_candidate(7);
        cache
            .store_pool_metadata(
                ChainKey::Ethereum,
                &HashMap::from([(v4, Ok(pool_metadata(7)))]),
            )
            .expect("store cached v4 pool");

        // The cache holds the true Ok metadata, but the whitelist (init asset only) does not
        // allow this pool's tokens — the gate must reject it on the way out, exactly as if
        // metadata resolution had failed, without the cache entry being disturbed.
        let whitelist = whitelist_of(&[ETHEREUM_USDC_TOKEN_ADDRESS]);
        let runtime = ClientEvmRuntime::new(
            test_subscriptions(),
            test_endpoints(),
            GraphEndpoints::empty(),
            cache,
            test_session_config(),
            Some(whitelist),
            Logger::sink(),
            View::sink(),
        );

        let result = runtime
            .cached_pool_metadata(ChainKey::Ethereum, hash(1), HashSet::from([v4]))
            .expect("resolve succeeds");

        assert_eq!(
            result.get(&v4),
            Some(&Err(client_evm::PoolMetadataFailure::TokenNotWhitelisted {
                token: pool_metadata(7).token0
            }))
        );
    }

    #[test]
    fn cached_pool_metadata_passes_pools_over_whitelisted_tokens() {
        let cache = in_memory_metadata_cache();
        let v4 = v4_pool_candidate(7);
        cache
            .store_pool_metadata(
                ChainKey::Ethereum,
                &HashMap::from([(v4, Ok(pool_metadata(7)))]),
            )
            .expect("store cached v4 pool");

        let metadata = pool_metadata(7);
        let whitelist = whitelist_of(&[
            ETHEREUM_USDC_TOKEN_ADDRESS,
            TokenAddress(metadata.token0, ChainKey::Ethereum),
            TokenAddress(metadata.token1, ChainKey::Ethereum),
        ]);
        let runtime = ClientEvmRuntime::new(
            test_subscriptions(),
            test_endpoints(),
            GraphEndpoints::empty(),
            cache,
            test_session_config(),
            Some(whitelist),
            Logger::sink(),
            View::sink(),
        );

        let result = runtime
            .cached_pool_metadata(ChainKey::Ethereum, hash(1), HashSet::from([v4]))
            .expect("resolve succeeds");

        assert_eq!(result.get(&v4), Some(&Ok(pool_metadata(7))));
    }

    #[test]
    fn run_optimization_effect_returns_no_events() {
        let runtime = ClientEvmRuntime::new(
            test_subscriptions(),
            test_endpoints(),
            GraphEndpoints::empty(),
            in_memory_metadata_cache(),
            test_session_config(),
            None,
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
        let config = test_session_config();

        // Ethereum USDC is the single global quote/init asset; the bidirectional 1:1 bridge to
        // Arbitrum USDC lets the solver close cross-chain cycles back to it.
        assert_eq!(config.source_asset, ETHEREUM_USDC_TOKEN_ADDRESS);
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
    fn default_session_config_bridges_native_eth_and_weth_both_ways() {
        let config = test_session_config();

        // Wrapping is 1:1, so native ETH (v4 `token0 = address(0)`) and WETH (v3) must be a
        // two-sided bridge on every ETH-native chain; otherwise v4 native-ETH pools are isolated
        // from WETH liquidity.
        for (native, weth) in [
            (ETHEREUM_NATIVE_TOKEN_ADDRESS, ETHEREUM_WETH_TOKEN_ADDRESS),
            (ARBITRUM_NATIVE_TOKEN_ADDRESS, ARBITRUM_WETH_TOKEN_ADDRESS),
        ] {
            assert!(config.bridges.contains(&(native, weth)));
            assert!(config.bridges.contains(&(weth, native)));
        }

        // Native currencies are never bridged directly across chains.
        assert!(
            !config
                .bridges
                .contains(&(ETHEREUM_NATIVE_TOKEN_ADDRESS, ARBITRUM_NATIVE_TOKEN_ADDRESS))
        );
    }

    #[test]
    fn default_session_config_includes_cross_chain_exposure_bridges_both_ways() {
        let config = test_session_config();
        let pairs = [
            // USDT exposure.
            (ETHEREUM_USDT_TOKEN_ADDRESS, ARBITRUM_USDT_TOKEN_ADDRESS),
            (ETHEREUM_USDT_TOKEN_ADDRESS, OPTIMISM_USDT_TOKEN_ADDRESS),
            (ETHEREUM_USDT_TOKEN_ADDRESS, POLYGON_USDT_TOKEN_ADDRESS),
            (ETHEREUM_USDT_TOKEN_ADDRESS, BNB_USDT_TOKEN_ADDRESS),
            (ETHEREUM_USDT_TOKEN_ADDRESS, AVALANCHE_USDT_TOKEN_ADDRESS),
            // BTC exposure.
            (ETHEREUM_WBTC_TOKEN_ADDRESS, ARBITRUM_WBTC_TOKEN_ADDRESS),
            (ETHEREUM_WBTC_TOKEN_ADDRESS, BASE_CBBTC_TOKEN_ADDRESS),
            (ETHEREUM_WBTC_TOKEN_ADDRESS, OPTIMISM_WBTC_TOKEN_ADDRESS),
            (ETHEREUM_WBTC_TOKEN_ADDRESS, POLYGON_WBTC_TOKEN_ADDRESS),
            (ETHEREUM_WBTC_TOKEN_ADDRESS, BNB_BTCB_TOKEN_ADDRESS),
            (ETHEREUM_WBTC_TOKEN_ADDRESS, AVALANCHE_WBTC_TOKEN_ADDRESS),
            // ETH exposure through wrapped ETH variants.
            (ETHEREUM_WETH_TOKEN_ADDRESS, ARBITRUM_WETH_TOKEN_ADDRESS),
            (ETHEREUM_WETH_TOKEN_ADDRESS, BASE_WETH_TOKEN_ADDRESS),
            (ETHEREUM_WETH_TOKEN_ADDRESS, OPTIMISM_WETH_TOKEN_ADDRESS),
            (ETHEREUM_WETH_TOKEN_ADDRESS, POLYGON_WETH_TOKEN_ADDRESS),
            (ETHEREUM_WETH_TOKEN_ADDRESS, BNB_WETH_TOKEN_ADDRESS),
            (ETHEREUM_WETH_TOKEN_ADDRESS, AVALANCHE_WETH_TOKEN_ADDRESS),
            // DAI exposure.
            (ETHEREUM_DAI_TOKEN_ADDRESS, OPTIMISM_DAI_TOKEN_ADDRESS),
        ];

        for (ethereum, remote) in pairs {
            assert!(
                config.bridges.contains(&(ethereum, remote)),
                "missing outbound bridge {ethereum:?} -> {remote:?}"
            );
            assert!(
                config.bridges.contains(&(remote, ethereum)),
                "missing return bridge {remote:?} -> {ethereum:?}"
            );
        }
    }

    /// Builds a validated whitelist allowing exactly `tokens`, going through the same
    /// file-schema path production uses.
    fn whitelist_of(tokens: &[TokenAddress]) -> TokenWhitelist {
        let mut chains: std::collections::BTreeMap<String, client_evm::ChainTokens> =
            std::collections::BTreeMap::new();
        for token in tokens {
            chains
                .entry(client_evm::drpc_network_path(token.1).to_string())
                .or_insert_with(|| client_evm::ChainTokens { tokens: Vec::new() })
                .tokens
                .push(client_evm::TokenEntry {
                    address: token.0,
                    symbol: None,
                    decimals: None,
                    examined_at: None,
                    tvl_usd: None,
                });
        }
        client_evm::TokenWhitelistFile {
            generated_at: None,
            examiner: None,
            chains,
        }
        .into_whitelist()
        .expect("valid whitelist")
    }

    #[test]
    fn session_config_without_whitelist_reproduces_the_default() {
        let (config, dropped) =
            optimization_session_config(None).expect("whitelist-free config");

        assert_eq!(config.source_asset, ETHEREUM_USDC_TOKEN_ADDRESS);
        assert_eq!(config.bridges, default_optimization_bridges());
        assert_eq!(config.whitelist, None);
        assert!(dropped.is_empty());
    }

    #[test]
    fn session_config_rejects_a_whitelist_without_the_init_asset() {
        let whitelist = whitelist_of(&[ETHEREUM_NATIVE_TOKEN_ADDRESS]);

        assert!(matches!(
            optimization_session_config(Some(&whitelist)),
            Err(CliError::InitAssetNotWhitelisted { .. })
        ));
    }

    #[test]
    fn session_config_drops_bridges_with_a_non_whitelisted_endpoint() {
        let allowed = [
            ETHEREUM_USDC_TOKEN_ADDRESS,
            ARBITRUM_USDC_TOKEN_ADDRESS,
            ETHEREUM_NATIVE_TOKEN_ADDRESS,
        ];
        let whitelist = whitelist_of(&allowed);

        let (config, dropped) =
            optimization_session_config(Some(&whitelist)).expect("whitelisted config");

        // Only the Ethereum⇄Arbitrum USDC pair survives: every other bridge touches a token
        // outside the whitelist (including ETH↔WETH, since WETH isn't allowed).
        assert_eq!(
            config.bridges,
            HashSet::from([
                (ETHEREUM_USDC_TOKEN_ADDRESS, ARBITRUM_USDC_TOKEN_ADDRESS),
                (ARBITRUM_USDC_TOKEN_ADDRESS, ETHEREUM_USDC_TOKEN_ADDRESS),
            ])
        );
        assert_eq!(
            dropped.len(),
            default_optimization_bridges().len() - config.bridges.len()
        );
        assert_eq!(
            config.whitelist,
            Some(HashSet::from(allowed)),
            "the optimizer receives exactly the whitelisted token set"
        );
    }

    #[test]
    fn empty_optimization_snapshot_is_dropped() {
        let runtime = ClientEvmRuntime::new(
            test_subscriptions(),
            test_endpoints(),
            GraphEndpoints::empty(),
            in_memory_metadata_cache(),
            test_session_config(),
            None,
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
            test_subscriptions(),
            test_endpoints(),
            GraphEndpoints::empty(),
            in_memory_metadata_cache(),
            test_session_config(),
            None,
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
            test_subscriptions(),
            test_endpoints(),
            GraphEndpoints::empty(),
            in_memory_metadata_cache(),
            test_session_config(),
            None,
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
            test_subscriptions(),
            test_endpoints(),
            GraphEndpoints::empty(),
            in_memory_metadata_cache(),
            test_session_config(),
            None,
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
            test_subscriptions(),
            test_endpoints(),
            GraphEndpoints::empty(),
            in_memory_metadata_cache(),
            test_session_config(),
            None,
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
        let handle = spawn_optimization_subscription(
            slot_receiver,
            test_session_config(),
            sender,
            Logger::sink(),
        );

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
            unexpected_pool_metadata_fetch,
            unexpected_token_metadata_fetch,
            unexpected_pool_data_fetch,
            unexpected_logs_range_fetch,
            unexpected_canonical_header_fetch,
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
            unexpected_pool_metadata_fetch,
            unexpected_token_metadata_fetch,
            unexpected_pool_data_fetch,
            unexpected_logs_range_fetch,
            unexpected_canonical_header_fetch,
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
            unexpected_pool_metadata_fetch,
            unexpected_token_metadata_fetch,
            unexpected_pool_data_fetch,
            unexpected_logs_range_fetch,
            unexpected_canonical_header_fetch,
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
            unexpected_pool_metadata_fetch,
            unexpected_token_metadata_fetch,
            unexpected_pool_data_fetch,
            unexpected_logs_range_fetch,
            unexpected_canonical_header_fetch,
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
            unexpected_pool_metadata_fetch,
            unexpected_token_metadata_fetch,
            unexpected_pool_data_fetch,
            unexpected_logs_range_fetch,
            unexpected_canonical_header_fetch,
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
            |requested_at, requested_candidates| {
                assert_eq!(requested_at, at);
                assert_eq!(requested_candidates, HashSet::from([candidate]));
                Ok(HashMap::new())
            },
            unexpected_token_metadata_fetch,
            unexpected_pool_data_fetch,
            unexpected_logs_range_fetch,
            unexpected_canonical_header_fetch,
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
            |requested_at, requested_candidates| {
                assert_eq!(requested_at, at);
                assert_eq!(requested_candidates, candidates);
                Err(ClientEvmError::InvalidConfig {
                    scope: ConfigScope::Http,
                    reason: "bad config".to_owned(),
                })
            },
            unexpected_token_metadata_fetch,
            unexpected_pool_data_fetch,
            unexpected_logs_range_fetch,
            unexpected_canonical_header_fetch,
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
            unexpected_pool_metadata_fetch,
            |requested_at, requested_tokens| {
                assert_eq!(requested_at, at);
                assert_eq!(requested_tokens, HashSet::from([token]));
                Ok(HashMap::new())
            },
            unexpected_pool_data_fetch,
            unexpected_logs_range_fetch,
            unexpected_canonical_header_fetch,
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
            unexpected_pool_metadata_fetch,
            |requested_at, requested_tokens| {
                assert_eq!(requested_at, at);
                assert_eq!(requested_tokens, tokens);
                Err(ClientEvmError::InvalidConfig {
                    scope: ConfigScope::Http,
                    reason: "bad config".to_owned(),
                })
            },
            unexpected_pool_data_fetch,
            unexpected_logs_range_fetch,
            unexpected_canonical_header_fetch,
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

    #[test]
    fn canonical_header_probe_success_maps_to_chain_event() -> Result<(), serde_json::Error> {
        let chain = ChainKey::Ethereum;
        let probed_hash = hash(2);
        let request_id = RequestId::<GetCanonicalHeaderAtHeight>::from_raw_for_test(12);
        let effect = kernel::Effect::Request(AnyIssuedRequest::CanonicalHeader(IssuedRequest {
            request_id,
            request_payload: GetCanonicalHeaderAtHeight { number: 9 },
        }));
        let header = block_header(probed_hash, hash(4))?; // header number is 0x9 = 9

        let events = execute_chain_effect_with(
            chain,
            effect,
            &Logger::sink(),
            unexpected_block_header_fetch,
            unexpected_block_logs_fetch,
            unexpected_pool_metadata_fetch,
            unexpected_token_metadata_fetch,
            unexpected_pool_data_fetch,
            unexpected_logs_range_fetch,
            |number| {
                assert_eq!(number, 9);
                Ok(Some(header))
            },
        );

        assert!(matches!(
            events.as_slice(),
            [Event::ChainEvent {
                chain: event_chain,
                event:
                    kernel::Event::CanonicalHeaderAtHeightReceived {
                        hash: event_hash,
                        number: event_number,
                        ..
                    },
            }] if *event_chain == chain && *event_hash == probed_hash && *event_number == 9
        ));
        Ok(())
    }

    #[test]
    fn canonical_header_probe_absent_maps_to_request_failed() {
        let chain = ChainKey::Ethereum;
        let request_id = RequestId::<GetCanonicalHeaderAtHeight>::from_raw_for_test(13);
        let effect = kernel::Effect::Request(AnyIssuedRequest::CanonicalHeader(IssuedRequest {
            request_id,
            request_payload: GetCanonicalHeaderAtHeight { number: 42 },
        }));

        let events = execute_chain_effect_with(
            chain,
            effect,
            &Logger::sink(),
            unexpected_block_header_fetch,
            unexpected_block_logs_fetch,
            unexpected_pool_metadata_fetch,
            unexpected_token_metadata_fetch,
            unexpected_pool_data_fetch,
            unexpected_logs_range_fetch,
            |_number| Ok(None),
        );

        assert!(matches!(
            events.as_slice(),
            [Event::ChainEvent {
                event: kernel::Event::RequestFailed { .. },
                ..
            }]
        ));
    }

    fn block_header_request_effect(
        block_hash: BlockHash,
    ) -> (kernel::Effect, RequestId<GetBlockHeader>) {
        let finalized_hash = hash(1);
        let observed_hash = hash(3);
        let state = kernel::State::init(finalized_hash, 1);

        let (_state, effects) = kernel::transition(
            ChainKey::Ethereum,
            state,
            kernel::Event::HeadObserved {
                hash: observed_hash,
                parent_hash: block_hash,
                logs_bloom: Bloom::default(),
                number: 3,
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

    fn pool_data_request_effect(
        at: BlockHash,
        pools: HashSet<PoolRef>,
    ) -> (kernel::Effect, RequestId<GetPoolData>) {
        let request_id = RequestId::from_raw_for_test(11);
        (
            kernel::Effect::Request(AnyIssuedRequest::PoolData(IssuedRequest {
                request_id,
                request_payload: GetPoolData { at, pools },
            })),
            request_id,
        )
    }

    fn logs_range_request_effect(
        from: u64,
        to: u64,
        covered: HashSet<BlockHash>,
    ) -> (kernel::Effect, RequestId<GetLogsRange>) {
        let request_id = RequestId::from_raw_for_test(12);
        (
            kernel::Effect::Request(AnyIssuedRequest::LogsRange(IssuedRequest {
                request_id,
                request_payload: GetLogsRange::new(from, to, covered)
                    .expect("test window is ordered"),
            })),
            request_id,
        )
    }

    #[test]
    fn logs_range_request_success_maps_to_chain_event_grouped_by_hash() {
        let chain = ChainKey::Ethereum;
        let block_hash = hash(7);
        let (effect, expected_request_id) =
            logs_range_request_effect(20, 21, HashSet::from([block_hash]));

        let events = execute_chain_effect_with(
            chain,
            effect,
            &Logger::sink(),
            unexpected_block_header_fetch,
            unexpected_block_logs_fetch,
            unexpected_pool_metadata_fetch,
            unexpected_token_metadata_fetch,
            unexpected_pool_data_fetch,
            |from, to| {
                assert_eq!(from, 20);
                assert_eq!(to, 21);
                Ok(vec![RangeLogBlock {
                    number: 20,
                    hash: block_hash,
                    logs: Vec::new(),
                }])
            },
            unexpected_canonical_header_fetch,
        );

        assert!(matches!(
            events.as_slice(),
            [Event::ChainEvent {
                chain: event_chain,
                event:
                    kernel::Event::BlockLogsRangeReceived {
                        request_id,
                        blocks,
                    },
            }] if *event_chain == chain
                && *request_id == expected_request_id
                && blocks.as_slice() == [(block_hash, Vec::new())]
        ));
    }

    #[test]
    fn logs_range_request_failure_maps_to_request_failed_event() {
        let chain = ChainKey::Ethereum;
        let (effect, expected_request_id) =
            logs_range_request_effect(20, 21, HashSet::from([hash(7)]));

        let events = execute_chain_effect_with(
            chain,
            effect,
            &Logger::sink(),
            unexpected_block_header_fetch,
            unexpected_block_logs_fetch,
            unexpected_pool_metadata_fetch,
            unexpected_token_metadata_fetch,
            unexpected_pool_data_fetch,
            |_from, _to| {
                Err(ClientEvmError::InvalidConfig {
                    scope: ConfigScope::Http,
                    reason: "bad config".to_owned(),
                })
            },
            unexpected_canonical_header_fetch,
        );

        assert!(matches!(
            events.as_slice(),
            [Event::ChainEvent {
                chain: event_chain,
                event:
                    kernel::Event::RequestFailed {
                        request_id: client_evm::AnyRequestId::LogsRange(request_id),
                    },
            }] if *event_chain == chain && *request_id == expected_request_id
        ));
    }

    #[test]
    fn pool_data_request_success_maps_to_chain_event() {
        let chain = ChainKey::Ethereum;
        let at = hash(2);
        let pool = pool_address(3);
        let (effect, expected_request_id) = pool_data_request_effect(at, HashSet::from([pool]));

        let events = execute_chain_effect_with(
            chain,
            effect,
            &Logger::sink(),
            unexpected_block_header_fetch,
            unexpected_block_logs_fetch,
            unexpected_pool_metadata_fetch,
            unexpected_token_metadata_fetch,
            |requested_at, requested_pools| {
                assert_eq!(requested_at, at);
                assert_eq!(requested_pools, HashSet::from([pool]));
                Ok(HashMap::new())
            },
            unexpected_logs_range_fetch,
            unexpected_canonical_header_fetch,
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
        let pool = pool_address(3);
        let (effect, expected_request_id) = pool_data_request_effect(at, HashSet::from([pool]));

        let events = execute_chain_effect_with(
            chain,
            effect,
            &Logger::sink(),
            unexpected_block_header_fetch,
            unexpected_block_logs_fetch,
            unexpected_pool_metadata_fetch,
            unexpected_token_metadata_fetch,
            |requested_at, requested_pools| {
                assert_eq!(requested_at, at);
                assert_eq!(requested_pools, HashSet::from([pool]));
                Err(ClientEvmError::InvalidConfig {
                    scope: ConfigScope::Http,
                    reason: "bad config".to_owned(),
                })
            },
            unexpected_logs_range_fetch,
            unexpected_canonical_header_fetch,
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

    fn unexpected_block_header_fetch(
        _block_hash: BlockHash,
    ) -> Result<Option<ClientHead>, ClientEvmError> {
        panic!("block header fetch must not be called")
    }

    fn unexpected_block_logs_fetch(_block_hash: BlockHash) -> Result<Vec<PoolLog>, ClientEvmError> {
        panic!("block logs fetch must not be called")
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

    fn unexpected_pool_data_fetch(
        _at: BlockHash,
        _pools: HashSet<PoolRef>,
    ) -> Result<HashMap<PoolRef, PoolDataResult>, ClientEvmError> {
        panic!("pool data fetch must not be called")
    }

    fn unexpected_logs_range_fetch(
        _from: u64,
        _to: u64,
    ) -> Result<Vec<RangeLogBlock>, ClientEvmError> {
        panic!("ranged logs fetch must not be called")
    }

    fn unexpected_canonical_header_fetch(
        _number: u64,
    ) -> Result<Option<ClientHead>, ClientEvmError> {
        panic!("canonical-header probe must not be called")
    }

    fn pool_candidate_address(last_byte: u8) -> ProtocolPoolKey {
        let address = format!("0x{}", format!("{last_byte:040x}"))
            .parse()
            .expect("test address must parse");

        ProtocolPoolKey::UniswapV3(address)
    }

    fn v4_pool_candidate(last_byte: u8) -> ProtocolPoolKey {
        ProtocolPoolKey::UniswapV4(client_evm::uniswap_v4::PoolId(hash(last_byte)))
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

    fn test_subscriptions() -> ChainSubscriptions {
        let mut ws = std::collections::BTreeMap::new();
        for &chain in client_evm::ACTIVE_CHAINS {
            ws.insert(
                chain,
                vec![client_evm::EndpointSpec::new(
                    "test",
                    "wss://example.invalid/ws",
                    1,
                )],
            );
        }
        ChainSubscriptions::new(ws).expect("test subscriptions")
    }

    // Endpoint pools pointing at an (unreachable) test URL — so any RPC a test triggers errors instead
    // of escaping to a real provider.
    fn test_endpoints() -> ChainEndpoints {
        let mut specs = std::collections::BTreeMap::new();
        for &chain in client_evm::ACTIVE_CHAINS {
            specs.insert(
                chain,
                vec![client_evm::EndpointSpec::new(
                    "test",
                    "https://example.invalid/http",
                    1,
                )],
            );
        }
        client_evm::assemble_chain_endpoints(&specs).expect("test endpoint assembly")
    }

    fn in_memory_metadata_cache() -> MetadataCache {
        MetadataCache::in_memory().expect("in-memory metadata cache")
    }

    fn test_session_config() -> OptimizationSessionConfig<TokenAddress> {
        let (config, _dropped) =
            optimization_session_config(None).expect("whitelist-free session config");
        config
    }
}
