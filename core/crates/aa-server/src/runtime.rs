//! The runtime adapter: the thin I/O shell around the pure [`crate::core`].
//!
//! Mirrors `aa-cli`'s runtime for a single chain — it executes the kernel's RPC-request effects and
//! the finalized-header poll through the shared `client_evm` fetch functions, and drives the WS
//! new-heads stream and the tick. It holds no optimization overlay, view, or multi-chain container.

use std::{
    collections::{HashMap, HashSet},
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc::Sender,
    },
    thread,
    time::{Duration, Instant},
};

use aa_framework::{Application, Runtime, Transition};
use client_evm::{
    AnyIssuedRequest, AnyRequestId, BlockHash, ChainEndpoints, ClientEvent, ClientEvmError,
    ClientHead, PoolDataResult, PoolLog, PoolMetadataResult, PoolRef, ProtocolPoolKey, RangeLogBlock,
    TokenAddress, TokenMetadataResult, fetch_block_header, fetch_block_logs,
    fetch_canonical_block_header_at, fetch_finalized_block_header, fetch_pool_data,
    fetch_pool_logs_in_range, fetch_pool_metadata, fetch_token_metadata, kernel, subscribe_new_heads,
};

use crate::core::{
    AnchorHeader, CHAIN, ServerEffect, ServerInput, ServerState, server_init, server_transition,
};

/// The single-chain server application (pure core wired to the framework).
pub struct ServerApp;

/// The framework subscriptions the server seeds: the live head stream and the driver tick.
pub enum ServerSubscription {
    NewHeads,
    Tick(Duration),
}

impl Application for ServerApp {
    type State = ServerState;
    type Input = ServerInput;
    type Effect = ServerEffect;
    type Subscription = ServerSubscription;

    fn init() -> Transition<ServerState, ServerEffect> {
        let (state, effects) = server_init();
        Transition { state, effects }
    }

    fn transition(state: ServerState, input: ServerInput) -> Transition<ServerState, ServerEffect> {
        let (state, effects) = server_transition(state, input);
        Transition { state, effects }
    }

    fn subscriptions() -> Vec<ServerSubscription> {
        vec![
            ServerSubscription::NewHeads,
            ServerSubscription::Tick(Duration::from_millis(1000)),
        ]
    }
}

/// The I/O runtime for [`ServerApp`]: one HTTP endpoint pool (with failover) plus one WS URL.
pub struct ServerRuntime {
    agent: ureq::Agent,
    endpoints: ChainEndpoints,
    ws_url: String,
    tick_interval: Duration,
    started: Instant,
    /// Wall-clock millis of the last emitted observation line, throttling the smoke log to ~1/s.
    last_log_millis: AtomicU64,
}

impl ServerRuntime {
    pub fn new(endpoints: ChainEndpoints, ws_url: String) -> ServerRuntime {
        ServerRuntime {
            agent: ureq::Agent::new_with_defaults(),
            endpoints,
            ws_url,
            tick_interval: Duration::from_millis(1000),
            started: Instant::now(),
            last_log_millis: AtomicU64::new(0),
        }
    }

    fn fetch_anchor(&self) -> Option<AnchorHeader> {
        match fetch_finalized_block_header(&self.agent, &self.endpoints, CHAIN) {
            Ok(Some(header)) => Some(AnchorHeader {
                hash: header.inner.hash,
                number: header.inner.inner.number,
            }),
            Ok(None) => None,
            Err(error) => {
                eprintln!("aa-server chain={CHAIN:?} finalized_fetch_failed error={error}");
                None
            }
        }
    }

    fn execute_kernel_request(&self, request: AnyIssuedRequest) -> kernel::Event {
        execute_request_with(
            request,
            |hash| fetch_block_header(&self.agent, &self.endpoints, CHAIN, hash),
            |hash| fetch_block_logs(&self.agent, &self.endpoints, CHAIN, hash),
            |at, candidates| fetch_pool_metadata(&self.agent, &self.endpoints, CHAIN, at, candidates),
            |at, tokens| fetch_token_metadata(&self.agent, &self.endpoints, CHAIN, at, tokens),
            |at, pools| fetch_pool_data(&self.agent, &self.endpoints, CHAIN, at, pools),
            |from, to| fetch_pool_logs_in_range(&self.agent, &self.endpoints, CHAIN, from, to),
            |number| fetch_canonical_block_header_at(&self.agent, &self.endpoints, CHAIN, number),
        )
    }
}

impl Runtime<ServerApp> for ServerRuntime {
    fn execute_effect(&self, effect: ServerEffect) -> Vec<ServerInput> {
        match effect {
            ServerEffect::FetchFinalizedHeader => {
                vec![ServerInput::FinalizedHeader(self.fetch_anchor())]
            }
            ServerEffect::Kernel(kernel::Effect::Request(request)) => {
                vec![ServerInput::Kernel(self.execute_kernel_request(request))]
            }
        }
    }

    fn spawn_subscription(&self, sender: &Sender<ServerInput>, subscription: ServerSubscription) {
        match subscription {
            ServerSubscription::NewHeads => {
                // One reconnecting WS connection; each new head is fed as a kernel `HeadObserved`.
                reconnect_loop(&format!("new_heads chain={CHAIN:?}"), || {
                    subscribe_new_heads(&self.ws_url, sender, map_new_head)
                });
            }
            ServerSubscription::Tick(interval) => loop {
                thread::sleep(interval);
                if sender.send(ServerInput::Tick).is_err() {
                    break;
                }
            },
        }
    }

    fn effect_pool_size(&self) -> usize {
        // Effects block on network I/O; size for the provider concurrency budget, not the CPU count.
        16
    }

    fn observe_state(&self, state: &ServerState) {
        // Throttle the smoke log to roughly once per second so warmup is visible without per-event spam.
        let now = self.started.elapsed().as_millis() as u64;
        if now.saturating_sub(self.last_log_millis.load(Ordering::Relaxed)) < self.tick_interval.as_millis() as u64
        {
            return;
        }
        self.last_log_millis.store(now, Ordering::Relaxed);

        match state {
            ServerState::AwaitingAnchor => {
                eprintln!("aa-server chain={CHAIN:?} status=awaiting_anchor")
            }
            ServerState::Running(kernel_state) => {
                let (finalized_hash, finalized_number) = kernel_state.finalized_head();
                let (_, frontier) = kernel_state.projected_pool_states(CHAIN);
                eprintln!(
                    "aa-server chain={CHAIN:?} status=running finalized={finalized_hash}@{finalized_number} \
                     canonical={} pools={} inflight={} behind={:?} ws_miss={} frontier={frontier}",
                    kernel_state.canonical_head(),
                    kernel_state.verified_pool_count(),
                    kernel_state.in_flight_request_count(),
                    kernel_state.blocks_behind(frontier),
                    kernel_state.ws_miss_count(),
                );
            }
        }
    }
}

/// Maps a live WS client event to a server input: only new heads matter to warmup; the pool-log,
/// subscribed, and closed events carry no kernel input on this stream.
fn map_new_head(event: ClientEvent) -> Option<ServerInput> {
    match event {
        ClientEvent::NewHead { header, .. } => Some(ServerInput::Kernel(kernel::Event::HeadObserved {
            hash: header.inner.hash,
            parent_hash: header.inner.inner.parent_hash,
            logs_bloom: header.inner.inner.logs_bloom,
            number: header.inner.inner.number,
        })),
        ClientEvent::PoolLogObserved { .. }
        | ClientEvent::Subscribed { .. }
        | ClientEvent::Closed { .. } => None,
    }
}

/// Reconnects forever with a fixed delay: a subscription attempt returns on close or error, and the
/// server is long-running, so a dead feed must always be retried (never silently give up).
fn reconnect_loop(kind: &str, mut attempt: impl FnMut() -> Result<(), ClientEvmError>) {
    loop {
        match attempt() {
            Ok(()) => eprintln!("aa-server subscription={kind} closed reconnecting"),
            Err(error) => eprintln!("aa-server subscription={kind} error={error} reconnecting"),
        }
        thread::sleep(Duration::from_millis(1000));
    }
}

/// Executes one kernel RPC request through injected fetchers, mapping the result to the kernel event
/// the response feeds back. Generic over the fetchers so the mapping is unit-testable with fakes; the
/// runtime passes the real `client_evm` fetch functions. Single-chain sibling of aa-cli's
/// `execute_chain_effect_with` (no `ChainEvent` wrapper).
#[allow(clippy::too_many_arguments)]
fn execute_request_with<FBH, FBL, FPM, FTM, FPD, FLR, FCH>(
    request: AnyIssuedRequest,
    fetch_block_header: FBH,
    fetch_block_logs: FBL,
    fetch_pool_metadata: FPM,
    fetch_token_metadata: FTM,
    fetch_pool_data: FPD,
    fetch_logs_range: FLR,
    fetch_canonical_header: FCH,
) -> kernel::Event
where
    FBH: FnOnce(BlockHash) -> Result<Option<ClientHead>, ClientEvmError>,
    FCH: FnOnce(u64) -> Result<Option<ClientHead>, ClientEvmError>,
    FBL: FnOnce(BlockHash) -> Result<Vec<PoolLog>, ClientEvmError>,
    FPM: FnOnce(
        BlockHash,
        HashSet<ProtocolPoolKey>,
    ) -> Result<HashMap<ProtocolPoolKey, PoolMetadataResult>, ClientEvmError>,
    FTM: FnOnce(
        BlockHash,
        HashSet<TokenAddress>,
    ) -> Result<HashMap<TokenAddress, TokenMetadataResult>, ClientEvmError>,
    FPD: FnOnce(
        BlockHash,
        HashSet<PoolRef>,
    ) -> Result<HashMap<PoolRef, PoolDataResult>, ClientEvmError>,
    FLR: FnOnce(u64, u64) -> Result<Vec<RangeLogBlock>, ClientEvmError>,
{
    match request {
        AnyIssuedRequest::BlockHeader(request) => {
            let request_id = request.request_id;
            match fetch_block_header(request.request_payload.block_hash) {
                Ok(Some(header)) => kernel::Event::BlockHeaderReceived {
                    request_id,
                    hash: header.inner.hash,
                    parent_hash: header.inner.inner.parent_hash,
                    logs_bloom: header.inner.inner.logs_bloom,
                    number: header.inner.inner.number,
                },
                Ok(None) => kernel::Event::BlockHeaderNotFound { request_id },
                Err(error) => request_failed(AnyRequestId::BlockHeader(request_id), &error),
            }
        }
        AnyIssuedRequest::BlockLogs(request) => {
            let request_id = request.request_id;
            match fetch_block_logs(request.request_payload.block_hash) {
                Ok(logs) => kernel::Event::BlockLogsReceived { request_id, logs },
                Err(error) => request_failed(AnyRequestId::BlockLogs(request_id), &error),
            }
        }
        AnyIssuedRequest::PoolMetadata(request) => {
            let request_id = request.request_id;
            let payload = request.request_payload;
            match fetch_pool_metadata(payload.at, payload.candidates) {
                Ok(metadata) => kernel::Event::PoolMetadataReceived {
                    request_id,
                    metadata,
                },
                Err(error) => request_failed(AnyRequestId::PoolMetadata(request_id), &error),
            }
        }
        AnyIssuedRequest::TokenMetadata(request) => {
            let request_id = request.request_id;
            let payload = request.request_payload;
            match fetch_token_metadata(payload.at, payload.tokens) {
                Ok(metadata) => kernel::Event::TokenMetadataReceived {
                    request_id,
                    metadata,
                },
                Err(error) => request_failed(AnyRequestId::TokenMetadata(request_id), &error),
            }
        }
        AnyIssuedRequest::PoolData(request) => {
            let request_id = request.request_id;
            let payload = request.request_payload;
            match fetch_pool_data(payload.at, payload.pools) {
                Ok(pools) => kernel::Event::PoolDataReceived { request_id, pools },
                Err(error) => request_failed(AnyRequestId::PoolData(request_id), &error),
            }
        }
        AnyIssuedRequest::LogsRange(request) => {
            let request_id = request.request_id;
            let from = request.request_payload.from_block();
            let to = request.request_payload.to_block();
            match fetch_logs_range(from, to) {
                Ok(blocks) => kernel::Event::BlockLogsRangeReceived {
                    request_id,
                    blocks: blocks
                        .into_iter()
                        .map(|block| (block.hash, block.logs))
                        .collect(),
                },
                Err(error) => request_failed(AnyRequestId::LogsRange(request_id), &error),
            }
        }
        AnyIssuedRequest::CanonicalHeader(request) => {
            let request_id = request.request_id;
            match fetch_canonical_header(request.request_payload.number) {
                Ok(Some(header)) => kernel::Event::CanonicalHeaderAtHeightReceived {
                    request_id,
                    hash: header.inner.hash,
                    number: header.inner.inner.number,
                },
                // Transient provider lag (the anchor height is below the tip): retry, don't treat as
                // a real absence.
                Ok(None) => kernel::Event::RequestFailed {
                    request_id: AnyRequestId::CanonicalHeader(request_id),
                },
                Err(error) => request_failed(AnyRequestId::CanonicalHeader(request_id), &error),
            }
        }
    }
}

fn request_failed(request_id: AnyRequestId, error: &ClientEvmError) -> kernel::Event {
    eprintln!("aa-server chain={CHAIN:?} request_failed request={request_id:?} error={error}");
    kernel::Event::RequestFailed { request_id }
}

#[cfg(test)]
mod tests {
    use super::*;
    use client_evm::{GetBlockHeader, GetBlockLogs, IssuedRequest, RequestId};

    fn hash(byte: u8) -> BlockHash {
        BlockHash::with_last_byte(byte)
    }

    fn unreachable_header(_: BlockHash) -> Result<Option<ClientHead>, ClientEvmError> {
        panic!("block header fetch should not be called")
    }
    fn unreachable_logs(_: BlockHash) -> Result<Vec<PoolLog>, ClientEvmError> {
        panic!("block logs fetch should not be called")
    }
    fn unreachable_pool_metadata(
        _: BlockHash,
        _: HashSet<ProtocolPoolKey>,
    ) -> Result<HashMap<ProtocolPoolKey, PoolMetadataResult>, ClientEvmError> {
        panic!("pool metadata fetch should not be called")
    }
    fn unreachable_token_metadata(
        _: BlockHash,
        _: HashSet<TokenAddress>,
    ) -> Result<HashMap<TokenAddress, TokenMetadataResult>, ClientEvmError> {
        panic!("token metadata fetch should not be called")
    }
    fn unreachable_pool_data(
        _: BlockHash,
        _: HashSet<PoolRef>,
    ) -> Result<HashMap<PoolRef, PoolDataResult>, ClientEvmError> {
        panic!("pool data fetch should not be called")
    }
    fn unreachable_logs_range(_: u64, _: u64) -> Result<Vec<RangeLogBlock>, ClientEvmError> {
        panic!("logs range fetch should not be called")
    }
    fn unreachable_canonical(_: u64) -> Result<Option<ClientHead>, ClientEvmError> {
        panic!("canonical header fetch should not be called")
    }

    /// Runs the executor with all fetchers unreachable except the block-header one.
    fn run_block_header(
        request: AnyIssuedRequest,
        fetch: impl FnOnce(BlockHash) -> Result<Option<ClientHead>, ClientEvmError>,
    ) -> kernel::Event {
        execute_request_with(
            request,
            fetch,
            unreachable_logs,
            unreachable_pool_metadata,
            unreachable_token_metadata,
            unreachable_pool_data,
            unreachable_logs_range,
            unreachable_canonical,
        )
    }

    #[test]
    fn block_header_absent_maps_to_not_found() {
        let request_id = RequestId::<GetBlockHeader>::from_raw_for_test(7);
        let request = AnyIssuedRequest::BlockHeader(IssuedRequest {
            request_id,
            request_payload: GetBlockHeader {
                block_hash: hash(1),
            },
        });

        let event = run_block_header(request, |_| Ok(None));

        assert!(matches!(
            event,
            kernel::Event::BlockHeaderNotFound { request_id: got } if got == request_id
        ));
    }

    #[test]
    fn block_header_fetch_error_maps_to_request_failed() {
        let request_id = RequestId::<GetBlockHeader>::from_raw_for_test(8);
        let request = AnyIssuedRequest::BlockHeader(IssuedRequest {
            request_id,
            request_payload: GetBlockHeader {
                block_hash: hash(1),
            },
        });

        let event = run_block_header(request, |_| {
            Err(ClientEvmError::MalformedResponse {
                context: "test".to_owned(),
                detail: "boom".to_owned(),
            })
        });

        assert!(matches!(
            event,
            kernel::Event::RequestFailed {
                request_id: AnyRequestId::BlockHeader(got),
            } if got == request_id
        ));
    }

    #[test]
    fn block_logs_success_maps_to_block_logs_received() {
        let request_id = RequestId::<GetBlockLogs>::from_raw_for_test(9);
        let request = AnyIssuedRequest::BlockLogs(IssuedRequest {
            request_id,
            request_payload: GetBlockLogs {
                block_hash: hash(2),
            },
        });

        let event = execute_request_with(
            request,
            unreachable_header,
            |_| Ok(Vec::new()),
            unreachable_pool_metadata,
            unreachable_token_metadata,
            unreachable_pool_data,
            unreachable_logs_range,
            unreachable_canonical,
        );

        assert!(matches!(
            event,
            kernel::Event::BlockLogsReceived { request_id: got, logs } if got == request_id && logs.is_empty()
        ));
    }
}
