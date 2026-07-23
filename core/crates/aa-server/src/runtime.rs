//! The runtime adapter: the thin I/O shell around the pure [`crate::core`].
//!
//! Mirrors `aa-cli`'s runtime for a single chain — it executes the kernel's RPC-request effects and
//! the finalized-header poll through the shared `client_evm` fetch functions, and drives the WS
//! new-heads stream and the tick. It holds no optimization overlay, view, or multi-chain container.

use std::{
    collections::{HashMap, HashSet},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
        mpsc::Sender,
    },
    thread,
    time::{Duration, Instant},
};

use aa_framework::{Application, Runtime, Transition};
use arc_swap::ArcSwap;
use client_evm::{
    AnyIssuedRequest, AnyRequestId, BlockHash, ChainEndpoints, ClientEvent, ClientEvmError,
    ClientHead, PoolDataResult, PoolLog, PoolMetadataResult, PoolRef, ProtocolPoolKey,
    RangeLogBlock, TokenAddress, TokenMetadataResult, fetch_block_header, fetch_block_logs,
    fetch_canonical_block_header_at, fetch_finalized_block_header, fetch_pool_data,
    fetch_pool_logs_in_range, fetch_pool_metadata, fetch_token_metadata, kernel,
    subscribe_new_heads,
};

use crate::{
    core::{
        AnchorHeader, CHAIN, ServerEffect, ServerInput, ServerState, server_init, server_transition,
    },
    serve::{ServerSnapshot, http_response, server_snapshot, strip_query},
};

/// The single-chain server application (pure core wired to the framework).
pub struct ServerApp;

/// The framework subscriptions the server seeds: the live head stream, the driver tick, and the
/// HTTP serving loop.
pub enum ServerSubscription {
    NewHeads,
    Tick(Duration),
    /// The blocking HTTP server; reads the published snapshot and answers `/health`. Sends no
    /// inputs back into the runtime (a read-only serving boundary).
    Serve,
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
            ServerSubscription::Serve,
        ]
    }
}

/// The I/O runtime for [`ServerApp`]: one HTTP endpoint pool (with failover) plus one WS URL.
pub struct ServerRuntime {
    agent: ureq::Agent,
    endpoints: ChainEndpoints,
    ws_url: String,
    /// Address the HTTP serving loop binds (e.g. `127.0.0.1:8080`).
    bind_addr: String,
    tick_interval: Duration,
    started: Instant,
    /// Wall-clock millis of the last emitted observation line, throttling the smoke log to ~1/s.
    last_log_millis: AtomicU64,
    /// The latest servable snapshot, republished from `observe_state` (inside the throttle) and
    /// read by the serve loop per request — the serve thread never touches kernel state. `ArcSwap`
    /// so publishes and per-request reads are lock-free (single writer, whole-value replace).
    snapshot: Arc<ArcSwap<ServerSnapshot>>,
}

impl ServerRuntime {
    pub fn new(endpoints: ChainEndpoints, ws_url: String, bind_addr: String) -> ServerRuntime {
        ServerRuntime {
            agent: ureq::Agent::new_with_defaults(),
            endpoints,
            ws_url,
            bind_addr,
            tick_interval: Duration::from_millis(1000),
            started: Instant::now(),
            last_log_millis: AtomicU64::new(0),
            snapshot: Arc::new(ArcSwap::from_pointee(ServerSnapshot::AwaitingAnchor)),
        }
    }

    /// Publishes the latest snapshot into the slot the serve loop reads. Lock-free: a single atomic
    /// pointer swap, so it never blocks (or is blocked by) a concurrent request read.
    fn publish(&self, snapshot: ServerSnapshot) {
        self.snapshot.store(Arc::new(snapshot));
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
            |at, candidates| {
                fetch_pool_metadata(&self.agent, &self.endpoints, CHAIN, at, candidates)
            },
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
            ServerSubscription::Serve => match tiny_http::Server::http(&self.bind_addr) {
                // A failed bind (bad/occupied address) is fatal only to serving: log loudly and let
                // this thread end. The kernel and its subscriptions keep running; no panic.
                Err(error) => {
                    eprintln!(
                        "aa-server serve bind_failed addr={} error={error}",
                        self.bind_addr
                    )
                }
                Ok(server) => {
                    eprintln!("aa-server serve listening addr={}", self.bind_addr);
                    serve_forever(server, self.snapshot.clone());
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
        if now.saturating_sub(self.last_log_millis.load(Ordering::Relaxed))
            < self.tick_interval.as_millis() as u64
        {
            return;
        }
        self.last_log_millis.store(now, Ordering::Relaxed);

        // Project once (the `Running` arm folds the frontier — the per-event hotspot, so this runs
        // at most ~1/s under the throttle), publish it for the serve loop, then log the same facts.
        let snapshot = server_snapshot(state);
        self.publish(snapshot.clone());

        match snapshot {
            ServerSnapshot::AwaitingAnchor => {
                eprintln!("aa-server chain={CHAIN:?} status=awaiting_anchor")
            }
            ServerSnapshot::Running {
                finalized: (finalized_hash, finalized_number),
                canonical,
                verified_pool_count,
                in_flight,
                ws_miss,
                behind,
                ..
            } => {
                eprintln!(
                    "aa-server chain={CHAIN:?} status=running finalized={finalized_hash}@{finalized_number} \
                     canonical={canonical} pools={verified_pool_count} inflight={in_flight} \
                     behind={behind:?} ws_miss={ws_miss}"
                );
            }
        }
    }
}

/// The blocking HTTP serving loop: one request at a time on a single thread. Each request reads the
/// method, path, and body, loads the current published snapshot (a lock-free `ArcSwap` read), and
/// answers via the pure [`http_response`]. A per-request `respond` I/O error (e.g. a client that
/// hung up) is logged and skipped so one broken connection cannot kill the loop.
fn serve_forever(server: tiny_http::Server, snapshot: Arc<ArcSwap<ServerSnapshot>>) {
    for mut request in server.incoming_requests() {
        let method = request.method().as_str().to_owned();
        let path = strip_query(request.url()).to_owned();
        // Read the body for `POST /slice`; a read failure leaves it empty, which the pure handler
        // rejects as a `400`. GET requests carry no body, so this is a no-op for `/health`.
        let mut body = String::new();
        if let Err(error) = request.as_reader().read_to_string(&mut body) {
            eprintln!("aa-server serve body_read_failed error={error}");
        }

        let current = snapshot.load_full();
        let response = http_response(&method, &path, &body, &current);

        let http =
            tiny_http::Response::from_string(response.body).with_status_code(response.status);
        if let Err(error) = request.respond(http) {
            eprintln!("aa-server serve respond_failed error={error}");
        }
    }
}

/// Maps a live WS client event to a server input: only new heads matter to warmup; the pool-log,
/// subscribed, and closed events carry no kernel input on this stream.
fn map_new_head(event: ClientEvent) -> Option<ServerInput> {
    match event {
        ClientEvent::NewHead { header, .. } => {
            Some(ServerInput::Kernel(kernel::Event::HeadObserved {
                hash: header.inner.hash,
                parent_hash: header.inner.inner.parent_hash,
                logs_bloom: header.inner.inner.logs_bloom,
                number: header.inner.inner.number,
            }))
        }
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

    fn test_runtime() -> ServerRuntime {
        let endpoints =
            ChainEndpoints::single(CHAIN, "primary", "http://localhost:8545".to_owned());
        ServerRuntime::new(
            endpoints,
            "ws://localhost:8546".to_owned(),
            "127.0.0.1:0".to_owned(),
        )
    }

    fn running_snapshot() -> ServerSnapshot {
        ServerSnapshot::Running {
            finalized: (hash(100), 100),
            canonical: hash(101),
            frontier: hash(101),
            verified_pool_count: 3,
            in_flight: 1,
            ws_miss: 2,
            behind: Some(4),
            pools: HashMap::new(),
        }
    }

    #[test]
    fn publish_updates_the_snapshot_slot() {
        let runtime = test_runtime();
        let snapshot = running_snapshot();

        runtime.publish(snapshot.clone());

        let stored = runtime.snapshot.load_full();
        assert_eq!(*stored, snapshot);
    }

    #[test]
    fn serve_answers_health_and_slice_over_a_loopback_socket() {
        let snapshot = running_snapshot();
        let slot = Arc::new(ArcSwap::from_pointee(snapshot.clone()));

        let server = tiny_http::Server::http("127.0.0.1:0").expect("bind ephemeral loopback port");
        let port = server.server_addr().to_ip().expect("ip listen addr").port();
        thread::spawn(move || serve_forever(server, slot));

        let agent = ureq::Agent::new_with_defaults();

        let mut response = agent
            .get(&format!("http://127.0.0.1:{port}/health"))
            .call()
            .expect("GET /health");
        assert_eq!(response.status().as_u16(), 200);
        let body = response.body_mut().read_to_string().expect("read body");
        assert_eq!(body, http_response("GET", "/health", "", &snapshot).body);

        // POST /slice reads the request body and answers via the same pure oracle.
        let slice_body = r#"{"pools":[]}"#;
        let mut slice = agent
            .post(&format!("http://127.0.0.1:{port}/slice"))
            .send(slice_body)
            .expect("POST /slice");
        assert_eq!(slice.status().as_u16(), 200);
        let slice_body_out = slice.body_mut().read_to_string().expect("read slice body");
        assert_eq!(
            slice_body_out,
            http_response("POST", "/slice", slice_body, &snapshot).body
        );

        match agent.get(&format!("http://127.0.0.1:{port}/slice")).call() {
            Err(ureq::Error::StatusCode(code)) => assert_eq!(code, 405),
            other => panic!("expected a 405 status error, got {other:?}"),
        }

        match agent.get(&format!("http://127.0.0.1:{port}/nope")).call() {
            Err(ureq::Error::StatusCode(code)) => assert_eq!(code, 404),
            other => panic!("expected a 404 status error, got {other:?}"),
        }
    }
}
