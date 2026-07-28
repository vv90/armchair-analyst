//! The composition root: the client engine as an [`aa_framework`] application. The pure `state`
//! reducer, the `optimizer` worker, and the `http` adapter are the leaf pieces; this module wires them
//! into the framework's event loop ([`aa_framework::Runtime::run`]) — the same Elm-style runtime the
//! kernel runs on. `run` owns the input channel, folds [`crate::transition`] over incoming events,
//! executes each emitted [`Effect`] on a thread-pool, feeds the results back as [`Event`]s, and drives
//! the periodic poll clock and the optimizer as subscriptions.
//!
//! Two roles map onto the framework's two traits:
//! - [`ClientEngineApp`] is the pure [`Application`]: `init`/`transition`/`subscriptions`, delegating
//!   the reducer verbatim.
//! - [`ClientEngineRuntime`] is the effectful [`Runtime`]. The fetch effects run inline in
//!   [`ClientEngineRuntime::execute_effect`] via the [`DataPlaneClient`] (synchronous HTTP). The
//!   optimizer cannot: its `OptimizationRunner` is GPU-capable and not `Send + Sync`, so it can never
//!   live in the shared runtime. Instead it runs on its own [`Subscription::Optimizer`] thread
//!   ([`crate::optimizer::run`]) that owns the runner internally; `execute_effect` just pushes the
//!   freshest reserves onto that thread's **coalescing** slot ([`crate::latest_slot`], latest-wins so a
//!   slow worker never accumulates a backlog) and the step results come back as [`Event`]s. The worker
//!   self-clocks its own `Continue`s. This is the same subscription-owned-worker shape aa-cli uses.
//!
//! [`ClientEngineRuntime::observe_state`] is the seam a later increment projects into a
//! `aa_client_api::ViewModel`.

use std::sync::Mutex;
use std::sync::mpsc::Sender;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use aa_framework::{Application, Runtime, Transition};
use client_evm::{ChainKey, ETHEREUM_USDC_TOKEN_ADDRESS, TokenAddress};
use optimization::{
    OptimizationBackendSelection, OptimizationSessionConfig, OptimizationStepConfig,
};

use crate::http::{DataPlaneClient, FetchRequest};
use crate::latest_slot::{LatestReceiver, LatestSender, latest_slot};
use crate::optimizer::{self, ReserveSnapshot};
use crate::state::{self, AppState, Effect, Event, Route, SessionConfig};

/// How often the poll clock fires. The framework's `subscriptions()` is a nullary static, so — like
/// aa-cli's hardcoded tick — the interval is a const rather than a per-session field for now
/// (configurable cadence is deferred with the rest of runtime config).
const POLL_INTERVAL: Duration = Duration::from_millis(500);

/// One optimizer step's per-call input amount, matching aa-cli's `default_optimization_step_config`
/// scale; `iterations` is the bounded grind budget of a single `run` between reserve refreshes.
/// Declared once: both the session config the reducer starts with and the budget the worker is built
/// with read this same const, so the two copies of a process-fixed value cannot disagree.
const STEP_CONFIG: OptimizationStepConfig = OptimizationStepConfig {
    input_amount: 1000.0,
    iterations: 10,
};

/// Which optimizer backend the worker initializes. Fixed for the process (never per-route), and read
/// from here by both holders for the same reason as [`STEP_CONFIG`].
const BACKEND: OptimizationBackendSelection = OptimizationBackendSelection::Cpu;

/// The long-lived, background work the runtime spawns. A narrow enum (not the effect type) so a
/// subscription can only be something the runtime actually knows how to spawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Subscription {
    /// Emit an [`Event::Tick`] every interval, forever (until the loop shuts down).
    Tick(Duration),
    /// Own the optimizer worker: pull the freshest reserves from its coalescing slot, self-clock the
    /// grind, and emit each step result as an event.
    Optimizer,
}

/// The chain the targeted `aa-server` is bound to. The wire is single-chain and carries no chain tag,
/// so the client supplies it and stamps it onto every projected pool/token — and onto the route.
const CLIENT_CHAIN: ChainKey = ChainKey::Ethereum;

/// Build the session config for a chosen route. Everything but the route is a const for now (single
/// Ethereum server, no bridges/whitelist, Cpu backend, fixed step) because the framework's
/// `init`/`subscriptions` are nullary statics; `chain`/`step`/`backend` configurability is deferred.
/// One source of truth so `default_session_config` and [`ClientConfig`] cannot drift.
fn session_config_for(route: Route) -> SessionConfig {
    SessionConfig {
        chain: CLIENT_CHAIN,
        optimization: OptimizationSessionConfig {
            source_asset: TokenAddress(route.source, CLIENT_CHAIN),
            output_asset: TokenAddress(route.output, CLIENT_CHAIN),
            bridges: std::collections::HashSet::new(),
            whitelist: None,
        },
        step: STEP_CONFIG,
        backend: BACKEND,
    }
}

/// The route `init()` seeds: closed-cycle USDC arbitrage, the permanent oracle. A placeholder, not a
/// policy — the framework's `init` is a nullary static, so `run_engine` replaces it with the caller's
/// route via an [`Event::SetRoute`] first input.
const DEFAULT_ROUTE: Route = Route {
    source: ETHEREUM_USDC_TOKEN_ADDRESS.0,
    output: ETHEREUM_USDC_TOKEN_ADDRESS.0,
};

/// The session config the engine starts with before its route is seeded.
fn default_session_config() -> SessionConfig {
    session_config_for(DEFAULT_ROUTE)
}

/// The caller-supplied configuration for one engine: which `aa-server` to target and the [`Route`] to
/// optimize. The route is user-facing (the app is a route explorer) and stays changeable at runtime
/// through [`Event::SetRoute`]; chain/backend/step stay const this increment. No default token is
/// buried in a `new`: a caller either asks for arbitrage on a named asset or for a named route.
#[derive(Clone, Debug)]
pub struct ClientConfig {
    /// The base URL of the `aa-server` data plane to fetch from.
    pub base_url: String,
    /// The asset pair to optimize (equal source/output ⇒ closed-cycle arbitrage).
    pub route: Route,
}

impl ClientConfig {
    /// Target `base_url` optimizing the closed arbitrage cycle on `asset`.
    pub fn arbitrage(base_url: String, asset: client_evm::Address) -> ClientConfig {
        ClientConfig {
            base_url,
            route: Route::arbitrage(asset),
        }
    }

    /// Target `base_url` optimizing `route` (an open best-execution path when its ends differ).
    pub fn for_route(base_url: String, route: Route) -> ClientConfig {
        ClientConfig { base_url, route }
    }
}

/// The pure application: `State = AppState`, `Input = Event`, `Effect = Effect`. All decision logic
/// lives in the `state` reducer; this impl only adapts it to the framework's shape.
pub struct ClientEngineApp;

impl Application for ClientEngineApp {
    type State = AppState;
    type Input = Event;
    type Effect = Effect;
    type Subscription = Subscription;

    /// Start awaiting the first slice, and kick off the cold-start fetches immediately so the engine
    /// doesn't idle a full poll interval before its first `/pools/meta` + `/health`.
    fn init() -> Transition<AppState, Effect> {
        // The cold start *is* a first tick: folding `Tick` over a fresh state issues the initial
        // `/pools/meta` + `/health` fetches (no catalog yet ⇒ meta, not slice) *and* records them in
        // the ledger, so there is a single issuance path and the first fetches occupy their slots.
        let (state, effects) =
            state::transition(AppState::started(default_session_config()), Event::Tick);
        Transition { state, effects }
    }

    fn transition(state: AppState, input: Event) -> Transition<AppState, Effect> {
        let (state, effects) = state::transition(state, input);
        Transition { state, effects }
    }

    fn subscriptions() -> Vec<Subscription> {
        vec![Subscription::Tick(POLL_INTERVAL), Subscription::Optimizer]
    }
}

/// The effectful runtime: owns the outbound HTTP client and the send end of the optimizer worker's
/// coalescing reserve slot. `Send + Sync` (the framework runs effects on a pool): the client and
/// [`LatestSender`] are `Send + Sync`, and the receiver is held behind a `Mutex` until claimed. The
/// optimizer worker itself — whose runner is not `Send + Sync` — is deliberately *not* here; it lives on
/// the [`Subscription::Optimizer`] thread.
///
/// It holds no route. Which pair is being optimized is `AppState`'s alone; it reaches the worker on
/// each [`Effect::PushReserves`], so there is no second copy here to fall out of step with the
/// reducer after a retarget. Only the backend and step budget — fixed for the process, never
/// per-route — are carried, because the worker needs them to build a runner.
pub struct ClientEngineRuntime {
    client: DataPlaneClient,
    /// Which optimizer backend the worker initializes (wgpu/cpu).
    backend: OptimizationBackendSelection,
    /// The per-step input amount and iteration budget the worker grinds with.
    step: OptimizationStepConfig,
    /// Coalescing sink for the freshest reserves (latest-wins, so a slow worker never backs up).
    reserve_sender: LatestSender<ReserveSnapshot>,
    /// The matching receiver, taken once by the optimizer subscription when it starts.
    reserve_inbox: Mutex<Option<LatestReceiver<ReserveSnapshot>>>,
}

impl ClientEngineRuntime {
    /// A runtime targeting one `aa-server` base URL, with the reserve slot created but its worker not
    /// yet spawned (the [`Subscription::Optimizer`] claims the receiver and owns the worker).
    pub fn new(base_url: String) -> ClientEngineRuntime {
        let (reserve_sender, reserve_inbox) = latest_slot();
        ClientEngineRuntime {
            client: DataPlaneClient::new(base_url),
            backend: BACKEND,
            step: STEP_CONFIG,
            reserve_sender,
            reserve_inbox: Mutex::new(Some(reserve_inbox)),
        }
    }
}

impl Runtime<ClientEngineApp> for ClientEngineRuntime {
    /// Execute one effect and return the events it produced. Fetch effects run inline against the data
    /// plane and yield exactly one event (the executor is total — faults come back as
    /// `Event::EffectFailed`, never a panic). `PushReserves` is asynchronous: the reserves are placed on
    /// the optimizer thread's coalescing slot and produce no event here; step results arrive later via
    /// the optimizer subscription.
    fn execute_effect(&self, effect: Effect) -> Vec<Event> {
        match effect {
            Effect::FetchMeta { id } => vec![self.client.handle(FetchRequest::Meta { id })],
            Effect::FetchHealth { id } => vec![self.client.handle(FetchRequest::Health { id })],
            Effect::FetchSlice { id, request } => {
                vec![self.client.handle(FetchRequest::Slice { id, request })]
            }
            Effect::PushReserves { reserves, session } => {
                // Coalescing send: overwrites any un-taken snapshot so the worker grinds the freshest
                // reserves. A closed slot (worker gone / shutting down) or poisoned lock is swallowed —
                // never a panic; the next productive slice simply pushes again.
                let _ = self
                    .reserve_sender
                    .send(ReserveSnapshot { reserves, session });
                Vec::new()
            }
        }
    }

    /// Run one background subscription. The poll clock sleeps then emits an [`Event::Tick`]; the
    /// optimizer subscription claims the reserve receiver and runs the worker, which owns its runner,
    /// self-clocks the grind, and forwards each step result as an event. Both return only when their
    /// channel closes (the engine is shutting down). Panic-free.
    fn spawn_subscription(&self, sender: &Sender<Event>, subscription: Subscription) {
        match subscription {
            Subscription::Tick(interval) => loop {
                thread::sleep(interval);
                if sender.send(Event::Tick).is_err() {
                    break;
                }
            },
            Subscription::Optimizer => {
                // Take the receiver exactly once; a second Optimizer subscription (there is only one)
                // would find it gone and simply return.
                let inbox = self
                    .reserve_inbox
                    .lock()
                    .ok()
                    .and_then(|mut slot| slot.take());
                if let Some(inbox) = inbox {
                    optimizer::run(inbox, self.backend, self.step, sender.clone());
                }
            }
        }
    }

    /// The observation seam. A later increment projects `state` into a `aa_client_api::ViewModel` and
    /// publishes it to the UI; for now the engine just runs.
    fn observe_state(&self, _state: &AppState) {}
}

/// Build the engine from a [`ClientConfig`] and start the framework loop on its own thread. Returns the
/// input sender (the runtime **command** seam) and the loop's join handle.
///
/// The caller's route reaches the engine as an [`Event::SetRoute`] first input: `init()` can only seed
/// a placeholder (the framework's `init` is a nullary static), so the route is *set*, exactly as a UI
/// will later set it. There is no second delivery path — the worker learns the route from the reserves
/// it is pushed — so this send races nothing: a slice that somehow beat it would be gated, and pushed,
/// against the placeholder route, and the retarget then re-initializes the worker on the next slice.
pub fn run(config: ClientConfig) -> (Sender<Event>, JoinHandle<()>) {
    let (sender, handle) = ClientEngineRuntime::new(config.base_url).run();
    let _ = sender.send(Event::SetRoute(config.route));
    (sender, handle)
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::sync::{Arc, Mutex as StdMutex};
    use std::thread::JoinHandle;
    use std::time::Instant;

    use aa_wire::{
        PoolCompleteness, PoolMetaEntry, PoolQuery, PoolSlice, PoolsMetaResponse, SliceResponse,
        TokenMetaEntry, WirePoolState,
    };
    use client_evm::{Address, ETHEREUM_WBTC_TOKEN_ADDRESS, PoolRef};
    use optimization::PoolReserves;

    use super::*;
    use crate::state::Phase;

    // Tick-0 price (`2^96`): keeps swap caps non-underflowing for any tick spacing.
    const SQRT_PRICE_TICK_0: u128 = 79_228_162_514_264_337_593_543_950_336;

    /// A loopback `aa-server` on an ephemeral port that answers each request from `handler`
    /// (`(method, path) -> (status, body)`); returns the port and the server thread's handle.
    fn loopback(
        handler: impl Fn(&str, &str) -> (u16, String) + Send + 'static,
    ) -> (u16, JoinHandle<()>) {
        let server = tiny_http::Server::http("127.0.0.1:0").expect("bind ephemeral loopback port");
        let port = server.server_addr().to_ip().expect("ip listen addr").port();
        let handle = std::thread::spawn(move || {
            for request in server.incoming_requests() {
                let method = request.method().as_str().to_owned();
                let path = request.url().to_owned();
                let (status, body) = handler(&method, &path);
                let response = tiny_http::Response::from_string(body).with_status_code(status);
                let _ = request.respond(response);
            }
        });
        (port, handle)
    }

    fn addr(byte: u8) -> Address {
        Address::from([byte; 20])
    }

    /// The USDC address the placeholder route arbitrages.
    fn usdc() -> Address {
        ETHEREUM_USDC_TOKEN_ADDRESS.0
    }

    fn v3_key(address: Address) -> PoolQuery {
        PoolQuery::UniswapV3 {
            address: format!("{address:#x}"),
        }
    }

    /// A catalog with one v3 pool over `(USDC, addr(2))` — a pair covering the placeholder route, so a
    /// complete slice for it is productive and initializes the optimizer.
    fn productive_meta() -> PoolsMetaResponse {
        PoolsMetaResponse {
            pools: vec![PoolMetaEntry {
                key: v3_key(addr(9)),
                token0: format!("{:#x}", usdc()),
                token1: format!("{:#x}", addr(2)),
                fee_pips: 3000,
                tick_spacing: 60,
            }],
            tokens: vec![
                TokenMetaEntry {
                    address: format!("{:#x}", usdc()),
                    decimals: 6,
                },
                TokenMetaEntry {
                    address: format!("{:#x}", addr(2)),
                    decimals: 18,
                },
            ],
        }
    }

    /// A complete slice for `productive_meta`'s pool at tick 0.
    fn productive_slice() -> SliceResponse {
        SliceResponse {
            block_hash: format!("{:#x}", client_evm::BlockHash::from([0xbb; 32])),
            confirmations: 2,
            pools: vec![PoolSlice {
                key: v3_key(addr(9)),
                state: PoolCompleteness::Complete {
                    state: WirePoolState {
                        sqrt_price_x96: format!("{SQRT_PRICE_TICK_0:#x}"),
                        tick: 0,
                        liquidity: format!("{:#x}", 1_000_000_000_000_000_000u128),
                    },
                },
            }],
        }
    }

    #[test]
    fn execute_effect_fetch_meta_hits_the_data_plane() {
        let body = serde_json::to_string(&productive_meta()).expect("serialize meta");
        let (port, _server) = loopback(move |method, path| {
            assert_eq!((method, path), ("GET", "/pools/meta"));
            (200, body.clone())
        });
        let runtime = ClientEngineRuntime::new(format!("http://127.0.0.1:{port}"));
        let id = crate::FetchId::from_raw_for_test(1);
        let events = runtime.execute_effect(Effect::FetchMeta { id });
        assert_eq!(
            events,
            vec![Event::MetaFetched {
                id,
                response: productive_meta()
            }]
        );
    }

    /// Minimal reserves of a given length, to tell one pushed snapshot from another.
    fn sample_reserves(count: usize) -> Vec<PoolReserves<PoolRef, TokenAddress>> {
        (0..count)
            .map(|i| PoolReserves {
                pool_id: PoolRef::uniswap_v3(addr(i as u8), ChainKey::Ethereum),
                token0: TokenAddress(addr(1), ChainKey::Ethereum),
                token1: TokenAddress(addr(2), ChainKey::Ethereum),
                value: optimization::VirtualReserveValues {
                    token_0: 1.0,
                    token_1: 1.0,
                    fee_multiplier: 0.997,
                    max_swap_0: 1.0,
                    max_swap_1: 1.0,
                },
            })
            .collect()
    }

    #[test]
    fn push_reserves_effect_coalesces_latest_into_the_slot() {
        let runtime = ClientEngineRuntime::new("http://127.0.0.1:1".to_owned());
        // The optimizer runs on its subscription thread, so `execute_effect` returns no event...
        assert!(
            runtime
                .execute_effect(Effect::PushReserves {
                    reserves: sample_reserves(1),
                    session: default_session_config().optimization,
                })
                .is_empty()
        );
        assert!(
            runtime
                .execute_effect(Effect::PushReserves {
                    reserves: sample_reserves(2),
                    session: default_session_config().optimization,
                })
                .is_empty()
        );

        // ...and the coalescing slot holds only the newest push (latest-wins, no backlog).
        let inbox = runtime
            .reserve_inbox
            .lock()
            .expect("inbox lock")
            .take()
            .expect("inbox present");
        let taken = inbox
            .try_take()
            .expect("slot readable")
            .expect("a snapshot is present");
        assert_eq!(taken.reserves.len(), 2);
        assert!(inbox.try_take().expect("slot readable").is_none());
    }

    #[test]
    fn tick_subscription_emits_ticks_then_stops_when_the_receiver_is_gone() {
        let runtime = ClientEngineRuntime::new("http://127.0.0.1:1".to_owned());
        let (event_tx, event_rx) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            runtime.spawn_subscription(&event_tx, Subscription::Tick(Duration::from_millis(1)));
        });
        assert_eq!(event_rx.recv().expect("first tick"), Event::Tick);
        // Dropping the receiver makes the next `send` fail, so the subscription loop returns.
        drop(event_rx);
        worker.join().expect("subscription thread joins");
    }

    /// A recording runtime: delegates every effect/subscription to a real [`ClientEngineRuntime`] and
    /// only overrides `observe_state` to capture each post-transition phase. Lets the end-to-end test
    /// watch the engine reach `Optimizing` without changing the production runtime.
    struct RecordingRuntime {
        inner: ClientEngineRuntime,
        observed: Arc<StdMutex<Vec<Phase>>>,
    }

    impl Runtime<ClientEngineApp> for RecordingRuntime {
        fn execute_effect(&self, effect: Effect) -> Vec<Event> {
            self.inner.execute_effect(effect)
        }

        fn spawn_subscription(&self, sender: &Sender<Event>, subscription: Subscription) {
            self.inner.spawn_subscription(sender, subscription);
        }

        fn observe_state(&self, state: &AppState) {
            if let Ok(mut observed) = self.observed.lock() {
                observed.push(state.phase.clone());
            }
        }
    }

    /// Block until `predicate` matches one of the observed phases, panicking with the whole recorded
    /// history after 10s. The engine self-clocks, so a phase is only reachable by waiting for it.
    fn wait_for_phase(
        observed: &Arc<StdMutex<Vec<Phase>>>,
        what: &str,
        predicate: impl Fn(&Phase) -> bool,
    ) {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let reached = observed
                .lock()
                .map(|phases| phases.iter().any(&predicate))
                .unwrap_or(false);
            if reached {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "engine did not reach {what} in time; observed: {:?}",
                observed
                    .lock()
                    .map(|phases| phases.clone())
                    .unwrap_or_default(),
            );
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    /// Forget every phase observed so far, so a later `wait_for_phase` can only match something the
    /// engine did *after* this point (the phase being awaited may already have occurred once).
    fn forget_observed(observed: &Arc<StdMutex<Vec<Phase>>>) {
        if let Ok(mut phases) = observed.lock() {
            phases.clear();
        }
    }

    #[test]
    fn engine_runs_from_cold_start_to_optimizing_over_a_loopback_data_plane() {
        let meta = serde_json::to_string(&productive_meta()).expect("serialize meta");
        let slice = serde_json::to_string(&productive_slice()).expect("serialize slice");
        let health = serde_json::to_string(&aa_wire::HealthResponse::AwaitingAnchor)
            .expect("serialize health");
        let (port, _server) = loopback(move |_method, path| match path {
            "/pools/meta" => (200, meta.clone()),
            "/slice" => (200, slice.clone()),
            "/health" => (200, health.clone()),
            _ => (404, String::new()),
        });

        let observed = Arc::new(StdMutex::new(Vec::new()));
        let runtime = RecordingRuntime {
            inner: ClientEngineRuntime::new(format!("http://127.0.0.1:{port}")),
            observed: observed.clone(),
        };
        // `run` loops forever (it self-clocks and grinds), so we never join it — we watch the recorded
        // phases until the productive slice has driven the engine into `Optimizing`.
        let (_sender, _handle) = <RecordingRuntime as Runtime<ClientEngineApp>>::run(runtime);

        wait_for_phase(&observed, "Optimizing", |phase| {
            matches!(phase, Phase::Optimizing { .. })
        });
    }

    /// A catalog with one v3 pool over `(WBTC, addr(2))` — reaches WBTC but never USDC, so a complete
    /// slice for it is productive only under a route whose output is WBTC, not the USDC arbitrage
    /// default. Reuses `productive_slice`'s pool key (`addr(9)`), so that slice fits this catalog.
    fn wbtc_route_meta() -> PoolsMetaResponse {
        PoolsMetaResponse {
            pools: vec![PoolMetaEntry {
                key: v3_key(addr(9)),
                token0: format!("{:#x}", ETHEREUM_WBTC_TOKEN_ADDRESS.0),
                token1: format!("{:#x}", addr(2)),
                fee_pips: 3000,
                tick_spacing: 60,
            }],
            tokens: vec![
                TokenMetaEntry {
                    address: format!("{:#x}", ETHEREUM_WBTC_TOKEN_ADDRESS.0),
                    decimals: 8,
                },
                TokenMetaEntry {
                    address: format!("{:#x}", addr(2)),
                    decimals: 18,
                },
            ],
        }
    }

    #[test]
    fn a_configured_open_route_reaches_optimizing_end_to_end() {
        // The whole seam in one run: `ClientConfig::for_route` → `run_engine` → the engine optimizing
        // an open `addr(2) → WBTC` route against a real (loopback) data plane. The catalog reaches
        // WBTC but never USDC, so the arbitrage default would sit in `AwaitingFirstSlice` forever —
        // reaching `Optimizing` proves the caller's route actually took effect. `RecordingRuntime`
        // wraps the real runtime only to observe phases; the route travels exactly as in `run_engine`.
        let meta = serde_json::to_string(&wbtc_route_meta()).expect("serialize meta");
        let slice = serde_json::to_string(&productive_slice()).expect("serialize slice");
        let health = serde_json::to_string(&aa_wire::HealthResponse::AwaitingAnchor)
            .expect("serialize health");
        let (port, _server) = loopback(move |_method, path| match path {
            "/pools/meta" => (200, meta.clone()),
            "/slice" => (200, slice.clone()),
            "/health" => (200, health.clone()),
            _ => (404, String::new()),
        });

        let config = ClientConfig::for_route(
            format!("http://127.0.0.1:{port}"),
            Route {
                source: addr(2),
                output: ETHEREUM_WBTC_TOKEN_ADDRESS.0,
            },
        );

        let observed = Arc::new(StdMutex::new(Vec::new()));
        let runtime = RecordingRuntime {
            inner: ClientEngineRuntime::new(config.base_url.clone()),
            observed: observed.clone(),
        };
        let (sender, _handle) = <RecordingRuntime as Runtime<ClientEngineApp>>::run(runtime);
        // Exactly what `run_engine` does with the config: set the caller's route as the first input.
        let _ = sender.send(Event::SetRoute(config.route));

        wait_for_phase(&observed, "Optimizing on the open route", |phase| {
            matches!(phase, Phase::Optimizing { .. })
        });
    }

    #[test]
    fn a_mid_session_retarget_restarts_optimization_on_the_new_route() {
        // What a UI will do: change the route while the engine is already optimizing. End to end, on
        // one live engine — the reducer drops the old route's results and re-gates the next slice, and
        // the worker, which learns the route from the reserves it is handed, re-initializes on it. The
        // engine must therefore leave `Optimizing`, then come back on its own within a poll interval.
        let meta = serde_json::to_string(&wbtc_route_meta()).expect("serialize meta");
        let slice = serde_json::to_string(&productive_slice()).expect("serialize slice");
        let health = serde_json::to_string(&aa_wire::HealthResponse::AwaitingAnchor)
            .expect("serialize health");
        let (port, _server) = loopback(move |_method, path| match path {
            "/pools/meta" => (200, meta.clone()),
            "/slice" => (200, slice.clone()),
            "/health" => (200, health.clone()),
            _ => (404, String::new()),
        });

        let observed = Arc::new(StdMutex::new(Vec::new()));
        let runtime = RecordingRuntime {
            inner: ClientEngineRuntime::new(format!("http://127.0.0.1:{port}")),
            observed: observed.clone(),
        };
        let (sender, _handle) = <RecordingRuntime as Runtime<ClientEngineApp>>::run(runtime);

        // Start on `addr(2) → WBTC`, and wait until it is really optimizing that route.
        let _ = sender.send(Event::SetRoute(Route {
            source: addr(2),
            output: ETHEREUM_WBTC_TOKEN_ADDRESS.0,
        }));
        wait_for_phase(&observed, "Optimizing on the first route", |phase| {
            matches!(phase, Phase::Optimizing { .. })
        });

        // Retarget to the reverse route, which the same catalog also serves.
        forget_observed(&observed);
        let _ = sender.send(Event::SetRoute(Route {
            source: ETHEREUM_WBTC_TOKEN_ADDRESS.0,
            output: addr(2),
        }));

        // The old route's results are gone...
        wait_for_phase(
            &observed,
            "AwaitingFirstSlice after the retarget",
            |phase| matches!(phase, Phase::AwaitingFirstSlice { .. }),
        );
        // ...and the engine recovers onto the new route with no further input.
        forget_observed(&observed);
        wait_for_phase(&observed, "Optimizing on the new route", |phase| {
            matches!(phase, Phase::Optimizing { .. })
        });
    }
}
