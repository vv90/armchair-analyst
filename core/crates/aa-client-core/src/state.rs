//! The headless engine's pure core: an `AppState` and a `transition` reducer with the same
//! `event → state → (state, effects)` shape as the kernel. It owns *all* of the client's decision
//! logic — when to poll the data plane, when to start optimizing — while holding **none** of the heavy
//! machinery. In particular the optimizer's mutable tensor state never lives here:
//! `optimization::OptimizationRunner` is a move-based state machine that owns the `Model`; the reducer
//! only hands the worker fresh reserves and the route they were gated against (via
//! [`Effect::PushReserves`]) and folds its results back in as
//! [`Event::OptimizerStepped`]. The grind cadence (`Continue` vs `NewReserves`) is the worker's, not the
//! reducer's. Keeping the derived heavy thing out of the reducible state is the same discipline the
//! kernel uses for reserves and pool folds.
//!
//! Everything in this module is pure and synchronous: the reducer performs no I/O and spawns nothing.
//! A driver (a later increment) executes the returned [`Effect`]s on real threads/HTTP and feeds the
//! outcomes back as [`Event`]s. That split makes the entire engine testable by feeding event
//! sequences and asserting `(state, effects)`, with no transport in the loop.

use std::str::FromStr;

use aa_wire::{HealthResponse, PoolsMetaResponse, SliceRequest, SliceResponse};
use client_evm::{Address, BlockHash, ChainKey, PoolRef, TokenAddress};
use optimization::{
    ExecutionPlan, OptimizationBackendSelection, OptimizationSessionConfig, OptimizationStepConfig,
    OptimizationStepResult, PoolReserves, reserves_reach_route,
};

use crate::pending::{FetchId, PendingFetches};
use crate::{Catalog, WireAdapterError};

/// Client-owned strategy and cadence for one server session. The optimizer's own config types are
/// reused verbatim (`OptimizationSessionConfig`/`OptimizationStepConfig`) so there is no mirror to
/// drift; the client just owns an instance of each. `chain` is the adapter's deferred open item — the
/// wire is single-chain and carries no chain tag, so the chain the server is bound to is supplied
/// here and stamped onto every parsed pool/token.
#[derive(Clone, Debug)]
pub struct SessionConfig {
    /// The chain the target server is bound to; stamped onto every projected `PoolRef`/`TokenAddress`.
    pub chain: ChainKey,
    /// The routing strategy the client optimizes for: the source/output route, bridge pairs, token
    /// whitelist. Retargeted in place by [`Event::SetRoute`], and the single owner of the route. The
    /// server never serves strategy, so this is purely client-side.
    pub optimization: OptimizationSessionConfig<TokenAddress>,
    /// Per-step input amount and the iteration budget of a single `run` — the bounded unit of work
    /// that keeps the (future) optimizer worker responsive between reserve refreshes.
    pub step: OptimizationStepConfig,
    /// Which optimizer backend to initialize (wgpu/cpu).
    pub backend: OptimizationBackendSelection,
}

/// The asset pair one optimization is for: spend `source`, maximize the received `output`. Equal ⇒ a
/// closed arbitrage cycle; distinct ⇒ an open best-execution path. This is what a user picks, so it
/// is also what the [`Event::SetRoute`] command and the client config carry.
///
/// Plain [`Address`]es rather than chain-stamped `TokenAddress`es: the session's `chain` is the one
/// place a chain is decided (the wire is single-chain and carries no chain tag), and it is stamped on
/// when the route meets the reserves. A route tagged with some *other* chain would match no reserve
/// and silently starve the optimizer — so it is made unrepresentable instead of validated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Route {
    /// The asset the committed input is denominated in.
    pub source: Address,
    /// The asset whose received amount the optimizer maximizes.
    pub output: Address,
}

impl Route {
    /// The closed arbitrage cycle on `asset`: spend it and maximize it (`source == output`).
    pub fn arbitrage(asset: Address) -> Route {
        Route {
            source: asset,
            output: asset,
        }
    }
}

/// The freshness envelope of the reserves currently driving the optimizer: which block the state was
/// valid at and how deep below the observed tip it sat. Consuming these `/slice` facts is what the
/// wire→reserves adapter deliberately left for the engine; candidate-lifetime re-evaluation against
/// later slices is a follow-up.
///
/// Deliberately carries no chain: the session is single-chain and [`SessionConfig::chain`] is that
/// fact's sole owner, so a view renders the chain from config rather than from the slice. Copying it
/// in here would be a second writable copy with no invariant tying the two together.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SliceProvenance {
    /// The frontier block hash the reserves were projected from.
    pub block_hash: BlockHash,
    /// Canonical blocks the observed tip was ahead of that frontier (reorg-depth / staleness).
    pub confirmations: u64,
}

/// The whole application state: immutable session config, the two facts that are true independently
/// of the lifecycle (the latest `/health` snapshot and the most recent fault), the [`Session`]
/// lifecycle itself, and the in-flight fetch ledger that gates and correlates data-plane requests.
///
/// `health` and `last_error` sit here rather than inside [`Session`] because no combination of them
/// with a lifecycle state is invalid: `/health` polls on its own clock, and a transport fault can be
/// recorded at any point without changing where the engine is. Anything whose *presence* is tied to
/// the lifecycle lives inside [`Session`]/[`Work`] instead, so it cannot be observed out of place.
#[derive(Clone, Debug)]
pub struct AppState {
    /// The session's fixed strategy/cadence for its whole lifetime.
    pub config: SessionConfig,
    /// The latest server freshness snapshot, if `/health` has been polled. Orthogonal to the
    /// lifecycle: the poll runs from the first tick and never gates anything.
    pub health: Option<HealthResponse>,
    /// The most recent recorded fault (fetch/adapter/optimizer), cleared when a good slice is
    /// applied. Orthogonal to the lifecycle *and* to why the engine is waiting: a failed `/health`
    /// poll says nothing about whether the route is servable, so the two are recorded separately
    /// rather than overwriting each other.
    pub last_error: Option<EffectError>,
    /// Where in the bootstrap→optimize lifecycle the engine currently is.
    pub session: Session,
    /// Which `/pools/meta`, `/slice`, and `/health` requests are outstanding — one per kind — so the
    /// poll loop never double-issues, retries a lost fetch on TTL, and rejects superseded responses.
    pub pending: PendingFetches,
}

impl AppState {
    /// A freshly armed engine: it has a config but has seen nothing from the server yet and has no
    /// request in flight. The first `Tick` issues the initial `/pools/meta` + `/health` fetches.
    pub fn started(config: SessionConfig) -> AppState {
        AppState {
            config,
            health: None,
            last_error: None,
            session: Session::NoCatalog,
            pending: PendingFetches::new(),
        }
    }

    /// The catalog currently held, if any. The only thing a `/slice` can be requested for or
    /// projected against, so every consumer — the poll loop, the projection, a future view — reads
    /// it through here rather than re-matching the lifecycle.
    pub fn catalog(&self) -> Option<&Catalog> {
        match &self.session {
            Session::NoCatalog => None,
            Session::Ready { catalog, .. } => Some(catalog),
        }
    }
}

/// The engine lifecycle's outer step: whether a catalog has been fetched. Everything downstream —
/// requesting a slice, projecting one, deciding a route is unservable, optimizing — requires one, so
/// nesting [`Work`] under `Ready` makes "any of that without a catalog" unrepresentable rather than
/// merely untested.
// Same shape as the server's `ServerState`, and allowed for the same reason: `Ready` is the large,
// permanent state and `NoCatalog` is the brief startup one. Boxing to shrink a variant the engine
// occupies for one round-trip would put an allocation behind every reserve refresh for the whole
// process lifetime.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq)]
pub enum Session {
    /// `/pools/meta` has not come back yet; there is nothing to project a slice against, so the poll
    /// loop asks for the catalog and nothing else.
    NoCatalog,
    /// The static catalog is held; `work` says how far past it the engine has got.
    Ready {
        /// The catalog every `/slice` request and projection is built from — validated at its own
        /// boundary, so holding one means holding a *usable* one. Route-independent, so it survives
        /// a retarget.
        catalog: Catalog,
        /// Whether a productive slice has been applied yet.
        work: Work,
    },
}

/// The lifecycle's inner step, under a held catalog (mirrors the server's `AwaitingAnchor →
/// Running`): the optimizer requires a non-empty reserve set covering the configured route, so it is
/// only driven once such a slice arrives. `latest`/`last_step`/`plan` exist only in `Optimizing`, so
/// "optimizer running but no reserves yet" stays unrepresentable.
#[derive(Clone, Debug, PartialEq)]
pub enum Work {
    /// No productive reserve snapshot has been applied yet; the optimizer has not been initialized.
    Awaiting(AwaitReason),
    /// The optimizer has been initialized and is being fed fresh reserves + grinding iterations.
    Optimizing {
        /// Freshness envelope of the reserves currently driving the optimizer.
        latest: SliceProvenance,
        /// The most recent optimizer step result, once one has come back.
        last_step: Option<OptimizationStepResult>,
        /// The executable plan recovered from the most recent step, if any.
        plan: Option<ExecutionPlan<PoolRef, TokenAddress>>,
    },
}

/// Why the engine is still awaiting its first productive slice. A view-facing status, not a control
/// signal — and *only* a wait-reason: a recorded fault is a separate, orthogonal fact
/// ([`AppState::last_error`]) rather than a third variant here, so an unrelated `/health` failure
/// cannot erase a `NoRoute` verdict.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AwaitReason {
    /// Waiting for the first slice to arrive (or for a retarget's first slice).
    Bootstrapping,
    /// A slice arrived but its reserves do not cover the configured route — either nothing to spend
    /// the source asset into or nothing yielding the output asset — so the optimizer is intentionally
    /// not initialized (it would abort on such a snapshot).
    NoRoute,
}

/// A fault surfaced into view state. Never a panic: a malformed server response or a failed fetch
/// degrades to a recorded, typed error, and the engine stays in a valid state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EffectError {
    /// A driver-side fetch effect failed; `what` names the request and `message` is its diagnostic.
    Fetch {
        /// Which data-plane request failed.
        what: FetchKind,
        /// The driver's error text.
        message: String,
    },
    /// Projecting a `/slice` payload into optimizer reserves failed (a data fault in the payload).
    Adapter(WireAdapterError),
    /// A `/slice` freshness field (currently `block_hash`) did not parse.
    Provenance {
        /// The wire field that failed to parse.
        field: &'static str,
        /// Its raw value.
        value: String,
    },
    /// The optimizer worker failed to initialize or advance the runner. `stage` names which; `message`
    /// is the optimizer error's text (the typed `optimization` errors don't cross the module boundary).
    Optimize {
        /// Whether the failure was at `init` or at `run`.
        stage: OptimizeStage,
        /// The optimizer error's diagnostic text.
        message: String,
    },
}

/// Which optimizer call a [`EffectError::Optimize`] refers to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptimizeStage {
    /// `OptimizationRunner::init` (first productive slice).
    Init,
    /// `OptimizationRunner::run` (a `NewReserves`/`Continue` step, or a `Run` before any `Init`).
    Run,
}

/// Which data-plane request a [`EffectError::Fetch`] refers to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FetchKind {
    /// `GET /pools/meta`.
    Meta,
    /// `GET /health`.
    Health,
    /// `POST /slice`.
    Slice,
}

/// Everything a driver can feed the reducer: the outcomes of the effects it was asked to perform. Each
/// fetch outcome carries the [`FetchId`] of the request it answers so the reducer can reject a
/// superseded (re-issued-past) response instead of applying it.
#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    /// The `/pools/meta` catalog was fetched for request `id`.
    MetaFetched {
        /// The id of the fetch this answers.
        id: FetchId,
        /// The fetched catalog.
        response: PoolsMetaResponse,
    },
    /// A `/slice` response was fetched for request `id`.
    SliceFetched {
        /// The id of the fetch this answers.
        id: FetchId,
        /// The fetched slice.
        response: SliceResponse,
    },
    /// A `/health` snapshot was fetched for request `id`.
    HealthFetched {
        /// The id of the fetch this answers.
        id: FetchId,
        /// The fetched health snapshot.
        response: HealthResponse,
    },
    /// A data-plane fetch for request `id` (of `kind`) failed with diagnostic `message`. Distinct from
    /// [`Event::EffectFailed`] (optimizer faults) so the reducer can free the right ledger slot.
    ///
    /// The driver reports the raw failure, not a built [`EffectError`]: `kind` names both the slot to
    /// free and the fault to record, so the two cannot name different requests, and a fetch outcome
    /// cannot masquerade as an adapter or optimizer fault.
    FetchFailed {
        /// The id of the fetch that failed.
        id: FetchId,
        /// Which data-plane request it was: the slot the reducer clears, and the `what` it records.
        kind: FetchKind,
        /// The driver's diagnostic text.
        message: String,
    },
    /// The optimizer completed a step (init or run), returning its scalar result and recovered plan.
    OptimizerStepped {
        /// The scalar step summary.
        result: OptimizationStepResult,
        /// The executable plan recovered from the trained weights, if extraction succeeded.
        plan: Option<ExecutionPlan<PoolRef, TokenAddress>>,
    },
    /// A non-fetch effect (the optimizer) failed.
    EffectFailed(EffectError),
    /// The poll clock fired.
    Tick,
    /// Retarget the engine at a different [`Route`]. The runtime **command** seam — the input
    /// `Sender` `run_engine` returns carries these, and it is the only event that is a user *intent*
    /// rather than the outcome of an effect. `run_engine` sends one at startup to seed the caller's
    /// route (the framework's `init` is a nullary static and can only seed a placeholder); a UI
    /// dispatches one per user route change, mid-session, with no further plumbing.
    SetRoute(Route),
}

/// A side effect the reducer asks the driver to perform. Executing an effect eventually produces one
/// or more [`Event`]s fed back into `transition`. Each fetch carries the [`FetchId`] the ledger issued,
/// which the driver echoes back on the outcome event.
#[derive(Clone, Debug)]
pub enum Effect {
    /// Fetch `GET /pools/meta`.
    FetchMeta {
        /// The ledger-issued id to echo on the outcome.
        id: FetchId,
    },
    /// Fetch `GET /health`.
    FetchHealth {
        /// The ledger-issued id to echo on the outcome.
        id: FetchId,
    },
    /// Fetch `POST /slice` for the given pool set.
    FetchSlice {
        /// The ledger-issued id to echo on the outcome.
        id: FetchId,
        /// The pools to request state for.
        request: SliceRequest,
    },
    /// Push the latest projected reserves — and the route they were gated against — to the optimizer
    /// worker's coalescing slot. The worker owns the grind cadence: it inits on the first snapshot,
    /// applies later ones as `NewReserves`, and self-continues when no fresh snapshot is waiting.
    ///
    /// The route rides along rather than being configured into the worker once, so [`AppState`] stays
    /// its **single owner**: a worker holding its own copy would keep optimizing the old pair after an
    /// [`Event::SetRoute`], silently disagreeing with the gate that admitted the reserves. Carrying it
    /// makes every snapshot self-describing, so the worker re-initializes exactly when the route it is
    /// handed stops matching the one it initialized on.
    PushReserves {
        /// The freshest productive reserves to grind (latest-wins; older un-taken pushes are dropped).
        reserves: Vec<PoolReserves<PoolRef, TokenAddress>>,
        /// The route (and routing constraints) these reserves were admitted for.
        session: OptimizationSessionConfig<TokenAddress>,
    },
}

/// The reducer: fold one [`Event`] into the state, returning the next state and the effects to run.
/// Pure and total — every event maps to a valid next state, and any data fault becomes a recorded
/// [`EffectError`] rather than a panic.
pub fn transition(mut state: AppState, event: Event) -> (AppState, Vec<Effect>) {
    let effects = reduce(&mut state, event);
    (state, effects)
}

/// The single dispatch point over [`Event`], so no variant is handled in two places. Takes `&mut`
/// rather than threading each field: with the lifecycle-independent facts (`health`, `last_error`)
/// out of [`Session`], most arms are a single field write and rebuilding the whole state around them
/// would be pure noise.
fn reduce(state: &mut AppState, event: Event) -> Vec<Effect> {
    match event {
        Event::Tick => on_tick(state),
        Event::MetaFetched { id, response } => {
            if state.pending.accept(FetchKind::Meta, id) {
                on_meta(state, response)
            } else {
                vec![]
            }
        }
        Event::SliceFetched { id, response } => {
            if state.pending.accept(FetchKind::Slice, id) {
                on_slice(state, response)
            } else {
                vec![]
            }
        }
        Event::HealthFetched { id, response } => {
            if state.pending.accept(FetchKind::Health, id) {
                state.health = Some(response);
            }
            vec![]
        }
        Event::FetchFailed { id, kind, message } => {
            if state.pending.accept(kind, id) {
                state.last_error = Some(EffectError::Fetch {
                    what: kind,
                    message,
                });
            }
            vec![]
        }
        Event::OptimizerStepped { result, plan } => {
            on_step(state, result, plan);
            vec![]
        }
        Event::EffectFailed(error) => {
            state.last_error = Some(error);
            vec![]
        }
        Event::SetRoute(route) => {
            on_set_route(state, route);
            vec![]
        }
    }
}

/// Retarget the optimized route. The route is retargeted in place and everything the optimizer
/// produced for the *previous* route is discarded: `last_step` and `plan` describe amounts and a swap
/// path for the old pair, so keeping them would leave them on display — attributed to the new route —
/// until fresh reserves land. The catalog survives (it is route-independent, and the slice already in
/// flight still answers it), so re-optimizing costs one poll interval and no refetch. Emits no
/// effects: the next slice pushes reserves through the retargeted gate on its own.
///
/// Setting the route already in force is a no-op, so `run_engine`'s startup seeding and a UI
/// re-sending the current pair never interrupt a running optimization.
fn on_set_route(state: &mut AppState, route: Route) {
    // The session's chain is stamped on here — the one place the client decides a chain — exactly as
    // `slice_to_reserves` stamps it onto every projected pool and token.
    let source = TokenAddress(route.source, state.config.chain);
    let output = TokenAddress(route.output, state.config.chain);
    let optimization = &mut state.config.optimization;
    if optimization.source_asset == source && optimization.output_asset == output {
        return;
    }
    optimization.source_asset = source;
    optimization.output_asset = output;

    // Only the *work* is route-dependent. The catalog is not, and neither is a recorded fault: it is
    // about the transport, so it outlives a retarget — uniformly, which it did not when it was
    // encoded one way while awaiting and another while optimizing.
    if let Session::Ready { work, .. } = &mut state.session {
        *work = Work::Awaiting(AwaitReason::Bootstrapping);
    }
}

/// The poll clock fired: advance the ledger clock, then ensure the desired fetches are in flight —
/// `/health` always, plus a fresh slice for the known catalog (else the catalog itself). `ensure`
/// gates on the ledger, so a fetch is issued only when that kind is free or its request has expired;
/// the periodic `Tick` subscription re-arms the clock, so the reducer no longer schedules it.
fn on_tick(state: &mut AppState) -> Vec<Effect> {
    state.pending.advance();
    let mut effects = Vec::new();
    if let Some(id) = state.pending.ensure(FetchKind::Health) {
        effects.push(Effect::FetchHealth { id });
    }
    if state.catalog().is_none() {
        if let Some(id) = state.pending.ensure(FetchKind::Meta) {
            effects.push(Effect::FetchMeta { id });
        }
        return effects;
    }
    // The ledger gate comes *before* the request is materialized: the catalog's prebuilt request is
    // still a clone per pool, and on most ticks the previous slice is still in flight, so building it
    // first would allocate the whole pool list only to drop it.
    if let Some(id) = state.pending.ensure(FetchKind::Slice)
        && let Some(catalog) = state.catalog()
    {
        effects.push(Effect::FetchSlice {
            id,
            request: catalog.slice_request().clone(),
        });
    }
    effects
}

/// The catalog arrived: validate it into a [`Catalog`], store it, and immediately request the first
/// slice for it (don't wait a full poll interval). The slice slot is free at this point, so `ensure`
/// issues. Validation is where every catalog data fault is settled — once, here, instead of being
/// rediscovered against each slice that arrives later.
fn on_meta(state: &mut AppState, response: PoolsMetaResponse) -> Vec<Effect> {
    let catalog = Catalog::parse(&response, state.config.chain);
    let request = catalog.slice_request().clone();
    match &mut state.session {
        // A refetched catalog replaces the held one; how far the engine has got is unaffected.
        Session::Ready { catalog: held, .. } => *held = catalog,
        session => {
            *session = Session::Ready {
                catalog,
                work: Work::Awaiting(AwaitReason::Bootstrapping),
            };
        }
    }
    match state.pending.ensure(FetchKind::Slice) {
        Some(id) => vec![Effect::FetchSlice { id, request }],
        None => vec![],
    }
}

/// A slice arrived: project it against the catalog and, if it yields a productive reserve set,
/// (re)drive the optimizer. Faults and unproductive snapshots are recorded without leaving a valid
/// state and without touching the optimizer.
fn on_slice(state: &mut AppState, slice: SliceResponse) -> Vec<Effect> {
    let Session::Ready { catalog, work } = &mut state.session else {
        // A slice is only ever requested once a catalog is held, and the catalog is never dropped, so
        // this is unreachable in practice — there is simply nothing to project the payload against.
        return vec![];
    };

    let reserves = match catalog.project(&slice) {
        Ok(reserves) => reserves,
        Err(error) => {
            state.last_error = Some(EffectError::Adapter(error));
            return vec![];
        }
    };
    let provenance = match parse_provenance(&slice) {
        Ok(provenance) => provenance,
        Err(error) => {
            state.last_error = Some(error);
            return vec![];
        }
    };
    if !is_productive(&reserves, &state.config) {
        match work {
            // Nothing loaded yet: record *why* the engine is still waiting.
            Work::Awaiting(reason) => *reason = AwaitReason::NoRoute,
            // Already grinding: a momentarily unreachable snapshot would abort the runner, so skip
            // it and keep optimizing the reserves already loaded.
            Work::Optimizing { .. } => {}
        }
        return vec![];
    }

    // The worker inits on the first productive snapshot and applies later ones as `NewReserves`;
    // either way what changes here is the provenance. Results carry over so a refresh does not blank
    // the view between steps — they describe the same route, which only `SetRoute` invalidates.
    let (last_step, plan) = match work {
        Work::Awaiting(_) => (None, None),
        Work::Optimizing {
            last_step, plan, ..
        } => (last_step.take(), plan.take()),
    };
    *work = Work::Optimizing {
        latest: provenance,
        last_step,
        plan,
    };
    state.last_error = None;
    vec![Effect::PushReserves {
        reserves,
        session: state.config.optimization.clone(),
    }]
}

/// A step came back: record its result and plan. No effect follows — the worker self-clocks, pulling
/// fresh reserves or self-continuing on its own thread, so the reducer never re-issues a `Continue`.
fn on_step(
    state: &mut AppState,
    result: OptimizationStepResult,
    plan: Option<ExecutionPlan<PoolRef, TokenAddress>>,
) {
    // A step result arriving while not optimizing is stale (e.g. it crossed a retarget); ignore it.
    if let Session::Ready {
        work:
            Work::Optimizing {
                last_step,
                plan: held,
                ..
            },
        ..
    } = &mut state.session
    {
        *last_step = Some(result);
        *held = plan;
    }
}

/// Whether a projected reserve set can actually drive the optimizer: non-empty and covering *both*
/// ends of the configured route — something to spend the source asset into and something that yields
/// the output asset (else `init`/`run` would abort with `EmptyReserves` /
/// `SourceAssetNotFound` / `OutputAssetNotFound`). The source half only bites on an open route: when
/// `source == output` reaching the output is reaching the source.
fn is_productive(reserves: &[PoolReserves<PoolRef, TokenAddress>], config: &SessionConfig) -> bool {
    !reserves.is_empty() && reserves_reach_route(reserves, &config.optimization)
}

/// Parses the slice's freshness envelope; the only fallible field is `block_hash`.
fn parse_provenance(slice: &SliceResponse) -> Result<SliceProvenance, EffectError> {
    let block_hash =
        BlockHash::from_str(&slice.block_hash).map_err(|_| EffectError::Provenance {
            field: "block_hash",
            value: slice.block_hash.clone(),
        })?;
    Ok(SliceProvenance {
        block_hash,
        confirmations: slice.confirmations,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use aa_wire::{
        PoolCompleteness, PoolMetaEntry, PoolQuery, PoolSlice, TokenMetaEntry, WirePoolState,
    };
    use optimization::OptimizationStepStatus;
    use proptest::prelude::*;

    use super::*;

    // Tick-0 price (`2^96`): keeps `swap_limit_x/y` non-underflowing for any tick spacing.
    const SQRT_PRICE_TICK_0: u128 = 79_228_162_514_264_337_593_543_950_336;
    const CHAIN: ChainKey = ChainKey::Ethereum;

    fn addr(byte: u8) -> client_evm::Address {
        client_evm::Address::from([byte; 20])
    }

    fn hash(byte: u8) -> BlockHash {
        BlockHash::from([byte; 32])
    }

    fn v3_key(byte: u8) -> PoolQuery {
        PoolQuery::UniswapV3 {
            address: format!("{:#x}", addr(byte)),
        }
    }

    fn wire_state() -> WirePoolState {
        WirePoolState {
            sqrt_price_x96: format!("{:#x}", SQRT_PRICE_TICK_0),
            tick: 0,
            liquidity: format!("{:#x}", 1_000_000_000_000_000_000u128),
        }
    }

    /// A session whose route is the closed cycle on `addr(1)`.
    fn config() -> SessionConfig {
        SessionConfig {
            chain: CHAIN,
            optimization: OptimizationSessionConfig {
                source_asset: TokenAddress(addr(1), CHAIN),
                output_asset: TokenAddress(addr(1), CHAIN),
                bridges: HashSet::new(),
                whitelist: None,
            },
            step: OptimizationStepConfig {
                input_amount: 1.0,
                iterations: 8,
            },
            backend: OptimizationBackendSelection::Cpu,
        }
    }

    /// Catalog with one v3 pool (`addr(9)`) over the given token pair.
    fn meta_over(token0: u8, token1: u8) -> PoolsMetaResponse {
        PoolsMetaResponse {
            pools: vec![PoolMetaEntry {
                key: v3_key(9),
                token0: format!("{:#x}", addr(token0)),
                token1: format!("{:#x}", addr(token1)),
                fee_pips: 3000,
                tick_spacing: 60,
            }],
            tokens: vec![
                TokenMetaEntry {
                    address: format!("{:#x}", addr(token0)),
                    decimals: 18,
                },
                TokenMetaEntry {
                    address: format!("{:#x}", addr(token1)),
                    decimals: 6,
                },
            ],
        }
    }

    /// A slice with the `addr(9)` pool complete, at block `hash(0xbb)`, 2 confirmations.
    fn slice_complete() -> SliceResponse {
        SliceResponse {
            block_hash: format!("{:#x}", hash(0xbb)),
            confirmations: 2,
            pools: vec![PoolSlice {
                key: v3_key(9),
                state: PoolCompleteness::Complete {
                    state: wire_state(),
                },
            }],
        }
    }

    /// A slice for the `addr(9)` pool at a chosen block/confirmations, so a test can distinguish two
    /// snapshots by provenance (freshness), not just by fetch id.
    fn slice_at(block: u8, confirmations: u64) -> SliceResponse {
        SliceResponse {
            block_hash: format!("{:#x}", hash(block)),
            confirmations,
            pools: vec![PoolSlice {
                key: v3_key(9),
                state: PoolCompleteness::Complete {
                    state: wire_state(),
                },
            }],
        }
    }

    /// Idle-tick a state until its in-flight slice request expires and the ledger re-issues it,
    /// returning the state and the *new* (current) slice id. After this, the previously-issued id is
    /// superseded — the ledger will reject its response — which is exactly the "two concurrent slice
    /// fetches on the wire" situation findings (a)/(e) are about.
    fn reissue_slice_after_ttl(mut state: AppState) -> (AppState, FetchId) {
        let mut reissued = None;
        for _ in 0..crate::pending::FETCH_TTL_TICKS {
            let (next, effects) = transition(state, Event::Tick);
            state = next;
            if let Some(id) = effects.iter().find_map(|e| match e {
                Effect::FetchSlice { id, .. } => Some(*id),
                _ => None,
            }) {
                reissued = Some(id);
            }
        }
        (state, reissued.expect("slice re-issued after TTL"))
    }

    fn step_result() -> OptimizationStepResult {
        OptimizationStepResult {
            status: OptimizationStepStatus::Updated,
            input_amount: 1.0,
            output_amount: 1.0,
            profit_amount: 0.0,
            reserves_count: 2,
            disabled_count: 0,
            pool_slots: 2,
            route_entropy: 0.0,
            effective_pools: 1.0,
            routed_pool_count: 1,
            iterations_completed: 8,
        }
    }

    /// The reserves of the `PushReserves` effect in a batch, if one was emitted.
    fn pushed_reserves(effects: &[Effect]) -> Option<&Vec<PoolReserves<PoolRef, TokenAddress>>> {
        effects.iter().find_map(|e| match e {
            Effect::PushReserves { reserves, .. } => Some(reserves),
            _ => None,
        })
    }

    fn any_push_reserves(effects: &[Effect]) -> bool {
        pushed_reserves(effects).is_some()
    }

    /// The id of the `FetchMeta` effect in a batch (the ledger issued it on the cold-start tick).
    fn meta_id(effects: &[Effect]) -> FetchId {
        effects
            .iter()
            .find_map(|e| match e {
                Effect::FetchMeta { id } => Some(*id),
                _ => None,
            })
            .expect("FetchMeta issued")
    }

    /// The id of the `FetchHealth` effect in a batch (issued on every tick the slot is free).
    fn health_id(effects: &[Effect]) -> FetchId {
        effects
            .iter()
            .find_map(|e| match e {
                Effect::FetchHealth { id } => Some(*id),
                _ => None,
            })
            .expect("FetchHealth issued")
    }

    /// The id of the `FetchSlice` effect in a batch.
    fn slice_id(effects: &[Effect]) -> FetchId {
        effects
            .iter()
            .find_map(|e| match e {
                Effect::FetchSlice { id, .. } => Some(*id),
                _ => None,
            })
            .expect("FetchSlice issued")
    }

    /// The id of the effect of `kind` in a batch, if one was issued. The kind-parametric counterpart
    /// of [`meta_id`]/[`health_id`]/[`slice_id`], for properties that quantify over `FetchKind`.
    fn issued_id(effects: &[Effect], kind: FetchKind) -> Option<FetchId> {
        effects.iter().find_map(|effect| match effect_kind(effect) {
            Some((issued, id)) if issued == kind => Some(id),
            _ => None,
        })
    }

    /// Cold-start tick → deliver the catalog. Returns the state (with a slice now in flight) and the
    /// `FetchSlice` id `on_meta` issued, so a caller can deliver a matching slice.
    fn after_catalog(config: SessionConfig, meta: PoolsMetaResponse) -> (AppState, FetchId) {
        let (state, effects) = transition(AppState::started(config), Event::Tick);
        let id = meta_id(&effects);
        let (state, effects) = transition(state, Event::MetaFetched { id, response: meta });
        let slice = slice_id(&effects);
        (state, slice)
    }

    /// Drives cold start → catalog → productive slice and asserts we land in `Optimizing`.
    fn optimizing_state() -> AppState {
        let (state, slice) = after_catalog(config(), meta_over(1, 2));
        let (state, effects) = transition(
            state,
            Event::SliceFetched {
                id: slice,
                response: slice_complete(),
            },
        );
        assert!(any_push_reserves(&effects));
        assert!(is_optimizing(&state));
        state
    }

    /// Whether the engine is grinding the optimizer.
    fn is_optimizing(state: &AppState) -> bool {
        matches!(
            state.session,
            Session::Ready {
                work: Work::Optimizing { .. },
                ..
            }
        )
    }

    /// The work under a held catalog, if there is one — the reduced form most assertions want.
    fn work_of(state: &AppState) -> Option<&Work> {
        match &state.session {
            Session::NoCatalog => None,
            Session::Ready { work, .. } => Some(work),
        }
    }

    /// Why the engine is waiting, if it is.
    fn await_reason(state: &AppState) -> Option<AwaitReason> {
        match work_of(state) {
            Some(Work::Awaiting(reason)) => Some(*reason),
            _ => None,
        }
    }

    /// Everything a transition may change except the route, bundled so "nothing changed" is one
    /// assertion. `SessionConfig` has no `PartialEq` (it embeds the optimizer's own config types), so
    /// the route is compared separately via [`route_of`].
    #[derive(Clone, Debug, PartialEq)]
    struct Observable {
        session: Session,
        health: Option<HealthResponse>,
        last_error: Option<EffectError>,
        pending: PendingFetches,
    }

    fn observable(state: &AppState) -> Observable {
        Observable {
            session: state.session.clone(),
            health: state.health.clone(),
            last_error: state.last_error.clone(),
            pending: state.pending.clone(),
        }
    }

    /// `config()` retargeted to spend `source` and maximize `output` (both `addr`-bytes, this chain).
    fn route_config(source: u8, output: u8) -> SessionConfig {
        let mut config = config();
        config.optimization.source_asset = TokenAddress(addr(source), CHAIN);
        config.optimization.output_asset = TokenAddress(addr(output), CHAIN);
        config
    }

    // ----------------------------------------------------------------------------------------
    // Property-based coverage of the reducer.
    //
    // Shape borrowed from `client-evm`'s kernel tests: a *symbolic* event enum with small index
    // fields (so proptest shrinks to a readable counterexample), a bounded random event sequence,
    // and an `assert_state_invariants` helper applied after **every** transition rather than only
    // at the end of a run. The invariants are the design claims this module's doc comments make —
    // route single-ownership, the productivity gate, effect/ledger correlation — turned into
    // executable checks over arbitrary histories.
    // ----------------------------------------------------------------------------------------

    /// Which [`FetchId`] a generated response carries. `Current` is the id the ledger most recently
    /// issued for that kind (a timely response); `Raw` is an arbitrary one (superseded, forged, or
    /// simply late). Drawn from the same small range the ledger mints from so collisions are real.
    #[derive(Clone, Copy, Debug)]
    enum IdChoice {
        Current,
        Raw(u64),
    }

    /// A symbolic event. Catalogs and slices are chosen by small index rather than generated
    /// wholesale, so the sequence explores *reducer* behaviour (ordering, gating, phase transitions)
    /// rather than re-testing the wire adapter, which has its own properties in `lib.rs`.
    #[derive(Clone, Copy, Debug)]
    enum GeneratedEvent {
        Tick,
        MetaFetched {
            id: IdChoice,
            catalog: u8,
        },
        SliceFetched {
            id: IdChoice,
            catalog: u8,
            block: u8,
            complete: bool,
        },
        HealthFetched {
            id: IdChoice,
        },
        FetchFailed {
            id: IdChoice,
            kind: FetchKind,
        },
        OptimizerStepped {
            output_amount: u8,
        },
        EffectFailed,
        SetRoute {
            source: u8,
            output: u8,
        },
    }

    fn id_choice() -> impl Strategy<Value = IdChoice> {
        prop_oneof![
            // Weighted towards `Current`: a run made only of rejected ids would never get past
            // bootstrap, so the interesting states would be unreachable.
            3 => Just(IdChoice::Current),
            1 => (0u64..24).prop_map(IdChoice::Raw),
        ]
    }

    fn fetch_kind() -> impl Strategy<Value = FetchKind> {
        prop_oneof![
            Just(FetchKind::Meta),
            Just(FetchKind::Health),
            Just(FetchKind::Slice),
        ]
    }

    fn generated_event() -> impl Strategy<Value = GeneratedEvent> {
        prop_oneof![
            3 => Just(GeneratedEvent::Tick),
            2 => (id_choice(), 0u8..CATALOG_COUNT)
                .prop_map(|(id, catalog)| GeneratedEvent::MetaFetched { id, catalog }),
            4 => (id_choice(), 0u8..CATALOG_COUNT, any::<u8>(), any::<bool>())
                .prop_map(|(id, catalog, block, complete)| GeneratedEvent::SliceFetched {
                    id,
                    catalog,
                    block,
                    complete,
                }),
            1 => id_choice().prop_map(|id| GeneratedEvent::HealthFetched { id }),
            1 => (id_choice(), fetch_kind())
                .prop_map(|(id, kind)| GeneratedEvent::FetchFailed { id, kind }),
            2 => any::<u8>().prop_map(|output_amount| GeneratedEvent::OptimizerStepped { output_amount }),
            1 => Just(GeneratedEvent::EffectFailed),
            2 => (0u8..TOKEN_COUNT, 0u8..TOKEN_COUNT)
                .prop_map(|(source, output)| GeneratedEvent::SetRoute { source, output }),
        ]
    }

    fn generated_event_sequence() -> impl Strategy<Value = Vec<GeneratedEvent>> {
        prop::collection::vec(generated_event(), 0..80)
    }

    /// Token bytes a generated route may name: `1..=4`. `1` and `2` are the tokens the catalogs are
    /// built over, so some routes are servable and others deliberately are not.
    const TOKEN_COUNT: u8 = 5;
    /// How many distinct catalogs the generator can serve.
    const CATALOG_COUNT: u8 = 3;

    /// The catalog for a generated index. Deliberately a small fixed set spanning the three cases
    /// that matter to the reducer: a pool pair covering the default route, one covering nothing the
    /// default route names, and a two-pool catalog (so a slice request carries more than one entry).
    fn catalog(index: u8) -> PoolsMetaResponse {
        match index % CATALOG_COUNT {
            0 => meta_over(1, 2),
            1 => meta_over(3, 4),
            _ => PoolsMetaResponse {
                pools: vec![
                    PoolMetaEntry {
                        key: v3_key(9),
                        token0: format!("{:#x}", addr(1)),
                        token1: format!("{:#x}", addr(2)),
                        fee_pips: 3000,
                        tick_spacing: 60,
                    },
                    PoolMetaEntry {
                        key: v3_key(10),
                        token0: format!("{:#x}", addr(2)),
                        token1: format!("{:#x}", addr(3)),
                        fee_pips: 500,
                        tick_spacing: 10,
                    },
                ],
                tokens: (1u8..=4)
                    .map(|byte| TokenMetaEntry {
                        address: format!("{:#x}", addr(byte)),
                        decimals: 18,
                    })
                    .collect(),
            },
        }
    }

    /// A slice answering `catalog(index)`: every pool of that catalog, all `Complete` or all
    /// `Incomplete`, at the given block.
    fn slice_for(index: u8, block: u8, complete: bool) -> SliceResponse {
        SliceResponse {
            block_hash: format!("{:#x}", hash(block)),
            confirmations: u64::from(block),
            pools: catalog(index)
                .pools
                .into_iter()
                .map(|entry| PoolSlice {
                    key: entry.key,
                    state: if complete {
                        PoolCompleteness::Complete {
                            state: wire_state(),
                        }
                    } else {
                        PoolCompleteness::Incomplete
                    },
                })
                .collect(),
        }
    }

    /// The fetch kind an effect requests, if it is a fetch.
    fn effect_kind(effect: &Effect) -> Option<(FetchKind, FetchId)> {
        match effect {
            Effect::FetchMeta { id } => Some((FetchKind::Meta, *id)),
            Effect::FetchHealth { id } => Some((FetchKind::Health, *id)),
            Effect::FetchSlice { id, .. } => Some((FetchKind::Slice, *id)),
            Effect::PushReserves { .. } => None,
        }
    }

    /// Every invariant the reducer must maintain, asserted after a single transition against the
    /// state it produced and the effects it emitted. Returns a message on the first violation.
    fn check_invariants(state: &AppState, effects: &[Effect]) -> Result<(), String> {
        // (1) The session chain is stamped on the route. `Route` carries bare `Address`es precisely
        // so a foreign-chain route is unrepresentable; a route tagged with another chain would match
        // no reserve and silently starve the optimizer forever.
        let route = &state.config.optimization;
        if route.source_asset.1 != state.config.chain || route.output_asset.1 != state.config.chain
        {
            return Err(format!(
                "route {route:?} is not stamped with the session chain {:?}",
                state.config.chain
            ));
        }

        // (2) Every emitted fetch is recorded as pending work under its own id — the client-side
        // analogue of the kernel's `assert_effects_are_well_formed`. A fetch effect whose id the
        // ledger does not hold would have its response rejected on arrival, stalling that kind
        // until the TTL; one recorded under the wrong kind would free the wrong slot.
        for effect in effects {
            if let Some((kind, id)) = effect_kind(effect)
                && !state.pending.clone().accept(kind, id)
            {
                return Err(format!(
                    "emitted {effect:?} is not recorded as pending {kind:?}"
                ));
            }
        }

        // (3) At most one fetch per kind per transition: two concurrent requests of a kind would
        // race, and only one could ever be accepted (the ledger holds a single slot).
        for kind in [FetchKind::Meta, FetchKind::Health, FetchKind::Slice] {
            let issued = effects
                .iter()
                .filter(|effect| effect_kind(effect).map(|(k, _)| k) == Some(kind))
                .count();
            if issued > 1 {
                return Err(format!(
                    "{issued} concurrent {kind:?} fetches in one transition"
                ));
            }
        }

        for effect in effects {
            if let Effect::PushReserves { reserves, session } = effect {
                // (4) `AppState` is the single owner of the route: a pushed snapshot always carries
                // the route currently in force. A snapshot describing a stale route would make the
                // worker keep optimizing the pair the user just navigated away from.
                if session != &state.config.optimization {
                    return Err(format!(
                        "pushed session {session:?} disagrees with the config's {:?}",
                        state.config.optimization
                    ));
                }
                // (5) The productivity gate holds at the push, not just at the call site. An
                // unproductive snapshot aborts `OptimizationRunner::init` — which is fatal to the
                // worker thread, and *nothing restarts it* — so this is a liveness invariant, not a
                // tidiness one.
                if reserves.is_empty() || !reserves_reach_route(reserves, session) {
                    return Err("pushed reserves do not cover the route they were gated for".into());
                }
            }
        }

        // (6) A slice is only ever requested for the catalog currently held, and asks for exactly
        // its pools — the request cannot drift from the catalog its response will be projected
        // against, which is what makes `UnknownPool` unreachable in practice.
        //
        // Its companion — "`NoRoute` is never recorded without a catalog to have projected a slice
        // against" — used to be invariant (7) here. `AwaitReason` now lives under `Session::Ready`,
        // so that state is unrepresentable and the check has nothing left to catch.
        for effect in effects {
            if let Effect::FetchSlice { request, .. } = effect {
                match state.catalog() {
                    Some(catalog) => {
                        if request.pools != catalog.slice_request().pools {
                            return Err("slice request does not match the held catalog".into());
                        }
                    }
                    None => {
                        return Err("slice requested with no catalog to project it against".into());
                    }
                }
            }
        }

        Ok(())
    }

    /// Folds a symbolic event sequence through the reducer, resolving `IdChoice::Current` against
    /// the ids the reducer actually issued (exactly as a real driver echoes them back) and checking
    /// every invariant after each step. Returns the final state.
    fn drive(config: SessionConfig, events: &[GeneratedEvent]) -> Result<AppState, TestCaseError> {
        let mut state = AppState::started(config);
        // The most recent id issued per kind, learned from the effects — the driver's own view.
        let mut live: Vec<(FetchKind, FetchId)> = Vec::new();
        let resolve =
            |live: &[(FetchKind, FetchId)], kind: FetchKind, choice: IdChoice| match choice {
                IdChoice::Raw(raw) => FetchId::from_raw_for_test(raw),
                IdChoice::Current => live
                    .iter()
                    .find(|(held, _)| *held == kind)
                    .map(|(_, id)| *id)
                    .unwrap_or_else(|| FetchId::from_raw_for_test(u64::MAX)),
            };

        for generated in events {
            let event = match *generated {
                GeneratedEvent::Tick => Event::Tick,
                GeneratedEvent::MetaFetched { id, catalog: index } => Event::MetaFetched {
                    id: resolve(&live, FetchKind::Meta, id),
                    response: catalog(index),
                },
                GeneratedEvent::SliceFetched {
                    id,
                    catalog: index,
                    block,
                    complete,
                } => Event::SliceFetched {
                    id: resolve(&live, FetchKind::Slice, id),
                    response: slice_for(index, block, complete),
                },
                GeneratedEvent::HealthFetched { id } => Event::HealthFetched {
                    id: resolve(&live, FetchKind::Health, id),
                    response: aa_wire::HealthResponse::AwaitingAnchor,
                },
                GeneratedEvent::FetchFailed { id, kind } => Event::FetchFailed {
                    id: resolve(&live, kind, id),
                    kind,
                    message: "generated".to_owned(),
                },
                GeneratedEvent::OptimizerStepped { output_amount } => {
                    let mut result = step_result();
                    result.output_amount = f32::from(output_amount);
                    Event::OptimizerStepped { result, plan: None }
                }
                GeneratedEvent::EffectFailed => Event::EffectFailed(EffectError::Optimize {
                    stage: OptimizeStage::Run,
                    message: "generated".to_owned(),
                }),
                GeneratedEvent::SetRoute { source, output } => Event::SetRoute(Route {
                    source: addr(source),
                    output: addr(output),
                }),
            };

            let (next, effects) = transition(state, event);
            if let Err(message) = check_invariants(&next, &effects) {
                return Err(TestCaseError::fail(format!(
                    "invariant violated after {generated:?}: {message}"
                )));
            }
            for (kind, id) in effects.iter().filter_map(effect_kind) {
                live.retain(|(held, _)| *held != kind);
                live.push((kind, id));
            }
            state = next;
        }

        Ok(state)
    }

    /// A route's identity as the reducer stores it, for comparing two states' routes.
    fn route_of(state: &AppState) -> (TokenAddress, TokenAddress) {
        (
            state.config.optimization.source_asset,
            state.config.optimization.output_asset,
        )
    }

    proptest! {
        /// The reducer is **total** and preserves every invariant across arbitrary histories: any
        /// interleaving of ticks, timely/superseded/forged responses, failures, optimizer steps, and
        /// mid-flight retargets leaves a valid state and well-formed effects. This is the property
        /// the whole crate rests on — the engine's decision logic is pure, so if it holds here it
        /// holds in production regardless of transport timing.
        #[test]
        fn transition_is_total_and_preserves_every_invariant(events in generated_event_sequence()) {
            drive(config(), &events)?;
        }

        /// The same, from an *open* route (source ≠ output) rather than the arbitrage default. Open
        /// routes take the other half of the productivity gate (the source asset must be spendable,
        /// not just the output reachable), so they reach states the closed-cycle run cannot.
        #[test]
        fn transition_preserves_every_invariant_on_an_open_route(
            events in generated_event_sequence(),
            source in 0u8..TOKEN_COUNT,
            output in 0u8..TOKEN_COUNT,
        ) {
            drive(route_config(source, output), &events)?;
        }

        /// Retargeting to the route already in force is a total no-op, from *any* reached state:
        /// same phase, same ledger, no effects. `run_engine` seeds the reducer with a `SetRoute`
        /// that may equal the default, and a UI can re-send the current pair on any redraw —
        /// neither may interrupt a running optimization or perturb the fetch ledger.
        #[test]
        fn setting_the_route_already_in_force_changes_nothing(events in generated_event_sequence()) {
            let state = drive(config(), &events)?;
            let (source, output) = route_of(&state);
            let before = observable(&state);

            let (next, effects) = transition(state, Event::SetRoute(Route {
                source: source.0,
                output: output.0,
            }));

            prop_assert!(effects.is_empty(), "a redundant retarget emitted {effects:?}");
            prop_assert_eq!(route_of(&next), (source, output));
            prop_assert_eq!(observable(&next), before);
        }

        /// A retarget always takes effect and always leaves the engine consistent: the config holds
        /// exactly the requested pair (chain-stamped), and no result attributed to the *previous*
        /// route survives into the new one. `last_step`/`plan` are amounts and a swap path for the
        /// old pair — leaving them would put them on screen under the new route's label.
        ///
        /// What is *not* route-dependent survives, and survives **uniformly**: the catalog, the
        /// `/health` snapshot, and any recorded fault. The fault is the sharp case — it describes the
        /// transport, so a retarget must neither invent nor erase one, no matter which lifecycle the
        /// engine was in when the retarget arrived.
        #[test]
        fn a_retarget_takes_effect_and_discards_the_previous_route_s_results(
            events in generated_event_sequence(),
            source in 0u8..TOKEN_COUNT,
            output in 0u8..TOKEN_COUNT,
        ) {
            let state = drive(config(), &events)?;
            let before = route_of(&state);
            let catalog_before = state.catalog().cloned();
            let health_before = state.health.clone();
            let error_before = state.last_error.clone();

            let (next, effects) = transition(state, Event::SetRoute(Route {
                source: addr(source),
                output: addr(output),
            }));

            prop_assert!(effects.is_empty(), "SetRoute emits no effects");
            prop_assert_eq!(
                route_of(&next),
                (TokenAddress(addr(source), CHAIN), TokenAddress(addr(output), CHAIN))
            );
            prop_assert_eq!(next.catalog().cloned(), catalog_before, "the catalog is route-independent");
            prop_assert_eq!(&next.health, &health_before, "`/health` is route-independent");
            prop_assert_eq!(&next.last_error, &error_before, "a fault is about the transport, not the route");

            // A *changed* route must drop back to awaiting; an unchanged one is the no-op above.
            if before != route_of(&next) {
                prop_assert_eq!(
                    await_reason(&next),
                    next.catalog().map(|_| AwaitReason::Bootstrapping),
                    "a retarget must discard the old route's results, got {:?}",
                    next.session
                );
            }
        }

        /// A response carrying an id the ledger is not holding is ignored *entirely*: no phase
        /// change, no ledger change, no effects. This is what makes TTL retry safe — after a
        /// re-issue two replies are on the wire, and applying the abandoned one would move the
        /// engine backwards onto staler reserves. Parametric over all three fetch kinds and over
        /// both success and failure outcomes.
        #[test]
        fn a_response_with_an_unheld_id_is_ignored_entirely(
            events in generated_event_sequence(),
            kind in fetch_kind(),
            raw in 0u64..64,
            failure in any::<bool>(),
        ) {
            let state = drive(config(), &events)?;
            let forged = FetchId::from_raw_for_test(raw);
            // Only exercise ids the ledger really is not holding.
            prop_assume!(!state.pending.clone().accept(kind, forged));

            let before = observable(&state);
            let event = if failure {
                Event::FetchFailed { id: forged, kind, message: "late".to_owned() }
            } else {
                match kind {
                    FetchKind::Meta => Event::MetaFetched { id: forged, response: catalog(0) },
                    FetchKind::Health => Event::HealthFetched {
                        id: forged,
                        response: aa_wire::HealthResponse::AwaitingAnchor,
                    },
                    FetchKind::Slice => Event::SliceFetched {
                        id: forged,
                        response: slice_for(0, 0xcc, true),
                    },
                }
            };

            let (next, effects) = transition(state, event);

            prop_assert!(effects.is_empty(), "an unheld id produced effects: {effects:?}");
            prop_assert_eq!(observable(&next), before);
        }

        /// A fetch failure records a fault naming the same request it frees. `kind` is the only
        /// source of that name — the driver reports a raw message, the reducer builds the typed
        /// error — so the fault a view renders and the slot the ledger reopens cannot name different
        /// requests. Parametric over all three kinds and over the diagnostic text; generalises
        /// `failed_fetch_frees_the_slot_for_reissue`, which pins the `Meta` case concretely.
        #[test]
        fn a_fetch_failure_names_the_kind_it_frees(kind in fetch_kind(), message in any::<String>()) {
            // `/health` and `/pools/meta` go out on the cold-start tick; a slice needs a catalog first.
            let (state, effects) = transition(AppState::started(config()), Event::Tick);
            let (state, effects) = match kind {
                FetchKind::Slice => transition(
                    state,
                    Event::MetaFetched { id: meta_id(&effects), response: catalog(1) },
                ),
                FetchKind::Meta | FetchKind::Health => (state, effects),
            };
            let Some(id) = issued_id(&effects, kind) else {
                return Err(TestCaseError::fail(format!("no {kind:?} fetch in flight")));
            };

            let (state, effects) = transition(
                state,
                Event::FetchFailed { id, kind, message: message.clone() },
            );
            prop_assert!(effects.is_empty(), "a failure schedules nothing: {effects:?}");
            prop_assert_eq!(
                &state.last_error,
                &Some(EffectError::Fetch { what: kind, message }),
                "the recorded fault must name the request that failed"
            );

            // The other half of the same fact: that kind's slot is free, so the next tick re-issues it.
            let (_state, effects) = transition(state, Event::Tick);
            let reissued = issued_id(&effects, kind);
            prop_assert!(
                reissued.is_some_and(|next| next != id),
                "the failed kind must be re-issued with a fresh id, got {reissued:?}"
            );
        }

        /// Optimizer results are last-write-wins and never schedule work: from any reached state, a
        /// run of steps leaves the last one recorded and emits nothing. The worker self-clocks its
        /// own `Continue`, so a reducer that emitted a follow-up effect per step would double-drive
        /// it. Direct analogue of the kernel's
        /// `optimization_step_completed_overwrites_previous_result`.
        #[test]
        fn optimizer_steps_are_last_write_wins_and_emit_nothing(
            events in generated_event_sequence(),
            outputs in prop::collection::vec(any::<u8>(), 1..8),
        ) {
            let mut state = drive(config(), &events)?;
            let optimizing = is_optimizing(&state);

            for output_amount in &outputs {
                let mut result = step_result();
                result.output_amount = f32::from(*output_amount);
                let (next, effects) = transition(state, Event::OptimizerStepped { result, plan: None });
                prop_assert!(effects.is_empty(), "a step scheduled work: {effects:?}");
                state = next;
            }

            match (work_of(&state), optimizing) {
                (Some(Work::Optimizing { last_step: Some(last), .. }), true) => {
                    let expected = outputs.last().copied().unwrap_or_default();
                    prop_assert_eq!(last.output_amount, f32::from(expected));
                }
                // A step that arrives while not optimizing is stale and is dropped by design.
                (_, false) => {}
                (work, _) => prop_assert!(false, "expected a recorded step, got {work:?}"),
            }
        }

        /// The in-flight gate holds *across* transitions, not just within one: over any window of
        /// `FETCH_TTL_TICKS - 1` consecutive ticks with no response delivered, each kind is fetched
        /// at most once. A slot re-issued at the start of the window cannot expire again inside it,
        /// so this bound is exact — and it is precisely what stops a slow or hung server from
        /// accumulating concurrent requests (review finding (a)).
        #[test]
        fn a_ttl_window_of_idle_ticks_issues_each_kind_at_most_once(
            events in generated_event_sequence(),
        ) {
            let mut state = drive(config(), &events)?;
            let mut issued = Vec::new();

            for _ in 0..(crate::pending::FETCH_TTL_TICKS - 1) {
                let (next, effects) = transition(state, Event::Tick);
                issued.extend(effects.iter().filter_map(effect_kind).map(|(kind, _)| kind));
                state = next;
            }

            for kind in [FetchKind::Meta, FetchKind::Health, FetchKind::Slice] {
                let count = issued.iter().filter(|entry| **entry == kind).count();
                prop_assert!(
                    count <= 1,
                    "{kind:?} was fetched {count} times inside one TTL window"
                );
            }
        }

        /// Liveness: whenever a catalog is held and the arriving slice is productive for the route,
        /// the engine *always* ends up optimizing and *always* pushes exactly one snapshot. The
        /// engine has no other way to make progress, so a gate that silently rejected a servable
        /// slice would leave it stuck in `AwaitingFirstSlice` forever with no error to show.
        #[test]
        fn a_productive_slice_always_drives_the_optimizer(
            events in generated_event_sequence(),
            index in 0u8..CATALOG_COUNT,
            block in any::<u8>(),
        ) {
            // Reach a state, then force a known catalog and a fresh slice request for it.
            let state = drive(config(), &events)?;
            let (state, effects) = transition(state, Event::Tick);
            let meta_slot = effects.iter().filter_map(effect_kind)
                .find(|(kind, _)| *kind == FetchKind::Meta)
                .map(|(_, id)| id);
            let (state, effects) = match meta_slot {
                Some(id) => transition(state, Event::MetaFetched { id, response: catalog(index) }),
                None => (state, effects),
            };
            let Some((_, slice)) = effects.iter().filter_map(effect_kind)
                .find(|(kind, _)| *kind == FetchKind::Slice)
            else {
                // No slice slot was free this tick; nothing to assert about.
                return Ok(());
            };
            let held = match state.catalog() {
                Some(catalog) => catalog.clone(),
                None => return Ok(()),
            };

            let response = slice_for(index, block, true);
            // Only meaningful when the slice really answers the catalog the reducer is holding.
            prop_assume!(
                held.slice_request().pools
                    == Catalog::parse(&catalog(index), CHAIN).slice_request().pools
            );

            let reserves = match held.project(&response) {
                Ok(reserves) => reserves,
                Err(_) => return Ok(()),
            };
            prop_assume!(!reserves.is_empty()
                && reserves_reach_route(&reserves, &state.config.optimization));

            let (next, effects) = transition(state, Event::SliceFetched { id: slice, response });

            prop_assert!(
                is_optimizing(&next),
                "a productive slice left the engine in {:?}",
                next.session
            );
            prop_assert_eq!(
                effects.iter().filter(|e| matches!(e, Effect::PushReserves { .. })).count(),
                1,
                "a productive slice must push exactly one snapshot"
            );
        }
    }

    #[test]
    fn set_route_retargets_the_config_without_effects() {
        // The runtime command seam: `SetRoute` retargets the route in place and emits nothing. From a
        // cold start there are no results to discard, so the lifecycle is untouched too (contrast
        // `set_route_to_a_new_route_discards_the_previous_route_s_results`).
        let state = AppState::started(config());
        let before = observable(&state);
        let (next, effects) = transition(
            state,
            Event::SetRoute(Route {
                source: addr(1),
                output: addr(2),
            }),
        );
        assert!(
            effects.is_empty(),
            "SetRoute emits no effects this increment"
        );
        assert_eq!(
            next.config.optimization.source_asset,
            TokenAddress(addr(1), CHAIN)
        );
        assert_eq!(
            next.config.optimization.output_asset,
            TokenAddress(addr(2), CHAIN)
        );
        assert_eq!(
            observable(&next),
            before,
            "the lifecycle is untouched by SetRoute"
        );
    }

    #[test]
    fn productivity_gate_honors_the_configured_output_asset() {
        // One and the same slice — a single pool over (addr(1), addr(2)) — drives the gate two ways.
        // With `output = addr(2)` the reserves reach the sink, so the engine starts optimizing; with
        // `output = addr(3)` (a token absent from the slice) they do not, so it stays awaiting. Proves
        // `is_productive` reads the *configured* output, not a hardcoded asset.
        let (state, slice) = after_catalog(route_config(1, 2), meta_over(1, 2));
        let (state, effects) = transition(
            state,
            Event::SliceFetched {
                id: slice,
                response: slice_complete(),
            },
        );
        assert!(
            any_push_reserves(&effects),
            "output = addr(2) is reachable ⇒ productive"
        );
        assert!(is_optimizing(&state));

        let (state, slice) = after_catalog(route_config(1, 3), meta_over(1, 2));
        let (state, effects) = transition(
            state,
            Event::SliceFetched {
                id: slice,
                response: slice_complete(),
            },
        );
        assert!(
            !any_push_reserves(&effects),
            "output = addr(3) is absent ⇒ not productive"
        );
        assert_eq!(await_reason(&state), Some(AwaitReason::NoRoute));
    }

    #[test]
    fn set_route_to_a_new_route_discards_the_previous_route_s_results() {
        // A retarget invalidates everything the optimizer produced for the old route. Staying in
        // `Optimizing` would leave `last_step`/`plan` — amounts and a swap path for the *previous*
        // pair — on display under the new route's label until a fresh slice lands. Drop back to
        // awaiting instead, keeping the catalog (route-independent) so no refetch is needed.
        let state = optimizing_state();
        let (state, _) = transition(
            state,
            Event::OptimizerStepped {
                result: step_result(),
                plan: None,
            },
        );
        assert!(matches!(
            work_of(&state),
            Some(Work::Optimizing {
                last_step: Some(_),
                ..
            })
        ));

        let (state, effects) = transition(
            state,
            Event::SetRoute(Route {
                source: addr(1),
                output: addr(2),
            }),
        );

        assert!(state.catalog().is_some(), "the catalog survives a retarget");
        assert_eq!(await_reason(&state), Some(AwaitReason::Bootstrapping));
        assert!(effects.is_empty(), "SetRoute emits no effects");
    }

    #[test]
    fn set_route_to_the_current_route_is_a_no_op() {
        // `run_engine` seeds the reducer with a `SetRoute` whose route may equal the default, and a UI
        // can re-send the current pair. Neither may interrupt a running optimization.
        let state = optimizing_state();
        let route = state.config.optimization.clone();
        let (state, effects) = transition(
            state,
            Event::SetRoute(Route {
                source: route.source_asset.0,
                output: route.output_asset.0,
            }),
        );

        assert!(is_optimizing(&state));
        assert!(effects.is_empty());
    }

    #[test]
    fn a_recorded_fault_survives_a_retarget_from_either_lifecycle() {
        // A fault describes the transport, not the route, so a retarget must not clear it — and must
        // not clear it *only sometimes*. While the fault lived in two places (a wait-status variant
        // when awaiting, a field when optimizing) the optimizing copy was silently dropped here,
        // because `on_set_route` rebuilt `AwaitingFirstSlice` from the parts it bothered to name.
        // One home, one rule: both halves of this test now exercise the same line.
        let retarget = Event::SetRoute(Route {
            source: addr(7),
            output: addr(8),
        });
        let fault = || {
            Event::EffectFailed(EffectError::Optimize {
                stage: OptimizeStage::Run,
                message: "boom".to_owned(),
            })
        };

        // From `Awaiting`: the case that always worked.
        let (state, _) = transition(AppState::started(config()), fault());
        let (state, _) = transition(state, retarget.clone());
        assert!(
            state.last_error.is_some(),
            "a fault recorded while awaiting must survive a retarget"
        );

        // From `Optimizing`: the case that silently lost the fault.
        let (state, _) = transition(optimizing_state(), fault());
        assert!(state.last_error.is_some(), "the fault is recorded");
        let (state, _) = transition(state, retarget);
        assert!(
            state.last_error.is_some(),
            "a fault recorded while optimizing must survive a retarget too"
        );
    }

    #[test]
    fn a_transport_fault_does_not_erase_a_no_route_verdict() {
        // Two independent facts about an idle engine: *why* it is not optimizing (this route is not
        // servable by the catalog) and *what last went wrong on the wire*. They were unioned into one
        // `AwaitStatus`, so recording either erased the other — a failed `/health` poll would wipe a
        // `NoRoute` verdict it says nothing about. Separate fields, so both are observable at once.
        let (state, effects) = transition(AppState::started(route_config(1, 3)), Event::Tick);
        let health = health_id(&effects);
        let (state, effects) = transition(
            state,
            Event::MetaFetched {
                id: meta_id(&effects),
                response: meta_over(1, 2),
            },
        );
        let (state, _) = transition(
            state,
            Event::SliceFetched {
                id: slice_id(&effects),
                response: slice_complete(),
            },
        );
        assert_eq!(await_reason(&state), Some(AwaitReason::NoRoute));

        let (state, _) = transition(
            state,
            Event::FetchFailed {
                id: health,
                kind: FetchKind::Health,
                message: "boom".to_owned(),
            },
        );

        assert_eq!(
            await_reason(&state),
            Some(AwaitReason::NoRoute),
            "an unrelated fetch failure must not overwrite why the engine is waiting"
        );
        assert!(matches!(
            state.last_error,
            Some(EffectError::Fetch {
                what: FetchKind::Health,
                ..
            })
        ));
    }

    #[test]
    fn productivity_gate_requires_the_source_asset_to_be_present() {
        // A route *out of* a token the catalog never mentions: the slice reaches the output asset
        // (`addr(2)`), so the output-reachability half of the gate passes, but nothing can be spent
        // from `addr(3)`. Pushing such a snapshot aborts the optimizer's model init — and that abort
        // is fatal to the worker thread — so the gate must reject it and keep the engine awaiting.
        let (state, slice) = after_catalog(route_config(3, 2), meta_over(1, 2));
        let (state, effects) = transition(
            state,
            Event::SliceFetched {
                id: slice,
                response: slice_complete(),
            },
        );
        assert!(
            !any_push_reserves(&effects),
            "source = addr(3) is absent from the slice ⇒ not productive"
        );
        assert_eq!(await_reason(&state), Some(AwaitReason::NoRoute));
    }

    #[test]
    fn cold_start_tick_bootstraps_catalog() {
        let (_state, effects) = transition(AppState::started(config()), Event::Tick);
        assert!(
            effects
                .iter()
                .any(|e| matches!(e, Effect::FetchMeta { .. }))
        );
        assert!(
            effects
                .iter()
                .any(|e| matches!(e, Effect::FetchHealth { .. }))
        );
        // Without a catalog yet, we must not request a slice.
        assert!(
            !effects
                .iter()
                .any(|e| matches!(e, Effect::FetchSlice { .. }))
        );
    }

    #[test]
    fn tick_with_catalog_requests_slice() {
        // Catalog known and its slice already delivered (slot free), over tokens with no init-asset
        // route so we stay in `AwaitingFirstSlice`.
        let (state, slice) = after_catalog(config(), meta_over(3, 4));
        let (state, _) = transition(
            state,
            Event::SliceFetched {
                id: slice,
                response: slice_complete(),
            },
        );
        let (_state, effects) = transition(state, Event::Tick);
        assert!(
            !effects
                .iter()
                .any(|e| matches!(e, Effect::FetchMeta { .. }))
        );
        assert!(
            effects
                .iter()
                .any(|e| matches!(e, Effect::FetchSlice { .. }))
        );
    }

    #[test]
    fn fresh_in_flight_slice_is_not_reissued_on_the_next_tick() {
        // `after_catalog` leaves a slice in flight; a tick right after must NOT issue another (gating).
        let (state, _slice) = after_catalog(config(), meta_over(1, 2));
        let (_state, effects) = transition(state, Event::Tick);
        assert!(
            !effects
                .iter()
                .any(|e| matches!(e, Effect::FetchSlice { .. }))
        );
    }

    #[test]
    fn meta_fetched_requests_slice_for_its_pools() {
        let (state, effects) = transition(AppState::started(config()), Event::Tick);
        let id = meta_id(&effects);
        let (state, effects) = transition(
            state,
            Event::MetaFetched {
                id,
                response: meta_over(1, 2),
            },
        );
        let request = effects
            .iter()
            .find_map(|e| match e {
                Effect::FetchSlice { request, .. } => Some(request),
                _ => None,
            })
            .expect("slice request");
        assert_eq!(request.pools, vec![v3_key(9)]);
        assert_eq!(await_reason(&state), Some(AwaitReason::Bootstrapping));
    }

    #[test]
    fn first_productive_slice_pushes_reserves() {
        let (state, slice) = after_catalog(config(), meta_over(1, 2));
        let (state, effects) = transition(
            state,
            Event::SliceFetched {
                id: slice,
                response: slice_complete(),
            },
        );

        // The reducer no longer inits directly: it hands the worker the projected reserves.
        let reserves = pushed_reserves(&effects).expect("push reserves effect");
        // The one v3 pool projects to forward + inverse.
        assert_eq!(reserves.len(), 2);

        let Some(Work::Optimizing {
            latest, last_step, ..
        }) = work_of(&state)
        else {
            panic!("expected Optimizing, got {:?}", state.session);
        };
        assert_eq!(latest.confirmations, 2);
        assert_eq!(latest.block_hash, hash(0xbb));
        assert!(last_step.is_none());
    }

    #[test]
    fn subsequent_slice_also_pushes_reserves() {
        let state = optimizing_state();
        // Issue a fresh slice (the first was consumed reaching `Optimizing`), then deliver it.
        let (state, effects) = transition(state, Event::Tick);
        let slice = slice_id(&effects);
        let (state, effects) = transition(
            state,
            Event::SliceFetched {
                id: slice,
                response: slice_complete(),
            },
        );
        // Init vs update is the worker's concern now; the reducer just pushes the freshest reserves.
        assert!(any_push_reserves(&effects));
        assert!(is_optimizing(&state));
    }

    #[test]
    fn step_records_result_without_re_issuing_continue() {
        let state = optimizing_state();
        let (state, effects) = transition(
            state,
            Event::OptimizerStepped {
                result: step_result(),
                plan: None,
            },
        );
        // The worker self-clocks its own `Continue`; the reducer emits no follow-up effect.
        assert!(effects.is_empty());
        let Some(Work::Optimizing { last_step, .. }) = work_of(&state) else {
            panic!("expected Optimizing");
        };
        assert!(last_step.is_some());
    }

    #[test]
    fn unproductive_slice_stays_awaiting_without_optimizing() {
        // Catalog over tokens that do NOT include the route asset (`addr(1)`).
        let (state, slice) = after_catalog(config(), meta_over(3, 4));
        let (state, effects) = transition(
            state,
            Event::SliceFetched {
                id: slice,
                response: slice_complete(),
            },
        );
        assert!(!any_push_reserves(&effects));
        assert_eq!(await_reason(&state), Some(AwaitReason::NoRoute));
    }

    #[test]
    fn bad_slice_hex_records_adapter_error_without_optimizing() {
        let (state, slice) = after_catalog(config(), meta_over(1, 2));
        // Same pool, but a malformed price string.
        let bad = SliceResponse {
            block_hash: format!("{:#x}", hash(0xbb)),
            confirmations: 0,
            pools: vec![PoolSlice {
                key: v3_key(9),
                state: PoolCompleteness::Complete {
                    state: WirePoolState {
                        sqrt_price_x96: "0xzz".to_owned(),
                        tick: 0,
                        liquidity: format!("{:#x}", 1u128),
                    },
                },
            }],
        };
        let (state, effects) = transition(
            state,
            Event::SliceFetched {
                id: slice,
                response: bad,
            },
        );
        assert!(!any_push_reserves(&effects));
        assert!(matches!(
            state.last_error,
            Some(EffectError::Adapter(WireAdapterError::HexParse { .. }))
        ));
        // The fault is recorded *beside* the wait-reason, not in place of it: a malformed payload
        // says nothing about whether the route is servable, so the engine is still bootstrapping.
        assert_eq!(await_reason(&state), Some(AwaitReason::Bootstrapping));
    }

    #[test]
    fn mismatched_slice_id_is_rejected_then_the_real_id_is_accepted() {
        let (state, slice) = after_catalog(config(), meta_over(1, 2));
        let before = observable(&state);

        // A slice carrying a stale id the ledger never issued is dropped: no effects, no state change.
        let stale = FetchId::from_raw_for_test(9999);
        let (state, effects) = transition(
            state,
            Event::SliceFetched {
                id: stale,
                response: slice_complete(),
            },
        );
        assert!(effects.is_empty());
        assert_eq!(observable(&state), before);

        // The real in-flight id is still accepted and drives the optimizer.
        let (state, effects) = transition(
            state,
            Event::SliceFetched {
                id: slice,
                response: slice_complete(),
            },
        );
        assert!(any_push_reserves(&effects));
        assert!(is_optimizing(&state));
    }

    #[test]
    fn expired_slice_is_reissued_with_a_new_id() {
        let (mut state, first) = after_catalog(config(), meta_over(1, 2));
        // Idle ticks (no slice delivered) until the TTL elapses; the slot is then re-issued.
        let mut reissued = None;
        for _ in 0..crate::pending::FETCH_TTL_TICKS {
            let (next, effects) = transition(state, Event::Tick);
            state = next;
            if let Some(effect_id) = effects.iter().find_map(|e| match e {
                Effect::FetchSlice { id, .. } => Some(*id),
                _ => None,
            }) {
                reissued = Some(effect_id);
            }
        }
        assert_ne!(reissued.expect("slice re-issued after TTL"), first);
    }

    #[test]
    fn failed_fetch_frees_the_slot_for_reissue() {
        let (state, effects) = transition(AppState::started(config()), Event::Tick);
        let id = meta_id(&effects);
        // A failure for the in-flight meta is recorded and frees the slot.
        let (state, effects) = transition(
            state,
            Event::FetchFailed {
                id,
                kind: FetchKind::Meta,
                message: "boom".to_owned(),
            },
        );
        assert!(effects.is_empty());
        assert!(matches!(state.last_error, Some(EffectError::Fetch { .. })));
        // The next tick re-issues meta with a fresh id (still no catalog).
        let (_state, effects) = transition(state, Event::Tick);
        assert_ne!(meta_id(&effects), id);
    }

    #[test]
    fn stale_failure_is_rejected() {
        let (state, effects) = transition(AppState::started(config()), Event::Tick);
        let real = meta_id(&effects);
        let before = observable(&state);
        // A failure carrying an id the ledger no longer holds is ignored — state is unchanged.
        let stale = FetchId::from_raw_for_test(9999);
        assert_ne!(real, stale);
        let (state, effects) = transition(
            state,
            Event::FetchFailed {
                id: stale,
                kind: FetchKind::Meta,
                message: "boom".to_owned(),
            },
        );
        assert!(effects.is_empty());
        assert_eq!(observable(&state), before);
    }

    // --- Review-2 findings (a) & (e): fetch in-flight gating and out-of-order slice application. ---
    // These pin the scenarios the grind-semantics review flagged before the `PendingFetches` ledger
    // existed. With the ledger they are regression guards: (a) one-per-kind gating with TTL retry, and
    // (e) `accept(id)` making `on_slice` apply only the latest-issued slice, rejecting a superseded one.

    #[test]
    fn slow_server_does_not_pile_up_concurrent_slice_fetches() {
        // Finding (a): `after_catalog` leaves a slice in flight. While it is outstanding (and within
        // its TTL), no tick may issue a second slice fetch — the gate is what prevents a slow server
        // from accumulating concurrent requests.
        let (mut state, _in_flight) = after_catalog(config(), meta_over(1, 2));
        for _ in 0..(crate::pending::FETCH_TTL_TICKS - 1) {
            let (next, effects) = transition(state, Event::Tick);
            state = next;
            assert!(
                !effects
                    .iter()
                    .any(|e| matches!(e, Effect::FetchSlice { .. })),
                "an in-flight slice must not be re-issued before its TTL elapses"
            );
        }
    }

    #[test]
    fn superseded_slice_from_a_prior_issuance_is_rejected() {
        // Finding (e): after a TTL re-issue, two slice ids exist on the wire — the stale first one and
        // the current second. The stale one's response must be dropped (its id is no longer held),
        // leaving the engine untouched; the current one is still accepted and drives the optimizer.
        let (state, stale) = after_catalog(config(), meta_over(1, 2));
        let (state, current) = reissue_slice_after_ttl(state);
        assert_ne!(stale, current);

        let (state, effects) = transition(
            state,
            Event::SliceFetched {
                id: stale,
                response: slice_complete(),
            },
        );
        assert!(!any_push_reserves(&effects));
        assert_eq!(await_reason(&state), Some(AwaitReason::Bootstrapping));

        let (state, effects) = transition(
            state,
            Event::SliceFetched {
                id: current,
                response: slice_complete(),
            },
        );
        assert!(any_push_reserves(&effects));
        assert!(is_optimizing(&state));
    }

    #[test]
    fn out_of_order_slice_resolution_keeps_the_latest_issued_provenance() {
        // Finding (e), the sharper edge: two concurrent slice fetches resolve out of order. The newer
        // (current) request lands first and sets provenance; the older (superseded) request lands
        // second carrying a *different* block — and must NOT overwrite `latest` backwards. The `accept`
        // id-gate is what makes the otherwise-blind `on_slice` overwrite safe.
        let state = optimizing_state();
        let (state, effects) = transition(state, Event::Tick);
        let stale = slice_id(&effects);
        let (state, current) = reissue_slice_after_ttl(state);
        assert_ne!(stale, current);

        // Current resolves first, at block 0xcc / 1 confirmation.
        let (state, _effects) = transition(
            state,
            Event::SliceFetched {
                id: current,
                response: slice_at(0xcc, 1),
            },
        );
        let Some(Work::Optimizing { latest, .. }) = work_of(&state) else {
            panic!("expected Optimizing");
        };
        assert_eq!(latest.block_hash, hash(0xcc));
        assert_eq!(latest.confirmations, 1);

        // The superseded older response arrives late at a different block and is rejected: no reserves
        // pushed, and provenance does not regress.
        let (state, effects) = transition(
            state,
            Event::SliceFetched {
                id: stale,
                response: slice_at(0xaa, 9),
            },
        );
        assert!(!any_push_reserves(&effects));
        let Some(Work::Optimizing { latest, .. }) = work_of(&state) else {
            panic!("expected Optimizing");
        };
        assert_eq!(
            latest.block_hash,
            hash(0xcc),
            "a superseded slice must not move `latest` provenance backwards"
        );
        assert_eq!(latest.confirmations, 1);
    }
}
