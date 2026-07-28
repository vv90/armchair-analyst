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
use crate::{WireAdapterError, slice_to_reserves};

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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SliceProvenance {
    /// The frontier block hash the reserves were projected from.
    pub block_hash: BlockHash,
    /// Canonical blocks the observed tip was ahead of that frontier (reorg-depth / staleness).
    pub confirmations: u64,
    /// The chain the reserves belong to (echoes the session's `chain`).
    pub chain: ChainKey,
}

/// The whole application state: immutable session config, the current lifecycle [`Phase`], and the
/// in-flight fetch ledger that gates and correlates data-plane requests.
#[derive(Clone, Debug)]
pub struct AppState {
    /// The session's fixed strategy/cadence for its whole lifetime.
    pub config: SessionConfig,
    /// Where in the bootstrap→optimize lifecycle the engine currently is.
    pub phase: Phase,
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
            phase: Phase::AwaitingFirstSlice {
                meta: None,
                health: None,
                status: AwaitStatus::Bootstrapping,
            },
            pending: PendingFetches::new(),
        }
    }
}

/// The engine lifecycle. `AwaitingFirstSlice` is the pre-optimizer state (mirrors the server's
/// `AwaitingAnchor → Running`): the optimizer requires a non-empty reserve set covering the
/// configured route, so it is only initialized once such a slice arrives. `last_step`/`plan`/`latest` only exist
/// in `Optimizing`, so "optimizer running but no reserves yet" is unrepresentable.
#[derive(Clone, Debug, PartialEq)]
pub enum Phase {
    /// No productive reserve snapshot has been applied yet; the optimizer has not been initialized.
    AwaitingFirstSlice {
        /// The static catalog, once fetched — needed to project any slice.
        meta: Option<PoolsMetaResponse>,
        /// The latest server freshness snapshot, if `/health` has been polled.
        health: Option<HealthResponse>,
        /// Why the engine is still waiting (bootstrapping, no route, or a recorded fault).
        status: AwaitStatus,
    },
    /// The optimizer has been initialized and is being fed fresh reserves + grinding iterations.
    Optimizing {
        /// The static catalog used to project each slice.
        meta: PoolsMetaResponse,
        /// Freshness envelope of the reserves currently driving the optimizer.
        latest: SliceProvenance,
        /// The most recent optimizer step result, once one has come back.
        last_step: Option<OptimizationStepResult>,
        /// The executable plan recovered from the most recent step, if any.
        plan: Option<ExecutionPlan<PoolRef, TokenAddress>>,
        /// The latest server freshness snapshot, if polled.
        health: Option<HealthResponse>,
        /// The most recent recorded fault (adapter/fetch), cleared when a good slice is applied.
        last_error: Option<EffectError>,
    },
}

/// Why the engine is still in `AwaitingFirstSlice`. A view-facing status, not a control signal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AwaitStatus {
    /// Waiting for the initial catalog/slice to arrive.
    Bootstrapping,
    /// A slice arrived but its reserves do not cover the configured route — either nothing to spend
    /// the source asset into or nothing yielding the output asset — so the optimizer is intentionally
    /// not initialized (it would abort on such a snapshot).
    NoRoute,
    /// A fetch or projection fault was recorded while still awaiting the first productive slice.
    Error(EffectError),
}

/// A fault surfaced into view state. Never a panic: a malformed server response or a failed fetch
/// degrades to a recorded, typed error, and the engine stays in a valid phase.
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
    /// A data-plane fetch for request `id` (of `kind`) failed; `error` is its recorded fault. Distinct
    /// from [`Event::EffectFailed`] (optimizer faults) so the reducer can free the right ledger slot.
    FetchFailed {
        /// The id of the fetch that failed.
        id: FetchId,
        /// Which data-plane request it was, so the reducer clears the matching slot.
        kind: FetchKind,
        /// The recorded fault.
        error: EffectError,
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

/// Builds the `POST /slice` request that asks for every catalog pool's current-tick state.
pub fn slice_request_for(meta: &PoolsMetaResponse) -> SliceRequest {
    SliceRequest {
        pools: meta.pools.iter().map(|entry| entry.key.clone()).collect(),
    }
}

/// The reducer: fold one [`Event`] into the state, returning the next state and the effects to run.
/// Pure and total — every event maps to a valid next phase, and any data fault becomes a recorded
/// [`EffectError`] rather than a panic.
pub fn transition(state: AppState, event: Event) -> (AppState, Vec<Effect>) {
    let AppState {
        mut config,
        phase,
        pending,
    } = state;
    let (phase, pending, effects) = reduce(&mut config, phase, pending, event);
    (
        AppState {
            config,
            phase,
            pending,
        },
        effects,
    )
}

/// `config` is `&mut` for the single event that retargets it ([`Event::SetRoute`], the command seam);
/// every other arm only reads it. Keeping that arm here rather than intercepting it in `transition`
/// leaves exactly one dispatch point over `Event`, so no variant is handled in two places.
fn reduce(
    config: &mut SessionConfig,
    phase: Phase,
    mut pending: PendingFetches,
    event: Event,
) -> (Phase, PendingFetches, Vec<Effect>) {
    match event {
        Event::Tick => on_tick(phase, pending),
        Event::MetaFetched { id, response } => {
            if pending.accept(FetchKind::Meta, id) {
                on_meta(response, phase, pending)
            } else {
                (phase, pending, vec![])
            }
        }
        Event::SliceFetched { id, response } => {
            if pending.accept(FetchKind::Slice, id) {
                let (phase, effects) = on_slice(config, phase, response);
                (phase, pending, effects)
            } else {
                (phase, pending, vec![])
            }
        }
        Event::HealthFetched { id, response } => {
            if pending.accept(FetchKind::Health, id) {
                (with_health(phase, response), pending, vec![])
            } else {
                (phase, pending, vec![])
            }
        }
        Event::FetchFailed { id, kind, error } => {
            if pending.accept(kind, id) {
                (with_error(phase, error), pending, vec![])
            } else {
                (phase, pending, vec![])
            }
        }
        Event::OptimizerStepped { result, plan } => {
            let (phase, effects) = on_step(phase, result, plan);
            (phase, pending, effects)
        }
        Event::EffectFailed(error) => (with_error(phase, error), pending, vec![]),
        Event::SetRoute(route) => (on_set_route(config, phase, route), pending, vec![]),
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
fn on_set_route(config: &mut SessionConfig, phase: Phase, route: Route) -> Phase {
    // The session's chain is stamped on here — the one place the client decides a chain — exactly as
    // `slice_to_reserves` stamps it onto every projected pool and token.
    let source = TokenAddress(route.source, config.chain);
    let output = TokenAddress(route.output, config.chain);
    if config.optimization.source_asset == source && config.optimization.output_asset == output {
        return phase;
    }
    config.optimization.source_asset = source;
    config.optimization.output_asset = output;

    let (meta, health, status) = match phase {
        Phase::Optimizing { meta, health, .. } => (Some(meta), health, AwaitStatus::Bootstrapping),
        Phase::AwaitingFirstSlice {
            meta,
            health,
            status,
        } => {
            // A recorded fetch/adapter fault is about the transport, not the route, so it outlives a
            // retarget; a `NoRoute` verdict was about the old route and does not.
            let status = match status {
                AwaitStatus::Error(error) => AwaitStatus::Error(error),
                _ => AwaitStatus::Bootstrapping,
            };
            (meta, health, status)
        }
    };
    Phase::AwaitingFirstSlice {
        meta,
        health,
        status,
    }
}

/// The poll clock fired: advance the ledger clock, then ensure the desired fetches are in flight —
/// `/health` always, plus a fresh slice for the known catalog (else the catalog itself). `ensure`
/// gates on the ledger, so a fetch is issued only when that kind is free or its request has expired;
/// the periodic `Tick` subscription re-arms the clock, so the reducer no longer schedules it.
fn on_tick(phase: Phase, mut pending: PendingFetches) -> (Phase, PendingFetches, Vec<Effect>) {
    pending.advance();
    let mut effects = Vec::new();
    if let Some(id) = pending.ensure(FetchKind::Health) {
        effects.push(Effect::FetchHealth { id });
    }
    match meta_of(&phase) {
        Some(meta) => {
            let request = slice_request_for(meta);
            if let Some(id) = pending.ensure(FetchKind::Slice) {
                effects.push(Effect::FetchSlice { id, request });
            }
        }
        None => {
            if let Some(id) = pending.ensure(FetchKind::Meta) {
                effects.push(Effect::FetchMeta { id });
            }
        }
    }
    (phase, pending, effects)
}

/// The catalog arrived: store it and immediately request the first slice for it (don't wait a full
/// poll interval). The slice slot is free at this point, so `ensure` issues.
fn on_meta(
    meta: PoolsMetaResponse,
    phase: Phase,
    mut pending: PendingFetches,
) -> (Phase, PendingFetches, Vec<Effect>) {
    let request = slice_request_for(&meta);
    let phase = set_meta(phase, meta);
    let effects = match pending.ensure(FetchKind::Slice) {
        Some(id) => vec![Effect::FetchSlice { id, request }],
        None => vec![],
    };
    (phase, pending, effects)
}

/// A slice arrived: project it against the catalog and, if it yields a productive reserve set,
/// (re)drive the optimizer. Faults and unproductive snapshots are recorded without leaving a valid
/// phase and without touching the optimizer.
fn on_slice(config: &SessionConfig, phase: Phase, slice: SliceResponse) -> (Phase, Vec<Effect>) {
    match phase {
        Phase::AwaitingFirstSlice {
            meta: Some(meta),
            health,
            status: _,
        } => {
            let reserves = match slice_to_reserves(&slice, &meta, config.chain) {
                Ok(reserves) => reserves,
                Err(error) => {
                    return (
                        Phase::AwaitingFirstSlice {
                            meta: Some(meta),
                            health,
                            status: AwaitStatus::Error(EffectError::Adapter(error)),
                        },
                        vec![],
                    );
                }
            };
            let provenance = match parse_provenance(&slice, config.chain) {
                Ok(provenance) => provenance,
                Err(error) => {
                    return (
                        Phase::AwaitingFirstSlice {
                            meta: Some(meta),
                            health,
                            status: AwaitStatus::Error(error),
                        },
                        vec![],
                    );
                }
            };
            if !is_productive(&reserves, config) {
                return (
                    Phase::AwaitingFirstSlice {
                        meta: Some(meta),
                        health,
                        status: AwaitStatus::NoRoute,
                    },
                    vec![],
                );
            }
            // The worker inits on this first productive snapshot; later slices arrive as `NewReserves`.
            (
                Phase::Optimizing {
                    meta,
                    latest: provenance,
                    last_step: None,
                    plan: None,
                    health,
                    last_error: None,
                },
                vec![Effect::PushReserves {
                    reserves,
                    session: config.optimization.clone(),
                }],
            )
        }
        Phase::AwaitingFirstSlice {
            meta: None,
            health,
            status,
        } => {
            // A slice with no catalog to project it against; nothing to do until `/pools/meta` lands.
            (
                Phase::AwaitingFirstSlice {
                    meta: None,
                    health,
                    status,
                },
                vec![],
            )
        }
        Phase::Optimizing {
            meta,
            latest,
            last_step,
            plan,
            health,
            last_error,
        } => {
            let reserves = match slice_to_reserves(&slice, &meta, config.chain) {
                Ok(reserves) => reserves,
                Err(error) => {
                    return (
                        Phase::Optimizing {
                            meta,
                            latest,
                            last_step,
                            plan,
                            health,
                            last_error: Some(EffectError::Adapter(error)),
                        },
                        vec![],
                    );
                }
            };
            let provenance = match parse_provenance(&slice, config.chain) {
                Ok(provenance) => provenance,
                Err(error) => {
                    return (
                        Phase::Optimizing {
                            meta,
                            latest,
                            last_step,
                            plan,
                            health,
                            last_error: Some(error),
                        },
                        vec![],
                    );
                }
            };
            if !is_productive(&reserves, config) {
                // A momentarily unreachable snapshot would abort the runner; skip it and keep
                // optimizing on the reserves already loaded.
                return (
                    Phase::Optimizing {
                        meta,
                        latest,
                        last_step,
                        plan,
                        health,
                        last_error,
                    },
                    vec![],
                );
            }
            // Push the freshest reserves; the worker coalesces and applies them as `NewReserves`.
            (
                Phase::Optimizing {
                    meta,
                    latest: provenance,
                    last_step,
                    plan,
                    health,
                    last_error: None,
                },
                vec![Effect::PushReserves {
                    reserves,
                    session: config.optimization.clone(),
                }],
            )
        }
    }
}

/// A step came back: record its result and plan. No effect follows — the worker self-clocks, pulling
/// fresh reserves or self-continuing on its own thread, so the reducer never re-issues a `Continue`.
fn on_step(
    phase: Phase,
    result: OptimizationStepResult,
    plan: Option<ExecutionPlan<PoolRef, TokenAddress>>,
) -> (Phase, Vec<Effect>) {
    match phase {
        Phase::Optimizing {
            meta,
            latest,
            health,
            last_error,
            ..
        } => (
            Phase::Optimizing {
                meta,
                latest,
                last_step: Some(result),
                plan,
                health,
                last_error,
            },
            vec![],
        ),
        // A step result while not optimizing is stale (e.g. after a reset); ignore it.
        other => (other, vec![]),
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
fn parse_provenance(
    slice: &SliceResponse,
    chain: ChainKey,
) -> Result<SliceProvenance, EffectError> {
    let block_hash =
        BlockHash::from_str(&slice.block_hash).map_err(|_| EffectError::Provenance {
            field: "block_hash",
            value: slice.block_hash.clone(),
        })?;
    Ok(SliceProvenance {
        block_hash,
        confirmations: slice.confirmations,
        chain,
    })
}

/// The catalog visible in the current phase, if any.
fn meta_of(phase: &Phase) -> Option<&PoolsMetaResponse> {
    match phase {
        Phase::AwaitingFirstSlice { meta, .. } => meta.as_ref(),
        Phase::Optimizing { meta, .. } => Some(meta),
    }
}

/// Stores/refreshes the catalog in whichever phase is current.
fn set_meta(phase: Phase, meta: PoolsMetaResponse) -> Phase {
    match phase {
        Phase::AwaitingFirstSlice { health, status, .. } => Phase::AwaitingFirstSlice {
            meta: Some(meta),
            health,
            status,
        },
        Phase::Optimizing {
            latest,
            last_step,
            plan,
            health,
            last_error,
            ..
        } => Phase::Optimizing {
            meta,
            latest,
            last_step,
            plan,
            health,
            last_error,
        },
    }
}

/// Stores the latest `/health` snapshot in whichever phase is current.
fn with_health(phase: Phase, health: HealthResponse) -> Phase {
    match phase {
        Phase::AwaitingFirstSlice { meta, status, .. } => Phase::AwaitingFirstSlice {
            meta,
            health: Some(health),
            status,
        },
        Phase::Optimizing {
            meta,
            latest,
            last_step,
            plan,
            last_error,
            ..
        } => Phase::Optimizing {
            meta,
            latest,
            last_step,
            plan,
            health: Some(health),
            last_error,
        },
    }
}

/// Records a fault into whichever phase is current, without changing the lifecycle.
fn with_error(phase: Phase, error: EffectError) -> Phase {
    match phase {
        Phase::AwaitingFirstSlice { meta, health, .. } => Phase::AwaitingFirstSlice {
            meta,
            health,
            status: AwaitStatus::Error(error),
        },
        Phase::Optimizing {
            meta,
            latest,
            last_step,
            plan,
            health,
            ..
        } => Phase::Optimizing {
            meta,
            latest,
            last_step,
            plan,
            health,
            last_error: Some(error),
        },
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use aa_wire::{
        PoolCompleteness, PoolMetaEntry, PoolQuery, PoolSlice, TokenMetaEntry, WirePoolState,
    };
    use optimization::OptimizationStepStatus;

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
        assert!(matches!(state.phase, Phase::Optimizing { .. }));
        state
    }

    /// `config()` retargeted to spend `source` and maximize `output` (both `addr`-bytes, this chain).
    fn route_config(source: u8, output: u8) -> SessionConfig {
        let mut config = config();
        config.optimization.source_asset = TokenAddress(addr(source), CHAIN);
        config.optimization.output_asset = TokenAddress(addr(output), CHAIN);
        config
    }

    #[test]
    fn set_route_retargets_the_config_without_effects() {
        // The runtime command seam: `SetRoute` retargets the route in place and emits nothing. From a
        // cold start there are no results to discard, so the phase is untouched too (contrast
        // `set_route_to_a_new_route_discards_the_previous_route_s_results`).
        let state = AppState::started(config());
        let phase_before = state.phase.clone();
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
        assert_eq!(next.phase, phase_before, "phase is untouched by SetRoute");
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
        assert!(matches!(state.phase, Phase::Optimizing { .. }));

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
        assert!(matches!(
            state.phase,
            Phase::AwaitingFirstSlice {
                status: AwaitStatus::NoRoute,
                ..
            }
        ));
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
            state.phase,
            Phase::Optimizing {
                last_step: Some(_),
                ..
            }
        ));

        let (state, effects) = transition(
            state,
            Event::SetRoute(Route {
                source: addr(1),
                output: addr(2),
            }),
        );

        match state.phase {
            Phase::AwaitingFirstSlice { meta, status, .. } => {
                assert!(meta.is_some(), "the catalog survives a retarget");
                assert_eq!(status, AwaitStatus::Bootstrapping);
            }
            other => panic!("a retarget must discard the old route's results, got {other:?}"),
        }
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

        assert!(matches!(state.phase, Phase::Optimizing { .. }));
        assert!(effects.is_empty());
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
        assert!(matches!(
            state.phase,
            Phase::AwaitingFirstSlice {
                status: AwaitStatus::NoRoute,
                ..
            }
        ));
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
        assert!(matches!(
            state.phase,
            Phase::AwaitingFirstSlice { meta: Some(_), .. }
        ));
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

        let Phase::Optimizing {
            latest, last_step, ..
        } = &state.phase
        else {
            panic!("expected Optimizing, got {:?}", state.phase);
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
        assert!(matches!(state.phase, Phase::Optimizing { .. }));
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
        let Phase::Optimizing { last_step, .. } = &state.phase else {
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
        assert!(matches!(
            state.phase,
            Phase::AwaitingFirstSlice {
                status: AwaitStatus::NoRoute,
                ..
            }
        ));
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
            state.phase,
            Phase::AwaitingFirstSlice {
                status: AwaitStatus::Error(EffectError::Adapter(WireAdapterError::HexParse { .. })),
                ..
            }
        ));
    }

    #[test]
    fn mismatched_slice_id_is_rejected_then_the_real_id_is_accepted() {
        let (state, slice) = after_catalog(config(), meta_over(1, 2));
        let phase_before = state.phase.clone();

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
        assert_eq!(state.phase, phase_before);

        // The real in-flight id is still accepted and drives the optimizer.
        let (state, effects) = transition(
            state,
            Event::SliceFetched {
                id: slice,
                response: slice_complete(),
            },
        );
        assert!(any_push_reserves(&effects));
        assert!(matches!(state.phase, Phase::Optimizing { .. }));
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
                error: EffectError::Fetch {
                    what: FetchKind::Meta,
                    message: "boom".to_owned(),
                },
            },
        );
        assert!(effects.is_empty());
        assert!(matches!(
            state.phase,
            Phase::AwaitingFirstSlice {
                status: AwaitStatus::Error(EffectError::Fetch { .. }),
                ..
            }
        ));
        // The next tick re-issues meta with a fresh id (still no catalog).
        let (_state, effects) = transition(state, Event::Tick);
        assert_ne!(meta_id(&effects), id);
    }

    #[test]
    fn stale_failure_is_rejected() {
        let (state, effects) = transition(AppState::started(config()), Event::Tick);
        let real = meta_id(&effects);
        let phase_before = state.phase.clone();
        // A failure carrying an id the ledger no longer holds is ignored — state is unchanged.
        let stale = FetchId::from_raw_for_test(9999);
        assert_ne!(real, stale);
        let (state, effects) = transition(
            state,
            Event::FetchFailed {
                id: stale,
                kind: FetchKind::Meta,
                error: EffectError::Fetch {
                    what: FetchKind::Meta,
                    message: "boom".to_owned(),
                },
            },
        );
        assert!(effects.is_empty());
        assert_eq!(state.phase, phase_before);
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
        assert!(matches!(state.phase, Phase::AwaitingFirstSlice { .. }));

        let (state, effects) = transition(
            state,
            Event::SliceFetched {
                id: current,
                response: slice_complete(),
            },
        );
        assert!(any_push_reserves(&effects));
        assert!(matches!(state.phase, Phase::Optimizing { .. }));
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
        let Phase::Optimizing { latest, .. } = &state.phase else {
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
        let Phase::Optimizing { latest, .. } = &state.phase else {
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
