//! The headless engine's pure core: an `AppState` and a `transition` reducer with the same
//! `event → state → (state, effects)` shape as the kernel. It owns *all* of the client's decision
//! logic — when to poll the data plane, when to (re)initialize the optimizer, when to keep grinding —
//! while holding **none** of the heavy machinery. In particular the optimizer's mutable tensor state
//! never lives here: `optimization::OptimizationRunner` is a move-based state machine that owns the
//! `Model`; the reducer only emits [`OptimizeCommand`]s to it (via [`Effect::Optimize`]) and folds its
//! results back in as [`Event::OptimizerStepped`]. Keeping the derived heavy thing out of the
//! reducible state is the same discipline the kernel uses for reserves and pool folds.
//!
//! Everything in this module is pure and synchronous: the reducer performs no I/O and spawns nothing.
//! A driver (a later increment) executes the returned [`Effect`]s on real threads/HTTP and feeds the
//! outcomes back as [`Event`]s. That split makes the entire engine testable by feeding event
//! sequences and asserting `(state, effects)`, with no transport in the loop.

use std::collections::HashSet;
use std::str::FromStr;

use aa_wire::{HealthResponse, PoolsMetaResponse, SliceRequest, SliceResponse};
use client_evm::{BlockHash, ChainKey, PoolRef, TokenAddress};
use optimization::{
    ExecutionPlan, OptimizationBackendSelection, OptimizationSessionConfig, OptimizationStepConfig,
    OptimizationStepResult, OptimizationStepUpdate, PoolReserves, reserves_reach_init_asset,
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
    /// The routing strategy the client optimizes for: init asset, bridge pairs, token whitelist. The
    /// server never serves strategy, so this is purely client-side.
    pub optimization: OptimizationSessionConfig<TokenAddress>,
    /// Per-step input amount and the iteration budget of a single `run` — the bounded unit of work
    /// that keeps the (future) optimizer worker responsive between reserve refreshes.
    pub step: OptimizationStepConfig,
    /// Which optimizer backend to initialize (wgpu/cpu).
    pub backend: OptimizationBackendSelection,
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
/// `AwaitingAnchor → Running`): the optimizer requires a non-empty reserve set that reaches the init
/// asset, so it is only initialized once such a slice arrives. `last_step`/`plan`/`latest` only exist
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
        /// Whether an optimizer command is in flight; backpressure so at most one is outstanding.
        awaiting_step: bool,
        /// The most recent recorded fault (adapter/fetch), cleared when a good slice is applied.
        last_error: Option<EffectError>,
    },
}

/// Why the engine is still in `AwaitingFirstSlice`. A view-facing status, not a control signal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AwaitStatus {
    /// Waiting for the initial catalog/slice to arrive.
    Bootstrapping,
    /// A slice arrived but its reserves cannot reach the configured init asset — nothing to arbitrage
    /// yet, so the optimizer is intentionally not initialized (it would abort on such a snapshot).
    NoInitAssetRoute,
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
    /// Drive the optimizer worker.
    Optimize(OptimizeCommand),
}

/// A command for the optimizer worker that owns the `OptimizationRunner`. `Init` bundles the config
/// the runner's `init` needs; `Run` reuses the optimizer's own `OptimizationStepUpdate` (a fresh
/// reserve snapshot, or `Continue` to grind more iterations on the current one).
#[derive(Clone, Debug)]
pub enum OptimizeCommand {
    /// Initialize the runner from the first productive reserve snapshot.
    Init {
        /// The projected reserves to seed the model with.
        reserves: Vec<PoolReserves<PoolRef, TokenAddress>>,
        /// The routing strategy (init asset / bridges / whitelist).
        session_config: OptimizationSessionConfig<TokenAddress>,
        /// The per-step input amount and iteration budget.
        step_config: OptimizationStepConfig,
        /// Which backend to initialize.
        backend: OptimizationBackendSelection,
    },
    /// Advance the already-initialized runner one step.
    Run(OptimizationStepUpdate<PoolRef, TokenAddress>),
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
        config,
        phase,
        pending,
    } = state;
    let (phase, pending, effects) = reduce(&config, phase, pending, event);
    (
        AppState {
            config,
            phase,
            pending,
        },
        effects,
    )
}

fn reduce(
    config: &SessionConfig,
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
                        status: AwaitStatus::NoInitAssetRoute,
                    },
                    vec![],
                );
            }
            let command = OptimizeCommand::Init {
                reserves,
                session_config: config.optimization.clone(),
                step_config: config.step,
                backend: config.backend,
            };
            (
                Phase::Optimizing {
                    meta,
                    latest: provenance,
                    last_step: None,
                    plan: None,
                    health,
                    awaiting_step: true,
                    last_error: None,
                },
                vec![Effect::Optimize(command)],
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
            awaiting_step,
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
                            awaiting_step,
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
                            awaiting_step,
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
                        awaiting_step,
                        last_error,
                    },
                    vec![],
                );
            }
            let command = OptimizeCommand::Run(OptimizationStepUpdate::NewReserves {
                reserves,
                disabled: HashSet::new(),
            });
            (
                Phase::Optimizing {
                    meta,
                    latest: provenance,
                    last_step,
                    plan,
                    health,
                    awaiting_step: true,
                    last_error: None,
                },
                vec![Effect::Optimize(command)],
            )
        }
    }
}

/// A step came back: record it and grind another bounded iteration chunk. A concurrently arriving
/// slice supersedes the `Continue` with a `NewReserves` on its own event.
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
                awaiting_step: true,
                last_error,
            },
            vec![Effect::Optimize(OptimizeCommand::Run(
                OptimizationStepUpdate::Continue,
            ))],
        ),
        // A step result while not optimizing is stale (e.g. after a reset); ignore it.
        other => (other, vec![]),
    }
}

/// Whether a projected reserve set can actually drive the optimizer: non-empty and reaching the
/// configured init asset (else `init`/`run` would abort with `EmptyReserves`/`InitAssetOutputNotFound`).
fn is_productive(reserves: &[PoolReserves<PoolRef, TokenAddress>], config: &SessionConfig) -> bool {
    !reserves.is_empty() && reserves_reach_init_asset(reserves, &config.optimization)
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
            awaiting_step,
            last_error,
            ..
        } => Phase::Optimizing {
            meta,
            latest,
            last_step,
            plan,
            health,
            awaiting_step,
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
            awaiting_step,
            last_error,
            ..
        } => Phase::Optimizing {
            meta,
            latest,
            last_step,
            plan,
            health: Some(health),
            awaiting_step,
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
            awaiting_step,
            ..
        } => Phase::Optimizing {
            meta,
            latest,
            last_step,
            plan,
            health,
            awaiting_step,
            last_error: Some(error),
        },
    }
}

#[cfg(test)]
mod tests {
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

    /// A session whose init asset is `addr(1)`.
    fn config() -> SessionConfig {
        SessionConfig {
            chain: CHAIN,
            optimization: OptimizationSessionConfig {
                init_asset: TokenAddress(addr(1), CHAIN),
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

    fn is_optimize_init(effect: &Effect) -> bool {
        matches!(effect, Effect::Optimize(OptimizeCommand::Init { .. }))
    }

    fn is_optimize_new_reserves(effect: &Effect) -> bool {
        matches!(
            effect,
            Effect::Optimize(OptimizeCommand::Run(
                OptimizationStepUpdate::NewReserves { .. }
            ))
        )
    }

    fn is_optimize_continue(effect: &Effect) -> bool {
        matches!(
            effect,
            Effect::Optimize(OptimizeCommand::Run(OptimizationStepUpdate::Continue))
        )
    }

    fn any_optimize(effects: &[Effect]) -> bool {
        effects.iter().any(|e| matches!(e, Effect::Optimize(_)))
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
        assert!(effects.iter().any(is_optimize_init));
        assert!(matches!(state.phase, Phase::Optimizing { .. }));
        state
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
    fn first_productive_slice_initializes_optimizer() {
        let (state, slice) = after_catalog(config(), meta_over(1, 2));
        let (state, effects) = transition(
            state,
            Event::SliceFetched {
                id: slice,
                response: slice_complete(),
            },
        );

        let reserves = effects
            .iter()
            .find_map(|e| match e {
                Effect::Optimize(OptimizeCommand::Init { reserves, .. }) => Some(reserves),
                _ => None,
            })
            .expect("init command");
        // The one v3 pool projects to forward + inverse.
        assert_eq!(reserves.len(), 2);

        let Phase::Optimizing {
            latest,
            awaiting_step,
            last_step,
            ..
        } = &state.phase
        else {
            panic!("expected Optimizing, got {:?}", state.phase);
        };
        assert_eq!(latest.confirmations, 2);
        assert_eq!(latest.block_hash, hash(0xbb));
        assert!(*awaiting_step);
        assert!(last_step.is_none());
    }

    #[test]
    fn subsequent_slice_runs_new_reserves_not_init() {
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
        assert!(effects.iter().any(is_optimize_new_reserves));
        assert!(!effects.iter().any(is_optimize_init));
        assert!(matches!(state.phase, Phase::Optimizing { .. }));
    }

    #[test]
    fn step_records_result_and_grinds_continue() {
        let state = optimizing_state();
        let (state, effects) = transition(
            state,
            Event::OptimizerStepped {
                result: step_result(),
                plan: None,
            },
        );
        assert!(effects.iter().any(is_optimize_continue));
        let Phase::Optimizing {
            last_step,
            awaiting_step,
            ..
        } = &state.phase
        else {
            panic!("expected Optimizing");
        };
        assert!(last_step.is_some());
        assert!(*awaiting_step);
    }

    #[test]
    fn unproductive_slice_stays_awaiting_without_optimizing() {
        // Catalog over tokens that do NOT include the init asset (`addr(1)`).
        let (state, slice) = after_catalog(config(), meta_over(3, 4));
        let (state, effects) = transition(
            state,
            Event::SliceFetched {
                id: slice,
                response: slice_complete(),
            },
        );
        assert!(!any_optimize(&effects));
        assert!(matches!(
            state.phase,
            Phase::AwaitingFirstSlice {
                status: AwaitStatus::NoInitAssetRoute,
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
        assert!(!any_optimize(&effects));
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

        // The real in-flight id is still accepted and initializes the optimizer.
        let (state, effects) = transition(
            state,
            Event::SliceFetched {
                id: slice,
                response: slice_complete(),
            },
        );
        assert!(effects.iter().any(is_optimize_init));
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
}
