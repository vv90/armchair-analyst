use std::{
    collections::{BTreeMap, HashMap},
    num::NonZeroUsize,
    time::Duration,
};

use alloy::primitives::{BlockHash, Bloom, U256};
use optimization::{
    ExecutionPlan, Invertible, OptimizationStepResult, PoolReserves, StepKind, VirtualReserveValues,
};
use thiserror::Error;

use crate::{
    LosslessPool, LosslessReplayError, PoolFee, PoolLog, PoolMetadata, PoolRef, PoolState,
    PoolStateError, TokenAddress, TokenAmountConversionError, TokenDecimals, bootstrap,
    chain::ChainKey, kernel, replay_plan_lossless, u256_token_amount_to_f32,
    utils::f32_token_amount_to_u256,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChainStatus {
    Initializing,
    Active,
}

/// Active-chain progress metrics surfaced to read models such as the CLI view.
/// Added so observers can show tracked-pool and fetch progress without inspecting kernel internals.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChainProgress {
    pub verified_pools: usize,
    /// Canonical blocks the tip is ahead of the chain's last dispatched optimization block.
    /// `None` while no optimization has been dispatched or the reference is off the current path.
    pub blocks_behind_tip: Option<usize>,
    /// Connected canonical path length from the finalized anchor to the tip — the window every
    /// optimization overlay recompute folds over. Growth past the chain's finalized-refresh
    /// target is the early signature of finalization not advancing (e.g. a wrong provisional
    /// finality constant), visible here before it shows up as CPU or memory. `None` while the
    /// tip is not yet connected to the anchor.
    pub canonical_window: Option<usize>,
    /// RPC requests currently in flight for the chain (dispatched, not yet answered): the per-chain
    /// fetch-backlog gauge.
    pub in_flight_requests: usize,
    /// Cumulative WS-miss count: how often an authoritative log fetch diverged from the streamed
    /// set it replaced. The trust gauge of the WS-primary flip — ~0 means the feeds have never
    /// been caught wrong.
    pub ws_misses: u64,
}

/// A render snapshot of one chain: lifecycle phase plus metrics that exist only while active.
/// Added as the single read model `observe` returns, keeping `ChainStatus` a pure lifecycle indicator.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChainObservation {
    /// Still bootstrapping. `buffered_events` counts the live subscription deliveries queued for
    /// replay at activation, so the size of the activation burst is visible while it accumulates.
    Initializing { buffered_events: usize },
    Active(ChainProgress),
}

pub struct State {
    chains: BTreeMap<ChainKey, ChainLifecycle>,
    latest_optimization_result: Option<OptimizationStepResult>,
    /// The lossless-replay verdict on the plan that arrived with `latest_optimization_result` —
    /// written only alongside it, so a stale result/verification pairing is unrepresentable.
    /// `None` when the latest step carried no plan.
    latest_plan_verification: Option<PlanVerification>,
    /// Latest fold-frontier block per chain for which optimization reserves were dispatched.
    /// Added so `RunOptimization` fires only when a chain's optimization fold frontier advances.
    last_optimized_block: BTreeMap<ChainKey, BlockHash>,
}

enum ChainLifecycle {
    /// A chain still fetching its anchor/discovery seed. The `Vec<SubscriptionData>` buffers the live
    /// heads/logs that arrive meanwhile so they can be replayed onto the seeded graph at activation,
    /// avoiding a per-block header walk to bridge `seed-tip..now`.
    Bootstrapping(bootstrap::State, Vec<SubscriptionData>),
    Active(kernel::State),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FinalizedRefreshPolicy {
    target_len: usize,
    /// Non-zero by type: the refresh predicate's bucket arithmetic divides by the stride, so a
    /// zero would be a build error here, not a runtime guard downstream.
    retry_stride: NonZeroUsize,
}

/// Compile-time-checked stride literal: a zero fails const evaluation (a build error), never the
/// runtime — every use below is a `const` item.
const fn nonzero_stride(value: usize) -> NonZeroUsize {
    match NonZeroUsize::new(value) {
        Some(stride) => stride,
        None => panic!("finalized-refresh retry stride must be non-zero"),
    }
}

const ETHEREUM_APPROX_FINALIZED_BLOCK_AGE: usize = 64;
const ETHEREUM_FINALIZED_RETENTION_MARGIN: usize = 8;
const ETHEREUM_FINALIZED_REFRESH_RETRY_STRIDE: NonZeroUsize = nonzero_stride(8);

/// Reorg-prone blocks nearest the observed tip left out of the seeded block graph.
const ETHEREUM_BOOTSTRAP_TIP_TRIM: usize = 4;
/// Ticks after which bootstrap activates best-effort (or abandons before the anchor is known).
const ETHEREUM_BOOTSTRAP_DEADLINE_TICKS: u64 = 180;

// PROVISIONAL — tune before activation. Arbitrum One produces ~one block per ~0.25s and inherits L1
// finality, so its block-denominated retention/look-back windows are larger than Ethereum's. These
// values do not affect runtime while Arbitrum is absent from `ACTIVE_CHAINS`; they are starting points
// for the activation chunk.
const ARBITRUM_APPROX_FINALIZED_BLOCK_AGE: usize = 1_000;
const ARBITRUM_FINALIZED_RETENTION_MARGIN: usize = 64;
const ARBITRUM_FINALIZED_REFRESH_RETRY_STRIDE: NonZeroUsize = nonzero_stride(32);

/// Reorg-prone blocks nearest the observed tip left out of the seeded block graph.
const ARBITRUM_BOOTSTRAP_TIP_TRIM: usize = 8;
/// Ticks after which bootstrap activates best-effort (or abandons before the anchor is known).
const ARBITRUM_BOOTSTRAP_DEADLINE_TICKS: u64 = 180;

// PROVISIONAL — tune before relying on finalized pruning. Base/Optimism are OP-stack rollups (~2s
// blocks): the unsafe sequencer head can reorg shallowly while true finality follows L1, so the
// block-denominated window is moderate. Values anchored to block time × an Ethereum-comparable reorg
// depth; revisit against observed reorg depth.
const BASE_APPROX_FINALIZED_BLOCK_AGE: usize = 200;
const BASE_FINALIZED_RETENTION_MARGIN: usize = 32;
const BASE_FINALIZED_REFRESH_RETRY_STRIDE: NonZeroUsize = nonzero_stride(16);
const BASE_BOOTSTRAP_TIP_TRIM: usize = 8;
const BASE_BOOTSTRAP_DEADLINE_TICKS: u64 = 180;

const OPTIMISM_APPROX_FINALIZED_BLOCK_AGE: usize = 200;
const OPTIMISM_FINALIZED_RETENTION_MARGIN: usize = 32;
const OPTIMISM_FINALIZED_REFRESH_RETRY_STRIDE: NonZeroUsize = nonzero_stride(16);
const OPTIMISM_BOOTSTRAP_TIP_TRIM: usize = 8;
const OPTIMISM_BOOTSTRAP_DEADLINE_TICKS: u64 = 180;

// PROVISIONAL — Polygon PoS (~2s blocks) has historically the deepest probabilistic reorgs of this
// set, so its retention/look-back windows are the largest.
const POLYGON_APPROX_FINALIZED_BLOCK_AGE: usize = 400;
const POLYGON_FINALIZED_RETENTION_MARGIN: usize = 64;
const POLYGON_FINALIZED_REFRESH_RETRY_STRIDE: NonZeroUsize = nonzero_stride(32);
const POLYGON_BOOTSTRAP_TIP_TRIM: usize = 16;
const POLYGON_BOOTSTRAP_DEADLINE_TICKS: u64 = 180;

// PROVISIONAL — BNB Chain (~3s blocks) has fast finality with occasional short reorgs.
const BNB_APPROX_FINALIZED_BLOCK_AGE: usize = 150;
const BNB_FINALIZED_RETENTION_MARGIN: usize = 32;
const BNB_FINALIZED_REFRESH_RETRY_STRIDE: NonZeroUsize = nonzero_stride(16);
const BNB_BOOTSTRAP_TIP_TRIM: usize = 12;
const BNB_BOOTSTRAP_DEADLINE_TICKS: u64 = 180;

// PROVISIONAL — Avalanche C-Chain (~2s blocks) has near-instant finality, so the smallest window.
const AVALANCHE_APPROX_FINALIZED_BLOCK_AGE: usize = 80;
const AVALANCHE_FINALIZED_RETENTION_MARGIN: usize = 16;
const AVALANCHE_FINALIZED_REFRESH_RETRY_STRIDE: NonZeroUsize = nonzero_stride(8);
const AVALANCHE_BOOTSTRAP_TIP_TRIM: usize = 6;
const AVALANCHE_BOOTSTRAP_DEADLINE_TICKS: u64 = 180;

/// One chain's projected reserves at the block they were derived from. The per-chain piece that
/// `pool_reserves_for_optimization` yields; the merge concatenates these into one optimizer envelope.
#[derive(Clone, Debug, PartialEq)]
pub struct ChainPoolReserves {
    pub block_hash: BlockHash,
    pub reserves: Vec<PoolReserves<PoolRef, TokenAddress>>,
}

/// The merged optimizer input spanning every active chain. Reserves stay a flat `Vec` because chain
/// identity already rides inside every `PoolRef`/`TokenAddress`; `block_hashes` records the block
/// each contributing chain was projected at (one entry per chain in the merge).
#[derive(Clone, Debug, PartialEq)]
pub struct OptimizationPoolReserves {
    pub block_hashes: BTreeMap<ChainKey, BlockHash>,
    pub reserves: Vec<PoolReserves<PoolRef, TokenAddress>>,
}

/// The lossless replay's verdict on the plan emitted with an optimization step, computed against the
/// freshest kernel state at the moment the step completed — the optimizer's claimed profit judged
/// against the market *now*, not against its own input snapshot.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PlanVerification {
    /// The lossless replay ran: `profit` = output − entry in decimal-normalized init-asset units
    /// (the same scale as the claimed `profit_amount`); `hit_tick_limit` marks a hop clamped at its
    /// tick boundary, making `profit` a conservative lower-fidelity bound rather than exact.
    Verified { profit: f32, hit_tick_limit: bool },
    Unverifiable(PlanVerificationFailure),
}

/// Why a plan could not be judged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanVerificationFailure {
    /// The init asset's chain is not active or its decimals are not verified.
    InitAssetUnknown,
    /// The f32 entry (or the output back-projection) has no representation.
    AmountConversion,
    /// The replay failed: a pool without resolvable lossless state, or a malformed plan step.
    Replay(LosslessReplayError),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoolReserveValueKind {
    Reserve0,
    Reserve1,
    MaxSwap0,
    MaxSwap1,
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum PoolReserveProjectionError {
    #[error("failed to calculate max swap 0 for pool {pool:?}: {source}")]
    SwapLimit0 {
        pool: PoolRef,
        source: PoolStateError,
    },

    #[error("failed to calculate max swap 1 for pool {pool:?}: {source}")]
    SwapLimit1 {
        pool: PoolRef,
        source: PoolStateError,
    },

    #[error("failed to convert {value:?} for pool {pool:?} token {token:?}: {source}")]
    AmountConversion {
        pool: PoolRef,
        token: TokenAddress,
        value: PoolReserveValueKind,
        source: TokenAmountConversionError,
    },
}

impl State {
    /// Creates multi-chain state with every requested chain bootstrapping its recent canonical window.
    /// Added so runtimes seed an inner kernel per active chain from each bootstrap outcome instead of
    /// an empty finalized state. The chain set is a parameter (production passes `ACTIVE_CHAINS`,
    /// tests pass a singleton) so the active count lives in one constant, not in this constructor.
    pub fn init(chains: &[ChainKey]) -> (State, Vec<Effect>) {
        let mut chain_map = BTreeMap::new();
        let mut effects = Vec::new();

        for &chain in chains {
            let (bootstrap_state, bootstrap_effects) = bootstrap::init(bootstrap_policy(chain));
            chain_map.insert(
                chain,
                ChainLifecycle::Bootstrapping(bootstrap_state, Vec::new()),
            );
            effects.extend(wrap_bootstrap_effects(chain, bootstrap_effects));
        }

        (
            State {
                chains: chain_map,
                latest_optimization_result: None,
                latest_plan_verification: None,
                last_optimized_block: BTreeMap::new(),
            },
            effects,
        )
    }

    /// Reports the most recent optimization step result reported by the optimization worker.
    /// Added so read models can surface optimization progress without inspecting the worker thread.
    pub fn latest_optimization_result(&self) -> Option<OptimizationStepResult> {
        self.latest_optimization_result
    }

    /// Reports the lossless-replay verdict on the latest optimization result's emitted plan —
    /// `None` when that step carried no plan. Added so read models can put the verified profit
    /// next to the optimizer's claimed one.
    pub fn latest_plan_verification(&self) -> Option<PlanVerification> {
        self.latest_plan_verification
    }

    /// Reports whether a configured chain is initializing (bootstrapping) or active.
    /// Added so callers can observe wrapper readiness without exposing the inner lifecycle representation.
    pub fn status(&self, chain: ChainKey) -> Option<ChainStatus> {
        self.chains
            .get(&chain)
            .map(|chain_state| match chain_state {
                ChainLifecycle::Bootstrapping(..) => ChainStatus::Initializing,
                ChainLifecycle::Active(_) => ChainStatus::Active,
            })
    }

    /// Builds a pure render snapshot for every configured chain, in chain order.
    /// Added as the single observation call read models make, so the view never chains accessors
    /// or derives progress for a non-active chain.
    pub fn observe(&self) -> Vec<(ChainKey, ChainObservation)> {
        self.chains
            .iter()
            .map(|(chain, chain_state)| {
                (
                    *chain,
                    observe_chain(chain_state, self.last_optimized_block.get(chain).copied()),
                )
            })
            .collect()
    }
}

/// Projects one chain lifecycle into its render observation.
/// Added so active-chain metrics are gathered in one place and stay unreachable while bootstrapping.
/// `last_optimized_block` is the chain's last dispatched optimization block, used as the cheap
/// reference for fetch progress; absent (not yet dispatched, or reorg-orphaned) renders as unknown.
fn observe_chain(
    chain_state: &ChainLifecycle,
    last_optimized_block: Option<BlockHash>,
) -> ChainObservation {
    match chain_state {
        ChainLifecycle::Bootstrapping(_, buffered) => ChainObservation::Initializing {
            buffered_events: buffered.len(),
        },
        ChainLifecycle::Active(chain_state) => ChainObservation::Active(ChainProgress {
            verified_pools: chain_state.verified_pool_count(),
            blocks_behind_tip: last_optimized_block
                .and_then(|reference| chain_state.blocks_behind(reference)),
            canonical_window: chain_state.canonical_path_len_from_finalized(),
            in_flight_requests: chain_state.in_flight_request_count(),
            ws_misses: chain_state.ws_miss_count(),
        }),
    }
}

/// Projects the requested chain's active kernel state and its optimization overlay into optimization reserves.
/// Added as the pure bridge from validated EVM pool state into the optimization crate's directional reserve model.
pub fn pool_reserves_for_optimization(
    state: &State,
    chain: ChainKey,
    update: &kernel::OptimizationStateUpdate,
) -> Result<Option<ChainPoolReserves>, PoolReserveProjectionError> {
    let Some(ChainLifecycle::Active(chain_state)) = state.chains.get(&chain) else {
        return Ok(None);
    };

    // The overlay carries owned folded states (no locations to resolve — the fold cannot
    // dangle); merge it over the finalized base for the full per-pool view.
    let overlay = update
        .pool_states
        .iter()
        .map(|(pool, pool_state)| (*pool, pool_state))
        .collect();

    let mut reserves = Vec::new();

    for (pool, pool_state) in
        sorted_pool_states_for_projection(chain_state.finalized_pool_snapshots(), overlay)
    {
        let Some((token0, token1, fee, token0_decimals, token1_decimals)) =
            projection_metadata(chain_state, chain, pool)
        else {
            return Ok(None);
        };

        let value = pool_reserve_values(
            pool,
            pool_state,
            fee,
            token0,
            token1,
            token0_decimals,
            token1_decimals,
        )?;

        let reserve = PoolReserves {
            token0,
            token1,
            pool_id: pool,
            value,
        };

        reserves.extend([reserve, reserve.inverse()]);
    }

    Ok(Some(ChainPoolReserves {
        block_hash: update.block_hash,
        reserves,
    }))
}

/// Rebuilds the merged optimizer input from every active chain's current kernel state. Pure over
/// `&State`: each call re-projects from the authoritative per-chain states rather than reading any
/// cached reserves, so the merge can never serve stale or drifted data. Chains that are still
/// bootstrapping or fail projection simply contribute nothing this round.
///
/// The one exception to re-projection: `precomputed` is the triggering chain's projection the
/// caller just computed from this same `state` (the dispatch gate needed it anyway), spliced in
/// verbatim so that chain's fold + projection is not immediately recomputed — a pure substitution
/// of an identical recomputation, never a cache.
fn merged_optimization_reserves(
    state: &State,
    precomputed_chain: ChainKey,
    precomputed: &ChainPoolReserves,
) -> OptimizationPoolReserves {
    let mut block_hashes = BTreeMap::new();
    let mut reserves = Vec::new();

    for (&chain, lifecycle) in &state.chains {
        let ChainLifecycle::Active(chain_state) = lifecycle else {
            continue;
        };
        if chain == precomputed_chain {
            block_hashes.insert(chain, precomputed.block_hash);
            reserves.extend(precomputed.reserves.iter().copied());
            continue;
        }
        let update = chain_state.optimization_update(chain);
        if let Ok(Some(chain_reserves)) = pool_reserves_for_optimization(state, chain, &update) {
            block_hashes.insert(chain, chain_reserves.block_hash);
            reserves.extend(chain_reserves.reserves);
        }
    }

    OptimizationPoolReserves {
        block_hashes,
        reserves,
    }
}

/// Merges finalized snapshots with update snapshots and returns pools in deterministic order.
/// Added so projections include the latest known state per pool while keeping output stable for model layout and tests.
fn sorted_pool_states_for_projection<'a>(
    finalized_pool_states: &'a HashMap<PoolRef, PoolState>,
    update_pool_states: HashMap<PoolRef, &'a PoolState>,
) -> Vec<(PoolRef, &'a PoolState)> {
    let mut pool_states = finalized_pool_states
        .iter()
        .map(|(pool, pool_state)| (*pool, pool_state))
        .collect::<HashMap<_, _>>();

    pool_states.extend(update_pool_states);

    let mut pool_states = pool_states.into_iter().collect::<Vec<_>>();
    pool_states.sort_by_key(|(pool, _)| *pool);
    pool_states
}

/// Collects verified pool metadata and token decimals needed for one pool projection.
/// Added so reserve generation can pause on incomplete registry data instead of emitting partially validated reserves.
fn projection_metadata(
    chain_state: &kernel::State,
    chain: ChainKey,
    pool: PoolRef,
) -> Option<(
    TokenAddress,
    TokenAddress,
    PoolFee,
    TokenDecimals,
    TokenDecimals,
)> {
    let PoolMetadata {
        token0,
        token1,
        fee,
    } = chain_state.verified_pool_metadata(pool)?;
    let token0 = TokenAddress(*token0, chain);
    let token1 = TokenAddress(*token1, chain);
    let token0_decimals = chain_state.verified_token_metadata(token0)?.decimals;
    let token1_decimals = chain_state.verified_token_metadata(token1)?.decimals;

    Some((token0, token1, *fee, token0_decimals, token1_decimals))
}

/// Converts one pool state into scaled virtual reserves, fee multiplier, and swap caps.
/// Added to keep Uniswap math, token scaling, and projection error context in one isolated pure step.
fn pool_reserve_values(
    pool: PoolRef,
    pool_state: &PoolState,
    fee: PoolFee,
    token0: TokenAddress,
    token1: TokenAddress,
    token0_decimals: TokenDecimals,
    token1_decimals: TokenDecimals,
) -> Result<VirtualReserveValues, PoolReserveProjectionError> {
    let reserve_0 = convert_pool_amount(
        pool,
        token0,
        PoolReserveValueKind::Reserve0,
        pool_state.virtual_reserve_x(),
        token0_decimals,
    )?;
    let reserve_1 = convert_pool_amount(
        pool,
        token1,
        PoolReserveValueKind::Reserve1,
        pool_state.virtual_reserve_y(),
        token1_decimals,
    )?;
    let tick_spacing = fee.tick_spacing();
    let max_swap_0 = pool_state
        .swap_limit_x(tick_spacing)
        .map_err(|source| PoolReserveProjectionError::SwapLimit0 { pool, source })
        .and_then(|amount| {
            convert_pool_amount(
                pool,
                token0,
                PoolReserveValueKind::MaxSwap0,
                amount,
                token0_decimals,
            )
        })?;
    let max_swap_1 = pool_state
        .swap_limit_y(tick_spacing)
        .map_err(|source| PoolReserveProjectionError::SwapLimit1 { pool, source })
        .and_then(|amount| {
            convert_pool_amount(
                pool,
                token1,
                PoolReserveValueKind::MaxSwap1,
                amount,
                token1_decimals,
            )
        })?;

    Ok(VirtualReserveValues {
        token_0: reserve_0,
        token_1: reserve_1,
        fee_multiplier: 1.0 - fee.pips() as f32 / 1_000_000.0,
        max_swap_0,
        max_swap_1,
    })
}

/// Scales a raw on-chain token amount into the optimizer's `f32` reserve representation.
/// Added so conversion failures carry the pool, token, and reserve field being projected.
fn convert_pool_amount(
    pool: PoolRef,
    token: TokenAddress,
    value: PoolReserveValueKind,
    amount: U256,
    decimals: TokenDecimals,
) -> Result<f32, PoolReserveProjectionError> {
    u256_token_amount_to_f32(amount, decimals).map_err(|source| {
        PoolReserveProjectionError::AmountConversion {
            pool,
            token,
            value,
            source,
        }
    })
}

/// Live subscription payload — exactly the two cases a `newHeads`/`logs` subscription can deliver.
/// Buffered verbatim while a chain bootstraps, then mapped to a `kernel::Event` on activation.
#[derive(Clone, Debug, PartialEq)]
pub enum SubscriptionData {
    NewHead {
        hash: BlockHash,
        parent_hash: BlockHash,
        logs_bloom: Bloom,
        number: u64,
    },
    PoolLog {
        block_hash: BlockHash,
        /// A consolidated, `log_index`-ordered batch for one block — the adapter's debounce window
        /// dedups a burst (and repeats across providers) into a single delivery.
        logs: Vec<PoolLog>,
    },
}

impl SubscriptionData {
    /// Maps the buffered subscription payload into the kernel event it drives.
    fn into_kernel_event(self) -> kernel::Event {
        match self {
            SubscriptionData::NewHead {
                hash,
                parent_hash,
                logs_bloom,
                number,
            } => kernel::Event::HeadObserved {
                hash,
                parent_hash,
                logs_bloom,
                number,
            },
            SubscriptionData::PoolLog { block_hash, logs } => {
                kernel::Event::LogObserved { block_hash, logs }
            }
        }
    }
}

pub enum Event {
    FinalizedHeaderReceived {
        chain: ChainKey,
        block_hash: BlockHash,
    },
    FinalizedHeaderUnavailable {
        chain: ChainKey,
    },
    /// Live subscription data (heads/logs). Carried as the narrow [`SubscriptionData`] rather than a
    /// `kernel::Event` so it can be buffered while a chain bootstraps without making RPC-response
    /// variants (which never originate from a subscription) representable in that buffer.
    SubscriptionData {
        chain: ChainKey,
        data: SubscriptionData,
    },
    /// A `kernel::Event` produced by executing an RPC effect for an already-active chain.
    ChainEvent {
        chain: ChainKey,
        event: kernel::Event,
    },
    BootstrapEvent {
        chain: ChainKey,
        event: bootstrap::Event,
    },
    OptimizationStepCompleted {
        result: OptimizationStepResult,
        /// The executable plan recovered from the step's trained weights; `None` when flow
        /// extraction degraded. Verified losslessly against the state at event-processing time.
        plan: Option<ExecutionPlan<PoolRef, TokenAddress>>,
    },
    Tick,
}

pub enum Subscription {
    NewHeadsSubscription(ChainKey),
    PoolEventsSubscription(ChainKey),
    TickSubscription(Duration),
    OptimizationSubscription,
}

pub enum Effect {
    FetchFinalizedHeader {
        chain: ChainKey,
    },
    ChainEffect {
        chain: ChainKey,
        effect: kernel::Effect,
    },
    BootstrapEffect {
        chain: ChainKey,
        effect: bootstrap::Effect,
    },
    RunOptimization {
        input: OptimizationPoolReserves,
    },
}

/// Dispatches one top-level multi-chain event through the pure state machine.
/// Added as the single runtime entry point that keeps chain lifecycle handling and effect wrapping centralized.
pub fn transition(state: State, event: Event) -> (State, Vec<Effect>) {
    match event {
        Event::FinalizedHeaderReceived { chain, block_hash } => {
            finalized_header_received(state, chain, block_hash)
        }
        Event::FinalizedHeaderUnavailable { chain } => finalized_header_unavailable(state, chain),
        Event::SubscriptionData { chain, data } => subscription_data(state, chain, data),
        Event::ChainEvent { chain, event } => chain_event(state, chain, event),
        Event::BootstrapEvent { chain, event } => bootstrap_event(state, chain, event),
        Event::OptimizationStepCompleted { result, plan } => {
            optimization_step_completed(state, result, plan)
        }
        Event::Tick => tick(state),
    }
}

/// Records the latest optimization step result reported by the optimization worker, and — when the
/// step carried a plan — its lossless-replay verdict against the current state, in lockstep.
/// It never produces effects and leaves the chain map untouched.
fn optimization_step_completed(
    state: State,
    result: OptimizationStepResult,
    plan: Option<ExecutionPlan<PoolRef, TokenAddress>>,
) -> (State, Vec<Effect>) {
    let verification = plan.map(|plan| verify_plan(&state, &plan));
    (
        State {
            latest_optimization_result: Some(result),
            latest_plan_verification: verification,
            ..state
        },
        Vec::new(),
    )
}

/// Judges an emitted plan by lossless replay against the current kernel state: the plan's own
/// advisory entry (so claimed and verified profit are compared at the same size) converted to raw
/// units, swapped through [`LosslessPool`]s resolved from the freshest per-chain fold, and the
/// terminal output projected back to the claimed profit's decimal-normalized scale.
fn verify_plan(state: &State, plan: &ExecutionPlan<PoolRef, TokenAddress>) -> PlanVerification {
    let TokenAddress(_, init_chain) = plan.init_asset;
    let Some(ChainLifecycle::Active(init_chain_state)) = state.chains.get(&init_chain) else {
        return PlanVerification::Unverifiable(PlanVerificationFailure::InitAssetUnknown);
    };
    let Some(decimals) = init_chain_state
        .verified_token_metadata(plan.init_asset)
        .map(|metadata| metadata.decimals)
    else {
        return PlanVerification::Unverifiable(PlanVerificationFailure::InitAssetUnknown);
    };

    let Ok(entry) = f32_token_amount_to_u256(plan.entry_amount, decimals) else {
        return PlanVerification::Unverifiable(PlanVerificationFailure::AmountConversion);
    };

    let book = lossless_pools_for_plan(state, plan);
    let outcome = match replay_plan_lossless(plan, |pool| book.get(&pool).cloned(), entry) {
        Ok(outcome) => outcome,
        Err(error) => {
            return PlanVerification::Unverifiable(PlanVerificationFailure::Replay(error));
        }
    };

    // Back-project the exact output onto the claimed profit's scale. Subtracting the advisory
    // entry rather than re-projecting the converted one skews by at most the ≤1-raw-unit
    // conversion floor — immaterial at gate precision.
    let Ok(output) = u256_token_amount_to_f32(outcome.output, decimals) else {
        return PlanVerification::Unverifiable(PlanVerificationFailure::AmountConversion);
    };
    PlanVerification::Verified {
        profit: output - plan.entry_amount,
        hit_tick_limit: outcome.hit_tick_limit,
    }
}

/// Resolves lossless pool states for every distinct pool the plan swaps through — the
/// `pool_reserves_for_optimization` lookup minus the f32 projection, batched so each involved
/// chain's optimization fold runs once (never per pool). An unresolvable pool (inactive chain,
/// unseeded state, unverified metadata, inconsistent tick state) is simply absent and surfaces as
/// [`LosslessReplayError::PoolNotFound`] at replay.
fn lossless_pools_for_plan(
    state: &State,
    plan: &ExecutionPlan<PoolRef, TokenAddress>,
) -> HashMap<PoolRef, LosslessPool> {
    let mut pools_by_chain: BTreeMap<ChainKey, Vec<PoolRef>> = BTreeMap::new();
    for step in &plan.steps {
        if let StepKind::Swap(pool) = step.kind {
            pools_by_chain.entry(pool.chain()).or_default().push(pool);
        }
    }

    let mut book = HashMap::new();
    for (chain, pools) in pools_by_chain {
        let Some(ChainLifecycle::Active(chain_state)) = state.chains.get(&chain) else {
            continue;
        };
        let overlay = chain_state.optimization_update(chain).pool_states;
        for pool in pools {
            let Some(pool_state) = overlay
                .get(&pool)
                .or_else(|| chain_state.finalized_pool_snapshots().get(&pool))
            else {
                continue;
            };
            let Some(metadata) = chain_state.verified_pool_metadata(pool) else {
                continue;
            };
            if let Ok(entry) = LosslessPool::from_pool_state(pool_state, metadata, chain) {
                book.insert(pool, entry);
            }
        }
    }
    book
}

/// Runs `f` on `chain`'s lifecycle entry (removed for ownership; reinserted unless `f` returns
/// `None`, which drops the chain), leaving every other `State` field untouched. The per-chain
/// handlers that neither read nor write the optimization fields all route through here, so the
/// remove/reinsert/rebuild dance has a single owner. An absent chain is a no-op.
fn with_chain_lifecycle(
    mut state: State,
    chain: ChainKey,
    f: impl FnOnce(ChainLifecycle) -> (Option<ChainLifecycle>, Vec<Effect>),
) -> (State, Vec<Effect>) {
    match state.chains.remove(&chain) {
        Some(lifecycle) => {
            let (lifecycle, effects) = f(lifecycle);
            if let Some(lifecycle) = lifecycle {
                state.chains.insert(chain, lifecycle);
            }
            (state, effects)
        }
        None => (state, Vec::new()),
    }
}

/// Advances an active chain's finalized boundary from a refreshed finalized header.
/// Added so the finalized-header refresh feed can drive each inner kernel's compaction; bootstrapping
/// chains fetch their own anchor and ignore this feed.
fn finalized_header_received(
    state: State,
    chain: ChainKey,
    block_hash: BlockHash,
) -> (State, Vec<Effect>) {
    with_chain_lifecycle(state, chain, |lifecycle| match lifecycle {
        ChainLifecycle::Active(chain_state) => {
            let (chain_state, effects) = kernel::transition(
                chain,
                chain_state,
                kernel::Event::FinalizedBlockObserved { block_hash },
            );
            (
                Some(ChainLifecycle::Active(chain_state)),
                effects
                    .into_iter()
                    .map(|effect| Effect::ChainEffect { chain, effect })
                    .collect(),
            )
        }
        other => (Some(other), Vec::new()),
    })
}

/// Ignores an unavailable finalized-header refresh.
/// Added so a failed active-chain finalized refresh is a no-op; the refresh is retried later and a
/// running chain is never torn down by a transient header fetch failure.
fn finalized_header_unavailable(state: State, _chain: ChainKey) -> (State, Vec<Effect>) {
    (state, Vec::new())
}

/// Forwards a bootstrap event to a bootstrapping chain and reflects the resulting lifecycle.
/// Added so bootstrap responses advance the phase machine, activate the inner kernel on completion,
/// or drop the chain when the anchor can never be obtained.
fn bootstrap_event(state: State, chain: ChainKey, event: bootstrap::Event) -> (State, Vec<Effect>) {
    with_chain_lifecycle(state, chain, |lifecycle| match lifecycle {
        ChainLifecycle::Bootstrapping(bootstrap_state, buffered) => {
            advance_bootstrap(chain, bootstrap_state, buffered, event)
        }
        other => (Some(other), Vec::new()),
    })
}

/// Runs one bootstrap transition and maps its completion onto the chain lifecycle.
/// Added as the single place that turns a bootstrap outcome into an active seeded kernel (or drops
/// the chain), keeping both the event and tick paths consistent.
fn advance_bootstrap(
    chain: ChainKey,
    bootstrap_state: bootstrap::State,
    buffered: Vec<SubscriptionData>,
    event: bootstrap::Event,
) -> (Option<ChainLifecycle>, Vec<Effect>) {
    let (bootstrap_state, effects) = bootstrap::transition(chain, bootstrap_state, event);
    let mut effects = wrap_bootstrap_effects(chain, effects);

    match bootstrap::completion(&bootstrap_state) {
        Some(bootstrap::Completion::Ready(outcome)) => {
            let (chain_state, activation_effects) = activate_bootstrap_outcome(outcome);
            effects.extend(
                activation_effects
                    .into_iter()
                    .map(|effect| Effect::ChainEffect { chain, effect }),
            );
            // Replay the live subscription data buffered during bootstrap onto the seeded kernel, so
            // the graph reaches the current tip from the events we already received instead of a
            // per-block header walk. Replay drives the inner kernel directly (no optimization
            // dispatch); the next live event re-derives the overlay.
            let (chain_state, replay_effects) =
                replay_buffered_subscription_data(chain, chain_state, buffered);
            effects.extend(replay_effects);
            (Some(ChainLifecycle::Active(chain_state)), effects)
        }
        Some(bootstrap::Completion::Abandoned) => (None, effects),
        None => (
            Some(ChainLifecycle::Bootstrapping(bootstrap_state, buffered)),
            effects,
        ),
    }
}

/// Folds the buffered subscription data through the freshly-activated kernel, threading state so each
/// scheduler dedups against the growing pending set. Wraps the resulting inner effects with the chain.
fn replay_buffered_subscription_data(
    chain: ChainKey,
    chain_state: kernel::State,
    buffered: Vec<SubscriptionData>,
) -> (kernel::State, Vec<Effect>) {
    buffered.into_iter().fold(
        (chain_state, Vec::new()),
        |(chain_state, mut effects), data| {
            let (chain_state, kernel_effects) =
                kernel::transition(chain, chain_state, data.into_kernel_event());
            effects.extend(
                kernel_effects
                    .into_iter()
                    .map(|effect| Effect::ChainEffect { chain, effect }),
            );
            (chain_state, effects)
        },
    )
}

/// Seeds an active kernel state from a completed bootstrap outcome.
/// Added as the pure Stage-2 bridge from the bootstrap outcome into the kernel's warm-start constructor.
fn activate_bootstrap_outcome(
    outcome: bootstrap::BootstrapOutcome,
) -> (kernel::State, Vec<kernel::Effect>) {
    let bootstrap::BootstrapOutcome {
        anchor,
        pool_snapshots,
        pool_registry,
        token_registry,
        seed_blocks,
    } = outcome;

    let seed_blocks = seed_blocks
        .into_iter()
        .map(|block| (block.hash, block.parent_hash, block.number, block.logs))
        .collect();

    kernel::State::activate_from_seed(
        anchor.hash,
        pool_snapshots,
        pool_registry,
        token_registry,
        seed_blocks,
    )
}

/// Wraps inner bootstrap effects with their owning chain key.
/// Added to keep bootstrap effect routing identical across init, event, and tick paths.
fn wrap_bootstrap_effects(chain: ChainKey, effects: Vec<bootstrap::Effect>) -> Vec<Effect> {
    effects
        .into_iter()
        .map(|effect| Effect::BootstrapEffect { chain, effect })
        .collect()
}

/// Routes live subscription data: a bootstrapping chain buffers it for replay at activation; an active
/// chain maps it to a `kernel::Event` and processes it immediately through the normal kernel path.
fn subscription_data(
    state: State,
    chain: ChainKey,
    data: SubscriptionData,
) -> (State, Vec<Effect>) {
    match state.chains.get(&chain) {
        Some(ChainLifecycle::Bootstrapping(..)) => buffer_subscription_data(state, chain, data),
        _ => chain_event(state, chain, data.into_kernel_event()),
    }
}

/// Appends subscription data to a bootstrapping chain's replay buffer; a no-op for any other state.
fn buffer_subscription_data(
    state: State,
    chain: ChainKey,
    data: SubscriptionData,
) -> (State, Vec<Effect>) {
    with_chain_lifecycle(state, chain, |lifecycle| match lifecycle {
        ChainLifecycle::Bootstrapping(bootstrap_state, mut buffered) => {
            buffered.push(data);
            (
                Some(ChainLifecycle::Bootstrapping(bootstrap_state, buffered)),
                Vec::new(),
            )
        }
        other => (Some(other), Vec::new()),
    })
}

/// Forwards an inner kernel event to an active chain and wraps its effects with the chain key.
/// Added to preserve chain isolation while letting callers drive per-chain kernel events through one wrapper.
fn chain_event(mut state: State, chain: ChainKey, event: kernel::Event) -> (State, Vec<Effect>) {
    // Not `with_chain_lifecycle`: this is the one handler that reads and writes the optimization
    // fields (`last_optimized_block`, the merged dispatch) around the reinserted chain.
    match state.chains.remove(&chain) {
        Some(ChainLifecycle::Active(chain_state)) => {
            // Measured before the transition (the state moves into it); on an inert outcome the
            // walk goes unused — the acceptable residual, negligible next to the skipped fold.
            let before_len = chain_state.canonical_path_len_from_finalized();
            // An inert transition returned state bit-identical, so every re-derivation below
            // (finalized-refresh probe, optimization overlay fold, dispatch gate) would
            // reproduce the previous event's results exactly: the path length cannot have
            // crossed a refresh bucket, and the fold frontier — hence the `changed` gate and
            // the (deterministic, this-chain-only) projection — is unchanged. Skip it all.
            let (chain_state, effects) = match kernel::transition_outcome(chain, chain_state, event)
            {
                kernel::TransitionOutcome::Inert(chain_state) => {
                    state.chains.insert(chain, ChainLifecycle::Active(chain_state));
                    return (state, Vec::new());
                }
                kernel::TransitionOutcome::Progressed(chain_state, effects) => {
                    (chain_state, effects)
                }
            };
            let after_len = chain_state.canonical_path_len_from_finalized();
            let refresh_policy = finalized_refresh_policy(chain);
            let should_fetch_finalized = kernel::should_fetch_finalized_header(
                before_len,
                after_len,
                refresh_policy.target_len,
                refresh_policy.retry_stride,
            );
            let mut effects = effects
                .into_iter()
                .map(|effect| Effect::ChainEffect { chain, effect })
                .collect::<Vec<_>>();

            if should_fetch_finalized {
                effects.push(Effect::FetchFinalizedHeader { chain });
            }

            // Gate the optimization fold on its frontier: the fold's result is consumed only
            // when the frontier advanced past the last dispatched block, so on every other
            // event — most log deliveries and scheduler-progressing responses — folding would be
            // wasted work. One walk decides the gate and feeds the fold
            // (`optimization_update_if_changed`), pinned by the kernel invariant tests to gating
            // on the frontier-only read and folding separately. Still zero cached state: keeping
            // a cached overlay valid across reorgs is complex and error-prone, so the full fold
            // recomputes from the finalized base — just only when its frontier (and hence the
            // dispatch gate) actually moved.
            let update = chain_state.optimization_update_if_changed(
                chain,
                state.last_optimized_block.get(&chain).copied(),
            );

            state.chains.insert(chain, ChainLifecycle::Active(chain_state));
            // Dispatch only when *this* chain's fold frontier advanced and its own projection is
            // ready. Record the hash only on that success so an unready (`Ok(None)`) or failed
            // (`Err`) chain retries this block next event. The dispatched input is then re-derived
            // across *all* active chains, so a slow chain rides along with its current state and a
            // fast chain never stalls waiting for it.
            //
            // A persistent `Err(PoolReserveProjectionError)` is deliberately silent here (this is
            // pure code — no logger to hand it to): the chain simply never dispatches. Its runtime
            // signature is `behind=?` never resolving in the gauge/view while `pools`/`inflight`
            // look healthy. If a live run ever shows that, surface the error through the
            // observation read model rather than threading a logger in here.
            if let Some(update) = update
                && let Ok(Some(chain_reserves)) =
                    pool_reserves_for_optimization(&state, chain, &update)
            {
                state.last_optimized_block.insert(chain, update.block_hash);
                let input = merged_optimization_reserves(&state, chain, &chain_reserves);
                if !input.reserves.is_empty() {
                    effects.push(Effect::RunOptimization { input });
                }
            }

            (state, effects)
        }
        Some(existing_chain) => {
            state.chains.insert(chain, existing_chain);
            (state, Vec::new())
        }
        None => (state, Vec::new()),
    }
}

/// Advances active and bootstrapping chains and forwards any retry or scheduler effects they produce.
/// Added so a single global tick can drive request TTL handling for active kernels and the bootstrap
/// retry/deadline timers for chains still warming up.
fn tick(mut state: State) -> (State, Vec<Effect>) {
    let (chains, effects) = std::mem::take(&mut state.chains).into_iter().fold(
        (BTreeMap::new(), Vec::new()),
        |(mut chains, mut effects), (chain, chain_state)| {
            match chain_state {
                ChainLifecycle::Bootstrapping(bootstrap_state, buffered) => {
                    let (lifecycle, chain_effects) =
                        advance_bootstrap(chain, bootstrap_state, buffered, bootstrap::Event::Tick);
                    if let Some(lifecycle) = lifecycle {
                        chains.insert(chain, lifecycle);
                    }
                    effects.extend(chain_effects);
                }
                ChainLifecycle::Active(chain_state) => {
                    let (chain_state, chain_effects) =
                        kernel::transition(chain, chain_state, kernel::Event::Tick);
                    chains.insert(chain, ChainLifecycle::Active(chain_state));
                    effects.extend(
                        chain_effects
                            .into_iter()
                            .map(|effect| Effect::ChainEffect { chain, effect }),
                    );
                }
            }

            (chains, effects)
        },
    );
    state.chains = chains;

    (state, effects)
}

fn finalized_refresh_policy(chain: ChainKey) -> FinalizedRefreshPolicy {
    match chain {
        ChainKey::Ethereum => FinalizedRefreshPolicy {
            target_len: ETHEREUM_APPROX_FINALIZED_BLOCK_AGE + ETHEREUM_FINALIZED_RETENTION_MARGIN,
            retry_stride: ETHEREUM_FINALIZED_REFRESH_RETRY_STRIDE,
        },
        ChainKey::Arbitrum => FinalizedRefreshPolicy {
            target_len: ARBITRUM_APPROX_FINALIZED_BLOCK_AGE + ARBITRUM_FINALIZED_RETENTION_MARGIN,
            retry_stride: ARBITRUM_FINALIZED_REFRESH_RETRY_STRIDE,
        },
        ChainKey::Base => FinalizedRefreshPolicy {
            target_len: BASE_APPROX_FINALIZED_BLOCK_AGE + BASE_FINALIZED_RETENTION_MARGIN,
            retry_stride: BASE_FINALIZED_REFRESH_RETRY_STRIDE,
        },
        ChainKey::Optimism => FinalizedRefreshPolicy {
            target_len: OPTIMISM_APPROX_FINALIZED_BLOCK_AGE + OPTIMISM_FINALIZED_RETENTION_MARGIN,
            retry_stride: OPTIMISM_FINALIZED_REFRESH_RETRY_STRIDE,
        },
        ChainKey::Polygon => FinalizedRefreshPolicy {
            target_len: POLYGON_APPROX_FINALIZED_BLOCK_AGE + POLYGON_FINALIZED_RETENTION_MARGIN,
            retry_stride: POLYGON_FINALIZED_REFRESH_RETRY_STRIDE,
        },
        ChainKey::Bnb => FinalizedRefreshPolicy {
            target_len: BNB_APPROX_FINALIZED_BLOCK_AGE + BNB_FINALIZED_RETENTION_MARGIN,
            retry_stride: BNB_FINALIZED_REFRESH_RETRY_STRIDE,
        },
        ChainKey::Avalanche => FinalizedRefreshPolicy {
            target_len: AVALANCHE_APPROX_FINALIZED_BLOCK_AGE + AVALANCHE_FINALIZED_RETENTION_MARGIN,
            retry_stride: AVALANCHE_FINALIZED_REFRESH_RETRY_STRIDE,
        },
    }
}

fn bootstrap_policy(chain: ChainKey) -> bootstrap::BootstrapPolicy {
    match chain {
        ChainKey::Ethereum => bootstrap::BootstrapPolicy {
            tip_trim: ETHEREUM_BOOTSTRAP_TIP_TRIM,
            deadline_ticks: ETHEREUM_BOOTSTRAP_DEADLINE_TICKS,
        },
        ChainKey::Arbitrum => bootstrap::BootstrapPolicy {
            tip_trim: ARBITRUM_BOOTSTRAP_TIP_TRIM,
            deadline_ticks: ARBITRUM_BOOTSTRAP_DEADLINE_TICKS,
        },
        ChainKey::Base => bootstrap::BootstrapPolicy {
            tip_trim: BASE_BOOTSTRAP_TIP_TRIM,
            deadline_ticks: BASE_BOOTSTRAP_DEADLINE_TICKS,
        },
        ChainKey::Optimism => bootstrap::BootstrapPolicy {
            tip_trim: OPTIMISM_BOOTSTRAP_TIP_TRIM,
            deadline_ticks: OPTIMISM_BOOTSTRAP_DEADLINE_TICKS,
        },
        ChainKey::Polygon => bootstrap::BootstrapPolicy {
            tip_trim: POLYGON_BOOTSTRAP_TIP_TRIM,
            deadline_ticks: POLYGON_BOOTSTRAP_DEADLINE_TICKS,
        },
        ChainKey::Bnb => bootstrap::BootstrapPolicy {
            tip_trim: BNB_BOOTSTRAP_TIP_TRIM,
            deadline_ticks: BNB_BOOTSTRAP_DEADLINE_TICKS,
        },
        ChainKey::Avalanche => bootstrap::BootstrapPolicy {
            tip_trim: AVALANCHE_BOOTSTRAP_TIP_TRIM,
            deadline_ticks: AVALANCHE_BOOTSTRAP_DEADLINE_TICKS,
        },
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashMap, HashSet};

    use alloy::primitives::{Address, BlockHash, Bloom, U160, U256, aliases::I24};
    use optimization::{
        Invertible, OptimizationStepResult, OptimizationStepStatus, VirtualReserveValues,
    };
    use proptest::prelude::*;

    use super::*;
    use crate::kernel;
    use crate::{
        PoolFee, PoolRef, PoolMetadata, PoolState, TokenAddress, TokenAmountConversionError,
        TokenDecimals, TokenMetadata, TrustedPoolRegistry, UniswapV3Fee, u256_token_amount_to_f32,
    };

    #[test]
    fn init_requests_finalized_header_and_marks_chain_bootstrapping() {
        let chain = ChainKey::Ethereum;
        let (state, effects) = State::init(&[chain]);

        assert_eq!(state.status(chain), Some(ChainStatus::Initializing));
        assert!(matches!(
            single_bootstrap_effect(&effects, chain),
            bootstrap::Effect::Request(
                crate::bootstrap::pending_requests::AnyIssuedRequest::FinalizedHeader(_)
            )
        ));
    }

    #[test]
    fn init_has_no_optimization_result() {
        let (state, _effects) = State::init(&[ChainKey::Ethereum]);

        assert_eq!(state.latest_optimization_result(), None);
    }

    #[test]
    fn init_seeds_each_chain_in_set_as_bootstrapping() {
        let (state, _effects) = State::init(crate::ACTIVE_CHAINS);

        let observations = state.observe();

        assert_eq!(observations.len(), crate::ACTIVE_CHAINS.len());
        for &chain in crate::ACTIVE_CHAINS {
            assert_eq!(state.status(chain), Some(ChainStatus::Initializing));
            assert!(
                observations.iter().any(|(observed_chain, observation)| {
                    *observed_chain == chain
                        && matches!(observation, ChainObservation::Initializing { .. })
                }),
                "active chain {chain:?} should be seeded as initializing"
            );
        }
    }

    // Maps a live-subscription new head into its kernel event.
    // This pins that the block number survives the SubscriptionData -> kernel::Event hop, the one
    // intermediate struct that carries it, so the graph's block-admission entry can read it.
    #[test]
    fn subscription_new_head_carries_block_number_into_head_observed_event() {
        let head = BlockHash::with_last_byte(20);
        let parent = BlockHash::with_last_byte(1);
        let number = 4_242;

        let event = SubscriptionData::NewHead {
            hash: head,
            parent_hash: parent,
            logs_bloom: Bloom::ZERO,
            number,
        }
        .into_kernel_event();

        assert!(matches!(
            event,
            kernel::Event::HeadObserved {
                hash,
                parent_hash,
                number: observed_number,
                ..
            } if hash == head && parent_hash == parent && observed_number == number
        ));
    }

    #[test]
    fn optimization_step_completed_stores_result_without_touching_chains() {
        let state = active_state_at(ChainKey::Ethereum, hash(1));
        let before = state.observe();
        let result = OptimizationStepResult {
            status: OptimizationStepStatus::Updated,
            input_amount: 1_000.0,
            output_amount: 1_012.0,
            profit_amount: 12.0,
            reserves_count: 3,
            disabled_count: 0,
            pool_slots: 3,
            route_entropy: 0.0,
            effective_pools: 0.0,
            routed_pool_count: 0,
            iterations_completed: 10,
        };

        let (state, effects) = transition(
            state,
            Event::OptimizationStepCompleted { result, plan: None },
        );

        assert!(effects.is_empty());
        assert_eq!(state.latest_optimization_result(), Some(result));
        assert_eq!(state.latest_plan_verification(), None);
        assert_eq!(state.observe(), before);
    }

    proptest! {
        #[test]
        fn optimization_step_completed_overwrites_previous_result(
            first in optimization_step_result_strategy(),
            second in optimization_step_result_strategy(),
        ) {
            let state = active_state_at(ChainKey::Ethereum, hash(1));

            let (state, _effects) = transition(
                state,
                Event::OptimizationStepCompleted { result: first, plan: None },
            );
            prop_assert_eq!(state.latest_optimization_result(), Some(first));

            let (state, effects) = transition(
                state,
                Event::OptimizationStepCompleted { result: second, plan: None },
            );
            prop_assert!(effects.is_empty());
            prop_assert_eq!(state.latest_optimization_result(), Some(second));
        }
    }

    /// A deep pool priced mid-tick (the WBTC/USDC state pinned in pool_state.rs's tests), so both
    /// swap limits are comfortably non-zero. `balanced_pool_state` sits exactly on the tick-0
    /// boundary, where the downward swap limit is zero and any hop in that direction clamps to
    /// nothing — useless for a round-trip verification.
    fn mid_tick_pool_state() -> PoolState {
        PoolState {
            sqrt_price_x96: U160::from_limbs([17134602959287796597, 139272449984, 0]),
            liquidity: 50170120777514,
            tick: I24::from_limbs([69583]),
        }
    }

    /// A single-chain state with one seeded, verified, mid-tick 0.3%-fee pool between two
    /// zero-decimal tokens — normalized amounts equal raw units, so verification arithmetic is
    /// directly readable.
    fn verification_state() -> State {
        projection_state(
            ChainKey::Ethereum,
            hash(1),
            HashMap::from([(pool(10), mid_tick_pool_state())]),
            HashMap::from([(
                pool(10),
                pool_metadata(token(1), token(2), UniswapV3Fee::Fee3000),
            )]),
            HashMap::from([(token(1), token_metadata(0)), (token(2), token_metadata(0))]),
        )
    }

    fn plan_swap_step(
        stage: usize,
        token_in: TokenAddress,
        token_out: TokenAddress,
        pool: PoolRef,
    ) -> optimization::ExecutableStep<PoolRef, TokenAddress> {
        optimization::ExecutableStep {
            stage,
            token_in,
            token_out,
            kind: StepKind::Swap(pool),
            weight: 1.0,
            amount_in: 0.0,
            amount_out: 0.0,
        }
    }

    fn round_trip_plan(through: PoolRef) -> ExecutionPlan<PoolRef, TokenAddress> {
        ExecutionPlan {
            init_asset: token(1),
            entry_amount: 100.0,
            steps: vec![
                plan_swap_step(0, token(1), token(2), through),
                plan_swap_step(1, token(2), token(1), through),
            ],
        }
    }

    fn completed_step_event(plan: Option<ExecutionPlan<PoolRef, TokenAddress>>) -> Event {
        Event::OptimizationStepCompleted {
            result: OptimizationStepResult {
                status: OptimizationStepStatus::Updated,
                input_amount: 100.0,
                output_amount: 100.0,
                profit_amount: 0.0,
                reserves_count: 1,
                disabled_count: 0,
                pool_slots: 1,
                route_entropy: 0.0,
                effective_pools: 0.0,
                routed_pool_count: 0,
                iterations_completed: 1,
            },
            plan,
        }
    }

    #[test]
    fn round_trip_plan_verifies_with_fee_loss() {
        let state = verification_state();

        let (state, effects) =
            transition(state, completed_step_event(Some(round_trip_plan(pool(10)))));

        assert!(effects.is_empty());
        let Some(PlanVerification::Verified {
            profit,
            hit_tick_limit,
        }) = state.latest_plan_verification()
        else {
            panic!(
                "expected a verified plan, got {:?}",
                state.latest_plan_verification()
            );
        };
        // A no-arbitrage round trip through one balanced pool loses exactly fees and integer
        // floors: strictly negative, but a small fraction of the 100-unit entry.
        assert!(profit < 0.0, "round trip must lose fees, got {profit}");
        assert!(profit > -10.0, "loss must stay fee-sized, got {profit}");
        assert!(!hit_tick_limit);
    }

    #[test]
    fn plan_through_unresolvable_pool_is_unverifiable_pool_not_found() {
        let state = verification_state();

        let (state, _effects) =
            transition(state, completed_step_event(Some(round_trip_plan(pool(99)))));

        assert_eq!(
            state.latest_plan_verification(),
            Some(PlanVerification::Unverifiable(
                PlanVerificationFailure::Replay(LosslessReplayError::PoolNotFound)
            ))
        );
    }

    #[test]
    fn plan_with_unverified_init_asset_is_unverifiable_init_asset_unknown() {
        let state = verification_state();
        let mut plan = round_trip_plan(pool(10));
        plan.init_asset = token(9); // no verified token metadata

        let (state, _effects) = transition(state, completed_step_event(Some(plan)));

        assert_eq!(
            state.latest_plan_verification(),
            Some(PlanVerification::Unverifiable(
                PlanVerificationFailure::InitAssetUnknown
            ))
        );
    }

    #[test]
    fn step_without_plan_clears_the_previous_verification() {
        let state = verification_state();
        let (state, _effects) =
            transition(state, completed_step_event(Some(round_trip_plan(pool(10)))));
        assert!(state.latest_plan_verification().is_some());

        let (state, _effects) = transition(state, completed_step_event(None));

        // The result and its verification stay in lockstep: a planless step stores the result
        // and clears the stale verdict rather than leaving it paired with the wrong result.
        assert!(state.latest_optimization_result().is_some());
        assert_eq!(state.latest_plan_verification(), None);
    }

    fn optimization_step_result_strategy() -> impl Strategy<Value = OptimizationStepResult> {
        (
            prop_oneof![
                Just(OptimizationStepStatus::Initialized),
                Just(OptimizationStepStatus::Updated),
                Just(OptimizationStepStatus::Extended),
                Just(OptimizationStepStatus::Reinitialized),
                Just(OptimizationStepStatus::Continued),
            ],
            -1.0e6f32..1.0e6f32,
            -1.0e6f32..1.0e6f32,
            -1.0e6f32..1.0e6f32,
            0usize..100,
            0usize..100,
            0usize..100,
        )
            .prop_map(
                |(
                    status,
                    input_amount,
                    output_amount,
                    profit_amount,
                    reserves_count,
                    disabled_count,
                    pool_slots,
                )| OptimizationStepResult {
                    status,
                    input_amount,
                    output_amount,
                    profit_amount,
                    reserves_count,
                    disabled_count,
                    pool_slots,
                    route_entropy: 0.0,
                    effective_pools: 0.0,
                    routed_pool_count: 0,
                    iterations_completed: 10,
                },
            )
    }

    #[test]
    fn observe_reports_initializing_while_bootstrapping() {
        let chain = ChainKey::Ethereum;
        let (state, _effects) = State::init(&[chain]);

        assert_eq!(
            state.observe(),
            vec![(
                chain,
                ChainObservation::Initializing { buffered_events: 0 }
            )]
        );
    }

    #[test]
    fn observe_counts_subscription_data_buffered_during_bootstrap() {
        let chain = ChainKey::Ethereum;
        let (state, _effects) = State::init(&[chain]);

        let (state, _effects) = transition(
            state,
            Event::SubscriptionData {
                chain,
                data: SubscriptionData::NewHead {
                    hash: hash(2),
                    parent_hash: hash(1),
                    logs_bloom: Bloom::ZERO,
                    number: 2,
                },
            },
        );

        assert_eq!(
            state.observe(),
            vec![(
                chain,
                ChainObservation::Initializing { buffered_events: 1 }
            )]
        );
    }

    #[test]
    fn observe_reports_unknown_distance_before_first_optimization_dispatch() {
        let pool = pool(10);
        let token0 = token(1);
        let token1 = token(2);
        let state = projection_state(
            ChainKey::Ethereum,
            hash(1),
            HashMap::new(),
            HashMap::from([(pool, pool_metadata(token0, token1, UniswapV3Fee::Fee3000))]),
            HashMap::from([(token0, token_metadata(18)), (token1, token_metadata(6))]),
        );

        // No optimization has been dispatched, so there is no reference block to measure against.
        assert_eq!(
            state.observe(),
            vec![(
                ChainKey::Ethereum,
                ChainObservation::Active(ChainProgress {
                    verified_pools: 1,
                    blocks_behind_tip: None,
                    canonical_window: Some(0),
                    in_flight_requests: 0,
                    ws_misses: 0,
                })
            )]
        );
    }

    #[test]
    fn observe_measures_distance_from_last_optimized_block() {
        let pool = pool(10);
        let token0 = token(1);
        let token1 = token(2);
        let mut state = projection_state(
            ChainKey::Ethereum,
            hash(1),
            HashMap::new(),
            HashMap::from([(pool, pool_metadata(token0, token1, UniswapV3Fee::Fee3000))]),
            HashMap::from([(token0, token_metadata(18)), (token1, token_metadata(6))]),
        );
        // The canonical tip equals the finalized anchor here, so optimizing at it is zero behind.
        state
            .last_optimized_block
            .insert(ChainKey::Ethereum, hash(1));

        assert_eq!(
            state.observe(),
            vec![(
                ChainKey::Ethereum,
                ChainObservation::Active(ChainProgress {
                    verified_pools: 1,
                    blocks_behind_tip: Some(0),
                    canonical_window: Some(0),
                    in_flight_requests: 0,
                    ws_misses: 0,
                })
            )]
        );
    }

    #[test]
    fn observe_reports_the_canonical_window_growing_with_connected_heads() {
        let chain = ChainKey::Ethereum;
        let state = active_state_at(chain, hash(0));

        let (state, _effects) = drive_connected_heads(state, chain, 3);

        // Three connected heads above the anchor: the fold window the gauge surfaces is exactly
        // the canonical path length.
        assert!(matches!(
            state.observe().as_slice(),
            [(
                observed_chain,
                ChainObservation::Active(ChainProgress {
                    canonical_window: Some(3),
                    ..
                })
            )] if *observed_chain == chain
        ));
    }

    #[test]
    fn subscription_type_lives_with_multi_chain_events_and_effects() {
        let new_heads = Subscription::NewHeadsSubscription(ChainKey::Ethereum);
        let tick = Subscription::TickSubscription(std::time::Duration::from_secs(1));
        let optimization = Subscription::OptimizationSubscription;

        assert!(matches!(
            new_heads,
            Subscription::NewHeadsSubscription(ChainKey::Ethereum)
        ));
        assert!(matches!(tick, Subscription::TickSubscription(_)));
        assert!(matches!(
            optimization,
            Subscription::OptimizationSubscription
        ));
    }

    #[test]
    fn bootstrap_handshake_activates_chain() {
        let chain = ChainKey::Ethereum;
        let finalized_hash = hash(1);
        let child_hash = hash(2);
        let state = drive_bootstrap_to_active(chain, finalized_hash);

        assert_eq!(state.status(chain), Some(ChainStatus::Active));

        let (state, _effects) = transition(
            state,
            Event::ChainEvent {
                chain,
                event: kernel::Event::HeadObserved {
                    logs_bloom: crate::Bloom::repeat_byte(0xff),
                    hash: child_hash,
                    parent_hash: finalized_hash,
                    number: block_number_for(child_hash),
                },
            },
        );
        // WS-primary: the backstop log fetch fires once the block sinks past the settle window.
        let (_state, effects) = observe_clear_padding(state, chain, child_hash, 0xE1);
        assert_single_log_request_chain_effect(&effects, chain, child_hash);
    }

    #[test]
    fn bootstrap_deadline_before_anchor_abandons_chain() {
        let chain = ChainKey::Ethereum;
        let (mut state, _effects) = State::init(&[chain]);
        let policy = bootstrap_policy(chain);

        for _ in 0..policy.deadline_ticks {
            let (next_state, _effects) = transition(state, Event::Tick);
            state = next_state;
        }

        assert_eq!(state.status(chain), None);
    }

    #[test]
    fn chain_event_for_inactive_chain_is_ignored() {
        let chain = ChainKey::Ethereum;
        let (mut state, _effects) = State::init(&[chain]);
        let policy = bootstrap_policy(chain);

        for _ in 0..policy.deadline_ticks {
            let (next_state, _effects) = transition(state, Event::Tick);
            state = next_state;
        }
        assert_eq!(state.status(chain), None);

        let (state, effects) = transition(
            state,
            Event::ChainEvent {
                chain,
                event: kernel::Event::Tick,
            },
        );

        assert_eq!(state.status(chain), None);
        assert!(effects.is_empty());
    }

    #[test]
    fn active_chain_event_routes_inner_effects_with_chain_key() {
        let chain = ChainKey::Ethereum;
        let finalized_hash = hash(1);
        let missing_parent_hash = hash(2);
        let observed_hash = hash(3);
        let state = active_state_at(chain, finalized_hash);

        let (_state, effects) = transition(
            state,
            Event::ChainEvent {
                chain,
                event: kernel::Event::HeadObserved {
                    logs_bloom: crate::Bloom::repeat_byte(0xff),
                    hash: observed_hash,
                    parent_hash: missing_parent_hash,
                    number: block_number_for(observed_hash),
                },
            },
        );

        assert_single_header_request_chain_effect(&effects, chain, missing_parent_hash);
    }

    #[test]
    fn tick_keeps_bootstrapping_chain_initializing_before_deadline() {
        let chain = ChainKey::Ethereum;
        let (state, _) = State::init(&[chain]);

        let (state, _effects) = transition(state, Event::Tick);

        assert_eq!(state.status(chain), Some(ChainStatus::Initializing));
    }

    #[test]
    fn tick_routes_inner_effects_with_chain_key() {
        let chain = ChainKey::Ethereum;
        let finalized_hash = hash(1);
        let missing_parent_hash = hash(2);
        let observed_hash = hash(3);
        let state = active_state_at(chain, finalized_hash);
        let (state, effects) = transition(
            state,
            Event::ChainEvent {
                chain,
                event: kernel::Event::HeadObserved {
                    logs_bloom: crate::Bloom::repeat_byte(0xff),
                    hash: observed_hash,
                    parent_hash: missing_parent_hash,
                    number: block_number_for(observed_hash),
                },
            },
        );
        assert_single_header_request_chain_effect(&effects, chain, missing_parent_hash);

        let (state, effects) = (0..crate::tick::REQUEST_TTL_FOR_TEST)
            .fold((state, Vec::new()), |(state, _effects), _| {
                transition(state, Event::Tick)
            });

        assert_eq!(state.status(chain), Some(ChainStatus::Active));
        assert_single_header_request_chain_effect(&effects, chain, missing_parent_hash);
    }

    #[test]
    fn connected_path_below_refresh_target_does_not_fetch_finalized_header() {
        let chain = ChainKey::Ethereum;
        let finalized_hash = hash(0);
        let policy = finalized_refresh_policy(chain);
        let state = active_state_at(chain, finalized_hash);
        let (_state, effects) = drive_connected_heads(state, chain, policy.target_len - 1);

        assert_no_fetch_finalized_header_effect(&effects, chain);
    }

    #[test]
    fn connected_path_crossing_refresh_target_fetches_finalized_header_once() {
        let chain = ChainKey::Ethereum;
        let finalized_hash = hash(0);
        let policy = finalized_refresh_policy(chain);
        let state = active_state_at(chain, finalized_hash);
        let (state, effects) = drive_connected_heads(state, chain, policy.target_len);

        assert_single_fetch_finalized_header_effect_for_chain(&effects, chain);

        let (_state, duplicate_effects) = observe_head(
            state,
            chain,
            hash_for_index(policy.target_len),
            hash_for_index(policy.target_len - 1),
        );

        assert_no_fetch_finalized_header_effect(&duplicate_effects, chain);
    }

    #[test]
    fn same_tip_log_event_after_refresh_target_does_not_duplicate_finalized_fetch() {
        let chain = ChainKey::Ethereum;
        let finalized_hash = hash(0);
        let policy = finalized_refresh_policy(chain);
        let state = active_state_at(chain, finalized_hash);
        let (state, effects) = drive_connected_heads(state, chain, policy.target_len);
        // WS-primary: the last head observation carries the backstop fetch for the block that just
        // crossed the settle window, not for the head itself.
        let request_id = assert_single_log_request_chain_effect(
            &effects,
            chain,
            hash_for_index(policy.target_len - kernel::STREAM_SETTLE_DEPTH),
        );

        let (_state, effects) = transition(
            state,
            Event::ChainEvent {
                chain,
                event: kernel::Event::BlockLogsReceived {
                    request_id,
                    logs: Vec::new(),
                },
            },
        );

        assert_no_fetch_finalized_header_effect(&effects, chain);
    }

    #[test]
    fn connected_heads_inside_retry_bucket_do_not_duplicate_finalized_fetch() {
        let chain = ChainKey::Ethereum;
        let finalized_hash = hash(0);
        let policy = finalized_refresh_policy(chain);
        let state = active_state_at(chain, finalized_hash);
        let (mut state, _effects) = drive_connected_heads(state, chain, policy.target_len);

        for block_index in policy.target_len + 1..policy.target_len + policy.retry_stride.get() {
            let (next_state, effects) = observe_head(
                state,
                chain,
                hash_for_index(block_index),
                hash_for_index(block_index - 1),
            );

            assert_no_fetch_finalized_header_effect(&effects, chain);
            state = next_state;
        }
    }

    #[test]
    fn connected_head_crossing_retry_bucket_fetches_finalized_header_once() {
        let chain = ChainKey::Ethereum;
        let finalized_hash = hash(0);
        let policy = finalized_refresh_policy(chain);
        let state = active_state_at(chain, finalized_hash);
        let (state, _effects) = drive_connected_heads(state, chain, policy.target_len);
        let mut state = state;
        let mut refresh_count = 0usize;

        for block_index in policy.target_len + 1..=policy.target_len + policy.retry_stride.get() {
            let (next_state, effects) = observe_head(
                state,
                chain,
                hash_for_index(block_index),
                hash_for_index(block_index - 1),
            );

            refresh_count += fetch_finalized_header_effect_count(&effects, chain);
            state = next_state;
        }

        assert_eq!(refresh_count, 1);
    }

    #[test]
    fn reconnecting_long_canonical_path_fetches_finalized_header_once() {
        let chain = ChainKey::Ethereum;
        let finalized_hash = hash(0);
        let policy = finalized_refresh_policy(chain);
        let state = active_state_at(chain, finalized_hash);
        let (mut state, mut effects) = observe_head(
            state,
            chain,
            hash_for_index(policy.target_len),
            hash_for_index(policy.target_len - 1),
        );
        let mut refresh_count = fetch_finalized_header_effect_count(&effects, chain);

        for missing_index in (1..policy.target_len).rev() {
            let missing_hash = hash_for_index(missing_index);
            let request_id = single_header_request_id(&effects, chain, missing_hash);
            let parent_hash = if missing_index == 1 {
                finalized_hash
            } else {
                hash_for_index(missing_index - 1)
            };

            let (next_state, next_effects) = transition(
                state,
                Event::ChainEvent {
                    chain,
                    event: kernel::Event::BlockHeaderReceived {
                        logs_bloom: crate::Bloom::repeat_byte(0xff),
                        request_id,
                        hash: missing_hash,
                        parent_hash,
                        number: block_number_for(missing_hash),
                    },
                },
            );

            refresh_count += fetch_finalized_header_effect_count(&next_effects, chain);
            state = next_state;
            effects = next_effects;
        }

        assert_eq!(refresh_count, 1);

        let (_state, duplicate_effects) = observe_head(
            state,
            chain,
            hash_for_index(policy.target_len),
            hash_for_index(policy.target_len - 1),
        );

        assert_no_fetch_finalized_header_effect(&duplicate_effects, chain);
    }

    #[test]
    fn finalized_header_received_for_active_chain_routes_to_inner_compaction() {
        let chain = ChainKey::Ethereum;
        let finalized_hash = hash(1);
        let compacted_hash = hash(2);
        let old_branch_hash = hash(3);
        let state = active_state_at(chain, finalized_hash);
        let (state, _effects) = transition(
            state,
            Event::ChainEvent {
                chain,
                event: kernel::Event::HeadObserved {
                    logs_bloom: crate::Bloom::repeat_byte(0xff),
                    hash: compacted_hash,
                    parent_hash: finalized_hash,
                    number: block_number_for(compacted_hash),
                },
            },
        );
        // WS-primary: sink the block past the settle window so the backstop issues its log fetch.
        let (state, effects) = observe_clear_padding(state, chain, compacted_hash, 0xE1);
        let log_request_id =
            assert_single_log_request_chain_effect(&effects, chain, compacted_hash);
        let (state, effects) = transition(
            state,
            Event::ChainEvent {
                chain,
                event: kernel::Event::BlockLogsReceived {
                    request_id: log_request_id,
                    logs: Default::default(),
                },
            },
        );
        // Resolving the block's logs advances the complete pool-state frontier to this block. The
        // empty test registry projects no reserves, so the merged optimizer input is empty and the
        // kernel emits nothing — it dispatches an optimization run only once some chain has
        // projectable reserves.
        assert!(effects.is_empty());

        let (state, effects) = transition(
            state,
            Event::FinalizedHeaderReceived {
                chain,
                block_hash: compacted_hash,
            },
        );

        assert!(effects.is_empty());

        let (_state, effects) = transition(
            state,
            Event::ChainEvent {
                chain,
                event: kernel::Event::HeadObserved {
                    logs_bloom: crate::Bloom::repeat_byte(0xff),
                    hash: old_branch_hash,
                    parent_hash: finalized_hash,
                    number: block_number_for(old_branch_hash),
                },
            },
        );

        assert_single_header_request_chain_effect(&effects, chain, finalized_hash);
    }

    #[test]
    fn run_optimization_effect_carries_chain_neutral_input() {
        let input = OptimizationPoolReserves {
            block_hashes: BTreeMap::from([(ChainKey::Ethereum, hash(1))]),
            reserves: Vec::new(),
        };

        let effect = Effect::RunOptimization {
            input: input.clone(),
        };

        assert!(matches!(
            effect,
            Effect::RunOptimization { input: effect_input }
                if effect_input == input
        ));
    }

    #[test]
    fn chain_event_emits_run_optimization_when_complete_block_advances() {
        let pool = pool(10);
        let token0 = token(1);
        let token1 = token(2);
        let pool_state = balanced_pool_state(1_000_000);
        let state = projection_state(
            ChainKey::Ethereum,
            hash(1),
            HashMap::from([(pool, pool_state.clone())]),
            HashMap::from([(pool, pool_metadata(token0, token1, UniswapV3Fee::Fee3000))]),
            HashMap::from([(token0, token_metadata(18)), (token1, token_metadata(6))]),
        );

        let (_state, effects) = transition(
            state,
            Event::ChainEvent {
                chain: ChainKey::Ethereum,
                event: kernel::Event::Tick,
            },
        );

        let inputs = effects
            .iter()
            .filter_map(|effect| match effect {
                Effect::RunOptimization { input } => Some(input),
                _ => None,
            })
            .collect::<Vec<_>>();

        assert_eq!(inputs.len(), 1);
        assert_eq!(
            inputs[0].block_hashes.get(&ChainKey::Ethereum),
            Some(&hash(1))
        );
        assert_directional_pair(&inputs[0].reserves, pool, token0, token1, &pool_state);
    }

    #[test]
    fn chain_event_does_not_re_emit_run_optimization_for_unchanged_complete_block() {
        let pool = pool(10);
        let token0 = token(1);
        let token1 = token(2);
        let state = projection_state(
            ChainKey::Ethereum,
            hash(1),
            HashMap::from([(pool, balanced_pool_state(1_000_000))]),
            HashMap::from([(pool, pool_metadata(token0, token1, UniswapV3Fee::Fee3000))]),
            HashMap::from([(token0, token_metadata(18)), (token1, token_metadata(6))]),
        );

        let (state, _effects) = transition(
            state,
            Event::ChainEvent {
                chain: ChainKey::Ethereum,
                event: kernel::Event::Tick,
            },
        );

        let (_state, effects) = transition(
            state,
            Event::ChainEvent {
                chain: ChainKey::Ethereum,
                event: kernel::Event::Tick,
            },
        );

        assert!(
            !effects
                .iter()
                .any(|effect| matches!(effect, Effect::RunOptimization { .. }))
        );
    }

    // Pins the inert skip: a fan-out re-delivery of the current tip produces NO effects at the
    // multi-chain level — no `ChainEffect`, no `FetchFinalizedHeader`, no `RunOptimization` —
    // and leaves the dispatch bookkeeping untouched. The kernel already proved the state
    // bit-identical (`TransitionOutcome::Inert`); this asserts the wrapper adds nothing on top.
    #[test]
    fn chain_event_for_duplicate_of_tip_head_produces_no_effects() {
        let state = projection_state(
            ChainKey::Ethereum,
            hash(1),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
        );
        let head_event = || Event::ChainEvent {
            chain: ChainKey::Ethereum,
            event: kernel::Event::HeadObserved {
                number: 2,
                logs_bloom: Bloom::default(),
                hash: hash(2),
                parent_hash: hash(1),
            },
        };

        let (state, _effects) = transition(state, head_event());
        let last_optimized_before = state.last_optimized_block.clone();

        let (state, effects) = transition(state, head_event());

        assert!(effects.is_empty());
        assert_eq!(state.last_optimized_block, last_optimized_before);
    }

    #[test]
    fn pool_reserves_projection_returns_none_when_ethereum_chain_is_inactive() {
        let (state, _) = State::init(&[ChainKey::Ethereum]);
        let update = projection_update(hash(1), HashMap::new());

        let reserves = pool_reserves_for_optimization(&state, ChainKey::Ethereum, &update).unwrap();

        assert_eq!(reserves, None);
    }

    #[test]
    fn pool_reserves_projection_returns_empty_reserves_for_complete_empty_update() {
        let state = projection_state(
            ChainKey::Ethereum,
            hash(1),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
        );
        let update = projection_update(hash(2), HashMap::new());

        let reserves = pool_reserves_for_optimization(&state, ChainKey::Ethereum, &update)
            .unwrap()
            .unwrap();

        assert_eq!(reserves.block_hash, update.block_hash);
        assert!(reserves.reserves.is_empty());
    }

    #[test]
    fn pool_reserves_projection_includes_finalized_snapshots() {
        let pool = pool(10);
        let token0 = token(1);
        let token1 = token(2);
        let pool_state = balanced_pool_state(1_000_000);
        let state = projection_state(
            ChainKey::Ethereum,
            hash(1),
            HashMap::from([(pool, pool_state.clone())]),
            HashMap::from([(pool, pool_metadata(token0, token1, UniswapV3Fee::Fee3000))]),
            HashMap::from([(token0, token_metadata(18)), (token1, token_metadata(6))]),
        );
        let update = projection_update(hash(2), HashMap::new());

        let reserves = pool_reserves_for_optimization(&state, ChainKey::Ethereum, &update)
            .unwrap()
            .unwrap();

        assert_eq!(reserves.reserves.len(), 2);
        assert_directional_pair(&reserves.reserves, pool, token0, token1, &pool_state);
    }

    #[test]
    fn pool_reserves_projection_selects_only_the_requested_chains_pools() {
        // Same raw addresses on both chains, distinguished only by their `ChainKey` — exactly the
        // collision the widened identity prevents.
        let ethereum_pool = pool_on(ChainKey::Ethereum, 10);
        let arbitrum_pool = pool_on(ChainKey::Arbitrum, 10);
        let ethereum_token0 = token_on(ChainKey::Ethereum, 1);
        let ethereum_token1 = token_on(ChainKey::Ethereum, 2);
        let arbitrum_token0 = token_on(ChainKey::Arbitrum, 1);
        let arbitrum_token1 = token_on(ChainKey::Arbitrum, 2);
        let ethereum_pool_state = balanced_pool_state(1_000_000);
        let arbitrum_pool_state = balanced_pool_state(2_000_000);

        let ethereum_state = projection_state(
            ChainKey::Ethereum,
            hash(1),
            HashMap::from([(ethereum_pool, ethereum_pool_state.clone())]),
            HashMap::from([(
                ethereum_pool,
                pool_metadata(ethereum_token0, ethereum_token1, UniswapV3Fee::Fee3000),
            )]),
            HashMap::from([
                (ethereum_token0, token_metadata(18)),
                (ethereum_token1, token_metadata(6)),
            ]),
        );
        let arbitrum_state = projection_state(
            ChainKey::Arbitrum,
            hash(2),
            HashMap::from([(arbitrum_pool, arbitrum_pool_state.clone())]),
            HashMap::from([(
                arbitrum_pool,
                pool_metadata(arbitrum_token0, arbitrum_token1, UniswapV3Fee::Fee3000),
            )]),
            HashMap::from([
                (arbitrum_token0, token_metadata(18)),
                (arbitrum_token1, token_metadata(6)),
            ]),
        );

        // Fold both active chains into one multi-chain state to prove selection happens by argument.
        let mut chains = ethereum_state.chains;
        chains.extend(arbitrum_state.chains);
        let state = State {
            chains,
            latest_optimization_result: None,
            latest_plan_verification: None,
            last_optimized_block: BTreeMap::new(),
        };

        let ethereum_update = projection_update(hash(3), HashMap::new());
        let arbitrum_update = projection_update(hash(4), HashMap::new());

        let ethereum_reserves =
            pool_reserves_for_optimization(&state, ChainKey::Ethereum, &ethereum_update)
                .unwrap()
                .unwrap();
        let arbitrum_reserves =
            pool_reserves_for_optimization(&state, ChainKey::Arbitrum, &arbitrum_update)
                .unwrap()
                .unwrap();

        // Each chain projects exactly its own pool (two directional entries) — no cross-chain bleed.
        assert_eq!(ethereum_reserves.reserves.len(), 2);
        assert_eq!(arbitrum_reserves.reserves.len(), 2);
        assert_directional_pair(
            &ethereum_reserves.reserves,
            ethereum_pool,
            ethereum_token0,
            ethereum_token1,
            &ethereum_pool_state,
        );
        assert_directional_pair(
            &arbitrum_reserves.reserves,
            arbitrum_pool,
            arbitrum_token0,
            arbitrum_token1,
            &arbitrum_pool_state,
        );
    }

    /// Builds a two-chain Active state whose pools share the same raw addresses across chains,
    /// distinguished only by `ChainKey` — the cross-chain collision the widened identity prevents,
    /// now exercised through the live merge instead of per-chain projection.
    fn merged_two_chain_state() -> State {
        let ethereum_state = projection_state(
            ChainKey::Ethereum,
            hash(1),
            HashMap::from([(
                pool_on(ChainKey::Ethereum, 10),
                balanced_pool_state(1_000_000),
            )]),
            HashMap::from([(
                pool_on(ChainKey::Ethereum, 10),
                pool_metadata(
                    token_on(ChainKey::Ethereum, 1),
                    token_on(ChainKey::Ethereum, 2),
                    UniswapV3Fee::Fee3000,
                ),
            )]),
            HashMap::from([
                (token_on(ChainKey::Ethereum, 1), token_metadata(18)),
                (token_on(ChainKey::Ethereum, 2), token_metadata(6)),
            ]),
        );
        let arbitrum_state = projection_state(
            ChainKey::Arbitrum,
            hash(2),
            HashMap::from([(
                pool_on(ChainKey::Arbitrum, 10),
                balanced_pool_state(2_000_000),
            )]),
            HashMap::from([(
                pool_on(ChainKey::Arbitrum, 10),
                pool_metadata(
                    token_on(ChainKey::Arbitrum, 1),
                    token_on(ChainKey::Arbitrum, 2),
                    UniswapV3Fee::Fee3000,
                ),
            )]),
            HashMap::from([
                (token_on(ChainKey::Arbitrum, 1), token_metadata(18)),
                (token_on(ChainKey::Arbitrum, 2), token_metadata(6)),
            ]),
        );

        let mut chains = ethereum_state.chains;
        chains.extend(arbitrum_state.chains);
        State {
            chains,
            latest_optimization_result: None,
            latest_plan_verification: None,
            last_optimized_block: BTreeMap::new(),
        }
    }

    /// The triggering chain's projection exactly as `chain_event` computes it before the merge —
    /// the value the merge splices in instead of recomputing.
    fn precomputed_reserves_for(state: &State, chain: ChainKey) -> ChainPoolReserves {
        let update = match state.chains.get(&chain) {
            Some(ChainLifecycle::Active(chain_state)) => chain_state.optimization_update(chain),
            _ => panic!("fixture chain must be active"),
        };
        pool_reserves_for_optimization(state, chain, &update)
            .unwrap()
            .unwrap()
    }

    #[test]
    fn merged_optimization_reserves_concatenates_every_active_chain() {
        let state = merged_two_chain_state();

        // Differential pin: handing the merge one chain's freshly-computed projection (as the
        // production call site does) must reproduce the recompute-everything output verbatim —
        // all assertions below are unchanged from before the precomputed splice existed.
        let precomputed = precomputed_reserves_for(&state, ChainKey::Ethereum);
        let merged = merged_optimization_reserves(&state, ChainKey::Ethereum, &precomputed);

        // Both chains contribute their directional pair into one flat Vec, each tagged with its own
        // block, and the same raw addresses on different chains stay distinct keys (no collision).
        assert_eq!(merged.reserves.len(), 4);
        assert_eq!(
            merged.block_hashes,
            BTreeMap::from([(ChainKey::Ethereum, hash(1)), (ChainKey::Arbitrum, hash(2))])
        );
        assert_directional_pair(
            &merged.reserves,
            pool_on(ChainKey::Ethereum, 10),
            token_on(ChainKey::Ethereum, 1),
            token_on(ChainKey::Ethereum, 2),
            &balanced_pool_state(1_000_000),
        );
        assert_directional_pair(
            &merged.reserves,
            pool_on(ChainKey::Arbitrum, 10),
            token_on(ChainKey::Arbitrum, 1),
            token_on(ChainKey::Arbitrum, 2),
            &balanced_pool_state(2_000_000),
        );
        let keys = merged
            .reserves
            .iter()
            .map(|reserve| (reserve.pool_id, reserve.token0, reserve.token1))
            .collect::<HashSet<_>>();
        assert_eq!(
            keys.len(),
            merged.reserves.len(),
            "reserve keys must be unique"
        );
    }

    #[test]
    fn merged_optimization_reserves_splices_the_precomputed_chain_verbatim() {
        let state = merged_two_chain_state();

        // A sentinel block hash no recompute could produce: it surfacing in the merge proves the
        // precomputed projection is spliced in, not recomputed from state.
        let precomputed = ChainPoolReserves {
            block_hash: hash(9),
            reserves: precomputed_reserves_for(&state, ChainKey::Arbitrum).reserves,
        };

        let merged = merged_optimization_reserves(&state, ChainKey::Arbitrum, &precomputed);

        assert_eq!(merged.block_hashes.get(&ChainKey::Arbitrum), Some(&hash(9)));
        // The other chain still contributes via its own recompute.
        assert_eq!(merged.block_hashes.get(&ChainKey::Ethereum), Some(&hash(1)));
        assert_eq!(merged.reserves.len(), 4);
    }

    #[test]
    fn chain_event_emits_merge_spanning_all_chains_when_one_chain_advances() {
        let state = merged_two_chain_state();

        // Advance only Arbitrum. The dispatched input is reconstructed across *all* active chains, so
        // it still carries Ethereum's reserves — proving the merge reads every chain, not just the
        // one that advanced, and that no per-chain reserves cache is involved.
        let (_state, effects) = transition(
            state,
            Event::ChainEvent {
                chain: ChainKey::Arbitrum,
                event: kernel::Event::Tick,
            },
        );

        let input = effects
            .iter()
            .find_map(|effect| match effect {
                Effect::RunOptimization { input } => Some(input),
                _ => None,
            })
            .expect("advancing a chain dispatches a merged optimization run");

        assert_eq!(input.reserves.len(), 4);
        assert_eq!(
            input.block_hashes,
            BTreeMap::from([(ChainKey::Ethereum, hash(1)), (ChainKey::Arbitrum, hash(2))])
        );
        assert_directional_pair(
            &input.reserves,
            pool_on(ChainKey::Ethereum, 10),
            token_on(ChainKey::Ethereum, 1),
            token_on(ChainKey::Ethereum, 2),
            &balanced_pool_state(1_000_000),
        );
    }

    #[test]
    fn pool_reserves_projection_update_snapshot_overwrites_finalized_snapshot() {
        let pool = pool(10);
        let token0 = token(1);
        let token1 = token(2);
        let finalized_pool_state = balanced_pool_state(1_000_000);
        let updated_pool_state = balanced_pool_state(2_000_000);
        let state = projection_state(
            ChainKey::Ethereum,
            hash(1),
            HashMap::from([(pool, finalized_pool_state)]),
            HashMap::from([(pool, pool_metadata(token0, token1, UniswapV3Fee::Fee3000))]),
            HashMap::from([(token0, token_metadata(18)), (token1, token_metadata(6))]),
        );
        let update = projection_update(hash(2), HashMap::from([(pool, updated_pool_state.clone())]));

        let reserves = pool_reserves_for_optimization(&state, ChainKey::Ethereum, &update)
            .unwrap()
            .unwrap();

        assert_directional_pair(
            &reserves.reserves,
            pool,
            token0,
            token1,
            &updated_pool_state,
        );
    }

    #[test]
    fn pool_reserves_projection_returns_none_when_pool_metadata_is_missing() {
        let pool = pool(10);
        let state = projection_state(
            ChainKey::Ethereum,
            hash(1),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
        );
        let update = projection_update(hash(2), HashMap::from([(pool, balanced_pool_state(1_000_000))]));

        let reserves = pool_reserves_for_optimization(&state, ChainKey::Ethereum, &update).unwrap();

        assert_eq!(reserves, None);
    }

    #[test]
    fn pool_reserves_projection_returns_none_when_token_metadata_is_missing() {
        let pool = pool(10);
        let token0 = token(1);
        let token1 = token(2);
        let state = projection_state(
            ChainKey::Ethereum,
            hash(1),
            HashMap::new(),
            HashMap::from([(pool, pool_metadata(token0, token1, UniswapV3Fee::Fee3000))]),
            HashMap::from([(token0, token_metadata(18))]),
        );
        let update = projection_update(hash(2), HashMap::from([(pool, balanced_pool_state(1_000_000))]));

        let reserves = pool_reserves_for_optimization(&state, ChainKey::Ethereum, &update).unwrap();

        assert_eq!(reserves, None);
    }

    #[test]
    fn pool_reserves_projection_resolves_native_currency_decimals_without_fetching() {
        let pool = pool(10);
        let native = TokenAddress(Address::ZERO, ChainKey::Ethereum);
        let token1 = token(2);
        let pool_state = balanced_pool_state(1_000_000);
        // Only token1's decimals are seeded. The native currency (token0 = address(0)) is never
        // fetched — it is not an ERC20 — yet the projection must still resolve it (intrinsically 18
        // decimals) and produce reserves instead of pausing on missing token metadata.
        let state = projection_state(
            ChainKey::Ethereum,
            hash(1),
            HashMap::from([(pool, pool_state.clone())]),
            HashMap::from([(pool, pool_metadata(native, token1, UniswapV3Fee::Fee3000))]),
            HashMap::from([(token1, token_metadata(6))]),
        );
        let update = projection_update(hash(2), HashMap::from([(pool, pool_state.clone())]));

        let reserves = pool_reserves_for_optimization(&state, ChainKey::Ethereum, &update)
            .unwrap()
            .unwrap();

        // `assert_directional_pair` expects token0 at 18 decimals — matching the intrinsic native
        // value — and token1 at 6.
        assert_directional_pair(&reserves.reserves, pool, native, token1, &pool_state);
    }

    #[test]
    fn pool_reserves_projection_returns_typed_error_when_amount_conversion_fails() {
        let pool = pool(10);
        let token0 = token(1);
        let token1 = token(2);
        let pool_state = PoolState {
            sqrt_price_x96: U160::from(1u8),
            liquidity: u128::MAX,
            tick: I24::from_limbs([0]),
        };
        let state = projection_state(
            ChainKey::Ethereum,
            hash(1),
            HashMap::new(),
            HashMap::from([(pool, pool_metadata(token0, token1, UniswapV3Fee::Fee3000))]),
            HashMap::from([(token0, token_metadata(0)), (token1, token_metadata(0))]),
        );
        let update = projection_update(hash(2), HashMap::from([(pool, pool_state)]));

        let error =
            pool_reserves_for_optimization(&state, ChainKey::Ethereum, &update).unwrap_err();

        assert!(matches!(
            error,
            PoolReserveProjectionError::AmountConversion {
                pool: error_pool,
                token: error_token,
                value: PoolReserveValueKind::Reserve0,
                source: TokenAmountConversionError::F32Overflow { .. },
            } if error_pool == pool && error_token == token0
        ));
    }

    #[test]
    fn pool_reserves_projection_returns_typed_error_when_swap_limit_fails() {
        let pool = pool(10);
        let token0 = token(1);
        let token1 = token(2);
        let inconsistent_pool_state = PoolState {
            sqrt_price_x96: U160::from(79228162514264337593543950336_u128),
            liquidity: 1_000_000,
            tick: I24::from_limbs([60]),
        };
        let state = projection_state(
            ChainKey::Ethereum,
            hash(1),
            HashMap::new(),
            HashMap::from([(pool, pool_metadata(token0, token1, UniswapV3Fee::Fee3000))]),
            HashMap::from([(token0, token_metadata(18)), (token1, token_metadata(6))]),
        );
        let update = projection_update(hash(2), HashMap::from([(pool, inconsistent_pool_state)]));

        let error =
            pool_reserves_for_optimization(&state, ChainKey::Ethereum, &update).unwrap_err();

        assert!(matches!(
            error,
            PoolReserveProjectionError::SwapLimit0 {
                pool: error_pool,
                ..
            } if error_pool == pool
        ));
    }

    proptest! {
        #[test]
        fn pool_reserves_projection_emits_two_finite_directional_entries_per_pool(
            pool_count in 0u8..8,
        ) {
            let finalized_hash = hash(1);
            let update_hash = hash(2);
            let pools = (0..pool_count)
                .map(|offset| {
                    let pool = pool(20 + offset);
                    let token0 = token(60 + offset.saturating_mul(2));
                    let token1 = token(61 + offset.saturating_mul(2));
                    let pool_state = balanced_pool_state(1_000_000 + u128::from(offset));
                    (pool, token0, token1, pool_state)
                })
                .collect::<Vec<_>>();
            let pool_metadata = pools
                .iter()
                .map(|(pool, token0, token1, _)| {
                    (*pool, pool_metadata(*token0, *token1, UniswapV3Fee::Fee3000))
                })
                .collect::<HashMap<_, _>>();
            let token_metadata = pools
                .iter()
                .flat_map(|(_, token0, token1, _)| {
                    [(*token0, token_metadata(18)), (*token1, token_metadata(6))]
                })
                .collect::<HashMap<_, _>>();
            let state = projection_state(ChainKey::Ethereum,
                finalized_hash,
                HashMap::new(),
                pool_metadata,
                token_metadata,
            );
            let update = projection_update(
                update_hash,
                pools
                    .iter()
                    .map(|(pool, _, _, pool_state)| (*pool, pool_state.clone()))
                    .collect(),
            );

            let reserves = pool_reserves_for_optimization(&state, ChainKey::Ethereum, &update)
                .unwrap()
                .unwrap();

            prop_assert_eq!(reserves.block_hash, update_hash);
            prop_assert_eq!(reserves.reserves.len(), usize::from(pool_count) * 2);

            let observed_directions = reserves
                .reserves
                .iter()
                .map(|reserve| (reserve.pool_id, reserve.token0, reserve.token1))
                .collect::<HashSet<_>>();

            for (pool, token0, token1, _) in pools {
                prop_assert!(observed_directions.contains(&(pool, token0, token1)));
                prop_assert!(observed_directions.contains(&(pool, token1, token0)));
            }

            for reserve in reserves.reserves {
                prop_assert!(reserve.value.token_0.is_finite());
                prop_assert!(reserve.value.token_1.is_finite());
                prop_assert!(reserve.value.max_swap_0.is_finite());
                prop_assert!(reserve.value.max_swap_1.is_finite());
                prop_assert!(reserve.value.token_0 >= 0.0);
                prop_assert!(reserve.value.token_1 >= 0.0);
                prop_assert!(reserve.value.max_swap_0 >= 0.0);
                prop_assert!(reserve.value.max_swap_1 >= 0.0);
                prop_assert!(reserve.value.fee_multiplier > 0.0);
                prop_assert!(reserve.value.fee_multiplier <= 1.0);
            }
        }
    }

    fn hash(value: u8) -> BlockHash {
        BlockHash::with_last_byte(value)
    }

    /// Test-only block number recovered from a block hash's trailing byte. These tests encode block
    /// identity in `BlockHash::with_last_byte(_)`, so this yields a per-hash-stable number for the
    /// log-sourced graph's block-admission entry. No production consumer of the plumbed `number` yet.
    fn block_number_for(hash: BlockHash) -> u64 {
        hash.0[31] as u64
    }

    fn pool(value: u8) -> PoolRef {
        pool_on(ChainKey::Ethereum, value)
    }

    fn token(value: u8) -> TokenAddress {
        token_on(ChainKey::Ethereum, value)
    }

    fn pool_on(chain: ChainKey, value: u8) -> PoolRef {
        PoolRef::uniswap_v3(Address::with_last_byte(value), chain)
    }

    fn token_on(chain: ChainKey, value: u8) -> TokenAddress {
        TokenAddress(Address::with_last_byte(value), chain)
    }

    fn pool_metadata(
        token0: TokenAddress,
        token1: TokenAddress,
        fee: UniswapV3Fee,
    ) -> PoolMetadata {
        PoolMetadata {
            token0: token0.0,
            token1: token1.0,
            fee: PoolFee::Tiered(fee),
        }
    }

    fn token_metadata(decimals: u8) -> TokenMetadata {
        TokenMetadata {
            decimals: token_decimals(decimals),
        }
    }

    fn token_decimals(decimals: u8) -> TokenDecimals {
        TokenDecimals::try_from_u256(U256::from(decimals)).unwrap()
    }

    fn balanced_pool_state(liquidity: u128) -> PoolState {
        PoolState {
            sqrt_price_x96: U160::from(79228162514264337593543950336_u128),
            liquidity,
            tick: I24::from_limbs([0]),
        }
    }

    fn projection_state(
        chain: ChainKey,
        finalized_hash: BlockHash,
        finalized_snapshots: HashMap<PoolRef, PoolState>,
        pool_metadata: HashMap<PoolRef, PoolMetadata>,
        token_metadata: HashMap<TokenAddress, TokenMetadata>,
    ) -> State {
        let pool_registry = TrustedPoolRegistry::new().with_metadata_results(
            chain,
            pool_metadata
                .into_iter()
                .map(|(pool, metadata)| {
                    let address = pool.uniswap_v3_address().expect("v3 pool");
                    (crate::ProtocolPoolKey::UniswapV3(address), Ok(metadata))
                })
                .collect(),
        );
        let token_registry = crate::TokenRegistry::new().with_metadata_results(
            token_metadata
                .into_iter()
                .map(|(token, metadata)| (token, Ok(metadata)))
                .collect(),
        );
        let chain_state = kernel::State::for_pool_reserve_projection_test(
            finalized_hash,
            finalized_snapshots,
            pool_registry,
            token_registry,
        );

        State {
            chains: BTreeMap::from([(chain, ChainLifecycle::Active(chain_state))]),
            latest_optimization_result: None,
            latest_plan_verification: None,
            last_optimized_block: BTreeMap::new(),
        }
    }

    /// Builds the owned overlay update the kernel's optimization read produces, so
    /// projection tests can exercise `pool_reserves_for_optimization` without driving log events
    /// through the fold.
    fn projection_update(
        block_hash: BlockHash,
        pool_states: HashMap<PoolRef, PoolState>,
    ) -> kernel::OptimizationStateUpdate {
        kernel::OptimizationStateUpdate {
            block_hash,
            pool_states,
        }
    }

    fn assert_directional_pair(
        reserves: &[optimization::PoolReserves<PoolRef, TokenAddress>],
        pool: PoolRef,
        token0: TokenAddress,
        token1: TokenAddress,
        pool_state: &PoolState,
    ) {
        let forward = reserves
            .iter()
            .find(|reserve| {
                reserve.pool_id == pool && reserve.token0 == token0 && reserve.token1 == token1
            })
            .unwrap();
        let reverse = reserves
            .iter()
            .find(|reserve| {
                reserve.pool_id == pool && reserve.token0 == token1 && reserve.token1 == token0
            })
            .unwrap();
        let expected_forward = expected_reserve_values(
            pool_state,
            UniswapV3Fee::Fee3000,
            token_metadata(18).decimals,
            token_metadata(6).decimals,
        );

        assert_eq!(forward.value, expected_forward);
        assert_eq!(reverse.value, expected_forward.inverse());
    }

    fn expected_reserve_values(
        pool_state: &PoolState,
        fee: UniswapV3Fee,
        token0_decimals: TokenDecimals,
        token1_decimals: TokenDecimals,
    ) -> VirtualReserveValues {
        let tick_spacing = u16::try_from(fee.tick_spacing()).unwrap();

        VirtualReserveValues {
            token_0: u256_token_amount_to_f32(pool_state.virtual_reserve_x(), token0_decimals)
                .unwrap(),
            token_1: u256_token_amount_to_f32(pool_state.virtual_reserve_y(), token1_decimals)
                .unwrap(),
            fee_multiplier: 1.0 - fee.pips() as f32 / 1_000_000.0,
            max_swap_0: u256_token_amount_to_f32(
                pool_state.swap_limit_x(tick_spacing).unwrap(),
                token0_decimals,
            )
            .unwrap(),
            max_swap_1: u256_token_amount_to_f32(
                pool_state.swap_limit_y(tick_spacing).unwrap(),
                token1_decimals,
            )
            .unwrap(),
        }
    }

    /// Returns the single bootstrap effect emitted for a chain, failing on zero or many.
    /// Bootstrap is sequential, so a non-terminal step always carries exactly one request.
    fn single_bootstrap_effect(effects: &[Effect], chain: ChainKey) -> &bootstrap::Effect {
        let matching = effects
            .iter()
            .filter_map(|effect| match effect {
                Effect::BootstrapEffect {
                    chain: effect_chain,
                    effect,
                } if *effect_chain == chain => Some(effect),
                _ => None,
            })
            .collect::<Vec<_>>();

        assert_eq!(matching.len(), 1);
        matching[0]
    }

    /// Builds an empty-but-well-formed success response for the outstanding bootstrap request.
    /// Empty windows/metadata drive the handshake to a `Ready` outcome anchored at `finalized_hash`.
    fn bootstrap_success_event(
        chain: ChainKey,
        finalized_hash: BlockHash,
        effect: &bootstrap::Effect,
    ) -> Event {
        use crate::bootstrap::pending_requests::AnyIssuedRequest;

        let bootstrap::Effect::Request(request) = effect;
        let event = match request {
            AnyIssuedRequest::FinalizedHeader(issued) => {
                bootstrap::Event::FinalizedHeaderReceived {
                    request_id: issued.request_id,
                    anchor: bootstrap::FinalizedAnchor {
                        hash: finalized_hash,
                        number: 0,
                    },
                }
            }
            AnyIssuedRequest::PoolCandidates(issued) => bootstrap::Event::PoolCandidatesReceived {
                request_id: issued.request_id,
                blocks: Vec::new(),
                scan_tip: 0,
                next_from: None,
            },
            AnyIssuedRequest::PoolMetadata(issued) => bootstrap::Event::PoolMetadataReceived {
                request_id: issued.request_id,
                metadata: HashMap::new(),
            },
            AnyIssuedRequest::TokenMetadata(issued) => bootstrap::Event::TokenMetadataReceived {
                request_id: issued.request_id,
                metadata: HashMap::new(),
            },
        };

        Event::BootstrapEvent { chain, event }
    }

    /// Drives a freshly initialized chain through the full bootstrap handshake to an active kernel.
    /// Feeds empty responses, so the activated kernel is anchored at `finalized_hash` with no seed.
    fn drive_bootstrap_to_active(chain: ChainKey, finalized_hash: BlockHash) -> State {
        let (mut state, mut effects) = State::init(&[chain]);

        while state.status(chain) == Some(ChainStatus::Initializing) {
            let event = bootstrap_success_event(
                chain,
                finalized_hash,
                single_bootstrap_effect(&effects, chain),
            );
            let result = transition(state, event);
            state = result.0;
            effects = result.1;
        }

        state
    }

    #[test]
    fn subscription_data_during_bootstrap_is_buffered_then_replayed_on_activation() {
        let chain = ChainKey::Ethereum;
        let finalized = BlockHash::with_last_byte(9);
        let head = BlockHash::with_last_byte(20);

        let (mut state, effects) = State::init(&[chain]);

        // A live head (building on the finalized anchor) arrives while bootstrapping. It must be
        // buffered — no effect, chain still initializing — not dropped or applied to a missing kernel.
        let (next_state, buffer_effects) = transition(
            state,
            Event::SubscriptionData {
                chain,
                data: SubscriptionData::NewHead {
                    hash: head,
                    parent_hash: finalized,
                    logs_bloom: Bloom::ZERO,
                    number: 20,
                },
            },
        );
        state = next_state;
        assert!(buffer_effects.is_empty());
        assert_eq!(state.status(chain), Some(ChainStatus::Initializing));

        // Complete bootstrap with empty responses (anchored at `finalized`, empty seed). Buffering left
        // the outstanding bootstrap request untouched, so the captured `effects` still drive it.
        let mut effects = effects;
        while state.status(chain) == Some(ChainStatus::Initializing) {
            let event =
                bootstrap_success_event(chain, finalized, single_bootstrap_effect(&effects, chain));
            let result = transition(state, event);
            state = result.0;
            effects = result.1;
        }

        // On activation the buffered head was replayed onto the seeded graph, so it is part of the
        // active kernel's canonical graph rather than lost to a header-walk gap.
        match state.chains.get(&chain) {
            Some(ChainLifecycle::Active(kernel_state)) => {
                assert!(
                    kernel_state.blocks_behind(head).is_some(),
                    "buffered head should be on the activated canonical path"
                );
            }
            _ => panic!("chain should be active after bootstrap"),
        }
    }

    fn assert_single_fetch_finalized_header_effect_for_chain(effects: &[Effect], chain: ChainKey) {
        assert_eq!(fetch_finalized_header_effect_count(effects, chain), 1);
    }

    fn assert_no_fetch_finalized_header_effect(effects: &[Effect], chain: ChainKey) {
        assert_eq!(fetch_finalized_header_effect_count(effects, chain), 0);
    }

    fn fetch_finalized_header_effect_count(effects: &[Effect], chain: ChainKey) -> usize {
        effects
            .iter()
            .filter(|effect| {
                matches!(
                    effect,
                    Effect::FetchFinalizedHeader { chain: effect_chain }
                        if *effect_chain == chain
                )
            })
            .count()
    }

    fn active_state_at(chain: ChainKey, finalized_hash: BlockHash) -> State {
        // No seed blocks, so activation issues no reconnection effects.
        let (chain_state, _effects) = kernel::State::activate_from_seed(
            finalized_hash,
            HashMap::new(),
            TrustedPoolRegistry::new(),
            crate::TokenRegistry::new(),
            Vec::new(),
        );

        State {
            chains: BTreeMap::from([(chain, ChainLifecycle::Active(chain_state))]),
            latest_optimization_result: None,
            latest_plan_verification: None,
            last_optimized_block: BTreeMap::new(),
        }
    }

    fn drive_connected_heads(
        mut state: State,
        chain: ChainKey,
        target_len: usize,
    ) -> (State, Vec<Effect>) {
        let mut effects = Vec::new();

        for block_index in 1..=target_len {
            let parent_hash = if block_index == 1 {
                hash(0)
            } else {
                hash_for_index(block_index - 1)
            };
            let result = observe_head(state, chain, hash_for_index(block_index), parent_hash);
            state = result.0;
            effects = result.1;
        }

        (state, effects)
    }

    fn observe_head(
        state: State,
        chain: ChainKey,
        hash: BlockHash,
        parent_hash: BlockHash,
    ) -> (State, Vec<Effect>) {
        transition(
            state,
            Event::ChainEvent {
                chain,
                event: kernel::Event::HeadObserved {
                    hash,
                    parent_hash,
                    logs_bloom: crate::Bloom::repeat_byte(0xff),
                    number: block_number_for(hash),
                },
            },
        )
    }

    /// Observes `STREAM_SETTLE_DEPTH` bloom-clear padding heads above `parent_hash`, sinking the
    /// blocks under test past the stream's settle window so the per-block backstop may fetch them
    /// (WS-primary — see the kernel tests' twin helper). Returns the state and the *last* padding
    /// transition's effects, where the uncovered holes' log requests surface.
    fn observe_clear_padding(
        state: State,
        chain: ChainKey,
        parent_hash: BlockHash,
        first_byte: u8,
    ) -> (State, Vec<Effect>) {
        let mut state = state;
        let mut parent_hash = parent_hash;
        let mut effects = Vec::new();
        for offset in 0..kernel::STREAM_SETTLE_DEPTH {
            let padding_hash = hash(first_byte + offset as u8);
            let (next_state, next_effects) = transition(
                state,
                Event::ChainEvent {
                    chain,
                    event: kernel::Event::HeadObserved {
                        hash: padding_hash,
                        parent_hash,
                        logs_bloom: crate::Bloom::default(),
                        number: block_number_for(padding_hash),
                    },
                },
            );
            state = next_state;
            effects = next_effects;
            parent_hash = padding_hash;
        }
        (state, effects)
    }

    fn hash_for_index(index: usize) -> BlockHash {
        hash(u8::try_from(index).expect("test block index must fit in u8"))
    }

    fn assert_single_header_request_chain_effect(
        effects: &[Effect],
        chain: ChainKey,
        block_hash: BlockHash,
    ) {
        let matching_effects = effects
            .iter()
            .filter(|effect| {
                matches!(
                    effect,
                    Effect::ChainEffect {
                        chain: effect_chain,
                        effect:
                            kernel::Effect::Request(
                                crate::AnyIssuedRequest::BlockHeader(request),
                            ),
                    } if *effect_chain == chain && request.request_payload.block_hash == block_hash
                )
            })
            .count();

        assert_eq!(matching_effects, 1);
    }

    fn single_header_request_id(
        effects: &[Effect],
        chain: ChainKey,
        block_hash: BlockHash,
    ) -> crate::RequestId<crate::GetBlockHeader> {
        let matching_effects = effects
            .iter()
            .filter_map(|effect| match effect {
                Effect::ChainEffect {
                    chain: effect_chain,
                    effect: kernel::Effect::Request(crate::AnyIssuedRequest::BlockHeader(request)),
                } if *effect_chain == chain && request.request_payload.block_hash == block_hash => {
                    Some(request.request_id)
                }
                _ => None,
            })
            .collect::<Vec<_>>();

        assert_eq!(matching_effects.len(), 1);
        matching_effects[0]
    }

    fn assert_single_log_request_chain_effect(
        effects: &[Effect],
        chain: ChainKey,
        block_hash: BlockHash,
    ) -> crate::RequestId<crate::GetBlockLogs> {
        let matching_effects = effects
            .iter()
            .filter_map(|effect| match effect {
                Effect::ChainEffect {
                    chain: effect_chain,
                    effect: kernel::Effect::Request(crate::AnyIssuedRequest::BlockLogs(request)),
                } if *effect_chain == chain && request.request_payload.block_hash == block_hash => {
                    Some(request.request_id)
                }
                _ => None,
            })
            .collect::<Vec<_>>();

        assert_eq!(matching_effects.len(), 1);
        matching_effects[0]
    }
}
