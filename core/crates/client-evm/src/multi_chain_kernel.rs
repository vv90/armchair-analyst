use std::{
    collections::{BTreeMap, HashMap},
    time::Duration,
};

use alloy::primitives::{BlockHash, U256};
use optimization::{Invertible, PoolReserves, VirtualReserveValues};
use thiserror::Error;

use crate::{
    PoolAddress, PoolMetadata, PoolState, PoolStateError, TokenAddress, TokenAmountConversionError,
    TokenDecimals, UniswapV3Fee, bootstrap, chain::ChainKey, kernel, u256_token_amount_to_f32,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChainStatus {
    Initializing,
    Active,
}

pub struct State {
    chains: BTreeMap<ChainKey, ChainLifecycle>,
}

enum ChainLifecycle {
    Bootstrapping(bootstrap::State),
    Active(kernel::State),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FinalizedRefreshPolicy {
    target_len: usize,
    retry_stride: usize,
}

const ETHEREUM_APPROX_FINALIZED_BLOCK_AGE: usize = 64;
const ETHEREUM_FINALIZED_RETENTION_MARGIN: usize = 8;
const ETHEREUM_FINALIZED_REFRESH_RETRY_STRIDE: usize = 8;

/// Pool-candidate discovery window scanned below the finalized anchor during bootstrap.
const ETHEREUM_BOOTSTRAP_LOOK_BACK_DEPTH: u64 = 64;
/// Reorg-prone blocks nearest the observed tip left out of the seeded block graph.
const ETHEREUM_BOOTSTRAP_TIP_TRIM: usize = 8;
/// Ticks after which bootstrap activates best-effort (or abandons before the anchor is known).
const ETHEREUM_BOOTSTRAP_DEADLINE_TICKS: u64 = 30;

#[derive(Clone, Debug, PartialEq)]
pub struct OptimizationPoolReserves {
    pub block_hash: BlockHash,
    pub reserves: Vec<PoolReserves<PoolAddress, TokenAddress>>,
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
        pool: PoolAddress,
        source: PoolStateError,
    },

    #[error("failed to calculate max swap 1 for pool {pool:?}: {source}")]
    SwapLimit1 {
        pool: PoolAddress,
        source: PoolStateError,
    },

    #[error("failed to convert {value:?} for pool {pool:?} token {token:?}: {source}")]
    AmountConversion {
        pool: PoolAddress,
        token: TokenAddress,
        value: PoolReserveValueKind,
        source: TokenAmountConversionError,
    },

    #[error("invalid tick spacing for pool {pool:?}: {tick_spacing}")]
    InvalidTickSpacing {
        pool: PoolAddress,
        tick_spacing: i32,
    },
}

impl State {
    /// Creates multi-chain state with one chain bootstrapping its recent canonical window.
    /// Added so runtimes seed an inner kernel from the bootstrap outcome instead of an empty finalized state.
    pub fn init(chain: ChainKey) -> (State, Vec<Effect>) {
        let mut chains = BTreeMap::new();
        let (bootstrap_state, bootstrap_effects) = bootstrap::init(bootstrap_policy(chain));

        chains.insert(chain, ChainLifecycle::Bootstrapping(bootstrap_state));

        (
            State { chains },
            wrap_bootstrap_effects(chain, bootstrap_effects),
        )
    }

    /// Reports whether a configured chain is initializing (bootstrapping) or active.
    /// Added so callers can observe wrapper readiness without exposing the inner lifecycle representation.
    pub fn status(&self, chain: ChainKey) -> Option<ChainStatus> {
        self.chains
            .get(&chain)
            .map(|chain_state| match chain_state {
                ChainLifecycle::Bootstrapping(_) => ChainStatus::Initializing,
                ChainLifecycle::Active(_) => ChainStatus::Active,
            })
    }
}

/// Projects the active Ethereum kernel state and a complete pool-state overlay into optimization reserves.
/// Added as the pure bridge from validated EVM pool state into the optimization crate's directional reserve model.
pub fn ethereum_pool_reserves_for_optimization(
    state: &State,
    update: &kernel::CompletePoolStateUpdate,
) -> Result<Option<OptimizationPoolReserves>, PoolReserveProjectionError> {
    let Some(ChainLifecycle::Active(chain_state)) = state.chains.get(&ChainKey::Ethereum) else {
        return Ok(None);
    };

    let mut reserves = Vec::new();

    for (pool, pool_state) in sorted_pool_states_for_projection(
        chain_state.finalized_pool_snapshots(),
        &update.pool_states,
    ) {
        let Some((token0, token1, fee, token0_decimals, token1_decimals)) =
            projection_metadata(chain_state, pool)
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

    Ok(Some(OptimizationPoolReserves {
        block_hash: update.block_hash,
        reserves,
    }))
}

/// Merges finalized snapshots with update snapshots and returns pools in deterministic order.
/// Added so projections include the latest known state per pool while keeping output stable for model layout and tests.
fn sorted_pool_states_for_projection<'a>(
    finalized_pool_states: &'a HashMap<PoolAddress, PoolState>,
    update_pool_states: &'a HashMap<PoolAddress, PoolState>,
) -> Vec<(PoolAddress, &'a PoolState)> {
    let mut pool_states = finalized_pool_states
        .iter()
        .map(|(pool, pool_state)| (*pool, pool_state))
        .collect::<HashMap<_, _>>();

    pool_states.extend(
        update_pool_states
            .iter()
            .map(|(pool, pool_state)| (*pool, pool_state)),
    );

    let mut pool_states = pool_states.into_iter().collect::<Vec<_>>();
    pool_states.sort_by_key(|(pool, _)| *pool);
    pool_states
}

/// Collects verified pool metadata and token decimals needed for one pool projection.
/// Added so reserve generation can pause on incomplete registry data instead of emitting partially validated reserves.
fn projection_metadata(
    chain_state: &kernel::State,
    pool: PoolAddress,
) -> Option<(
    TokenAddress,
    TokenAddress,
    UniswapV3Fee,
    TokenDecimals,
    TokenDecimals,
)> {
    let PoolMetadata {
        token0,
        token1,
        fee,
    } = chain_state.verified_pool_metadata(pool)?;
    let token0 = TokenAddress(*token0);
    let token1 = TokenAddress(*token1);
    let token0_decimals = chain_state.verified_token_metadata(token0)?.decimals;
    let token1_decimals = chain_state.verified_token_metadata(token1)?.decimals;

    Some((token0, token1, *fee, token0_decimals, token1_decimals))
}

/// Converts one pool state into scaled virtual reserves, fee multiplier, and swap caps.
/// Added to keep Uniswap math, token scaling, and projection error context in one isolated pure step.
fn pool_reserve_values(
    pool: PoolAddress,
    pool_state: &PoolState,
    fee: UniswapV3Fee,
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
    let tick_spacing = tick_spacing_for_pool(pool, fee)?;
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

/// Derives a projection-ready tick spacing from the verified fee tier.
/// Added to keep the fee-tier boundary typed before calling pool-state swap-limit math.
fn tick_spacing_for_pool(
    pool: PoolAddress,
    fee: UniswapV3Fee,
) -> Result<u16, PoolReserveProjectionError> {
    let tick_spacing = fee.tick_spacing();

    u16::try_from(tick_spacing)
        .map_err(|_| PoolReserveProjectionError::InvalidTickSpacing { pool, tick_spacing })
}

/// Scales a raw on-chain token amount into the optimizer's `f32` reserve representation.
/// Added so conversion failures carry the pool, token, and reserve field being projected.
fn convert_pool_amount(
    pool: PoolAddress,
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

pub enum Event {
    FinalizedHeaderReceived {
        chain: ChainKey,
        block_hash: BlockHash,
    },
    FinalizedHeaderUnavailable {
        chain: ChainKey,
    },
    ChainEvent {
        chain: ChainKey,
        event: kernel::Event,
    },
    BootstrapEvent {
        chain: ChainKey,
        event: bootstrap::Event,
    },
    Tick,
}

pub enum Subscription {
    NewHeadsSubscription(ChainKey),
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
        Event::ChainEvent { chain, event } => chain_event(state, chain, event),
        Event::BootstrapEvent { chain, event } => bootstrap_event(state, chain, event),
        Event::Tick => tick(state),
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
    let mut chains = state.chains;

    match chains.remove(&chain) {
        Some(ChainLifecycle::Active(chain_state)) => {
            let (chain_state, effects) = kernel::transition(
                chain_state,
                kernel::Event::FinalizedBlockObserved { block_hash },
            );
            chains.insert(chain, ChainLifecycle::Active(chain_state));
            (
                State { chains },
                effects
                    .into_iter()
                    .map(|effect| Effect::ChainEffect { chain, effect })
                    .collect(),
            )
        }
        Some(other) => {
            chains.insert(chain, other);
            (State { chains }, Vec::new())
        }
        None => (State { chains }, Vec::new()),
    }
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
    let mut chains = state.chains;

    match chains.remove(&chain) {
        Some(ChainLifecycle::Bootstrapping(bootstrap_state)) => {
            let (lifecycle, effects) = advance_bootstrap(chain, bootstrap_state, event);
            if let Some(lifecycle) = lifecycle {
                chains.insert(chain, lifecycle);
            }
            (State { chains }, effects)
        }
        Some(other) => {
            chains.insert(chain, other);
            (State { chains }, Vec::new())
        }
        None => (State { chains }, Vec::new()),
    }
}

/// Runs one bootstrap transition and maps its completion onto the chain lifecycle.
/// Added as the single place that turns a bootstrap outcome into an active seeded kernel (or drops
/// the chain), keeping both the event and tick paths consistent.
fn advance_bootstrap(
    chain: ChainKey,
    bootstrap_state: bootstrap::State,
    event: bootstrap::Event,
) -> (Option<ChainLifecycle>, Vec<Effect>) {
    let (bootstrap_state, effects) = bootstrap::transition(bootstrap_state, event);
    let mut effects = wrap_bootstrap_effects(chain, effects);

    match bootstrap::completion(&bootstrap_state) {
        Some(bootstrap::Completion::Ready(outcome)) => {
            let (chain_state, activation_effects) = activate_bootstrap_outcome(outcome);
            effects.extend(
                activation_effects
                    .into_iter()
                    .map(|effect| Effect::ChainEffect { chain, effect }),
            );
            (Some(ChainLifecycle::Active(chain_state)), effects)
        }
        Some(bootstrap::Completion::Abandoned) => (None, effects),
        None => (
            Some(ChainLifecycle::Bootstrapping(bootstrap_state)),
            effects,
        ),
    }
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
        .map(|block| (block.hash, block.parent_hash, block.candidates))
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

/// Forwards an inner kernel event to an active chain and wraps its effects with the chain key.
/// Added to preserve chain isolation while letting callers drive per-chain kernel events through one wrapper.
fn chain_event(state: State, chain: ChainKey, event: kernel::Event) -> (State, Vec<Effect>) {
    let mut chains = state.chains;

    match chains.remove(&chain) {
        Some(ChainLifecycle::Active(chain_state)) => {
            let before_len = chain_state.canonical_path_len_from_finalized();
            let (chain_state, effects) = kernel::transition(chain_state, event);
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

            chains.insert(chain, ChainLifecycle::Active(chain_state));

            (State { chains }, effects)
        }
        Some(existing_chain) => {
            chains.insert(chain, existing_chain);
            (State { chains }, Vec::new())
        }
        None => (State { chains }, Vec::new()),
    }
}

/// Advances active and bootstrapping chains and forwards any retry or scheduler effects they produce.
/// Added so a single global tick can drive request TTL handling for active kernels and the bootstrap
/// retry/deadline timers for chains still warming up.
fn tick(state: State) -> (State, Vec<Effect>) {
    let (chains, effects) = state.chains.into_iter().fold(
        (BTreeMap::new(), Vec::new()),
        |(mut chains, mut effects), (chain, chain_state)| {
            match chain_state {
                ChainLifecycle::Bootstrapping(bootstrap_state) => {
                    let (lifecycle, chain_effects) =
                        advance_bootstrap(chain, bootstrap_state, bootstrap::Event::Tick);
                    if let Some(lifecycle) = lifecycle {
                        chains.insert(chain, lifecycle);
                    }
                    effects.extend(chain_effects);
                }
                ChainLifecycle::Active(chain_state) => {
                    let (chain_state, chain_effects) =
                        kernel::transition(chain_state, kernel::Event::Tick);
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

    (State { chains }, effects)
}

fn finalized_refresh_policy(chain: ChainKey) -> FinalizedRefreshPolicy {
    match chain {
        ChainKey::Ethereum => FinalizedRefreshPolicy {
            target_len: ETHEREUM_APPROX_FINALIZED_BLOCK_AGE + ETHEREUM_FINALIZED_RETENTION_MARGIN,
            retry_stride: ETHEREUM_FINALIZED_REFRESH_RETRY_STRIDE,
        },
    }
}

fn bootstrap_policy(chain: ChainKey) -> bootstrap::BootstrapPolicy {
    match chain {
        ChainKey::Ethereum => bootstrap::BootstrapPolicy {
            look_back_depth: ETHEREUM_BOOTSTRAP_LOOK_BACK_DEPTH,
            tip_trim: ETHEREUM_BOOTSTRAP_TIP_TRIM,
            deadline_ticks: ETHEREUM_BOOTSTRAP_DEADLINE_TICKS,
        },
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use alloy::primitives::{Address, BlockHash, U160, U256, aliases::I24};
    use optimization::{Invertible, VirtualReserveValues};
    use proptest::prelude::*;

    use super::*;
    use crate::kernel;
    use crate::{
        PoolAddress, PoolMetadata, PoolState, TokenAddress, TokenAmountConversionError,
        TokenDecimals, TokenMetadata, TrustedPoolRegistry, UniswapV3Fee, u256_token_amount_to_f32,
    };

    #[test]
    fn init_requests_finalized_header_and_marks_chain_bootstrapping() {
        let chain = ChainKey::Ethereum;
        let (state, effects) = State::init(chain);

        assert_eq!(state.status(chain), Some(ChainStatus::Initializing));
        assert!(matches!(
            single_bootstrap_effect(&effects, chain),
            bootstrap::Effect::Request(
                crate::bootstrap::pending_requests::AnyIssuedRequest::FinalizedHeader(_)
            )
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

        let (_state, effects) = transition(
            state,
            Event::ChainEvent {
                chain,
                event: kernel::Event::HeadObserved {
                    hash: child_hash,
                    parent_hash: finalized_hash,
                },
            },
        );
        assert_single_log_request_chain_effect(&effects, chain, child_hash);
    }

    #[test]
    fn bootstrap_deadline_before_anchor_abandons_chain() {
        let chain = ChainKey::Ethereum;
        let (mut state, _effects) = State::init(chain);
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
        let (mut state, _effects) = State::init(chain);
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
                    hash: observed_hash,
                    parent_hash: missing_parent_hash,
                },
            },
        );

        assert_single_header_request_chain_effect(&effects, chain, missing_parent_hash);
    }

    #[test]
    fn tick_keeps_bootstrapping_chain_initializing_before_deadline() {
        let chain = ChainKey::Ethereum;
        let (state, _) = State::init(chain);

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
                    hash: observed_hash,
                    parent_hash: missing_parent_hash,
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
        let request_id = assert_single_log_request_chain_effect(
            &effects,
            chain,
            hash_for_index(policy.target_len),
        );

        let (_state, effects) = transition(
            state,
            Event::ChainEvent {
                chain,
                event: kernel::Event::BlockLogsReceived {
                    request_id,
                    logs: HashSet::new(),
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

        for block_index in policy.target_len + 1..policy.target_len + policy.retry_stride {
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

        for block_index in policy.target_len + 1..=policy.target_len + policy.retry_stride {
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
                        request_id,
                        hash: missing_hash,
                        parent_hash,
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
        let (state, effects) = transition(
            state,
            Event::ChainEvent {
                chain,
                event: kernel::Event::HeadObserved {
                    hash: compacted_hash,
                    parent_hash: finalized_hash,
                },
            },
        );
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
                    hash: old_branch_hash,
                    parent_hash: finalized_hash,
                },
            },
        );

        assert_single_header_request_chain_effect(&effects, chain, finalized_hash);
    }

    #[test]
    fn run_optimization_effect_carries_chain_neutral_input() {
        let input = OptimizationPoolReserves {
            block_hash: hash(1),
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
    fn pool_reserves_projection_returns_none_when_ethereum_chain_is_inactive() {
        let (state, _) = State::init(ChainKey::Ethereum);
        let update = kernel::CompletePoolStateUpdate {
            block_hash: hash(1),
            pool_states: HashMap::new(),
        };

        let reserves = ethereum_pool_reserves_for_optimization(&state, &update).unwrap();

        assert_eq!(reserves, None);
    }

    #[test]
    fn pool_reserves_projection_returns_empty_reserves_for_complete_empty_update() {
        let state = projection_state(hash(1), HashMap::new(), HashMap::new(), HashMap::new());
        let update = kernel::CompletePoolStateUpdate {
            block_hash: hash(2),
            pool_states: HashMap::new(),
        };

        let reserves = ethereum_pool_reserves_for_optimization(&state, &update)
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
            hash(1),
            HashMap::from([(pool, pool_state.clone())]),
            HashMap::from([(pool, pool_metadata(token0, token1, UniswapV3Fee::Fee3000))]),
            HashMap::from([(token0, token_metadata(18)), (token1, token_metadata(6))]),
        );
        let update = kernel::CompletePoolStateUpdate {
            block_hash: hash(2),
            pool_states: HashMap::new(),
        };

        let reserves = ethereum_pool_reserves_for_optimization(&state, &update)
            .unwrap()
            .unwrap();

        assert_eq!(reserves.reserves.len(), 2);
        assert_directional_pair(&reserves.reserves, pool, token0, token1, &pool_state);
    }

    #[test]
    fn pool_reserves_projection_update_snapshot_overwrites_finalized_snapshot() {
        let pool = pool(10);
        let token0 = token(1);
        let token1 = token(2);
        let finalized_pool_state = balanced_pool_state(1_000_000);
        let updated_pool_state = balanced_pool_state(2_000_000);
        let state = projection_state(
            hash(1),
            HashMap::from([(pool, finalized_pool_state)]),
            HashMap::from([(pool, pool_metadata(token0, token1, UniswapV3Fee::Fee3000))]),
            HashMap::from([(token0, token_metadata(18)), (token1, token_metadata(6))]),
        );
        let update = kernel::CompletePoolStateUpdate {
            block_hash: hash(2),
            pool_states: HashMap::from([(pool, updated_pool_state.clone())]),
        };

        let reserves = ethereum_pool_reserves_for_optimization(&state, &update)
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
        let state = projection_state(hash(1), HashMap::new(), HashMap::new(), HashMap::new());
        let update = kernel::CompletePoolStateUpdate {
            block_hash: hash(2),
            pool_states: HashMap::from([(pool, balanced_pool_state(1_000_000))]),
        };

        let reserves = ethereum_pool_reserves_for_optimization(&state, &update).unwrap();

        assert_eq!(reserves, None);
    }

    #[test]
    fn pool_reserves_projection_returns_none_when_token_metadata_is_missing() {
        let pool = pool(10);
        let token0 = token(1);
        let token1 = token(2);
        let state = projection_state(
            hash(1),
            HashMap::new(),
            HashMap::from([(pool, pool_metadata(token0, token1, UniswapV3Fee::Fee3000))]),
            HashMap::from([(token0, token_metadata(18))]),
        );
        let update = kernel::CompletePoolStateUpdate {
            block_hash: hash(2),
            pool_states: HashMap::from([(pool, balanced_pool_state(1_000_000))]),
        };

        let reserves = ethereum_pool_reserves_for_optimization(&state, &update).unwrap();

        assert_eq!(reserves, None);
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
            hash(1),
            HashMap::new(),
            HashMap::from([(pool, pool_metadata(token0, token1, UniswapV3Fee::Fee3000))]),
            HashMap::from([(token0, token_metadata(0)), (token1, token_metadata(0))]),
        );
        let update = kernel::CompletePoolStateUpdate {
            block_hash: hash(2),
            pool_states: HashMap::from([(pool, pool_state)]),
        };

        let error = ethereum_pool_reserves_for_optimization(&state, &update).unwrap_err();

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
            hash(1),
            HashMap::new(),
            HashMap::from([(pool, pool_metadata(token0, token1, UniswapV3Fee::Fee3000))]),
            HashMap::from([(token0, token_metadata(18)), (token1, token_metadata(6))]),
        );
        let update = kernel::CompletePoolStateUpdate {
            block_hash: hash(2),
            pool_states: HashMap::from([(pool, inconsistent_pool_state)]),
        };

        let error = ethereum_pool_reserves_for_optimization(&state, &update).unwrap_err();

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
            let update = kernel::CompletePoolStateUpdate {
                block_hash: update_hash,
                pool_states: pools
                    .iter()
                    .map(|(pool, _, _, pool_state)| (*pool, pool_state.clone()))
                    .collect(),
            };
            let state = projection_state(
                finalized_hash,
                HashMap::new(),
                pool_metadata,
                token_metadata,
            );

            let reserves = ethereum_pool_reserves_for_optimization(&state, &update)
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

    fn pool(value: u8) -> PoolAddress {
        PoolAddress(Address::with_last_byte(value))
    }

    fn token(value: u8) -> TokenAddress {
        TokenAddress(Address::with_last_byte(value))
    }

    fn pool_metadata(
        token0: TokenAddress,
        token1: TokenAddress,
        fee: UniswapV3Fee,
    ) -> PoolMetadata {
        PoolMetadata {
            token0: token0.0,
            token1: token1.0,
            fee,
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
        finalized_hash: BlockHash,
        finalized_snapshots: HashMap<PoolAddress, PoolState>,
        pool_metadata: HashMap<PoolAddress, PoolMetadata>,
        token_metadata: HashMap<TokenAddress, TokenMetadata>,
    ) -> State {
        let finalized_state = kernel::FinalizedState::with_pool_snapshots_for_test(
            finalized_hash,
            finalized_snapshots,
        );
        let pool_registry = TrustedPoolRegistry::new().with_metadata_results(
            pool_metadata
                .into_iter()
                .map(|(pool, metadata)| (crate::PoolCandidateAddress(pool.0), Ok(metadata)))
                .collect(),
        );
        let token_registry = crate::TokenRegistry::new().with_metadata_results(
            token_metadata
                .into_iter()
                .map(|(token, metadata)| (token, Ok(metadata)))
                .collect(),
        );
        let chain_state = kernel::State::for_pool_reserve_projection_test(
            finalized_state,
            pool_registry,
            token_registry,
        );

        State {
            chains: BTreeMap::from([(ChainKey::Ethereum, ChainLifecycle::Active(chain_state))]),
        }
    }

    fn assert_directional_pair(
        reserves: &[optimization::PoolReserves<PoolAddress, TokenAddress>],
        pool: PoolAddress,
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
            },
            AnyIssuedRequest::PoolMetadata(issued) => bootstrap::Event::PoolMetadataReceived {
                request_id: issued.request_id,
                metadata: HashMap::new(),
            },
            AnyIssuedRequest::TokenMetadata(issued) => bootstrap::Event::TokenMetadataReceived {
                request_id: issued.request_id,
                metadata: HashMap::new(),
            },
            AnyIssuedRequest::PoolData(issued) => bootstrap::Event::PoolDataReceived {
                request_id: issued.request_id,
                pools: HashMap::new(),
            },
        };

        Event::BootstrapEvent { chain, event }
    }

    /// Drives a freshly initialized chain through the full bootstrap handshake to an active kernel.
    /// Feeds empty responses, so the activated kernel is anchored at `finalized_hash` with no seed.
    fn drive_bootstrap_to_active(chain: ChainKey, finalized_hash: BlockHash) -> State {
        let (mut state, mut effects) = State::init(chain);

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
                event: kernel::Event::HeadObserved { hash, parent_hash },
            },
        )
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
