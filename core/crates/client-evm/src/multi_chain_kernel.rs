use std::collections::{BTreeMap, HashMap};

use alloy::primitives::{BlockHash, U256};
use optimization::{Invertible, PoolReserves, VirtualReserveValues};
use thiserror::Error;

use crate::{
    PoolAddress, PoolMetadata, PoolState, PoolStateError, TokenAddress, TokenAmountConversionError,
    TokenDecimals, UniswapV3Fee, chain::ChainKey, kernel, u256_token_amount_to_f32,
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
    Initializing,
    Active(kernel::State),
}

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
    /// Creates multi-chain state with one chain waiting for its finalized anchor.
    /// Added so runtimes can bootstrap an inner kernel only after the chain's finalized block is known.
    pub fn init(chain: ChainKey) -> (State, Vec<Effect>) {
        let mut chains = BTreeMap::new();

        if chains.contains_key(&chain) {
            return (State { chains }, Vec::new());
        }

        chains.insert(chain, ChainLifecycle::Initializing);

        (
            State { chains },
            vec![Effect::FetchFinalizedHeader { chain }],
        )
    }

    /// Reports whether a configured chain is initializing or active.
    /// Added so callers can observe wrapper readiness without exposing the inner lifecycle representation.
    pub fn status(&self, chain: ChainKey) -> Option<ChainStatus> {
        self.chains
            .get(&chain)
            .map(|chain_state| match chain_state {
                ChainLifecycle::Initializing => ChainStatus::Initializing,
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
    Tick,
}

pub enum Effect {
    FetchFinalizedHeader {
        chain: ChainKey,
    },
    ChainEffect {
        chain: ChainKey,
        effect: kernel::Effect,
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
        Event::Tick => tick(state),
    }
}

/// Handles finalized header observations for both bootstrap and active-chain compaction.
/// Added so the same finalized-header feed can initialize chains and later advance each inner kernel's finalized boundary.
fn finalized_header_received(
    state: State,
    chain: ChainKey,
    block_hash: BlockHash,
) -> (State, Vec<Effect>) {
    let mut chains = state.chains;

    match chains.remove(&chain) {
        Some(ChainLifecycle::Initializing) => {
            let finalized_state = kernel::FinalizedState::empty_at(block_hash);
            chains.insert(
                chain,
                ChainLifecycle::Active(kernel::State::init(finalized_state)),
            );
            (State { chains }, Vec::new())
        }
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
        None => (State { chains }, Vec::new()),
    }
}

/// Removes a chain that could not fetch the finalized header during bootstrap.
/// Added so failed initialization does not leave an unusable chain in the initializing lifecycle forever.
fn finalized_header_unavailable(state: State, chain: ChainKey) -> (State, Vec<Effect>) {
    let mut chains = state.chains;

    if matches!(chains.get(&chain), Some(ChainLifecycle::Initializing)) {
        chains.remove(&chain);
    }

    (State { chains }, Vec::new())
}

/// Forwards an inner kernel event to an active chain and wraps its effects with the chain key.
/// Added to preserve chain isolation while letting callers drive per-chain kernel events through one wrapper.
fn chain_event(state: State, chain: ChainKey, event: kernel::Event) -> (State, Vec<Effect>) {
    let mut chains = state.chains;

    match chains.remove(&chain) {
        Some(ChainLifecycle::Active(chain_state)) => {
            let (chain_state, effects) = kernel::transition(chain_state, event);
            chains.insert(chain, ChainLifecycle::Active(chain_state));

            (
                State { chains },
                effects
                    .into_iter()
                    .map(|effect| Effect::ChainEffect { chain, effect })
                    .collect(),
            )
        }
        Some(existing_chain) => {
            chains.insert(chain, existing_chain);
            (State { chains }, Vec::new())
        }
        None => (State { chains }, Vec::new()),
    }
}

/// Advances active chains and forwards any retry or scheduler effects they produce.
/// Added so a single global tick can drive request TTL handling across all configured chains.
fn tick(state: State) -> (State, Vec<Effect>) {
    let (chains, effects) = state.chains.into_iter().fold(
        (BTreeMap::new(), Vec::new()),
        |(mut chains, mut effects), (chain, chain_state)| {
            match chain_state {
                ChainLifecycle::Initializing => {
                    chains.insert(chain, ChainLifecycle::Initializing);
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
    fn init_requests_finalized_header_and_marks_chain_initializing() {
        let chain = ChainKey::Ethereum;
        let (state, effects) = State::init(chain);

        assert_eq!(state.status(chain), Some(ChainStatus::Initializing));
        assert_single_fetch_finalized_header_effect(&effects, chain);
    }

    #[test]
    fn finalized_header_received_for_initializing_chain_activates_chain() {
        let chain = ChainKey::Ethereum;
        let finalized_hash = hash(1);
        let child_hash = hash(2);
        let (state, _) = State::init(chain);

        let (state, effects) = transition(
            state,
            Event::FinalizedHeaderReceived {
                chain,
                block_hash: finalized_hash,
            },
        );

        assert_eq!(state.status(chain), Some(ChainStatus::Active));
        assert!(effects.is_empty());

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
    fn finalized_header_unavailable_for_initializing_chain_removes_chain() {
        let chain = ChainKey::Ethereum;
        let (state, _) = State::init(chain);

        let (state, effects) = transition(state, Event::FinalizedHeaderUnavailable { chain });

        assert_eq!(state.status(chain), None);
        assert!(effects.is_empty());
    }

    #[test]
    fn chain_event_for_inactive_chain_is_ignored() {
        let chain = ChainKey::Ethereum;
        let (state, _) = State::init(chain);
        let (state, _) = transition(state, Event::FinalizedHeaderUnavailable { chain });

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
        let (state, _) = State::init(chain);
        let (state, _) = transition(
            state,
            Event::FinalizedHeaderReceived {
                chain,
                block_hash: finalized_hash,
            },
        );

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
    fn tick_is_ignored_for_initializing_chains() {
        let chain = ChainKey::Ethereum;
        let (state, _) = State::init(chain);

        let (state, effects) = transition(state, Event::Tick);

        assert_eq!(state.status(chain), Some(ChainStatus::Initializing));
        assert!(effects.is_empty());
    }

    #[test]
    fn tick_routes_inner_effects_with_chain_key() {
        let chain = ChainKey::Ethereum;
        let finalized_hash = hash(1);
        let missing_parent_hash = hash(2);
        let observed_hash = hash(3);
        let (state, _) = State::init(chain);
        let (state, _) = transition(
            state,
            Event::FinalizedHeaderReceived {
                chain,
                block_hash: finalized_hash,
            },
        );
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
    fn finalized_header_received_for_active_chain_routes_to_inner_compaction() {
        let chain = ChainKey::Ethereum;
        let finalized_hash = hash(1);
        let compacted_hash = hash(2);
        let old_branch_hash = hash(3);
        let (state, _) = State::init(chain);
        let (state, _) = transition(
            state,
            Event::FinalizedHeaderReceived {
                chain,
                block_hash: finalized_hash,
            },
        );
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

    fn assert_single_fetch_finalized_header_effect(effects: &[Effect], chain: ChainKey) {
        assert_eq!(effects.len(), 1);
        assert!(matches!(
            &effects[0],
            Effect::FetchFinalizedHeader { chain: effect_chain }
                if *effect_chain == chain
        ));
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
                                crate::pending_requests::AnyIssuedRequest::BlockHeader(request),
                            ),
                    } if *effect_chain == chain && request.request_payload.block_hash == block_hash
                )
            })
            .count();

        assert_eq!(matching_effects, 1);
    }

    fn assert_single_log_request_chain_effect(
        effects: &[Effect],
        chain: ChainKey,
        block_hash: BlockHash,
    ) -> crate::pending_requests::RequestId<crate::pending_requests::GetBlockLogs> {
        let matching_effects =
            effects
                .iter()
                .filter_map(|effect| match effect {
                    Effect::ChainEffect {
                        chain: effect_chain,
                        effect:
                            kernel::Effect::Request(
                                crate::pending_requests::AnyIssuedRequest::BlockLogs(request),
                            ),
                    } if *effect_chain == chain
                        && request.request_payload.block_hash == block_hash =>
                    {
                        Some(request.request_id)
                    }
                    _ => None,
                })
                .collect::<Vec<_>>();

        assert_eq!(matching_effects.len(), 1);
        matching_effects[0]
    }
}
