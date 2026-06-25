//! Decoded Uniswap v3 / v4 pool-state log events.
//!
//! A [`PoolLog`] is the minimal projection of an on-chain log that the kernel can fold into a
//! [`crate::PoolState`]: the emitting pool's protocol-tagged identity, the intra-block ordering key,
//! and the state-relevant event payload. Logs that do not affect pool state (v3 Collect/Flash/
//! cardinality/fee-protocol, or any unrecognized event) decode to `None` and are dropped at this
//! boundary.

use alloy::{
    primitives::{U160, aliases::I24},
    rpc::types::Log,
    sol_types::SolEvent,
};

use crate::PoolState;
use crate::pool_state::ProtocolPoolKey;
use crate::uniswap_v3::{Burn, Initialize, Mint, Swap};
use crate::uniswap_v4::{self, PoolId};

/// A state-relevant Uniswap v3 or v4 log, projected to exactly the fields pool-state derivation
/// needs. The pool's protocol-tagged identity is its own contract address (v3) or its
/// [`PoolId`](crate::uniswap_v4::PoolId) (v4).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PoolLog {
    pub pool: ProtocolPoolKey,
    pub log_index: u64,
    pub event: PoolLogEvent,
}

/// The state-relevant payload of a Uniswap v3 or v4 pool log.
///
/// `Swap` and `Initialize` carry an absolute post-event snapshot; `Mint` and `Burn` carry a
/// liquidity delta that applies only when the pool's current tick is inside `[tick_lower,
/// tick_upper)`. v4's single `ModifyLiquidity` event maps onto `Mint`/`Burn` by the sign of its
/// signed `liquidityDelta`. All other pool events are state-irrelevant and never produce a
/// `PoolLogEvent`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PoolLogEvent {
    Initialize {
        sqrt_price_x96: U160,
        tick: I24,
    },
    Swap {
        sqrt_price_x96: U160,
        tick: I24,
        liquidity: u128,
    },
    Mint {
        tick_lower: I24,
        tick_upper: I24,
        amount: u128,
    },
    Burn {
        tick_lower: I24,
        tick_upper: I24,
        amount: u128,
    },
}

/// Decodes a raw log into a [`PoolLog`], or `None` when the log is not a state-relevant pool event
/// or lacks an intra-block ordering index. Pure: dispatches on `topic0` against the v3 and v4 event
/// signature hashes (which are disjoint) and decodes the matching `SolEvent`. A v3 log is keyed by
/// its emitting address; a v4 log is keyed by the `PoolId` carried in its indexed `id` topic.
pub fn decode_pool_log(log: &Log) -> Option<PoolLog> {
    let log_index = log.log_index?;
    let data = &log.inner.data;
    let topic0 = *log.topic0()?;

    let (pool, event) = if topic0 == Swap::SIGNATURE_HASH {
        let swap = Swap::decode_log_data(data).ok()?;
        (
            ProtocolPoolKey::UniswapV3(log.address()),
            PoolLogEvent::Swap {
                sqrt_price_x96: swap.sqrtPriceX96,
                tick: swap.tick,
                liquidity: swap.liquidity,
            },
        )
    } else if topic0 == Initialize::SIGNATURE_HASH {
        let initialize = Initialize::decode_log_data(data).ok()?;
        (
            ProtocolPoolKey::UniswapV3(log.address()),
            PoolLogEvent::Initialize {
                sqrt_price_x96: initialize.sqrtPriceX96,
                tick: initialize.tick,
            },
        )
    } else if topic0 == Mint::SIGNATURE_HASH {
        let mint = Mint::decode_log_data(data).ok()?;
        (
            ProtocolPoolKey::UniswapV3(log.address()),
            PoolLogEvent::Mint {
                tick_lower: mint.tickLower,
                tick_upper: mint.tickUpper,
                amount: mint.amount,
            },
        )
    } else if topic0 == Burn::SIGNATURE_HASH {
        let burn = Burn::decode_log_data(data).ok()?;
        (
            ProtocolPoolKey::UniswapV3(log.address()),
            PoolLogEvent::Burn {
                tick_lower: burn.tickLower,
                tick_upper: burn.tickUpper,
                amount: burn.amount,
            },
        )
    } else if topic0 == uniswap_v4::Swap::SIGNATURE_HASH {
        let swap = uniswap_v4::Swap::decode_log_data(data).ok()?;
        (
            ProtocolPoolKey::UniswapV4(PoolId(swap.id)),
            PoolLogEvent::Swap {
                sqrt_price_x96: swap.sqrtPriceX96,
                tick: swap.tick,
                liquidity: swap.liquidity,
            },
        )
    } else if topic0 == uniswap_v4::Initialize::SIGNATURE_HASH {
        let initialize = uniswap_v4::Initialize::decode_log_data(data).ok()?;
        (
            ProtocolPoolKey::UniswapV4(PoolId(initialize.id)),
            PoolLogEvent::Initialize {
                sqrt_price_x96: initialize.sqrtPriceX96,
                tick: initialize.tick,
            },
        )
    } else if topic0 == uniswap_v4::ModifyLiquidity::SIGNATURE_HASH {
        let modify = uniswap_v4::ModifyLiquidity::decode_log_data(data).ok()?;
        // v4 collapses v3's Mint/Burn into one signed delta: a non-negative delta adds liquidity
        // (Mint), a negative delta removes it (Burn). The magnitude is bounded by the pool's
        // `uint128` liquidity; a value that does not fit drops the log to the GetPoolData fallback
        // rather than panicking.
        let amount = u128::try_from(modify.liquidityDelta.unsigned_abs()).ok()?;
        let event = if modify.liquidityDelta.is_negative() {
            PoolLogEvent::Burn {
                tick_lower: modify.tickLower,
                tick_upper: modify.tickUpper,
                amount,
            }
        } else {
            PoolLogEvent::Mint {
                tick_lower: modify.tickLower,
                tick_upper: modify.tickUpper,
                amount,
            }
        };
        (ProtocolPoolKey::UniswapV4(PoolId(modify.id)), event)
    } else {
        return None;
    };

    Some(PoolLog {
        pool,
        log_index,
        event,
    })
}

/// Folds an ordered run of a single pool's log events into its post-run [`PoolState`], starting
/// from `base` (the pool's latest snapshot before this run, if any).
///
/// `ordered` must be in `log_index` order. `Swap`/`Initialize` carry an absolute snapshot and set
/// state outright (`Initialize` implies liquidity 0), so they *seed* the run independent of `base`.
/// `Mint`/`Burn` adjust liquidity by their amount only when the running tick is inside
/// `[tick_lower, tick_upper)`; a delta reached before the run is seeded (no `base` and no prior
/// absolute event) has nothing to apply to and is skipped. The result is therefore the post-run
/// state whenever the run is seeded by `base` or by an absolute event within it — i.e. a block
/// containing a swap derives without any base. Returns `None` only when the run ends still unseeded
/// (a delta-only run with no `base`) or when a liquidity adjustment overflows; the caller then
/// leaves the block unsnapshotted and the existing `GetPoolData` path covers it. An empty run
/// returns `base`.
pub fn derive_pool_state(base: Option<&PoolState>, ordered: &[&PoolLogEvent]) -> Option<PoolState> {
    let mut running = base.cloned();

    for event in ordered {
        match event {
            PoolLogEvent::Swap {
                sqrt_price_x96,
                tick,
                liquidity,
            } => {
                running = Some(PoolState {
                    sqrt_price_x96: *sqrt_price_x96,
                    tick: *tick,
                    liquidity: *liquidity,
                });
            }
            PoolLogEvent::Initialize {
                sqrt_price_x96,
                tick,
            } => {
                running = Some(PoolState {
                    sqrt_price_x96: *sqrt_price_x96,
                    tick: *tick,
                    liquidity: 0,
                });
            }
            PoolLogEvent::Mint {
                tick_lower,
                tick_upper,
                amount,
            } => {
                if let Some(current) = &running {
                    running = Some(apply_liquidity_delta(
                        current,
                        *tick_lower,
                        *tick_upper,
                        |l| l.checked_add(*amount),
                    )?);
                }
            }
            PoolLogEvent::Burn {
                tick_lower,
                tick_upper,
                amount,
            } => {
                if let Some(current) = &running {
                    running = Some(apply_liquidity_delta(
                        current,
                        *tick_lower,
                        *tick_upper,
                        |l| l.checked_sub(*amount),
                    )?);
                }
            }
        }
    }

    running
}

/// Adjusts `current`'s liquidity via `delta` only when its tick is active in
/// `[tick_lower, tick_upper)`; out-of-range deltas leave the snapshot untouched. `None` when the
/// adjustment overflows.
fn apply_liquidity_delta(
    current: &PoolState,
    tick_lower: I24,
    tick_upper: I24,
    delta: impl FnOnce(u128) -> Option<u128>,
) -> Option<PoolState> {
    let liquidity = if tick_lower <= current.tick && current.tick < tick_upper {
        delta(current.liquidity)?
    } else {
        current.liquidity
    };

    Some(PoolState {
        sqrt_price_x96: current.sqrt_price_x96,
        tick: current.tick,
        liquidity,
    })
}

#[cfg(test)]
mod tests {
    use alloy::primitives::{
        Address, B256, I256, U256,
        aliases::U24,
        b256,
    };
    use proptest::prelude::*;

    use super::*;

    fn tick(value: i32) -> I24 {
        I24::try_from(value).expect("tick fixture in range")
    }

    fn v3(address: Address) -> ProtocolPoolKey {
        ProtocolPoolKey::UniswapV3(address)
    }

    fn log_with(address: Address, log_index: u64, event: impl SolEvent) -> Log {
        Log {
            inner: alloy::primitives::Log {
                address,
                data: event.encode_log_data(),
            },
            log_index: Some(log_index),
            ..Default::default()
        }
    }

    #[test]
    fn decodes_swap_to_absolute_snapshot_fields() {
        let address = Address::with_last_byte(0xAA);
        let log = log_with(
            address,
            7,
            Swap {
                sender: Address::with_last_byte(1),
                recipient: Address::with_last_byte(2),
                amount0: I256::try_from(-12345).unwrap(),
                amount1: I256::try_from(6789).unwrap(),
                sqrtPriceX96: U160::from(123456789u128),
                liquidity: 555_000u128,
                tick: tick(85176),
            },
        );

        assert_eq!(
            decode_pool_log(&log),
            Some(PoolLog {
                pool: v3(address),
                log_index: 7,
                event: PoolLogEvent::Swap {
                    sqrt_price_x96: U160::from(123456789u128),
                    tick: tick(85176),
                    liquidity: 555_000u128,
                },
            })
        );
    }

    #[test]
    fn decodes_initialize_to_absolute_snapshot_fields() {
        let address = Address::with_last_byte(0xBB);
        let log = log_with(
            address,
            0,
            Initialize {
                sqrtPriceX96: U160::from(999u128),
                tick: tick(-42),
            },
        );

        assert_eq!(
            decode_pool_log(&log),
            Some(PoolLog {
                pool: v3(address),
                log_index: 0,
                event: PoolLogEvent::Initialize {
                    sqrt_price_x96: U160::from(999u128),
                    tick: tick(-42),
                },
            })
        );
    }

    #[test]
    fn decodes_mint_to_liquidity_delta_fields() {
        let address = Address::with_last_byte(0xCC);
        let log = log_with(
            address,
            3,
            Mint {
                sender: Address::with_last_byte(1),
                owner: Address::with_last_byte(2),
                tickLower: tick(-600),
                tickUpper: tick(600),
                amount: 42u128,
                amount0: U256::from(10u64),
                amount1: U256::from(20u64),
            },
        );

        assert_eq!(
            decode_pool_log(&log),
            Some(PoolLog {
                pool: v3(address),
                log_index: 3,
                event: PoolLogEvent::Mint {
                    tick_lower: tick(-600),
                    tick_upper: tick(600),
                    amount: 42u128,
                },
            })
        );
    }

    #[test]
    fn decodes_burn_to_liquidity_delta_fields() {
        let address = Address::with_last_byte(0xDD);
        let log = log_with(
            address,
            9,
            Burn {
                owner: Address::with_last_byte(2),
                tickLower: tick(120),
                tickUpper: tick(240),
                amount: 7u128,
                amount0: U256::from(1u64),
                amount1: U256::from(2u64),
            },
        );

        assert_eq!(
            decode_pool_log(&log),
            Some(PoolLog {
                pool: v3(address),
                log_index: 9,
                event: PoolLogEvent::Burn {
                    tick_lower: tick(120),
                    tick_upper: tick(240),
                    amount: 7u128,
                },
            })
        );
    }

    #[test]
    fn ignores_state_irrelevant_collect_event() {
        use crate::uniswap_v3::Collect;

        let log = log_with(
            Address::with_last_byte(0xEE),
            1,
            Collect {
                owner: Address::with_last_byte(2),
                recipient: Address::with_last_byte(3),
                tickLower: tick(-60),
                tickUpper: tick(60),
                amount0: 1u128,
                amount1: 2u128,
            },
        );

        assert_eq!(decode_pool_log(&log), None);
    }

    #[test]
    fn ignores_log_without_intra_block_index() {
        let address = Address::with_last_byte(0xAA);
        let mut log = log_with(
            address,
            7,
            Swap {
                sender: Address::with_last_byte(1),
                recipient: Address::with_last_byte(2),
                amount0: I256::ZERO,
                amount1: I256::ZERO,
                sqrtPriceX96: U160::from(1u128),
                liquidity: 1u128,
                tick: tick(0),
            },
        );
        log.log_index = None;

        assert_eq!(decode_pool_log(&log), None);
    }

    // An arbitrary but fixed v4 pool id; the decode must surface it as the log's identity.
    const V4_POOL_ID: B256 =
        b256!("21c67e77068de97969ba93d4aab21826d33ca12bb9f565d8496e8fda8a82ca27");

    fn v4_manager() -> Address {
        uniswap_v4::ETHEREUM_UNISWAP_V4_POOL_MANAGER_ADDRESS
    }

    #[test]
    fn decodes_v4_swap_to_absolute_snapshot_keyed_by_pool_id() {
        // The emitting address is the singleton PoolManager; identity comes from the indexed id.
        let log = log_with(
            v4_manager(),
            4,
            uniswap_v4::Swap {
                id: V4_POOL_ID,
                sender: Address::with_last_byte(1),
                amount0: -12345i128,
                amount1: 6789i128,
                sqrtPriceX96: U160::from(123456789u128),
                liquidity: 555_000u128,
                tick: tick(85176),
                fee: U24::from(500u32),
            },
        );

        assert_eq!(
            decode_pool_log(&log),
            Some(PoolLog {
                pool: ProtocolPoolKey::UniswapV4(PoolId(V4_POOL_ID)),
                log_index: 4,
                event: PoolLogEvent::Swap {
                    sqrt_price_x96: U160::from(123456789u128),
                    tick: tick(85176),
                    liquidity: 555_000u128,
                },
            })
        );
    }

    #[test]
    fn decodes_v4_initialize_to_absolute_snapshot_keyed_by_pool_id() {
        let log = log_with(
            v4_manager(),
            0,
            uniswap_v4::Initialize {
                id: V4_POOL_ID,
                currency0: Address::ZERO,
                currency1: Address::with_last_byte(2),
                fee: U24::from(500u32),
                tickSpacing: tick(10),
                hooks: Address::ZERO,
                sqrtPriceX96: U160::from(999u128),
                tick: tick(-42),
            },
        );

        assert_eq!(
            decode_pool_log(&log),
            Some(PoolLog {
                pool: ProtocolPoolKey::UniswapV4(PoolId(V4_POOL_ID)),
                log_index: 0,
                event: PoolLogEvent::Initialize {
                    sqrt_price_x96: U160::from(999u128),
                    tick: tick(-42),
                },
            })
        );
    }

    #[test]
    fn decodes_v4_modify_liquidity_positive_delta_to_mint() {
        let log = log_with(
            v4_manager(),
            3,
            uniswap_v4::ModifyLiquidity {
                id: V4_POOL_ID,
                sender: Address::with_last_byte(1),
                tickLower: tick(-600),
                tickUpper: tick(600),
                liquidityDelta: I256::try_from(42).expect("fits int256"),
                salt: B256::ZERO,
            },
        );

        assert_eq!(
            decode_pool_log(&log),
            Some(PoolLog {
                pool: ProtocolPoolKey::UniswapV4(PoolId(V4_POOL_ID)),
                log_index: 3,
                event: PoolLogEvent::Mint {
                    tick_lower: tick(-600),
                    tick_upper: tick(600),
                    amount: 42u128,
                },
            })
        );
    }

    #[test]
    fn decodes_v4_modify_liquidity_negative_delta_to_burn() {
        let log = log_with(
            v4_manager(),
            9,
            uniswap_v4::ModifyLiquidity {
                id: V4_POOL_ID,
                sender: Address::with_last_byte(1),
                tickLower: tick(120),
                tickUpper: tick(240),
                liquidityDelta: I256::try_from(-7).expect("fits int256"),
                salt: B256::ZERO,
            },
        );

        assert_eq!(
            decode_pool_log(&log),
            Some(PoolLog {
                pool: ProtocolPoolKey::UniswapV4(PoolId(V4_POOL_ID)),
                log_index: 9,
                event: PoolLogEvent::Burn {
                    tick_lower: tick(120),
                    tick_upper: tick(240),
                    amount: 7u128,
                },
            })
        );
    }

    #[test]
    fn v4_modify_liquidity_delta_exceeding_u128_is_dropped() {
        // A magnitude past u128::MAX cannot be a real v4 liquidity delta; it must not panic.
        let too_big = I256::try_from(U256::from(u128::MAX)).expect("fits int256") + I256::ONE;
        let log = log_with(
            v4_manager(),
            1,
            uniswap_v4::ModifyLiquidity {
                id: V4_POOL_ID,
                sender: Address::with_last_byte(1),
                tickLower: tick(-1),
                tickUpper: tick(1),
                liquidityDelta: too_big,
                salt: B256::ZERO,
            },
        );

        assert_eq!(decode_pool_log(&log), None);
    }

    fn state(sqrt_price_x96: u128, tick_value: i32, liquidity: u128) -> PoolState {
        PoolState {
            sqrt_price_x96: U160::from(sqrt_price_x96),
            tick: tick(tick_value),
            liquidity,
        }
    }

    #[test]
    fn initialize_sets_liquidity_to_zero_regardless_of_base() {
        let base = state(99, 99, 9999);
        let initialize = PoolLogEvent::Initialize {
            sqrt_price_x96: U160::from(123u128),
            tick: tick(5),
        };

        assert_eq!(
            derive_pool_state(Some(&base), &[&initialize]),
            Some(state(123, 5, 0))
        );
    }

    #[test]
    fn in_range_mint_adds_liquidity() {
        let base = state(10, 50, 1000);
        let mint = PoolLogEvent::Mint {
            tick_lower: tick(0),
            tick_upper: tick(100),
            amount: 500,
        };

        assert_eq!(
            derive_pool_state(Some(&base), &[&mint]),
            Some(state(10, 50, 1500))
        );
    }

    #[test]
    fn mint_range_is_half_open_lower_inclusive_upper_exclusive() {
        // tick == tick_lower is active.
        let at_lower = state(10, 0, 1000);
        let mint = PoolLogEvent::Mint {
            tick_lower: tick(0),
            tick_upper: tick(100),
            amount: 500,
        };
        assert_eq!(
            derive_pool_state(Some(&at_lower), &[&mint]),
            Some(state(10, 0, 1500))
        );

        // tick == tick_upper is inactive (exclusive), so liquidity is untouched.
        let at_upper = state(10, 100, 1000);
        assert_eq!(derive_pool_state(Some(&at_upper), &[&mint]), Some(at_upper));
    }

    #[test]
    fn out_of_range_mint_is_a_noop_even_for_max_amount() {
        let base = state(10, 0, 1000);
        let mint = PoolLogEvent::Mint {
            tick_lower: tick(100),
            tick_upper: tick(200),
            amount: u128::MAX,
        };

        assert_eq!(derive_pool_state(Some(&base), &[&mint]), Some(base));
    }

    #[test]
    fn in_range_mint_overflow_returns_none() {
        let base = state(10, 0, u128::MAX);
        let mint = PoolLogEvent::Mint {
            tick_lower: tick(-1),
            tick_upper: tick(1),
            amount: 1,
        };

        assert_eq!(derive_pool_state(Some(&base), &[&mint]), None);
    }

    #[test]
    fn in_range_burn_underflow_returns_none() {
        let base = state(10, 0, 0);
        let burn = PoolLogEvent::Burn {
            tick_lower: tick(-1),
            tick_upper: tick(1),
            amount: 1,
        };

        assert_eq!(derive_pool_state(Some(&base), &[&burn]), None);
    }

    #[test]
    fn delta_only_run_without_base_returns_none() {
        let mint = PoolLogEvent::Mint {
            tick_lower: tick(-1),
            tick_upper: tick(1),
            amount: 1,
        };
        let burn = PoolLogEvent::Burn {
            tick_lower: tick(-1),
            tick_upper: tick(1),
            amount: 1,
        };

        // With no base and no absolute event to seed the run, there is nothing to apply the
        // deltas to, so the block is left underivable for the GetPoolData fallback.
        assert_eq!(derive_pool_state(None, &[&mint]), None);
        assert_eq!(derive_pool_state(None, &[&burn]), None);
        assert_eq!(derive_pool_state(None, &[&mint, &burn]), None);
    }

    #[test]
    fn trailing_absolute_event_seeds_a_run_with_leading_deltas() {
        let mint = PoolLogEvent::Mint {
            tick_lower: tick(-1),
            tick_upper: tick(1),
            amount: 1,
        };
        let swap = PoolLogEvent::Swap {
            sqrt_price_x96: U160::from(7u128),
            tick: tick(3),
            liquidity: 9,
        };

        // A leading delta with no base is skipped (it has nothing to apply to); a later absolute
        // event seeds the run, so the result is fully determined by that event regardless of base.
        assert_eq!(
            derive_pool_state(None, &[&mint, &swap]),
            Some(state(7, 3, 9))
        );
    }

    #[test]
    fn absolute_event_establishes_base_for_a_following_delta() {
        let swap = PoolLogEvent::Swap {
            sqrt_price_x96: U160::from(7u128),
            tick: tick(0),
            liquidity: 100,
        };
        let mint = PoolLogEvent::Mint {
            tick_lower: tick(-10),
            tick_upper: tick(10),
            amount: 50,
        };

        assert_eq!(
            derive_pool_state(None, &[&swap, &mint]),
            Some(state(7, 0, 150))
        );
    }

    #[test]
    fn empty_run_returns_base() {
        let base = state(10, 0, 1000);
        assert_eq!(derive_pool_state(Some(&base), &[]), Some(base));
        assert_eq!(derive_pool_state(None, &[]), None);
    }

    fn tick_value() -> impl Strategy<Value = I24> {
        (-8_388_608i32..=8_388_607i32).prop_map(|value| I24::try_from(value).unwrap())
    }

    fn any_pool_state() -> impl Strategy<Value = PoolState> {
        (any::<u128>(), tick_value(), any::<u128>()).prop_map(
            |(sqrt_price_x96, tick, liquidity)| PoolState {
                sqrt_price_x96: U160::from(sqrt_price_x96),
                tick,
                liquidity,
            },
        )
    }

    fn any_swap_event() -> impl Strategy<Value = PoolLogEvent> {
        (any::<u128>(), tick_value(), any::<u128>()).prop_map(|(sqrt, tick, liquidity)| {
            PoolLogEvent::Swap {
                sqrt_price_x96: U160::from(sqrt),
                tick,
                liquidity,
            }
        })
    }

    fn any_delta_event() -> impl Strategy<Value = PoolLogEvent> {
        (any::<bool>(), tick_value(), tick_value(), any::<u128>()).prop_map(
            |(is_mint, tick_lower, tick_upper, amount)| {
                if is_mint {
                    PoolLogEvent::Mint {
                        tick_lower,
                        tick_upper,
                        amount,
                    }
                } else {
                    PoolLogEvent::Burn {
                        tick_lower,
                        tick_upper,
                        amount,
                    }
                }
            },
        )
    }

    fn is_absolute(event: &PoolLogEvent) -> bool {
        matches!(
            event,
            PoolLogEvent::Initialize { .. } | PoolLogEvent::Swap { .. }
        )
    }

    proptest! {
        #[test]
        fn swap_sets_absolute_state_regardless_of_base(
            base in prop::option::of(any_pool_state()),
            sqrt in any::<u128>(),
            tick in tick_value(),
            liquidity in any::<u128>(),
        ) {
            let swap = PoolLogEvent::Swap {
                sqrt_price_x96: U160::from(sqrt),
                tick,
                liquidity,
            };
            let expected = PoolState {
                sqrt_price_x96: U160::from(sqrt),
                tick,
                liquidity,
            };

            prop_assert_eq!(derive_pool_state(base.as_ref(), &[&swap]), Some(expected.clone()));
            // Re-applying the same absolute snapshot is idempotent.
            prop_assert_eq!(derive_pool_state(base.as_ref(), &[&swap, &swap]), Some(expected));
        }

        #[test]
        fn last_absolute_event_determines_snapshot(
            base in prop::option::of(any_pool_state()),
            first in any_swap_event(),
            second in any_swap_event(),
        ) {
            prop_assert_eq!(
                derive_pool_state(base.as_ref(), &[&first, &second]),
                derive_pool_state(None, &[&second]),
            );
        }

        #[test]
        fn a_run_with_an_absolute_event_derives_base_free(
            // Build a run guaranteed to contain at least one absolute event: a mix of leading
            // events, a mandatory swap, then a mix of trailing events.
            leading in prop::collection::vec(
                prop_oneof![any_delta_event(), any_swap_event()],
                0..4,
            ),
            anchor in any_swap_event(),
            trailing in prop::collection::vec(
                prop_oneof![any_delta_event(), any_swap_event()],
                0..4,
            ),
        ) {
            let run: Vec<PoolLogEvent> = leading
                .into_iter()
                .chain(std::iter::once(anchor))
                .chain(trailing)
                .collect();
            let ordered: Vec<&PoolLogEvent> = run.iter().collect();

            // No base required: the snapshot equals folding from the first absolute event, since
            // everything before it is overwritten by it.
            let first_absolute = ordered
                .iter()
                .position(|event| is_absolute(event))
                .expect("run contains the anchor swap");

            prop_assert_eq!(
                derive_pool_state(None, &ordered),
                derive_pool_state(None, &ordered[first_absolute..]),
            );

            // It derives to Some unless a trailing in-range liquidity delta overflows; absent that,
            // an absolute-seeded run is always derivable without a base.
            if derive_pool_state(None, &ordered[first_absolute..]).is_some() {
                prop_assert!(derive_pool_state(None, &ordered).is_some());
            }
        }

        #[test]
        fn mint_then_equal_burn_restores_base(
            base in any_pool_state(),
            tick_lower in tick_value(),
            tick_upper in tick_value(),
            amount in any::<u128>(),
        ) {
            let mint = PoolLogEvent::Mint { tick_lower, tick_upper, amount };
            let burn = PoolLogEvent::Burn { tick_lower, tick_upper, amount };

            // When the mint overflows in-range the run is underivable; the cancellation property
            // only concerns runs that derive.
            prop_assume!(derive_pool_state(Some(&base), &[&mint]).is_some());

            prop_assert_eq!(
                derive_pool_state(Some(&base), &[&mint, &burn]),
                Some(base.clone())
            );
        }
    }
}
