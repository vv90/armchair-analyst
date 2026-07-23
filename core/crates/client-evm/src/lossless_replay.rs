//! Lossless (Tier A) evaluation of an [`ExecutionPlan`] against authoritative U256 pool state
//! (`optimization.md` Step 12).
//!
//! The optimization crate's f32 [`optimization::replay_plan`] replays a plan against the *projected*
//! f32 reserves — already a lossy downcast of the on-chain integers, so it can only prove the optimizer
//! self-consistent. This module is the integer twin: the same weight-driven sequential replay, but over
//! [`LosslessPool`] entries built straight from [`PoolState`]'s U256 virtual reserves, so the f32
//! downcast error is gone entirely. Within the current tick a Uniswap v3/v4 pool *is exactly* a
//! constant-product AMM over its virtual reserves; `swap_limit` (the tick boundary) caps each hop and
//! flags, via [`LosslessOutcome::hit_tick_limit`], when a hop left that exact regime.
//!
//! Semantics are pinned to the f32 oracle (`replay::replay_steps`), which differentially validates this
//! module: each stage reads its start-of-stage balance snapshot and writes a fresh next-stage map;
//! steps run in plan order; the book is mutable, so a pool reused across stages sees moved reserves —
//! same-direction reuse consolidates to the single-net-swap value (constant-product path independence)
//! and opposite-direction reuse pays its true fee-losing cost. Reserve mutation follows the oracle's
//! convention of adding the *pre-fee* input to the input-side reserve; unlike on-chain v3 (where fees
//! do not accrete into virtual liquidity) that skews conservative — a later same-direction hop sees
//! slightly worse reserves — so the gate never overstates. Remaining fee/rounding deltas versus
//! on-chain `SwapMath` are a few wei (a later fidelity tier), irrelevant to a profitability gate.
//!
//! A [`StepKind::Bridge`] step is a raw-unit 1:1 transfer — the integer image of the oracle's
//! zero-cost token-unit pass. That identity only holds when both endpoints have equal decimals;
//! `verify_plan` (the kernel caller) gates on verified-equal decimals before invoking this fold, so
//! the precondition is established outside rather than threaded through as a resolver.

use std::collections::HashMap;
use std::collections::hash_map::Entry;

use alloy::primitives::{U256, U512};
use optimization::{ExecutionPlan, StepKind};
use thiserror::Error;

use crate::ChainKey;
use crate::kernel::pool_registry::PoolMetadata;
use crate::kernel::token_registry::TokenAddress;
use crate::pool_state::{PoolRef, PoolState, PoolStateError};

/// Uniswap fee denominator: `fee_pips` is parts-per-million of the input taken as fee.
const PIPS_DENOMINATOR: u32 = 1_000_000;

/// Fixed-point denominator for applying an f32 plan `weight` to a U256 balance.
const WEIGHT_SCALE: u64 = 1_000_000_000;

/// One pool's live U256 reserves during a lossless replay — the integer twin of the f32 replay's
/// `PoolBookEntry`. Both swap directions share this single entry; a swap mutates it in place so a later
/// swap through the same pool sees the moved reserves.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LosslessPool {
    pub token0: TokenAddress,
    pub token1: TokenAddress,
    /// `virtual_reserve_x` — the token0 side.
    pub reserve0: U256,
    /// `virtual_reserve_y` — the token1 side.
    pub reserve1: U256,
    pub fee_pips: u32,
    /// Cap on the after-fee token0 input before the price leaves the current tick range.
    pub swap_limit_0: U256,
    /// Cap on the after-fee token1 input before the price leaves the current tick range.
    pub swap_limit_1: U256,
}

/// The terminal result of a lossless replay: the exact `init_asset` amount the route returns, and
/// whether any hop was clamped at its tick boundary (beyond which constant product is no longer the
/// exact in-tick math, so `output` is a conservative lower fidelity bound rather than exact).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LosslessOutcome {
    pub output: U256,
    pub hit_tick_limit: bool,
}

/// Surfaced when a plan and the resolver's view of the market do not belong together. Mirrors
/// `optimization::ReplayError`; keeps the replay panic-free rather than indexing into a missing entry.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum LosslessReplayError {
    /// The plan swaps through a pool the resolver could not produce a [`LosslessPool`] for.
    #[error("plan references a pool the resolver could not produce a lossless state for")]
    PoolNotFound,
    /// A step's `token_in` is neither token of the pool it names.
    #[error("step's input token is not one of its pool's two tokens")]
    TokenNotInPool,
}

/// The realized result of one swap hop.
struct Hop {
    out: U256,
    hit_limit: bool,
}

impl LosslessPool {
    /// Builds the lossless entry from the kernel's authoritative pool state and verified metadata —
    /// the same derivation the optimizer projection performs (`pool_reserve_values`), minus the f32
    /// downcast. Fails only where `swap_limit_x/y` does (inconsistent tick/price state).
    pub fn from_pool_state(
        state: &PoolState,
        metadata: &PoolMetadata,
        chain: ChainKey,
    ) -> Result<LosslessPool, PoolStateError> {
        let tick_spacing = metadata.fee.tick_spacing();
        Ok(LosslessPool {
            token0: TokenAddress(metadata.token0, chain),
            token1: TokenAddress(metadata.token1, chain),
            reserve0: state.virtual_reserve_x(),
            reserve1: state.virtual_reserve_y(),
            fee_pips: metadata.fee.pips(),
            swap_limit_0: state.swap_limit_x(tick_spacing)?,
            swap_limit_1: state.swap_limit_y(tick_spacing)?,
        })
    }

    /// Exact integer constant-product swap of `amount_in` of `token_in` for the other token, mutating
    /// the entry — the U256 twin of the f32 `PoolBookEntry::swap`: capped-after-fee input,
    /// `y·u/(x+u)`, pre-fee input added to the input-side reserve. All intermediates are U512-widened,
    /// every narrowing is bounded by construction (`after_fee ≤ amount_in`, `out ≤ y`), and reserve
    /// mutation saturates, so the path cannot panic or overflow.
    fn swap(
        &mut self,
        token_in: TokenAddress,
        amount_in: U256,
    ) -> Result<Hop, LosslessReplayError> {
        let forward = token_in == self.token0;
        let (x, y, limit) = if forward {
            (self.reserve0, self.reserve1, self.swap_limit_0)
        } else if token_in == self.token1 {
            (self.reserve1, self.reserve0, self.swap_limit_1)
        } else {
            return Err(LosslessReplayError::TokenNotInPool);
        };

        // A malformed fee of 100% or more keeps nothing: after_fee = 0, out = 0, never a panic.
        let keep = PIPS_DENOMINATOR.saturating_sub(self.fee_pips);
        let after_fee =
            U256::from(U512::from(amount_in) * U512::from(keep) / U512::from(PIPS_DENOMINATOR));

        let hit_limit = after_fee > limit;
        let capped = if hit_limit { limit } else { after_fee };

        let denominator = U512::from(x) + U512::from(capped);
        let out = if denominator.is_zero() {
            U256::ZERO
        } else {
            U256::from(U512::from(y) * U512::from(capped) / denominator)
        };

        if forward {
            self.reserve0 = self.reserve0.saturating_add(amount_in);
            self.reserve1 = self.reserve1.saturating_sub(out);
        } else {
            self.reserve1 = self.reserve1.saturating_add(amount_in);
            self.reserve0 = self.reserve0.saturating_sub(out);
        }

        Ok(Hop { out, hit_limit })
    }
}

/// Applies an f32 plan `weight` to a U256 balance as a fixed-point fraction. Rounding strands at most
/// dust (never overspends) — the same "fraction of realized balance" property the f32 replay relies
/// on. A `NaN` weight survives `clamp` but saturates to `0` in the float-to-int cast, routing nothing.
fn weighted_amount(balance: U256, weight: f32) -> U256 {
    let scaled = (f64::from(weight.clamp(0.0, 1.0)) * WEIGHT_SCALE as f64).round() as u64;
    let scaled = scaled.min(WEIGHT_SCALE);
    U256::from(U512::from(balance) * U512::from(scaled) / U512::from(WEIGHT_SCALE))
}

/// Replays `plan` sequentially against lossless pool states, starting with the whole `entry_amount`
/// (raw integer units of the plan's `init_asset`), and returns the exact terminal `init_asset` amount.
///
/// The book is built from the **resolver**, one call per distinct pool the plan swaps through —
/// O(pools in the plan), never the whole market. `resolve` returns `None` when a pool cannot be turned
/// into a [`LosslessPool`] (absent from the freshest state, or its state is inconsistent), which
/// surfaces as [`LosslessReplayError::PoolNotFound`].
///
/// `entry_amount` is deliberately independent of the plan's advisory f32 `entry_amount`: revalidation
/// may size the entry differently, and optimality is size-dependent — the caller owns keeping the two
/// consistent when comparing against the optimizer's claimed profit.
pub fn replay_plan_lossless(
    plan: &ExecutionPlan<PoolRef, TokenAddress>,
    resolve: impl Fn(PoolRef) -> Option<LosslessPool>,
    entry_amount: U256,
) -> Result<LosslessOutcome, LosslessReplayError> {
    let mut book: HashMap<PoolRef, LosslessPool> = HashMap::new();
    for step in &plan.steps {
        let StepKind::Swap(pool) = step.kind else {
            continue;
        };
        if let Entry::Vacant(vacant) = book.entry(pool) {
            vacant.insert(resolve(pool).ok_or(LosslessReplayError::PoolNotFound)?);
        }
    }

    let mut balances: HashMap<TokenAddress, U256> = HashMap::new();
    balances.insert(plan.init_asset, entry_amount);
    let mut hit_tick_limit = false;

    let Some(max_stage) = plan.steps.iter().map(|step| step.stage).max() else {
        // No steps to route: the entry never leaves the init asset.
        return Ok(LosslessOutcome {
            output: entry_amount,
            hit_tick_limit: false,
        });
    };

    for stage in 0..=max_stage {
        let mut next: HashMap<TokenAddress, U256> = HashMap::new();
        for step in plan.steps.iter().filter(|step| step.stage == stage) {
            let available = balances.get(&step.token_in).copied().unwrap_or(U256::ZERO);
            let amount_in = weighted_amount(available, step.weight);
            if amount_in.is_zero() {
                continue;
            }
            match step.kind {
                StepKind::Swap(pool) => {
                    let entry = book
                        .get_mut(&pool)
                        .ok_or(LosslessReplayError::PoolNotFound)?;
                    let hop = entry.swap(step.token_in, amount_in)?;
                    hit_tick_limit |= hop.hit_limit;
                    let slot = next.entry(step.token_out).or_insert(U256::ZERO);
                    *slot = slot.saturating_add(hop.out);
                }
                // A well-formed plan only carries a token to itself; mirror the oracle and drop the
                // share of a malformed cross-token carry rather than fabricating a conversion.
                StepKind::Carry if step.token_in == step.token_out => {
                    let slot = next.entry(step.token_out).or_insert(U256::ZERO);
                    *slot = slot.saturating_add(amount_in);
                }
                StepKind::Carry => {}
                // Raw-unit 1:1 transfer across a configured bridge pair, mirroring the oracle's
                // zero-cost token-unit pass. Only correct for equal-decimal endpoints, which
                // `verify_plan` gates before this fold runs; the token-unit f32 oracle needs no gate.
                StepKind::Bridge => {
                    let slot = next.entry(step.token_out).or_insert(U256::ZERO);
                    *slot = slot.saturating_add(amount_in);
                }
            }
        }
        balances = next;
    }

    Ok(LosslessOutcome {
        output: balances
            .get(&plan.init_asset)
            .copied()
            .unwrap_or(U256::ZERO),
        hit_tick_limit,
    })
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use alloy::primitives::{Address, U160, aliases::I24};
    use optimization::{ExecutableStep, PoolReserves, VirtualReserveValues, replay_plan};
    use proptest::prelude::*;

    use crate::kernel::pool_registry::{PoolFee, UniswapV3Fee};
    use crate::kernel::token_registry::TokenDecimals;
    use crate::utils::u256_token_amount_to_f32;

    use super::*;

    const FEE_PIPS: u32 = 3000;

    /// Large enough to never clamp in tests, small enough to stay f32-representable for the oracle.
    fn no_limit() -> U256 {
        U256::from(10u8).pow(U256::from(30u8))
    }

    fn token(byte: u8) -> TokenAddress {
        TokenAddress(Address::with_last_byte(byte), ChainKey::Ethereum)
    }

    fn pool_ref(byte: u8) -> PoolRef {
        PoolRef::uniswap_v3(Address::with_last_byte(byte), ChainKey::Ethereum)
    }

    fn pool(
        token0: TokenAddress,
        token1: TokenAddress,
        reserve0: u128,
        reserve1: u128,
    ) -> LosslessPool {
        LosslessPool {
            token0,
            token1,
            reserve0: U256::from(reserve0),
            reserve1: U256::from(reserve1),
            fee_pips: FEE_PIPS,
            swap_limit_0: no_limit(),
            swap_limit_1: no_limit(),
        }
    }

    fn swap_step(
        stage: usize,
        token_in: TokenAddress,
        token_out: TokenAddress,
        pool: PoolRef,
        weight: f32,
    ) -> ExecutableStep<PoolRef, TokenAddress> {
        ExecutableStep {
            stage,
            token_in,
            token_out,
            kind: StepKind::Swap(pool),
            weight,
            amount_in: 0.0,
            amount_out: 0.0,
        }
    }

    fn carry_step(
        stage: usize,
        token: TokenAddress,
        weight: f32,
    ) -> ExecutableStep<PoolRef, TokenAddress> {
        ExecutableStep {
            stage,
            token_in: token,
            token_out: token,
            kind: StepKind::Carry,
            weight,
            amount_in: 0.0,
            amount_out: 0.0,
        }
    }

    fn resolver(book: Vec<(PoolRef, LosslessPool)>) -> impl Fn(PoolRef) -> Option<LosslessPool> {
        move |pool| {
            book.iter()
                .find(|(id, _)| *id == pool)
                .map(|(_, entry)| entry.clone())
        }
    }

    fn to_f32(amount: U256) -> f32 {
        let zero_decimals = TokenDecimals::try_from_u256(U256::ZERO).unwrap();
        u256_token_amount_to_f32(amount, zero_decimals).unwrap()
    }

    /// Derives the f32 oracle reserves from the *same* U256 numbers (decimals 0), so the only
    /// divergence between the two replays is fp-vs-integer arithmetic plus weight fixed-point.
    fn f32_reserves_from(
        pool_id: PoolRef,
        entry: &LosslessPool,
    ) -> PoolReserves<PoolRef, TokenAddress> {
        PoolReserves {
            pool_id,
            token0: entry.token0,
            token1: entry.token1,
            value: VirtualReserveValues {
                token_0: to_f32(entry.reserve0),
                token_1: to_f32(entry.reserve1),
                fee_multiplier: 1.0 - entry.fee_pips as f32 / PIPS_DENOMINATOR as f32,
                max_swap_0: to_f32(entry.swap_limit_0),
                max_swap_1: to_f32(entry.swap_limit_1),
            },
        }
    }

    fn assert_relative_eq(lossless: U256, oracle: f32, tolerance: f32) {
        let lossless = to_f32(lossless);
        let scale = lossless.abs().max(oracle.abs()).max(1.0);
        assert!(
            (lossless - oracle).abs() <= tolerance * scale,
            "lossless {lossless} vs oracle {oracle} beyond relative tolerance {tolerance}"
        );
    }

    /// A mispriced two-pool loop: pool 1 prices B at 2 per A, pool 2 at 1 per A — buying B on pool 1
    /// and selling it on pool 2 roughly doubles the entry.
    fn mispriced_loop() -> (
        ExecutionPlan<PoolRef, TokenAddress>,
        Vec<(PoolRef, LosslessPool)>,
    ) {
        let (a, b) = (token(1), token(2));
        let (p1, p2) = (pool_ref(11), pool_ref(12));
        let book = vec![
            (p1, pool(a, b, 1_000_000_000, 2_000_000_000)),
            (p2, pool(a, b, 1_000_000_000, 1_000_000_000)),
        ];
        let plan = ExecutionPlan {
            init_asset: a,
            entry_amount: 100_000.0,
            steps: vec![swap_step(0, a, b, p1, 1.0), swap_step(1, b, a, p2, 1.0)],
        };
        (plan, book)
    }

    #[test]
    fn resolver_is_invoked_once_per_distinct_plan_pool() {
        let (plan, book) = mispriced_loop();
        let resolve = resolver(book);
        let calls = RefCell::new(Vec::new());
        let counting = |pool: PoolRef| {
            calls.borrow_mut().push(pool);
            resolve(pool)
        };

        replay_plan_lossless(&plan, counting, U256::from(100_000u64)).unwrap();

        let mut calls = calls.into_inner();
        calls.sort();
        assert_eq!(calls, vec![pool_ref(11), pool_ref(12)]);
    }

    #[test]
    fn constructor_matches_pool_state_derivations() {
        // The WBTC/USDC state pinned in pool_state.rs's tests (private there; PoolState fields are
        // pub, so the literal is duplicated rather than widening that constant's visibility).
        let state = PoolState {
            sqrt_price_x96: U160::from_limbs([17134602959287796597, 139272449984, 0]),
            liquidity: 50170120777514,
            tick: I24::from_limbs([69583]),
        };
        let metadata = PoolMetadata {
            token0: Address::with_last_byte(1),
            token1: Address::with_last_byte(2),
            fee: PoolFee::Tiered(UniswapV3Fee::Fee3000),
        };

        let entry = LosslessPool::from_pool_state(&state, &metadata, ChainKey::Ethereum).unwrap();

        assert_eq!(
            entry.token0,
            TokenAddress(metadata.token0, ChainKey::Ethereum)
        );
        assert_eq!(
            entry.token1,
            TokenAddress(metadata.token1, ChainKey::Ethereum)
        );
        assert_eq!(entry.reserve0, state.virtual_reserve_x());
        assert_eq!(entry.reserve1, state.virtual_reserve_y());
        assert_eq!(entry.fee_pips, 3000);
        assert_eq!(entry.swap_limit_0, state.swap_limit_x(60).unwrap());
        assert_eq!(entry.swap_limit_1, state.swap_limit_y(60).unwrap());
    }

    #[test]
    fn profitable_loop_matches_f32_oracle() {
        let (plan, book) = mispriced_loop();
        let reserves: Vec<_> = book
            .iter()
            .map(|(id, entry)| f32_reserves_from(*id, entry))
            .collect();

        let outcome = replay_plan_lossless(&plan, resolver(book), U256::from(100_000u64)).unwrap();
        let oracle = replay_plan(&plan, &reserves).unwrap();

        assert!(!outcome.hit_tick_limit);
        assert!(
            outcome.output > U256::from(100_000u64),
            "loop is profitable"
        );
        assert_relative_eq(outcome.output, oracle, 1e-3);
    }

    #[test]
    fn balanced_round_trip_loses_only_fees() {
        let (a, b) = (token(1), token(2));
        let p = pool_ref(11);
        let book = vec![(p, pool(a, b, 1_000_000_000, 1_000_000_000))];
        let plan = ExecutionPlan {
            init_asset: a,
            entry_amount: 100_000.0,
            steps: vec![swap_step(0, a, b, p, 1.0), swap_step(1, b, a, p, 1.0)],
        };
        let entry = U256::from(100_000u64);

        let outcome = replay_plan_lossless(&plan, resolver(book), entry).unwrap();

        assert!(outcome.output < entry, "no-arb round trip must lose fees");
        // Two 0.3% fee legs on a deep pool: the loss is small, not catastrophic.
        assert!(outcome.output > entry - U256::from(2_000u64));
    }

    #[test]
    fn missing_pool_is_pool_not_found() {
        let (plan, _) = mispriced_loop();

        let error = replay_plan_lossless(&plan, |_| None, U256::from(100u64)).unwrap_err();

        assert_eq!(error, LosslessReplayError::PoolNotFound);
    }

    #[test]
    fn foreign_token_is_token_not_in_pool() {
        let (a, b, c) = (token(1), token(2), token(3));
        let p = pool_ref(11);
        let book = vec![(p, pool(a, b, 1_000_000, 1_000_000))];
        let plan = ExecutionPlan {
            init_asset: c,
            entry_amount: 1_000.0,
            steps: vec![swap_step(0, c, b, p, 1.0)],
        };

        let error = replay_plan_lossless(&plan, resolver(book), U256::from(1_000u64)).unwrap_err();

        assert_eq!(error, LosslessReplayError::TokenNotInPool);
    }

    #[test]
    fn same_direction_reuse_consolidates_at_zero_fee() {
        // Constant-product path independence: swapping Δ₁ then Δ₂ through the same pool in the same
        // direction equals one swap of Δ₁+Δ₂ — exactly, at zero fee (the pre-fee reserve convention
        // breaks the exact identity for fee > 0). Integer floors may differ by a couple of units.
        let (a, b) = (token(1), token(2));
        let mut split = pool(a, b, 1_000_000_000, 3_000_000_000);
        split.fee_pips = 0;
        let mut whole = split.clone();

        let first = split.swap(a, U256::from(40_000u64)).unwrap();
        let second = split.swap(a, U256::from(60_000u64)).unwrap();
        let consolidated = whole.swap(a, U256::from(100_000u64)).unwrap();

        let split_total = first.out + second.out;
        let difference = split_total.abs_diff(consolidated.out);
        assert!(
            difference <= U256::from(2u8),
            "split {split_total} vs consolidated {} differ by more than floor slack",
            consolidated.out
        );
    }

    #[test]
    fn reuse_and_carry_plan_matches_f32_oracle() {
        // Stage 0 splits the entry: half swaps A→B through P, half carries. Stage 1 re-enters P in
        // the SAME direction with the carried half (the mutating book sees moved reserves) and
        // carries stage 0's B. Stage 2 sells all B back through a second pool.
        let (a, b) = (token(1), token(2));
        let (p, q) = (pool_ref(11), pool_ref(12));
        let book = vec![
            (p, pool(a, b, 1_000_000_000, 2_000_000_000)),
            (q, pool(a, b, 1_000_000_000, 1_000_000_000)),
        ];
        let plan = ExecutionPlan {
            init_asset: a,
            entry_amount: 100_000.0,
            steps: vec![
                swap_step(0, a, b, p, 0.5),
                carry_step(0, a, 0.5),
                swap_step(1, a, b, p, 1.0),
                carry_step(1, b, 1.0),
                swap_step(2, b, a, q, 1.0),
            ],
        };
        let reserves: Vec<_> = book
            .iter()
            .map(|(id, entry)| f32_reserves_from(*id, entry))
            .collect();

        let outcome = replay_plan_lossless(&plan, resolver(book), U256::from(100_000u64)).unwrap();
        let oracle = replay_plan(&plan, &reserves).unwrap();

        assert_relative_eq(outcome.output, oracle, 1e-3);
    }

    #[test]
    fn tick_limit_clamps_and_flags() {
        let (a, b) = (token(1), token(2));
        let p = pool_ref(11);
        let mut entry = pool(a, b, 1_000_000, 1_000_000);
        entry.swap_limit_0 = U256::from(500u64);
        let book = vec![(p, entry)];
        let plan = ExecutionPlan {
            init_asset: a,
            entry_amount: 10_000.0,
            steps: vec![swap_step(0, a, b, p, 1.0)],
        };

        // after_fee = 10_000 × 0.997 = 9_970 > 500, so the hop is capped at the limit:
        // out = 1_000_000 × 500 / (1_000_000 + 500).
        let expected = U256::from(1_000_000u64 * 500 / 1_000_500);
        let outcome = replay_plan_lossless(&plan, resolver(book), U256::from(10_000u64)).unwrap();

        assert!(outcome.hit_tick_limit);
        // The plan ends in B, not the init asset, so read the hop directly for the exact value.
        let mut direct = pool(a, b, 1_000_000, 1_000_000);
        direct.swap_limit_0 = U256::from(500u64);
        let hop = direct.swap(a, U256::from(10_000u64)).unwrap();
        assert!(hop.hit_limit);
        assert_eq!(hop.out, expected);
        assert_eq!(outcome.output, U256::ZERO);
    }

    fn bridge_step(
        stage: usize,
        token_in: TokenAddress,
        token_out: TokenAddress,
        weight: f32,
    ) -> ExecutableStep<PoolRef, TokenAddress> {
        ExecutableStep {
            stage,
            token_in,
            token_out,
            kind: StepKind::Bridge,
            weight,
            amount_in: 0.0,
            amount_out: 0.0,
        }
    }

    #[test]
    fn bridged_plan_matches_f32_oracle() {
        // A bridge hop is a raw-unit 1:1 transfer: swap a→b through P, bridge b→c, swap c→a
        // through Q. Differential against the f32 oracle on the same numbers.
        let (a, b, c) = (token(1), token(2), token(3));
        let (p, q) = (pool_ref(11), pool_ref(12));
        let book = vec![
            (p, pool(a, b, 1_000_000_000, 2_000_000_000)),
            (q, pool(c, a, 3_000_000_000, 1_500_000_000)),
        ];
        let plan = ExecutionPlan {
            init_asset: a,
            entry_amount: 100_000.0,
            steps: vec![
                swap_step(0, a, b, p, 1.0),
                bridge_step(1, b, c, 1.0),
                swap_step(2, c, a, q, 1.0),
            ],
        };
        let reserves: Vec<_> = book
            .iter()
            .map(|(id, entry)| f32_reserves_from(*id, entry))
            .collect();

        let outcome = replay_plan_lossless(&plan, resolver(book), U256::from(100_000u64)).unwrap();
        let oracle = replay_plan(&plan, &reserves).unwrap();

        assert_relative_eq(outcome.output, oracle, 1e-3);
    }

    #[test]
    fn malformed_cross_token_carry_drops_bridge_passes() {
        // The same cross-token round-trip shape: as malformed `Carry` steps the shares are lost
        // (never a fabricated conversion); as `Bridge` steps it is a zero-cost identity round trip.
        // Pins the two folds' invariants aligned with the f32 oracle's.
        let (a, b) = (token(1), token(2));
        let cross = |kind: StepKind<PoolRef>, stage, token_in, token_out| ExecutableStep {
            stage,
            token_in,
            token_out,
            kind,
            weight: 1.0,
            amount_in: 0.0,
            amount_out: 0.0,
        };
        let round_trip = |kind: StepKind<PoolRef>| ExecutionPlan {
            init_asset: a,
            entry_amount: 1_000.0,
            steps: vec![cross(kind, 0, a, b), cross(kind, 1, b, a)],
        };
        let entry = U256::from(1_000u64);

        let carried =
            replay_plan_lossless(&round_trip(StepKind::Carry), resolver(vec![]), entry).unwrap();
        assert_eq!(
            carried.output,
            U256::ZERO,
            "malformed cross-token carries must lose their share"
        );

        let bridged =
            replay_plan_lossless(&round_trip(StepKind::Bridge), resolver(vec![]), entry).unwrap();
        assert_eq!(bridged.output, entry, "a bridge round trip is zero-cost");
    }

    /// A fixed reuse-inclusive plan shape over two pools, driven by proptest inputs. Exercises
    /// same-direction reuse of P (stages 0 and 1), reuse of Q (stages 1 and 2), and carries.
    fn reuse_plan(split: f32) -> ExecutionPlan<PoolRef, TokenAddress> {
        let (a, b) = (token(1), token(2));
        let (p, q) = (pool_ref(11), pool_ref(12));
        ExecutionPlan {
            init_asset: a,
            entry_amount: 0.0,
            steps: vec![
                swap_step(0, a, b, p, split),
                carry_step(0, a, 1.0 - split),
                swap_step(1, a, b, p, 1.0),
                swap_step(1, b, a, q, 1.0),
                swap_step(2, b, a, q, 1.0),
                carry_step(2, a, 1.0),
            ],
        }
    }

    /// A fixed bridge-inclusive plan shape over three tokens and two pools (P: a↔b, Q: c↔a, with
    /// b→c only reachable via bridge). Exercises bridge hops feeding and fed by swaps, P reuse
    /// (stages 0 and 1), Q reuse (stages 2 and 3), and a carry.
    fn bridged_plan(split: f32) -> ExecutionPlan<PoolRef, TokenAddress> {
        let (a, b, c) = (token(1), token(2), token(3));
        let (p, q) = (pool_ref(11), pool_ref(12));
        ExecutionPlan {
            init_asset: a,
            entry_amount: 0.0,
            steps: vec![
                swap_step(0, a, b, p, split),
                carry_step(0, a, 1.0 - split),
                swap_step(1, a, b, p, 1.0),
                bridge_step(1, b, c, 1.0),
                swap_step(2, c, a, q, 1.0),
                bridge_step(2, b, c, 1.0),
                carry_step(3, a, 1.0),
                swap_step(3, c, a, q, 1.0),
            ],
        }
    }

    /// The two-pool book for [`bridged_plan`] (P: a↔b, Q: c↔a).
    fn bridged_book(
        reserve_p0: u128,
        reserve_p1: u128,
        reserve_q0: u128,
        reserve_q1: u128,
    ) -> Vec<(PoolRef, LosslessPool)> {
        let (a, b, c) = (token(1), token(2), token(3));
        vec![
            (pool_ref(11), pool(a, b, reserve_p0, reserve_p1)),
            (pool_ref(12), pool(c, a, reserve_q0, reserve_q1)),
        ]
    }

    proptest! {
        /// The mutating U256 twin tracks the mutating f32 oracle through pool reuse, not just
        /// unique-pool routes. Pool price ratios are bounded (each hop scales a balance by at most
        /// ~2×) so intermediate balances stay large: integer floor rounding loses at most one unit
        /// per hop and later hops amplify a lost unit by their rate, giving a small explicit
        /// absolute slack on top of the f32 relative error.
        #[test]
        fn differential_reuse_plan_tracks_oracle(
            reserve_p0 in 1_000_000_000u64..1_000_000_000_000,
            ratio_p in 0.5f64..2.0,
            reserve_q0 in 1_000_000_000u64..1_000_000_000_000,
            ratio_q in 0.5f64..2.0,
            entry in 100_000u64..1_000_000,
            split in 0.0f32..=1.0,
        ) {
            let reserve_p1 = (reserve_p0 as f64 * ratio_p) as u128;
            let reserve_q1 = (reserve_q0 as f64 * ratio_q) as u128;
            let (a, b) = (token(1), token(2));
            let book = vec![
                (pool_ref(11), pool(a, b, reserve_p0.into(), reserve_p1)),
                (pool_ref(12), pool(a, b, reserve_q0.into(), reserve_q1)),
            ];
            let mut plan = reuse_plan(split);
            plan.entry_amount = entry as f32;
            let reserves: Vec<_> = book
                .iter()
                .map(|(id, entry)| f32_reserves_from(*id, entry))
                .collect();

            let outcome = replay_plan_lossless(&plan, resolver(book), U256::from(entry)).unwrap();
            let oracle = replay_plan(&plan, &reserves).unwrap();

            let lossless = to_f32(outcome.output);
            let scale = lossless.abs().max(oracle.abs()).max(1.0);
            // ≤6 hops, ≤1 floored unit each, amplified ≤2× by each of ≤3 later hops: ≤48 units.
            prop_assert!(
                (lossless - oracle).abs() <= 1e-2 * scale + 48.0,
                "lossless {lossless} vs oracle {oracle}",
            );
        }

        /// The bridged plan shape tracks the oracle too: bridge hops are exact 1:1 in both domains
        /// (only the weight fixed-point differs), so the slack story matches the reuse case with the
        /// deeper 4-stage shape (≤8 hops, each floored unit amplified ≤2× by each later hop).
        #[test]
        fn differential_bridged_plan_tracks_oracle(
            reserve_p0 in 1_000_000_000u64..1_000_000_000_000,
            ratio_p in 0.5f64..2.0,
            reserve_q0 in 1_000_000_000u64..1_000_000_000_000,
            ratio_q in 0.5f64..2.0,
            entry in 100_000u64..1_000_000,
            split in 0.0f32..=1.0,
        ) {
            let reserve_p1 = (reserve_p0 as f64 * ratio_p) as u128;
            let reserve_q1 = (reserve_q0 as f64 * ratio_q) as u128;
            let book = bridged_book(reserve_p0.into(), reserve_p1, reserve_q0.into(), reserve_q1);
            let mut plan = bridged_plan(split);
            plan.entry_amount = entry as f32;
            let reserves: Vec<_> = book
                .iter()
                .map(|(id, entry)| f32_reserves_from(*id, entry))
                .collect();

            let outcome = replay_plan_lossless(&plan, resolver(book), U256::from(entry)).unwrap();
            let oracle = replay_plan(&plan, &reserves).unwrap();

            let lossless = to_f32(outcome.output);
            let scale = lossless.abs().max(oracle.abs()).max(1.0);
            prop_assert!(
                (lossless - oracle).abs() <= 1e-2 * scale + 96.0,
                "lossless {lossless} vs oracle {oracle}",
            );
        }

        /// Never panics and never conjures value out of a hop: for arbitrary reserves, limits, fee,
        /// and amount — including a full-range `weight` bit pattern in the plan path — the swap output
        /// is bounded by the output-side reserve.
        #[test]
        fn swap_is_panic_free_and_bounded(
            reserve0 in any::<u128>(),
            reserve1 in any::<u128>(),
            limit in any::<u128>(),
            fee_pips in any::<u32>(),
            amount in any::<u128>(),
        ) {
            let (a, b) = (token(1), token(2));
            let mut entry = LosslessPool {
                token0: a,
                token1: b,
                reserve0: U256::from(reserve0),
                reserve1: U256::from(reserve1),
                fee_pips,
                swap_limit_0: U256::from(limit),
                swap_limit_1: U256::from(limit),
            };

            let hop = entry.swap(a, U256::from(amount)).unwrap();

            prop_assert!(hop.out <= U256::from(reserve1));
        }

        /// The full replay is panic-free for arbitrary numeric inputs, including non-finite weights
        /// (`NaN`/infinities route as zero/full share respectively, never a crash).
        #[test]
        fn replay_is_panic_free(
            reserve_p0 in any::<u128>(),
            reserve_p1 in any::<u128>(),
            reserve_q0 in any::<u128>(),
            reserve_q1 in any::<u128>(),
            entry in any::<u128>(),
            split in any::<f32>(),
        ) {
            let (a, b) = (token(1), token(2));
            let book = vec![
                (pool_ref(11), pool(a, b, reserve_p0, reserve_p1)),
                (pool_ref(12), pool(a, b, reserve_q0, reserve_q1)),
            ];
            let plan = reuse_plan(split);

            let outcome = replay_plan_lossless(&plan, resolver(book), U256::from(entry));

            prop_assert!(outcome.is_ok());

            // Same numeric inputs through the bridge-inclusive shape (its book pairs c↔a for Q).
            let bridged_outcome = replay_plan_lossless(
                &bridged_plan(split),
                resolver(bridged_book(reserve_p0, reserve_p1, reserve_q0, reserve_q1)),
                U256::from(entry),
            );

            prop_assert!(bridged_outcome.is_ok());
        }
    }
}
