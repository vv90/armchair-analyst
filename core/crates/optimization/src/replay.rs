//! Exact scalar replay of a trained model's extracted flows (`optimization.md` Step 12, slice 1).
//!
//! The differentiable [`crate::model::Model`] forward is a *soft, parallel* solution: it softmax-
//! splits each input token across every pool row and quotes each cell's constant-product output
//! against the **frozen** snapshot reserves, independently, then sums. It never sequences one swap's
//! effect onto the next. [`crate::model::Model::extract_flows`] reads that off faithfully, so simply
//! re-summing the flows reproduces `evaluate` — a tautology.
//!
//! [`replay_flows`] instead executes the trained routing as a *sequential* series of swaps against a
//! **mutating** reserve book, threading realized balances. Comparing its output to `evaluate` yields
//! the approximation error the Step 12 gate rejects candidates on: ~0 for deep, distinct pools (the
//! "executable" green light), and non-zero exactly where the parallel-on-original-reserves forward is
//! wrong — a pool reused across stages/directions, where sequential execution sees reserves the
//! forward ignored.
//!
//! The replay is **weight-driven**: it applies each cell's post-softmax share (`FlowRecord::weight`)
//! to the token balance actually on hand, not the recorded `amount_in` (which is pre-fee and computed
//! against original upstream balances). Because softmax is per-column, the shares over rows sum to one
//! for each input token, so `balance × weight` exactly partitions each token's realized balance — no
//! over-spend, no clamping. This is the `s_{t+1} = T_a(s_t)` transition later phases generalize, and
//! it holds one flow set fixed so the same call can replay it against *perturbed* reserves (Step 8).
//!
//! This module is also the intended home for the scalar `quote`/`apply_swap` engine that the roadmap
//! keeps **out** of the differentiable core (a scalar dispatch can't be batched or backpropagated).

use std::collections::HashMap;
use std::hash::Hash;

use thiserror::Error;

use crate::model::FlowRecord;
use crate::plan::{ExecutionPlan, StepKind};
use crate::pool_reserves::PoolReserves;

/// Surfaced when a flow set and a reserve snapshot do not belong together. Neither variant can occur
/// for flows extracted from a model built on the same snapshot; they keep the replay panic-free
/// rather than indexing into a missing entry.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum ReplayError {
    /// A flow references a `pool_id` absent from the reserve snapshot.
    #[error("flow references a pool id absent from the reserve snapshot")]
    PoolNotFound,
    /// A flow's `token_in` is neither token of the pool it names.
    #[error("flow's input token is not one of its pool's two tokens")]
    TokenNotInPool,
}

/// One pool's live reserves during a replay. Both swap directions share this single entry; a swap
/// mutates it in place so a later swap through the same pool sees the moved reserves.
struct PoolBookEntry<I> {
    token0: I,
    token1: I,
    reserve0: f32,
    reserve1: f32,
    fee: f32,
    max_swap_0: f32,
    max_swap_1: f32,
}

impl<I: Copy + PartialEq> PoolBookEntry<I> {
    /// Exact constant-product swap of `amount_in` of `token_in` for the other token, mutating the
    /// entry. Mirrors the scalar oracle `forward_reference`: capped-after-fee input, `y·u/(x+u+ε)`,
    /// clamped at zero. Constant product guarantees `out < y`, so the output reserve stays positive.
    fn swap(&mut self, token_in: I, amount_in: f32) -> Result<f32, ReplayError> {
        let forward = token_in == self.token0;
        let (x, y, max_swap) = if forward {
            (self.reserve0, self.reserve1, self.max_swap_0)
        } else if token_in == self.token1 {
            (self.reserve1, self.reserve0, self.max_swap_1)
        } else {
            return Err(ReplayError::TokenNotInPool);
        };

        let capped = (amount_in * self.fee).min(max_swap);
        let out = (y * capped / (x + capped + f32::EPSILON)).max(0.0);

        if forward {
            self.reserve0 += amount_in;
            self.reserve1 -= out;
        } else {
            self.reserve1 += amount_in;
            self.reserve0 -= out;
        }
        Ok(out)
    }
}

/// Replays `flows` sequentially against a mutable book built from `reserves`, starting with the whole
/// `input` in `init_asset`, and returns the exact terminal `init_asset` amount.
///
/// Stages run in ascending order (`FlowRecord::stage`: `0` = `layer_in` … `LAYERS + 1` = `layer_out`).
/// Mirroring the forward, each stage reads the start-of-stage balances and writes a fresh next-stage
/// balance map (so unrouted balances vanish, exactly as the forward replaces its token vector). Within
/// a stage, `amount_in = balance(token_in) × weight`; a `Some(pool_id)` cell swaps through — and
/// mutates — its book entry, a bypass cell (`token_in == token_out`, no pool) carries the amount
/// through, and an empty cell (no pool, distinct tokens) contributes nothing, losing its softmax share
/// just as the zero-reserve forward cell does.
pub fn replay_flows<U, I>(
    flows: &[FlowRecord<U, I>],
    reserves: &[PoolReserves<U, I>],
    init_asset: I,
    input: f32,
) -> Result<f32, ReplayError>
where
    U: Copy + Eq + Hash,
    I: Copy + Eq + Hash,
{
    let steps = flows.iter().map(|flow| ReplayStep {
        stage: flow.stage,
        token_in: flow.token_in,
        token_out: flow.token_out,
        pool: flow.pool_id,
        weight: flow.weight,
    });
    replay_steps(steps, reserves, init_asset, input)
}

/// Replays a discrete [`ExecutionPlan`] (the candidate-extraction output) against an f32 reserve book,
/// the reference oracle for the plan client-evm will execute losslessly. Same weight-driven semantics
/// as [`replay_flows`]: [`StepKind::Swap`] swaps through and mutates its pool, [`StepKind::Carry`]
/// carries the token through unchanged. The plan supplies its own `init_asset` and absolute
/// `entry_amount`.
pub fn replay_plan<U, I>(
    plan: &ExecutionPlan<U, I>,
    reserves: &[PoolReserves<U, I>],
) -> Result<f32, ReplayError>
where
    U: Copy + Eq + Hash,
    I: Copy + Eq + Hash,
{
    let steps = plan.steps.iter().map(|step| ReplayStep {
        stage: step.stage,
        token_in: step.token_in,
        token_out: step.token_out,
        pool: match step.kind {
            StepKind::Swap(pool) => Some(pool),
            StepKind::Carry => None,
        },
        weight: step.weight,
    });
    replay_steps(steps, reserves, plan.init_asset, plan.entry_amount)
}

/// The normalized unit both entry points fold over: a weighted hop through an optional pool. `pool`
/// is `None` for a carry (`token_in == token_out`, the amount passes through) and, for raw flows only,
/// an empty cell (`token_in != token_out`, whose share is lost — a plan never carries such a step).
struct ReplayStep<U, I> {
    stage: usize,
    token_in: I,
    token_out: I,
    pool: Option<U>,
    weight: f32,
}

/// Shared sequential replay: build a mutable pool book from `reserves`, seed the whole `input` into
/// `init_asset`, then fold the steps stage by stage (ascending), reading each stage's start balances
/// and writing a fresh next-stage map so unrouted balances vanish. Returns the terminal `init_asset`
/// amount.
fn replay_steps<U, I>(
    steps: impl Iterator<Item = ReplayStep<U, I>> + Clone,
    reserves: &[PoolReserves<U, I>],
    init_asset: I,
    input: f32,
) -> Result<f32, ReplayError>
where
    U: Copy + Eq + Hash,
    I: Copy + Eq + Hash,
{
    let mut book: HashMap<U, PoolBookEntry<I>> = HashMap::new();
    for reserve in reserves {
        // One entry per pool id; the snapshot may carry both directions (they mirror each other), so
        // the first-seen orientation is authoritative and `swap` resolves direction by token.
        book.entry(reserve.pool_id)
            .or_insert_with(|| PoolBookEntry {
                token0: reserve.token0,
                token1: reserve.token1,
                reserve0: reserve.value.token_0,
                reserve1: reserve.value.token_1,
                fee: reserve.value.fee_multiplier,
                max_swap_0: reserve.value.max_swap_0,
                max_swap_1: reserve.value.max_swap_1,
            });
    }

    let mut balances: HashMap<I, f32> = HashMap::new();
    balances.insert(init_asset, input);

    let Some(max_stage) = steps.clone().map(|step| step.stage).max() else {
        // No steps to route: the input never leaves the init asset.
        return Ok(input);
    };

    for stage in 0..=max_stage {
        let mut next: HashMap<I, f32> = HashMap::new();
        for step in steps.clone().filter(|step| step.stage == stage) {
            let available = balances.get(&step.token_in).copied().unwrap_or(0.0);
            let amount_in = available * step.weight;
            if amount_in <= 0.0 {
                continue;
            }
            match step.pool {
                Some(pool_id) => {
                    let entry = book.get_mut(&pool_id).ok_or(ReplayError::PoolNotFound)?;
                    let out = entry.swap(step.token_in, amount_in)?;
                    *next.entry(step.token_out).or_default() += out;
                }
                None if step.token_in == step.token_out => {
                    *next.entry(step.token_out).or_default() += amount_in;
                }
                None => {}
            }
        }
        balances = next;
    }

    Ok(balances.get(&init_asset).copied().unwrap_or(0.0))
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use burn::backend::{Autodiff, NdArray};
    use proptest::prelude::*;

    use crate::model::Model;
    use crate::pool_reserves::VirtualReserveValues;
    use crate::tokens::test as tokens;
    use crate::tokens::test::TokenAddress;
    use crate::utils::Invertible;

    use super::*;

    type CpuBackend = Autodiff<NdArray<f32>>;

    const FEE: f32 = 0.997;
    const DEEP: f32 = 1_000_000_000.0;

    fn pool(
        token0: TokenAddress,
        token1: TokenAddress,
        pool_id: i32,
        reserve0: f32,
        reserve1: f32,
    ) -> PoolReserves<i32, TokenAddress> {
        PoolReserves {
            token0,
            token1,
            pool_id,
            value: VirtualReserveValues {
                token_0: reserve0,
                token_1: reserve1,
                fee_multiplier: FEE,
                max_swap_0: f32::MAX,
                max_swap_1: f32::MAX,
            },
        }
    }

    /// Expands each pool into both swap directions (the model needs directional reserves).
    fn both_directions(
        pools: Vec<PoolReserves<i32, TokenAddress>>,
    ) -> Vec<PoolReserves<i32, TokenAddress>> {
        pools
            .into_iter()
            .flat_map(|reserve| [reserve, reserve.inverse()])
            .collect()
    }

    /// A deep, balanced USDC/WETH/WBTC triangle: no cross-pool arbitrage, slippage negligible at the
    /// sizes probed, so replay and `evaluate` should agree tightly.
    fn deep_no_arbitrage_universe() -> Vec<PoolReserves<i32, TokenAddress>> {
        let usdc = tokens::USDC.address;
        let weth = tokens::WETH.address;
        let wbtc = tokens::WBTC.address;
        both_directions(vec![
            pool(usdc, weth, 1, DEEP, DEEP),
            pool(weth, wbtc, 2, DEEP, DEEP),
            pool(wbtc, usdc, 3, DEEP, DEEP),
        ])
    }

    fn init_model(
        reserves: Vec<PoolReserves<i32, TokenAddress>>,
    ) -> Model<CpuBackend, i32, TokenAddress, 1> {
        Model::<CpuBackend, i32, TokenAddress, 1>::init(
            tokens::USDC.address,
            reserves,
            &HashSet::new(),
            &HashSet::new(),
        )
        .expect("model init failed")
    }

    #[test]
    fn deep_pools_replay_matches_evaluate() {
        // The headline Step 12 verify: sequential exact replay of the trained routing reproduces the
        // differentiable prediction on deep, distinct constant-product pools — the "this soft solution
        // is executable as written" green light. Any weights work; `extract_flows`, `evaluate` and
        // `replay_flows` all read the same freshly-initialized model.
        let reserves = deep_no_arbitrage_universe();
        let model = init_model(reserves.clone());
        let input = 1_000.0;

        let flows = model.extract_flows(input).expect("extract_flows failed");
        let exact =
            replay_flows(&flows, &reserves, tokens::USDC.address, input).expect("replay failed");
        let predicted = model.evaluate(input);

        let tolerance = predicted.abs().max(exact.abs()) * 1e-3 + 1e-3;
        assert!(
            (exact - predicted).abs() <= tolerance,
            "replay {exact} diverged from evaluate {predicted} beyond {tolerance}"
        );
        assert!(exact.is_finite() && exact >= 0.0, "replay produced {exact}");
    }

    #[test]
    fn no_arbitrage_replay_never_profits() {
        // Executable + no fake profit: replaying a deep balanced market recovers a positive amount
        // strictly below the input (only fees and slippage are lost), matching the closed-route
        // golden invariant but through the exact-replay path.
        let reserves = deep_no_arbitrage_universe();
        let model = init_model(reserves.clone());
        let input = 1_000.0;

        let flows = model.extract_flows(input).expect("extract_flows failed");
        let exact =
            replay_flows(&flows, &reserves, tokens::USDC.address, input).expect("replay failed");

        assert!(exact > 0.0, "replay recovered nothing: {exact}");
        assert!(
            exact < input,
            "no-arbitrage replay profited: {exact} >= {input}"
        );
    }

    #[test]
    fn same_pool_round_trip_diverges_from_naive() {
        // Anti-tautology: route the full input A->B->A through one *moderate* pool (stage 0 forward,
        // stage 1 bypass carrying B, stage 2 reverse through the same pool). The forward would quote
        // both legs against the original reserves; sequential replay quotes the return leg against the
        // reserves the first leg moved — favorably, since the pool is now A-rich / B-poor — so exact
        // recovers materially MORE than the naive double-quote. This proves replay sequences state the
        // forward does not, and pins the exact value against an independent hand computation.
        let token_a = tokens::USDC.address;
        let token_b = tokens::WETH.address;
        let (ra, rb) = (1_000.0f32, 1_000.0f32);
        let input = 100.0f32;

        let reserves = vec![PoolReserves {
            token0: token_a,
            token1: token_b,
            pool_id: 1,
            value: VirtualReserveValues {
                token_0: ra,
                token_1: rb,
                fee_multiplier: FEE,
                max_swap_0: f32::MAX,
                max_swap_1: f32::MAX,
            },
        }];

        // `replay_flows` is weight-driven; the `amount_in`/`amount_out` fields are unused here.
        let flow = |stage, token_in, token_out, pool_id| FlowRecord {
            stage,
            token_in,
            token_out,
            pool_id,
            amount_in: 0.0,
            amount_out: 0.0,
            weight: 1.0,
        };
        let flows = vec![
            flow(0, token_a, token_b, Some(1)),
            flow(1, token_b, token_b, None),
            flow(2, token_b, token_a, Some(1)),
        ];

        // Independent hand computation of the exact sequential result.
        let eps = f32::EPSILON;
        let cap0 = input * FEE;
        let out1 = rb * cap0 / (ra + cap0 + eps);
        let (ra2, rb2) = (ra + input, rb - out1);
        let cap2 = out1 * FEE;
        let exact_expected = ra2 * cap2 / (rb2 + cap2 + eps);
        // Naive: the return leg quoted against the ORIGINAL reserves.
        let naive = ra * cap2 / (rb + cap2 + eps);

        let exact = replay_flows(&flows, &reserves, token_a, input).expect("replay failed");

        assert!(
            (exact - exact_expected).abs() <= exact_expected.abs() * 1e-5 + 1e-6,
            "replay {exact} did not match hand-computed sequential {exact_expected}"
        );
        assert!(
            exact > naive + 1.0,
            "same-pool round trip must beat the naive double-quote: {exact} vs {naive}"
        );
    }

    proptest! {
        #[test]
        fn replay_stays_finite_and_non_negative(
            r0 in 1_000.0f32..DEEP,
            r1 in 1_000.0f32..DEEP,
            r2 in 1_000.0f32..DEEP,
            r3 in 1_000.0f32..DEEP,
            r4 in 1_000.0f32..DEEP,
            r5 in 1_000.0f32..DEEP,
            input in 1.0f32..1_000_000.0,
        ) {
            // Arbitrary (possibly arbitraged, possibly shallow) universes must never make replay
            // diverge, error, or go negative — the pipeline stays crash- and NaN-free.
            let usdc = tokens::USDC.address;
            let weth = tokens::WETH.address;
            let wbtc = tokens::WBTC.address;
            let reserves = both_directions(vec![
                pool(usdc, weth, 1, r0, r1),
                pool(weth, wbtc, 2, r2, r3),
                pool(wbtc, usdc, 3, r4, r5),
            ]);
            let model = init_model(reserves.clone());

            let flows = model.extract_flows(input).expect("extract_flows failed");
            let exact = replay_flows(&flows, &reserves, usdc, input).expect("replay failed");

            prop_assert!(exact.is_finite(), "replay produced non-finite {exact}");
            prop_assert!(exact >= 0.0, "replay produced negative {exact}");
        }
    }
}
