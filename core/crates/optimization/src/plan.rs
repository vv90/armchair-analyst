//! Candidate extraction (`optimization.md` Step 12): turning the trained model's soft, dense flows
//! into a discrete, executable [`ExecutionPlan`].
//!
//! [`crate::model::Model::extract_flows`] reads the differentiable routing off faithfully — one
//! [`crate::model::FlowRecord`] per layout cell, including empty padding cells that carry no action.
//! [`build_plan`] folds that into the plan that actually gets executed: it drops empty and below-
//! threshold cells, merges cells that hit the same pool in the same direction within a stage, and
//! renormalizes the surviving per-token shares so no balance is stranded.
//!
//! The plan is **hybrid-encoded**: one absolute `entry_amount` (the committed capital the route was
//! optimized for — optimality is size-dependent, so the size is pinned) plus per-step *fractions*
//! (`weight`) for all routing. Downstream hop amounts are emergent — they depend on the exact realized
//! output of the prior hop, which the lossless executor (client-evm) computes differently from the f32
//! forward — so the plan carries fractions of the realized balance, never f32-predicted amounts. The
//! recorded `amount_in`/`amount_out` are advisory only (pruning, gas, display); they are not the
//! source of truth for execution.
//!
//! [`crate::replay::replay_plan`] replays a plan against an f32 reserve book as the reference oracle;
//! the real, exact, lossless simulation lives in client-evm.

use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use crate::model::FlowRecord;

/// What an [`ExecutableStep`] does. There is no "empty" variant: empty layout cells (no pool, distinct
/// tokens, not a configured bridge pair) carry no action and are dropped by [`build_plan`], so they
/// cannot appear in a plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StepKind<U> {
    /// Swap through the named pool in the `token_in -> token_out` direction.
    Swap(U),
    /// Carry the token straight through this stage unchanged (a bypass cell, `token_in == token_out`).
    Carry,
    /// Convert `token_in` to `token_out` 1:1 across a configured bridge pair — zero cost, no fee, no
    /// pool. The executable form of a bypass-masked bridge cell (the forward carries the pre-fee
    /// routed input through such a cell unswapped).
    Bridge,
}

/// One executable action in a plan. `weight` is the renormalized fraction of the realized `token_in`
/// balance to route here; `amount_in`/`amount_out` are the model's f32 predictions, retained only as
/// advisory metadata (pruning thresholds, gas estimation, display) — not execution truth.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExecutableStep<U, I> {
    pub stage: usize,
    pub token_in: I,
    pub token_out: I,
    pub kind: StepKind<U>,
    pub weight: f32,
    pub amount_in: f32,
    pub amount_out: f32,
}

/// A discrete, dependency-ordered executable route recovered from a trained model. Steps are sorted by
/// ascending `stage` (stage *i* feeds *i + 1*, so stage order is already topological; steps within a
/// stage are parallel and independent).
#[derive(Clone, Debug, PartialEq)]
pub struct ExecutionPlan<U, I> {
    pub init_asset: I,
    pub entry_amount: f32,
    pub steps: Vec<ExecutableStep<U, I>>,
}

/// Folds a faithful flow set into an executable plan.
///
/// 1. **Classify & drop empties.** A cell whose `(token_in, token_out)` pair is in `bridges` becomes
///    [`StepKind::Bridge`] — checked *before* `pool_id`, because a bridge cell can overlap a real
///    pool in the layout while the forward bypass-masks it to an identity carry (classifying it as a
///    swap would replay a fee-paying hop the model never quoted). Otherwise `Some(pool)` becomes
///    [`StepKind::Swap`]; a no-pool cell with `token_in == token_out` becomes [`StepKind::Carry`];
///    a remaining no-pool cell with distinct tokens is layout padding and is dropped.
/// 2. **Prune tiny flows** below `min_weight`.
/// 3. **Merge** cells sharing `(stage, token_in, token_out, kind)` — the same pool and direction within
///    a stage — summing their weight and advisory amounts.
/// 4. **Renormalize** the surviving `weight`s within each `(stage, token_in)` group to sum to one, so
///    the dropped/pruned share is redistributed to real actions rather than stranding balance. On a
///    fully-connected market with no pruning this is a no-op (softmax columns already sum to one).
/// 5. **Order** by ascending `stage`.
pub fn build_plan<U, I>(
    flows: &[FlowRecord<U, I>],
    init_asset: I,
    entry_amount: f32,
    min_weight: f32,
    bridges: &HashSet<(I, I)>,
) -> ExecutionPlan<U, I>
where
    U: Copy + Eq + Hash,
    I: Copy + Eq + Hash,
{
    // One executable cell, identified by stage + direction + kind so duplicates collapse.
    #[derive(Clone, Copy, PartialEq, Eq, Hash)]
    struct CellKey<U, I> {
        stage: usize,
        token_in: I,
        token_out: I,
        kind: StepKind<U>,
    }
    #[derive(Clone, Copy, Default)]
    struct Accum {
        weight: f32,
        amount_in: f32,
        amount_out: f32,
    }

    // Merge cells sharing a key, summing weight and advisory amounts.
    let mut merged: HashMap<CellKey<U, I>, Accum> = HashMap::new();
    for flow in flows {
        // Bridge membership wins over `pool_id`: a bridge cell overlapping a pool is bypass-masked
        // in the forward, so its flow was carried 1:1, never swap-quoted.
        let kind = if flow.token_in != flow.token_out
            && bridges.contains(&(flow.token_in, flow.token_out))
        {
            StepKind::Bridge
        } else {
            match flow.pool_id {
                Some(pool) => StepKind::Swap(pool),
                None if flow.token_in == flow.token_out => StepKind::Carry,
                None => continue, // empty padding cell — no executable action
            }
        };
        if flow.weight < min_weight {
            continue; // prune tiny flow
        }
        let acc = merged
            .entry(CellKey {
                stage: flow.stage,
                token_in: flow.token_in,
                token_out: flow.token_out,
                kind,
            })
            .or_default();
        acc.weight += flow.weight;
        acc.amount_in += flow.amount_in;
        acc.amount_out += flow.amount_out;
    }

    // Per-(stage, token_in) surviving weight totals, for renormalization.
    let mut group_totals: HashMap<(usize, I), f32> = HashMap::new();
    for (key, acc) in &merged {
        *group_totals.entry((key.stage, key.token_in)).or_insert(0.0) += acc.weight;
    }

    let mut steps: Vec<ExecutableStep<U, I>> = merged
        .into_iter()
        .map(|(key, acc)| {
            let total = group_totals
                .get(&(key.stage, key.token_in))
                .copied()
                .unwrap_or(0.0);
            let normalized = if total > 0.0 { acc.weight / total } else { 0.0 };
            ExecutableStep {
                stage: key.stage,
                token_in: key.token_in,
                token_out: key.token_out,
                kind: key.kind,
                weight: normalized,
                amount_in: acc.amount_in,
                amount_out: acc.amount_out,
            }
        })
        .collect();
    // Stage order is the only ordering replay depends on; within a stage steps are independent.
    steps.sort_by_key(|step| step.stage);

    ExecutionPlan {
        init_asset,
        entry_amount,
        steps,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use burn::backend::{Autodiff, NdArray};
    use proptest::prelude::*;

    use crate::model::{FlowRecord, Model};
    use crate::pool_reserves::{PoolReserves, VirtualReserveValues};
    use crate::replay::replay_plan;
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

    fn both_directions(
        pools: Vec<PoolReserves<i32, TokenAddress>>,
    ) -> Vec<PoolReserves<i32, TokenAddress>> {
        pools
            .into_iter()
            .flat_map(|reserve| [reserve, reserve.inverse()])
            .collect()
    }

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

    /// Sums the surviving weights of each `(stage, token_in)` group in a plan.
    fn group_weight_sums(
        plan: &ExecutionPlan<i32, TokenAddress>,
    ) -> HashMap<(usize, TokenAddress), f32> {
        let mut sums: HashMap<(usize, TokenAddress), f32> = HashMap::new();
        for step in &plan.steps {
            *sums.entry((step.stage, step.token_in)).or_insert(0.0) += step.weight;
        }
        sums
    }

    #[test]
    fn pruning_barely_changes_value() {
        // Headline Step 12 Verify: the executable plan reproduces the differentiable prediction on
        // deep, distinct pools. The triangle connects every token pair, so no cell is empty and a tiny
        // min_weight prunes nothing — renormalization is a no-op and the plan replays to `evaluate`.
        let reserves = deep_no_arbitrage_universe();
        let model = init_model(reserves.clone());
        let input = 1_000.0;

        let flows = model.extract_flows(input).expect("extract_flows failed");
        let plan = build_plan(&flows, tokens::USDC.address, input, 1e-4, &HashSet::new());
        let replayed = replay_plan(&plan, &reserves).expect("replay_plan failed");
        let predicted = model.evaluate(input);

        let tolerance = predicted.abs().max(replayed.abs()) * 1e-3 + 1e-3;
        assert!(
            (replayed - predicted).abs() <= tolerance,
            "plan replay {replayed} diverged from evaluate {predicted} beyond {tolerance}"
        );
    }

    #[test]
    fn plan_without_bridges_has_only_swap_or_carry_and_groups_sum_to_one() {
        // Every step is an executable action (no empty cells survive; no bridges are configured, so
        // no `Bridge` steps either), and each token's surviving shares within a stage sum to one —
        // nothing is stranded, nothing over-committed.
        let reserves = deep_no_arbitrage_universe();
        let model = init_model(reserves.clone());
        let input = 1_000.0;

        let flows = model.extract_flows(input).expect("extract_flows failed");
        let plan = build_plan(&flows, tokens::USDC.address, input, 1e-4, &HashSet::new());

        assert!(!plan.steps.is_empty(), "plan has no steps");
        assert_eq!(plan.entry_amount, input);
        assert_eq!(plan.init_asset, tokens::USDC.address);

        for ((stage, token_in), sum) in group_weight_sums(&plan) {
            assert!(
                (sum - 1.0).abs() <= 1e-4,
                "stage {stage} token weights sum to {sum}, expected 1.0 (token {token_in:?})"
            );
        }
    }

    #[test]
    fn pruning_drops_tiny_and_renormalizes_survivors() {
        // Hand-built stage-0 split of the init asset: two dominant pools plus one dust flow. A
        // min_weight above the dust weight drops it, and the two survivors renormalize to sum to one.
        let usdc = tokens::USDC.address;
        let weth = tokens::WETH.address;
        let wbtc = tokens::WBTC.address;
        let step = |token_out, pool_id, weight| FlowRecord {
            stage: 0,
            token_in: usdc,
            token_out,
            pool_id: Some(pool_id),
            amount_in: 0.0,
            amount_out: 0.0,
            weight,
        };
        let flows = vec![
            step(weth, 1, 0.60),
            step(wbtc, 2, 0.39),
            step(weth, 3, 0.01), // dust — below min_weight
        ];

        let plan = build_plan(&flows, usdc, 1_000.0, 0.05, &HashSet::new());

        assert_eq!(plan.steps.len(), 2, "dust flow should have been pruned");
        assert!(
            plan.steps.iter().all(|s| s.kind != StepKind::Swap(3)),
            "pool 3 (dust) must not survive"
        );
        let sum: f32 = plan.steps.iter().map(|s| s.weight).sum();
        assert!(
            (sum - 1.0).abs() <= 1e-5,
            "survivors must renormalize to 1.0, got {sum}"
        );
        // 0.60 / 0.99 ≈ 0.606; the ratio between survivors is preserved.
        let pool1 = plan
            .steps
            .iter()
            .find(|s| s.kind == StepKind::Swap(1))
            .expect("pool 1 missing");
        assert!((pool1.weight - 0.60 / 0.99).abs() <= 1e-4);
    }

    #[test]
    fn merge_folds_same_pool_direction_duplicates() {
        // Two cells hitting the same pool in the same direction within a stage merge into one step,
        // summing weights and advisory amounts.
        let usdc = tokens::USDC.address;
        let weth = tokens::WETH.address;
        let dup = |weight, amount_in, amount_out| FlowRecord {
            stage: 1,
            token_in: usdc,
            token_out: weth,
            pool_id: Some(7),
            amount_in,
            amount_out,
            weight,
        };
        let flows = vec![dup(0.3, 30.0, 29.0), dup(0.7, 70.0, 68.0)];

        let plan = build_plan(&flows, usdc, 100.0, 1e-4, &HashSet::new());

        assert_eq!(plan.steps.len(), 1, "duplicates should merge into one step");
        let step = plan.steps.first().expect("no step");
        assert_eq!(step.kind, StepKind::Swap(7));
        // Weight renormalizes to 1.0 (sole survivor); advisory amounts are the raw sum.
        assert!((step.weight - 1.0).abs() <= 1e-6);
        assert!(
            (step.amount_in - 100.0).abs() <= 1e-4,
            "amount_in {}",
            step.amount_in
        );
        assert!(
            (step.amount_out - 97.0).abs() <= 1e-4,
            "amount_out {}",
            step.amount_out
        );
    }

    #[test]
    fn bridge_flow_survives_as_bridge_step() {
        // A pool-less distinct-token flow whose pair is configured as a bridge becomes a `Bridge`
        // step; the same shape with an unconfigured pair is still layout padding and is dropped.
        let usdc = tokens::USDC.address;
        let weth = tokens::WETH.address;
        let wbtc = tokens::WBTC.address;
        let carry = |token_in, token_out| FlowRecord {
            stage: 1,
            token_in,
            token_out,
            pool_id: None::<i32>,
            amount_in: 50.0,
            amount_out: 50.0,
            weight: 0.5,
        };
        let flows = vec![carry(weth, wbtc), carry(wbtc, usdc)];
        let bridges: HashSet<_> = [(weth, wbtc)].into_iter().collect();

        let plan = build_plan(&flows, usdc, 100.0, 1e-4, &bridges);

        assert_eq!(plan.steps.len(), 1, "only the configured pair survives");
        let step = plan.steps.first().expect("no step");
        assert_eq!(step.kind, StepKind::Bridge);
        assert_eq!((step.token_in, step.token_out), (weth, wbtc));
    }

    #[test]
    fn bridge_over_pool_cell_classifies_as_bridge() {
        // A bridge cell can overlap a real pool in the layout; the forward bypass-masks it to an
        // identity carry, so classification must yield `Bridge`, not a fee-paying `Swap`.
        let weth = tokens::WETH.address;
        let wbtc = tokens::WBTC.address;
        let flows = vec![FlowRecord {
            stage: 1,
            token_in: weth,
            token_out: wbtc,
            pool_id: Some(9),
            amount_in: 50.0,
            amount_out: 50.0,
            weight: 1.0,
        }];
        let bridges: HashSet<_> = [(weth, wbtc)].into_iter().collect();

        let plan = build_plan(&flows, weth, 100.0, 1e-4, &bridges);

        assert_eq!(plan.steps.len(), 1);
        assert_eq!(
            plan.steps.first().expect("no step").kind,
            StepKind::Bridge,
            "bridge membership must win over the overlapping pool id"
        );
    }

    #[test]
    fn bridge_merges_and_renormalizes_with_swaps() {
        // Within one `(stage, token_in)` group: a swap (to a non-bridge token), two bridge cells
        // (merging into one step), and a dust swap below `min_weight`. Survivors renormalize to sum
        // to one with the merged bridge weight intact.
        let usdc = tokens::USDC.address;
        let weth = tokens::WETH.address;
        let wbtc = tokens::WBTC.address;
        let flow = |token_out, pool_id, weight| FlowRecord {
            stage: 0,
            token_in: usdc,
            token_out,
            pool_id,
            amount_in: 0.0,
            amount_out: 0.0,
            weight,
        };
        let flows = vec![
            flow(wbtc, Some(1), 0.50),
            flow(weth, None, 0.30),    // bridge cell
            flow(weth, None, 0.10),    // duplicate bridge cell — merges with the one above
            flow(wbtc, Some(2), 0.01), // dust — below min_weight
        ];
        let bridges: HashSet<_> = [(usdc, weth)].into_iter().collect();

        let plan = build_plan(&flows, usdc, 100.0, 0.05, &bridges);

        assert_eq!(
            plan.steps.len(),
            2,
            "swap + merged bridge survive, dust pruned"
        );
        let sum: f32 = plan.steps.iter().map(|s| s.weight).sum();
        assert!(
            (sum - 1.0).abs() <= 1e-5,
            "survivors renormalize to 1.0, got {sum}"
        );
        let bridge = plan
            .steps
            .iter()
            .find(|s| s.kind == StepKind::Bridge)
            .expect("bridge step missing");
        assert!(
            (bridge.weight - 0.40 / 0.90).abs() <= 1e-5,
            "merged bridge weight {}, expected {}",
            bridge.weight,
            0.40 / 0.90
        );
    }

    #[test]
    fn model_with_bridges_emits_bridge_steps() {
        // End-to-end pin for the extraction hole: a model whose only WETH→WBTC edge is a configured
        // bridge routes real flow through the bypass cell, and `build_plan` must surface it as a
        // `Bridge` step instead of dropping it as padding.
        let usdc = tokens::USDC.address;
        let weth = tokens::WETH.address;
        let wbtc = tokens::WBTC.address;
        let reserves = both_directions(vec![
            pool(usdc, weth, 1, DEEP, DEEP),
            pool(wbtc, usdc, 2, DEEP, DEEP),
        ]);
        let bridges: HashSet<_> = [(weth, wbtc), (wbtc, weth)].into_iter().collect();
        let model = Model::<CpuBackend, i32, TokenAddress, 1>::init(
            usdc,
            reserves,
            &bridges,
            &HashSet::new(),
        )
        .expect("model init failed");

        let flows = model.extract_flows(1_000.0).expect("extract_flows failed");
        let plan = build_plan(&flows, usdc, 1_000.0, 1e-4, &bridges);

        let bridge_steps: Vec<_> = plan
            .steps
            .iter()
            .filter(|s| s.kind == StepKind::Bridge)
            .collect();
        assert!(
            !bridge_steps.is_empty(),
            "a bridged model must emit at least one Bridge step"
        );
        assert!(
            bridge_steps
                .iter()
                .all(|s| bridges.contains(&(s.token_in, s.token_out))),
            "every Bridge step's pair must be a configured bridge"
        );
    }

    proptest! {
        #[test]
        fn build_then_replay_stays_finite_and_conservative(
            r0 in 1_000.0f32..DEEP,
            r1 in 1_000.0f32..DEEP,
            r2 in 1_000.0f32..DEEP,
            r3 in 1_000.0f32..DEEP,
            r4 in 1_000.0f32..DEEP,
            r5 in 1_000.0f32..DEEP,
            input in 1.0f32..1_000_000.0,
        ) {
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
            let plan = build_plan(&flows, usdc, input, 1e-4, &HashSet::new());
            let replayed = replay_plan(&plan, &reserves).expect("replay_plan failed");

            prop_assert!(replayed.is_finite(), "replay produced non-finite {replayed}");
            prop_assert!(replayed >= 0.0, "replay produced negative {replayed}");

            let mut sums: HashMap<(usize, TokenAddress), f32> = HashMap::new();
            for step in &plan.steps {
                *sums.entry((step.stage, step.token_in)).or_insert(0.0) += step.weight;
            }
            for sum in sums.values() {
                prop_assert!(*sum <= 1.0 + 1e-4, "group weights sum to {sum} > 1");
            }
        }

        #[test]
        fn build_then_replay_bridged_universe_stays_finite_and_conservative(
            r0 in 1_000.0f32..DEEP,
            r1 in 1_000.0f32..DEEP,
            r2 in 1_000.0f32..DEEP,
            r3 in 1_000.0f32..DEEP,
            input in 1.0f32..1_000_000.0,
        ) {
            // Same conservativeness envelope as above, but the only WETH<->WBTC edge is a configured
            // bridge, so the extracted plan routes real flow through `Bridge` steps end to end.
            let usdc = tokens::USDC.address;
            let weth = tokens::WETH.address;
            let wbtc = tokens::WBTC.address;
            let reserves = both_directions(vec![
                pool(usdc, weth, 1, r0, r1),
                pool(wbtc, usdc, 2, r2, r3),
            ]);
            let bridges: HashSet<_> = [(weth, wbtc), (wbtc, weth)].into_iter().collect();
            let model = Model::<CpuBackend, i32, TokenAddress, 1>::init(
                usdc,
                reserves.clone(),
                &bridges,
                &HashSet::new(),
            )
            .expect("model init failed");

            let flows = model.extract_flows(input).expect("extract_flows failed");
            let plan = build_plan(&flows, usdc, input, 1e-4, &bridges);
            let replayed = replay_plan(&plan, &reserves).expect("replay_plan failed");

            prop_assert!(replayed.is_finite(), "replay produced non-finite {replayed}");
            prop_assert!(replayed >= 0.0, "replay produced negative {replayed}");

            let mut sums: HashMap<(usize, TokenAddress), f32> = HashMap::new();
            for step in &plan.steps {
                *sums.entry((step.stage, step.token_in)).or_insert(0.0) += step.weight;
            }
            for sum in sums.values() {
                prop_assert!(*sum <= 1.0 + 1e-4, "group weights sum to {sum} > 1");
            }
        }
    }
}
