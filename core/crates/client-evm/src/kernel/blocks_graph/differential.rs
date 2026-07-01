//! Stage-2 differential proptest: the new `BlocksGraph::reanchored_to` fold/re-root must agree with
//! the legacy `State::with_finalized_block_observed` finalization path.
//!
//! TEMPORARY SCAFFOLDING — delete this whole file and its `mod differential;` declaration at the
//! Stage-4 swap, when the legacy finalization path itself is removed.
//!
//! Placed as a child of `blocks_graph` (not a sibling) so it is a descendant of both `blocks_graph`
//! (the new graph's private internals, incl. `ConnectedHash` construction) and `kernel` (the
//! module-private legacy `State`/`BlockNode`/`BlocksGraph`/`FinalizedState` and the registries).
//!
//! What is compared, and how divergences are normalized:
//!  - **Two-sided equivalence** (`new_finalization_matches_legacy`): both paths finalize to the same
//!    target and produce the same finalized snapshot, restricted to base-resident pools. To make them
//!    comparable, generated canonical paths are **all `Complete`** (legacy would otherwise stop at the
//!    latest fully-complete block ≤ target — partial compaction), and comparison is on **base keys
//!    only** (the new fold folds only snapshot-resident pools; legacy also snapshots path-discovered
//!    pools — see `registry_only_pool_*`).
//!  - **One-sided A4** (`incomplete_path_is_refused_with_hole_ranges`): new-only — an incomplete path
//!    is refused with the exact hole ranges; legacy has no equivalent (it silently compacts short).
//!
//! The legacy per-block *derivation* (`BlockLogsReceived` → snapshot) is not the subject here — it is
//! not what `reanchored_to` replaces — so per-block snapshots are planted directly via the same
//! `derive_pool_state` primitive, isolating the finalization *orchestration* both sides own.

use std::collections::{BTreeMap, HashMap, HashSet};

use alloy::primitives::{Address, BlockHash, Bloom, U160, aliases::I24};
use proptest::prelude::*;

// New (log-sourced) graph under test.
use super::{
    AnchoredRef, BlockData, BlockLogs, BlocksGraph as NewGraph, ConnectedHash, ConnectedNode, Node,
    ReanchorError,
};
// Legacy finalization path being matched.
use crate::kernel::pending_requests::PendingRequests;
use crate::kernel::pool_registry::{
    PoolFee, PoolMetadata, PoolMetadataResult, TrustedPoolRegistry, UniswapV3Fee,
};
use crate::kernel::token_registry::TokenRegistry;
use crate::kernel::{BlockNode, BlocksGraph as LegacyGraph, FinalizedState, PoolLogsStatus, State};
use crate::tick::Tick;
use crate::{
    ChainKey, PoolLog, PoolLogEvent, PoolRef, PoolState, ProtocolPoolKey, derive_pool_state,
};

const CHAIN: ChainKey = ChainKey::Ethereum;

// ---- fixtures ------------------------------------------------------------------------------------

fn dh(index: usize) -> BlockHash {
    BlockHash::with_last_byte(index as u8)
}

fn tk(value: i32) -> I24 {
    I24::try_from(value).expect("tick fixture in range")
}

fn ps(sqrt: u128, tick: i32, liquidity: u128) -> PoolState {
    PoolState {
        sqrt_price_x96: U160::from(sqrt),
        tick: tk(tick),
        liquidity,
    }
}

/// The fixed base pool set — all v3, all present in the finalized snapshot.
const POOL_BYTES: [u8; 3] = [0xA1, 0xA2, 0xA3];

fn pool_of(index: usize) -> PoolRef {
    PoolRef::uniswap_v3(Address::with_last_byte(POOL_BYTES[index]), CHAIN)
}

/// Base liquidity is large so bounded generated mints never overflow.
fn base_snapshot() -> HashMap<PoolRef, PoolState> {
    HashMap::from([
        (pool_of(0), ps(1_000, 0, 1_000_000)),
        (pool_of(1), ps(2_000, 10, 2_000_000)),
        (pool_of(2), ps(3_000, -10, 3_000_000)),
    ])
}

fn pool_meta() -> PoolMetadata {
    PoolMetadata {
        token0: Address::with_last_byte(1),
        token1: Address::with_last_byte(2),
        fee: PoolFee::Tiered(UniswapV3Fee::Fee500),
    }
}

/// A registry that verifies every base pool key plus any `extra` keys, so legacy's trusted-pool scan
/// resolves them and finalization advances.
fn registry(extra: &[ProtocolPoolKey]) -> TrustedPoolRegistry {
    let mut results: HashMap<ProtocolPoolKey, PoolMetadataResult> = (0..POOL_BYTES.len())
        .map(|index| (pool_of(index).key, Ok(pool_meta())))
        .collect();
    for key in extra {
        results.insert(*key, Ok(pool_meta()));
    }
    TrustedPoolRegistry::new().with_metadata_results(CHAIN, results)
}

#[allow(deprecated)]
fn pool_log(pool: PoolRef, event: PoolLogEvent) -> PoolLog {
    PoolLog {
        pool: pool.key,
        // Deprecated; intra-block order is the BTreeMap key assigned below.
        log_index: 0,
        event,
    }
}

/// Keys logs by arrival order into the `Complete` payload the new graph reads.
fn complete(logs: Vec<PoolLog>) -> BlockLogs {
    BlockLogs::Complete(
        logs.into_iter()
            .enumerate()
            .map(|(index, log)| (index as u64, log))
            .collect(),
    )
}

/// A block's `(pool index, event)` list rendered as new-graph `Complete` logs.
fn complete_events(events: &[(usize, PoolLogEvent)]) -> BlockLogs {
    complete(
        events
            .iter()
            .map(|(index, event)| pool_log(pool_of(*index), event.clone()))
            .collect(),
    )
}

/// Keys logs by arrival order into the `Streamed` (best-effort WS) payload — read only under
/// `AllowStreamed` (the optimization view), refused by the finalization fold.
fn streamed(logs: Vec<PoolLog>) -> BlockLogs {
    BlockLogs::Streamed(
        logs.into_iter()
            .enumerate()
            .map(|(index, log)| (index as u64, log))
            .collect(),
    )
}

/// A block's `(pool index, event)` list as `Streamed` or `Complete` logs per `is_streamed`. The
/// optimization fold (`AllowStreamed`) reads both identically, so the choice only exercises the
/// authority seam — both must contribute to match the legacy overlay (which stores snapshots
/// regardless of Streamed/Complete).
fn view_events(events: &[(usize, PoolLogEvent)], is_streamed: bool) -> BlockLogs {
    let logs: Vec<PoolLog> = events
        .iter()
        .map(|(index, event)| pool_log(pool_of(*index), event.clone()))
        .collect();
    if is_streamed {
        streamed(logs)
    } else {
        complete(logs)
    }
}

// ---- scenario model + generator ------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Scenario {
    node_count: usize,
    /// `parents[0] == 0` (anchor); `parents[i] < i`, so the block set is anchor-rooted and acyclic.
    parents: Vec<usize>,
    /// Depth from the anchor; the canonical path to any node is number-contiguous.
    number: Vec<u64>,
    /// `events[i]` is node `i`'s `(pool index, event)` list; `events[0]` is empty (the anchor).
    events: Vec<Vec<(usize, PoolLogEvent)>>,
    /// The connected node to finalize to / observe as the optimization head (`1..node_count`).
    target: usize,
    /// An admission permutation of `1..node_count` (stresses the new graph's order-independence).
    order: Vec<usize>,
    /// Per node, whether the optimization-view builder plants its logs `Streamed` (best-effort WS)
    /// rather than `Complete`; `streamed[0]` (the anchor) is unused. Ignored by the Stage-2
    /// finalization builders, which are all-`Complete`.
    streamed: Vec<bool>,
}

fn swap_strategy() -> impl Strategy<Value = PoolLogEvent> {
    (1u128..1_000_000u128, -50i32..50i32, 1u128..100_000u128).prop_map(
        |(sqrt, tick, liquidity)| PoolLogEvent::Swap {
            sqrt_price_x96: U160::from(sqrt),
            tick: tk(tick),
            liquidity,
        },
    )
}

fn mint_strategy() -> impl Strategy<Value = PoolLogEvent> {
    (-50i32..50i32, -50i32..50i32, 1u128..1_000u128).prop_map(|(a, b, amount)| {
        let (lower, upper) = if a <= b { (a, b) } else { (b, a) };
        let upper = if lower == upper { upper + 1 } else { upper };
        PoolLogEvent::Mint {
            tick_lower: tk(lower),
            tick_upper: tk(upper),
            amount,
        }
    })
}

fn event_strategy() -> impl Strategy<Value = (usize, PoolLogEvent)> {
    (
        0usize..POOL_BYTES.len(),
        prop_oneof![swap_strategy(), mint_strategy()],
    )
}

fn scenario_strategy() -> impl Strategy<Value = Scenario> {
    (2usize..10)
        .prop_flat_map(|node_count| {
            let admit: Vec<usize> = (1..node_count).collect();
            (
                Just(node_count),
                prop::collection::vec(any::<usize>(), node_count - 1),
                prop::collection::vec(
                    prop::collection::vec(event_strategy(), 0..3),
                    node_count - 1,
                ),
                1usize..node_count,
                Just(admit).prop_shuffle(),
                prop::collection::vec(any::<bool>(), node_count - 1),
            )
        })
        .prop_map(
            |(node_count, parent_choices, node_events, target, order, streamed_choices)| {
                let parents: Vec<usize> = (0..node_count)
                    .map(|index| if index == 0 { 0 } else { parent_choices[index - 1] % index })
                    .collect();
                let mut number = vec![0u64; node_count];
                for index in 1..node_count {
                    number[index] = number[parents[index]] + 1;
                }
                let mut events = vec![Vec::new()];
                events.extend(node_events);
                let mut streamed = vec![false];
                streamed.extend(streamed_choices);
                Scenario {
                    node_count,
                    parents,
                    number,
                    events,
                    target,
                    order,
                    streamed,
                }
            },
        )
}

// ---- interpreters --------------------------------------------------------------------------------

/// Builds the new graph from a scenario: admit every block in `order` (children may precede parents,
/// exercising pending→promotion), then set each block's logs to the `Complete` payload it carries.
fn build_new(scenario: &Scenario) -> NewGraph {
    let mut graph = NewGraph::new(dh(0));
    for &node in &scenario.order {
        graph = graph
            .with_block(
                dh(node),
                dh(scenario.parents[node]),
                scenario.number[node],
                Bloom::repeat_byte(0xFF),
            )
            .expect("generated admission succeeds");
    }
    for node in 1..scenario.node_count {
        let logs = complete_events(&scenario.events[node]);
        match graph.nodes.get_mut(&dh(node)) {
            Some(Node::Connected(connected)) => connected.data.logs = logs,
            _ => panic!("every generated block must be connected after full admission"),
        }
    }
    graph
}

/// Builds the new graph for the optimization view: like [`build_new`] but plants each block's logs as
/// `Streamed` or `Complete` per `scenario.streamed`, then sets `observed_head` to the target so the
/// optimization read folds the canonical suffix `anchor → target`.
fn build_new_optimization(scenario: &Scenario) -> NewGraph {
    let mut graph = NewGraph::new(dh(0));
    for &node in &scenario.order {
        graph = graph
            .with_block(
                dh(node),
                dh(scenario.parents[node]),
                scenario.number[node],
                Bloom::repeat_byte(0xFF),
            )
            .expect("generated admission succeeds");
    }
    for node in 1..scenario.node_count {
        let logs = view_events(&scenario.events[node], scenario.streamed[node]);
        match graph.nodes.get_mut(&dh(node)) {
            Some(Node::Connected(connected)) => connected.data.logs = logs,
            _ => panic!("every generated block must be connected after full admission"),
        }
    }
    graph.with_observed_head(dh(scenario.target))
}

/// Builds the pre-finalization legacy `State`, planting each block's absolute per-pool snapshots via
/// an incremental `derive_pool_state` fold (block = parent snapshot + block logs) and setting the
/// canonical tip to the target. Shared by the finalization builder ([`build_legacy`]) and the
/// optimization-view read (`optimization_view_matches_legacy`).
fn legacy_state(scenario: &Scenario) -> State {
    let base = base_snapshot();

    // `state_at[node]` is every base pool's absolute state at `node`; `stored[node]` is the subset a
    // block would have snapshotted (the pools it touched) — what legacy's derivation persists.
    let mut state_at: Vec<HashMap<PoolRef, PoolState>> = vec![HashMap::new(); scenario.node_count];
    let mut stored: Vec<HashMap<PoolRef, PoolState>> = vec![HashMap::new(); scenario.node_count];
    state_at[0] = base.clone();
    for node in 1..scenario.node_count {
        let mut current = state_at[scenario.parents[node]].clone();
        let mut by_pool: BTreeMap<usize, Vec<PoolLogEvent>> = BTreeMap::new();
        for (pool_index, event) in &scenario.events[node] {
            by_pool.entry(*pool_index).or_default().push(event.clone());
        }
        let mut snapshots = HashMap::new();
        for (pool_index, events) in by_pool {
            let pool = pool_of(pool_index);
            let run: Vec<&PoolLogEvent> = events.iter().collect();
            let folded = derive_pool_state(current.get(&pool), &run)
                .expect("base pool fold is always derivable");
            current.insert(pool, folded.clone());
            snapshots.insert(pool, folded);
        }
        stored[node] = snapshots;
        state_at[node] = current;
    }

    let blocks: HashMap<BlockHash, BlockNode> = (1..scenario.node_count)
        .map(|node| {
            let candidates: HashSet<ProtocolPoolKey> = scenario.events[node]
                .iter()
                .map(|(pool_index, _)| pool_of(*pool_index).key)
                .collect();
            (
                dh(node),
                BlockNode {
                    parent_hash: dh(scenario.parents[node]),
                    logs_bloom: None,
                    pool_logs: PoolLogsStatus::Resolved(candidates),
                    pool_snapshots: stored[node].clone(),
                    pool_data_failures: HashMap::new(),
                },
            )
        })
        .collect();

    State {
        blocks: LegacyGraph(blocks),
        canonical_tip: dh(scenario.target),
        pending_requests: PendingRequests::new(),
        finalized_state: FinalizedState {
            block_hash: dh(0),
            pool_snapshots: base,
        },
        pool_registry: registry(&[]),
        token_registry: TokenRegistry::new(),
        tick: Tick::initial(),
        streamed_logs: HashMap::new(),
        // Unused here: this scaffolding drives the new graph directly, not via `State.log_shadow`.
        log_shadow: crate::kernel::LogGraphShadow::new(dh(0), HashMap::new()),
    }
}

/// The legacy finalization path: builds the pre-finalization state, then finalizes to the target.
fn build_legacy(scenario: &Scenario) -> State {
    legacy_state(scenario).with_finalized_block_observed(CHAIN, dh(scenario.target))
}

/// The legacy optimization overlay at the canonical tip, merged over the finalized base — the read
/// the kernel's optimization dispatch performs today (`latest_complete_pool_state_update` +
/// `resolve_complete_pool_states`, then overlaid on the finalized snapshots).
fn legacy_optimization_snapshot(state: &State) -> HashMap<PoolRef, PoolState> {
    let update = state
        .latest_complete_pool_state_update(CHAIN)
        .expect("an all-resolved path yields a complete overlay");
    let overlay = state
        .resolve_complete_pool_states(&update)
        .expect("every overlay location is present");
    let mut merged = base_snapshot();
    for (pool, pool_state) in overlay {
        merged.insert(pool, pool_state.clone());
    }
    merged
}

/// Every connected node's parent chain reaches the anchor (the post-reanchor form of T2).
fn connected_reaches_anchor(graph: &NewGraph, start: BlockHash) -> bool {
    let mut current = start;
    let mut visited = HashSet::new();
    loop {
        match graph.connected(current) {
            Some(connected) => match &connected.parent {
                AnchoredRef::Anchor => return true,
                AnchoredRef::Block(ConnectedHash(parent)) => {
                    if !visited.insert(current) {
                        return false;
                    }
                    current = *parent;
                }
            },
            None => return false,
        }
    }
}

// ---- properties ----------------------------------------------------------------------------------

proptest! {
    /// Two-sided: on an all-`Complete` canonical path, the new fold/re-root produces the same
    /// finalized snapshot (on base-resident pools) as the legacy finalization, and both advance to
    /// the same target.
    #[test]
    fn new_finalization_matches_legacy(scenario in scenario_strategy()) {
        let base = base_snapshot();
        let (new_graph, new_snapshot) = build_new(&scenario)
            .reanchored_to(ConnectedHash(dh(scenario.target)), &base, Address::ZERO)
            .expect("an all-Complete path must fold");
        let legacy = build_legacy(&scenario);

        // Full advance — no partial compaction, since every path block is Complete.
        prop_assert_eq!(legacy.finalized_state.block_hash, dh(scenario.target));

        // Snapshots agree on base-resident keys.
        for pool in base.keys() {
            prop_assert_eq!(
                new_snapshot.get(pool),
                legacy.finalized_state.pool_snapshots.get(pool),
                "mismatch for pool {:?}", pool
            );
        }

        // New post-state: anchor advanced, no pending survive, every node reaches the anchor.
        prop_assert_eq!(new_graph.anchor, dh(scenario.target));
        prop_assert!(
            new_graph.nodes.values().all(|node| matches!(node, Node::Connected(_))),
            "no pending nodes may survive finalization"
        );
        for hash in new_graph.nodes.keys() {
            prop_assert!(connected_reaches_anchor(&new_graph, *hash), "node {:?} must reach anchor", hash);
        }

        // Legacy post-state: the finalized block is no longer a recent block.
        prop_assert!(!legacy.blocks.0.contains_key(&dh(scenario.target)));
    }
}

proptest! {
    /// Two-sided: the new graph's `optimization_pool_states` (fold to the observed head under
    /// `AllowStreamed`, reading best-effort `Streamed` logs as well as `Complete`) produces the same
    /// per-pool overlay — on base-resident keys — as the legacy optimization read at the canonical
    /// tip. Blocks are a random Streamed/Complete mix; because `AllowStreamed` must read both, a
    /// dropped stream would diverge from the legacy overlay (which stores snapshots regardless of
    /// log kind), so the mix directly exercises the authority seam. No finalization: the fold spans
    /// the unfinalized region `anchor → head`, unlike the Stage-2 finalization test.
    #[test]
    fn optimization_view_matches_legacy(scenario in scenario_strategy()) {
        let base = base_snapshot();
        let new_snapshot = build_new_optimization(&scenario).optimization_pool_states(&base);
        let legacy_snapshot = legacy_optimization_snapshot(&legacy_state(&scenario));

        for pool in base.keys() {
            prop_assert_eq!(
                new_snapshot.get(pool),
                legacy_snapshot.get(pool),
                "optimization overlay mismatch for pool {:?}", pool
            );
        }
    }
}

/// Coalesces `is_complete[i] == false` positions (block number `i + 1`) into inclusive ranges.
fn expected_hole_ranges(is_complete: &[bool]) -> Vec<(u64, u64)> {
    let mut ranges: Vec<(u64, u64)> = Vec::new();
    for (index, &done) in is_complete.iter().enumerate() {
        if done {
            continue;
        }
        let number = (index + 1) as u64;
        match ranges.last_mut() {
            Some(last) if number == last.1 + 1 => last.1 = number,
            _ => ranges.push((number, number)),
        }
    }
    ranges
}

/// A linear connected chain block `i` (number `i`, `1..=len`) with a bloom that hits the watched base
/// pool, `Complete` (empty logs) or `Unknown` per `is_complete`.
fn build_linear(is_complete: &[bool]) -> NewGraph {
    let mut nodes = HashMap::new();
    let mut parent = AnchoredRef::Anchor;
    for (index, &done) in is_complete.iter().enumerate() {
        let number = (index + 1) as u64;
        let hash = dh(number as usize);
        let logs = if done {
            BlockLogs::Complete(BTreeMap::new())
        } else {
            BlockLogs::Unknown
        };
        nodes.insert(
            hash,
            Node::Connected(ConnectedNode {
                parent,
                data: BlockData {
                    number,
                    logs_bloom: Some(Bloom::repeat_byte(0xFF)),
                    logs,
                },
            }),
        );
        parent = AnchoredRef::Block(ConnectedHash(hash));
    }
    NewGraph {
        anchor: dh(0),
        nodes,
        observed_head: dh(is_complete.len()),
    }
}

proptest! {
    /// One-sided A4: an incomplete (non-`Complete`, bloom-hit) path is refused with the exact hole
    /// ranges and the graph handed back unchanged; a fully-complete path folds.
    #[test]
    fn incomplete_path_is_refused_with_hole_ranges(
        is_complete in prop::collection::vec(any::<bool>(), 1..8)
    ) {
        let base = base_snapshot();
        let graph = build_linear(&is_complete);
        let expected = expected_hole_ranges(&is_complete);
        let node_count = is_complete.len();

        let result = graph.reanchored_to(ConnectedHash(dh(node_count)), &base, Address::ZERO);

        if expected.is_empty() {
            prop_assert!(result.is_ok(), "a fully-complete path must fold");
        } else {
            match result {
                Err(ReanchorError::Incomplete(returned, ranges)) => {
                    let got: Vec<(u64, u64)> = ranges.iter().map(|range| (range.from, range.to)).collect();
                    prop_assert_eq!(got, expected);
                    // Graph handed back unchanged.
                    prop_assert_eq!(returned.anchor, dh(0));
                    prop_assert_eq!(returned.nodes.len(), node_count);
                }
                Ok(_) => prop_assert!(false, "an incomplete path must be refused"),
            }
        }
    }
}

// ---- targeted unit tests -------------------------------------------------------------------------

#[test]
fn reanchor_to_anchor_is_noop() {
    let base = base_snapshot();
    let graph = NewGraph::new(dh(0))
        .with_block(dh(1), dh(0), 1, Bloom::repeat_byte(0xFF))
        .expect("admission succeeds");

    let (graph, snapshot) = graph
        .reanchored_to(ConnectedHash(dh(0)), &base, Address::ZERO)
        .expect("reanchoring to the current anchor is a no-op");

    assert_eq!(graph.anchor, dh(0));
    assert_eq!(snapshot, base);
}

#[test]
fn registry_only_pool_is_excluded_from_new_fold_but_present_in_legacy() {
    let base = base_snapshot();
    let extra = PoolRef::uniswap_v3(Address::with_last_byte(0xC1), CHAIN);
    // Self-seeding swaps (absolute), so a snapshot is derivable for both pools with no prior base.
    let base_swap = PoolLogEvent::Swap {
        sqrt_price_x96: U160::from(7_000u128),
        tick: tk(4),
        liquidity: 55_555,
    };
    let extra_swap = PoolLogEvent::Swap {
        sqrt_price_x96: U160::from(9_000u128),
        tick: tk(6),
        liquidity: 4_242,
    };
    let base0_state = ps(7_000, 4, 55_555);
    let extra_state = ps(9_000, 6, 4_242);

    // New graph: one block off the anchor carrying both pools' swaps.
    let mut graph = NewGraph::new(dh(0))
        .with_block(dh(1), dh(0), 1, Bloom::repeat_byte(0xFF))
        .expect("admission succeeds");
    let logs = complete(vec![
        pool_log(pool_of(0), base_swap.clone()),
        pool_log(extra, extra_swap.clone()),
    ]);
    match graph.nodes.get_mut(&dh(1)) {
        Some(Node::Connected(connected)) => connected.data.logs = logs,
        _ => panic!("block must be connected"),
    }
    let (_, new_snapshot) = graph
        .reanchored_to(ConnectedHash(dh(1)), &base, Address::ZERO)
        .expect("path folds");

    // The registry-only pool has no base, so the new fold excludes it; the base pool is folded.
    assert!(!new_snapshot.contains_key(&extra));
    assert_eq!(new_snapshot.get(&pool_of(0)), Some(&base0_state));

    // Legacy: the same block, with both snapshots planted and both pools verified.
    let candidates = HashSet::from([pool_of(0).key, extra.key]);
    let snapshots = HashMap::from([(pool_of(0), base0_state.clone()), (extra, extra_state.clone())]);
    let blocks = HashMap::from([(
        dh(1),
        BlockNode {
            parent_hash: dh(0),
            logs_bloom: None,
            pool_logs: PoolLogsStatus::Resolved(candidates),
            pool_snapshots: snapshots,
            pool_data_failures: HashMap::new(),
        },
    )]);
    let legacy = State {
        blocks: LegacyGraph(blocks),
        canonical_tip: dh(1),
        pending_requests: PendingRequests::new(),
        finalized_state: FinalizedState {
            block_hash: dh(0),
            pool_snapshots: base,
        },
        pool_registry: registry(&[extra.key]),
        token_registry: TokenRegistry::new(),
        tick: Tick::initial(),
        streamed_logs: HashMap::new(),
        // Unused here: this scaffolding drives the new graph directly, not via `State.log_shadow`.
        log_shadow: crate::kernel::LogGraphShadow::new(dh(0), HashMap::new()),
    }
    .with_finalized_block_observed(CHAIN, dh(1));

    assert_eq!(legacy.finalized_state.block_hash, dh(1));
    // Legacy keeps the registry-only pool; both agree on the base pool.
    assert_eq!(legacy.finalized_state.pool_snapshots.get(&extra), Some(&extra_state));
    assert_eq!(legacy.finalized_state.pool_snapshots.get(&pool_of(0)), Some(&base0_state));
}
