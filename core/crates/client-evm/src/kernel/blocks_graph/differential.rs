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
//!  - **Two-sided exact** (`new_finalization_matches_legacy`): on an **all-`Complete`** rendering of
//!    the canonical path, both sides finalize fully to the target and produce the same finalized
//!    snapshot on every tracked pool.
//!  - **Two-sided partial** (`partial_finalization_matches_legacy`): with mixed per-block log
//!    authority (`Complete`/`Streamed`/`Unknown`, bloom hit/clear — [`BlockKind`]), `finalized_to`
//!    and legacy partial compaction advance to the **same frontier** and agree on the snapshot there.
//!  - **Two-sided optimization** (`optimization_view_matches_legacy`): the `AllowStreamed` fold to
//!    the observed head matches the overlay of a fully-informed legacy state, and both report the
//!    same frontier hash.
//!  - **One-sided A4** (`incomplete_path_is_refused_with_hole_ranges`): new-only — an incomplete path
//!    is refused with the exact hole ranges; legacy has no equivalent (it silently compacts short).
//!
//! Scenarios may also plant a **post-bootstrap-discovered pool** — verified but absent from `base` —
//! whose per-block runs are always Swap-led (absolute), so both sides derive it identically (the
//! Blocker-1 seeding class). Delta-only (Mint-led) discovered pools are deliberately *not* generated:
//! that case genuinely diverges until Blocker 1b lands — see
//! `delta_only_new_pool_new_graph_advances_but_legacy_waits`.
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

/// The fixed tracked pool set — all v3. The first [`BASE_POOL_COUNT`] are present in the finalized
/// snapshot; the last is the post-bootstrap-discovered pool (verified, absent from `base`), which
/// the folds must absolute-seed from its Swap-led runs.
const POOL_BYTES: [u8; 4] = [0xA1, 0xA2, 0xA3, 0xA4];
const BASE_POOL_COUNT: usize = 3;
const DISCOVERED_POOL_INDEX: usize = 3;

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

/// The verified tracked set spanning exactly the base pools — verified == base keeps a fold
/// restricted to them. Used by the targeted tests below (which widen it explicitly) and the A4
/// range property (whose linear chains carry no discovered pool).
fn base_verified() -> HashSet<PoolRef> {
    base_snapshot().keys().copied().collect()
}

/// The verified tracked set the generated scenarios hand the folds: the base pools plus the
/// discovered pool — so every generated fold exercises watching and absolute-seeding a pool with no
/// `base` entry, and comparisons span all tracked pools.
fn scenario_verified() -> HashSet<PoolRef> {
    (0..POOL_BYTES.len()).map(pool_of).collect()
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

/// A block's `(pool index, event)` list rendered as new-graph `Streamed` (best-effort WS) logs.
fn streamed_events(events: &[(usize, PoolLogEvent)]) -> BlockLogs {
    streamed(
        events
            .iter()
            .map(|(index, event)| pool_log(pool_of(*index), event.clone()))
            .collect(),
    )
}

/// A generated block's log authority — the per-block axis both sides render from.
///
/// `UnknownClear` models a block whose logs were never fetched but whose header bloom proves no
/// watched pool emitted: legacy's scheduler promotes it to `Resolved` (empty) without a fetch, and
/// the new graph's frontier walks past it — it stops no scan on either side. `UnknownHit` is the
/// true hole: no logs, and a bloom that may touch a watched pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockKind {
    Complete,
    Streamed,
    UnknownClear,
    UnknownHit,
}

/// The header bloom a block is admitted with: zero (proven untouched) for `UnknownClear`, saturated
/// (may touch everything) otherwise. Only `Unknown` blocks ever have their bloom consulted, but the
/// saturated default keeps every non-clear block conservatively bloom-hit.
fn admission_bloom(kind: BlockKind) -> Bloom {
    match kind {
        BlockKind::UnknownClear => Bloom::default(),
        _ => Bloom::repeat_byte(0xFF),
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
    /// Per-node log authority for the mixed-authority builders/interpreters; `kind[0]` (the anchor)
    /// is unused. `Unknown*` nodes carry no events (their content is unreadable on both sides, and a
    /// clear bloom *proves* emptiness). The exact-finalization property ignores this axis and
    /// renders everything `Complete`.
    kind: Vec<BlockKind>,
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
        0usize..BASE_POOL_COUNT,
        prop_oneof![swap_strategy(), mint_strategy()],
    )
}

/// The discovered pool's per-block run: always Swap-led (absolute), so both sides can derive a state
/// for it with no prior `base` entry regardless of which branch or admission order the block lands
/// in. Delta-only (Mint-led) runs are the Blocker-1b divergence and are deliberately not generated.
fn discovered_run_strategy() -> impl Strategy<Value = Vec<PoolLogEvent>> {
    (
        swap_strategy(),
        prop::collection::vec(prop_oneof![swap_strategy(), mint_strategy()], 0..2),
    )
        .prop_map(|(lead, tail)| {
            let mut run = vec![lead];
            run.extend(tail);
            run
        })
}

/// `Complete` is weighted dominant so folds usually advance deep enough to exercise the seams the
/// rarer kinds plant (streamed reads, holes, bloom-proven skips).
fn block_kind_strategy() -> impl Strategy<Value = BlockKind> {
    prop_oneof![
        4 => Just(BlockKind::Complete),
        2 => Just(BlockKind::Streamed),
        1 => Just(BlockKind::UnknownClear),
        1 => Just(BlockKind::UnknownHit),
    ]
}

fn scenario_strategy() -> impl Strategy<Value = Scenario> {
    (2usize..10)
        .prop_flat_map(|node_count| {
            let admit: Vec<usize> = (1..node_count).collect();
            (
                Just(node_count),
                prop::collection::vec(any::<usize>(), node_count - 1),
                prop::collection::vec(
                    (
                        prop::collection::vec(event_strategy(), 0..3),
                        prop::option::weighted(0.3, discovered_run_strategy()),
                    ),
                    node_count - 1,
                ),
                1usize..node_count,
                Just(admit).prop_shuffle(),
                prop::collection::vec(block_kind_strategy(), node_count - 1),
            )
        })
        .prop_map(
            |(node_count, parent_choices, node_events, target, order, kind_choices)| {
                let parents: Vec<usize> = (0..node_count)
                    .map(|index| if index == 0 { 0 } else { parent_choices[index - 1] % index })
                    .collect();
                let mut number = vec![0u64; node_count];
                for index in 1..node_count {
                    number[index] = number[parents[index]] + 1;
                }
                let mut kind = vec![BlockKind::Complete];
                kind.extend(kind_choices);
                let mut events = vec![Vec::new()];
                for (node, (base_events, discovered_run)) in node_events.into_iter().enumerate() {
                    // `Unknown` blocks carry no events: a clear bloom *proves* the block is empty,
                    // and a bloom-hit block's content is unreadable on both sides — planting some
                    // anyway would leak data neither interpreter is allowed to see.
                    if matches!(
                        kind[node + 1],
                        BlockKind::UnknownClear | BlockKind::UnknownHit
                    ) {
                        events.push(Vec::new());
                        continue;
                    }
                    let mut block_events = base_events;
                    if let Some(run) = discovered_run {
                        block_events
                            .extend(run.into_iter().map(|event| (DISCOVERED_POOL_INDEX, event)));
                    }
                    events.push(block_events);
                }
                Scenario {
                    node_count,
                    parents,
                    number,
                    events,
                    target,
                    order,
                    kind,
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

/// Builds the new graph rendering each block per its [`BlockKind`]: admit every block in `order`
/// with the kind's bloom ([`admission_bloom`]), then plant `Complete`/`Streamed` payloads —
/// `Unknown*` blocks keep the admission default (`BlockLogs::Unknown`, bloom deciding hole vs skip).
fn build_new_kinds(scenario: &Scenario) -> NewGraph {
    let mut graph = NewGraph::new(dh(0));
    for &node in &scenario.order {
        graph = graph
            .with_block(
                dh(node),
                dh(scenario.parents[node]),
                scenario.number[node],
                admission_bloom(scenario.kind[node]),
            )
            .expect("generated admission succeeds");
    }
    for node in 1..scenario.node_count {
        let logs = match scenario.kind[node] {
            BlockKind::Complete => complete_events(&scenario.events[node]),
            BlockKind::Streamed => streamed_events(&scenario.events[node]),
            BlockKind::UnknownClear | BlockKind::UnknownHit => continue,
        };
        match graph.nodes.get_mut(&dh(node)) {
            Some(Node::Connected(connected)) => connected.data.logs = logs,
            _ => panic!("every generated block must be connected after full admission"),
        }
    }
    graph
}

/// The canonical node-index path anchor → target (anchor excluded), derived from `parents`.
fn canonical_path(scenario: &Scenario) -> Vec<usize> {
    let mut path = Vec::new();
    let mut node = scenario.target;
    while node != 0 {
        path.push(node);
        node = scenario.parents[node];
    }
    path.reverse();
    path
}

/// The independently-computed frontier both sides must stop at: the last canonical block before the
/// first block whose kind `blocks` the given fold (node `0` — the anchor — when the very first path
/// block already blocks). Guards against a shared bug in the two interpreters masking a divergence.
fn expected_frontier(scenario: &Scenario, blocks: impl Fn(BlockKind) -> bool) -> usize {
    let mut frontier = 0;
    for &node in &canonical_path(scenario) {
        if blocks(scenario.kind[node]) {
            break;
        }
        frontier = node;
    }
    frontier
}

/// Builds the pre-finalization legacy `State`, planting each block's absolute per-pool snapshots via
/// an incremental `derive_pool_state` fold (block = parent snapshot + block logs) and setting the
/// canonical tip to the target. Shared by all comparisons; `kinds` decides each block's rendering
/// (callers pass `scenario.kind`, an all-`Complete` vector, or a "fully informed" mapping):
///  - `Complete` → `Resolved` candidates + derived snapshots (authoritative logs arrived);
///  - `Streamed` → `Partial` candidates, no snapshots (subscription logs are provisional and never
///    unblock the legacy scans);
///  - `UnknownClear` → `Resolved` empty, no snapshots (the production scheduler promotes a
///    bloom-clear block without a fetch — `resolve_empty_candidates` in the kernel);
///  - `UnknownHit` → `Unknown`, stopping every legacy scan.
fn legacy_state(scenario: &Scenario, kinds: &[BlockKind]) -> State {
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
            // Base pools fold from `base`; the discovered pool's runs are Swap-led by construction.
            let folded = derive_pool_state(current.get(&pool), &run)
                .expect("every generated per-block run is derivable");
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
            let (pool_logs, pool_snapshots) = match kinds[node] {
                BlockKind::Complete => {
                    (PoolLogsStatus::Resolved(candidates), stored[node].clone())
                }
                BlockKind::Streamed => (PoolLogsStatus::Partial(candidates), HashMap::new()),
                // `Unknown*` blocks have no events, so `candidates` is empty either way; the clear
                // case renders as the scheduler's fetch-free `Resolved` promotion.
                BlockKind::UnknownClear => (PoolLogsStatus::Resolved(candidates), HashMap::new()),
                BlockKind::UnknownHit => (PoolLogsStatus::Unknown, HashMap::new()),
            };
            (
                dh(node),
                BlockNode {
                    parent_hash: dh(scenario.parents[node]),
                    logs_bloom: None,
                    pool_logs,
                    pool_snapshots,
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

/// The legacy finalization path on an all-`Complete` rendering (the exact-equivalence premise):
/// builds the pre-finalization state, then finalizes to the target.
fn build_legacy(scenario: &Scenario) -> State {
    legacy_state(scenario, &vec![BlockKind::Complete; scenario.node_count])
        .with_finalized_block_observed(CHAIN, dh(scenario.target))
}

/// The legacy optimization overlay at the canonical tip, merged over the finalized base, plus the
/// reference block hash the overlay is valid at — the read the kernel's optimization dispatch
/// performs today (`latest_complete_pool_state_update` + `resolve_complete_pool_states`, then
/// overlaid on the finalized snapshots).
fn legacy_optimization_snapshot(state: &State) -> (HashMap<PoolRef, PoolState>, BlockHash) {
    let update = state
        .latest_complete_pool_state_update(CHAIN)
        .expect("a connected path yields an overlay descriptor");
    let overlay = state
        .resolve_complete_pool_states(&update)
        .expect("every overlay location is present");
    let mut merged = base_snapshot();
    for (pool, pool_state) in overlay {
        merged.insert(pool, pool_state.clone());
    }
    (merged, update.block_hash)
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
    /// Two-sided: on an all-`Complete` rendering of the canonical path, the new fold/re-root
    /// produces the same finalized snapshot as the legacy finalization — on every tracked pool,
    /// including the discovered pool both sides must introduce identically — and both advance to
    /// the same target.
    #[test]
    fn new_finalization_matches_legacy(scenario in scenario_strategy()) {
        let base = base_snapshot();
        let verified = scenario_verified();
        let (new_graph, new_snapshot) = build_new(&scenario)
            .reanchored_to(ConnectedHash(dh(scenario.target)), &base, &verified, Address::ZERO)
            .expect("an all-Complete path must fold");
        let legacy = build_legacy(&scenario);

        // Full advance — no partial compaction, since every path block is Complete.
        prop_assert_eq!(legacy.finalized_state.block_hash, dh(scenario.target));

        // Snapshots agree on every tracked pool (absence must agree too: a discovered pool whose
        // logs sit off the canonical path stays out of both snapshots).
        for pool in &verified {
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
    /// per-pool overlay — on every tracked pool, discovered included — and the same frontier hash as
    /// the legacy optimization read at the canonical tip. Legacy is rendered **fully informed** on
    /// `Streamed` blocks (`Resolved` + snapshots): the fold reads streamed logs, and its equivalence
    /// target is the overlay legacy builds once the same logs arrive authoritatively — a dropped
    /// stream would therefore diverge. A bloom-hit `Unknown` block stops both sides at the same
    /// frontier (legacy's scan break, the new fold's hole); a bloom-clear one stops neither.
    #[test]
    fn optimization_view_matches_legacy(scenario in scenario_strategy()) {
        let base = base_snapshot();
        let verified = scenario_verified();
        let frontier = expected_frontier(
            &scenario,
            |kind| matches!(kind, BlockKind::UnknownHit),
        );

        // The changed-pool overlay plus the frontier hash the reserves are valid at; the overlay is
        // merged onto `base` for comparison (an untouched pool reads through to `base`), mirroring
        // the production optimization consumer.
        let (new_overlay, new_frontier) = build_new_kinds(&scenario)
            .with_observed_head(dh(scenario.target))
            .optimization_pool_states(&base, &verified, Address::ZERO);

        let informed: Vec<BlockKind> = scenario
            .kind
            .iter()
            .map(|kind| match kind {
                BlockKind::Streamed => BlockKind::Complete,
                other => *other,
            })
            .collect();
        let (legacy_snapshot, legacy_frontier) =
            legacy_optimization_snapshot(&legacy_state(&scenario, &informed));

        prop_assert_eq!(new_frontier, dh(frontier), "new frontier");
        prop_assert_eq!(legacy_frontier, dh(frontier), "legacy frontier");
        for pool in &verified {
            prop_assert_eq!(
                new_overlay.get(pool).or_else(|| base.get(pool)),
                legacy_snapshot.get(pool),
                "optimization overlay mismatch for pool {:?}", pool
            );
        }
    }
}

proptest! {
    /// Two-sided partial compaction under mixed log authority: `finalized_to` — the production
    /// shadow finalization entry — and legacy `with_finalized_block_observed` advance to the same
    /// frontier (the last canonical block before the first `Streamed` or bloom-hit `Unknown` block)
    /// and agree on the finalized snapshot there, discovered pool included. A bloom-clear `Unknown`
    /// block stops neither side: legacy's scheduler promotes it to `Resolved` (empty) without a
    /// fetch, and the new frontier walks past a proven-untouched block.
    #[test]
    fn partial_finalization_matches_legacy(scenario in scenario_strategy()) {
        let base = base_snapshot();
        let verified = scenario_verified();
        let frontier = expected_frontier(
            &scenario,
            |kind| matches!(kind, BlockKind::Streamed | BlockKind::UnknownHit),
        );

        let (new_graph, new_snapshot) = build_new_kinds(&scenario)
            .finalized_to(dh(scenario.target), &base, &verified, Address::ZERO);
        let legacy = legacy_state(&scenario, &scenario.kind)
            .with_finalized_block_observed(CHAIN, dh(scenario.target));

        prop_assert_eq!(new_graph.anchor, dh(frontier), "new frontier");
        prop_assert_eq!(legacy.finalized_state.block_hash, dh(frontier), "legacy frontier");
        for pool in &verified {
            prop_assert_eq!(
                new_snapshot.get(pool),
                legacy.finalized_state.pool_snapshots.get(pool),
                "finalized snapshot mismatch for pool {:?}", pool
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

        let result = graph.reanchored_to(ConnectedHash(dh(node_count)), &base, &base_verified(), Address::ZERO);

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
        .reanchored_to(ConnectedHash(dh(0)), &base, &base_verified(), Address::ZERO)
        .expect("reanchoring to the current anchor is a no-op");

    assert_eq!(graph.anchor, dh(0));
    assert_eq!(snapshot, base);
}

#[test]
fn registry_verified_new_pool_is_absolute_seeded_into_new_fold() {
    // Blocker 1: a pool discovered after bootstrap (absent from `base`) but registry-verified is
    // absolute-seeded into the new fold from its own Swap via `derive_pool_state(None, run)` — so it
    // now enters the finalized snapshot, matching legacy which snapshots the path-discovered pool.
    let base = base_snapshot();
    let extra = PoolRef::uniswap_v3(Address::with_last_byte(0xC1), CHAIN);
    let mut verified = base_verified();
    verified.insert(extra);
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
        .reanchored_to(ConnectedHash(dh(1)), &base, &verified, Address::ZERO)
        .expect("path folds");

    // The verified new pool is absolute-seeded; the base pool is folded.
    assert_eq!(new_snapshot.get(&extra), Some(&extra_state));
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
    // Both paths now keep the new pool and agree on the base pool.
    assert_eq!(legacy.finalized_state.pool_snapshots.get(&extra), Some(&extra_state));
    assert_eq!(
        new_snapshot.get(&extra),
        legacy.finalized_state.pool_snapshots.get(&extra)
    );
    assert_eq!(legacy.finalized_state.pool_snapshots.get(&pool_of(0)), Some(&base0_state));
}

#[test]
fn delta_only_new_pool_new_graph_advances_but_legacy_waits() {
    // Blocker 1b divergence (one-sided, documented). A verified new pool (absent from `base`) whose
    // only path log is a Mint is un-baseable: `derive_pool_state(None, [Mint])` is `None`. The two
    // paths deliberately differ here, and Blocker 1b must reconcile them before the legacy delete:
    //   - New graph: a `Complete` block is never a finalization hole, so `reanchored_to` advances the
    //     anchor past it and simply skips the un-baseable pool (it will be seeded at the anchor via
    //     `GetPoolData` — Blocker 1b).
    //   - Legacy: `latest_complete_pool_state_update_from` only marks a block complete once every
    //     verified candidate has a snapshot (`invalid_pools.is_empty()`), so the un-baseable pool
    //     *blocks* finalization — legacy waits for `GetPoolData` while the block is still unfinalized.
    let base = base_snapshot();
    let extra = PoolRef::uniswap_v3(Address::with_last_byte(0xC2), CHAIN);
    let mut verified = base_verified();
    verified.insert(extra);

    let base_swap = PoolLogEvent::Swap {
        sqrt_price_x96: U160::from(7_000u128),
        tick: tk(4),
        liquidity: 55_555,
    };
    let base0_state = ps(7_000, 4, 55_555);
    let extra_mint = PoolLogEvent::Mint {
        tick_lower: tk(-10),
        tick_upper: tk(10),
        amount: 1_000,
    };

    // New graph: a block carrying the base pool's swap plus the new pool's delta-only mint.
    let mut graph = NewGraph::new(dh(0))
        .with_block(dh(1), dh(0), 1, Bloom::repeat_byte(0xFF))
        .expect("admission succeeds");
    let logs = complete(vec![
        pool_log(pool_of(0), base_swap.clone()),
        pool_log(extra, extra_mint.clone()),
    ]);
    match graph.nodes.get_mut(&dh(1)) {
        Some(Node::Connected(connected)) => connected.data.logs = logs,
        _ => panic!("block must be connected"),
    }
    let (new_graph, new_snapshot) = graph
        .reanchored_to(ConnectedHash(dh(1)), &base, &verified, Address::ZERO)
        .expect("a Complete path folds (the delta-only pool is simply skipped)");

    // New graph advanced, seeded the base pool, and skipped the un-baseable new pool.
    assert_eq!(new_graph.anchor, dh(1));
    assert!(!new_snapshot.contains_key(&extra));
    assert_eq!(new_snapshot.get(&pool_of(0)), Some(&base0_state));

    // Legacy: the block snapshots only the base pool (its derivation yields nothing for the
    // delta-only new pool), so the un-baseable verified candidate holds finalization at the anchor.
    let candidates = HashSet::from([pool_of(0).key, extra.key]);
    let snapshots = HashMap::from([(pool_of(0), base0_state.clone())]);
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
        log_shadow: crate::kernel::LogGraphShadow::new(dh(0), HashMap::new()),
    }
    .with_finalized_block_observed(CHAIN, dh(1));

    // Legacy did NOT advance: the un-baseable pool blocks the block from being "complete".
    assert_eq!(legacy.finalized_state.block_hash, dh(0));
}
