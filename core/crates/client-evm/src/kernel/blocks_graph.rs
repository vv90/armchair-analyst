#![cfg_attr(not(test), allow(dead_code))]

use std::collections::{BTreeMap, HashMap};

use alloy::primitives::{BlockHash, Bloom};

use crate::PoolLog;

/// The volatile block tree: recent, non-finalized blocks rooted at the finalized anchor.
///
/// Invariants are split between STRUCTURAL (unrepresentable — the types admit no invalid value) and
/// RUNTIME (upheld by the transition methods and covered by tests). See
/// `KERNEL_BLOCKS_GRAPH_INVARIANTS.md` for the full contract.
///
/// Structural here:
///  - `anchor` is the single root and the sole home of the finalized hash (no node carries it).
///  - one `nodes` entry per hash (key uniqueness), and that entry is `Connected` XOR `Pending` —
///    a block cannot be in both states, nor in neither.
///  - a connected node's parent and the canonical tip are `AnchoredRef` — the "missing/unknown
///    parent" case the old `while current != finalized { … else break }` walk handled does not
///    exist.
///  - per-block logs are a `BTreeMap` keyed by intra-block index: deduped and ordered for free; the
///    index lives only in the key.
///  - non-anchor blocks carry no pool snapshot (absolute state is reconstructed by fold-on-demand;
///    the finalized snapshot stays in `FinalizedState`, its sole home).
///
/// Runtime (upheld by the methods, tested): referential integrity of every `ConnectedHash` /
/// `AnchoredRef`, hence parent-walk termination and acyclicity; that a `Pending` node is genuinely
/// not yet anchor-reachable; bloom consistency; the monotone `Unknown → Streamed → Complete` log
/// ladder; the finalization re-root/prune and its foldability gate; and the `pending` bound.
#[derive(Debug)]
struct BlocksGraph {
    anchor: BlockHash,
    nodes: HashMap<BlockHash, Node>,
    canonical_tip: AnchoredRef,
}

/// A block is connected XOR pending. One keyed entry, exactly one of these states — disjointness is
/// structural, not asserted.
#[derive(Debug)]
enum Node {
    Connected(ConnectedNode),
    Pending(PendingNode),
}

/// A reference into the connected tree. TOTAL: there is no "missing" variant, because a connected
/// node's parent and the canonical tip are connected by construction. Replaces the
/// `while current != finalized { … else break }` walk pattern with an exhaustive `match`.
#[derive(Debug)]
enum AnchoredRef {
    Anchor,
    Block(ConnectedHash),
}

/// A `BlockHash` proven (by the minting path) to be a `Connected` node. A construction-time proof
/// that a lookup will hit, minted only by the insertion/promotion paths and consumed within the same
/// transition. Referential integrity is a RUNTIME property (this is a plain newtype) and is tested.
#[derive(Debug)]
struct ConnectedHash(BlockHash);

/// A block on a complete path to the anchor. Differs from `PendingNode` ONLY in the parent type:
/// the parent is a proven `AnchoredRef`, not a raw hash.
#[derive(Debug)]
struct ConnectedNode {
    parent: AnchoredRef,
    data: BlockData,
}

/// A block whose ancestry is not yet connected to the anchor (its parent is absent, or present but
/// itself pending). Parent is an unproven raw hash. Promotion re-types this parent into an
/// `AnchoredRef` and moves the node into the `Connected` variant.
#[derive(Debug)]
struct PendingNode {
    parent: BlockHash,
    data: BlockData,
}

/// Per-block payload, independent of the block's connectivity. Shared by connected and pending
/// nodes so promotion never reshapes data — it only re-types the parent reference. Carries no
/// snapshot: absolute pool state is folded on demand, never stored off the anchor.
#[derive(Debug)]
struct BlockData {
    /// Header `logsBloom` when the block entered from a header; `None` for header-less nodes.
    logs_bloom: Option<Bloom>,
    logs: BlockLogs,
}

/// Per-block decoded pool logs, keyed by intra-block index. The key is the ordering/dedup index; new
/// logic reads it from the key and never from `PoolLog::log_index` (deprecated, removed at swap).
// `Streamed`/`Complete` are constructed by the log-merge transition (invariant L5), a later
// increment; only `Unknown` is reachable through `with_block` today.
#[derive(Debug)]
#[allow(dead_code)]
enum BlockLogs {
    Unknown,
    Streamed(BTreeMap<u64, PoolLog>),
    Complete(BTreeMap<u64, PoolLog>),
}

/// Why an admission was rejected. Every non-fatal variant hands the (unmodified) graph back, since
/// `with_block` takes `self` by value. Mirrors the legacy `with_new_block` rejections, minus
/// `CycleDetected`: a freshly inserted hash is a leaf nothing points to, so admission cannot create
/// a cycle (self-parent — the one degenerate case — is rejected up front).
// The returned graph is consumed by the kernel's call sites at the swap (Stage 4); until then the
// tests only assert the rejection variant, not the recovered graph.
#[derive(Debug)]
#[allow(dead_code)]
enum NewBlockError {
    /// `hash == parent`.
    SelfParent(BlocksGraph),
    /// `hash == anchor` — a re-announce of the finalized block (legacy `ExistingBlock`).
    AnchorReadmit(BlocksGraph),
    /// `hash` is already present with the same parent — an idempotent no-op (legacy `ExistingBlock`).
    DuplicateBlock(BlocksGraph),
    /// `hash` is already present with a *different* parent — a fatal conflict for the caller.
    ConflictingParent(BlocksGraph),
}

/// The raw hash a connected node's parent reference points at (the anchor for `Anchor`).
fn connected_parent_hash(parent: &AnchoredRef, anchor: BlockHash) -> BlockHash {
    match parent {
        AnchoredRef::Anchor => anchor,
        AnchoredRef::Block(ConnectedHash(hash)) => *hash,
    }
}

impl BlocksGraph {
    fn new(anchor: BlockHash) -> BlocksGraph {
        BlocksGraph {
            anchor,
            nodes: HashMap::new(),
            canonical_tip: AnchoredRef::Anchor,
        }
    }

    /// True when no recent blocks are tracked yet — only the anchor exists.
    fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// The connected node for `hash`, if present and connected. O(1), replacing the legacy
    /// parent-walk used to (re)decide connectivity.
    fn connected(&self, hash: BlockHash) -> Option<&ConnectedNode> {
        match self.nodes.get(&hash) {
            Some(Node::Connected(node)) => Some(node),
            _ => None,
        }
    }

    /// The per-block payload for `hash` regardless of connectivity.
    fn get(&self, hash: BlockHash) -> Option<&BlockData> {
        self.nodes.get(&hash).map(|node| match node {
            Node::Connected(node) => &node.data,
            Node::Pending(node) => &node.data,
        })
    }

    /// Admit a block from a header. Classifies it as `Connected` (parent is the anchor or an
    /// existing connected node) or `Pending`, and — when it lands connected — promotes any pending
    /// subtree it now connects to the anchor (see `promote_reachable`). Does NOT move
    /// `canonical_tip`: admission and tip advancement are separate concerns (as in the legacy
    /// `HeadObserved` path).
    fn with_block(self, hash: BlockHash, parent: BlockHash, bloom: Bloom) -> Result<BlocksGraph, NewBlockError> {
        if hash == parent {
            return Err(NewBlockError::SelfParent(self));
        }
        if hash == self.anchor {
            return Err(NewBlockError::AnchorReadmit(self));
        }
        if let Some(existing) = self.nodes.get(&hash) {
            let existing_parent = match existing {
                Node::Connected(node) => connected_parent_hash(&node.parent, self.anchor),
                Node::Pending(node) => node.parent,
            };
            return if existing_parent == parent {
                Err(NewBlockError::DuplicateBlock(self))
            } else {
                Err(NewBlockError::ConflictingParent(self))
            };
        }

        // Connectivity is decided here, at insert time, rather than recomputed by a parent-walk.
        let connects =
            parent == self.anchor || matches!(self.nodes.get(&parent), Some(Node::Connected(_)));

        let BlocksGraph {
            anchor,
            mut nodes,
            canonical_tip,
        } = self;
        let data = BlockData {
            logs_bloom: Some(bloom),
            logs: BlockLogs::Unknown,
        };

        if connects {
            let parent_ref = if parent == anchor {
                AnchoredRef::Anchor
            } else {
                AnchoredRef::Block(ConnectedHash(parent))
            };
            nodes.insert(hash, Node::Connected(ConnectedNode { parent: parent_ref, data }));
            // The new block may be the awaited parent of a pending subtree — connect it now (T6).
            Ok(BlocksGraph { anchor, nodes, canonical_tip }.promote_reachable(ConnectedHash(hash)))
        } else {
            nodes.insert(hash, Node::Pending(PendingNode { parent, data }));
            Ok(BlocksGraph { anchor, nodes, canonical_tip })
        }
    }

    /// Promotes every pending node now reachable through `newly_connected` — and transitively its
    /// newly-connected descendants — into `Connected`, re-typing each raw parent into an
    /// `AnchoredRef`. Exhaustive: on return no pending node descends from a connected one (T6).
    /// `ConnectedHash` is minted only here and on the `with_block` insert path (I1).
    fn promote_reachable(self, newly_connected: ConnectedHash) -> BlocksGraph {
        let BlocksGraph {
            anchor,
            mut nodes,
            canonical_tip,
        } = self;
        let mut frontier = vec![newly_connected];
        while let Some(ConnectedHash(connected_hash)) = frontier.pop() {
            let children: Vec<BlockHash> = nodes
                .iter()
                .filter_map(|(child_hash, node)| match node {
                    Node::Pending(pending) if pending.parent == connected_hash => Some(*child_hash),
                    _ => None,
                })
                .collect();
            for child_hash in children {
                let Some(Node::Pending(pending)) = nodes.remove(&child_hash) else {
                    continue;
                };
                nodes.insert(
                    child_hash,
                    Node::Connected(ConnectedNode {
                        parent: AnchoredRef::Block(ConnectedHash(connected_hash)),
                        data: pending.data,
                    }),
                );
                frontier.push(ConnectedHash(child_hash));
            }
        }
        BlocksGraph { anchor, nodes, canonical_tip }
    }

    // --- planned interface (each lands test-first) ---------------------------------------------
    //
    // derived view, no longer re-walked: the dangling parents still being backfilled
    // fn missing_parents(&self) -> impl Iterator<Item = BlockHash> + '_;
    //
    // canonical queries take NO finalized_hash param — the anchor is owned
    // fn canonical_oldest_to_newest(&self) -> Vec<ConnectedHash>;
    //
    // finalization: re-root at a connected descendant, prune non-descendants, reclassify
    // fn reanchored_to(self, new_anchor: ConnectedHash) -> BlocksGraph;
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use proptest::prelude::*;

    use super::*;

    fn graph_hash(index: usize) -> BlockHash {
        BlockHash::with_last_byte(index as u8)
    }

    fn block_data() -> BlockData {
        BlockData {
            logs_bloom: Some(Bloom::repeat_byte(0)),
            logs: BlockLogs::Unknown,
        }
    }

    // --- invariant helper (mirrors `assert_state_invariants` in kernel/mod.rs) ------------------

    /// Asserts the structural/topology invariants of the graph. Called after every admission in the
    /// property test and at the end of every unit test, so any regression in classification or
    /// promotion is caught immediately.
    fn assert_graph_invariants(graph: &BlocksGraph) {
        assert_anchor_not_in_nodes(graph); // T5
        assert_connected_parents_resolve_to_anchor(graph); // T2 / T3 / I1
        assert_canonical_tip_resolves(graph); // T4
        assert_no_pending_reaches_anchor(graph); // T6
    }

    fn assert_anchor_not_in_nodes(graph: &BlocksGraph) {
        assert!(
            !graph.nodes.contains_key(&graph.anchor),
            "anchor must not be present in nodes"
        );
    }

    fn assert_connected_parents_resolve_to_anchor(graph: &BlocksGraph) {
        for node in graph.nodes.values() {
            let Node::Connected(connected) = node else {
                continue;
            };
            let mut visited = HashSet::new();
            let mut current = &connected.parent;
            loop {
                match current {
                    AnchoredRef::Anchor => break,
                    AnchoredRef::Block(ConnectedHash(hash)) => {
                        assert!(visited.insert(*hash), "connected parent walk must not cycle");
                        match graph.nodes.get(hash) {
                            Some(Node::Connected(parent)) => current = &parent.parent,
                            Some(Node::Pending(_)) => {
                                panic!("connected node parent must be connected, not pending")
                            }
                            None => panic!("connected node parent must resolve to a present node"),
                        }
                    }
                }
            }
        }
    }

    fn assert_canonical_tip_resolves(graph: &BlocksGraph) {
        if let AnchoredRef::Block(ConnectedHash(hash)) = &graph.canonical_tip {
            assert!(
                matches!(graph.nodes.get(hash), Some(Node::Connected(_))),
                "canonical tip must resolve to a present connected node"
            );
        }
    }

    fn assert_no_pending_reaches_anchor(graph: &BlocksGraph) {
        for node in graph.nodes.values() {
            let Node::Pending(pending) = node else {
                continue;
            };
            assert!(
                !pending_reaches_anchor(graph, pending.parent),
                "pending node must not be reachable to the anchor"
            );
        }
    }

    /// Walks a raw parent chain. A pending node is *reachable* iff this walk reaches the anchor or a
    /// connected node; it must instead dead-end at an absent hash. A cycle counts as unreachable
    /// (and is independently rejected for connected nodes).
    fn pending_reaches_anchor(graph: &BlocksGraph, start_parent: BlockHash) -> bool {
        let mut visited = HashSet::new();
        let mut current = start_parent;
        loop {
            if current == graph.anchor {
                return true;
            }
            if !visited.insert(current) {
                return false;
            }
            match graph.nodes.get(&current) {
                Some(Node::Connected(_)) => return true,
                Some(Node::Pending(pending)) => current = pending.parent,
                None => return false,
            }
        }
    }

    // --- generator + property tests -------------------------------------------------------------

    #[derive(Debug, Clone)]
    struct AdmissionPlan {
        /// `parents[i]` is the parent node index of node `i`; node 0 is the anchor. Always strictly
        /// less than `i`, so the generated graph is rooted at the anchor and acyclic by construction.
        parents: Vec<usize>,
        /// Two independent admission permutations of nodes `1..node_count`.
        order_a: Vec<usize>,
        order_b: Vec<usize>,
    }

    /// Generates a rooted acyclic block set plus two arbitrary admission orders, so blocks can
    /// arrive children-before-parents (→ pending → later promoted). Adapts `generated_chain_strategy`
    /// from kernel/mod.rs.
    fn admission_plan_strategy() -> impl Strategy<Value = AdmissionPlan> {
        (2usize..32)
            .prop_flat_map(|node_count| {
                let admit_order: Vec<usize> = (1..node_count).collect();
                (
                    Just(node_count),
                    prop::collection::vec(any::<usize>(), node_count - 1),
                    Just(admit_order.clone()).prop_shuffle(),
                    Just(admit_order).prop_shuffle(),
                )
            })
            .prop_map(|(node_count, parent_choices, order_a, order_b)| {
                let parents = (0..node_count)
                    .map(|index| {
                        if index == 0 {
                            0
                        } else {
                            parent_choices[index - 1] % index
                        }
                    })
                    .collect();
                AdmissionPlan {
                    parents,
                    order_a,
                    order_b,
                }
            })
    }

    fn admit_in_order(plan: &AdmissionPlan, order: &[usize]) -> BlocksGraph {
        let mut graph = BlocksGraph::new(graph_hash(0));
        for &node in order {
            graph = graph
                .with_block(graph_hash(node), graph_hash(plan.parents[node]), Bloom::repeat_byte(0))
                .expect("generated admission must succeed");
        }
        graph
    }

    /// `(hash → (is_connected, parent_hash))` — a content view independent of arrival order, for
    /// comparing graphs built from the same block set in different orders.
    fn graph_fingerprint(graph: &BlocksGraph) -> BTreeMap<BlockHash, (bool, BlockHash)> {
        graph
            .nodes
            .iter()
            .map(|(hash, node)| {
                let entry = match node {
                    Node::Connected(connected) => {
                        (true, connected_parent_hash(&connected.parent, graph.anchor))
                    }
                    Node::Pending(pending) => (false, pending.parent),
                };
                (*hash, entry)
            })
            .collect()
    }

    proptest! {
        /// The central property: classification + exhaustive promotion uphold every structural and
        /// topology invariant after each admission, across all insertion orders. Because the whole
        /// set is anchor-rooted, every block is connected once fully admitted.
        #[test]
        fn with_block_preserves_graph_invariants_under_any_order(plan in admission_plan_strategy()) {
            let mut graph = BlocksGraph::new(graph_hash(0));
            for &node in &plan.order_a {
                graph = graph
                    .with_block(graph_hash(node), graph_hash(plan.parents[node]), Bloom::repeat_byte(0))
                    .expect("generated admission must succeed");
                assert_graph_invariants(&graph);
            }
            for &node in &plan.order_a {
                prop_assert!(graph.connected(graph_hash(node)).is_some());
            }
        }

        /// Promotion makes the final graph a pure function of the block set, not arrival order.
        #[test]
        fn with_block_is_order_independent(plan in admission_plan_strategy()) {
            let fingerprint_a = graph_fingerprint(&admit_in_order(&plan, &plan.order_a));
            let fingerprint_b = graph_fingerprint(&admit_in_order(&plan, &plan.order_b));
            prop_assert!(!fingerprint_a.is_empty());
            prop_assert_eq!(fingerprint_a, fingerprint_b);
        }
    }

    // --- negative invariant tests (corrupt state, confirm the checker fires) --------------------

    #[test]
    #[should_panic(expected = "anchor must not be present in nodes")]
    fn invariants_reject_anchor_in_nodes() {
        let anchor = graph_hash(1);
        let mut nodes = HashMap::new();
        nodes.insert(
            anchor,
            Node::Connected(ConnectedNode {
                parent: AnchoredRef::Anchor,
                data: block_data(),
            }),
        );
        let graph = BlocksGraph {
            anchor,
            nodes,
            canonical_tip: AnchoredRef::Anchor,
        };
        assert_graph_invariants(&graph);
    }

    #[test]
    #[should_panic(expected = "connected node parent must resolve to a present node")]
    fn invariants_reject_dangling_connected_parent() {
        let anchor = graph_hash(1);
        let mut nodes = HashMap::new();
        nodes.insert(
            graph_hash(2),
            Node::Connected(ConnectedNode {
                parent: AnchoredRef::Block(ConnectedHash(graph_hash(3))),
                data: block_data(),
            }),
        );
        let graph = BlocksGraph {
            anchor,
            nodes,
            canonical_tip: AnchoredRef::Anchor,
        };
        assert_graph_invariants(&graph);
    }

    #[test]
    #[should_panic(expected = "connected node parent must be connected, not pending")]
    fn invariants_reject_connected_parent_pointing_at_pending() {
        let anchor = graph_hash(1);
        let mut nodes = HashMap::new();
        nodes.insert(
            graph_hash(2),
            Node::Pending(PendingNode {
                parent: graph_hash(9),
                data: block_data(),
            }),
        );
        nodes.insert(
            graph_hash(3),
            Node::Connected(ConnectedNode {
                parent: AnchoredRef::Block(ConnectedHash(graph_hash(2))),
                data: block_data(),
            }),
        );
        let graph = BlocksGraph {
            anchor,
            nodes,
            canonical_tip: AnchoredRef::Anchor,
        };
        assert_graph_invariants(&graph);
    }

    #[test]
    #[should_panic(expected = "connected parent walk must not cycle")]
    fn invariants_reject_connected_cycle() {
        let anchor = graph_hash(1);
        let first = graph_hash(2);
        let second = graph_hash(3);
        let mut nodes = HashMap::new();
        nodes.insert(
            first,
            Node::Connected(ConnectedNode {
                parent: AnchoredRef::Block(ConnectedHash(second)),
                data: block_data(),
            }),
        );
        nodes.insert(
            second,
            Node::Connected(ConnectedNode {
                parent: AnchoredRef::Block(ConnectedHash(first)),
                data: block_data(),
            }),
        );
        let graph = BlocksGraph {
            anchor,
            nodes,
            canonical_tip: AnchoredRef::Anchor,
        };
        assert_graph_invariants(&graph);
    }

    #[test]
    #[should_panic(expected = "pending node must not be reachable to the anchor")]
    fn invariants_reject_pending_reachable_to_anchor() {
        let anchor = graph_hash(1);
        let mut nodes = HashMap::new();
        nodes.insert(
            graph_hash(2),
            Node::Pending(PendingNode {
                parent: anchor,
                data: block_data(),
            }),
        );
        let graph = BlocksGraph {
            anchor,
            nodes,
            canonical_tip: AnchoredRef::Anchor,
        };
        assert_graph_invariants(&graph);
    }

    // --- classification -------------------------------------------------------------------------

    #[test]
    fn parent_anchor_admits_connected() {
        // A child of the anchor connects directly: parent ref is `Anchor`, and the header's bloom
        // and an empty (Unknown) log set are stored as the block's payload.
        let anchor = graph_hash(1);
        let block = graph_hash(2);
        let bloom = Bloom::repeat_byte(0x11);
        let graph = BlocksGraph::new(anchor)
            .with_block(block, anchor, bloom)
            .expect("admitting a child of the anchor must succeed");
        assert!(matches!(
            graph.nodes.get(&block),
            Some(Node::Connected(ConnectedNode {
                parent: AnchoredRef::Anchor,
                ..
            }))
        ));
        let data = graph.get(block).expect("admitted block must carry payload");
        assert_eq!(data.logs_bloom, Some(bloom));
        assert!(matches!(data.logs, BlockLogs::Unknown));
        assert_graph_invariants(&graph);
    }

    #[test]
    fn parent_connected_admits_connected() {
        // A child of an existing connected node connects: parent ref names that node.
        let anchor = graph_hash(1);
        let parent = graph_hash(2);
        let child = graph_hash(3);
        let graph = BlocksGraph::new(anchor)
            .with_block(parent, anchor, Bloom::repeat_byte(0))
            .unwrap()
            .with_block(child, parent, Bloom::repeat_byte(0))
            .unwrap();
        assert!(matches!(
            graph.nodes.get(&child),
            Some(Node::Connected(ConnectedNode {
                parent: AnchoredRef::Block(ConnectedHash(hash)),
                ..
            })) if *hash == parent
        ));
        assert_graph_invariants(&graph);
    }

    #[test]
    fn parent_absent_admits_pending() {
        // An unknown parent leaves the block pending, holding the raw parent hash.
        let anchor = graph_hash(1);
        let absent_parent = graph_hash(2);
        let block = graph_hash(3);
        let graph = BlocksGraph::new(anchor)
            .with_block(block, absent_parent, Bloom::repeat_byte(0))
            .unwrap();
        assert!(matches!(
            graph.nodes.get(&block),
            Some(Node::Pending(PendingNode { parent, .. })) if *parent == absent_parent
        ));
        assert_graph_invariants(&graph);
    }

    #[test]
    fn parent_pending_admits_pending() {
        // A child of a pending node is itself pending (its ancestry still does not reach the anchor).
        let anchor = graph_hash(1);
        let absent = graph_hash(2);
        let pending = graph_hash(3);
        let child = graph_hash(4);
        let graph = BlocksGraph::new(anchor)
            .with_block(pending, absent, Bloom::repeat_byte(0))
            .unwrap()
            .with_block(child, pending, Bloom::repeat_byte(0))
            .unwrap();
        assert!(matches!(
            graph.nodes.get(&child),
            Some(Node::Pending(PendingNode { parent, .. })) if *parent == pending
        ));
        assert_graph_invariants(&graph);
    }

    // --- promotion ------------------------------------------------------------------------------

    #[test]
    fn connecting_block_promotes_pending_child() {
        // Admitting the awaited parent connects the previously-pending child in the same transition.
        let anchor = graph_hash(1);
        let parent = graph_hash(2);
        let child = graph_hash(3);
        let graph = BlocksGraph::new(anchor)
            .with_block(child, parent, Bloom::repeat_byte(0))
            .unwrap()
            .with_block(parent, anchor, Bloom::repeat_byte(0))
            .unwrap();
        assert!(matches!(
            graph.nodes.get(&child),
            Some(Node::Connected(ConnectedNode {
                parent: AnchoredRef::Block(ConnectedHash(hash)),
                ..
            })) if *hash == parent
        ));
        assert_graph_invariants(&graph);
    }

    #[test]
    fn connecting_block_promotes_pending_subtree() {
        // A whole pending chain promotes transitively when its root connects to the anchor (T6).
        let anchor = graph_hash(1);
        let b2 = graph_hash(2);
        let b3 = graph_hash(3);
        let b4 = graph_hash(4);
        let graph = BlocksGraph::new(anchor)
            .with_block(b4, b3, Bloom::repeat_byte(0))
            .unwrap()
            .with_block(b3, b2, Bloom::repeat_byte(0))
            .unwrap()
            .with_block(b2, anchor, Bloom::repeat_byte(0))
            .unwrap();
        assert!(graph.connected(b2).is_some());
        assert!(graph.connected(b3).is_some());
        assert!(graph.connected(b4).is_some());
        assert_graph_invariants(&graph);
    }

    #[test]
    fn pending_admission_promotes_nothing() {
        // Admitting a still-disconnected block connects neither it nor its pending children.
        let anchor = graph_hash(1);
        let b2 = graph_hash(2);
        let b3 = graph_hash(3);
        let b4 = graph_hash(4);
        let graph = BlocksGraph::new(anchor)
            .with_block(b4, b3, Bloom::repeat_byte(0))
            .unwrap()
            .with_block(b3, b2, Bloom::repeat_byte(0))
            .unwrap();
        assert!(matches!(graph.nodes.get(&b3), Some(Node::Pending(_))));
        assert!(matches!(graph.nodes.get(&b4), Some(Node::Pending(_))));
        assert_graph_invariants(&graph);
    }

    // --- errors ---------------------------------------------------------------------------------

    #[test]
    fn self_parent_is_rejected() {
        let anchor = graph_hash(1);
        let block = graph_hash(2);
        let error = BlocksGraph::new(anchor)
            .with_block(block, block, Bloom::repeat_byte(0))
            .unwrap_err();
        assert!(matches!(error, NewBlockError::SelfParent(_)));
    }

    #[test]
    fn anchor_readmit_is_rejected() {
        let anchor = graph_hash(1);
        let error = BlocksGraph::new(anchor)
            .with_block(anchor, graph_hash(2), Bloom::repeat_byte(0))
            .unwrap_err();
        assert!(matches!(error, NewBlockError::AnchorReadmit(_)));
    }

    #[test]
    fn exact_duplicate_is_noop() {
        let anchor = graph_hash(1);
        let block = graph_hash(2);
        let error = BlocksGraph::new(anchor)
            .with_block(block, anchor, Bloom::repeat_byte(0))
            .unwrap()
            .with_block(block, anchor, Bloom::repeat_byte(0))
            .unwrap_err();
        assert!(matches!(error, NewBlockError::DuplicateBlock(_)));
    }

    #[test]
    fn conflicting_parent_is_rejected() {
        let anchor = graph_hash(1);
        let block = graph_hash(3);
        let error = BlocksGraph::new(anchor)
            .with_block(block, anchor, Bloom::repeat_byte(0))
            .unwrap()
            .with_block(block, graph_hash(2), Bloom::repeat_byte(0))
            .unwrap_err();
        assert!(matches!(error, NewBlockError::ConflictingParent(_)));
    }

    #[test]
    fn blocks_graph_starts_empty() {
        let graph = BlocksGraph::new(BlockHash::ZERO);

        assert!(graph.is_empty());
    }
}
