use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use alloy::primitives::{Address, BlockHash, Bloom, BloomInput};

use crate::{PoolLog, PoolLogEvent, PoolRef, PoolState, ProtocolPoolKey, derive_pool_state};

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
///  - a connected node's parent is an `AnchoredRef` — the "missing/unknown parent" case the old
///    `while current != finalized { … else break }` walk handled does not exist for connected nodes.
///  - per-block logs are a `BTreeMap` keyed by intra-block index: deduped and ordered for free; the
///    index lives only in the key.
///  - non-anchor blocks carry no pool snapshot (absolute state is reconstructed by fold-on-demand;
///    the finalized snapshot lives beside the graph in the kernel `State`, its sole home).
///
/// Runtime (upheld by the methods, tested): referential integrity of every `ConnectedHash` /
/// `AnchoredRef`, hence parent-walk termination and acyclicity; that a `Pending` node is genuinely
/// not yet anchor-reachable; bloom consistency; the monotone `Unknown → Streamed → Complete` log
/// ladder; the finalization re-root/prune and its foldability gate; and the `pending` bound.
#[derive(Debug)]
pub(crate) struct BlocksGraph {
    anchor: BlockHash,
    nodes: HashMap<BlockHash, Node>,
    /// The latest head the feed reported — genuine external input (consensus picks the head; the DAG
    /// alone cannot, multiple connected leaves can exist), so it is stored, not derived. It may name a
    /// `Pending` (or, transiently, absent) block: the canonical chain is derived on demand and is empty
    /// until the head connects, then auto-completes (see `canonical_oldest_to_newest`). Mirrors the
    /// legacy `State.canonical_tip`, set unconditionally to the observed head.
    observed_head: BlockHash,
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
    /// Header block number. The canonical chain is number-contiguous, so the backfill query
    /// ([`BlocksGraph::missing_complete_ranges_to`], the finalization ranged-getLogs verification)
    /// coalesces unresolved blocks into numeric `eth_getLogs` ranges from this.
    number: u64,
    /// Header `logsBloom` when the block entered from a header; `None` for header-less nodes.
    logs_bloom: Option<Bloom>,
    logs: BlockLogs,
}

/// Per-block decoded pool logs, keyed by intra-block index. The key is the ordering/dedup index,
/// populated from `PoolLog::log_index` at the ingestion boundary (L3); reads use the key.
/// Authority is monotone `Unknown → Streamed → Complete` (invariant L5), advanced by
/// [`BlocksGraph::with_streamed_logs`] / [`BlocksGraph::with_complete_logs`].
#[derive(Debug)]
enum BlockLogs {
    Unknown,
    Streamed(BTreeMap<u64, PoolLog>),
    Complete(BTreeMap<u64, PoolLog>),
}

/// The outcome of an admission. Returned *alongside* the graph (which admission hands back
/// unchanged on any refusal — reorg-safety decision (b), refuse-and-keep), so no variant needs to
/// carry state. Mirrors the legacy `with_new_block` rejections, minus `CycleDetected`: a freshly
/// inserted hash is a leaf nothing points to, so admission cannot create a cycle (self-parent — the
/// one degenerate case — is refused up front).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Admission {
    /// The block entered the graph (connected or pending).
    Admitted,
    /// Refused: `hash == parent`.
    SelfParent,
    /// Refused: `hash == anchor` — a re-announce of the finalized block (legacy `ExistingBlock`).
    AnchorReadmit,
    /// Refused: `hash` is already present with the same parent — an idempotent no-op (legacy
    /// `ExistingBlock`).
    DuplicateBlock,
    /// Refused: `hash` is already present with a *different* parent — provably bad data; the
    /// first-seen block is kept.
    ConflictingParent,
    /// Refused: the block would land `Pending` but the pending staging area is already at its cap
    /// ([`MAX_PENDING_BLOCKS`]); the block is dropped (refuse-when-full, mirroring the legacy
    /// `MAX_STREAMED_LOG_BLOCKS` buffer). The authoritative path / a later header re-observation,
    /// and ultimately finalization, recover it.
    PendingBufferFull,
}

/// An inclusive range of block numbers to backfill authoritatively (ranged `eth_getLogs`). Emitted
/// by [`BlocksGraph::missing_complete_ranges_to`] for the runs of canonical blocks whose logs are
/// not yet `Complete` but whose bloom may touch a tracked pool. `hashes` snapshots exactly the hole
/// blocks the range coalesced from (ascending, matching `from..=to` minus excluded numbers): the
/// scheduler carries them in the request so the response handler can complete-empty a covered block
/// absent from the (topics-only) response — absence proves emptiness only for blocks the request
/// demonstrably covered, never for whatever the provider's fork happened to be.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MissingRange {
    pub(crate) from: u64,
    pub(crate) to: u64,
    pub(crate) hashes: Vec<BlockHash>,
}

/// Which per-block log authority the fold ([`BlocksGraph::folded_pool_states`]) is allowed to read.
/// The single axis distinguishing the two folds: finalization needs authoritative logs only;
/// the optimization read (a later increment) also accepts best-effort streamed logs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Authority {
    /// Only `Complete` blocks contribute logs — the authoritative path used by finalization.
    RequireComplete,
    /// `Streamed` (best-effort WS) blocks contribute too — the freshest-state path used by the
    /// optimization read.
    AllowStreamed,
}

/// Caps the pending staging area (invariant B1), bounding it when blocks arrive whose parent never
/// connects to the anchor (e.g. an orphaned fork below finalization). Refuse-when-full at admission;
/// the principled prune of stale forks happens later at finalization. Mirrors the legacy
/// `MAX_STREAMED_LOG_BLOCKS` subscription buffer.
const MAX_PENDING_BLOCKS: usize = 1024;

/// Whether a block's `logs_bloom` may carry a log from any `watched` address. The block bloom is a
/// consensus field with no false negatives, so a clear result *proves* the block is untouched. A
/// header-less block (no bloom) cannot be cleared, so it conservatively "may touch". With nothing
/// watched, no block is ever needed. Mirrors the legacy `block_may_touch_trusted_pool`.
fn bloom_may_touch(bloom: Option<Bloom>, watched: &HashSet<Address>) -> bool {
    if watched.is_empty() {
        return false;
    }
    match bloom {
        Some(bloom) => watched
            .iter()
            .any(|address| bloom.contains_input(BloomInput::Raw(address.as_slice()))),
        None => true,
    }
}

/// Keys incoming logs by their intra-block `log_index` into the `BTreeMap` a block stores (L1/L3).
/// This ingestion boundary is the one place `PoolLog::log_index` is read: the external index becomes
/// the map key here, and all downstream logic orders/dedups by the key alone. Positional keying
/// would be wrong — streamed fragments would each restart at 0 and collide on union.
fn key_by_log_index(logs: Vec<PoolLog>) -> BTreeMap<u64, PoolLog> {
    logs.into_iter().map(|log| (log.log_index, log)).collect()
}

/// The per-block logs a fold of the given `authority` may read: `Complete` always (authoritative),
/// `Streamed` only when streamed logs are allowed (the optimization view). `Unknown` never has logs.
/// The single point where the finalization/optimization authority axis is decided.
fn readable_logs(logs: &BlockLogs, authority: Authority) -> Option<&BTreeMap<u64, PoolLog>> {
    match (logs, authority) {
        (BlockLogs::Complete(logs), _) => Some(logs),
        (BlockLogs::Streamed(logs), Authority::AllowStreamed) => Some(logs),
        _ => None,
    }
}

/// Whether `start` is a connected descendant of `ancestor`: its connected parent chain reaches
/// `ancestor` before the anchor. A pending or absent link, or the anchor, ends the walk as "not a
/// descendant". Used by finalization to decide which nodes survive the re-root.
fn connected_descends_from(
    nodes: &HashMap<BlockHash, Node>,
    start: BlockHash,
    ancestor: BlockHash,
) -> bool {
    let mut current = start;
    let mut visited = HashSet::new();
    loop {
        if current == ancestor {
            return true;
        }
        if !visited.insert(current) {
            return false;
        }
        match nodes.get(&current) {
            Some(Node::Connected(connected)) => match &connected.parent {
                AnchoredRef::Anchor => return false,
                AnchoredRef::Block(ConnectedHash(parent)) => current = *parent,
            },
            _ => return false,
        }
    }
}

/// The bloom addresses to watch for a set of tracked pools: each v3 pool's own contract address, and
/// the chain's v4 PoolManager when any v4 pool is tracked (v4 pools share the singleton emitter).
/// Registry-free — identity comes from the pool key itself. Spans the whole `verified` tracked set,
/// not just `base`: a pool whose absolute-state logs seed it into the fold ([`folded_overlay`]) must
/// also be watched, so a bloom-touching block on its path is treated as a hole until it is foldable.
/// `v4_manager: None` means the chain deploys no PoolManager — a tracked v4 pool then contributes
/// nothing (unreachable in practice: v4 pools are only ever discovered through the manager).
fn watched_addresses(verified: &HashSet<PoolRef>, v4_manager: Option<Address>) -> HashSet<Address> {
    let mut watched = HashSet::new();
    for pool_ref in verified {
        match pool_ref.key {
            ProtocolPoolKey::UniswapV3(address) => {
                watched.insert(address);
            }
            ProtocolPoolKey::UniswapV4(_) => {
                watched.extend(v4_manager);
            }
        }
    }
    watched
}

/// The raw hash a connected node's parent reference points at (the anchor for `Anchor`).
fn connected_parent_hash(parent: &AnchoredRef, anchor: BlockHash) -> BlockHash {
    match parent {
        AnchoredRef::Anchor => anchor,
        AnchoredRef::Block(ConnectedHash(hash)) => *hash,
    }
}

impl BlocksGraph {
    pub(crate) fn new(anchor: BlockHash) -> BlocksGraph {
        BlocksGraph {
            anchor,
            nodes: HashMap::new(),
            observed_head: anchor,
        }
    }

    /// Builds a graph pre-populated from the bootstrap seed window (the seed-activation warmup fix):
    /// each seed block enters through the same connect-or-pending classification as live admission,
    /// but header-less (`logs_bloom: None`) and with its ranged-getLogs payload stored as
    /// authoritative `Complete`. The range query is topics-only (no address filter), so a present
    /// block's payload is its full pool-vocabulary log set and a gap-filler's empty set is *proven*
    /// empty — both independent of the verified-pool set, which may grow after activation (`Complete`
    /// blocks never consult the bloom, so the missing header is semantically inert).
    ///
    /// `observed_head` stays at the anchor, mirroring legacy activation (`canonical_tip` starts at
    /// the finalized hash; live replay advances it). Degenerate entries (self-parent, anchor
    /// re-admit, duplicate/conflicting hash, cap overflow) are skipped refuse-and-keep, as in
    /// [`BlocksGraph::admitted`].
    pub(crate) fn from_seed(
        anchor: BlockHash,
        blocks: Vec<(BlockHash, BlockHash, u64, Vec<PoolLog>)>,
    ) -> BlocksGraph {
        blocks.into_iter().fold(
            BlocksGraph::new(anchor),
            |graph, (hash, parent, number, logs)| {
                let data = BlockData {
                    number,
                    logs_bloom: None,
                    logs: BlockLogs::Complete(key_by_log_index(logs)),
                };
                graph.with_data_capped(hash, parent, data, MAX_PENDING_BLOCKS).0
            },
        )
    }

    /// The connected node for `hash`, if present and connected. O(1), replacing the legacy
    /// parent-walk used to (re)decide connectivity.
    fn connected(&self, hash: BlockHash) -> Option<&ConnectedNode> {
        match self.nodes.get(&hash) {
            Some(Node::Connected(node)) => Some(node),
            _ => None,
        }
    }

    /// How many blocks are currently `Pending` (the bounded staging area, invariant B1). O(n) over
    /// the node set; bounded, so a cached count is a possible later optimization, not needed now.
    fn pending_count(&self) -> usize {
        self.nodes
            .values()
            .filter(|node| matches!(node, Node::Pending(_)))
            .count()
    }

    /// Admit a block from a header (production entry point), bounding the pending staging area at
    /// [`MAX_PENDING_BLOCKS`]. See [`BlocksGraph::with_block_capped`] for the semantics.
    fn with_block(
        self,
        hash: BlockHash,
        parent: BlockHash,
        number: u64,
        bloom: Bloom,
    ) -> (BlocksGraph, Admission) {
        self.with_block_capped(hash, parent, number, bloom, MAX_PENDING_BLOCKS)
    }

    /// Kernel admission entry: [`with_block`], discarding the outcome. Every refusal keeps the
    /// (unchanged) graph — reorg-safety decision (b) in `KERNEL_BLOCKS_GRAPH_REORG_SAFETY.md`: a
    /// `ConflictingParent` (provably bad data) is refused and the first-seen block kept, not reset;
    /// every other refusal (self-parent, anchor-readmit, duplicate, pending-buffer-full) is already
    /// a benign no-op. The kernel is pure, so a refusal is silently dropped (no telemetry channel);
    /// finalization ultimately prunes any poisoned fork.
    pub(crate) fn admitted(
        self,
        hash: BlockHash,
        parent: BlockHash,
        number: u64,
        bloom: Bloom,
    ) -> BlocksGraph {
        self.with_block(hash, parent, number, bloom).0
    }

    /// Admit a block from a header. Classifies it as `Connected` (parent is the anchor or an
    /// existing connected node) or `Pending`, and — when it lands connected — promotes any pending
    /// subtree it now connects to the anchor (see `promote_reachable`). Does NOT set `observed_head`:
    /// admission and tip advancement are separate concerns (the kernel composes this with
    /// `with_observed_head`, mirroring the legacy `HeadObserved` path).
    ///
    /// `max_pending` bounds the pending staging area (B1): a block that would land `Pending` is
    /// refused with [`Admission::PendingBufferFull`] once `pending_count() >= max_pending`. The
    /// cap gates only the pending branch — a connecting admission (and the promotion it triggers,
    /// which only shrinks the pending set) is never refused. `with_block` calls this with
    /// [`MAX_PENDING_BLOCKS`]; tests drive it at a small cap.
    fn with_block_capped(
        self,
        hash: BlockHash,
        parent: BlockHash,
        number: u64,
        bloom: Bloom,
        max_pending: usize,
    ) -> (BlocksGraph, Admission) {
        let data = BlockData {
            number,
            logs_bloom: Some(bloom),
            logs: BlockLogs::Unknown,
        };
        self.with_data_capped(hash, parent, data, max_pending)
    }

    /// The admission core shared by live headers ([`with_block_capped`], bloom + `Unknown` logs) and
    /// the bootstrap seed ([`from_seed`], header-less + `Complete` logs): classification, the B1 cap,
    /// and pending promotion are identical for both; only the per-block payload differs.
    fn with_data_capped(
        self,
        hash: BlockHash,
        parent: BlockHash,
        data: BlockData,
        max_pending: usize,
    ) -> (BlocksGraph, Admission) {
        if hash == parent {
            return (self, Admission::SelfParent);
        }
        if hash == self.anchor {
            return (self, Admission::AnchorReadmit);
        }
        if let Some(existing) = self.nodes.get(&hash) {
            let existing_parent = match existing {
                Node::Connected(node) => connected_parent_hash(&node.parent, self.anchor),
                Node::Pending(node) => node.parent,
            };
            let admission = if existing_parent == parent {
                Admission::DuplicateBlock
            } else {
                Admission::ConflictingParent
            };
            return (self, admission);
        }

        // Connectivity is decided here, at insert time, rather than recomputed by a parent-walk.
        let connects =
            parent == self.anchor || matches!(self.nodes.get(&parent), Some(Node::Connected(_)));

        // B1: refuse a block that would land pending once the staging area is at the cap. The cap
        // gates only this branch — a connecting admission (and its promotion, which only shrinks
        // pending) is never refused. The graph is handed back unchanged.
        if !connects && self.pending_count() >= max_pending {
            return (self, Admission::PendingBufferFull);
        }

        let BlocksGraph {
            anchor,
            mut nodes,
            observed_head,
        } = self;

        let graph = if connects {
            let parent_ref = if parent == anchor {
                AnchoredRef::Anchor
            } else {
                AnchoredRef::Block(ConnectedHash(parent))
            };
            nodes.insert(hash, Node::Connected(ConnectedNode { parent: parent_ref, data }));
            // The new block may be the awaited parent of a pending subtree — connect it now (T6).
            BlocksGraph { anchor, nodes, observed_head }.promote_reachable(ConnectedHash(hash))
        } else {
            nodes.insert(hash, Node::Pending(PendingNode { parent, data }));
            BlocksGraph { anchor, nodes, observed_head }
        };
        (graph, Admission::Admitted)
    }

    /// Promotes every pending node now reachable through `newly_connected` — and transitively its
    /// newly-connected descendants — into `Connected`, re-typing each raw parent into an
    /// `AnchoredRef`. Exhaustive: on return no pending node descends from a connected one (T6).
    /// `ConnectedHash` is minted only here and on the `with_block` insert path (I1).
    fn promote_reachable(self, newly_connected: ConnectedHash) -> BlocksGraph {
        let BlocksGraph {
            anchor,
            mut nodes,
            observed_head,
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
        BlocksGraph { anchor, nodes, observed_head }
    }

    /// Records the latest observed head (the kernel's `HeadObserved` analog, composed after
    /// `with_block` admits the block). Set unconditionally, like the legacy `State.canonical_tip`:
    /// the head may be `Pending` — the canonical chain stays empty until it connects, then
    /// auto-completes (see `canonical_oldest_to_newest`). Updates the field only when `hash` is the
    /// anchor or a present node, so the "`observed_head` is the anchor or present" invariant cannot be
    /// violated here; an absent hash (which the admit-first kernel never produces) is a no-op.
    pub(crate) fn with_observed_head(self, hash: BlockHash) -> BlocksGraph {
        if hash == self.anchor || self.nodes.contains_key(&hash) {
            BlocksGraph { observed_head: hash, ..self }
        } else {
            self
        }
    }

    /// A present block's mutable payload, regardless of its connectivity — the log-merge transitions
    /// write here so promotion never has to move logs. `None` when the hash is absent (pruned/never
    /// admitted); the anchor has no node and so is never returned.
    fn block_data_mut(&mut self, hash: BlockHash) -> Option<&mut BlockData> {
        self.nodes.get_mut(&hash).map(|node| match node {
            Node::Connected(connected) => &mut connected.data,
            Node::Pending(pending) => &mut pending.data,
        })
    }

    /// Merges best-effort streamed (WS) logs into a present block, upholding L5. `Unknown → Streamed`;
    /// a further `Streamed` merge grows the set by union on the intra-block index. Never steps back
    /// off an authoritative `Complete` block (the merge is a no-op there), and a no-op when the block
    /// is absent. Keys the logs by `log_index` (L3) via [`key_by_log_index`].
    pub(crate) fn with_streamed_logs(mut self, hash: BlockHash, logs: Vec<PoolLog>) -> BlocksGraph {
        if let Some(data) = self.block_data_mut(hash) {
            match &mut data.logs {
                // Authoritative logs stand — L5 forbids Complete → Streamed.
                BlockLogs::Complete(_) => {}
                BlockLogs::Streamed(existing) => existing.extend(key_by_log_index(logs)),
                BlockLogs::Unknown => data.logs = BlockLogs::Streamed(key_by_log_index(logs)),
            }
        }
        self
    }

    /// Records authoritative (getLogs) logs for a present block, upholding L5: sets `Complete`, which
    /// supersedes any prior `Unknown`/`Streamed` (replace semantics — `Complete` is the authoritative
    /// full set). Idempotent on an already-`Complete` block (same hash ⇒ same logs) and a no-op when
    /// the block is absent. Keys the logs by `log_index` (L3) via [`key_by_log_index`].
    pub(crate) fn with_complete_logs(mut self, hash: BlockHash, logs: Vec<PoolLog>) -> BlocksGraph {
        if let Some(data) = self.block_data_mut(hash) {
            data.logs = BlockLogs::Complete(key_by_log_index(logs));
        }
        self
    }

    /// Whether an incoming authoritative log set *diverges* from the streamed set a present block
    /// holds: true iff the block is `Streamed` and the `log_index` key-sets differ (a missed or
    /// spurious stream delivery). The WS-miss metric — the kernel reads this immediately before the
    /// [`with_complete_logs`] replacement to count how often the trusted-at-tip stream was wrong
    /// (`Complete ⊇ Streamed` is an expectation *measured* here, not assumed; replace semantics stays
    /// the authority). `Unknown` (nothing streamed — the hole/backstop path) and `Complete`
    /// (idempotent re-receive) blocks never count, nor does an absent block. A pure query: the graph
    /// stays write-monotone.
    pub(crate) fn streamed_log_mismatch(&self, hash: BlockHash, logs: &[PoolLog]) -> bool {
        let Some(node) = self.nodes.get(&hash) else {
            return false;
        };
        let data = match node {
            Node::Connected(connected) => &connected.data,
            Node::Pending(pending) => &pending.data,
        };
        match &data.logs {
            BlockLogs::Streamed(existing) => {
                let incoming: BTreeSet<u64> = logs.iter().map(|log| log.log_index).collect();
                existing.keys().copied().collect::<BTreeSet<u64>>() != incoming
            }
            BlockLogs::Unknown | BlockLogs::Complete(_) => false,
        }
    }

    /// The canonical chain from oldest (child of the anchor) to newest (`observed_head`), derived on
    /// demand — never stored. When `observed_head` is `Connected`, its parent chain reaches the anchor
    /// by T2, so the walk is total and gap-free; when it is the anchor, `Pending`, or absent, there is
    /// no foldable suffix yet and the chain is empty. Takes no `finalized_hash` — the anchor is owned.
    fn canonical_oldest_to_newest(&self) -> Vec<ConnectedHash> {
        // The shared walk stops immediately on a non-connected start, so a pending/absent/anchor
        // head yields an empty chain (the `ConnectedHash` here is a walk start, not an I1 proof).
        self.connected_oldest_to_newest(ConnectedHash(self.observed_head))
    }

    /// The dangling ancestors still to backfill: the distinct, sorted set of `Pending` nodes' parent
    /// hashes that are absent from the graph (and not the anchor). A *present* pending parent is
    /// awaited transitively, not fetched — only the gap root is reported. Connected nodes never
    /// contribute (their parents resolve by T2). No `finalized_hash` parameter — the anchor is owned.
    /// Sorted+deduped so downstream backfill request-id assignment is deterministic (mirrors the
    /// legacy `missing_seed_parents`). The production header-backfill source: the kernel requests
    /// each of these (minus in-flight) via `GetBlockHeader`.
    pub(crate) fn missing_parents(&self) -> Vec<BlockHash> {
        let mut missing: Vec<BlockHash> = self
            .nodes
            .values()
            .filter_map(|node| match node {
                Node::Pending(pending)
                    if pending.parent != self.anchor
                        && !self.nodes.contains_key(&pending.parent) =>
                {
                    Some(pending.parent)
                }
                _ => None,
            })
            .collect();
        missing.sort_unstable();
        missing.dedup();
        missing
    }

    /// The raw parent hash of any present node — proven (`AnchoredRef`) for connected nodes, unproven
    /// for pending ones. The present-chain walks below follow it without caring about connectivity,
    /// mirroring the legacy graph's flat parent-pointer walk.
    fn node_parent_hash(&self, node: &Node) -> BlockHash {
        match node {
            Node::Connected(connected) => connected_parent_hash(&connected.parent, self.anchor),
            Node::Pending(pending) => pending.parent,
        }
    }

    /// The *present* chain from `observed_head` toward the anchor, newest→oldest: follows raw parent
    /// pointers through connected AND pending nodes and stops at the anchor or the first absent
    /// block. This deliberately reaches past the connected region — during a header backfill the
    /// suffix above the gap is still `Pending`, and the schedulers must keep fetching its logs and
    /// validating its candidates in parallel with the headers, exactly as the legacy tip walk did.
    /// Newest-first so request emission order (and thus request-id assignment) matches legacy.
    /// A cycle (representable only among pending nodes, inert otherwise) yields an empty walk,
    /// mirroring the legacy visited-guard.
    fn present_chain_from_head(&self) -> Vec<(BlockHash, &BlockData)> {
        let mut chain = Vec::new();
        let mut current = self.observed_head;
        let mut visited = HashSet::new();
        while current != self.anchor {
            if !visited.insert(current) {
                return Vec::new();
            }
            let Some(node) = self.nodes.get(&current) else {
                break;
            };
            let data = match node {
                Node::Connected(connected) => &connected.data,
                Node::Pending(pending) => &pending.data,
            };
            chain.push((current, data));
            current = self.node_parent_hash(node);
        }
        chain
    }

    /// The tip-side *holes* still needing a per-block `GetBlockLogs` backstop, newest→oldest:
    /// present-chain blocks whose logs are `Unknown` (the WS stream delivered nothing — a `Streamed`
    /// block is trusted at the tip and never re-fetched; authoritative verification is
    /// finalization's ranged fetch), whose bloom may touch a `trusted` address, at depth ≥
    /// `min_depth` below the observed head (giving the stream time to deliver before falling back
    /// to RPC), minus `exclude` (the in-flight request set — the graph is pure and holds no request
    /// state).
    ///
    /// The bloom gate mirrors the legacy scheduler, NOT the fold's `watched_addresses`: `trusted` is
    /// the scheduler-owned set (each verified pool's address plus the v4 PoolManager unconditionally —
    /// the discovery anchor). An empty `trusted` set yields no holes ([`bloom_may_touch`]) — the
    /// WS-primary trust flip retired the gate-inactive fetch-everything warmup; discovery rides the
    /// topic-filtered stream and the bootstrap range scan. The gate re-evaluates against the
    /// *current* trusted set, so a block turns fetchable when a later-verified pool's address hits
    /// its bloom. `present_chain_from_head` walks newest→oldest, so the enumeration index *is* the
    /// depth below the head (0 = the head itself).
    pub(crate) fn unknown_log_hole_hashes(
        &self,
        trusted: &HashSet<Address>,
        exclude: &HashSet<BlockHash>,
        min_depth: usize,
    ) -> Vec<BlockHash> {
        self.present_chain_from_head()
            .into_iter()
            .enumerate()
            .filter(|(depth, (hash, data))| {
                *depth >= min_depth
                    && matches!(data.logs, BlockLogs::Unknown)
                    && !exclude.contains(hash)
                    && bloom_may_touch(data.logs_bloom, trusted)
            })
            .map(|(_, (hash, _))| hash)
            .collect()
    }

    /// Candidate pools named by stored logs on the present chain, newest→oldest, one non-empty set
    /// per block. Candidates are derived from the logs themselves (`log.pool` — never stored, the
    /// L6 discipline), from `Streamed` as well as `Complete` payloads so subscription-discovered
    /// pools validate before the authoritative fetch lands (legacy `Partial` behaved the same).
    /// Registry filtering (known/pending) stays kernel-side; the graph knows nothing of registries.
    pub(crate) fn pool_log_candidates_from_head(&self) -> Vec<(BlockHash, HashSet<ProtocolPoolKey>)> {
        self.present_chain_from_head()
            .into_iter()
            .filter_map(|(hash, data)| {
                let logs = match &data.logs {
                    BlockLogs::Unknown => return None,
                    BlockLogs::Streamed(logs) | BlockLogs::Complete(logs) => logs,
                };
                let candidates: HashSet<ProtocolPoolKey> =
                    logs.values().map(|log| log.pool).collect();
                (!candidates.is_empty()).then_some((hash, candidates))
            })
            .collect()
    }

    /// How many present-chain blocks separate `observed_head` from `reference` (0 when the head *is*
    /// the reference; the reference block itself is not counted). `None` when `reference` is not on
    /// the present chain — absent, on a side fork, or beyond a gap. Follows raw parent pointers like
    /// [`present_chain_from_head`], mirroring the legacy `connected_path_hashes_oldest_to_newest`
    /// walk that fed `blocks_behind` and `canonical_path_len_from_finalized` (pass the anchor as
    /// `reference` for the latter).
    pub(crate) fn distance_from_head(&self, reference: BlockHash) -> Option<usize> {
        let mut current = self.observed_head;
        let mut distance = 0;
        let mut visited = HashSet::new();
        while current != reference {
            if !visited.insert(current) {
                return None;
            }
            let node = self.nodes.get(&current)?;
            distance += 1;
            current = self.node_parent_hash(node);
        }
        Some(distance)
    }

    /// Every admitted block hash, connected or pending — the graph's key-space. Finalization prunes
    /// block-targeted pending requests against this after the re-root, mirroring the legacy
    /// `blocks.hashes()` (the anchor is not a node and block-scoped requests never target it).
    pub(crate) fn node_hashes(&self) -> HashSet<BlockHash> {
        self.nodes.keys().copied().collect()
    }

    /// The finalized anchor hash — its sole home (invariant A1). Production reads it for the
    /// finalized-distance metric ([`distance_from_head`](Self::distance_from_head) to the anchor)
    /// and wherever the kernel needs the finalized boundary.
    pub(crate) fn anchor_hash(&self) -> BlockHash {
        self.anchor
    }

    /// Whether `hash` is an admitted block (connected or pending; the anchor is not a node). The
    /// kernel uses it to route a streamed log to the block versus the pre-head staging buffer, and
    /// to drain that buffer only once the block actually entered the graph.
    pub(crate) fn contains(&self, hash: BlockHash) -> bool {
        self.nodes.contains_key(&hash)
    }

    /// The per-block payloads on the canonical path `anchor → target`, oldest→newest. `target` is
    /// connected (the caller's [`ConnectedHash`] proof), so the walk reaches the anchor by T2.
    fn connected_path_data(&self, target: ConnectedHash) -> Vec<&BlockData> {
        // Every hash the shared walk yields is connected by construction, so the lookup resolves.
        self.connected_oldest_to_newest(target)
            .into_iter()
            .filter_map(|ConnectedHash(hash)| self.connected(hash).map(|node| &node.data))
            .collect()
    }

    /// The block-number ranges on the canonical path `anchor → target` whose logs must be fetched
    /// authoritatively (ranged `eth_getLogs`) before the path is fully foldable — the finalization
    /// verification query behind [`missing_complete_ranges_to`].
    ///
    /// Value-free (reads no pool state): a block is a *hole* iff its logs are not yet `Complete`
    /// **and** its `logs_bloom` may touch a `watched` address (a tracked pool's v3 contract address,
    /// or the v4 PoolManager) **and** its number is not in `exclude_numbers` (numbers already covered
    /// by an in-flight ranged request — the graph is pure and holds no request state; an excluded
    /// number splits a would-be range like any non-hole). A header-less node (no bloom) is
    /// conservatively a hole. Consecutive hole block numbers are coalesced into inclusive
    /// [`MissingRange`]s carrying their hole hashes (the canonical path is number-contiguous, but
    /// coalescing by numeric adjacency is correct regardless). An empty result means the path is
    /// fully foldable.
    ///
    /// `target` must be connected (caller holds the [`ConnectedHash`] proof); a target that is the
    /// anchor yields no ranges.
    fn missing_complete_ranges(
        &self,
        target: ConnectedHash,
        watched: &HashSet<Address>,
        exclude_numbers: &HashSet<u64>,
    ) -> Vec<MissingRange> {
        let mut ranges: Vec<MissingRange> = Vec::new();
        for ConnectedHash(hash) in self.connected_oldest_to_newest(target) {
            // Every hash the shared walk yields is connected by construction, so the lookup resolves.
            let Some(node) = self.connected(hash) else {
                break;
            };
            let data = &node.data;
            let is_hole = !matches!(data.logs, BlockLogs::Complete(_))
                && bloom_may_touch(data.logs_bloom, watched)
                && !exclude_numbers.contains(&data.number);
            if !is_hole {
                continue;
            }
            match ranges.last_mut() {
                Some(last) if data.number == last.to + 1 => {
                    last.to = data.number;
                    last.hashes.push(hash);
                }
                _ => ranges.push(MissingRange {
                    from: data.number,
                    to: data.number,
                    hashes: vec![hash],
                }),
            }
        }
        ranges
    }

    /// Kernel entry for the finalization-time ranged-getLogs verification (the WS-primary
    /// counterpart of [`finalized_to`], which pre-selects a hole-free frontier and therefore stalls
    /// at these very holes): the [`MissingRange`]s on the canonical path `anchor → target`, or empty
    /// when `target` is absent/pending/unconnected or off the canonical chain — mirroring
    /// [`finalized_to`]'s guards (same `verified`/`v4_manager` watched-set inputs), so a target the
    /// finalization no-ops on schedules no fetches either (and a fully-advanced target, no longer a
    /// node, yields nothing: today's quiet case).
    pub(crate) fn missing_complete_ranges_to(
        &self,
        target: BlockHash,
        verified: &HashSet<PoolRef>,
        v4_manager: Option<Address>,
        exclude_numbers: &HashSet<u64>,
    ) -> Vec<MissingRange> {
        if self.connected(target).is_none() {
            return Vec::new();
        }
        if !self
            .canonical_oldest_to_newest()
            .iter()
            .any(|ConnectedHash(hash)| *hash == target)
        {
            return Vec::new();
        }
        let watched = watched_addresses(verified, v4_manager);
        self.missing_complete_ranges(ConnectedHash(target), &watched, exclude_numbers)
    }

    /// The per-pool **overlay** of the canonical path `anchor → target` folded over `base`: only pools
    /// whose state actually changes on the path appear; an untouched pool is absent and the caller
    /// reads `base` for it. The single fold engine — see [`folded_pool_states`] for the full-snapshot
    /// wrapper and [`optimization_pool_states`] for the overlay consumer.
    ///
    /// Single pass: walk the path once ([`connected_path_data`], oldest→newest) and bucket each
    /// **readable** log's `&PoolLogEvent` by its [`ProtocolPoolKey`] — buckets borrow, nothing is
    /// cloned. Block order plus each block's `BTreeMap` `log_index` order give every bucket the exact
    /// fold order [`derive_pool_state`] requires. Then fold each touched pool once from its `base`
    /// state, keyed by the log's own [`ProtocolPoolKey`] (never the registry): a pool absent from
    /// `base` has no bucket match and is skipped (absent, never stale).
    /// `authority` selects which blocks contribute logs (see [`Authority`]). A pool whose run is
    /// underivable (a liquidity overflow — impossible for protocol-bounded `Complete` data) produces no
    /// overlay entry, so the caller keeps its base state. Cost is `O(B + L)` over the path plus cheap
    /// per-`base`-pool bucket probes, cloning only the pools that changed.
    ///
    /// **Seeding.** A `verified` pool absent from `base` (discovered after bootstrap) is seeded from
    /// its own path logs via `derive_pool_state(None, run)`: it yields a state iff the run begins with
    /// an absolute event (Swap/Initialize). A Mint/Burn-only run stays underivable here and instead
    /// gains its base from the anchor-height seed path (`schedule_finalized_pool_seed_requests` →
    /// `finalized_snapshot`, Blockers 1b/1c), after which it folds through the `base` branch. Only
    /// `verified` keys are ever seeded, so topic-spoofing non-pools never enter and every seeded pool
    /// already has registry metadata.
    fn folded_overlay(
        &self,
        base: &HashMap<PoolRef, PoolState>,
        verified: &HashSet<PoolRef>,
        target: ConnectedHash,
        authority: Authority,
    ) -> HashMap<PoolRef, PoolState> {
        let mut buckets: HashMap<ProtocolPoolKey, Vec<&PoolLogEvent>> = HashMap::new();
        for data in self.connected_path_data(target) {
            if let Some(logs) = readable_logs(&data.logs, authority) {
                for log in logs.values() {
                    buckets.entry(log.pool).or_default().push(&log.event);
                }
            }
        }

        let mut overlay = HashMap::new();
        for (pool_ref, base_state) in base {
            if let Some(run) = buckets.get(&pool_ref.key) {
                if let Some(folded) = derive_pool_state(Some(base_state), run) {
                    overlay.insert(*pool_ref, folded);
                }
            }
        }
        for pool_ref in verified {
            if base.contains_key(pool_ref) {
                continue;
            }
            if let Some(run) = buckets.get(&pool_ref.key) {
                if let Some(folded) = derive_pool_state(None, run) {
                    overlay.insert(*pool_ref, folded);
                }
            }
        }
        overlay
    }

    /// The full absolute snapshot of every tracked pool at `target`: `base` with the path's changes
    /// ([`folded_overlay`]) applied. Folds **only** pools present in `base` (others are seeded via
    /// the deferred anchor-height seeding); a pool whose run is underivable keeps its base state.
    /// Clones `base` once — the
    /// finalization path ([`reanchored_to`]) needs a fresh owned snapshot to store; the per-event
    /// optimization read uses the bare overlay instead. Borrows `base`: shared read.
    fn folded_pool_states(
        &self,
        base: &HashMap<PoolRef, PoolState>,
        verified: &HashSet<PoolRef>,
        target: ConnectedHash,
        authority: Authority,
    ) -> HashMap<PoolRef, PoolState> {
        let mut snapshot = base.clone();
        snapshot.extend(self.folded_overlay(base, verified, target, authority));
        snapshot
    }

    /// The latest connected block on the canonical path `anchor → target` whose whole prefix is
    /// foldable under `authority`: walk oldest→newest, advancing the frontier across each block, and
    /// **stop before the first "blocker"** — a block with no readable logs for this authority
    /// ([`readable_logs`] is `None`) **and** a `logs_bloom` that may touch a watched pool
    /// ([`bloom_may_touch`]). A block that is bloom-clear (proven untouched) is skipped, not a blocker,
    /// so the frontier advances past it. Returns the last good block, or `ConnectedHash(anchor)` when
    /// the first canonical block already blocks or the chain is empty (a non-connected `target` yields
    /// an empty walk, [`connected_oldest_to_newest`]).
    ///
    /// The single "stop at the first unfoldable block" predicate shared by both folds, differing only
    /// on which logs count as readable ([`Authority`]): finalization ([`Authority::RequireComplete`])
    /// treats a `Streamed` block as a blocker, the optimization read ([`Authority::AllowStreamed`])
    /// folds through it and blocks only on an `Unknown` bloom-touching one.
    fn foldable_frontier(
        &self,
        target: ConnectedHash,
        watched: &HashSet<Address>,
        authority: Authority,
    ) -> ConnectedHash {
        let mut frontier = ConnectedHash(self.anchor);
        for ConnectedHash(hash) in self.connected_oldest_to_newest(target) {
            let Some(node) = self.connected(hash) else {
                break;
            };
            let blocks = readable_logs(&node.data.logs, authority).is_none()
                && bloom_may_touch(node.data.logs_bloom, watched);
            if blocks {
                break;
            }
            frontier = ConnectedHash(hash);
        }
        frontier
    }

    /// The optimization read: the per-pool **overlay** of freshest state, folding the canonical path
    /// `anchor → observed_head` over `base` while reading **best-effort `Streamed`** logs as well as
    /// authoritative `Complete` ones ([`Authority::AllowStreamed`]) — so the optimizer runs on the most
    /// recent state, not just the last fully-verified one.
    ///
    /// **Stops at the first `Unknown` block whose bloom may touch a watched pool** ([`foldable_frontier`]):
    /// its logs could move a tracked pool, so folding *past* it onto pre-gap state would be wrong. The
    /// read never fails — it returns the overlay folded up to (and including) the last good block before
    /// that gap, together with **that frontier block's hash** so the caller knows the height the reserves
    /// are valid at (the caller merges the overlay onto `base`; an untouched pool is absent). When
    /// `observed_head` is the anchor, `Pending`, or absent, the overlay is empty and the hash is the
    /// anchor. An `Unknown` block with a *clear* bloom is proven untouched and does not stop the fold.
    ///
    /// The finalization counterpart is [`reanchored_to`] ([`Authority::RequireComplete`], to the anchor,
    /// full snapshot); this one never mutates and clones nothing beyond the changed pools.
    ///
    /// The production optimization read, called by `State::optimization_update`.
    /// `v4_manager` is the chain's v4 PoolManager address (`None` when the chain deploys none), used for v4 bloom-touch checks.
    pub(crate) fn optimization_pool_states(
        &self,
        base: &HashMap<PoolRef, PoolState>,
        verified: &HashSet<PoolRef>,
        v4_manager: Option<Address>,
    ) -> (HashMap<PoolRef, PoolState>, BlockHash) {
        let watched = watched_addresses(verified, v4_manager);
        let frontier =
            self.foldable_frontier(ConnectedHash(self.observed_head), &watched, Authority::AllowStreamed);
        let frontier_hash = frontier.0;
        (
            self.folded_overlay(base, verified, frontier, Authority::AllowStreamed),
            frontier_hash,
        )
    }

    /// Advances the anchor to `new_anchor`, folding the now-final logs into `base` and pruning every
    /// block that no longer descends from the new anchor (A3), returning the re-rooted graph and the
    /// advanced finalized snapshot.
    ///
    /// Precondition (A4): the canonical path `anchor → new_anchor` is fully `Complete`/bloom-clear
    /// for the tracked pools. The sole production caller, [`finalized_to`], guarantees this by
    /// selecting `new_anchor` via [`foldable_frontier`] ([`Authority::RequireComplete`]) — the gate
    /// lives there, not here.
    ///
    /// `base` is borrowed; a fresh snapshot is returned.
    fn reanchored_to(
        self,
        new_anchor: ConnectedHash,
        base: &HashMap<PoolRef, PoolState>,
        verified: &HashSet<PoolRef>,
    ) -> (BlocksGraph, HashMap<PoolRef, PoolState>) {
        let new_anchor_hash = new_anchor.0;
        // Defensive: a ConnectedHash never names the anchor (I1), but the query is pure over any input.
        if new_anchor_hash == self.anchor {
            return (self, base.clone());
        }

        // Fold the now-final prefix (borrowed) before consuming self for the re-root.
        let new_snapshot = self.folded_pool_states(
            base,
            verified,
            ConnectedHash(new_anchor_hash),
            Authority::RequireComplete,
        );

        let BlocksGraph {
            anchor: _,
            nodes,
            observed_head,
        } = self;

        // Retain the connected descendants of the new anchor; drop the new anchor itself (it becomes
        // the root), all other forks, and every pending node (none descend — T6).
        let retained: HashSet<BlockHash> = nodes
            .iter()
            .filter_map(|(hash, node)| {
                (matches!(node, Node::Connected(_))
                    && *hash != new_anchor_hash
                    && connected_descends_from(&nodes, *hash, new_anchor_hash))
                .then_some(*hash)
            })
            .collect();

        let nodes: HashMap<BlockHash, Node> = nodes
            .into_iter()
            .filter(|(hash, _)| retained.contains(hash))
            .map(|(hash, node)| {
                // Reclassify a direct child of the new anchor: its parent reference now resolves to
                // the root, so it becomes `Anchor`. Deeper nodes keep their (also-retained) parent.
                let node = match node {
                    Node::Connected(mut connected) => {
                        if matches!(
                            &connected.parent,
                            AnchoredRef::Block(ConnectedHash(parent)) if *parent == new_anchor_hash
                        ) {
                            connected.parent = AnchoredRef::Anchor;
                        }
                        Node::Connected(connected)
                    }
                    pending => pending,
                };
                (hash, node)
            })
            .collect();

        // Keep the observed head if it survived (or is the new anchor); otherwise reset to the anchor.
        let observed_head = if observed_head == new_anchor_hash || nodes.contains_key(&observed_head) {
            observed_head
        } else {
            new_anchor_hash
        };

        (
            BlocksGraph {
                anchor: new_anchor_hash,
                nodes,
                observed_head,
            },
            new_snapshot,
        )
    }

    /// Kernel finalization entry: advance the anchor toward `target` (the observed finalized block) as
    /// far as the canonical prefix is fully foldable, folding into `base`. Mirrors legacy
    /// partial-compaction (reorg-safety decision (c) in `KERNEL_BLOCKS_GRAPH_REORG_SAFETY.md`): it
    /// reanchors to the *latest complete connected block ≤ target*, so a hole short of `target`
    /// advances only to just before that hole (fold-on-demand), and an absent/pending/unconnected/
    /// off-canonical `target` is a no-op. Owns the A4 gate: the chosen sub-target is hole-free by
    /// construction ([`foldable_frontier`], [`Authority::RequireComplete`]), satisfying
    /// [`reanchored_to`]'s precondition. The returned snapshot replaces the caller's finalized base
    /// only when the anchor actually advances.
    pub(crate) fn finalized_to(
        self,
        target: BlockHash,
        base: &HashMap<PoolRef, PoolState>,
        verified: &HashSet<PoolRef>,
        v4_manager: Option<Address>,
    ) -> (BlocksGraph, HashMap<PoolRef, PoolState>) {
        // Only a connected target is finalizable; an absent/pending one has no foldable prefix yet.
        if self.connected(target).is_none() {
            return (self, base.clone());
        }
        // Decision (d) in `KERNEL_BLOCKS_GRAPH_REORG_SAFETY.md`: only a target on the canonical
        // chain (anchor → observed_head) may finalize — a connected side-fork target would prune
        // the head branch; no-op and wait for the head to catch up instead.
        if !self
            .canonical_oldest_to_newest()
            .iter()
            .any(|ConnectedHash(hash)| *hash == target)
        {
            return (self, base.clone());
        }
        let watched = watched_addresses(verified, v4_manager);

        // The latest complete connected block whose prefix `anchor → it` is entirely foldable —
        // legacy's "latest complete ≤ target". `RequireComplete` makes a non-`Complete` bloom-touching
        // block a blocker (the finalization "hole"), so the shared frontier walk stops just before it.
        // A frontier equal to the anchor means nothing is foldable yet.
        let frontier = self.foldable_frontier(ConnectedHash(target), &watched, Authority::RequireComplete);

        // A frontier at the anchor is `reanchored_to`'s base-clone no-op — nothing foldable, stay put.
        self.reanchored_to(frontier, base, verified)
    }

    /// The single connected-chain walk: the canonical chain `anchor → target`, oldest→newest, as
    /// connected hashes. [`canonical_oldest_to_newest`] (from the head) and [`connected_path_data`]
    /// (payload projection) both delegate here. When `target` is connected (the caller's
    /// [`ConnectedHash`] proof), the walk reaches the anchor gap-free and acyclically (T2); a
    /// non-connected start yields the empty chain.
    fn connected_oldest_to_newest(&self, target: ConnectedHash) -> Vec<ConnectedHash> {
        let mut chain = Vec::new();
        let mut current = target.0;
        while let Some(node) = self.connected(current) {
            chain.push(ConnectedHash(current));
            match node.parent {
                AnchoredRef::Anchor => break,
                AnchoredRef::Block(ConnectedHash(parent)) => current = parent,
            }
        }
        chain.reverse();
        chain
    }
}

/// Plain-`BlockHash` test accessors for the kernel's behavior tests. `ConnectedHash` is
/// module-private (the connectivity proof must not cross the module boundary), so these project it
/// away for assertions.
#[cfg(test)]
impl BlocksGraph {
    pub(crate) fn observed_head_hash(&self) -> BlockHash {
        self.observed_head
    }

    /// True when no recent blocks are tracked yet — only the anchor exists.
    fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// The per-block payload for `hash` regardless of connectivity.
    fn get(&self, hash: BlockHash) -> Option<&BlockData> {
        self.nodes.get(&hash).map(|node| match node {
            Node::Connected(node) => &node.data,
            Node::Pending(node) => &node.data,
        })
    }

    /// The canonical chain hashes, oldest→newest (empty when the head is not connected).
    pub(crate) fn canonical_hashes(&self) -> Vec<BlockHash> {
        self.canonical_oldest_to_newest()
            .into_iter()
            .map(|ConnectedHash(hash)| hash)
            .collect()
    }

    /// The raw parent hash a present node references (proven for connected, raw for pending).
    pub(crate) fn parent_hash_for_test(&self, hash: BlockHash) -> Option<BlockHash> {
        self.nodes
            .get(&hash)
            .map(|node| self.node_parent_hash(node))
    }

    /// `Some(true)` when the block's logs are authoritative (`Complete`), `Some(false)` for
    /// `Unknown`/`Streamed`, `None` when the block is absent.
    pub(crate) fn has_complete_logs_for_test(&self, hash: BlockHash) -> Option<bool> {
        self.get(hash)
            .map(|data| matches!(data.logs, BlockLogs::Complete(_)))
    }

    /// The candidate pools named by a present block's stored logs (empty while `Unknown`).
    pub(crate) fn log_candidates_for_test(&self, hash: BlockHash) -> Option<HashSet<ProtocolPoolKey>> {
        self.get(hash).map(|data| match &data.logs {
            BlockLogs::Unknown => HashSet::new(),
            BlockLogs::Streamed(logs) | BlockLogs::Complete(logs) => {
                logs.values().map(|log| log.pool).collect()
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeSet, HashSet};

    use alloy::primitives::{U160, aliases::I24};
    use proptest::prelude::*;

    use super::*;
    use crate::ChainKey;

    fn graph_hash(index: usize) -> BlockHash {
        BlockHash::with_last_byte(index as u8)
    }

    fn block_data() -> BlockData {
        BlockData {
            number: 0,
            logs_bloom: Some(Bloom::repeat_byte(0)),
            logs: BlockLogs::Unknown,
        }
    }

    /// Unwraps a derived canonical chain to its raw hashes, for order-sensitive comparison.
    fn chain_hashes(chain: &[ConnectedHash]) -> Vec<BlockHash> {
        chain.iter().map(|ConnectedHash(hash)| *hash).collect()
    }

    // --- invariant helper (mirrors `assert_state_invariants` in kernel/mod.rs) ------------------

    /// Asserts the structural/topology invariants of the graph. Called after every admission in the
    /// property test and at the end of every unit test, so any regression in classification or
    /// promotion is caught immediately.
    fn assert_graph_invariants(graph: &BlocksGraph) {
        assert_anchor_not_in_nodes(graph); // T5
        assert_connected_parents_resolve_to_anchor(graph); // T2 / T3 / I1
        assert_observed_head_present(graph); // T4
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

    fn assert_observed_head_present(graph: &BlocksGraph) {
        assert!(
            graph.observed_head == graph.anchor || graph.nodes.contains_key(&graph.observed_head),
            "observed head must be the anchor or a present node"
        );
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
                .with_block(graph_hash(node), graph_hash(plan.parents[node]), node as u64, Bloom::repeat_byte(0)).0;
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
                    .with_block(graph_hash(node), graph_hash(plan.parents[node]), node as u64, Bloom::repeat_byte(0)).0;
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

        /// B1: the pending staging area never exceeds the cap, in any insertion order, while every
        /// topology invariant continues to hold. Refused admissions hand the (unchanged) graph back,
        /// so the fold continues from it. (Does NOT assert full connectivity or order-independence:
        /// once the cap bites, a refused pending block can leave its descendants permanently
        /// unconnected and the outcome depends on arrival order.)
        #[test]
        fn with_block_capped_never_exceeds_bound(plan in admission_plan_strategy()) {
            const MAX: usize = 3;
            let mut graph = BlocksGraph::new(graph_hash(0));
            for &node in &plan.order_a {
                let (next, admission) = graph.with_block_capped(
                    graph_hash(node),
                    graph_hash(plan.parents[node]),
                    node as u64,
                    Bloom::repeat_byte(0),
                    MAX,
                );
                prop_assert!(
                    matches!(admission, Admission::Admitted | Admission::PendingBufferFull),
                    "unexpected admission outcome: {admission:?}"
                );
                graph = next;
                prop_assert!(graph.pending_count() <= MAX);
                assert_graph_invariants(&graph);
            }
        }

        /// The canonical chain derived from a connected head is exactly the parent path from that
        /// head down to the anchor (oldest→newest), with every element resolving to a connected node.
        /// Compared against an independent recompute over the generated parent links, across shapes.
        #[test]
        fn canonical_chain_is_connected_path_to_anchor(
            plan in admission_plan_strategy(),
            head_seed in any::<usize>(),
        ) {
            let node_count = plan.parents.len();
            // A non-anchor node index (the set is anchor-rooted, so every such node is connected).
            let head = 1 + head_seed % (node_count - 1);
            let graph = admit_in_order(&plan, &plan.order_a).with_observed_head(graph_hash(head));

            // Independently recompute the parent path head → anchor (node 0), oldest→newest.
            let mut expected = Vec::new();
            let mut current = head;
            while current != 0 {
                expected.push(graph_hash(current));
                current = plan.parents[current];
            }
            expected.reverse();

            let chain = graph.canonical_oldest_to_newest();
            prop_assert_eq!(chain_hashes(&chain), expected);
            for ConnectedHash(hash) in &chain {
                prop_assert!(graph.connected(*hash).is_some());
            }
        }

        /// `missing_parents()` is exactly the absent, non-anchor hashes referenced by pending nodes —
        /// sound, complete, and strictly ascending — at every step of an incremental fold (where
        /// genuine gaps exist before promotion closes them).
        #[test]
        fn missing_parents_are_exactly_absent_referenced_parents(plan in admission_plan_strategy()) {
            let mut graph = BlocksGraph::new(graph_hash(0));
            for &node in &plan.order_a {
                graph = graph
                    .with_block(graph_hash(node), graph_hash(plan.parents[node]), node as u64, Bloom::repeat_byte(0)).0;
                let missing = graph.missing_parents();

                // distinct + strictly ascending
                for pair in missing.windows(2) {
                    prop_assert!(pair[0] < pair[1]);
                }
                // soundness: every reported hash is absent, not the anchor, and referenced by a pending node
                for &hash in &missing {
                    prop_assert!(hash != graph.anchor);
                    prop_assert!(!graph.nodes.contains_key(&hash));
                    prop_assert!(graph
                        .nodes
                        .values()
                        .any(|node| matches!(node, Node::Pending(p) if p.parent == hash)));
                }
                // completeness: every pending node's absent, non-anchor parent is reported
                for node in graph.nodes.values() {
                    if let Node::Pending(pending) = node {
                        if pending.parent != graph.anchor && !graph.nodes.contains_key(&pending.parent) {
                            prop_assert!(missing.contains(&pending.parent));
                        }
                    }
                }
            }
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
            observed_head: anchor,
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
            observed_head: anchor,
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
            observed_head: anchor,
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
            observed_head: anchor,
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
            observed_head: anchor,
        };
        assert_graph_invariants(&graph);
    }

    #[test]
    #[should_panic(expected = "observed head must be the anchor or a present node")]
    fn invariants_reject_absent_observed_head() {
        let anchor = graph_hash(1);
        let graph = BlocksGraph {
            anchor,
            nodes: HashMap::new(),
            observed_head: graph_hash(2),
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
            .with_block(block, anchor, 0, bloom).0;
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
            .with_block(parent, anchor, 0, Bloom::repeat_byte(0)).0
            .with_block(child, parent, 0, Bloom::repeat_byte(0)).0;
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
            .with_block(block, absent_parent, 0, Bloom::repeat_byte(0)).0;
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
            .with_block(pending, absent, 0, Bloom::repeat_byte(0)).0
            .with_block(child, pending, 0, Bloom::repeat_byte(0)).0;
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
            .with_block(child, parent, 0, Bloom::repeat_byte(0)).0
            .with_block(parent, anchor, 0, Bloom::repeat_byte(0)).0;
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
            .with_block(b4, b3, 0, Bloom::repeat_byte(0)).0
            .with_block(b3, b2, 0, Bloom::repeat_byte(0)).0
            .with_block(b2, anchor, 0, Bloom::repeat_byte(0)).0;
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
            .with_block(b4, b3, 0, Bloom::repeat_byte(0)).0
            .with_block(b3, b2, 0, Bloom::repeat_byte(0)).0;
        assert!(matches!(graph.nodes.get(&b3), Some(Node::Pending(_))));
        assert!(matches!(graph.nodes.get(&b4), Some(Node::Pending(_))));
        assert_graph_invariants(&graph);
    }

    // --- errors ---------------------------------------------------------------------------------

    #[test]
    fn self_parent_is_rejected() {
        let anchor = graph_hash(1);
        let block = graph_hash(2);
        let (_, admission) = BlocksGraph::new(anchor).with_block(block, block, 0, Bloom::repeat_byte(0));
        assert_eq!(admission, Admission::SelfParent);
    }

    #[test]
    fn anchor_readmit_is_rejected() {
        let anchor = graph_hash(1);
        let (_, admission) =
            BlocksGraph::new(anchor).with_block(anchor, graph_hash(2), 0, Bloom::repeat_byte(0));
        assert_eq!(admission, Admission::AnchorReadmit);
    }

    #[test]
    fn exact_duplicate_is_noop() {
        let anchor = graph_hash(1);
        let block = graph_hash(2);
        let (_, admission) = BlocksGraph::new(anchor)
            .with_block(block, anchor, 0, Bloom::repeat_byte(0)).0
            .with_block(block, anchor, 0, Bloom::repeat_byte(0));
        assert_eq!(admission, Admission::DuplicateBlock);
    }

    #[test]
    fn conflicting_parent_is_rejected() {
        let anchor = graph_hash(1);
        let block = graph_hash(3);
        let (_, admission) = BlocksGraph::new(anchor)
            .with_block(block, anchor, 0, Bloom::repeat_byte(0)).0
            .with_block(block, graph_hash(2), 0, Bloom::repeat_byte(0));
        assert_eq!(admission, Admission::ConflictingParent);
    }

    // --- pending bound (B1) ---------------------------------------------------------------------

    /// Admits `max` disconnected blocks (distinct absent parents) so the staging area is exactly
    /// full, then a further disconnected block. The last one is refused and the graph handed back
    /// unchanged.
    fn graph_with_full_pending(anchor: BlockHash, max: usize) -> BlocksGraph {
        let mut graph = BlocksGraph::new(anchor);
        for index in 0..max {
            // parents are absent (and distinct from any block hash) so every block lands pending.
            let block = graph_hash(10 + index);
            let absent_parent = graph_hash(100 + index);
            graph = graph
                .with_block_capped(block, absent_parent, 0, Bloom::repeat_byte(0), max).0;
        }
        graph
    }

    #[test]
    fn pending_bound_refuses_new_when_full() {
        // At the cap, a further pending block is refused with PendingBufferFull and the graph is
        // returned unchanged (same pending count, the refused hash absent).
        let anchor = graph_hash(1);
        let max = 3;
        let graph = graph_with_full_pending(anchor, max);
        assert_eq!(graph.pending_count(), max);

        let refused = graph_hash(50);
        let (graph, admission) =
            graph.with_block_capped(refused, graph_hash(200), 0, Bloom::repeat_byte(0), max);
        assert_eq!(admission, Admission::PendingBufferFull);
        assert_eq!(graph.pending_count(), max);
        assert!(graph.get(refused).is_none());
        assert_graph_invariants(&graph);
    }

    #[test]
    fn connected_admission_succeeds_at_pending_cap() {
        // The cap gates only the pending branch: a child of the anchor connects even when the
        // pending staging area is full.
        let anchor = graph_hash(1);
        let max = 3;
        let graph = graph_with_full_pending(anchor, max);
        assert_eq!(graph.pending_count(), max);

        let child = graph_hash(2);
        let graph = graph
            .with_block_capped(child, anchor, 0, Bloom::repeat_byte(0), max).0;
        assert!(graph.connected(child).is_some());
        assert_eq!(graph.pending_count(), max);
        assert_graph_invariants(&graph);
    }

    #[test]
    fn promotion_not_blocked_at_pending_cap() {
        // A pending child counts toward a full buffer; admitting its awaited (anchor-rooted) parent
        // connects and promotes the child — never refused, and the pending count drops.
        let anchor = graph_hash(1);
        let max = 3;
        let parent = graph_hash(2);
        let child = graph_hash(3);
        // Fill to the cap with the child plus (max - 1) unrelated disconnected blocks.
        let mut graph = BlocksGraph::new(anchor)
            .with_block_capped(child, parent, 0, Bloom::repeat_byte(0), max).0;
        for index in 0..(max - 1) {
            graph = graph
                .with_block_capped(graph_hash(10 + index), graph_hash(100 + index), 0, Bloom::repeat_byte(0), max).0;
        }
        assert_eq!(graph.pending_count(), max);

        let graph = graph
            .with_block_capped(parent, anchor, 0, Bloom::repeat_byte(0), max).0;
        assert!(graph.connected(parent).is_some());
        assert!(graph.connected(child).is_some());
        assert_eq!(graph.pending_count(), max - 1);
        assert_graph_invariants(&graph);
    }

    // --- canonical chain (observed head) --------------------------------------------------------

    #[test]
    fn canonical_chain_empty_when_head_is_anchor() {
        // A fresh graph's head is the anchor: no non-anchor blocks, so the chain is empty.
        let graph = BlocksGraph::new(graph_hash(1));
        assert!(graph.canonical_oldest_to_newest().is_empty());
        assert_graph_invariants(&graph);
    }

    #[test]
    fn canonical_chain_single_connected_head() {
        // A child of the anchor, observed as head, is the whole canonical chain.
        let anchor = graph_hash(1);
        let child = graph_hash(2);
        let graph = BlocksGraph::new(anchor)
            .with_block(child, anchor, 0, Bloom::repeat_byte(0)).0
            .with_observed_head(child);
        assert_eq!(chain_hashes(&graph.canonical_oldest_to_newest()), vec![child]);
        assert_graph_invariants(&graph);
    }

    #[test]
    fn canonical_chain_full_path() {
        // The chain runs oldest (child of the anchor) to newest (the head).
        let anchor = graph_hash(1);
        let b2 = graph_hash(2);
        let b3 = graph_hash(3);
        let graph = BlocksGraph::new(anchor)
            .with_block(b2, anchor, 0, Bloom::repeat_byte(0)).0
            .with_block(b3, b2, 0, Bloom::repeat_byte(0)).0
            .with_observed_head(b3);
        assert_eq!(chain_hashes(&graph.canonical_oldest_to_newest()), vec![b2, b3]);
        assert_graph_invariants(&graph);
    }

    #[test]
    fn canonical_chain_empty_when_head_pending() {
        // A head observed before its parent lands pending; no connected suffix exists yet.
        let anchor = graph_hash(1);
        let absent_parent = graph_hash(2);
        let head = graph_hash(3);
        let graph = BlocksGraph::new(anchor)
            .with_block(head, absent_parent, 0, Bloom::repeat_byte(0)).0
            .with_observed_head(head);
        assert!(matches!(graph.nodes.get(&head), Some(Node::Pending(_))));
        assert!(graph.canonical_oldest_to_newest().is_empty());
        assert_graph_invariants(&graph);
    }

    #[test]
    fn canonical_chain_autocompletes_after_backfill() {
        // The defining property: the head is observed while pending (chain empty); when its missing
        // parent is later backfilled and promotion connects the head, the chain auto-completes with
        // NO further with_observed_head call.
        let anchor = graph_hash(1);
        let parent = graph_hash(2);
        let head = graph_hash(3);
        let graph = BlocksGraph::new(anchor)
            .with_block(head, parent, 0, Bloom::repeat_byte(0)).0
            .with_observed_head(head);
        assert!(graph.canonical_oldest_to_newest().is_empty());

        let graph = graph.with_block(parent, anchor, 0, Bloom::repeat_byte(0)).0;
        assert_eq!(chain_hashes(&graph.canonical_oldest_to_newest()), vec![parent, head]);
        assert_graph_invariants(&graph);
    }

    #[test]
    fn canonical_chain_follows_reorg() {
        // Repointing the head to a sibling connected fork is a pure, non-destructive pointer move:
        // the chain follows the head, both forks stay in the graph.
        let anchor = graph_hash(1);
        let fork_a = graph_hash(2);
        let fork_b = graph_hash(3);
        let graph = BlocksGraph::new(anchor)
            .with_block(fork_a, anchor, 0, Bloom::repeat_byte(0)).0
            .with_block(fork_b, anchor, 0, Bloom::repeat_byte(0)).0
            .with_observed_head(fork_a);
        assert_eq!(chain_hashes(&graph.canonical_oldest_to_newest()), vec![fork_a]);

        let graph = graph.with_observed_head(fork_b);
        assert_eq!(chain_hashes(&graph.canonical_oldest_to_newest()), vec![fork_b]);
        assert_graph_invariants(&graph);
    }

    #[test]
    fn with_observed_head_ignores_absent_block() {
        // Setting the head to a hash that is neither the anchor nor present is a no-op (keeps the
        // "observed head is the anchor or present" invariant un-violable at the setter).
        let anchor = graph_hash(1);
        let absent = graph_hash(2);
        let graph = BlocksGraph::new(anchor).with_observed_head(absent);
        assert_eq!(graph.observed_head, anchor);
        assert_graph_invariants(&graph);
    }

    // --- missing parents (backfill view) --------------------------------------------------------

    #[test]
    fn missing_parents_empty_graph() {
        let graph = BlocksGraph::new(graph_hash(1));
        assert!(graph.missing_parents().is_empty());
        assert_graph_invariants(&graph);
    }

    #[test]
    fn missing_parents_all_connected() {
        // An anchor-rooted chain has no gaps: every parent is present.
        let anchor = graph_hash(1);
        let b2 = graph_hash(2);
        let b3 = graph_hash(3);
        let graph = BlocksGraph::new(anchor)
            .with_block(b2, anchor, 0, Bloom::repeat_byte(0)).0
            .with_block(b3, b2, 0, Bloom::repeat_byte(0)).0;
        assert!(graph.missing_parents().is_empty());
        assert_graph_invariants(&graph);
    }

    #[test]
    fn missing_parents_single_dangling() {
        // A pending block's absent parent is the one ancestor to backfill.
        let anchor = graph_hash(1);
        let absent_parent = graph_hash(2);
        let block = graph_hash(3);
        let graph = BlocksGraph::new(anchor)
            .with_block(block, absent_parent, 0, Bloom::repeat_byte(0)).0;
        assert_eq!(graph.missing_parents(), vec![absent_parent]);
        assert_graph_invariants(&graph);
    }

    #[test]
    fn missing_parents_dedups_shared_absent_parent() {
        // Two pending blocks awaiting the same absent parent report it once.
        let anchor = graph_hash(1);
        let shared_absent = graph_hash(2);
        let b3 = graph_hash(3);
        let b4 = graph_hash(4);
        let graph = BlocksGraph::new(anchor)
            .with_block(b3, shared_absent, 0, Bloom::repeat_byte(0)).0
            .with_block(b4, shared_absent, 0, Bloom::repeat_byte(0)).0;
        assert_eq!(graph.missing_parents(), vec![shared_absent]);
        assert_graph_invariants(&graph);
    }

    #[test]
    fn missing_parents_excludes_present_pending_parent() {
        // b4 → b3 (b3 absent on arrival), then b3 → absent_root. Both pending; b3 is present, so it is
        // not missing (awaited transitively) — only the gap root, absent_root, is reported.
        let anchor = graph_hash(1);
        let absent_root = graph_hash(2);
        let b3 = graph_hash(3);
        let b4 = graph_hash(4);
        let graph = BlocksGraph::new(anchor)
            .with_block(b4, b3, 0, Bloom::repeat_byte(0)).0
            .with_block(b3, absent_root, 0, Bloom::repeat_byte(0)).0;
        assert!(matches!(graph.nodes.get(&b3), Some(Node::Pending(_))));
        assert!(matches!(graph.nodes.get(&b4), Some(Node::Pending(_))));
        assert_eq!(graph.missing_parents(), vec![absent_root]);
        assert_graph_invariants(&graph);
    }

    #[test]
    fn missing_parents_sorted() {
        // Absent parents admitted out of order are returned strictly ascending.
        let anchor = graph_hash(1);
        let graph = BlocksGraph::new(anchor)
            .with_block(graph_hash(10), graph_hash(30), 0, Bloom::repeat_byte(0)).0
            .with_block(graph_hash(11), graph_hash(20), 0, Bloom::repeat_byte(0)).0
            .with_block(graph_hash(12), graph_hash(25), 0, Bloom::repeat_byte(0)).0;
        assert_eq!(
            graph.missing_parents(),
            vec![graph_hash(20), graph_hash(25), graph_hash(30)]
        );
        assert_graph_invariants(&graph);
    }

    #[test]
    fn missing_parents_empty_after_promotion() {
        // Once the awaited parent arrives and connects, the gap closes and nothing remains to fetch.
        let anchor = graph_hash(1);
        let parent = graph_hash(2);
        let child = graph_hash(3);
        let graph = BlocksGraph::new(anchor)
            .with_block(child, parent, 0, Bloom::repeat_byte(0)).0;
        assert_eq!(graph.missing_parents(), vec![parent]);

        let graph = graph.with_block(parent, anchor, 0, Bloom::repeat_byte(0)).0;
        assert!(graph.missing_parents().is_empty());
        assert_graph_invariants(&graph);
    }

    // --- scheduling queries (present-chain walk from the head) ---------------------------------

    #[test]
    fn log_holes_walk_present_chain_newest_first() {
        let anchor = graph_hash(1);
        let graph = BlocksGraph::new(anchor)
            .with_block(graph_hash(2), anchor, 2, hit_bloom()).0
            .with_block(graph_hash(3), graph_hash(2), 3, hit_bloom()).0
            .with_block(graph_hash(4), graph_hash(3), 4, hit_bloom()).0
            .with_observed_head(graph_hash(4));
        assert_eq!(
            graph.unknown_log_hole_hashes(&watched(), &HashSet::new(), 0),
            vec![graph_hash(4), graph_hash(3), graph_hash(2)]
        );
        assert_graph_invariants(&graph);
    }

    #[test]
    fn log_holes_reach_pending_suffix_and_stop_at_gap() {
        // b4/b5 are pending above the absent b3: the walk fetches their logs in parallel with the
        // header backfill, and stops at the gap — the connected b2 below it is not reached until
        // the headers connect (mirrors the legacy tip walk, which broke at the first absent block).
        let anchor = graph_hash(1);
        let graph = BlocksGraph::new(anchor)
            .with_block(graph_hash(2), anchor, 2, hit_bloom()).0
            .with_block(graph_hash(4), graph_hash(3), 4, hit_bloom()).0
            .with_block(graph_hash(5), graph_hash(4), 5, hit_bloom()).0
            .with_observed_head(graph_hash(5));
        assert_eq!(
            graph.unknown_log_hole_hashes(&watched(), &HashSet::new(), 0),
            vec![graph_hash(5), graph_hash(4)]
        );
        assert_graph_invariants(&graph);
    }

    #[test]
    fn log_holes_skip_complete_streamed_and_in_flight() {
        // Complete is authoritative (never refetched); an in-flight request is deduped by the
        // caller-supplied exclude set; and — the WS-primary trust flip — a Streamed block is trusted
        // at the tip and NOT re-fetched (pre-flip behavior fetched it; finalization's ranged
        // verification is now its authoritative check).
        let anchor = graph_hash(1);
        let graph = BlocksGraph::new(anchor)
            .with_block(graph_hash(2), anchor, 2, hit_bloom()).0
            .with_block(graph_hash(3), graph_hash(2), 3, hit_bloom()).0
            .with_block(graph_hash(4), graph_hash(3), 4, hit_bloom()).0
            .with_complete_logs(graph_hash(2), Vec::new())
            .with_streamed_logs(graph_hash(3), Vec::new())
            .with_observed_head(graph_hash(4));
        assert_eq!(
            graph.unknown_log_hole_hashes(&watched(), &HashSet::new(), 0),
            vec![graph_hash(4)]
        );
        let exclude = HashSet::from([graph_hash(4)]);
        assert!(
            graph
                .unknown_log_hole_hashes(&watched(), &exclude, 0)
                .is_empty()
        );
        assert_graph_invariants(&graph);
    }

    #[test]
    fn log_holes_respect_the_depth_gate() {
        // min_depth leaves the newest blocks to the WS stream: depth counts down from the head
        // (head = 0), so with min_depth 2 only b2 (depth 2) qualifies.
        let anchor = graph_hash(1);
        let graph = BlocksGraph::new(anchor)
            .with_block(graph_hash(2), anchor, 2, hit_bloom()).0
            .with_block(graph_hash(3), graph_hash(2), 3, hit_bloom()).0
            .with_block(graph_hash(4), graph_hash(3), 4, hit_bloom()).0
            .with_observed_head(graph_hash(4));
        assert_eq!(
            graph.unknown_log_hole_hashes(&watched(), &HashSet::new(), 2),
            vec![graph_hash(2)]
        );
        assert_graph_invariants(&graph);
    }

    #[test]
    fn log_holes_bloom_clear_and_empty_trusted_yield_nothing() {
        // A bloom-clear block is provably untouched and skipped. An empty trusted set yields no
        // holes at all: the WS-primary flip retired the gate-inactive fetch-everything warmup
        // (discovery rides the topic-filtered stream and the bootstrap range scan).
        let anchor = graph_hash(1);
        let graph = BlocksGraph::new(anchor)
            .with_block(graph_hash(2), anchor, 2, clear_bloom()).0
            .with_block(graph_hash(3), graph_hash(2), 3, hit_bloom()).0
            .with_observed_head(graph_hash(3));
        assert_eq!(
            graph.unknown_log_hole_hashes(&watched(), &HashSet::new(), 0),
            vec![graph_hash(3)]
        );
        assert!(
            graph
                .unknown_log_hole_hashes(&HashSet::new(), &HashSet::new(), 0)
                .is_empty()
        );
        assert_graph_invariants(&graph);
    }

    proptest! {
        /// The tip-hole query never names a block that has any streamed/complete logs, sits above
        /// the depth gate, or fails the bloom gate — the WS-primary "streamed is never re-fetched"
        /// pin at the graph level.
        #[test]
        fn log_holes_are_only_unknown_bloom_hit_blocks_at_depth(
            specs in prop::collection::vec((0u8..3, any::<bool>()), 1..12),
            min_depth in 0usize..6,
        ) {
            let anchor = graph_hash(1);
            let mut graph = BlocksGraph::new(anchor);
            let mut parent = anchor;
            for (index, (logs_kind, bloom_hit)) in specs.iter().enumerate() {
                let hash = graph_hash(index + 2);
                let bloom = if *bloom_hit { hit_bloom() } else { clear_bloom() };
                graph = graph.with_block(hash, parent, (index + 2) as u64, bloom).0;
                graph = match logs_kind {
                    0 => graph,
                    1 => graph.with_streamed_logs(hash, Vec::new()),
                    _ => graph.with_complete_logs(hash, Vec::new()),
                };
                parent = hash;
            }
            let graph = graph.with_observed_head(parent);

            let holes = graph.unknown_log_hole_hashes(&watched(), &HashSet::new(), min_depth);

            // Independently expected: Unknown + bloom-hit blocks at depth ≥ min_depth, newest first.
            let expected: Vec<BlockHash> = specs
                .iter()
                .enumerate()
                .rev()
                .filter_map(|(index, (logs_kind, bloom_hit))| {
                    let depth = specs.len() - 1 - index;
                    (*logs_kind == 0 && *bloom_hit && depth >= min_depth)
                        .then(|| graph_hash(index + 2))
                })
                .collect();
            prop_assert_eq!(holes, expected);
        }
    }

    #[test]
    fn candidates_come_from_streamed_and_complete_logs_newest_first() {
        let pool_a = v3_pool(0xA1);
        let pool_b = v3_pool(0xB2);
        let anchor = graph_hash(1);
        let graph = BlocksGraph::new(anchor)
            .with_block(graph_hash(2), anchor, 2, hit_bloom()).0
            .with_block(graph_hash(3), graph_hash(2), 3, hit_bloom()).0
            .with_block(graph_hash(4), graph_hash(3), 4, hit_bloom()).0
            .with_complete_logs(graph_hash(2), vec![pool_log(pool_a, sw(0))])
            .with_streamed_logs(graph_hash(3), vec![pool_log(pool_b, sw(1))])
            .with_observed_head(graph_hash(4));
        // b4 is Unknown (no logs yet), so it contributes no candidate entry.
        assert_eq!(
            graph.pool_log_candidates_from_head(),
            vec![
                (graph_hash(3), HashSet::from([pool_b.key])),
                (graph_hash(2), HashSet::from([pool_a.key])),
            ]
        );
        assert_graph_invariants(&graph);
    }

    #[test]
    fn candidates_skip_empty_complete_block() {
        // A Complete block with no pool logs names no candidates — no empty entries downstream.
        let anchor = graph_hash(1);
        let graph = BlocksGraph::new(anchor)
            .with_block(graph_hash(2), anchor, 2, hit_bloom()).0
            .with_complete_logs(graph_hash(2), Vec::new())
            .with_observed_head(graph_hash(2));
        assert!(graph.pool_log_candidates_from_head().is_empty());
        assert_graph_invariants(&graph);
    }

    #[test]
    fn distance_from_head_counts_blocks_above_reference() {
        let anchor = graph_hash(1);
        let graph = BlocksGraph::new(anchor)
            .with_block(graph_hash(2), anchor, 2, clear_bloom()).0
            .with_block(graph_hash(3), graph_hash(2), 3, clear_bloom()).0
            .with_block(graph_hash(4), graph_hash(3), 4, clear_bloom()).0
            .with_observed_head(graph_hash(4));
        assert_eq!(graph.distance_from_head(graph_hash(4)), Some(0));
        assert_eq!(graph.distance_from_head(graph_hash(2)), Some(2));
        assert_eq!(graph.distance_from_head(anchor), Some(3));
        assert_graph_invariants(&graph);
    }

    #[test]
    fn distance_from_head_is_none_off_the_present_chain() {
        // A fork sibling is not on the head's parent chain.
        let anchor = graph_hash(1);
        let graph = BlocksGraph::new(anchor)
            .with_block(graph_hash(2), anchor, 2, clear_bloom()).0
            .with_block(graph_hash(3), anchor, 2, clear_bloom()).0
            .with_observed_head(graph_hash(2));
        assert_eq!(graph.distance_from_head(graph_hash(3)), None);

        // A pending head above an absent parent cannot reach the anchor — but the walk counts
        // through pending nodes and stops AT the reference, so the absent gap root itself is
        // reachable as a reference (mirrors the legacy flat parent-pointer walk exactly).
        let gapped = BlocksGraph::new(anchor)
            .with_block(graph_hash(5), graph_hash(4), 5, clear_bloom()).0
            .with_observed_head(graph_hash(5));
        assert_eq!(gapped.distance_from_head(anchor), None);
        assert_eq!(gapped.distance_from_head(graph_hash(5)), Some(0));
        assert_eq!(gapped.distance_from_head(graph_hash(4)), Some(1));
    }

    // --- finalization gate (missing complete ranges) -------------------------------------------

    fn watched_addr() -> Address {
        Address::with_last_byte(0xAA)
    }

    fn watched() -> HashSet<Address> {
        HashSet::from([watched_addr()])
    }

    /// A bloom that hits the watched address (so a non-`Complete` block carrying it is a hole).
    fn hit_bloom() -> Bloom {
        let mut bloom = Bloom::default();
        bloom.accrue(BloomInput::Raw(watched_addr().as_slice()));
        bloom
    }

    /// An all-zero bloom: hits no address, so the block is provably untouched by any tracked pool.
    fn clear_bloom() -> Bloom {
        Bloom::default()
    }

    /// Builds a connected linear chain rooted at the anchor (`graph_hash(1)`): block `i` (0-based) has
    /// hash `graph_hash(i + 2)`, parent block `i - 1` (block 0's parent is the anchor). Each spec is
    /// `(number, logs_bloom, is_complete)`. `observed_head` is the last block; the target returned is
    /// that last block as a `ConnectedHash`. `specs` must be non-empty.
    fn gate_chain(specs: &[(u64, Option<Bloom>, bool)]) -> (BlocksGraph, ConnectedHash) {
        let anchor = graph_hash(1);
        let mut nodes = HashMap::new();
        let mut parent = AnchoredRef::Anchor;
        let mut last = anchor;
        for (index, (number, bloom, complete)) in specs.iter().enumerate() {
            let hash = graph_hash(index + 2);
            let logs = if *complete {
                BlockLogs::Complete(BTreeMap::new())
            } else {
                BlockLogs::Unknown
            };
            nodes.insert(
                hash,
                Node::Connected(ConnectedNode {
                    parent,
                    data: BlockData {
                        number: *number,
                        logs_bloom: *bloom,
                        logs,
                    },
                }),
            );
            parent = AnchoredRef::Block(ConnectedHash(hash));
            last = hash;
        }
        let graph = BlocksGraph {
            anchor,
            nodes,
            observed_head: last,
        };
        (graph, ConnectedHash(last))
    }

    /// Shorthand for the expected [`MissingRange`] of a contiguous hole run: `gate_chain` gives
    /// block number `n` the hash `graph_hash(n)`, so the covered hashes follow from the bounds.
    fn missing_range(from: u64, to: u64) -> MissingRange {
        MissingRange {
            from,
            to,
            hashes: (from..=to).map(|number| graph_hash(number as usize)).collect(),
        }
    }

    fn no_excludes() -> HashSet<u64> {
        HashSet::new()
    }

    #[test]
    fn gate_all_complete_yields_no_ranges() {
        let (graph, target) = gate_chain(&[(2, Some(hit_bloom()), true), (3, Some(hit_bloom()), true)]);
        assert!(
            graph
                .missing_complete_ranges(target, &watched(), &no_excludes())
                .is_empty()
        );
        assert_graph_invariants(&graph);
    }

    #[test]
    fn gate_single_non_complete_bloom_hit_is_a_range() {
        let (graph, target) = gate_chain(&[
            (2, Some(hit_bloom()), true),
            (3, Some(hit_bloom()), false),
            (4, Some(hit_bloom()), true),
        ]);
        assert_eq!(
            graph.missing_complete_ranges(target, &watched(), &no_excludes()),
            vec![missing_range(3, 3)]
        );
        assert_graph_invariants(&graph);
    }

    #[test]
    fn gate_bloom_clear_non_complete_is_not_a_hole() {
        // A bloom-clear block provably touches no tracked pool, so it never needs fetching even
        // though its logs are not Complete.
        let (graph, target) =
            gate_chain(&[(2, Some(clear_bloom()), false), (3, Some(clear_bloom()), false)]);
        assert!(
            graph
                .missing_complete_ranges(target, &watched(), &no_excludes())
                .is_empty()
        );
        assert_graph_invariants(&graph);
    }

    #[test]
    fn gate_coalesces_consecutive_holes() {
        let (graph, target) = gate_chain(&[
            (2, Some(hit_bloom()), false),
            (3, Some(hit_bloom()), false),
            (4, Some(hit_bloom()), true),
        ]);
        assert_eq!(
            graph.missing_complete_ranges(target, &watched(), &no_excludes()),
            vec![missing_range(2, 3)]
        );
        assert_graph_invariants(&graph);
    }

    #[test]
    fn gate_separates_non_adjacent_holes() {
        let (graph, target) = gate_chain(&[
            (2, Some(hit_bloom()), false),
            (3, Some(hit_bloom()), true),
            (4, Some(hit_bloom()), false),
        ]);
        assert_eq!(
            graph.missing_complete_ranges(target, &watched(), &no_excludes()),
            vec![missing_range(2, 2), missing_range(4, 4)]
        );
        assert_graph_invariants(&graph);
    }

    #[test]
    fn gate_excluded_number_splits_a_range() {
        // A number already covered by an in-flight ranged request is not re-requested, and it
        // splits the coalesced run around it exactly like a non-hole.
        let (graph, target) = gate_chain(&[
            (2, Some(hit_bloom()), false),
            (3, Some(hit_bloom()), false),
            (4, Some(hit_bloom()), false),
        ]);
        assert_eq!(
            graph.missing_complete_ranges(target, &watched(), &HashSet::from([3])),
            vec![missing_range(2, 2), missing_range(4, 4)]
        );
        assert_graph_invariants(&graph);
    }

    #[test]
    fn gate_headerless_non_complete_is_a_hole() {
        // No bloom ⇒ we cannot prove the block is clear, so (with pools watched) it must be fetched.
        let (graph, target) = gate_chain(&[(2, None, false)]);
        assert_eq!(
            graph.missing_complete_ranges(target, &watched(), &no_excludes()),
            vec![missing_range(2, 2)]
        );
        assert_graph_invariants(&graph);
    }

    #[test]
    fn gate_wrapper_guards_mirror_finalization() {
        // The kernel-facing wrapper no-ops exactly where `finalized_to` does: an absent target has
        // no foldable prefix, and a connected side-fork target must not drive fetches for a path
        // finalization would refuse to fold. A canonical mid-path target works. Watched-set inputs
        // are `finalized_to`'s own (verified pools + v4 manager), derived internally.
        let verified = HashSet::from([watched_pool()]);
        let anchor = graph_hash(1);
        let graph = BlocksGraph::new(anchor)
            .with_block(graph_hash(2), anchor, 2, hit_bloom()).0
            .with_block(graph_hash(3), graph_hash(2), 3, hit_bloom()).0
            // Side fork off b2, connected but not canonical once the head sits on b3.
            .with_block(graph_hash(9), graph_hash(2), 3, hit_bloom()).0
            .with_observed_head(graph_hash(3));
        assert!(
            graph
                .missing_complete_ranges_to(graph_hash(7), &verified, None, &no_excludes())
                .is_empty()
        );
        assert!(
            graph
                .missing_complete_ranges_to(graph_hash(9), &verified, None, &no_excludes())
                .is_empty()
        );
        assert_eq!(
            graph.missing_complete_ranges_to(graph_hash(2), &verified, None, &no_excludes()),
            vec![missing_range(2, 2)]
        );
        assert_graph_invariants(&graph);
    }

    #[test]
    fn gate_streamed_is_a_hole() {
        // Streamed is best-effort (invariant L5), not authoritative: only Complete clears a hit block.
        let anchor = graph_hash(1);
        let block = graph_hash(2);
        let mut nodes = HashMap::new();
        nodes.insert(
            block,
            Node::Connected(ConnectedNode {
                parent: AnchoredRef::Anchor,
                data: BlockData {
                    number: 2,
                    logs_bloom: Some(hit_bloom()),
                    logs: BlockLogs::Streamed(BTreeMap::new()),
                },
            }),
        );
        let graph = BlocksGraph {
            anchor,
            nodes,
            observed_head: block,
        };
        assert_eq!(
            graph.missing_complete_ranges(ConnectedHash(block), &watched(), &no_excludes()),
            vec![MissingRange {
                from: 2,
                to: 2,
                hashes: vec![block]
            }]
        );
        assert_graph_invariants(&graph);
    }

    #[test]
    fn gate_empty_watched_yields_no_ranges() {
        // With no tracked pools, no block's logs are ever needed — not even a headerless one.
        let (graph, target) = gate_chain(&[(2, Some(hit_bloom()), false), (3, None, false)]);
        assert!(
            graph
                .missing_complete_ranges(target, &HashSet::new(), &no_excludes())
                .is_empty()
        );
        assert_graph_invariants(&graph);
    }

    proptest! {
        /// The gate's ranges cover exactly the canonical-path blocks that are not `Complete` and may
        /// touch a watched pool — coalesced, ascending, and non-adjacent. Compared against an
        /// independent recompute over the generated chain.
        #[test]
        fn gate_ranges_exactly_cover_unresolved_bloom_hit_blocks(
            specs in prop::collection::vec((any::<bool>(), 0u8..3), 1..12),
        ) {
            let chain: Vec<(u64, Option<Bloom>, bool)> = specs
                .iter()
                .enumerate()
                .map(|(index, (complete, kind))| {
                    let bloom = match kind {
                        0 => Some(hit_bloom()),
                        1 => Some(clear_bloom()),
                        _ => None,
                    };
                    ((index + 2) as u64, bloom, *complete)
                })
                .collect();
            let (graph, target) = gate_chain(&chain);
            let ranges = graph.missing_complete_ranges(target, &watched(), &no_excludes());

            // Independently expected hole block numbers (ascending, since numbers are contiguous).
            let holes: Vec<u64> = chain
                .iter()
                .filter_map(|(number, bloom, complete)| {
                    let touches = match bloom {
                        Some(bloom) => bloom.contains_input(BloomInput::Raw(watched_addr().as_slice())),
                        None => true,
                    };
                    (!complete && touches).then_some(*number)
                })
                .collect();

            let mut covered: Vec<u64> = Vec::new();
            for range in &ranges {
                prop_assert!(range.from <= range.to);
                covered.extend(range.from..=range.to);
                // The covered hashes name exactly the range's blocks (gate_chain keys hash by number).
                let expected_hashes: Vec<BlockHash> =
                    (range.from..=range.to).map(|number| graph_hash(number as usize)).collect();
                prop_assert_eq!(&range.hashes, &expected_hashes);
            }
            // Exact match ⇒ sound (only holes covered) and complete (every hole covered).
            prop_assert_eq!(covered, holes);
            // Properly coalesced: ranges are ascending with a real gap between them.
            for pair in ranges.windows(2) {
                prop_assert!(pair[0].to + 1 < pair[1].from);
            }
        }
    }

    // --- finalization fold (folded_pool_states) ------------------------------------------------

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

    fn v3_pool(byte: u8) -> PoolRef {
        PoolRef::uniswap_v3(Address::with_last_byte(byte), ChainKey::Ethereum)
    }

    /// The tracked/verified set for a fold test: exactly the `base` pools. Existing tests seed no new
    /// pool, so verified == base keeps the fold behavior identical to before seeding was added.
    fn verified_of(base: &HashMap<PoolRef, PoolState>) -> HashSet<PoolRef> {
        base.keys().copied().collect()
    }

    fn pool_log(pool: PoolRef, event: PoolLogEvent) -> PoolLog {
        PoolLog {
            pool: pool.key,
            // Deprecated; intra-block order is the BTreeMap key (assigned by `complete`/`streamed`).
            log_index: 0,
            event,
        }
    }

    /// Builds a connected linear chain (bloom always a hit, so it never affects the fold), block `i`
    /// carrying `BlockLogs` `blocks[i].1` with number `blocks[i].0`. Returns the graph and last block
    /// as target. `BlockLogs` is moved in, so each test constructs its own logs.
    fn fold_chain(blocks: Vec<(u64, BlockLogs)>) -> (BlocksGraph, ConnectedHash) {
        let anchor = graph_hash(1);
        let mut nodes = HashMap::new();
        let mut parent = AnchoredRef::Anchor;
        let mut last = anchor;
        for (index, (number, logs)) in blocks.into_iter().enumerate() {
            let hash = graph_hash(index + 2);
            nodes.insert(
                hash,
                Node::Connected(ConnectedNode {
                    parent,
                    data: BlockData {
                        number,
                        logs_bloom: Some(hit_bloom()),
                        logs,
                    },
                }),
            );
            parent = AnchoredRef::Block(ConnectedHash(hash));
            last = hash;
        }
        let graph = BlocksGraph {
            anchor,
            nodes,
            observed_head: last,
        };
        (graph, ConnectedHash(last))
    }

    fn keyed(logs: Vec<PoolLog>) -> BTreeMap<u64, PoolLog> {
        logs.into_iter()
            .enumerate()
            .map(|(index, log)| (index as u64, log))
            .collect()
    }

    fn complete(logs: Vec<PoolLog>) -> BlockLogs {
        BlockLogs::Complete(keyed(logs))
    }

    fn streamed(logs: Vec<PoolLog>) -> BlockLogs {
        BlockLogs::Streamed(keyed(logs))
    }

    // --- log-merge transition (L5) --------------------------------------------------------------

    /// A distinct swap keyed off `n`, so different `log_index`es carry visibly different logs.
    fn sw(n: u64) -> PoolLogEvent {
        PoolLogEvent::Swap {
            sqrt_price_x96: U160::from(n + 1),
            tick: tk(0),
            liquidity: u128::from(n) + 1,
        }
    }

    /// A pool log carrying an explicit intra-block `log_index` — the key the merge transition uses.
    fn log_at(log_index: u64) -> PoolLog {
        PoolLog {
            pool: v3_pool(0xA1).key,
            log_index,
            event: sw(log_index),
        }
    }

    fn logs_at(indices: &[u64]) -> Vec<PoolLog> {
        indices.iter().map(|&index| log_at(index)).collect()
    }

    /// `(is_complete, sorted keys)` for a block's logs, or `None` when still `Unknown`.
    fn log_keys(logs: &BlockLogs) -> Option<(bool, Vec<u64>)> {
        match logs {
            BlockLogs::Unknown => None,
            BlockLogs::Streamed(map) => Some((false, map.keys().copied().collect())),
            BlockLogs::Complete(map) => Some((true, map.keys().copied().collect())),
        }
    }

    fn block_logs(graph: &BlocksGraph, hash: BlockHash) -> Option<(bool, Vec<u64>)> {
        log_keys(&graph.get(hash).expect("block present").logs)
    }

    /// A graph with a single connected block (`graph_hash(1)`) off the anchor, logs `Unknown`.
    fn one_block_graph() -> BlocksGraph {
        BlocksGraph::new(graph_hash(0))
            .with_block(graph_hash(1), graph_hash(0), 1, hit_bloom()).0
    }

    #[test]
    fn streamed_logs_promote_unknown_to_streamed() {
        let graph = one_block_graph().with_streamed_logs(graph_hash(1), logs_at(&[2, 5]));
        assert_eq!(block_logs(&graph, graph_hash(1)), Some((false, vec![2, 5])));
        assert_graph_invariants(&graph);
    }

    #[test]
    fn streamed_logs_grow_by_union_on_log_index() {
        let graph = one_block_graph()
            .with_streamed_logs(graph_hash(1), logs_at(&[2, 5]))
            .with_streamed_logs(graph_hash(1), logs_at(&[3]));
        // Union on the true index — not positional, which would collide fragments at key 0.
        assert_eq!(block_logs(&graph, graph_hash(1)), Some((false, vec![2, 3, 5])));
    }

    #[test]
    fn complete_logs_replace_prior_streamed() {
        // Replace semantics (chosen over union): `Complete` is authoritative and self-sufficient.
        let graph = one_block_graph()
            .with_streamed_logs(graph_hash(1), logs_at(&[2, 5]))
            .with_complete_logs(graph_hash(1), logs_at(&[7]));
        assert_eq!(block_logs(&graph, graph_hash(1)), Some((true, vec![7])));
    }

    #[test]
    fn streamed_after_complete_is_a_noop() {
        // L5: never step backward off the authoritative `Complete`.
        let graph = one_block_graph()
            .with_complete_logs(graph_hash(1), logs_at(&[7]))
            .with_streamed_logs(graph_hash(1), logs_at(&[2]));
        assert_eq!(block_logs(&graph, graph_hash(1)), Some((true, vec![7])));
    }

    #[test]
    fn complete_logs_are_idempotent() {
        let graph = one_block_graph()
            .with_complete_logs(graph_hash(1), logs_at(&[7, 9]))
            .with_complete_logs(graph_hash(1), logs_at(&[7, 9]));
        assert_eq!(block_logs(&graph, graph_hash(1)), Some((true, vec![7, 9])));
    }

    #[test]
    fn logs_on_an_absent_block_are_a_noop() {
        let graph = one_block_graph()
            .with_streamed_logs(graph_hash(9), logs_at(&[1]))
            .with_complete_logs(graph_hash(9), logs_at(&[1]));
        assert_eq!(block_logs(&graph, graph_hash(1)), None); // admitted block untouched
        assert!(graph.get(graph_hash(9)).is_none()); // absent hash never materialized
        assert_graph_invariants(&graph);
    }

    #[test]
    fn logs_apply_to_a_pending_block() {
        // Parent `graph_hash(1)` is absent, so `graph_hash(2)` lands pending; logs still attach
        // (`BlockData` is shared across the connectivity split).
        let graph = BlocksGraph::new(graph_hash(0))
            .with_block(graph_hash(2), graph_hash(1), 2, hit_bloom()).0
            .with_streamed_logs(graph_hash(2), logs_at(&[4]));
        assert!(matches!(graph.nodes.get(&graph_hash(2)), Some(Node::Pending(_))));
        assert_eq!(block_logs(&graph, graph_hash(2)), Some((false, vec![4])));
        assert_graph_invariants(&graph);
    }

    // --- WS-miss detection (streamed vs authoritative divergence) -------------------------------

    #[test]
    fn streamed_mismatch_fires_only_on_key_set_divergence() {
        // The metric counts a streamed set whose `log_index` keys differ from the incoming
        // authoritative set — missing a log or carrying a spurious one both count; agreement does
        // not, whatever the payloads (dedup/authority is keyed, L3).
        let graph = one_block_graph().with_streamed_logs(graph_hash(1), logs_at(&[3, 5]));
        assert!(!graph.streamed_log_mismatch(graph_hash(1), &logs_at(&[3, 5])));
        assert!(graph.streamed_log_mismatch(graph_hash(1), &logs_at(&[3]))); // spurious stream log
        assert!(graph.streamed_log_mismatch(graph_hash(1), &logs_at(&[3, 5, 7]))); // missed log
        assert!(graph.streamed_log_mismatch(graph_hash(1), &[])); // non-empty stream vs empty truth
    }

    #[test]
    fn streamed_mismatch_ignores_unknown_complete_and_absent_blocks() {
        // Unknown = nothing streamed (the hole/backstop path, not a stream miss); Complete = an
        // idempotent authoritative re-receive; absent = pruned/never admitted. None are misses.
        let graph = one_block_graph();
        assert!(!graph.streamed_log_mismatch(graph_hash(1), &logs_at(&[3])));
        let graph = graph.with_complete_logs(graph_hash(1), logs_at(&[3]));
        assert!(!graph.streamed_log_mismatch(graph_hash(1), &logs_at(&[3, 4])));
        assert!(!graph.streamed_log_mismatch(graph_hash(9), &logs_at(&[3])));
    }

    #[derive(Debug, Clone)]
    enum LogOp {
        Streamed(Vec<u64>),
        Complete(Vec<u64>),
    }

    /// The reference model of L5: the expected authority + key set after a sequence of merges.
    #[derive(Debug, Clone, PartialEq)]
    enum ExpectedLogs {
        Unknown,
        Streamed(BTreeSet<u64>),
        Complete(BTreeSet<u64>),
    }

    fn log_op_strategy() -> impl Strategy<Value = LogOp> {
        let indices = prop::collection::vec(0u64..8, 0..5);
        prop_oneof![
            indices.clone().prop_map(LogOp::Streamed),
            indices.prop_map(LogOp::Complete),
        ]
    }

    proptest! {
        /// L5 on the merge op: over any sequence of streamed/complete merges the authority is
        /// monotone (`Unknown → Streamed → Complete`, never backward), `Streamed` only grows, and
        /// `Complete` replaces (the chosen rule). The graph's logs match the reference model at
        /// every step.
        #[test]
        fn log_merge_upholds_l5(ops in prop::collection::vec(log_op_strategy(), 1..12)) {
            let mut graph = one_block_graph();
            let mut expected = ExpectedLogs::Unknown;
            let mut prev_rank = 0u8;
            for op in ops {
                expected = match (op.clone(), expected) {
                    (LogOp::Streamed(idx), ExpectedLogs::Unknown) => {
                        ExpectedLogs::Streamed(idx.into_iter().collect())
                    }
                    (LogOp::Streamed(idx), ExpectedLogs::Streamed(mut prev)) => {
                        prev.extend(idx);
                        ExpectedLogs::Streamed(prev)
                    }
                    // A streamed merge onto Complete is a no-op.
                    (LogOp::Streamed(_), ExpectedLogs::Complete(keys)) => ExpectedLogs::Complete(keys),
                    (LogOp::Complete(idx), _) => ExpectedLogs::Complete(idx.into_iter().collect()),
                };
                graph = match op {
                    LogOp::Streamed(idx) => graph.with_streamed_logs(graph_hash(1), logs_at(&idx)),
                    LogOp::Complete(idx) => graph.with_complete_logs(graph_hash(1), logs_at(&idx)),
                };

                let actual = match &graph.get(graph_hash(1)).unwrap().logs {
                    BlockLogs::Unknown => ExpectedLogs::Unknown,
                    BlockLogs::Streamed(map) => ExpectedLogs::Streamed(map.keys().copied().collect()),
                    BlockLogs::Complete(map) => ExpectedLogs::Complete(map.keys().copied().collect()),
                };
                prop_assert_eq!(&actual, &expected);

                let rank = match actual {
                    ExpectedLogs::Unknown => 0,
                    ExpectedLogs::Streamed(_) => 1,
                    ExpectedLogs::Complete(_) => 2,
                };
                prop_assert!(rank >= prev_rank, "authority must never step backward");
                prev_rank = rank;
            }
            assert_graph_invariants(&graph);
        }
    }

    #[test]
    fn fold_empty_path_logs_returns_base() {
        let pool = v3_pool(0xA1);
        let base = HashMap::from([(pool, ps(10, 5, 1000))]);
        let (graph, target) = fold_chain(vec![(2, complete(vec![]))]);
        assert_eq!(
            graph.folded_pool_states(&base, &verified_of(&base), target, Authority::RequireComplete),
            base
        );
        assert_graph_invariants(&graph);
    }

    #[test]
    fn fold_swap_supersedes_base() {
        let pool = v3_pool(0xA1);
        let base = HashMap::from([(pool, ps(10, 5, 1000))]);
        let (graph, target) = fold_chain(vec![(
            2,
            complete(vec![pool_log(
                pool,
                PoolLogEvent::Swap {
                    sqrt_price_x96: U160::from(777u128),
                    tick: tk(9),
                    liquidity: 42,
                },
            )]),
        )]);
        assert_eq!(
            graph.folded_pool_states(&base, &verified_of(&base), target, Authority::RequireComplete),
            HashMap::from([(pool, ps(777, 9, 42))])
        );
        assert_graph_invariants(&graph);
    }

    #[test]
    fn fold_in_range_mint_adds_to_base_liquidity() {
        let pool = v3_pool(0xA1);
        let base = HashMap::from([(pool, ps(10, 5, 1000))]);
        let (graph, target) = fold_chain(vec![(
            2,
            complete(vec![pool_log(
                pool,
                PoolLogEvent::Mint {
                    tick_lower: tk(0),
                    tick_upper: tk(10),
                    amount: 500,
                },
            )]),
        )]);
        assert_eq!(
            graph.folded_pool_states(&base, &verified_of(&base), target, Authority::RequireComplete),
            HashMap::from([(pool, ps(10, 5, 1500))])
        );
        assert_graph_invariants(&graph);
    }

    #[test]
    fn fold_ignores_pool_absent_from_base_and_unverified() {
        // A Complete block carries a Swap for a pool that is neither in `base` nor `verified`: it is
        // not a seed candidate (only verified keys are seeded), so it must not appear in the result.
        let tracked = v3_pool(0xA1);
        let untracked = v3_pool(0xB2);
        let base = HashMap::from([(tracked, ps(10, 5, 1000))]);
        let (graph, target) = fold_chain(vec![(
            2,
            complete(vec![pool_log(
                untracked,
                PoolLogEvent::Swap {
                    sqrt_price_x96: U160::from(1u128),
                    tick: tk(1),
                    liquidity: 1,
                },
            )]),
        )]);
        // `verified_of(&base)` excludes `untracked`, so it is never seeded.
        assert_eq!(
            graph.folded_pool_states(&base, &verified_of(&base), target, Authority::RequireComplete),
            base
        );
        assert_graph_invariants(&graph);
    }

    #[test]
    fn fold_seeds_verified_pool_absent_from_base_via_absolute_log() {
        // Blocker 1: a verified pool absent from `base` (discovered after bootstrap) whose run begins
        // with an absolute Swap is seeded into the fold from `derive_pool_state(None, run)`.
        let tracked = v3_pool(0xA1);
        let new_pool = v3_pool(0xC1);
        let base = HashMap::from([(tracked, ps(10, 5, 1000))]);
        let verified = HashSet::from([tracked, new_pool]);
        let (graph, target) = fold_chain(vec![(
            2,
            complete(vec![pool_log(
                new_pool,
                PoolLogEvent::Swap {
                    sqrt_price_x96: U160::from(777u128),
                    tick: tk(9),
                    liquidity: 42,
                },
            )]),
        )]);
        assert_eq!(
            graph.folded_pool_states(&base, &verified, target, Authority::RequireComplete),
            HashMap::from([(tracked, ps(10, 5, 1000)), (new_pool, ps(777, 9, 42))])
        );
        assert_graph_invariants(&graph);
    }

    #[test]
    fn fold_does_not_seed_verified_pool_from_delta_only_run() {
        // A verified new pool whose only log is a Mint has no absolute anchor, so
        // `derive_pool_state(None, [Mint])` is `None`: the fold leaves it unseeded here — its base
        // comes from the anchor-height seed path (`schedule_finalized_pool_seed_requests`) instead.
        let new_pool = v3_pool(0xC1);
        let base: HashMap<PoolRef, PoolState> = HashMap::new();
        let verified = HashSet::from([new_pool]);
        let (graph, target) = fold_chain(vec![(
            2,
            complete(vec![pool_log(
                new_pool,
                PoolLogEvent::Mint {
                    tick_lower: tk(0),
                    tick_upper: tk(10),
                    amount: 500,
                },
            )]),
        )]);
        assert!(
            graph
                .folded_overlay(&base, &verified, target, Authority::RequireComplete)
                .is_empty()
        );
        assert_graph_invariants(&graph);
    }

    #[test]
    fn finalization_waits_for_complete_before_seeding_new_pool() {
        // Watched-set widening: a verified pool absent from `base` whose Swap sits in a Streamed-only
        // (not yet Complete) bloom-touching block is a finalization hole — `finalized_to` must NOT
        // advance the anchor past it (which would drop the swap forever), it waits for Complete logs.
        // `watched_pool()` sits at `watched_addr()`, the one address `fold_chain`'s `hit_bloom` sets,
        // so the block's bloom actually touches it.
        let new_pool = watched_pool();
        let base: HashMap<PoolRef, PoolState> = HashMap::new();
        let verified = HashSet::from([new_pool]);
        let swap = PoolLogEvent::Swap {
            sqrt_price_x96: U160::from(777u128),
            tick: tk(9),
            liquidity: 42,
        };

        // Streamed: the block is a hole under RequireComplete, so the anchor stays put and nothing seeds.
        let (graph, target) = fold_chain(vec![(2, streamed(vec![pool_log(new_pool, swap.clone())]))]);
        let (graph, snapshot) = graph.finalized_to(target.0, &base, &verified, None);
        assert_eq!(graph.anchor, graph_hash(1));
        assert!(snapshot.is_empty());

        // Complete: the same block now folds — the anchor advances and the new pool is absolute-seeded.
        let (graph, target) = fold_chain(vec![(2, complete(vec![pool_log(new_pool, swap)]))]);
        let (graph, snapshot) = graph.finalized_to(target.0, &base, &verified, None);
        assert_eq!(graph.anchor, graph_hash(2));
        assert_eq!(snapshot.get(&new_pool), Some(&ps(777, 9, 42)));
    }

    #[test]
    fn finalization_ignores_connected_side_fork_target() {
        // Decision (d): a finality signal for a connected block OFF the canonical chain
        // (anchor → observed_head) must not reanchor — advancing would prune the head branch on a
        // possibly transient head/finality feed disagreement. No-op and wait for the head instead.
        let pool = v3_pool(0xA1);
        let base = HashMap::from([(pool, ps(10, 5, 1000))]);
        let (mut graph, _tip) = fold_chain(vec![(2, complete(vec![]))]);
        // A second child of the anchor: connected and fully foldable, but the head is on block 2.
        let fork = graph_hash(3);
        graph.nodes.insert(
            fork,
            Node::Connected(ConnectedNode {
                parent: AnchoredRef::Anchor,
                data: BlockData {
                    number: 2,
                    logs_bloom: Some(hit_bloom()),
                    logs: complete(vec![pool_log(
                        pool,
                        PoolLogEvent::Swap {
                            sqrt_price_x96: U160::from(777u128),
                            tick: tk(9),
                            liquidity: 42,
                        },
                    )]),
                },
            }),
        );

        let (graph, snapshot) = graph.finalized_to(fork, &base, &verified_of(&base), None);

        assert_eq!(graph.anchor, graph_hash(1));
        assert_eq!(graph.observed_head, graph_hash(2));
        assert!(graph.nodes.contains_key(&graph_hash(2)));
        assert_eq!(snapshot, base);
        assert_graph_invariants(&graph);
    }

    #[test]
    fn finalization_advances_to_mid_canonical_target() {
        // The canonical-membership gate must not over-refuse: a target strictly between the anchor
        // and the observed head is on the canonical chain and finalizes normally.
        let pool = v3_pool(0xA1);
        let base = HashMap::from([(pool, ps(10, 5, 1000))]);
        let (graph, _tip) = fold_chain(vec![
            (
                2,
                complete(vec![pool_log(
                    pool,
                    PoolLogEvent::Swap {
                        sqrt_price_x96: U160::from(777u128),
                        tick: tk(9),
                        liquidity: 42,
                    },
                )]),
            ),
            (3, complete(vec![])),
        ]);

        let (graph, snapshot) =
            graph.finalized_to(graph_hash(2), &base, &verified_of(&base), None);

        assert_eq!(graph.anchor, graph_hash(2));
        assert_eq!(graph.observed_head, graph_hash(3));
        assert_eq!(snapshot.get(&pool), Some(&ps(777, 9, 42)));
        assert_graph_invariants(&graph);
    }

    #[test]
    fn fold_composes_across_blocks_in_path_order() {
        // Swap in the first block sets absolute state; mint in the second adjusts it.
        let pool = v3_pool(0xA1);
        let base = HashMap::from([(pool, ps(10, 5, 1000))]);
        let (graph, target) = fold_chain(vec![
            (
                2,
                complete(vec![pool_log(
                    pool,
                    PoolLogEvent::Swap {
                        sqrt_price_x96: U160::from(50u128),
                        tick: tk(0),
                        liquidity: 100,
                    },
                )]),
            ),
            (
                3,
                complete(vec![pool_log(
                    pool,
                    PoolLogEvent::Mint {
                        tick_lower: tk(-10),
                        tick_upper: tk(10),
                        amount: 25,
                    },
                )]),
            ),
        ]);
        assert_eq!(
            graph.folded_pool_states(&base, &verified_of(&base), target, Authority::RequireComplete),
            HashMap::from([(pool, ps(50, 0, 125))])
        );
        assert_graph_invariants(&graph);
    }

    #[test]
    fn fold_require_complete_ignores_streamed_logs() {
        // Under RequireComplete a Streamed block contributes nothing, so the pool keeps its base.
        let pool = v3_pool(0xA1);
        let base = HashMap::from([(pool, ps(10, 5, 1000))]);
        let (graph, target) = fold_chain(vec![(
            2,
            streamed(vec![pool_log(
                pool,
                PoolLogEvent::Swap {
                    sqrt_price_x96: U160::from(777u128),
                    tick: tk(9),
                    liquidity: 42,
                },
            )]),
        )]);
        assert_eq!(
            graph.folded_pool_states(&base, &verified_of(&base), target, Authority::RequireComplete),
            base
        );
        assert_graph_invariants(&graph);
    }

    #[test]
    fn fold_allow_streamed_reads_streamed_logs() {
        // The authority seam: the same Streamed block IS read under AllowStreamed (optimization view).
        let pool = v3_pool(0xA1);
        let base = HashMap::from([(pool, ps(10, 5, 1000))]);
        let (graph, target) = fold_chain(vec![(
            2,
            streamed(vec![pool_log(
                pool,
                PoolLogEvent::Swap {
                    sqrt_price_x96: U160::from(777u128),
                    tick: tk(9),
                    liquidity: 42,
                },
            )]),
        )]);
        assert_eq!(
            graph.folded_pool_states(&base, &verified_of(&base), target, Authority::AllowStreamed),
            HashMap::from([(pool, ps(777, 9, 42))])
        );
        assert_graph_invariants(&graph);
    }

    // --- optimization read: stop-on-unknown (goal 3) --------------------------------------------

    /// A `Swap` event, which sets absolute pool state (so the fold result is the last swap on the path).
    fn swap_to(sqrt: u128, tick: i32, liquidity: u128) -> PoolLogEvent {
        PoolLogEvent::Swap {
            sqrt_price_x96: U160::from(sqrt),
            tick: tk(tick),
            liquidity,
        }
    }

    /// A `Swap` for the watched pool, wrapped as `Complete` block logs.
    fn complete_swap(sqrt: u128, tick: i32, liquidity: u128) -> BlockLogs {
        complete(vec![pool_log(watched_pool(), swap_to(sqrt, tick, liquidity))])
    }

    /// Builds a connected linear chain off the anchor with explicit per-block `(logs_bloom, logs)`;
    /// block `i` (0-based) has hash `graph_hash(i + 2)` and number `i + 1`, and `observed_head` is the
    /// last block. Gives the per-block bloom/authority control `fold_chain` (all-hit) does not.
    fn opt_chain(blocks: Vec<(Option<Bloom>, BlockLogs)>) -> BlocksGraph {
        let anchor = graph_hash(1);
        let mut nodes = HashMap::new();
        let mut parent = AnchoredRef::Anchor;
        let mut last = anchor;
        for (index, (bloom, logs)) in blocks.into_iter().enumerate() {
            let hash = graph_hash(index + 2);
            nodes.insert(
                hash,
                Node::Connected(ConnectedNode {
                    parent,
                    data: BlockData {
                        number: (index as u64) + 1,
                        logs_bloom: bloom,
                        logs,
                    },
                }),
            );
            parent = AnchoredRef::Block(ConnectedHash(hash));
            last = hash;
        }
        BlocksGraph {
            anchor,
            nodes,
            observed_head: last,
        }
    }

    #[test]
    fn optimization_stops_at_unknown_bloom_touching_block() {
        // [good][good][Unknown+hit][good]: fold the two good blocks, stop before the gap, and never
        // reach the post-gap block. Frontier is the last folded (2nd good) block.
        let pool = watched_pool();
        let base = HashMap::from([(pool, ps(10, 5, 1000))]);
        let graph = opt_chain(vec![
            (Some(hit_bloom()), complete_swap(100, 1, 10)),
            (Some(hit_bloom()), complete_swap(200, 2, 20)),
            (Some(hit_bloom()), BlockLogs::Unknown),
            (Some(hit_bloom()), complete_swap(999, 9, 99)),
        ]);
        let (overlay, frontier) = graph.optimization_pool_states(&base, &verified_of(&base), None);
        assert_eq!(overlay, HashMap::from([(pool, ps(200, 2, 20))]));
        assert_eq!(frontier, graph_hash(3));
        assert_graph_invariants(&graph);
    }

    #[test]
    fn optimization_skips_unknown_with_clear_bloom() {
        // An Unknown block with a clear bloom is proven untouched, so the fold continues past it to the
        // head; the post-gap swap is applied.
        let pool = watched_pool();
        let base = HashMap::from([(pool, ps(10, 5, 1000))]);
        let graph = opt_chain(vec![
            (Some(hit_bloom()), complete_swap(100, 1, 10)),
            (Some(clear_bloom()), BlockLogs::Unknown),
            (Some(hit_bloom()), complete_swap(200, 2, 20)),
        ]);
        let (overlay, frontier) = graph.optimization_pool_states(&base, &verified_of(&base), None);
        assert_eq!(overlay, HashMap::from([(pool, ps(200, 2, 20))]));
        assert_eq!(frontier, graph_hash(4));
        assert_graph_invariants(&graph);
    }

    #[test]
    fn optimization_folds_through_streamed_block() {
        // A Streamed (best-effort WS) block is readable under AllowStreamed, so it never stops the fold
        // and its swap contributes to the overlay.
        let pool = watched_pool();
        let base = HashMap::from([(pool, ps(10, 5, 1000))]);
        let graph = opt_chain(vec![(
            Some(hit_bloom()),
            streamed(vec![pool_log(pool, swap_to(150, 3, 15))]),
        )]);
        let (overlay, frontier) = graph.optimization_pool_states(&base, &verified_of(&base), None);
        assert_eq!(overlay, HashMap::from([(pool, ps(150, 3, 15))]));
        assert_eq!(frontier, graph_hash(2));
        assert_graph_invariants(&graph);
    }

    #[test]
    fn optimization_stops_at_header_less_unknown_block() {
        // A header-less Unknown block (no bloom) conservatively may-touch, so it stops the fold; the
        // post-gap block is never reached.
        let pool = watched_pool();
        let base = HashMap::from([(pool, ps(10, 5, 1000))]);
        let graph = opt_chain(vec![
            (Some(hit_bloom()), complete_swap(100, 1, 10)),
            (None, BlockLogs::Unknown),
            (Some(hit_bloom()), complete_swap(999, 9, 99)),
        ]);
        let (overlay, frontier) = graph.optimization_pool_states(&base, &verified_of(&base), None);
        assert_eq!(overlay, HashMap::from([(pool, ps(100, 1, 10))]));
        assert_eq!(frontier, graph_hash(2));
        assert_graph_invariants(&graph);
    }

    #[test]
    fn optimization_no_gap_folds_to_head() {
        // With no unknown blocks the fold spans the whole path; frontier is the observed head.
        let pool = watched_pool();
        let base = HashMap::from([(pool, ps(10, 5, 1000))]);
        let graph = opt_chain(vec![
            (Some(hit_bloom()), complete_swap(100, 1, 10)),
            (Some(hit_bloom()), complete_swap(200, 2, 20)),
        ]);
        let (overlay, frontier) = graph.optimization_pool_states(&base, &verified_of(&base), None);
        assert_eq!(overlay, HashMap::from([(pool, ps(200, 2, 20))]));
        assert_eq!(frontier, graph.observed_head);
        assert_eq!(frontier, graph_hash(3));
        assert_graph_invariants(&graph);
    }

    #[test]
    fn optimization_empty_graph_yields_empty_overlay_at_anchor() {
        // No recent blocks: nothing to fold, overlay empty, frontier is the anchor.
        let pool = watched_pool();
        let base = HashMap::from([(pool, ps(10, 5, 1000))]);
        let graph = BlocksGraph::new(graph_hash(0));
        let (overlay, frontier) = graph.optimization_pool_states(&base, &verified_of(&base), None);
        assert!(overlay.is_empty());
        assert_eq!(frontier, graph_hash(0));
    }

    #[test]
    fn optimization_non_connected_head_yields_empty_overlay_at_anchor() {
        // `observed_head` points at a Pending block (its parent never connected): there is no foldable
        // suffix, so the overlay is empty and the frontier is the anchor.
        let pool = watched_pool();
        let base = HashMap::from([(pool, ps(10, 5, 1000))]);
        let graph = BlocksGraph::new(graph_hash(0))
            .with_block(graph_hash(3), graph_hash(2), 3, hit_bloom()).0
            .with_observed_head(graph_hash(3));
        assert!(matches!(graph.nodes.get(&graph_hash(3)), Some(Node::Pending(_))));
        let (overlay, frontier) = graph.optimization_pool_states(&base, &verified_of(&base), None);
        assert!(overlay.is_empty());
        assert_eq!(frontier, graph_hash(0));
        assert_graph_invariants(&graph);
    }

    // --- finalization re-root (reanchored_to) ---------------------------------------------------

    /// A tracked pool whose v3 address is the watched address, so [`hit_bloom`] marks its blocks.
    fn watched_pool() -> PoolRef {
        PoolRef::uniswap_v3(watched_addr(), ChainKey::Ethereum)
    }

    /// A complete-logs connected node with a clear bloom (so it never triggers the gate).
    fn clear_complete_node(parent: AnchoredRef, number: u64) -> Node {
        Node::Connected(ConnectedNode {
            parent,
            data: BlockData {
                number,
                logs_bloom: Some(clear_bloom()),
                logs: BlockLogs::Complete(BTreeMap::new()),
            },
        })
    }

    #[test]
    fn reanchor_to_anchor_is_noop() {
        let anchor = graph_hash(1);
        let graph = BlocksGraph::new(anchor)
            .with_block(graph_hash(2), anchor, 2, Bloom::repeat_byte(0)).0;
        let base = HashMap::from([(v3_pool(0xA1), ps(1, 1, 1))]);
        // Reanchor to the current anchor is a base-clone no-op.
        let (graph, snapshot) = graph.reanchored_to(ConnectedHash(anchor), &base, &verified_of(&base));
        assert_eq!(graph.anchor, anchor);
        assert!(graph.connected(graph_hash(2)).is_some());
        assert_eq!(snapshot, base);
        assert_graph_invariants(&graph);
    }

    #[test]
    fn finalization_refuses_unknown_bloom_hit_block() {
        // A4: an Unknown-logs block whose bloom hits a tracked pool is a hole — finalization must
        // not advance over it; the graph and the caller's snapshot (borrowed) are untouched, and the
        // hole is exactly what `missing_complete_ranges` reports for the finalization backfill.
        let pool = watched_pool();
        let base = HashMap::from([(pool, ps(10, 5, 1000))]);
        let (graph, target) = fold_chain(vec![(7, BlockLogs::Unknown)]);
        let target_hash = target.0;
        let watched = watched_addresses(&verified_of(&base), None);
        assert_eq!(
            graph.missing_complete_ranges(target, &watched, &no_excludes()),
            vec![MissingRange {
                from: 7,
                to: 7,
                hashes: vec![target_hash]
            }]
        );
        let (graph, snapshot) = graph.finalized_to(target_hash, &base, &verified_of(&base), None);
        assert_eq!(graph.anchor, graph_hash(1));
        assert!(graph.connected(graph_hash(2)).is_some());
        assert_eq!(snapshot, base);
        assert_graph_invariants(&graph);
    }

    #[test]
    fn reanchor_advances_prunes_and_reclassifies() {
        // anchor → b2 → b3 → b4 (all Complete). Reanchor to b2: b2 becomes the anchor (removed from
        // nodes), b3 is reclassified to an Anchor parent, b4 keeps its Block(b3) parent.
        let (graph, _) = fold_chain(vec![
            (2, complete(vec![])),
            (3, complete(vec![])),
            (4, complete(vec![])),
        ]);
        let (b2, b3, b4) = (graph_hash(2), graph_hash(3), graph_hash(4));
        let (graph, snapshot) = graph.reanchored_to(ConnectedHash(b2), &HashMap::new(), &HashSet::new());
        assert!(snapshot.is_empty());
        assert_eq!(graph.anchor, b2);
        assert!(graph.nodes.get(&b2).is_none());
        assert!(matches!(
            graph.nodes.get(&b3),
            Some(Node::Connected(ConnectedNode {
                parent: AnchoredRef::Anchor,
                ..
            }))
        ));
        assert!(matches!(
            graph.nodes.get(&b4),
            Some(Node::Connected(ConnectedNode {
                parent: AnchoredRef::Block(ConnectedHash(parent)),
                ..
            })) if *parent == b3
        ));
        assert_eq!(graph.observed_head, b4);
        assert_graph_invariants(&graph);
    }

    #[test]
    fn reanchor_prunes_sibling_forks() {
        // anchor has two children b2 and b3; b2 has a child b4. Reanchor to b2 drops the b3 fork.
        let anchor = graph_hash(1);
        let (b2, b3, b4) = (graph_hash(2), graph_hash(3), graph_hash(4));
        let mut nodes = HashMap::new();
        nodes.insert(b2, clear_complete_node(AnchoredRef::Anchor, 2));
        nodes.insert(b3, clear_complete_node(AnchoredRef::Anchor, 2));
        nodes.insert(b4, clear_complete_node(AnchoredRef::Block(ConnectedHash(b2)), 3));
        let graph = BlocksGraph {
            anchor,
            nodes,
            observed_head: b4,
        };
        let (graph, _) = graph.reanchored_to(ConnectedHash(b2), &HashMap::new(), &HashSet::new());
        assert_eq!(graph.anchor, b2);
        assert!(graph.nodes.get(&b3).is_none());
        assert!(graph.nodes.get(&b2).is_none());
        assert!(matches!(
            graph.nodes.get(&b4),
            Some(Node::Connected(ConnectedNode {
                parent: AnchoredRef::Anchor,
                ..
            }))
        ));
        assert_graph_invariants(&graph);
    }

    #[test]
    fn reanchor_resets_observed_head_when_pruned() {
        // observed_head sits on the b3 sibling fork; reanchoring to b2 prunes it, so the head resets
        // to the new anchor.
        let anchor = graph_hash(1);
        let (b2, b3) = (graph_hash(2), graph_hash(3));
        let mut nodes = HashMap::new();
        nodes.insert(b2, clear_complete_node(AnchoredRef::Anchor, 2));
        nodes.insert(b3, clear_complete_node(AnchoredRef::Anchor, 2));
        let graph = BlocksGraph {
            anchor,
            nodes,
            observed_head: b3,
        };
        let (graph, _) = graph.reanchored_to(ConnectedHash(b2), &HashMap::new(), &HashSet::new());
        assert_eq!(graph.observed_head, b2);
        assert_graph_invariants(&graph);
    }

    #[test]
    fn reanchor_to_observed_head_empties_chain() {
        // Finalizing right up to the head: the head becomes the anchor, no recent blocks remain, and
        // the canonical chain is empty.
        let (graph, target) = fold_chain(vec![(2, complete(vec![])), (3, complete(vec![]))]);
        let b3 = graph_hash(3);
        let (graph, _) = graph.reanchored_to(target, &HashMap::new(), &HashSet::new());
        assert_eq!(graph.anchor, b3);
        assert!(graph.nodes.is_empty());
        assert_eq!(graph.observed_head, b3);
        assert!(graph.canonical_oldest_to_newest().is_empty());
        assert_graph_invariants(&graph);
    }

    #[test]
    fn reanchor_folds_finalized_snapshot() {
        // A Complete swap on the finalized prefix advances the pool's snapshot at the new anchor.
        let pool = watched_pool();
        let base = HashMap::from([(pool, ps(10, 5, 1000))]);
        let (graph, target) = fold_chain(vec![(
            2,
            complete(vec![pool_log(
                pool,
                PoolLogEvent::Swap {
                    sqrt_price_x96: U160::from(777u128),
                    tick: tk(9),
                    liquidity: 42,
                },
            )]),
        )]);
        let (graph, snapshot) = graph.reanchored_to(target, &base, &verified_of(&base));
        assert_eq!(snapshot, HashMap::from([(pool, ps(777, 9, 42))]));
        assert_eq!(graph.anchor, graph_hash(2));
        assert!(graph.nodes.is_empty());
        assert_graph_invariants(&graph);
    }

    proptest! {
        /// Reanchoring to any connected node (with no tracked pools, so the gate always passes)
        /// re-roots the graph: it upholds every invariant, retains exactly the descendants of the new
        /// anchor, leaves no pending, and keeps a valid observed head — across all shapes and orders.
        #[test]
        fn reanchor_preserves_invariants_and_prunes_to_descendants(
            plan in admission_plan_strategy(),
            anchor_seed in any::<usize>(),
        ) {
            let node_count = plan.parents.len();
            let new_anchor_idx = 1 + anchor_seed % (node_count - 1);
            let new_anchor_hash = graph_hash(new_anchor_idx);
            let graph = admit_in_order(&plan, &plan.order_a)
                .with_observed_head(graph_hash(node_count - 1));

            // Independently: the strict descendants of new_anchor by the generated parent links.
            let mut expected: HashSet<BlockHash> = HashSet::new();
            for node in 1..node_count {
                if node == new_anchor_idx {
                    continue;
                }
                let mut current = node;
                while current != 0 {
                    if current == new_anchor_idx {
                        expected.insert(graph_hash(node));
                        break;
                    }
                    current = plan.parents[current];
                }
            }

            let (regraphed, snapshot) =
                graph.reanchored_to(ConnectedHash(new_anchor_hash), &HashMap::new(), &HashSet::new());
            prop_assert!(snapshot.is_empty());
            prop_assert_eq!(regraphed.anchor, new_anchor_hash);
            assert_graph_invariants(&regraphed);
            prop_assert!(regraphed.nodes.values().all(|node| matches!(node, Node::Connected(_))));
            let retained: HashSet<BlockHash> = regraphed.nodes.keys().copied().collect();
            prop_assert_eq!(retained, expected);
        }
    }

    #[test]
    fn blocks_graph_starts_empty() {
        let graph = BlocksGraph::new(BlockHash::ZERO);

        assert!(graph.is_empty());
    }

    // --- bootstrap seed (from_seed) --------------------------------------------------------------

    /// A pool log with an explicit intra-block index for a given pool — seed payloads are keyed by
    /// `log_index` exactly like live ingestion.
    fn seed_log(log_index: u64, pool: PoolRef, event: PoolLogEvent) -> PoolLog {
        PoolLog {
            pool: pool.key,
            log_index,
            event,
        }
    }

    /// The seed window under test: anchor(1) → s2 (absolute swap) → s3 (gap filler, proven-empty
    /// logs) → s4 (in-range mint). Folding it for [`watched_pool`] from any base yields
    /// `ps(500, 5, base_liquidity-agnostic 5_000 + 100)` since the swap is absolute.
    fn seeded_graph() -> BlocksGraph {
        let pool = watched_pool();
        BlocksGraph::from_seed(
            graph_hash(1),
            vec![
                (
                    graph_hash(2),
                    graph_hash(1),
                    2,
                    vec![seed_log(
                        0,
                        pool,
                        PoolLogEvent::Swap {
                            sqrt_price_x96: U160::from(500),
                            tick: tk(5),
                            liquidity: 5_000,
                        },
                    )],
                ),
                (graph_hash(3), graph_hash(2), 3, vec![]),
                (
                    graph_hash(4),
                    graph_hash(3),
                    4,
                    vec![seed_log(
                        1,
                        pool,
                        PoolLogEvent::Mint {
                            tick_lower: tk(0),
                            tick_upper: tk(10),
                            amount: 100,
                        },
                    )],
                ),
            ],
        )
    }

    #[test]
    fn seed_builds_connected_header_less_complete_chain() {
        let graph = seeded_graph();
        for index in 2..=4 {
            let node = graph
                .connected(graph_hash(index))
                .expect("seed block must be connected");
            assert_eq!(node.data.logs_bloom, None, "seed blocks are header-less");
            assert!(matches!(node.data.logs, BlockLogs::Complete(_)));
        }
        // Activation mirrors legacy: the head starts at the anchor; live replay advances it.
        assert_eq!(graph.observed_head, graph_hash(1));
        assert!(graph.canonical_hashes().is_empty());
        assert_graph_invariants(&graph);
    }

    #[test]
    fn seed_window_folds_into_optimization_read_once_head_connects() {
        // The first live block lands on the seeded chain and the whole window folds immediately —
        // the warmup property: no header walk is needed before serving recent state.
        let pool = watched_pool();
        let base = HashMap::from([(pool, ps(10, 0, 1_000))]);
        let graph = seeded_graph()
            .admitted(graph_hash(5), graph_hash(4), 5, clear_bloom())
            .with_observed_head(graph_hash(5));
        let (overlay, frontier) =
            graph.optimization_pool_states(&base, &verified_of(&base), None);
        assert_eq!(frontier, graph_hash(5));
        assert_eq!(overlay.get(&pool), Some(&ps(500, 5, 5_100)));
        assert_graph_invariants(&graph);
    }

    #[test]
    fn seed_finalizes_through_header_less_window() {
        // RequireComplete folds the seeded window: header-less (`None`-bloom) blocks are foldable
        // because they are `Complete` — including the empty gap filler, whose emptiness the ranged
        // response proved. The anchor advances with no backfill ranges. The head must have connected
        // through the window first (decision (d): off-canonical targets no-op) — same as legacy,
        // whose canonical tip also sits at the anchor until the first post-activation head.
        let pool = watched_pool();
        let base = HashMap::from([(pool, ps(10, 0, 1_000))]);
        let (graph, snapshot) = seeded_graph()
            .with_observed_head(graph_hash(4))
            .finalized_to(graph_hash(4), &base, &verified_of(&base), None);
        assert_eq!(graph.anchor, graph_hash(4));
        assert_eq!(snapshot.get(&pool), Some(&ps(500, 5, 5_100)));
        assert_graph_invariants(&graph);
    }

    #[test]
    fn seed_window_seeds_pool_verified_after_activation() {
        // Registry growth after activation: the seed stores logs registry-free, so a pool verified
        // only later folds out of the seeded window exactly as it would from live blocks.
        let pool = watched_pool();
        let base: HashMap<PoolRef, PoolState> = HashMap::new();
        let graph = seeded_graph().with_observed_head(graph_hash(4));

        let (before, _) = graph.optimization_pool_states(&base, &HashSet::new(), None);
        assert!(before.is_empty(), "unverified pool must stay invisible");

        let (after, frontier) =
            graph.optimization_pool_states(&base, &HashSet::from([pool]), None);
        assert_eq!(frontier, graph_hash(4));
        assert_eq!(after.get(&pool), Some(&ps(500, 5, 5_100)));
        assert_graph_invariants(&graph);
    }

    #[test]
    fn seed_skips_degenerate_entries_first_seen_kept() {
        let pool = watched_pool();
        let graph = BlocksGraph::from_seed(
            graph_hash(1),
            vec![
                (graph_hash(2), graph_hash(1), 2, vec![seed_log(0, pool, sw(7))]),
                // Self-parent, anchor re-admit, and a conflicting duplicate: skipped, first-seen kept.
                (graph_hash(9), graph_hash(9), 9, vec![]),
                (graph_hash(1), graph_hash(2), 1, vec![]),
                (graph_hash(2), graph_hash(4), 2, vec![]),
            ],
        );
        assert_eq!(graph.node_hashes(), HashSet::from([graph_hash(2)]));
        let node = graph
            .connected(graph_hash(2))
            .expect("first-seen block must stay connected");
        assert!(matches!(&node.parent, AnchoredRef::Anchor));
        assert_graph_invariants(&graph);
    }
}
