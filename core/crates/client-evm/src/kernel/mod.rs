use std::collections::{HashMap, HashSet};

use alloy::primitives::{Address, BlockHash, Bloom, BloomInput};

pub(crate) mod pending_requests;
pub(crate) mod pool_registry;
pub(crate) mod token_registry;

use self::{pending_requests::*, pool_registry::*, token_registry::*};
use crate::{ChainKey, PoolLog, PoolLogEvent, derive_pool_state, pool_state::*, tick::Tick, uniswap_v4};

enum PoolLogsStatus {
    Unknown,
    /// Logs observed via the best-effort subscription: candidates are real but the set is not
    /// guaranteed complete, so it is provisional and never gates the optimizer or finalization.
    Partial(HashSet<ProtocolPoolKey>),
    /// Logs from an authoritative `GetBlockLogs`: the candidate set is complete.
    Resolved(HashSet<ProtocolPoolKey>),
}

impl PoolLogsStatus {
    /// The observed candidate set for `Partial`/`Resolved`, or `None` while `Unknown`. Lets
    /// consumers that treat any observation alike (candidate discovery, dirty tracking, metadata
    /// validation) avoid distinguishing provisional from authoritative.
    fn candidates(&self) -> Option<&HashSet<ProtocolPoolKey>> {
        match self {
            PoolLogsStatus::Unknown => None,
            PoolLogsStatus::Partial(candidates) | PoolLogsStatus::Resolved(candidates) => {
                Some(candidates)
            }
        }
    }
}

struct BlockNode {
    parent_hash: BlockHash,
    /// The block header's `logsBloom`, when the block entered the graph from a header (head
    /// observation or `GetBlockHeader`). `None` for blocks materialized without a header
    /// (finalized anchor, bootstrap-inferred), which are never bloom-gated.
    logs_bloom: Option<Bloom>,
    pool_logs: PoolLogsStatus,
    pool_snapshots: HashMap<PoolRef, PoolState>,
    pool_data_failures: HashMap<PoolRef, PoolDataFailure>,
}

enum NewBlockError {
    SelfParentBlock(BlockHash, BlocksGraph),
    ExistingBlock(BlocksGraph),
    ConflictingBlockParent,
    CycleDetected,
}

struct BlocksGraph(HashMap<BlockHash, BlockNode>);

impl BlocksGraph {
    /// Creates the volatile graph for recent, non-finalized blocks.
    /// Added so resets can discard reorg-prone block state without touching finalized or registry state.
    fn new() -> BlocksGraph {
        BlocksGraph(HashMap::new())
    }

    /// Looks up a tracked recent block without exposing the backing map.
    /// Keeps graph access centralized so future refactors can preserve graph invariants behind this type.
    fn get(&self, hash: &BlockHash) -> Option<&BlockNode> {
        self.0.get(hash)
    }

    /// Removes and returns one pool's snapshot from a block, taking ownership without cloning.
    /// Added so finalized compaction can move snapshots out of soon-pruned blocks into the snapshot.
    fn take_pool_snapshot(
        &mut self,
        block_hash: BlockHash,
        pool: PoolRef,
    ) -> Option<PoolState> {
        self.0.get_mut(&block_hash)?.pool_snapshots.remove(&pool)
    }

    /// Inserts a block header and returns the first missing parent, if the ancestry is incomplete.
    /// Added to make header ingestion validate duplicate, conflicting, self-parent, and cyclic ancestry before state is accepted.
    fn with_new_block(
        self,
        hash: BlockHash,
        parent_hash: BlockHash,
        logs_bloom: Bloom,
        finalized_hash: BlockHash,
    ) -> Result<(BlocksGraph, Option<BlockHash>), NewBlockError> {
        if hash == parent_hash {
            return Err(NewBlockError::SelfParentBlock(hash, self));
        }

        if hash == finalized_hash {
            return Err(NewBlockError::ExistingBlock(self));
        }

        if let Some(block) = self.get(&hash) {
            if block.parent_hash == parent_hash {
                return Err(NewBlockError::ExistingBlock(self));
            }

            return Err(NewBlockError::ConflictingBlockParent);
        }

        let BlocksGraph(mut blocks) = self;

        blocks.insert(
            hash,
            BlockNode {
                parent_hash,
                logs_bloom: Some(logs_bloom),
                pool_logs: PoolLogsStatus::Unknown,
                pool_snapshots: HashMap::new(),
                pool_data_failures: HashMap::new(),
            },
        );

        match find_missing_block_hash(&blocks, hash, finalized_hash) {
            Ok(None) => Ok((BlocksGraph(blocks), None)),
            Ok(Some(missing_hash)) => Ok((BlocksGraph(blocks), Some(missing_hash))),
            Err(BlocksGraphCycleError) => Err(NewBlockError::CycleDetected),
        }
    }

    /// Sets a block's pool-logs status (`Partial` for provisional subscription logs, `Resolved` for
    /// authoritative `GetBlockLogs`). No-ops if the block is absent.
    fn with_pool_logs(self, block_hash: BlockHash, status: PoolLogsStatus) -> BlocksGraph {
        let BlocksGraph(mut blocks) = self;

        if let Some(block) = blocks.get_mut(&block_hash) {
            block.pool_logs = status;
        }

        BlocksGraph(blocks)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    /// Projects a block's raw candidate logs into trusted pool logs using the current registry.
    /// Added so validation state stays in `TrustedPoolRegistry` and block state does not duplicate trusted/rejected facts.
    fn trusted_pool_logs(
        &self,
        chain: ChainKey,
        block_hash: BlockHash,
        registry: &TrustedPoolRegistry,
    ) -> Option<TrustedPoolLogs> {
        self.get(&block_hash)
            .map(|block| match block.pool_logs.candidates() {
                None => TrustedPoolLogs::Unknown,
                Some(candidates) => registry.trusted_pool_logs(chain, candidates),
            })
    }

    /// Returns the present canonical suffix from oldest known block to tip.
    /// Added for schedulers that need chronological context while safely stopping at missing parents or cycles.
    fn present_canonical_hashes_oldest_to_newest(
        &self,
        tip_hash: BlockHash,
        finalized_hash: BlockHash,
    ) -> Vec<BlockHash> {
        let mut current_hash = tip_hash;
        let mut hashes = Vec::new();
        let mut visited = HashSet::new();

        while current_hash != finalized_hash {
            if !visited.insert(current_hash) {
                return Vec::new();
            }

            let Some(block) = self.get(&current_hash) else {
                break;
            };

            hashes.push(current_hash);
            current_hash = block.parent_hash;
        }

        hashes.reverse();
        hashes
    }

    /// Returns a complete parent path from finalized exclusive to the requested block.
    /// Added for pure queries that must reject disconnected or cyclic ancestry instead of using a partial prefix.
    fn connected_path_hashes_oldest_to_newest(
        &self,
        start_hash: BlockHash,
        finalized_hash: BlockHash,
    ) -> Option<Vec<BlockHash>> {
        let mut current_hash = start_hash;
        let mut hashes = Vec::new();
        let mut visited = HashSet::new();

        while current_hash != finalized_hash {
            if !visited.insert(current_hash) {
                return None;
            }

            let block = self.get(&current_hash)?;

            hashes.push(current_hash);
            current_hash = block.parent_hash;
        }

        hashes.reverse();
        Some(hashes)
    }

    /// Checks whether a block is on a fully connected canonical path.
    /// Added so finalized observations can only compact the branch currently selected by the kernel.
    fn connected_path_contains(
        &self,
        tip_hash: BlockHash,
        target_hash: BlockHash,
        finalized_hash: BlockHash,
    ) -> bool {
        target_hash == finalized_hash
            || self
                .connected_path_hashes_oldest_to_newest(tip_hash, finalized_hash)
                .is_some_and(|hashes| hashes.contains(&target_hash))
    }

    /// Retains only recent blocks that descend from the new finalized boundary.
    /// Added so finalized snapshot compaction can drop old canonical blocks and side branches together.
    fn retaining_descendants_of(self, finalized_hash: BlockHash) -> BlocksGraph {
        let BlocksGraph(blocks) = self;
        let retained_hashes = blocks
            .keys()
            .copied()
            .filter(|hash| block_descends_from(&blocks, *hash, finalized_hash))
            .collect::<HashSet<_>>();

        BlocksGraph(
            blocks
                .into_iter()
                .filter(|(hash, _)| retained_hashes.contains(hash))
                .collect(),
        )
    }

    /// Returns the hashes still present in recent block storage.
    /// Added so pending block-scoped requests can be pruned after block compaction.
    fn hashes(&self) -> HashSet<BlockHash> {
        self.0.keys().copied().collect()
    }

    /// Applies requested pool-state results to a target block snapshot.
    /// Added to ensure stale, missing-block, or overbroad RPC responses cannot write unrequested pool state.
    fn with_pool_data(
        self,
        block_hash: BlockHash,
        requested_pools: HashSet<PoolRef>,
        pool_results: HashMap<PoolRef, PoolDataResult>,
    ) -> BlocksGraph {
        let BlocksGraph(mut blocks) = self;

        if let Some(block) = blocks.get_mut(&block_hash) {
            for pool in requested_pools {
                let Some(pool_result) = pool_results.get(&pool) else {
                    continue;
                };

                match pool_result {
                    Ok(pool_state) => {
                        block.pool_snapshots.insert(pool, pool_state.clone());
                        block.pool_data_failures.remove(&pool);
                    }
                    Err(pool_failure) if !block.pool_snapshots.contains_key(&pool) => {
                        block.pool_data_failures.insert(pool, pool_failure.clone());
                    }
                    Err(_) => {}
                }
            }
        }

        BlocksGraph(blocks)
    }

    /// Records log-derived pool snapshots onto a block.
    ///
    /// Unlike [`BlocksGraph::with_pool_data`] there is no failure map: derivation either produces a
    /// snapshot (stored here) or it does not (the pool is simply left uncovered, so the existing
    /// `GetPoolData` path picks it up). A derived snapshot is authoritative for its block; it
    /// overwrites any prior snapshot for the same pool on that block.
    fn with_derived_pool_state(
        self,
        block_hash: BlockHash,
        derived: HashMap<PoolRef, PoolState>,
    ) -> BlocksGraph {
        let BlocksGraph(mut blocks) = self;

        if let Some(block) = blocks.get_mut(&block_hash) {
            for (pool, pool_state) in derived {
                block.pool_snapshots.insert(pool, pool_state);
            }
        }

        BlocksGraph(blocks)
    }

    /// Builds the next pool-state request (issued at the tip) for verified dirty pools on the
    /// canonical path that are not already covered.
    ///
    /// Semantics: a trusted pool is "dirty" at a block when that block's resolved logs name it, and
    /// "covered" at a block when that block holds its snapshot or failure marker, or an in-flight
    /// `GetPoolData` request targets that block. Walking the present canonical suffix oldest→newest, a
    /// pool needs a read iff its latest dirty block is newer than its latest covered block — i.e. its
    /// state changed after the last time we read, attempted, or are reading it. Coverage at *any* suffix
    /// block (not only the tip) suppresses the read, so a snapshot taken at an earlier block stops being
    /// re-requested every head until something re-dirties the pool; re-dirtying at a newer block
    /// re-adds it. The result is read once per change instead of once per block.
    ///
    /// CAVEAT (known follow-up): a `pool_data_failures` entry counts as coverage, so a *transient* fetch
    /// error filters the pool out of `dirty_pools` until it is re-dirtied. Between the failure and the
    /// next log that names the pool, we hold no fresh snapshot for it and could optimize over an
    /// outdated/absent state. This is acceptable only as a stopgap; a bounded retry/backoff for transient
    /// per-pool failures should replace this so stale state cannot persist silently.
    fn unknown_present_canonical_pool_data_request(
        &self,
        chain: ChainKey,
        tip_hash: BlockHash,
        finalized_hash: BlockHash,
        registry: &TrustedPoolRegistry,
        pending_pool_data_by_block: &HashMap<BlockHash, HashSet<PoolRef>>,
    ) -> Option<(BlockHash, HashSet<PoolRef>)> {
        let hashes = self.present_canonical_hashes_oldest_to_newest(tip_hash, finalized_hash);
        let target_hash = hashes.last().copied()?;
        let mut dirty_pools = HashSet::new();

        for block_hash in hashes {
            let Some(block) = self.get(&block_hash) else {
                continue;
            };

            if let Some(candidates) = block.pool_logs.candidates() {
                if let TrustedPoolLogs::Resolved(pools) =
                    registry.trusted_pool_logs(chain, candidates)
                {
                    dirty_pools.extend(pools);
                }
            }

            // Oldest→newest, so the latest touch wins: coverage on this block clears the pool, a
            // newer dirty block re-adds it. This skips reads for pools already known/failed/pending
            // at a block at-or-after their last change, not just at the tip.
            for pool in block.pool_snapshots.keys() {
                dirty_pools.remove(pool);
            }
            for pool in block.pool_data_failures.keys() {
                dirty_pools.remove(pool);
            }
            if let Some(pending_pools) = pending_pool_data_by_block.get(&block_hash) {
                for pool in pending_pools {
                    dirty_pools.remove(pool);
                }
            }
        }

        (!dirty_pools.is_empty()).then_some((target_hash, dirty_pools))
    }

    /// Up to `limit` verified pools (lowest address first) with no snapshot or failure anywhere
    /// reachable — neither in the finalized snapshot nor on the present canonical suffix. These are the
    /// never-covered pools the dirty-pool path never asks for because they have not traded; the
    /// background backfill snapshots them so the frontier broadens over time. Address-sorted for stable,
    /// resumable chunking: each fetched chunk becomes covered and drops out of the next selection, so
    /// the sweep advances without a stored cursor.
    fn uncovered_verified_pool_chunk(
        &self,
        chain: ChainKey,
        tip_hash: BlockHash,
        finalized_hash: BlockHash,
        finalized_snapshots: &HashMap<PoolRef, PoolState>,
        registry: &TrustedPoolRegistry,
        limit: usize,
    ) -> HashSet<PoolRef> {
        let mut covered: HashSet<PoolRef> = finalized_snapshots.keys().copied().collect();
        for block_hash in self.present_canonical_hashes_oldest_to_newest(tip_hash, finalized_hash) {
            if let Some(block) = self.get(&block_hash) {
                covered.extend(block.pool_snapshots.keys().copied());
                covered.extend(block.pool_data_failures.keys().copied());
            }
        }

        let mut uncovered = registry
            .verified_addresses(chain)
            .into_iter()
            .map(|address| PoolRef::uniswap_v3(address, chain))
            .filter(|pool| !covered.contains(pool))
            .collect::<Vec<_>>();
        uncovered.sort();
        uncovered.into_iter().take(limit).collect()
    }

    /// Finds present canonical blocks whose logs are still unknown and not pending.
    /// Added so every connected canonical block eventually gets log data without duplicating in-flight requests.
    fn unknown_present_canonical_log_hashes(
        &self,
        tip_hash: BlockHash,
        finalized_hash: BlockHash,
        pending_log_hashes: &HashSet<BlockHash>,
    ) -> Vec<BlockHash> {
        let mut current_hash = tip_hash;
        let mut hashes = Vec::new();
        let mut visited = HashSet::new();

        while current_hash != finalized_hash {
            if !visited.insert(current_hash) {
                return Vec::new();
            }

            let Some(block) = self.get(&current_hash) else {
                break;
            };

            // `Partial` logs are provisional, so they still need an authoritative `GetBlockLogs`.
            if !matches!(block.pool_logs, PoolLogsStatus::Resolved(_))
                && !pending_log_hashes.contains(&current_hash)
            {
                hashes.push(current_hash);
            }

            current_hash = block.parent_hash;
        }

        hashes
    }

    /// Groups unknown canonical log candidates into metadata validation requests.
    /// Added to prevent arbitrary contracts that emit matching topics from being treated as real Uniswap pools.
    fn unknown_present_canonical_pool_metadata_requests(
        &self,
        chain: ChainKey,
        tip_hash: BlockHash,
        finalized_hash: BlockHash,
        registry: &TrustedPoolRegistry,
        pending_candidates: &HashSet<ProtocolPoolKey>,
    ) -> Vec<(BlockHash, HashSet<ProtocolPoolKey>)> {
        let mut current_hash = tip_hash;
        let mut requests = Vec::new();
        let mut visited = HashSet::new();
        let mut unavailable_candidates = pending_candidates.clone();

        while current_hash != finalized_hash {
            if !visited.insert(current_hash) {
                return Vec::new();
            }

            let Some(block) = self.get(&current_hash) else {
                break;
            };

            if let Some(candidates) = block.pool_logs.candidates() {
                let request_candidates = candidates
                    .iter()
                    .copied()
                    .filter(|candidate| {
                        !registry.is_known(chain, *candidate)
                            && !unavailable_candidates.contains(candidate)
                    })
                    .collect::<HashSet<_>>();

                if !request_candidates.is_empty() {
                    unavailable_candidates.extend(request_candidates.iter().copied());
                    requests.push((current_hash, request_candidates));
                }
            }

            current_hash = block.parent_hash;
        }

        requests
    }

    /// Groups verified-pool tokens whose metadata is not known into token metadata requests.
    /// Added so later reserve projection can scale exact on-chain amounts by trusted token decimals.
    fn unknown_present_canonical_token_metadata_requests(
        &self,
        chain: ChainKey,
        tip_hash: BlockHash,
        finalized_hash: BlockHash,
        pool_registry: &TrustedPoolRegistry,
        token_registry: &TokenRegistry,
        pending_tokens: &HashSet<TokenAddress>,
    ) -> Vec<(BlockHash, HashSet<TokenAddress>)> {
        let mut current_hash = tip_hash;
        let mut requests = Vec::new();
        let mut visited = HashSet::new();
        let mut unavailable_tokens = pending_tokens.clone();

        while current_hash != finalized_hash {
            if !visited.insert(current_hash) {
                return Vec::new();
            }

            let Some(block) = self.get(&current_hash) else {
                break;
            };

            if let Some(candidates) = block.pool_logs.candidates() {
                let tokens = candidates
                    .iter()
                    .filter_map(|candidate| pool_registry.verified_pool(chain, *candidate))
                    .filter_map(|pool| pool_registry.verified_metadata(pool))
                    .flat_map(|metadata| {
                        [
                            TokenAddress(metadata.token0, chain),
                            TokenAddress(metadata.token1, chain),
                        ]
                    })
                    .filter(|token| {
                        !token_registry.is_known(*token) && !unavailable_tokens.contains(token)
                    })
                    .collect::<HashSet<_>>();

                if !tokens.is_empty() {
                    unavailable_tokens.extend(tokens.iter().copied());
                    requests.push((current_hash, tokens));
                }
            }

            current_hash = block.parent_hash;
        }

        requests
    }
}

pub struct FinalizedState {
    pub block_hash: BlockHash,
    pool_snapshots: HashMap<PoolRef, PoolState>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompletePoolStateUpdate {
    pub block_hash: BlockHash,
    /// For each affected pool, the canonical block holding its latest snapshot at-or-before
    /// `block_hash`. Resolve into pool states via `State::resolve_complete_pool_states`; valid only
    /// against the block graph it was computed from (consume before any block mutation).
    pub pool_snapshot_blocks: HashMap<PoolRef, BlockHash>,
}

impl FinalizedState {
    /// Creates a finalized snapshot with only its block hash and no pool snapshots.
    /// Added as the bootstrap path before finalized pool-state persistence exists.
    pub fn empty_at(block_hash: BlockHash) -> FinalizedState {
        FinalizedState {
            block_hash,
            pool_snapshots: HashMap::new(),
        }
    }

    #[cfg(test)]
    /// Creates a finalized snapshot with explicit pool snapshots for projection tests.
    /// Added so tests can exercise read-only projections without driving unrelated header/log scheduling.
    pub(crate) fn with_pool_snapshots_for_test(
        block_hash: BlockHash,
        pool_snapshots: HashMap<PoolRef, PoolState>,
    ) -> FinalizedState {
        FinalizedState {
            block_hash,
            pool_snapshots,
        }
    }
}

pub struct State {
    blocks: BlocksGraph,
    canonical_tip: BlockHash,
    pending_requests: PendingRequests,
    finalized_state: FinalizedState,
    pool_registry: TrustedPoolRegistry,
    token_registry: TokenRegistry,
    tick: Tick,
    /// Subscription-observed logs for blocks not yet in the graph, keyed by block hash. Drained
    /// into the block when it enters via a head/header observation. Bounded by
    /// [`MAX_STREAMED_LOG_BLOCKS`]; raw input staging only, safe to drop.
    streamed_logs: HashMap<BlockHash, Vec<PoolLog>>,
}

/// Caps how many not-yet-known blocks can hold buffered subscription logs at once, bounding the
/// staging map when observed logs for a block arrive but its head never does (e.g. a reorg).
const MAX_STREAMED_LOG_BLOCKS: usize = 1024;

impl State {
    /// Creates kernel state from a finalized snapshot with no pending requests or recent blocks.
    /// Added as the pure state-machine entry point for runtimes that will feed events and execute effects.
    pub fn init(finalized_state: FinalizedState) -> State {
        State {
            blocks: BlocksGraph::new(),
            canonical_tip: finalized_state.block_hash,
            pending_requests: PendingRequests::new(),
            finalized_state,
            pool_registry: TrustedPoolRegistry::new(),
            token_registry: TokenRegistry::new(),
            tick: Tick::initial(),
            streamed_logs: HashMap::new(),
        }
    }

    /// Activates kernel state from a bootstrap seed: a finalized anchor with its pool snapshots,
    /// validated pool/token registries, and recent canonical blocks whose pool logs are already resolved.
    /// Added so a chain starts warm from the bootstrap outcome instead of warming up from an empty finalized state.
    pub fn activate_from_seed(
        finalized_hash: BlockHash,
        finalized_pool_snapshots: HashMap<PoolRef, PoolState>,
        pool_registry: TrustedPoolRegistry,
        token_registry: TokenRegistry,
        seed_blocks: Vec<(BlockHash, BlockHash, HashSet<ProtocolPoolKey>)>,
    ) -> (State, Vec<Effect>) {
        let blocks = seed_blocks
            .into_iter()
            .map(|(hash, parent_hash, candidates)| {
                (
                    hash,
                    BlockNode {
                        parent_hash,
                        logs_bloom: None,
                        pool_logs: PoolLogsStatus::Resolved(candidates),
                        pool_snapshots: HashMap::new(),
                        pool_data_failures: HashMap::new(),
                    },
                )
            })
            .collect();
        let blocks = BlocksGraph(blocks);
        let tick = Tick::initial();

        // Seed blocks can reference ancestors that were not themselves seeded: a no-log block is
        // absent from the candidate window, so the segment above it floats. Request a header for
        // each distinct missing ancestor so ancestry reconstruction reconnects every seeded block
        // down to the finalized anchor, the same way `HeadObserved` does for one gap at a time.
        let (pending_requests, effects) = missing_seed_parents(&blocks, finalized_hash)
            .into_iter()
            .fold(
                (PendingRequests::new(), Vec::new()),
                |(pending_requests, mut effects), missing_hash| {
                    let request_payload = GetBlockHeader {
                        block_hash: missing_hash,
                    };
                    let (pending_requests, request_id) =
                        pending_requests.with_new_request(request_payload.clone(), tick);
                    effects.push(Effect::Request(AnyIssuedRequest::BlockHeader(
                        IssuedRequest {
                            request_id,
                            request_payload,
                        },
                    )));
                    (pending_requests, effects)
                },
            );

        let state = State {
            blocks,
            canonical_tip: finalized_hash,
            pending_requests,
            finalized_state: FinalizedState {
                block_hash: finalized_hash,
                pool_snapshots: finalized_pool_snapshots,
            },
            pool_registry,
            token_registry,
            tick,
            streamed_logs: HashMap::new(),
        };

        (state, effects)
    }

    /// Exposes finalized pool snapshots to pure read models.
    /// Added so multi-chain projections can merge finalized state with a complete recent-block overlay without mutating kernel state.
    pub(crate) fn finalized_pool_snapshots(&self) -> &HashMap<PoolRef, PoolState> {
        &self.finalized_state.pool_snapshots
    }

    /// Looks up verified pool metadata without exposing registry internals.
    /// Added so projections can refuse incomplete data while keeping validation ownership inside the kernel registry.
    pub(crate) fn verified_pool_metadata(&self, pool: PoolRef) -> Option<&PoolMetadata> {
        self.pool_registry.verified_metadata(pool)
    }

    /// Looks up verified token metadata without exposing registry internals.
    /// Added so projections can scale raw on-chain amounts only after token decimals have been validated.
    pub(crate) fn verified_token_metadata(&self, token: TokenAddress) -> Option<&TokenMetadata> {
        self.token_registry.verified_metadata(token)
    }

    /// Counts pools the registry has verified.
    /// Added so read models can surface tracked-pool progress without reaching into the registry.
    pub(crate) fn verified_pool_count(&self) -> usize {
        self.pool_registry.verified_size()
    }

    /// Counts RPC requests currently in flight: dispatched but not yet answered.
    /// Added so read models can surface per-chain fetch backlog without reaching into the request store.
    pub(crate) fn in_flight_request_count(&self) -> usize {
        self.pending_requests.len()
    }

    /// Latest complete pool-state overlay anchored at the current canonical tip.
    /// Added so optimization dispatch can read the tip's fully-fetched pool state without exposing `canonical_tip`.
    pub(crate) fn latest_complete_pool_state_update(
        &self,
        chain: ChainKey,
    ) -> Option<CompletePoolStateUpdate> {
        self.latest_complete_pool_state_update_from(chain, self.canonical_tip)
    }

    /// Counts canonical blocks the tip is ahead of `reference_hash` on a connected path.
    /// Added so read models can measure fetch progress from an already-known frontier block
    /// (such as the last dispatched optimization block) without rebuilding the complete
    /// pool-state overlay; `None` mirrors a reference that is off the tip's connected path.
    pub(crate) fn blocks_behind(&self, reference_hash: BlockHash) -> Option<usize> {
        self.blocks
            .connected_path_hashes_oldest_to_newest(self.canonical_tip, reference_hash)
            .map(|hashes| hashes.len())
    }

    #[cfg(test)]
    /// Builds kernel state from projection-relevant parts for tests.
    /// Added to keep projection tests focused on pure reserve generation instead of replaying unrelated RPC scheduling events.
    pub(crate) fn for_pool_reserve_projection_test(
        finalized_state: FinalizedState,
        pool_registry: TrustedPoolRegistry,
        token_registry: TokenRegistry,
    ) -> State {
        State {
            blocks: BlocksGraph::new(),
            canonical_tip: finalized_state.block_hash,
            pending_requests: PendingRequests::new(),
            finalized_state,
            pool_registry,
            token_registry,
            tick: Tick::initial(),
            streamed_logs: HashMap::new(),
        }
    }

    #[cfg(test)]
    /// Seeds a block carrying the given pool snapshots so projection tests can resolve overlay
    /// descriptors (`pool_snapshot_blocks`) that point at it.
    pub(crate) fn with_overlay_block_for_test(
        mut self,
        block_hash: BlockHash,
        pool_snapshots: HashMap<PoolRef, PoolState>,
    ) -> State {
        self.blocks.0.insert(
            block_hash,
            BlockNode {
                parent_hash: self.finalized_state.block_hash,
                logs_bloom: None,
                pool_logs: PoolLogsStatus::Resolved(HashSet::new()),
                pool_snapshots,
                pool_data_failures: HashMap::new(),
            },
        );
        self
    }

    /// Rebuilds volatile chain state around a finalized anchor while preserving registries and tick.
    /// Added for reorg/inconsistency recovery where recent blocks are unsafe but immutable pool/token facts remain valid.
    fn reset(
        finalized_state: FinalizedState,
        tick: Tick,
        pool_registry: TrustedPoolRegistry,
        token_registry: TokenRegistry,
    ) -> State {
        State {
            blocks: BlocksGraph::new(),
            canonical_tip: finalized_state.block_hash,
            pending_requests: PendingRequests::new(),
            finalized_state,
            pool_registry,
            token_registry,
            tick,
            streamed_logs: HashMap::new(),
        }
    }

    /// Measures the connected canonical path length from the finalized anchor to the current tip.
    /// Added so wrappers can trigger finalized-header refreshes from graph distance without inspecting graph internals.
    pub(crate) fn canonical_path_len_from_finalized(&self) -> Option<usize> {
        self.blocks
            .connected_path_hashes_oldest_to_newest(
                self.canonical_tip,
                self.finalized_state.block_hash,
            )
            .map(|hashes| hashes.len())
    }

    /// Finds the latest complete pool-state overlay on a connected path from an arbitrary block.
    /// Added so optimization and compaction can share one readiness query without triggering effects.
    pub fn latest_complete_pool_state_update_from(
        &self,
        chain: ChainKey,
        start_block_hash: BlockHash,
    ) -> Option<CompletePoolStateUpdate> {
        let hashes = self.blocks.connected_path_hashes_oldest_to_newest(
            start_block_hash,
            self.finalized_state.block_hash,
        )?;

        // Single forward pass: accumulate snapshot *locations* (cheap `BlockHash`es) and promote a
        // block to "complete" by flushing the pending buffer into `committed` only when every
        // affected pool has a snapshot. The committed map is returned as the overlay descriptor;
        // callers resolve it into pool states lazily, so no `PoolState` is cloned here.
        let mut affected_pools: HashSet<PoolRef> = HashSet::new();
        let mut invalid_pools: HashSet<PoolRef> = HashSet::new();
        let mut committed: HashMap<PoolRef, BlockHash> = HashMap::new();
        let mut pending: HashMap<PoolRef, BlockHash> = HashMap::new();
        let mut latest_complete_hash = self.finalized_state.block_hash;

        for block_hash in hashes {
            // A path hash with no backing block means the ancestry broke under us, so the overlay
            // is unknown rather than "complete so far": fail the whole query.
            let block = self.blocks.get(&block_hash)?;

            // Unresolved logs or pending candidate validation stop the scan; keep the latest
            // complete overlay found before this block.
            let Some(trusted_pools) = self.trusted_pools_for_complete_pool_state_scan(chain, block)
            else {
                break;
            };

            affected_pools.extend(trusted_pools.iter().copied());
            invalid_pools.extend(trusted_pools);

            for pool in block.pool_snapshots.keys() {
                if affected_pools.contains(pool) {
                    invalid_pools.remove(pool);
                    pending.insert(*pool, block_hash);
                }
            }

            if invalid_pools.is_empty() {
                // Flush is O(pending), not O(committed): each snapshot update flushes at most once.
                committed.extend(pending.drain());
                latest_complete_hash = block_hash;
            }
        }

        // Pending entries left past the latest complete block are the incomplete tail and are
        // dropped. `committed` is the overlay descriptor (pool -> snapshot location at or before
        // the frontier); resolution is deferred to the caller.
        Some(CompletePoolStateUpdate {
            block_hash: latest_complete_hash,
            pool_snapshot_blocks: committed,
        })
    }

    /// Resolves an overlay descriptor into borrowed pool states against the current block graph.
    /// Returns `None` if any location is missing: that is a broken invariant, so the whole overlay
    /// is untrustworthy and callers must not act on a partial result.
    pub(crate) fn resolve_complete_pool_states<'a>(
        &'a self,
        update: &CompletePoolStateUpdate,
    ) -> Option<HashMap<PoolRef, &'a PoolState>> {
        update
            .pool_snapshot_blocks
            .iter()
            .map(|(pool, block_hash)| {
                let pool_state = self.blocks.get(block_hash)?.pool_snapshots.get(pool)?;
                Some((*pool, pool_state))
            })
            .collect()
    }

    /// Resolves a block's pool logs into trusted pools for readiness scanning.
    /// Added so unresolved logs or pending candidate validation stop the scan without losing the latest prior complete overlay.
    fn trusted_pools_for_complete_pool_state_scan(
        &self,
        chain: ChainKey,
        block: &BlockNode,
    ) -> Option<HashSet<PoolRef>> {
        match &block.pool_logs {
            // `Partial` is provisional: it must not contribute to a "complete" overlay, so it stops
            // the scan exactly like `Unknown`. This is the single gate that keeps the optimizer and
            // finalization acting only on authoritative `Resolved` state.
            PoolLogsStatus::Unknown | PoolLogsStatus::Partial(_) => None,
            PoolLogsStatus::Resolved(candidates) => {
                match self.pool_registry.trusted_pool_logs(chain, candidates) {
                    TrustedPoolLogs::Resolved(pools) => Some(pools),
                    TrustedPoolLogs::Unknown | TrustedPoolLogs::PendingValidation => None,
                }
            }
        }
    }

    /// Records an authoritative block-logs response: resolves the block's candidate set, derives a
    /// snapshot for every trusted pool the logs can be folded forward for, and stores both.
    ///
    /// Derivation is best-effort: a pool whose base is uncertain or whose fold yields nothing is
    /// left unsnapshotted, so the existing dirty/`GetPoolData` machinery covers it unchanged. No
    /// scheduler change is needed — a derived snapshot simply keeps the pool out of the dirty set.
    fn with_block_logs_applied(
        self,
        chain: ChainKey,
        block_hash: BlockHash,
        logs: Vec<PoolLog>,
        complete: bool,
    ) -> State {
        let Some(block) = self.blocks.get(&block_hash) else {
            // The target block was pruned/reset between request and response; drop the logs.
            return self;
        };
        let parent_hash = block.parent_hash;

        // A provisional (subscription) update never downgrades or re-derives an already
        // authoritative block: the `Resolved` snapshots stand.
        if !complete && matches!(block.pool_logs, PoolLogsStatus::Resolved(_)) {
            return self;
        }

        // `log.pool` is the protocol-tagged identity (v3 contract address or v4 PoolId); both
        // protocols become candidates here.
        let observed = logs.iter().map(|log| log.pool).collect::<HashSet<_>>();
        let status = if complete {
            PoolLogsStatus::Resolved(observed)
        } else if let PoolLogsStatus::Partial(existing) = &block.pool_logs {
            // Subscription logs arrive in fragments; accumulate the provisional candidate set.
            PoolLogsStatus::Partial(existing.union(&observed).copied().collect())
        } else {
            PoolLogsStatus::Partial(observed)
        };

        // Group this block's logs by the trusted pool that emitted them, preserving intra-block
        // order via `log_index`.
        let mut events_by_pool: HashMap<PoolRef, Vec<(u64, &PoolLogEvent)>> = HashMap::new();
        for log in &logs {
            if let Some(pool) = self.pool_registry.verified_pool(chain, log.pool) {
                events_by_pool
                    .entry(pool)
                    .or_default()
                    .push((log.log_index, &log.event));
            }
        }

        let mut derived = HashMap::new();
        for (pool, mut indexed_events) in events_by_pool {
            // An uncertain base resolves to `None`: a block whose logs contain an absolute event
            // (Swap/Initialize) still derives from that event alone, while a delta-only block with
            // no base derives to `None` and falls back to `GetPoolData` exactly as before.
            let base = self.resolve_pool_base(chain, parent_hash, pool);

            indexed_events.sort_by_key(|(log_index, _)| *log_index);
            let ordered = indexed_events
                .into_iter()
                .map(|(_, event)| event)
                .collect::<Vec<_>>();

            if let Some(pool_state) = derive_pool_state(base.as_ref(), &ordered) {
                derived.insert(pool, pool_state);
            }
        }

        let blocks = self
            .blocks
            .with_pool_logs(block_hash, status)
            .with_derived_pool_state(block_hash, derived);

        State { blocks, ..self }
    }

    /// Stages subscription logs for a block not yet in the graph. Bounded by
    /// [`MAX_STREAMED_LOG_BLOCKS`]: once full, logs for further new blocks are dropped (the
    /// authoritative `GetBlockLogs` still covers them once the block arrives).
    fn with_streamed_logs_buffered(mut self, block_hash: BlockHash, logs: Vec<PoolLog>) -> State {
        if self.streamed_logs.len() >= MAX_STREAMED_LOG_BLOCKS
            && !self.streamed_logs.contains_key(&block_hash)
        {
            return self;
        }

        self.streamed_logs
            .entry(block_hash)
            .or_default()
            .extend(logs);
        self
    }

    /// Applies and clears any staged subscription logs for a block that has just entered the graph.
    fn with_streamed_logs_drained(mut self, chain: ChainKey, block_hash: BlockHash) -> State {
        match self.streamed_logs.remove(&block_hash) {
            Some(logs) => self.with_block_logs_applied(chain, block_hash, logs, false),
            None => self,
        }
    }

    /// Resolves a pool's snapshot as of `parent_hash` for forward derivation by walking the
    /// canonical ancestry newest→oldest. The first ancestor that changed the pool fixes the base to
    /// its snapshot; if no ancestor changed the pool, the base is the finalized snapshot. Returns
    /// `None` when the base is *uncertain* — an ancestor could have changed the pool without a
    /// recorded snapshot (its logs are not yet resolved, or its change was not snapshotted). A
    /// `None` base is still derivable when the block itself carries an absolute event; otherwise the
    /// caller leaves the pool for the `GetPoolData` fallback.
    fn resolve_pool_base(
        &self,
        chain: ChainKey,
        parent_hash: BlockHash,
        pool: PoolRef,
    ) -> Option<PoolState> {
        let path = self
            .blocks
            .connected_path_hashes_oldest_to_newest(parent_hash, self.finalized_state.block_hash)?;

        for block_hash in path.iter().rev() {
            let block = self.blocks.get(block_hash)?;
            let trusted_pools = self.trusted_pools_for_complete_pool_state_scan(chain, block)?;

            if trusted_pools.contains(&pool) {
                return block.pool_snapshots.get(&pool).cloned();
            }
        }

        self.finalized_state.pool_snapshots.get(&pool).cloned()
    }

    /// Attempts to compact recent block data into the finalized snapshot.
    /// Added to bound block graph growth without remembering failed finalized observations.
    fn with_finalized_block_observed(self, chain: ChainKey, block_hash: BlockHash) -> State {
        if block_hash == self.finalized_state.block_hash {
            return self;
        }

        if !self.blocks.connected_path_contains(
            self.canonical_tip,
            block_hash,
            self.finalized_state.block_hash,
        ) {
            return self;
        }

        let Some(update) = self.latest_complete_pool_state_update_from(chain, block_hash) else {
            return self;
        };

        if update.block_hash == self.finalized_state.block_hash {
            return self;
        }

        // Validate every overlay location resolves before mutating: a missing snapshot is a broken
        // invariant, so abort compaction (return self unchanged) rather than persist a partial,
        // untrustworthy finalized snapshot.
        if self.resolve_complete_pool_states(&update).is_none() {
            return self;
        }

        let State {
            mut blocks,
            canonical_tip,
            pending_requests,
            finalized_state,
            pool_registry,
            token_registry,
            tick,
            streamed_logs: _,
        } = self;

        // Move each pool's latest snapshot out of its block instead of cloning. Every referenced
        // block is at or before the frontier, so `retaining_descendants_of` below prunes them all
        // and no retained block is mutated; presence is guaranteed by the validation above.
        let mut pool_snapshots = finalized_state.pool_snapshots;
        for (pool, snapshot_hash) in &update.pool_snapshot_blocks {
            if let Some(pool_state) = blocks.take_pool_snapshot(*snapshot_hash, *pool) {
                pool_snapshots.insert(*pool, pool_state);
            }
        }

        let finalized_state = FinalizedState {
            block_hash: update.block_hash,
            pool_snapshots,
        };
        let blocks = blocks.retaining_descendants_of(finalized_state.block_hash);
        let retained_blocks = blocks.hashes();
        let canonical_tip = if retained_blocks.contains(&canonical_tip) {
            canonical_tip
        } else {
            finalized_state.block_hash
        };
        let pending_requests = pending_requests.retaining_block_targets(&retained_blocks);

        State {
            blocks,
            canonical_tip,
            pending_requests,
            finalized_state,
            pool_registry,
            token_registry,
            tick,
            // Finalization evicts the staging buffer: any logs whose head has not arrived by now
            // are almost certainly orphaned, and a real future block re-fetches authoritatively.
            streamed_logs: HashMap::new(),
        }
    }
}

/// Decides whether a canonical path length transition should refresh the finalized header.
/// Added to make finalized refresh edge-triggered instead of repeatedly firing for every event above target.
pub(crate) fn should_fetch_finalized_header(
    before_len: Option<usize>,
    after_len: Option<usize>,
    target_len: usize,
    retry_stride: usize,
) -> bool {
    let Some(after_len) = after_len else {
        return false;
    };

    if after_len < target_len {
        return false;
    }

    match before_len {
        None => true,
        Some(before_len) if before_len < target_len => true,
        Some(before_len) => {
            finalized_refresh_bucket(before_len, target_len, retry_stride)
                < finalized_refresh_bucket(after_len, target_len, retry_stride)
        }
    }
}

fn finalized_refresh_bucket(len: usize, target_len: usize, retry_stride: usize) -> usize {
    if retry_stride == 0 {
        return 0;
    }

    len.saturating_sub(target_len) / retry_stride
}

pub enum Event {
    HeadObserved {
        hash: BlockHash,
        parent_hash: BlockHash,
        logs_bloom: Bloom,
    },
    FinalizedBlockObserved {
        block_hash: BlockHash,
    },
    BlockHeaderReceived {
        request_id: RequestId<GetBlockHeader>,
        hash: BlockHash,
        parent_hash: BlockHash,
        logs_bloom: Bloom,
    },
    BlockHeaderNotFound {
        request_id: RequestId<GetBlockHeader>,
    },
    BlockLogsReceived {
        request_id: RequestId<GetBlockLogs>,
        logs: Vec<PoolLog>,
    },
    /// Best-effort logs from the live subscription: provisional (`Partial`), not authoritative.
    LogObserved {
        block_hash: BlockHash,
        logs: Vec<PoolLog>,
    },
    PoolMetadataReceived {
        request_id: RequestId<GetPoolMetadata>,
        metadata: HashMap<ProtocolPoolKey, PoolMetadataResult>,
    },
    TokenMetadataReceived {
        request_id: RequestId<GetTokenMetadata>,
        metadata: HashMap<TokenAddress, TokenMetadataResult>,
    },
    PoolDataReceived {
        request_id: RequestId<GetPoolData>,
        pools: HashMap<PoolRef, PoolDataResult>,
    },
    RequestFailed {
        request_id: AnyRequestId,
    },
    Tick,
}

pub enum Effect {
    Request(AnyIssuedRequest),
}

struct BlocksGraphCycleError;

/// Checks whether a recent block's parent walk reaches a finalized boundary.
/// Added so compaction can retain descendants while removing the finalized block itself.
fn block_descends_from(
    blocks: &HashMap<BlockHash, BlockNode>,
    block_hash: BlockHash,
    finalized_hash: BlockHash,
) -> bool {
    if block_hash == finalized_hash {
        return false;
    }

    let mut visited = HashSet::new();
    let mut current_hash = block_hash;

    while current_hash != finalized_hash {
        if !visited.insert(current_hash) {
            return false;
        }

        let Some(block) = blocks.get(&current_hash) else {
            return false;
        };

        current_hash = block.parent_hash;
    }

    true
}

/// Tests whether a block's `logsBloom` may contain a log from any of the `trusted` addresses.
/// The block `logsBloom` is a consensus field with no false negatives, so a `false` result proves
/// none of the trusted pools emitted in the block; a `true` result is a maybe (bloom false positives).
/// Added so the authoritative log fetch can be skipped for blocks that provably touch no trusted pool.
fn block_may_touch_trusted_pool(bloom: &Bloom, trusted: &HashSet<Address>) -> bool {
    trusted
        .iter()
        .any(|address| bloom.contains_input(BloomInput::Raw(address.as_slice())))
}

/// Emits log-fetch requests for present canonical blocks whose logs are unknown.
/// Added so header connectivity automatically drives pool-affecting log discovery.
fn schedule_unknown_canonical_log_requests(
    chain: ChainKey,
    state: State,
    mut effects: Vec<Effect>,
) -> (State, Vec<Effect>) {
    let State {
        mut blocks,
        canonical_tip,
        pending_requests,
        finalized_state,
        pool_registry,
        token_registry,
        tick,
        streamed_logs,
    } = state;

    let pending_log_hashes = pending_requests.pending_block_log_hashes();
    let block_hashes = blocks.unknown_present_canonical_log_hashes(
        canonical_tip,
        finalized_state.block_hash,
        &pending_log_hashes,
    );
    // The bloom gate only fires once there is a verified pool whose log completeness to protect; with
    // none, the per-block fetch is still the discovery channel and every bloom-bearing block is
    // fetched. Capture that "gate active" decision from the verified set *before* adding the v4
    // PoolManager discovery anchor, so warmup behavior is unchanged.
    let mut trusted_addresses = pool_registry.verified_addresses(chain);
    let gate_active = !trusted_addresses.is_empty();
    if let Some(manager) = uniswap_v4::pool_manager_address(chain) {
        // Anchor on the singleton PoolManager so a block carrying only v4 activity is never
        // bloom-skipped; otherwise, once any v3 pool is verified, new v4 pools would never be found.
        trusted_addresses.insert(manager);
    }
    let mut pending_requests = pending_requests;

    for block_hash in block_hashes {
        // Skip the authoritative fetch only when the block's header bloom proves none of the
        // trusted pools emitted here. The bloom has no false negatives, so a trusted pool that
        // did emit is never skipped, keeping trusted-pool log completeness unchanged. With no
        // trusted pools yet, the per-block fetch is still the discovery channel, so we fetch.
        let resolve_empty_candidates = blocks.get(&block_hash).and_then(|block| {
            let bloom = block.logs_bloom.as_ref()?;
            (gate_active && !block_may_touch_trusted_pool(bloom, &trusted_addresses))
                .then(|| block.pool_logs.candidates().cloned().unwrap_or_default())
        });

        match resolve_empty_candidates {
            // The block touches no trusted pool: promote it to `Resolved` (preserving any
            // subscription-discovered candidates) without a fetch, so finalization is unblocked.
            Some(candidates) => {
                blocks = blocks.with_pool_logs(block_hash, PoolLogsStatus::Resolved(candidates));
            }
            None => {
                let request_payload = GetBlockLogs { block_hash };
                let (next_pending_requests, request_id) =
                    pending_requests.with_new_request(request_payload.clone(), tick);

                pending_requests = next_pending_requests;
                effects.push(Effect::Request(AnyIssuedRequest::BlockLogs(
                    IssuedRequest {
                        request_id,
                        request_payload,
                    },
                )));
            }
        }
    }

    (
        State {
            blocks,
            canonical_tip,
            pending_requests,
            finalized_state,
            pool_registry,
            token_registry,
            tick,
            streamed_logs,
        },
        effects,
    )
}

/// Emits pool metadata requests for unvalidated candidate addresses on the canonical path.
/// Added to turn log emitters into verified/rejected registry entries before using them as pools.
fn schedule_unknown_canonical_pool_metadata_requests(
    chain: ChainKey,
    state: State,
    mut effects: Vec<Effect>,
) -> (State, Vec<Effect>) {
    let State {
        blocks,
        canonical_tip,
        pending_requests,
        finalized_state,
        pool_registry,
        token_registry,
        tick,
        streamed_logs,
    } = state;

    let pending_candidates = pending_requests.pending_pool_metadata_candidates();
    let requests = blocks.unknown_present_canonical_pool_metadata_requests(
        chain,
        canonical_tip,
        finalized_state.block_hash,
        &pool_registry,
        &pending_candidates,
    );
    let mut pending_requests = pending_requests;

    for (block_hash, candidates) in requests {
        let request_payload = GetPoolMetadata {
            at: block_hash,
            candidates,
        };
        let (next_pending_requests, request_id) =
            pending_requests.with_new_request(request_payload.clone(), tick);

        pending_requests = next_pending_requests;
        effects.push(Effect::Request(AnyIssuedRequest::PoolMetadata(
            IssuedRequest {
                request_id,
                request_payload,
            },
        )));
    }

    (
        State {
            blocks,
            canonical_tip,
            pending_requests,
            finalized_state,
            pool_registry,
            token_registry,
            tick,
            streamed_logs,
        },
        effects,
    )
}

/// Emits token metadata requests for tokens referenced by verified canonical pools.
/// Added so reserve projection can use known decimals and avoid guessing token scale.
fn schedule_unknown_canonical_token_metadata_requests(
    chain: ChainKey,
    state: State,
    mut effects: Vec<Effect>,
) -> (State, Vec<Effect>) {
    let State {
        blocks,
        canonical_tip,
        pending_requests,
        finalized_state,
        pool_registry,
        token_registry,
        tick,
        streamed_logs,
    } = state;

    let pending_tokens = pending_requests.pending_token_metadata_tokens();
    let requests = blocks.unknown_present_canonical_token_metadata_requests(
        chain,
        canonical_tip,
        finalized_state.block_hash,
        &pool_registry,
        &token_registry,
        &pending_tokens,
    );
    let mut pending_requests = pending_requests;

    for (block_hash, tokens) in requests {
        let request_payload = GetTokenMetadata {
            at: block_hash,
            tokens,
        };
        let (next_pending_requests, request_id) =
            pending_requests.with_new_request(request_payload.clone(), tick);

        pending_requests = next_pending_requests;
        effects.push(Effect::Request(AnyIssuedRequest::TokenMetadata(
            IssuedRequest {
                request_id,
                request_payload,
            },
        )));
    }

    (
        State {
            blocks,
            canonical_tip,
            pending_requests,
            finalized_state,
            pool_registry,
            token_registry,
            tick,
            streamed_logs,
        },
        effects,
    )
}

/// Emits a pool-state request for trusted dirty pools accumulated on the canonical path.
/// Added so affected pools get fresh snapshots that can later feed reserve projection and optimization.
fn schedule_unknown_canonical_pool_data_request(
    chain: ChainKey,
    state: State,
    mut effects: Vec<Effect>,
) -> (State, Vec<Effect>) {
    let State {
        blocks,
        canonical_tip,
        pending_requests,
        finalized_state,
        pool_registry,
        token_registry,
        tick,
        streamed_logs,
    } = state;

    let pending_pool_data_by_block = pending_requests.pending_pool_data_pools_by_block();
    let request = blocks.unknown_present_canonical_pool_data_request(
        chain,
        canonical_tip,
        finalized_state.block_hash,
        &pool_registry,
        &pending_pool_data_by_block,
    );
    let mut pending_requests = pending_requests;

    if let Some((at, pools)) = request {
        let request_payload = GetPoolData { at, pools };
        let (next_pending_requests, request_id) =
            pending_requests.with_new_request(request_payload.clone(), tick);

        pending_requests = next_pending_requests;
        effects.push(Effect::Request(AnyIssuedRequest::PoolData(IssuedRequest {
            request_id,
            request_payload,
        })));
    }

    (
        State {
            blocks,
            canonical_tip,
            pending_requests,
            finalized_state,
            pool_registry,
            token_registry,
            tick,
            streamed_logs,
        },
        effects,
    )
}

/// Pools per background backfill request. Bounds each `GetPoolData` against the provider's per-request
/// budget; the next chunk waits until this one resolves, so the sweep is sequential.
const REGISTRY_BACKFILL_CHUNK: usize = 100;

/// Background sweep that snapshots never-covered registry pools in bounded chunks, but only when no
/// request is pending — so chain reconstruction and frontier freshness (which keep the pending tier
/// busy) always preempt it. Runs after every event from the kernel `transition` tail, so progress
/// tracks tip cadence rather than the coarse tick. One chunk is issued at a time: the chunk itself is
/// pending until it resolves, after which the next idle event issues the following chunk. Each fetched
/// pool becomes covered and leaves the uncovered set, so the sweep advances without a stored cursor.
fn schedule_registry_backfill_pool_data(
    chain: ChainKey,
    state: State,
    mut effects: Vec<Effect>,
) -> (State, Vec<Effect>) {
    if !state.pending_requests.is_empty() {
        return (state, effects);
    }

    let State {
        blocks,
        canonical_tip,
        pending_requests,
        finalized_state,
        pool_registry,
        token_registry,
        tick,
        streamed_logs,
    } = state;

    let pools = blocks.uncovered_verified_pool_chunk(
        chain,
        canonical_tip,
        finalized_state.block_hash,
        &finalized_state.pool_snapshots,
        &pool_registry,
        REGISTRY_BACKFILL_CHUNK,
    );
    let mut pending_requests = pending_requests;

    if !pools.is_empty() {
        let request_payload = GetPoolData {
            at: canonical_tip,
            pools,
        };
        let (next_pending_requests, request_id) =
            pending_requests.with_new_request(request_payload.clone(), tick);

        pending_requests = next_pending_requests;
        effects.push(Effect::Request(AnyIssuedRequest::PoolData(IssuedRequest {
            request_id,
            request_payload,
        })));
    }

    (
        State {
            blocks,
            canonical_tip,
            pending_requests,
            finalized_state,
            pool_registry,
            token_registry,
            tick,
            streamed_logs,
        },
        effects,
    )
}

/// Runs every canonical follow-up scheduler in the order needed for dependencies between requests.
/// Added so all transitions converge through one scheduling path instead of duplicating follow-up logic per event arm.
fn schedule_unknown_canonical_requests(
    chain: ChainKey,
    state: State,
    effects: Vec<Effect>,
) -> (State, Vec<Effect>) {
    let (state, effects) = schedule_unknown_canonical_log_requests(chain, state, effects);
    let (state, effects) = schedule_unknown_canonical_pool_metadata_requests(chain, state, effects);
    let (state, effects) =
        schedule_unknown_canonical_token_metadata_requests(chain, state, effects);
    schedule_unknown_canonical_pool_data_request(chain, state, effects)
}

/// Issues a block-header request for a missing ancestor, unless one for that hash is already in flight.
/// Added so every ancestry-reconnection site (head observations and header-received walks) dedups
/// against pending headers through one path: a converging missing parent is fetched once, not per event.
/// When deduped, returns no effect — the existing in-flight request carries the walk forward.
fn request_missing_header(
    pending_requests: PendingRequests,
    tick: Tick,
    missing_hash: BlockHash,
) -> (PendingRequests, Vec<Effect>) {
    if pending_requests.has_pending_header_request(missing_hash) {
        return (pending_requests, Vec::new());
    }

    let request_payload = GetBlockHeader {
        block_hash: missing_hash,
    };
    let (pending_requests, request_id) =
        pending_requests.with_new_request(request_payload.clone(), tick);

    (
        pending_requests,
        vec![Effect::Request(AnyIssuedRequest::BlockHeader(
            IssuedRequest {
                request_id,
                request_payload,
            },
        ))],
    )
}

/// Applies one kernel event to state and returns the side effects the runtime must execute.
/// Added as the pure deterministic boundary between EVM client state transitions and impure RPC/runtime work.
pub fn transition(chain: ChainKey, state: State, event: Event) -> (State, Vec<Effect>) {
    let (state, effects) = match event {
        Event::HeadObserved {
            hash,
            parent_hash,
            logs_bloom,
        } => {
            match state.blocks.with_new_block(
                hash,
                parent_hash,
                logs_bloom,
                state.finalized_state.block_hash,
            ) {
                Ok((blocks, None)) => schedule_unknown_canonical_requests(
                    chain,
                    State {
                        blocks,
                        canonical_tip: hash,
                        ..state
                    }
                    .with_streamed_logs_drained(chain, hash),
                    vec![],
                ),
                Ok((blocks, Some(missing_hash))) => {
                    let (pending_requests, effects) =
                        request_missing_header(state.pending_requests, state.tick, missing_hash);

                    schedule_unknown_canonical_requests(
                        chain,
                        State {
                            blocks,
                            canonical_tip: hash,
                            pending_requests,
                            ..state
                        }
                        .with_streamed_logs_drained(chain, hash),
                        effects,
                    )
                }
                Err(NewBlockError::SelfParentBlock(missing_hash, blocks)) => {
                    let (pending_requests, effects) =
                        request_missing_header(state.pending_requests, state.tick, missing_hash);
                    (
                        State {
                            blocks,
                            pending_requests,
                            ..state
                        },
                        effects,
                    )
                }
                Err(NewBlockError::ExistingBlock(blocks)) => schedule_unknown_canonical_requests(
                    chain,
                    State {
                        blocks,
                        canonical_tip: hash,
                        ..state
                    },
                    vec![],
                ),
                Err(NewBlockError::ConflictingBlockParent) => (
                    State {
                        pending_requests: state.pending_requests,
                        ..State::reset(
                            state.finalized_state,
                            state.tick,
                            state.pool_registry,
                            state.token_registry,
                        )
                    },
                    vec![],
                ),
                Err(NewBlockError::CycleDetected) => (
                    State {
                        pending_requests: state.pending_requests,
                        ..State::reset(
                            state.finalized_state,
                            state.tick,
                            state.pool_registry,
                            state.token_registry,
                        )
                    },
                    vec![],
                ),
            }
        }
        Event::FinalizedBlockObserved { block_hash } => (
            state.with_finalized_block_observed(chain, block_hash),
            vec![],
        ),
        Event::BlockHeaderReceived {
            request_id,
            hash,
            parent_hash,
            logs_bloom,
        } => {
            let (pending_requests, request_payload) = state.pending_requests.take(&request_id);

            let retry_request_payload = match request_payload {
                // found what we were looking for, no retry needed
                Some(PendingPayload { payload, .. }) if payload.block_hash == hash => None,
                // shouldn't happen, but something else was returned, still need the original request to succeed
                Some(payload) => Some(payload),
                // Unsolicited response, no matching request, ignore without retrying
                None => None,
            };

            let (new_state, effects, should_schedule_log_requests) =
                match state.blocks.with_new_block(
                    hash,
                    parent_hash,
                    logs_bloom,
                    state.finalized_state.block_hash,
                ) {
                    Ok((blocks, None)) => (
                        State {
                            blocks,
                            pending_requests,
                            ..state
                        },
                        vec![],
                        true,
                    ),
                    Ok((blocks, Some(missing_hash))) => {
                        let (pending_requests, effects) =
                            request_missing_header(pending_requests, state.tick, missing_hash);

                        (
                            State {
                                blocks,
                                pending_requests,
                                ..state
                            },
                            effects,
                            true,
                        )
                    }
                    Err(NewBlockError::SelfParentBlock(missing_hash, blocks)) => {
                        let (pending_requests, effects) =
                            request_missing_header(pending_requests, state.tick, missing_hash);
                        (
                            State {
                                blocks,
                                pending_requests,
                                ..state
                            },
                            effects,
                            false,
                        )
                    }
                    Err(NewBlockError::ExistingBlock(blocks)) => (
                        State {
                            blocks,
                            pending_requests,
                            ..state
                        },
                        vec![],
                        true,
                    ),
                    Err(NewBlockError::ConflictingBlockParent) => (
                        State {
                            pending_requests,
                            ..State::reset(
                                state.finalized_state,
                                state.tick,
                                state.pool_registry,
                                state.token_registry,
                            )
                        },
                        vec![],
                        false,
                    ),
                    Err(NewBlockError::CycleDetected) => (
                        State {
                            pending_requests,
                            ..State::reset(
                                state.finalized_state,
                                state.tick,
                                state.pool_registry,
                                state.token_registry,
                            )
                        },
                        vec![],
                        false,
                    ),
                };

            // A header can be the first time a block enters the graph, so drain any logs the
            // subscription staged for it before its head was processed.
            let new_state = new_state.with_streamed_logs_drained(chain, hash);

            let (new_state, effects) = if let Some(PendingPayload {
                payload: request_payload,
                ..
            }) = retry_request_payload
            {
                let (pending_requests, request_id) = new_state
                    .pending_requests
                    .with_new_request(request_payload.clone(), state.tick);

                (
                    State {
                        pending_requests,
                        ..new_state
                    },
                    effects
                        .into_iter()
                        .chain([Effect::Request(AnyIssuedRequest::BlockHeader(
                            IssuedRequest {
                                request_id,
                                request_payload,
                            },
                        ))])
                        .collect(),
                )
            } else {
                (new_state, effects)
            };

            if should_schedule_log_requests {
                schedule_unknown_canonical_requests(chain, new_state, effects)
            } else {
                (new_state, effects)
            }
        }
        Event::BlockHeaderNotFound { request_id } => {
            let (pending_requests, payload) = state.pending_requests.take(&request_id);
            if let Some(_) = payload {
                (
                    State {
                        pending_requests,
                        ..State::reset(
                            state.finalized_state,
                            state.tick,
                            state.pool_registry,
                            state.token_registry,
                        )
                    },
                    vec![],
                )
            } else {
                (
                    State {
                        pending_requests,
                        ..state
                    },
                    vec![],
                )
            }
        }
        Event::BlockLogsReceived { request_id, logs } => {
            let (pending_requests, request_payload) = state.pending_requests.take(&request_id);
            match request_payload {
                Some(PendingPayload {
                    payload: GetBlockLogs { block_hash },
                    ..
                }) => {
                    let state = State {
                        pending_requests,
                        ..state
                    }
                    .with_block_logs_applied(chain, block_hash, logs, true);

                    schedule_unknown_canonical_requests(chain, state, vec![])
                }
                None => (
                    State {
                        pending_requests,
                        ..state
                    },
                    vec![],
                ),
            }
        }
        Event::LogObserved { block_hash, logs } => {
            if state.blocks.get(&block_hash).is_some() {
                // Known block: apply provisionally, then schedule the authoritative `GetBlockLogs`
                // (the block is now `Partial`, which still needs a complete fetch).
                let state = state.with_block_logs_applied(chain, block_hash, logs, false);
                schedule_unknown_canonical_requests(chain, state, vec![])
            } else {
                // The head has not arrived yet: stage the logs until the block enters the graph.
                (state.with_streamed_logs_buffered(block_hash, logs), vec![])
            }
        }
        Event::PoolMetadataReceived {
            request_id,
            metadata,
        } => {
            let (pending_requests, request_payload) = state.pending_requests.take(&request_id);
            match request_payload {
                Some(PendingPayload {
                    payload:
                        GetPoolMetadata {
                            candidates: requested_candidates,
                            ..
                        },
                    ..
                }) => {
                    let metadata = metadata
                        .into_iter()
                        .filter(|(candidate, _)| requested_candidates.contains(candidate))
                        .collect::<HashMap<_, _>>();
                    let pool_registry = state.pool_registry.with_metadata_results(chain, metadata);

                    schedule_unknown_canonical_requests(
                        chain,
                        State {
                            pending_requests,
                            pool_registry,
                            ..state
                        },
                        vec![],
                    )
                }
                None => (
                    State {
                        pending_requests,
                        ..state
                    },
                    vec![],
                ),
            }
        }
        Event::TokenMetadataReceived {
            request_id,
            metadata,
        } => {
            let (pending_requests, request_payload) = state.pending_requests.take(&request_id);
            match request_payload {
                Some(PendingPayload {
                    payload:
                        GetTokenMetadata {
                            tokens: requested_tokens,
                            ..
                        },
                    ..
                }) => {
                    let metadata = metadata
                        .into_iter()
                        .filter(|(token, _)| requested_tokens.contains(token))
                        .collect::<HashMap<_, _>>();
                    let token_registry = state.token_registry.with_metadata_results(metadata);

                    schedule_unknown_canonical_requests(
                        chain,
                        State {
                            pending_requests,
                            token_registry,
                            ..state
                        },
                        vec![],
                    )
                }
                None => (
                    State {
                        pending_requests,
                        ..state
                    },
                    vec![],
                ),
            }
        }
        Event::PoolDataReceived { request_id, pools } => {
            let (pending_requests, request_payload) = state.pending_requests.take(&request_id);
            match request_payload {
                Some(PendingPayload {
                    payload:
                        GetPoolData {
                            at,
                            pools: requested_pools,
                        },
                    ..
                }) => {
                    let blocks = state.blocks.with_pool_data(at, requested_pools, pools);

                    (
                        State {
                            blocks,
                            pending_requests,
                            ..state
                        },
                        vec![],
                    )
                }
                None => (
                    State {
                        pending_requests,
                        ..state
                    },
                    vec![],
                ),
            }
        }
        Event::RequestFailed { request_id } => {
            let (pending_requests, issued_request) =
                state.pending_requests.retry(request_id, state.tick);

            (
                State {
                    pending_requests,
                    ..state
                },
                issued_request.into_iter().map(Effect::Request).collect(),
            )
        }
        Event::Tick => {
            let tick = state.tick.next();
            let (pending_requests, issued_requests) = state.pending_requests.retry_expired(tick);

            (
                State {
                    pending_requests,
                    tick,
                    ..state
                },
                issued_requests.into_iter().map(Effect::Request).collect(),
            )
        }
    };

    // After every event, when the priority tier is idle (no request pending), spend one slot on the
    // background registry pool-state backfill. Tip cadence (not the coarse tick) drives this, so fast
    // chains make progress between blocks; any pending priority request pauses it.
    schedule_registry_backfill_pool_data(chain, state, effects)
}

/// Walks from a tip toward finality and returns the first missing block hash, if any.
/// Added so disconnected heads can request exactly the next ancestry gap needed for reconstruction.
fn find_missing_block_hash(
    blocks: &HashMap<BlockHash, BlockNode>,
    tip_hash: BlockHash,
    finalized_hash: BlockHash,
) -> Result<Option<BlockHash>, BlocksGraphCycleError> {
    let mut visited = HashSet::new();
    let mut current_hash = tip_hash;

    while current_hash != finalized_hash {
        if !visited.insert(current_hash) {
            return Err(BlocksGraphCycleError);
        }

        let Some(block) = blocks.get(&current_hash) else {
            return Ok(Some(current_hash));
        };

        current_hash = block.parent_hash;
    }

    Ok(None)
}

/// Collects the distinct first missing ancestors across every seeded block, sorted so request-id
/// assignment stays deterministic. A seeded block whose parent was not itself seeded contributes
/// that gap hash; a cycle (which a well-formed seed never produces) contributes nothing.
fn missing_seed_parents(blocks: &BlocksGraph, finalized_hash: BlockHash) -> Vec<BlockHash> {
    // A well-formed seed graph never cycles; treat the impossible cycle error as "no gap".
    let mut missing = blocks
        .0
        .keys()
        .filter_map(|hash| {
            find_missing_block_hash(&blocks.0, *hash, finalized_hash).unwrap_or_default()
        })
        .collect::<Vec<_>>();
    missing.sort_unstable();
    missing.dedup();
    missing
}

#[cfg(test)]
mod tests {
    use super::*;

    use alloy::primitives::{Address, U160, U256, aliases::I24};

    use crate::tick::REQUEST_TTL_FOR_TEST as REQUEST_TTL;
    use crate::{PoolLogEvent, RangeLogBlock, bootstrap};
    use proptest::prelude::*;

    /// An all-ones `logsBloom` that matches every address, so a block carrying it is never
    /// bloom-gated. The behavior-preserving default for tests whose blocks enter the graph via a
    /// header event but that do not exercise the gate.
    fn bloom_matching_any() -> Bloom {
        Bloom::repeat_byte(0xff)
    }

    /// Builds a `logsBloom` seeded with each address exactly as a node accrues a log's emitter,
    /// so `block_may_touch_trusted_pool` sees the same bits a real header would carry.
    fn bloom_containing(addresses: &[Address]) -> Bloom {
        let mut bloom = Bloom::default();
        for address in addresses {
            bloom.accrue(BloomInput::Raw(address.as_slice()));
        }
        bloom
    }

    #[test]
    fn block_may_touch_trusted_pool_detects_a_seeded_address() {
        let trusted = Address::with_last_byte(1);
        let other = Address::with_last_byte(2);
        let bloom = bloom_containing(&[trusted]);

        assert!(block_may_touch_trusted_pool(
            &bloom,
            &HashSet::from([trusted])
        ));
        assert!(!block_may_touch_trusted_pool(
            &bloom,
            &HashSet::from([other])
        ));
    }

    #[test]
    fn block_may_touch_trusted_pool_is_false_for_empty_trusted_set() {
        let bloom = bloom_containing(&[Address::with_last_byte(1)]);
        assert!(!block_may_touch_trusted_pool(&bloom, &HashSet::new()));
    }

    #[derive(Debug)]
    struct GeneratedChain {
        parents: Vec<usize>,
        observed_heads: Vec<usize>,
    }

    #[derive(Clone, Copy, Debug)]
    enum GeneratedEvent {
        HeadObserved {
            hash_index: usize,
            parent_index: usize,
        },
        BlockHeaderReceived {
            request_id: u8,
            hash_index: usize,
            parent_index: usize,
        },
        BlockHeaderNotFound {
            request_id: u8,
        },
        RequestFailed {
            request_id: u8,
        },
        Tick,
    }

    #[derive(Clone, Copy, Debug)]
    enum GeneratedRetry {
        Failure,
        Expiration,
    }

    #[derive(Clone, Debug)]
    enum GeneratedRequestPayload {
        GetBlockLogs { block_index: u8 },
        GetBlockHeader { block_index: u8 },
    }

    #[derive(Clone, Debug)]
    enum ExpectedRequestPayload {
        GetBlockLogs { block_hash: BlockHash },
        GetBlockHeader { block_hash: BlockHash },
    }

    /// Generates varied rooted block graphs and observed heads.
    /// These inputs drive ancestry reconstruction properties without hard-coding only linear chains.
    fn generated_chain_strategy() -> impl Strategy<Value = GeneratedChain> {
        (2usize..24)
            .prop_flat_map(|node_count| {
                (
                    Just(node_count),
                    prop::collection::vec(any::<usize>(), node_count - 1),
                    prop::collection::vec(1usize..node_count, 1..64),
                )
            })
            .prop_map(|(node_count, parent_choices, observed_heads)| {
                let parents = (0..node_count)
                    .map(|node_index| {
                        if node_index == 0 {
                            0
                        } else {
                            parent_choices
                                .get(node_index - 1)
                                .copied()
                                .unwrap_or_default()
                                % node_index
                        }
                    })
                    .collect();

                GeneratedChain {
                    parents,
                    observed_heads,
                }
            })
    }

    /// Generates linear chains with reobserved heads.
    /// This keeps recovery properties focused on reset behavior rather than branch selection.
    fn generated_linear_chain_strategy() -> impl Strategy<Value = GeneratedChain> {
        (3usize..24)
            .prop_flat_map(|node_count| {
                (
                    Just(node_count),
                    prop::collection::vec(2usize..node_count, 1..64),
                )
            })
            .prop_map(|(node_count, observed_heads)| GeneratedChain {
                parents: (0..node_count)
                    .map(|node_index| node_index.saturating_sub(1))
                    .collect(),
                observed_heads,
            })
    }

    /// Generates mixed kernel events with arbitrary ids and hashes.
    /// The sequence properties use it to shake out safety invariant violations across unexpected event orderings.
    fn generated_event_strategy() -> impl Strategy<Value = GeneratedEvent> {
        prop_oneof![
            (0usize..16, 0usize..16).prop_map(|(hash_index, parent_index)| {
                GeneratedEvent::HeadObserved {
                    hash_index,
                    parent_index,
                }
            }),
            (0u8..16, 0usize..16, 0usize..16).prop_map(|(request_id, hash_index, parent_index)| {
                GeneratedEvent::BlockHeaderReceived {
                    request_id,
                    hash_index,
                    parent_index,
                }
            },),
            (0u8..16).prop_map(|request_id| GeneratedEvent::BlockHeaderNotFound { request_id }),
            (0u8..16).prop_map(|request_id| GeneratedEvent::RequestFailed { request_id }),
            Just(GeneratedEvent::Tick),
        ]
    }

    /// Builds bounded random event histories.
    /// This is the shared input for state-machine properties that need many transitions but finite shrinking.
    fn generated_event_sequence_strategy() -> impl Strategy<Value = Vec<GeneratedEvent>> {
        prop::collection::vec(generated_event_strategy(), 1..128)
    }

    /// Generates retry plans made of failures and expirations.
    /// Header reconstruction properties use it to prove delayed success still works after retry churn.
    fn generated_retry_plans_strategy() -> impl Strategy<Value = Vec<Vec<GeneratedRetry>>> {
        prop::collection::vec(
            prop::collection::vec(
                prop_oneof![
                    Just(GeneratedRetry::Failure),
                    Just(GeneratedRetry::Expiration),
                ],
                0..8,
            ),
            24,
        )
    }

    /// Generates request payload cases for retry tests.
    /// This keeps payload-identity assertions covering both header and log request types.
    fn generated_request_payload_strategy() -> impl Strategy<Value = GeneratedRequestPayload> {
        prop_oneof![
            any::<u8>()
                .prop_map(|block_index| GeneratedRequestPayload::GetBlockLogs { block_index }),
            any::<u8>()
                .prop_map(|block_index| GeneratedRequestPayload::GetBlockHeader { block_index }),
        ]
    }

    /// Asserts the core shape invariants for kernel state.
    /// Centralizing these checks makes each transition test catch graph, tip, and pool-data bookkeeping regressions.
    fn assert_state_invariants(state: &State) {
        assert_finalized_block_not_in_recent_blocks(state);
        assert_no_self_parent_blocks(state);
        assert_canonical_tip_is_known_or_finalized(state);
        assert_parent_walks_do_not_cycle(state);
        assert_pool_snapshots_and_failures_do_not_overlap(state);
    }

    /// Asserts no two in-flight header requests target the same block hash.
    /// This pairs with `assert_missing_parent_is_pending` ("every missing parent has a request") to
    /// guarantee each missing ancestor is fetched exactly once. Only asserted against states reached
    /// through production transitions (not the hand-seeded fixtures that inject duplicate requests on
    /// purpose to exercise id-matching), since the dedup is a property production maintains, not one
    /// preserved from an arbitrary seed.
    fn assert_no_duplicate_pending_header_requests(state: &State) {
        assert_eq!(
            state
                .pending_requests
                .pending_header_request_count_for_test(),
            state
                .pending_requests
                .pending_header_hashes_for_test()
                .len(),
            "no two pending header requests may target the same block hash"
        );
    }

    /// Asserts every present canonical block with unknown logs has an active log request.
    /// This protects the log scheduler from leaving canonical blocks permanently unqueried.
    fn assert_canonical_unknown_logs_are_pending(state: &State) {
        let pending_log_hashes = state.pending_requests.pending_block_log_hashes();
        let mut visited = HashSet::new();
        let mut current_hash = state.canonical_tip;

        while current_hash != state.finalized_state.block_hash {
            assert!(
                visited.insert(current_hash),
                "canonical walk must not cycle"
            );

            let Some(block) = state.blocks.get(&current_hash) else {
                break;
            };

            // `Partial` logs are not yet authoritative, so they must still have a pending fetch.
            if !matches!(block.pool_logs, PoolLogsStatus::Resolved(_)) {
                assert!(
                    pending_log_hashes.contains(&current_hash),
                    "canonical block without resolved logs must have a pending log request"
                );
            }

            current_hash = block.parent_hash;
        }
    }

    /// Asserts resolved canonical log candidates are either known by the registry or pending validation.
    /// This keeps candidate discovery connected to the trust boundary.
    fn assert_canonical_resolved_candidates_are_known_or_pending(state: &State) {
        let pending_candidates = state.pending_requests.pending_pool_metadata_candidates();
        let mut visited = HashSet::new();
        let mut current_hash = state.canonical_tip;

        while current_hash != state.finalized_state.block_hash {
            assert!(
                visited.insert(current_hash),
                "canonical walk must not cycle"
            );

            let Some(block) = state.blocks.get(&current_hash) else {
                break;
            };

            if let Some(candidates) = block.pool_logs.candidates() {
                for candidate in candidates {
                    assert!(
                        state.pool_registry.is_known(ChainKey::Ethereum, *candidate)
                            || pending_candidates.contains(candidate),
                        "canonical observed log candidate must be known or pending metadata validation"
                    );
                }
            }

            current_hash = block.parent_hash;
        }
    }

    /// Asserts every present canonical block has resolved logs after log-draining helpers run.
    /// Properties use it to distinguish incomplete draining from scheduler bugs.
    fn assert_present_canonical_logs_are_resolved(state: &State) {
        let mut visited = HashSet::new();
        let mut current_hash = state.canonical_tip;

        while current_hash != state.finalized_state.block_hash {
            assert!(
                visited.insert(current_hash),
                "canonical walk must not cycle"
            );

            let Some(block) = state.blocks.get(&current_hash) else {
                break;
            };

            assert!(
                matches!(block.pool_logs, PoolLogsStatus::Resolved(_)),
                "present canonical block logs must be resolved"
            );

            current_hash = block.parent_hash;
        }
    }

    /// Checks that every emitted request effect is recorded as pending with the same payload.
    /// This guards the runtime/kernel contract that effects are executable work, not orphan notifications.
    fn assert_effects_are_well_formed(state: &State, effects: &[Effect]) {
        for effect in effects {
            match effect {
                Effect::Request(AnyIssuedRequest::BlockHeader(IssuedRequest {
                    request_id,
                    request_payload,
                })) => {
                    let pending_request = state
                        .pending_requests
                        .get(request_id)
                        .expect("emitted header request must be recorded as pending");

                    assert_eq!(
                        pending_request.payload.block_hash,
                        request_payload.block_hash
                    );
                }
                Effect::Request(AnyIssuedRequest::BlockLogs(IssuedRequest {
                    request_id,
                    request_payload,
                })) => {
                    let pending_request = state
                        .pending_requests
                        .get(request_id)
                        .expect("emitted logs request must be recorded as pending");

                    assert_eq!(
                        pending_request.payload.block_hash,
                        request_payload.block_hash
                    );
                }
                Effect::Request(AnyIssuedRequest::PoolData(IssuedRequest {
                    request_id,
                    request_payload,
                })) => {
                    let pending_request = state
                        .pending_requests
                        .get(request_id)
                        .expect("emitted pool data request must be recorded as pending");

                    assert_eq!(pending_request.payload.at, request_payload.at);
                    assert_eq!(pending_request.payload.pools, request_payload.pools);
                }
                Effect::Request(AnyIssuedRequest::PoolMetadata(IssuedRequest {
                    request_id,
                    request_payload,
                })) => {
                    let pending_request = state
                        .pending_requests
                        .get(request_id)
                        .expect("emitted pool metadata request must be recorded as pending");

                    assert_eq!(pending_request.payload.at, request_payload.at);
                    assert_eq!(
                        pending_request.payload.candidates,
                        request_payload.candidates
                    );
                }
                Effect::Request(AnyIssuedRequest::TokenMetadata(IssuedRequest {
                    request_id,
                    request_payload,
                })) => {
                    let pending_request = state
                        .pending_requests
                        .get(request_id)
                        .expect("emitted token metadata request must be recorded as pending");

                    assert_eq!(pending_request.payload.at, request_payload.at);
                    assert_eq!(pending_request.payload.tokens, request_payload.tokens);
                }
            }
        }
    }

    /// Asserts the first missing ancestor for a tracked tip has a pending header request.
    /// This keeps disconnected canonical paths from stalling.
    fn assert_missing_parent_is_pending(state: &State, tip_hash: BlockHash) {
        let missing_hash = match find_missing_block_hash(
            &state.blocks.0,
            tip_hash,
            state.finalized_state.block_hash,
        ) {
            Ok(missing_hash) => missing_hash,
            Err(BlocksGraphCycleError) => panic!("canonical parent walk must not cycle"),
        };

        if let Some(missing_hash) = missing_hash {
            assert!(
                state
                    .pending_requests
                    .pending_header_hashes_for_test()
                    .contains(&missing_hash),
                "missing canonical parent must have a pending header request"
            );
        }
    }

    /// Applies the missing-parent assertion to every tracked recent block.
    /// This broadens ancestry progress checks beyond the canonical tip.
    fn assert_missing_parents_for_known_blocks_are_pending(state: &State) {
        for block_hash in state.blocks.0.keys() {
            assert_missing_parent_is_pending(state, *block_hash);
        }
    }

    /// Asserts finalized block hash is absent from recent block storage.
    /// This preserves the boundary between immutable finalized state and volatile reorg state.
    fn assert_finalized_block_not_in_recent_blocks(state: &State) {
        assert!(
            !state
                .blocks
                .0
                .contains_key(&state.finalized_state.block_hash),
            "finalized block must not be present in recent blocks"
        );
    }

    /// Asserts no recent block points to itself as parent.
    /// This catches the smallest ancestry cycle before graph walks depend on the data.
    fn assert_no_self_parent_blocks(state: &State) {
        for (hash, block) in &state.blocks.0 {
            assert_ne!(
                hash, &block.parent_hash,
                "block must not reference itself as parent"
            );
        }
    }

    /// Asserts canonical tip is either finalized or present in recent blocks.
    /// This keeps schedulers anchored to inspectable state.
    fn assert_canonical_tip_is_known_or_finalized(state: &State) {
        assert!(
            state.canonical_tip == state.finalized_state.block_hash
                || state.blocks.0.contains_key(&state.canonical_tip),
            "canonical tip must be finalized or present in recent blocks"
        );
    }

    /// Walks every parent chain and asserts it terminates without cycles.
    /// This protects canonical and scheduler traversals from non-termination.
    fn assert_parent_walks_do_not_cycle(state: &State) {
        for start_hash in state.blocks.0.keys() {
            let mut visited = HashSet::new();
            let mut current_hash = *start_hash;

            while let Some(block) = state.blocks.get(&current_hash) {
                assert!(visited.insert(current_hash), "parent walk must not cycle");
                current_hash = block.parent_hash;
            }
        }
    }

    /// Asserts no pool has both a snapshot and failure marker on the same block.
    /// This keeps pool-data state interpretable when scheduling retries or projecting reserves.
    fn assert_pool_snapshots_and_failures_do_not_overlap(state: &State) {
        for block in state.blocks.0.values() {
            for pool in block.pool_snapshots.keys() {
                assert!(
                    !block.pool_data_failures.contains_key(pool),
                    "pool snapshot and pool data failure must not overlap"
                );
            }
        }
    }

    /// Maps generated node indexes to deterministic block hashes.
    /// This gives properties stable, compact hash fixtures without arbitrary hash construction.
    fn hash_for_node(node_index: usize) -> BlockHash {
        BlockHash::with_last_byte((node_index + 1) as u8)
    }

    /// Converts generated event cases into real kernel events.
    /// This separates shrinking-friendly inputs from the concrete transition API.
    fn event_from_generated(generated_event: GeneratedEvent) -> Event {
        match generated_event {
            GeneratedEvent::HeadObserved {
                hash_index,
                parent_index,
            } => Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: hash_for_node(hash_index),
                parent_hash: hash_for_node(parent_index),
            },
            GeneratedEvent::BlockHeaderReceived {
                request_id,
                hash_index,
                parent_index,
            } => Event::BlockHeaderReceived {
                logs_bloom: bloom_matching_any(),
                request_id: RequestId::from_raw_for_test(u64::from(request_id)),
                hash: hash_for_node(hash_index),
                parent_hash: hash_for_node(parent_index),
            },
            GeneratedEvent::BlockHeaderNotFound { request_id } => Event::BlockHeaderNotFound {
                request_id: RequestId::from_raw_for_test(u64::from(request_id)),
            },
            GeneratedEvent::RequestFailed { request_id } => Event::RequestFailed {
                request_id: AnyRequestId::BlockHeader(RequestId::from_raw_for_test(u64::from(
                    request_id,
                ))),
            },
            GeneratedEvent::Tick => Event::Tick,
        }
    }

    /// Converts generated request payload cases into real expected payloads.
    /// Retry properties use this to compare payload identity after ids are replaced.
    fn request_payload_from_generated(payload: GeneratedRequestPayload) -> ExpectedRequestPayload {
        match payload {
            GeneratedRequestPayload::GetBlockLogs { block_index } => {
                ExpectedRequestPayload::GetBlockLogs {
                    block_hash: hash_for_node(usize::from(block_index)),
                }
            }
            GeneratedRequestPayload::GetBlockHeader { block_index } => {
                ExpectedRequestPayload::GetBlockHeader {
                    block_hash: hash_for_node(usize::from(block_index)),
                }
            }
        }
    }

    /// Builds a test tick from a raw counter.
    /// This keeps timing scenarios readable while staying behind the test-only Tick constructor.
    fn tick(value: u64) -> Tick {
        Tick::from_raw_for_test(value)
    }

    /// Wraps a header request id in a request-failed event.
    /// This keeps retry tests focused on behavior instead of enum plumbing.
    fn request_failed_for_header(request_id: RequestId<GetBlockHeader>) -> Event {
        Event::RequestFailed {
            request_id: AnyRequestId::BlockHeader(request_id),
        }
    }

    // Exercises the empty finalized-state constructor.
    // This locks down the bootstrap shape before recent chain tracking or persisted pool snapshots exist.
    #[test]
    fn finalized_state_empty_at_stores_hash_with_empty_pool_snapshots() {
        let finalized_hash = BlockHash::with_last_byte(1);

        let finalized_state = FinalizedState::empty_at(finalized_hash);

        assert_eq!(finalized_state.block_hash, finalized_hash);
        assert!(finalized_state.pool_snapshots.is_empty());
    }

    // Exercises State::init from a finalized anchor.
    // This confirms initialization itself does not schedule work or create recent blocks.
    #[test]
    fn state_init_from_finalized_state_starts_with_empty_tracking() {
        let finalized_hash = BlockHash::with_last_byte(1);

        let state = State::init(FinalizedState::empty_at(finalized_hash));

        assert_empty_initial_state_at(&state, finalized_hash);
        assert!(state.blocks.0.is_empty());
        assert!(state.finalized_state.pool_snapshots.is_empty());
        assert_eq!(state.tick.raw_for_test(), Tick::initial().raw_for_test());
    }

    /// Issues an expected request through PendingRequests.
    /// Retry tests use this to create fixtures through the same path as production scheduling.
    fn issue_expected_request(
        pending_requests: PendingRequests,
        payload: ExpectedRequestPayload,
        tick: Tick,
    ) -> (PendingRequests, AnyRequestId) {
        match payload {
            ExpectedRequestPayload::GetBlockLogs { block_hash } => {
                let (pending_requests, request_id) =
                    pending_requests.with_new_request(GetBlockLogs { block_hash }, tick);

                (pending_requests, AnyRequestId::BlockLogs(request_id))
            }
            ExpectedRequestPayload::GetBlockHeader { block_hash } => {
                let (pending_requests, request_id) =
                    pending_requests.with_new_request(GetBlockHeader { block_hash }, tick);

                (pending_requests, AnyRequestId::BlockHeader(request_id))
            }
        }
    }

    /// Checks whether a pending request still carries the expected payload.
    /// This catches retries that preserve only the request kind while losing the original target.
    fn pending_payload_matches(
        pending_requests: &PendingRequests,
        request_id: AnyRequestId,
        expected: &ExpectedRequestPayload,
    ) -> bool {
        match (request_id, expected) {
            (
                AnyRequestId::BlockLogs(request_id),
                ExpectedRequestPayload::GetBlockLogs { block_hash },
            ) => pending_requests
                .get(&request_id)
                .is_some_and(|request| request.payload.block_hash == *block_hash),
            (
                AnyRequestId::BlockHeader(request_id),
                ExpectedRequestPayload::GetBlockHeader { block_hash },
            ) => pending_requests
                .get(&request_id)
                .is_some_and(|request| request.payload.block_hash == *block_hash),
            _ => false,
        }
    }

    /// Extracts raw ids from typed request wrappers.
    /// Reset and retry tests use this to compare freshness across request kinds.
    fn any_request_id_raw(request_id: AnyRequestId) -> u64 {
        match request_id {
            AnyRequestId::BlockHeader(request_id) => request_id.raw_for_test(),
            AnyRequestId::BlockLogs(request_id) => request_id.raw_for_test(),
            AnyRequestId::PoolData(request_id) => request_id.raw_for_test(),
            AnyRequestId::PoolMetadata(request_id) => request_id.raw_for_test(),
            AnyRequestId::TokenMetadata(request_id) => request_id.raw_for_test(),
        }
    }

    /// Returns a generated node's parent index with finality as the fallback.
    /// This makes ancestry helpers total for fuzzed indexes.
    fn parent_index(chain: &GeneratedChain, node_index: usize) -> usize {
        chain.parents.get(node_index).copied().unwrap_or_default()
    }

    /// Finds the generated node index for a concrete block hash.
    /// Replay properties need this to answer emitted header requests from generated chains.
    fn node_index_for_hash(chain: &GeneratedChain, hash: BlockHash) -> Option<usize> {
        (0..chain.parents.len()).find(|node_index| hash_for_node(*node_index) == hash)
    }

    /// Computes the parent map that should exist after observing and reconstructing generated heads.
    /// This is the oracle for chain-replay properties.
    fn expected_observed_ancestor_closure(chain: &GeneratedChain) -> HashMap<BlockHash, BlockHash> {
        let mut expected = HashMap::new();

        for head_index in &chain.observed_heads {
            let mut current_index = *head_index;

            while current_index != 0 {
                let parent_index = parent_index(chain, current_index);
                expected.insert(hash_for_node(current_index), hash_for_node(parent_index));
                current_index = parent_index;
            }
        }

        expected
    }

    /// Applies an event and recursively answers header requests from the generated chain.
    /// This models a healthy RPC layer so reconstruction properties can focus on kernel state.
    fn apply_event_and_drain_block_headers(
        mut state: State,
        chain: &GeneratedChain,
        event: Event,
    ) -> State {
        let (next_state, effects) = transition(ChainKey::Ethereum, state, event);
        state = next_state;
        assert_state_invariants(&state);

        let mut pending_effects = effects;

        while let Some(effect) = pending_effects.pop() {
            match effect {
                Effect::Request(AnyIssuedRequest::BlockHeader(IssuedRequest {
                    request_id,
                    request_payload: GetBlockHeader { block_hash },
                })) => {
                    let block_index = node_index_for_hash(chain, block_hash)
                        .expect("requested block header must belong to generated chain");
                    assert_ne!(block_index, 0, "finalized header must not be requested");

                    let parent_hash = hash_for_node(parent_index(chain, block_index));
                    let (next_state, effects) = transition(
                        ChainKey::Ethereum,
                        state,
                        Event::BlockHeaderReceived {
                            logs_bloom: bloom_matching_any(),
                            request_id,
                            hash: block_hash,
                            parent_hash,
                        },
                    );

                    state = next_state;
                    assert_state_invariants(&state);
                    pending_effects.extend(effects);
                }
                Effect::Request(AnyIssuedRequest::BlockLogs(IssuedRequest {
                    request_id, ..
                })) => {
                    let (next_state, effects) = transition(
                        ChainKey::Ethereum,
                        state,
                        Event::BlockLogsReceived {
                            request_id,
                            logs: Vec::new(),
                        },
                    );

                    state = next_state;
                    assert_state_invariants(&state);
                    pending_effects.extend(effects);
                }
                Effect::Request(AnyIssuedRequest::PoolData(_)) => {}
                Effect::Request(AnyIssuedRequest::PoolMetadata(_)) => {}
                Effect::Request(AnyIssuedRequest::TokenMetadata(_)) => {}
            }
        }

        state
    }

    /// Applies an event while injecting configured failures or expirations before header success.
    /// This proves ancestry reconstruction survives retry churn.
    fn apply_event_and_drain_block_headers_with_retries(
        mut state: State,
        chain: &GeneratedChain,
        event: Event,
        retry_plans: &[Vec<GeneratedRetry>],
    ) -> State {
        let (next_state, effects) = transition(ChainKey::Ethereum, state, event);
        state = next_state;
        assert_state_invariants(&state);

        let mut pending_effects = effects;

        while let Some(effect) = pending_effects.pop() {
            match effect {
                Effect::Request(AnyIssuedRequest::BlockHeader(IssuedRequest {
                    request_id,
                    request_payload: GetBlockHeader { block_hash },
                })) => {
                    let block_index = node_index_for_hash(chain, block_hash)
                        .expect("requested block header must belong to generated chain");
                    let parent_hash = hash_for_node(parent_index(chain, block_index));
                    let mut current_request_id = request_id;

                    for retry in retry_plans.get(block_index).into_iter().flatten() {
                        let (next_state, effects) = match retry {
                            GeneratedRetry::Failure => transition(
                                ChainKey::Ethereum,
                                state,
                                request_failed_for_header(current_request_id),
                            ),
                            GeneratedRetry::Expiration => advance_ticks(state, REQUEST_TTL),
                        };

                        state = next_state;
                        current_request_id =
                            assert_single_block_header_request_effect(&effects, block_hash);
                        assert_state_invariants(&state);
                        assert_effects_are_well_formed(&state, &effects);
                        assert_missing_parents_for_known_blocks_are_pending(&state);
                    }

                    let (next_state, effects) = transition(
                        ChainKey::Ethereum,
                        state,
                        Event::BlockHeaderReceived {
                            logs_bloom: bloom_matching_any(),
                            request_id: current_request_id,
                            hash: block_hash,
                            parent_hash,
                        },
                    );

                    state = next_state;
                    assert_state_invariants(&state);
                    assert_effects_are_well_formed(&state, &effects);
                    pending_effects.extend(effects);
                }
                Effect::Request(AnyIssuedRequest::BlockLogs(IssuedRequest {
                    request_id, ..
                })) => {
                    let (next_state, effects) = transition(
                        ChainKey::Ethereum,
                        state,
                        Event::BlockLogsReceived {
                            request_id,
                            logs: Vec::new(),
                        },
                    );

                    state = next_state;
                    assert_state_invariants(&state);
                    assert_effects_are_well_formed(&state, &effects);
                    pending_effects.extend(effects);
                }
                Effect::Request(AnyIssuedRequest::PoolData(_)) => {}
                Effect::Request(AnyIssuedRequest::PoolMetadata(_)) => {}
                Effect::Request(AnyIssuedRequest::TokenMetadata(_)) => {}
            }
        }

        state
    }

    /// Builds a minimal recent block with unknown logs.
    /// Scheduler and invariant tests use it when pool contents are irrelevant.
    fn block_with_parent(parent_hash: BlockHash) -> BlockNode {
        BlockNode {
            logs_bloom: None,
            parent_hash,
            pool_logs: PoolLogsStatus::Unknown,
            pool_snapshots: HashMap::new(),
            pool_data_failures: HashMap::new(),
        }
    }

    /// Builds a clean kernel state anchored at a finalized hash.
    /// This gives tests empty registries, no pending work, and deterministic tick state.
    fn empty_state_at(finalized_hash: BlockHash) -> State {
        State {
            blocks: BlocksGraph::new(),
            canonical_tip: finalized_hash,
            pending_requests: PendingRequests::new(),
            finalized_state: FinalizedState {
                block_hash: finalized_hash,
                pool_snapshots: HashMap::new(),
            },
            pool_registry: TrustedPoolRegistry::new(),
            token_registry: TokenRegistry::new(),
            tick: tick(0),
            streamed_logs: HashMap::new(),
        }
    }

    /// Asserts chain state was reset to the finalized anchor.
    /// This captures the expected recovery shape after unsafe ancestry observations.
    fn assert_chain_reset_at(state: &State, finalized_hash: BlockHash) {
        assert!(state.blocks.0.is_empty());
        assert_eq!(state.canonical_tip, finalized_hash);
        assert_eq!(state.finalized_state.block_hash, finalized_hash);
    }

    /// Asserts the common empty-state baseline.
    /// This keeps initialization tests aligned with reset expectations without duplicating fields.
    fn assert_empty_initial_state_at(state: &State, finalized_hash: BlockHash) {
        assert_chain_reset_at(state, finalized_hash);
        assert!(state.pending_requests.is_empty_for_test());
    }

    /// Asserts a single inserted block has unknown logs and no pool data.
    /// This verifies header ingestion before log/data fetches enrich the block.
    fn assert_single_unknown_block(state: &State, hash: BlockHash, parent_hash: BlockHash) {
        let block = state
            .blocks
            .get(&hash)
            .expect("expected block to be present");

        assert_eq!(block.parent_hash, parent_hash);
        assert!(matches!(block.pool_logs, PoolLogsStatus::Unknown));
        assert!(block.pool_snapshots.is_empty());
    }

    /// Asserts a tracked block has the expected parent and no pool snapshots.
    /// This lets ancestry tests ignore log status when it is scenario-specific.
    fn assert_single_block_with_parent(state: &State, hash: BlockHash, parent_hash: BlockHash) {
        let block = state
            .blocks
            .get(&hash)
            .expect("expected block to be present");

        assert_eq!(block.parent_hash, parent_hash);
        assert!(block.pool_snapshots.is_empty());
    }

    /// Asserts resolved log candidates were stored on one block.
    /// This catches decoded-log application bugs without depending on candidate ordering.
    fn assert_resolved_pool_logs(
        state: &State,
        hash: BlockHash,
        parent_hash: BlockHash,
        expected_logs: &HashSet<ProtocolPoolKey>,
    ) {
        let block = state
            .blocks
            .get(&hash)
            .expect("expected block to be present");

        assert_eq!(block.parent_hash, parent_hash);
        assert!(block.pool_snapshots.is_empty());

        let PoolLogsStatus::Resolved(logs) = &block.pool_logs else {
            panic!("expected resolved pool logs");
        };

        assert_eq!(logs.len(), expected_logs.len());
        for pool_address in expected_logs {
            assert!(logs.contains(pool_address));
        }
    }

    /// Asserts the resolved candidate set matches exactly.
    /// This is used when ordering is irrelevant but membership/cardinality must be preserved.
    fn assert_resolved_candidate_logs(
        state: &State,
        hash: BlockHash,
        parent_hash: BlockHash,
        expected_logs: &HashSet<ProtocolPoolKey>,
    ) {
        let block = state
            .blocks
            .get(&hash)
            .expect("expected block to be present");

        assert_eq!(block.parent_hash, parent_hash);
        assert!(block.pool_snapshots.is_empty());

        let PoolLogsStatus::Resolved(logs) = &block.pool_logs else {
            panic!("expected resolved pool logs");
        };

        assert_eq!(logs, expected_logs);
    }

    /// Asserts trusted logs are derived from resolved candidates and registry state.
    /// This keeps block storage from duplicating trust decisions.
    fn assert_trusted_pool_logs_resolved(
        state: &State,
        hash: BlockHash,
        expected_pools: HashSet<PoolRef>,
    ) {
        assert_eq!(
            state
                .blocks
                .trusted_pool_logs(ChainKey::Ethereum, hash, &state.pool_registry)
                .expect("block must be present"),
            TrustedPoolLogs::Resolved(expected_pools)
        );
    }

    /// Extracts the single expected header request for a block.
    /// Tests use it to fail on missing or duplicate scheduling.
    fn assert_single_block_header_request_effect(
        effects: &[Effect],
        block_hash: BlockHash,
    ) -> RequestId<GetBlockHeader> {
        let request_ids = effects
            .iter()
            .filter_map(|effect| match effect {
                Effect::Request(AnyIssuedRequest::BlockHeader(IssuedRequest {
                    request_id,
                    request_payload:
                        GetBlockHeader {
                            block_hash: requested_hash,
                        },
                })) if *requested_hash == block_hash => Some(*request_id),
                _ => None,
            })
            .collect::<Vec<_>>();

        assert_eq!(request_ids.len(), 1);
        request_ids[0]
    }

    /// Extracts the single expected log request for a block.
    /// Tests use it to fail on missing or duplicate log scheduling.
    fn assert_single_block_log_request_effect(
        effects: &[Effect],
        block_hash: BlockHash,
    ) -> RequestId<GetBlockLogs> {
        let request_ids = effects
            .iter()
            .filter_map(|effect| match effect {
                Effect::Request(AnyIssuedRequest::BlockLogs(IssuedRequest {
                    request_id,
                    request_payload:
                        GetBlockLogs {
                            block_hash: requested_hash,
                        },
                })) if *requested_hash == block_hash => Some(*request_id),
                _ => None,
            })
            .collect::<Vec<_>>();

        assert_eq!(request_ids.len(), 1);
        request_ids[0]
    }

    /// Asserts no authoritative block-log request was emitted for `block_hash`.
    /// Documents the bloom gate skipping the fetch for a block that touches no trusted pool.
    fn assert_no_block_log_request_effect(effects: &[Effect], block_hash: BlockHash) {
        assert!(!effects.iter().any(|effect| matches!(
            effect,
            Effect::Request(AnyIssuedRequest::BlockLogs(IssuedRequest {
                request_payload: GetBlockLogs {
                    block_hash: requested_hash,
                },
                ..
            })) if *requested_hash == block_hash
        )));
    }

    /// Extracts the single expected pool-data request for a block and pool set.
    /// This locks scheduler targeting to one snapshot block and exact pools.
    fn assert_single_pool_data_request_effect(
        effects: &[Effect],
        at: BlockHash,
        pools: &HashSet<PoolRef>,
    ) -> RequestId<GetPoolData> {
        let request_ids = effects
            .iter()
            .filter_map(|effect| match effect {
                Effect::Request(AnyIssuedRequest::PoolData(IssuedRequest {
                    request_id,
                    request_payload:
                        GetPoolData {
                            at: requested_at,
                            pools: requested_pools,
                        },
                })) if *requested_at == at && requested_pools == pools => Some(*request_id),
                _ => None,
            })
            .collect::<Vec<_>>();

        assert_eq!(request_ids.len(), 1);
        request_ids[0]
    }

    /// Collects pool-data request payloads from effects.
    /// Property tests use it to reason about all emitted snapshot work.
    fn pool_data_request_payloads_from_effects(
        effects: &[Effect],
    ) -> Vec<(BlockHash, HashSet<PoolRef>)> {
        effects
            .iter()
            .filter_map(|effect| match effect {
                Effect::Request(AnyIssuedRequest::PoolData(IssuedRequest {
                    request_payload: GetPoolData { at, pools },
                    ..
                })) => Some((*at, pools.clone())),
                _ => None,
            })
            .collect()
    }

    /// Asserts no pool-data requests were emitted.
    /// This documents scenarios where scheduling is blocked, unnecessary, or already covered.
    fn assert_no_pool_data_request_effect(effects: &[Effect]) {
        assert!(pool_data_request_payloads_from_effects(effects).is_empty());
    }

    /// Extracts the single expected pool-metadata request.
    /// This keeps validation scheduler tests strict about block and candidate batching.
    fn assert_single_pool_metadata_request_effect(
        effects: &[Effect],
        at: BlockHash,
        candidates: &HashSet<ProtocolPoolKey>,
    ) -> RequestId<GetPoolMetadata> {
        let request_ids = effects
            .iter()
            .filter_map(|effect| match effect {
                Effect::Request(AnyIssuedRequest::PoolMetadata(IssuedRequest {
                    request_id,
                    request_payload:
                        GetPoolMetadata {
                            at: requested_at,
                            candidates: requested_candidates,
                        },
                })) if *requested_at == at && requested_candidates == candidates => {
                    Some(*request_id)
                }
                _ => None,
            })
            .collect::<Vec<_>>();

        assert_eq!(request_ids.len(), 1);
        request_ids[0]
    }

    /// Extracts the single expected token-metadata request.
    /// This keeps token scheduler tests strict about block and token batching.
    fn assert_single_token_metadata_request_effect(
        effects: &[Effect],
        at: BlockHash,
        tokens: &HashSet<TokenAddress>,
    ) -> RequestId<GetTokenMetadata> {
        let request_ids = effects
            .iter()
            .filter_map(|effect| match effect {
                Effect::Request(AnyIssuedRequest::TokenMetadata(IssuedRequest {
                    request_id,
                    request_payload:
                        GetTokenMetadata {
                            at: requested_at,
                            tokens: requested_tokens,
                        },
                })) if *requested_at == at && requested_tokens == tokens => Some(*request_id),
                _ => None,
            })
            .collect::<Vec<_>>();

        assert_eq!(request_ids.len(), 1);
        request_ids[0]
    }

    /// Extracts a single header or log request and checks its payload.
    /// Retry tests use it when request kind is generated.
    fn assert_single_request_effect(
        effects: &[Effect],
        expected_payload: &ExpectedRequestPayload,
    ) -> AnyRequestId {
        assert_eq!(effects.len(), 1);

        match (&effects[0], expected_payload) {
            (
                Effect::Request(AnyIssuedRequest::BlockHeader(IssuedRequest {
                    request_id,
                    request_payload,
                })),
                ExpectedRequestPayload::GetBlockHeader { block_hash },
            ) => {
                assert_eq!(request_payload.block_hash, *block_hash);
                AnyRequestId::BlockHeader(*request_id)
            }
            (
                Effect::Request(AnyIssuedRequest::BlockLogs(IssuedRequest {
                    request_id,
                    request_payload,
                })),
                ExpectedRequestPayload::GetBlockLogs { block_hash },
            ) => {
                assert_eq!(request_payload.block_hash, *block_hash);
                AnyRequestId::BlockLogs(*request_id)
            }
            _ => panic!("expected single matching request effect"),
        }
    }

    /// Builds a deterministic pool address fixture.
    /// This keeps unit and property tests compact while avoiding repeated Address construction.
    fn pool_address(last_byte: u8) -> PoolRef {
        PoolRef::uniswap_v3(Address::with_last_byte(last_byte), ChainKey::Ethereum)
    }

    /// Builds a deterministic candidate address fixture.
    /// This preserves the identity relationship between a log emitter and its potential pool.
    fn pool_candidate_address(last_byte: u8) -> ProtocolPoolKey {
        ProtocolPoolKey::UniswapV3(Address::with_last_byte(last_byte))
    }

    /// Builds the `BlockLogsReceived` payload that names exactly `candidates`. Each log is a
    /// liquidity delta with no base, so it derives to no snapshot: candidate-only tests keep their
    /// existing semantics (the block resolves and the pools are dirtied, nothing is derived).
    fn pool_logs(candidates: &HashSet<ProtocolPoolKey>) -> Vec<PoolLog> {
        candidates
            .iter()
            .enumerate()
            .map(|(index, candidate)| PoolLog {
                pool: *candidate,
                log_index: index as u64,
                event: PoolLogEvent::Mint {
                    tick_lower: I24::try_from(-1).expect("test tick fits int24"),
                    tick_upper: I24::try_from(1).expect("test tick fits int24"),
                    amount: 1,
                },
            })
            .collect()
    }

    /// Builds a single `Swap` log whose absolute snapshot equals `pool_state`.
    fn swap_log(
        candidate: ProtocolPoolKey,
        log_index: u64,
        pool_state: &PoolState,
    ) -> PoolLog {
        PoolLog {
            pool: candidate,
            log_index,
            event: PoolLogEvent::Swap {
                sqrt_price_x96: pool_state.sqrt_price_x96,
                tick: pool_state.tick,
                liquidity: pool_state.liquidity,
            },
        }
    }

    /// Builds a single `Mint` log carrying a liquidity delta over `[tick_lower, tick_upper)`.
    fn mint_log(
        candidate: ProtocolPoolKey,
        log_index: u64,
        tick_lower: i32,
        tick_upper: i32,
        amount: u128,
    ) -> PoolLog {
        PoolLog {
            pool: candidate,
            log_index,
            event: PoolLogEvent::Mint {
                tick_lower: I24::try_from(tick_lower).expect("test tick fits int24"),
                tick_upper: I24::try_from(tick_upper).expect("test tick fits int24"),
                amount,
            },
        }
    }

    /// Reads a block's derived/stored snapshot for a pool, if any.
    fn block_pool_snapshot<'a>(
        state: &'a State,
        block_hash: BlockHash,
        pool: PoolRef,
    ) -> Option<&'a PoolState> {
        state.blocks.get(&block_hash)?.pool_snapshots.get(&pool)
    }

    /// Builds pool metadata fixtures directly.
    /// This lets registry/kernel tests avoid exercising RPC metadata decoding.
    fn pool_metadata(token0: u8, token1: u8, fee: UniswapV3Fee) -> PoolMetadata {
        PoolMetadata {
            token0: Address::with_last_byte(token0),
            token1: Address::with_last_byte(token1),
            fee: PoolFee::Tiered(fee),
        }
    }

    /// Builds a deterministic token address fixture.
    /// Scheduler tests use it as a stable key for token registry state.
    fn token_address(last_byte: u8) -> TokenAddress {
        TokenAddress(Address::with_last_byte(last_byte), ChainKey::Ethereum)
    }

    /// Builds token metadata with supported decimals.
    /// This lets token-registry tests set known metadata without decoding RPC results.
    fn token_metadata(decimals: u8) -> TokenMetadata {
        TokenMetadata {
            decimals: TokenDecimals::try_from_u256(U256::from(decimals))
                .expect("test decimals must be supported"),
        }
    }

    /// Builds a distinguishable pool state fixture.
    /// Pool-data tests use it to confirm the right snapshot lands on the right block and pool.
    fn pool_state(last_byte: u8) -> PoolState {
        PoolState {
            sqrt_price_x96: U160::from(u64::from(last_byte) + 1),
            tick: I24::try_from(i32::from(last_byte)).expect("test tick must fit int24"),
            liquidity: u128::from(last_byte) + 10,
        }
    }

    /// Creates deterministic mixed pool-data results from a byte.
    /// Properties use it to cover success and failure entries without separate generators.
    fn pool_data_result_for_byte(last_byte: u8) -> PoolDataResult {
        if last_byte % 2 == 0 {
            Ok(pool_state(last_byte))
        } else {
            Err(PoolDataFailure::CallFailed(PoolDataCall::Slot0))
        }
    }

    /// Builds a resolved block with explicit log candidates and pool snapshots.
    /// Latest-complete-query tests use it to shape path readiness directly.
    fn resolved_block_with_snapshots(
        parent_hash: BlockHash,
        candidates: HashSet<ProtocolPoolKey>,
        pool_snapshots: HashMap<PoolRef, PoolState>,
    ) -> BlockNode {
        BlockNode {
            logs_bloom: None,
            parent_hash,
            pool_logs: PoolLogsStatus::Resolved(candidates),
            pool_snapshots,
            pool_data_failures: HashMap::new(),
        }
    }

    /// Builds the expected (block, resolved pool states) overlay for query tests.
    /// The query now returns snapshot *locations*, so tests compare against resolved states.
    fn complete_pool_state_update(
        block_hash: BlockHash,
        pool_states: HashMap<PoolRef, PoolState>,
    ) -> (BlockHash, HashMap<PoolRef, PoolState>) {
        (block_hash, pool_states)
    }

    /// Runs the overlay query and resolves its locations into owned pool states for comparison.
    /// `None` propagates both a missing overlay and an unresolvable location (a broken invariant).
    fn resolved_complete_pool_state_update_from(
        state: &State,
        start: BlockHash,
    ) -> Option<(BlockHash, HashMap<PoolRef, PoolState>)> {
        let update = state.latest_complete_pool_state_update_from(ChainKey::Ethereum, start)?;
        let pool_states = state
            .resolve_complete_pool_states(&update)?
            .into_iter()
            .map(|(pool, pool_state)| (pool, pool_state.clone()))
            .collect();
        Some((update.block_hash, pool_states))
    }

    /// Resolves the latest complete overlay anchored at the canonical tip.
    fn resolved_complete_pool_state_update(
        state: &State,
    ) -> Option<(BlockHash, HashMap<PoolRef, PoolState>)> {
        resolved_complete_pool_state_update_from(state, state.canonical_tip)
    }

    /// Asserts a pool snapshot equals the expected state on a block.
    /// This confirms successful pool-data responses mutate the intended target only.
    fn assert_pool_snapshot(
        state: &State,
        block_hash: BlockHash,
        pool: PoolRef,
        expected: &PoolState,
    ) {
        let block = state
            .blocks
            .get(&block_hash)
            .expect("expected block to be present");

        assert_eq!(block.pool_snapshots.get(&pool), Some(expected));
    }

    /// Asserts a block has no snapshot for a pool.
    /// This catches stale, failed, and unrequested pool-data writes.
    fn assert_no_pool_snapshot(state: &State, block_hash: BlockHash, pool: PoolRef) {
        let block = state
            .blocks
            .get(&block_hash)
            .expect("expected block to be present");

        assert!(!block.pool_snapshots.contains_key(&pool));
    }

    /// Asserts a block records the expected pool-data failure for a pool.
    /// This protects retry-suppression semantics for deterministic per-pool failures.
    fn assert_pool_failure(
        state: &State,
        block_hash: BlockHash,
        pool: PoolRef,
        expected: &PoolDataFailure,
    ) {
        let block = state
            .blocks
            .get(&block_hash)
            .expect("expected block to be present");

        assert_eq!(block.pool_data_failures.get(&pool), Some(expected));
    }

    /// Asserts a block has no failure marker for a pool.
    /// This keeps successes and unrequested entries from leaving misleading failure state.
    fn assert_no_pool_failure(state: &State, block_hash: BlockHash, pool: PoolRef) {
        let block = state
            .blocks
            .get(&block_hash)
            .expect("expected block to be present");

        assert!(!block.pool_data_failures.contains_key(&pool));
    }

    /// Advances the state by a fixed number of ticks and returns emitted effects.
    /// Retry tests use it to drive TTL scenarios without duplicating loops.
    fn advance_ticks(mut state: State, count: u64) -> (State, Vec<Effect>) {
        let mut effects = Vec::new();

        for _ in 0..count {
            let (next_state, tick_effects) = transition(ChainKey::Ethereum, state, Event::Tick);
            state = next_state;
            effects.extend(tick_effects);
        }

        (state, effects)
    }

    /// Satisfies emitted log requests with empty results.
    /// Chain-reconstruction properties use it when only ancestry, not pool contents, matters.
    fn drain_block_log_effects(mut state: State, effects: &[Effect]) -> State {
        for effect in effects {
            let Effect::Request(AnyIssuedRequest::BlockLogs(IssuedRequest { request_id, .. })) =
                effect
            else {
                continue;
            };

            let (next_state, effects) = transition(
                ChainKey::Ethereum,
                state,
                Event::BlockLogsReceived {
                    request_id: *request_id,
                    logs: Vec::new(),
                },
            );

            state = next_state;
            assert!(effects.is_empty());
            assert_state_invariants(&state);
        }

        state
    }

    /// Converts effects into the set of requested header hashes, panicking on other effect kinds.
    /// This is for scenarios expected to emit only header work.
    fn header_hashes_from_effects(effects: &[Effect]) -> HashSet<BlockHash> {
        effects
            .iter()
            .map(|effect| match effect {
                Effect::Request(AnyIssuedRequest::BlockHeader(IssuedRequest {
                    request_payload:
                        GetBlockHeader {
                            block_hash: requested_hash,
                        },
                    ..
                })) => *requested_hash,
                _ => panic!("expected block header request effect"),
            })
            .collect()
    }

    /// Collects header request hashes from mixed effects.
    /// This lets assertions inspect header scheduling without rejecting unrelated follow-up effects.
    fn header_request_hashes_from_effects(effects: &[Effect]) -> HashSet<BlockHash> {
        effects
            .iter()
            .filter_map(|effect| match effect {
                Effect::Request(AnyIssuedRequest::BlockHeader(IssuedRequest {
                    request_payload:
                        GetBlockHeader {
                            block_hash: requested_hash,
                        },
                    ..
                })) => Some(*requested_hash),
                _ => None,
            })
            .collect()
    }

    /// Collects block-log request hashes from mixed effects.
    /// This lets assertions inspect log scheduling without rejecting unrelated follow-up effects.
    fn block_log_request_hashes_from_effects(effects: &[Effect]) -> HashSet<BlockHash> {
        effects
            .iter()
            .filter_map(|effect| match effect {
                Effect::Request(AnyIssuedRequest::BlockLogs(IssuedRequest {
                    request_payload:
                        GetBlockLogs {
                            block_hash: requested_hash,
                        },
                    ..
                })) => Some(*requested_hash),
                _ => None,
            })
            .collect()
    }

    /// Asserts the exact header and log request sets in an effect batch.
    /// This keeps scheduler-output tests concise while still detecting extra work.
    fn assert_request_hashes(
        effects: &[Effect],
        expected_header_hashes: HashSet<BlockHash>,
        expected_log_hashes: HashSet<BlockHash>,
    ) {
        assert_eq!(
            header_request_hashes_from_effects(effects),
            expected_header_hashes
        );
        assert_eq!(
            block_log_request_hashes_from_effects(effects),
            expected_log_hashes
        );
        assert_eq!(
            effects.len(),
            expected_header_hashes.len() + expected_log_hashes.len()
        );
    }

    /// Returns pending header hashes through the test-only API.
    /// This keeps tests independent of PendingRequests' internal map layout.
    fn pending_header_hashes(state: &State) -> HashSet<BlockHash> {
        state.pending_requests.pending_header_hashes_for_test()
    }

    /// Reads the dispatch tick for an active header request.
    /// Retry tests use it to prove timing is based on the current request id.
    fn active_request_dispatch_tick(
        pending_requests: &PendingRequests,
        request_id: RequestId<GetBlockHeader>,
    ) -> Tick {
        pending_requests
            .header_dispatch_tick_for_test(&request_id)
            .expect("active request must have a dispatch tick")
    }

    /// Asserts every active request has exactly one dispatch tick.
    /// This catches bookkeeping leaks when retries replace ids.
    fn assert_active_requests_have_exactly_one_dispatch_tick(state: &State) {
        assert_eq!(
            state.pending_requests.dispatch_ticks_for_test().len(),
            state.pending_requests.len_for_test(),
            "active request must have exactly one dispatch tick"
        );
    }

    /// Asserts no active request is already expired at the current tick.
    /// This proves retry processing refreshes dispatch age for all active work.
    fn assert_no_active_request_is_expired(state: &State) {
        for dispatch_tick in state.pending_requests.dispatch_ticks_for_test() {
            assert!(
                !state.tick.is_expired_since(dispatch_tick),
                "active request must not remain expired after a tick"
            );
        }
    }

    // Corrupts state by inserting the finalized hash into recent blocks.
    // This keeps the invariant checker enforcing the finalized/recent-state boundary.
    #[test]
    #[should_panic(expected = "finalized block must not be present in recent blocks")]
    fn state_invariants_reject_finalized_block_in_recent_blocks() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let mut state = empty_state_at(finalized_hash);

        state.blocks.0.insert(
            finalized_hash,
            block_with_parent(BlockHash::with_last_byte(0)),
        );

        assert_state_invariants(&state);
    }

    // Corrupts state with a direct self-parent edge.
    // This ensures invariant checks catch the smallest parent cycle before schedulers traverse it.
    #[test]
    #[should_panic(expected = "block must not reference itself as parent")]
    fn state_invariants_reject_self_parent_block() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let mut state = empty_state_at(finalized_hash);

        state.canonical_tip = block_hash;
        state
            .blocks
            .0
            .insert(block_hash, block_with_parent(block_hash));

        assert_state_invariants(&state);
    }

    // Points the canonical tip at an absent block.
    // This keeps invariant checks from accepting scheduler state that cannot be inspected.
    #[test]
    #[should_panic(expected = "canonical tip must be finalized or present in recent blocks")]
    fn state_invariants_reject_unknown_canonical_tip() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let mut state = empty_state_at(finalized_hash);

        state.canonical_tip = BlockHash::with_last_byte(2);

        assert_state_invariants(&state);
    }

    // Creates a two-block parent cycle.
    // This ensures graph invariants reject ancestry that would make walks non-terminating.
    #[test]
    #[should_panic(expected = "parent walk must not cycle")]
    fn state_invariants_reject_parent_cycle() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let first_hash = BlockHash::with_last_byte(2);
        let second_hash = BlockHash::with_last_byte(3);
        let mut state = empty_state_at(finalized_hash);

        state
            .blocks
            .0
            .insert(first_hash, block_with_parent(second_hash));
        state
            .blocks
            .0
            .insert(second_hash, block_with_parent(first_hash));

        assert_state_invariants(&state);
    }

    // Observes a malformed head whose parent is itself.
    // This preserves the recovery behavior of fetching the header without accepting unsafe graph state.
    #[test]
    fn head_observed_with_self_parent_fetches_header_without_changing_state() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let head_hash = BlockHash::with_last_byte(2);
        let state = empty_state_at(finalized_hash);

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: head_hash,
                parent_hash: head_hash,
            },
        );

        assert!(next_state.blocks.0.is_empty());
        assert_eq!(next_state.canonical_tip, finalized_hash);
        let request_id = assert_single_block_header_request_effect(&effects, head_hash);
        assert!(next_state.pending_requests.contains(&request_id));
        assert_state_invariants(&next_state);
    }

    // Observes a known block with a conflicting parent.
    // This ensures contradictory ancestry resets volatile chain state instead of merging histories.
    #[test]
    fn head_observed_with_conflicting_parent_resets_to_initial_finalized_state() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let head_hash = BlockHash::with_last_byte(2);
        let original_parent_hash = finalized_hash;
        let conflicting_parent_hash = BlockHash::with_last_byte(3);
        let mut state = empty_state_at(finalized_hash);

        state.canonical_tip = head_hash;
        state
            .blocks
            .0
            .insert(head_hash, block_with_parent(original_parent_hash));

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: head_hash,
                parent_hash: conflicting_parent_hash,
            },
        );

        assert_chain_reset_at(&next_state, finalized_hash);
        assert!(next_state.pending_requests.is_empty_for_test());
        assert!(effects.is_empty());
        assert_state_invariants(&next_state);
    }

    // Forces a reset with verified and rejected pool registry entries present.
    // This protects immutable pool metadata from being discarded with volatile blocks.
    #[test]
    fn chain_reset_preserves_pool_registry() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let head_hash = BlockHash::with_last_byte(2);
        let original_parent_hash = finalized_hash;
        let conflicting_parent_hash = BlockHash::with_last_byte(3);
        let verified_candidate = pool_candidate_address(4);
        let rejected_candidate = pool_candidate_address(5);
        let metadata = pool_metadata(6, 7, UniswapV3Fee::Fee3000);
        let mut state = empty_state_at(finalized_hash);

        state.pool_registry = TrustedPoolRegistry::new().with_metadata_results(
            ChainKey::Ethereum,
            HashMap::from([
                (verified_candidate, Ok(metadata.clone())),
                (
                    rejected_candidate,
                    Err(PoolMetadataFailure::FactoryReturnedZero),
                ),
            ]),
        );
        state.canonical_tip = head_hash;
        state
            .blocks
            .0
            .insert(head_hash, block_with_parent(original_parent_hash));

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: head_hash,
                parent_hash: conflicting_parent_hash,
            },
        );

        assert_chain_reset_at(&next_state, finalized_hash);
        // A reset issues no reconstruction requests; the preserved verified pool is now uncovered, so
        // the only follow-up is the background backfill snapshotting it at the (finalized) tip.
        assert_no_priority_effects(&effects);
        assert_eq!(
            next_state
                .pool_registry
                .verified_metadata(PoolRef { key: verified_candidate, chain: ChainKey::Ethereum }),
            Some(&metadata)
        );
        assert!(next_state.pool_registry.is_rejected(rejected_candidate));
        assert_state_invariants(&next_state);
    }

    // Forces a reset with token registry entries present.
    // This protects token decimals and terminal token failures because they are independent of recent reorg state.
    #[test]
    fn chain_reset_preserves_token_registry() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let head_hash = BlockHash::with_last_byte(2);
        let original_parent_hash = finalized_hash;
        let conflicting_parent_hash = BlockHash::with_last_byte(3);
        let verified_token = token_address(4);
        let unsupported_token = token_address(5);
        let metadata = token_metadata(6);
        let mut state = empty_state_at(finalized_hash);

        state.token_registry = TokenRegistry::new().with_metadata_results(HashMap::from([
            (verified_token, Ok(metadata.clone())),
            (
                unsupported_token,
                Err(TokenMetadataFailure::CallFailed(
                    TokenMetadataCall::Decimals,
                )),
            ),
        ]));
        state.canonical_tip = head_hash;
        state
            .blocks
            .0
            .insert(head_hash, block_with_parent(original_parent_hash));

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: head_hash,
                parent_hash: conflicting_parent_hash,
            },
        );

        assert_chain_reset_at(&next_state, finalized_hash);
        assert!(effects.is_empty());
        assert_eq!(
            next_state.token_registry.verified_metadata(verified_token),
            Some(&metadata)
        );
        assert!(next_state.token_registry.is_unsupported(unsupported_token));
        assert_state_invariants(&next_state);
    }

    // Activates kernel state from a bootstrap seed and reads back the finalized snapshots and registries.
    // This confirms the warm-start constructor carries finalized state and validated metadata into the kernel.
    #[test]
    fn activate_from_seed_exposes_finalized_snapshots_and_registries() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let candidate = pool_candidate_address(4);
        let pool = pool_address(4);
        let token0 = token_address(6);
        let token1 = token_address(7);
        let metadata = pool_metadata(6, 7, UniswapV3Fee::Fee3000);
        let snapshot = pool_state(9);

        let pool_registry = TrustedPoolRegistry::new().with_metadata_results(
            ChainKey::Ethereum,
            HashMap::from([(candidate, Ok(metadata.clone()))]),
        );
        let token_registry = TokenRegistry::new().with_metadata_results(HashMap::from([
            (token0, Ok(token_metadata(18))),
            (token1, Ok(token_metadata(6))),
        ]));

        let (state, effects) = State::activate_from_seed(
            finalized_hash,
            HashMap::from([(pool, snapshot.clone())]),
            pool_registry,
            token_registry,
            Vec::new(),
        );

        // No seed blocks, so nothing to reconnect to the anchor.
        assert!(effects.is_empty());
        assert_eq!(state.finalized_state.block_hash, finalized_hash);
        assert_eq!(
            state.finalized_pool_snapshots(),
            &HashMap::from([(pool, snapshot)])
        );
        assert_eq!(state.verified_pool_metadata(pool), Some(&metadata));
        assert_eq!(
            state.verified_token_metadata(token0),
            Some(&token_metadata(18))
        );
        assert_eq!(state.canonical_tip, finalized_hash);
        assert_state_invariants(&state);
    }

    // Seeds a recent block with resolved pool logs, then observes a head built on top of it.
    // This proves the seeded logs and registries are honored: only the new head needs a log fetch,
    // and the verified pool from the seeded block is immediately scheduled for pool data.
    #[test]
    fn activate_from_seed_seeds_resolved_logs_so_new_head_skips_log_refetch() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let seed_hash = BlockHash::with_last_byte(2);
        let new_head = BlockHash::with_last_byte(3);
        let candidate = pool_candidate_address(4);
        let pool = pool_address(4);
        let metadata = pool_metadata(6, 7, UniswapV3Fee::Fee3000);

        let pool_registry = TrustedPoolRegistry::new().with_metadata_results(
            ChainKey::Ethereum,
            HashMap::from([(candidate, Ok(metadata))]),
        );
        let token_registry = TokenRegistry::new().with_metadata_results(HashMap::from([
            (token_address(6), Ok(token_metadata(18))),
            (token_address(7), Ok(token_metadata(6))),
        ]));

        let (state, activation_effects) = State::activate_from_seed(
            finalized_hash,
            HashMap::new(),
            pool_registry,
            token_registry,
            vec![(seed_hash, finalized_hash, HashSet::from([candidate]))],
        );

        // The seed block's parent is the finalized anchor, so nothing needs reconnecting.
        assert!(activation_effects.is_empty());

        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: new_head,
                parent_hash: seed_hash,
            },
        );

        // The seeded block already has resolved logs, so no log request targets it.
        let seed_log_requests = effects
            .iter()
            .filter(|effect| {
                matches!(
                    effect,
                    Effect::Request(AnyIssuedRequest::BlockLogs(IssuedRequest {
                        request_payload: GetBlockLogs { block_hash },
                        ..
                    })) if *block_hash == seed_hash
                )
            })
            .count();
        assert_eq!(seed_log_requests, 0);

        // Only the new head needs its logs fetched.
        assert_single_block_log_request_effect(&effects, new_head);
        // The verified pool from the seeded block's resolved logs is scheduled at the tip.
        assert_single_pool_data_request_effect(&effects, new_head, &HashSet::from([pool]));
        assert_eq!(state.canonical_tip, new_head);
        assert_state_invariants(&state);
    }

    // A seed whose real block sits above a no-log gap is bridged by a filler (empty candidates)
    // parented to the anchor. The kernel treats the filler as an ordinary resolved, log-free block:
    // activation issues no `GetBlockHeader`, and a head built on the seed connects straight through
    // the filler down to the anchor.
    #[test]
    fn activate_from_seed_with_filler_bridges_gap_without_header_requests() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let filler = BlockHash::with_last_byte(9);
        let seed_hash = BlockHash::with_last_byte(2);
        let new_head = BlockHash::with_last_byte(3);
        let candidate = pool_candidate_address(4);
        let metadata = pool_metadata(6, 7, UniswapV3Fee::Fee3000);

        let pool_registry = TrustedPoolRegistry::new().with_metadata_results(
            ChainKey::Ethereum,
            HashMap::from([(candidate, Ok(metadata))]),
        );
        let token_registry = TokenRegistry::new().with_metadata_results(HashMap::from([
            (token_address(6), Ok(token_metadata(18))),
            (token_address(7), Ok(token_metadata(6))),
        ]));

        let (state, activation_effects) = State::activate_from_seed(
            finalized_hash,
            HashMap::new(),
            pool_registry,
            token_registry,
            vec![
                // Filler bridging the gap to the anchor, then the real block parented to the filler.
                (filler, finalized_hash, HashSet::new()),
                (seed_hash, filler, HashSet::from([candidate])),
            ],
        );

        // The graph is fully connected, so no ancestor needs fetching.
        assert!(activation_effects.is_empty());

        let (state, _effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: new_head,
                parent_hash: seed_hash,
            },
        );

        assert_eq!(state.canonical_tip, new_head);
        // head -> seed -> filler -> anchor: three connected blocks above the finalized anchor.
        assert_eq!(state.canonical_path_len_from_finalized(), Some(3));
        assert_state_invariants(&state);
    }

    // A reorg whose new branch re-enters a bridged gap is handled normally: the new branch's real
    // blocks are materialized and connect to the anchor, while the filler-bridged old branch is left
    // orphaned (pruned only at finalization) — no reset, because a filler's synthetic hash is never
    // re-inserted or mistaken for a real one.
    #[test]
    fn reorg_into_bridged_gap_materializes_new_branch_without_reset() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let filler = BlockHash::with_last_byte(9);
        let seed_hash = BlockHash::with_last_byte(2);
        let old_head = BlockHash::with_last_byte(3);
        let new_head = BlockHash::with_last_byte(5);
        let new_gap_block = BlockHash::with_last_byte(6);
        let candidate = pool_candidate_address(4);
        let metadata = pool_metadata(6, 7, UniswapV3Fee::Fee3000);

        let pool_registry = TrustedPoolRegistry::new().with_metadata_results(
            ChainKey::Ethereum,
            HashMap::from([(candidate, Ok(metadata))]),
        );
        let token_registry = TokenRegistry::new().with_metadata_results(HashMap::from([
            (token_address(6), Ok(token_metadata(18))),
            (token_address(7), Ok(token_metadata(6))),
        ]));

        // Seed: anchor <- filler <- seed_hash, then a head built on the seed.
        let (state, _effects) = State::activate_from_seed(
            finalized_hash,
            HashMap::new(),
            pool_registry,
            token_registry,
            vec![
                (filler, finalized_hash, HashSet::new()),
                (seed_hash, filler, HashSet::from([candidate])),
            ],
        );
        let (state, _effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: old_head,
                parent_hash: seed_hash,
            },
        );
        assert_eq!(state.canonical_tip, old_head);

        // Reorg: a competing head whose parent is a real block inside the formerly-bridged region.
        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: new_head,
                parent_hash: new_gap_block,
            },
        );
        // The new branch's missing real ancestor is fetched — the graph grew, it did not reset.
        let header_request_id = assert_single_block_header_request_effect(&effects, new_gap_block);
        assert_eq!(state.canonical_tip, new_head);

        // The real gap block resolves, forking at the anchor: the new branch connects through.
        let (state, _effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::BlockHeaderReceived {
                logs_bloom: bloom_matching_any(),
                request_id: header_request_id,
                hash: new_gap_block,
                parent_hash: finalized_hash,
            },
        );

        assert_eq!(state.canonical_tip, new_head);
        // new_head -> new_gap_block -> anchor: the winning branch is real, with no filler.
        assert_eq!(state.canonical_path_len_from_finalized(), Some(2));
        // The filler and the old branch survive as an orphan (no reset would have kept them).
        assert!(state.blocks.0.contains_key(&filler));
        assert!(state.blocks.0.contains_key(&seed_hash));
        assert!(state.blocks.0.contains_key(&old_head));
        assert_state_invariants(&state);
    }

    // ===== Bootstrap-activation invariant coverage =====
    // These exercises feed real `bootstrap` outcomes through `activate_from_seed` (mirroring
    // `multi_chain_kernel::activate_bootstrap_outcome`) and hold the resulting kernel state to
    // every kernel invariant — the core shape set plus the scheduler-level guarantees — so a
    // warm-started chain is never weaker than a cold one. The goal: any state reachable from a
    // `bootstrap::Completion::Ready` outcome satisfies all kernel state invariants.

    /// The finalized anchor every bootstrap-activation exercise shares. Its hash byte (220) stays
    /// clear of the per-number block hashes (derived from numbers just above 100) so fixtures
    /// never collide.
    fn bootstrap_anchor() -> bootstrap::FinalizedAnchor {
        bootstrap::FinalizedAnchor {
            hash: BlockHash::with_last_byte(220),
            number: 100,
        }
    }

    /// Compact, collision-free block hash for a canonical block at `number`.
    fn block_hash_for_number(number: u64) -> BlockHash {
        BlockHash::with_last_byte(number as u8)
    }

    /// The single pool-event candidate a block at `number` contributes to its range-logs entry.
    fn candidate_for_number(number: u64) -> ProtocolPoolKey {
        pool_candidate_address(number as u8)
    }

    /// A one-block range-logs entry, as the bootstrap candidate scan would report it.
    fn range_log_block(number: u64) -> RangeLogBlock {
        RangeLogBlock {
            number,
            hash: block_hash_for_number(number),
            candidates: HashSet::from([candidate_for_number(number)]),
        }
    }

    /// Low byte of an address, used to key the deterministic success/failure fixtures.
    fn address_last_byte(address: &Address) -> u8 {
        address.into_array().into_iter().next_back().unwrap_or(0)
    }

    /// The single outstanding bootstrap request carried by a transition, if any.
    fn bootstrap_outstanding(
        effects: Vec<bootstrap::Effect>,
    ) -> Option<bootstrap::AnyIssuedRequest> {
        effects
            .into_iter()
            .map(|bootstrap::Effect::Request(issued)| issued)
            .next()
    }

    /// A well-formed response answering whatever bootstrap request is outstanding, with metadata and
    /// token results decided by the caller-supplied failing-byte sets.
    fn bootstrap_response(
        issued: &bootstrap::AnyIssuedRequest,
        window: &[RangeLogBlock],
        metadata_fails: &HashSet<u8>,
        token_fails: &HashSet<u8>,
    ) -> bootstrap::Event {
        match issued {
            bootstrap::AnyIssuedRequest::FinalizedHeader(request) => {
                bootstrap::Event::FinalizedHeaderReceived {
                    request_id: request.request_id,
                    anchor: bootstrap_anchor(),
                }
            }
            bootstrap::AnyIssuedRequest::PoolCandidates(request) => {
                bootstrap::Event::PoolCandidatesReceived {
                    request_id: request.request_id,
                    blocks: window.to_vec(),
                }
            }
            bootstrap::AnyIssuedRequest::PoolMetadata(request) => {
                let metadata = request
                    .request_payload
                    .candidates
                    .iter()
                    .map(|candidate| {
                        let byte = address_last_byte(&candidate.uniswap_v3_address().expect("v3 pool"));
                        let result = if metadata_fails.contains(&byte) {
                            Err(PoolMetadataFailure::FactoryReturnedZero)
                        } else {
                            Ok(pool_metadata(
                                byte,
                                byte.wrapping_add(64),
                                UniswapV3Fee::Fee3000,
                            ))
                        };
                        (*candidate, result)
                    })
                    .collect();
                bootstrap::Event::PoolMetadataReceived {
                    request_id: request.request_id,
                    metadata,
                }
            }
            bootstrap::AnyIssuedRequest::TokenMetadata(request) => {
                let metadata = request
                    .request_payload
                    .tokens
                    .iter()
                    .map(|token| {
                        let result = if token_fails.contains(&address_last_byte(&token.0)) {
                            Err(TokenMetadataFailure::CallFailed(
                                TokenMetadataCall::Decimals,
                            ))
                        } else {
                            Ok(token_metadata(18))
                        };
                        (*token, result)
                    })
                    .collect();
                bootstrap::Event::TokenMetadataReceived {
                    request_id: request.request_id,
                    metadata,
                }
            }
        }
    }

    /// Drives a real bootstrap from `init` to its terminal completion. It answers each outstanding
    /// request, but stops after `deliver_limit` deliveries and ticks to the deadline instead, so
    /// partial runs surface the *degraded* best-effort outcomes the kernel must also tolerate.
    fn drive_bootstrap_to_completion(
        policy: bootstrap::BootstrapPolicy,
        window: &[RangeLogBlock],
        metadata_fails: &HashSet<u8>,
        token_fails: &HashSet<u8>,
        deliver_limit: usize,
    ) -> bootstrap::Completion {
        let (mut state, effects) = bootstrap::init(policy);
        let mut outstanding = bootstrap_outstanding(effects);
        let mut delivered = 0;

        loop {
            if let Some(completion) = bootstrap::completion(&state) {
                return completion;
            }

            let event = match outstanding.as_ref() {
                Some(request) if delivered < deliver_limit => {
                    delivered += 1;
                    bootstrap_response(request, window, metadata_fails, token_fails)
                }
                _ => bootstrap::Event::Tick,
            };

            let (next_state, effects) = bootstrap::transition(ChainKey::Ethereum, state, event);
            state = next_state;
            if let Some(request) = bootstrap_outstanding(effects) {
                outstanding = Some(request);
            }
        }
    }

    /// Activates a kernel state from a bootstrap outcome exactly as `multi_chain_kernel` does,
    /// returning the activation effects so tests can hold them to the effect well-formedness check.
    fn activate_bootstrap_outcome_for_test(
        outcome: bootstrap::BootstrapOutcome,
    ) -> (State, Vec<Effect>) {
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

        State::activate_from_seed(
            anchor.hash,
            pool_snapshots,
            pool_registry,
            token_registry,
            seed_blocks,
        )
    }

    /// Asserts every continuously-held kernel invariant: the core shape set plus the
    /// scheduler-level guarantees `assert_state_invariants` omits but a healthy state always meets.
    fn assert_all_kernel_invariants(state: &State) {
        assert_state_invariants(state);
        assert_canonical_unknown_logs_are_pending(state);
        assert_canonical_resolved_candidates_are_known_or_pending(state);
        assert_missing_parents_for_known_blocks_are_pending(state);
    }

    /// A fresh head hash for building on top of a seeded graph, distinct from every fixture hash.
    fn fresh_head_hash() -> BlockHash {
        BlockHash::with_last_byte(250)
    }

    /// Asserts the transition scheduled no priority (reconstruction/freshness) work. The only follow-up
    /// permitted once the pending tier is idle is the background registry pool-state backfill, which is
    /// always a `GetPoolData` request — so any other effect indicates unexpected priority scheduling.
    fn assert_no_priority_effects(effects: &[Effect]) {
        for effect in effects {
            assert!(
                matches!(effect, Effect::Request(AnyIssuedRequest::PoolData(_))),
                "expected only background backfill pool-data effects"
            );
        }
    }

    #[test]
    fn idle_transition_backfills_uncovered_verified_pools_one_chunk_at_a_time() {
        let finalized = BlockHash::with_last_byte(1);
        let candidate = pool_candidate_address(3);
        let mut state = empty_state_at(finalized);
        state.pool_registry = registry_verifying(candidate);

        // Idle pending tier: a tick (no priority work) issues exactly one backfill request covering the
        // uncovered verified pool, snapshotted at the (finalized) tip.
        let (state, effects) = transition(ChainKey::Ethereum, state, Event::Tick);
        let request = match effects.as_slice() {
            [Effect::Request(AnyIssuedRequest::PoolData(request))] => request,
            _ => panic!("expected exactly one backfill pool-data request"),
        };
        assert_eq!(request.request_payload.at, finalized);
        assert!(request.request_payload.pools.contains(&pool_address(3)));

        // The chunk is now pending, so a second idle tick issues no further backfill (sequential).
        let (_state, effects) = transition(ChainKey::Ethereum, state, Event::Tick);
        assert!(effects.is_empty());
    }

    #[test]
    fn backfill_is_suppressed_while_a_request_is_pending() {
        let finalized = BlockHash::with_last_byte(1);
        let candidate = pool_candidate_address(3);
        let mut state = empty_state_at(finalized);
        state.pool_registry = registry_verifying(candidate);

        // A fresh (non-expired) request occupies the priority tier; the tick must not start a backfill.
        let (pending_requests, _request_id) = state.pending_requests.with_new_request(
            GetBlockHeader {
                block_hash: BlockHash::with_last_byte(50),
            },
            state.tick,
        );
        state.pending_requests = pending_requests;

        let (_state, effects) = transition(ChainKey::Ethereum, state, Event::Tick);
        assert!(effects.is_empty());
    }

    // Positive control: a clean full bootstrap (contiguous window, all calls succeed) activates
    // into a state that satisfies every invariant, both at activation and after a head builds on
    // the seeded tip. If this fails, the harness or the bridge mapping is wrong, not the logic.
    #[test]
    fn clean_full_bootstrap_activation_satisfies_all_invariants() {
        let policy = bootstrap::BootstrapPolicy {
            tip_trim: 0,
            deadline_ticks: 100,
        };
        let window = vec![
            range_log_block(101),
            range_log_block(102),
            range_log_block(103),
        ];
        let outcome = match drive_bootstrap_to_completion(
            policy,
            &window,
            &HashSet::new(),
            &HashSet::new(),
            5,
        ) {
            bootstrap::Completion::Ready(outcome) => outcome,
            other => panic!("expected ready outcome, got {other:?}"),
        };

        let seed_tip = outcome
            .seed_blocks
            .last()
            .expect("contiguous window seeds blocks")
            .hash;
        let (state, activation_effects) = activate_bootstrap_outcome_for_test(outcome);
        assert_all_kernel_invariants(&state);
        assert_effects_are_well_formed(&state, &activation_effects);

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: fresh_head_hash(),
                parent_hash: seed_tip,
            },
        );
        assert_all_kernel_invariants(&next_state);
        assert_effects_are_well_formed(&next_state, &effects);
    }

    // A bootstrap that hits its deadline while validating pools degrades to a Ready outcome that
    // carries seed blocks but an empty pool registry. Once such a seeded block joins the canonical
    // path, its resolved log candidates must still be known or pending validation — otherwise the
    // pools in that block are silently orphaned and never validated.
    #[test]
    fn degraded_bootstrap_before_pool_metadata_keeps_canonical_candidates_known_or_pending() {
        let policy = bootstrap::BootstrapPolicy {
            tip_trim: 0,
            deadline_ticks: 3,
        };
        let window = vec![range_log_block(101), range_log_block(102)];
        // Deliver the finalized header and pool candidates, then stall to the deadline.
        let outcome = match drive_bootstrap_to_completion(
            policy,
            &window,
            &HashSet::new(),
            &HashSet::new(),
            2,
        ) {
            bootstrap::Completion::Ready(outcome) => outcome,
            other => panic!("expected degraded ready outcome, got {other:?}"),
        };

        // Precondition: this is the degraded shape — seeded blocks, but no validated pools.
        assert!(!outcome.seed_blocks.is_empty());
        assert_eq!(outcome.pool_registry, TrustedPoolRegistry::new());

        let seed_tip = outcome
            .seed_blocks
            .last()
            .expect("seed blocks present")
            .hash;
        let (state, _activation_effects) = activate_bootstrap_outcome_for_test(outcome);
        let (next_state, _effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: fresh_head_hash(),
                parent_hash: seed_tip,
            },
        );

        assert_canonical_resolved_candidates_are_known_or_pending(&next_state);
    }

    // A no-log block leaves a gap in the range-logs window. Rather than dropping the blocks above it
    // (which would force a per-block header walk to reconnect them), the bootstrap bridges the gap
    // with a single synthetic filler, so the activated graph is fully connected and the deep
    // header-walk never starts.
    #[test]
    fn bootstrap_log_gap_is_bridged_by_a_filler_without_header_requests() {
        let policy = bootstrap::BootstrapPolicy {
            tip_trim: 0,
            deadline_ticks: 100,
        };
        // Block 103 has no pool logs (absent from the window), leaving a gap between 102 and 104.
        let window = vec![
            range_log_block(101),
            range_log_block(102),
            range_log_block(104),
            range_log_block(105),
        ];
        let outcome = match drive_bootstrap_to_completion(
            policy,
            &window,
            &HashSet::new(),
            &HashSet::new(),
            5,
        ) {
            bootstrap::Completion::Ready(outcome) => outcome,
            other => panic!("expected ready outcome, got {other:?}"),
        };

        // Every seed block links to a seeded ancestor or the anchor: nothing floats.
        let seeded = outcome
            .seed_blocks
            .iter()
            .map(|block| block.hash)
            .collect::<HashSet<_>>();
        let has_floating_parent = outcome.seed_blocks.iter().any(|block| {
            !seeded.contains(&block.parent_hash) && block.parent_hash != bootstrap_anchor().hash
        });
        assert!(
            !has_floating_parent,
            "the gap must be bridged, leaving no floating parent"
        );

        // The bridge is a filler: a seeded block with no candidates (real blocks each contribute one).
        let filler_count = outcome
            .seed_blocks
            .iter()
            .filter(|block| block.candidates.is_empty())
            .count();
        assert_eq!(
            filler_count, 1,
            "the single gap is bridged by exactly one filler"
        );

        let (state, activation_effects) = activate_bootstrap_outcome_for_test(outcome);
        // The deep header-walk never starts: a fully connected seed needs no header fetch.
        assert!(
            activation_effects.is_empty(),
            "a fully connected seed issues no header request"
        );
        assert_missing_parents_for_known_blocks_are_pending(&state);
        assert_effects_are_well_formed(&state, &activation_effects);
    }

    proptest! {
        // Every kernel invariant holds on the state activated from any Ready bootstrap outcome —
        // full or degraded, with arbitrary log gaps, tip trimming, and metadata/token/data results.
        #[test]
        fn bootstrap_activation_satisfies_all_kernel_invariants(
            offsets in prop::collection::hash_set(1u64..40, 0..16),
            metadata_fails in prop::collection::hash_set(any::<u8>(), 0..16),
            token_fails in prop::collection::hash_set(any::<u8>(), 0..16),
            tip_trim in 0usize..6,
            deliver_limit in 0usize..6,
        ) {
            let anchor_number = bootstrap_anchor().number;
            let window = offsets
                .iter()
                .map(|offset| range_log_block(anchor_number + offset))
                .collect::<Vec<_>>();
            let policy = bootstrap::BootstrapPolicy {
                tip_trim,
                deadline_ticks: 4,
            };

            if let bootstrap::Completion::Ready(outcome) = drive_bootstrap_to_completion(
                policy,
                &window,
                &metadata_fails,
                &token_fails,
                deliver_limit,
            ) {
                let (state, activation_effects) = activate_bootstrap_outcome_for_test(outcome);
                assert_all_kernel_invariants(&state);
                assert_effects_are_well_formed(&state, &activation_effects);
            }
        }

        // Every kernel invariant still holds after a new head builds on the seeded tip, bringing
        // the seed blocks onto the canonical path where the scheduler invariants bite. Emitted
        // effects must also be well-formed (recorded as pending work).
        #[test]
        fn bootstrap_activation_then_head_satisfies_all_kernel_invariants(
            offsets in prop::collection::hash_set(1u64..40, 0..16),
            metadata_fails in prop::collection::hash_set(any::<u8>(), 0..16),
            token_fails in prop::collection::hash_set(any::<u8>(), 0..16),
            tip_trim in 0usize..6,
            deliver_limit in 0usize..6,
        ) {
            let anchor_number = bootstrap_anchor().number;
            let window = offsets
                .iter()
                .map(|offset| range_log_block(anchor_number + offset))
                .collect::<Vec<_>>();
            let policy = bootstrap::BootstrapPolicy {
                tip_trim,
                deadline_ticks: 4,
            };

            if let bootstrap::Completion::Ready(outcome) = drive_bootstrap_to_completion(
                policy,
                &window,
                &metadata_fails,
                &token_fails,
                deliver_limit,
            ) {
                let seed_tip = outcome.seed_blocks.last().map(|block| block.hash);
                let (state, activation_effects) = activate_bootstrap_outcome_for_test(outcome);
                assert_all_kernel_invariants(&state);
                assert_effects_are_well_formed(&state, &activation_effects);

                if let Some(seed_tip) = seed_tip {
                    let (next_state, effects) = transition(ChainKey::Ethereum,
                        state,
                        Event::HeadObserved {
                            logs_bloom: bloom_matching_any(),
                            hash: fresh_head_hash(),
                            parent_hash: seed_tip,
                        },
                    );
                    assert_all_kernel_invariants(&next_state);
                    assert_effects_are_well_formed(&next_state, &effects);
                }
            }
        }
    }

    // Reobserves an already tracked head with the same parent.
    // This keeps block graph updates idempotent while still allowing follow-up scheduling.
    #[test]
    fn head_observed_with_duplicate_matching_block_does_not_change_state() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let head_hash = BlockHash::with_last_byte(2);
        let mut state = empty_state_at(finalized_hash);

        state.canonical_tip = head_hash;
        state
            .blocks
            .0
            .insert(head_hash, block_with_parent(finalized_hash));

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: head_hash,
                parent_hash: finalized_hash,
            },
        );

        assert_eq!(next_state.blocks.0.len(), 1);
        assert_eq!(next_state.canonical_tip, head_hash);
        assert_single_unknown_block(&next_state, head_hash, finalized_hash);
        assert_request_hashes(&effects, HashSet::new(), HashSet::from([head_hash]));
        assert_state_invariants(&next_state);
    }

    // Observes a head that would close an ancestry cycle.
    // This keeps cycle detection on the reset path instead of retaining corrupt graph state.
    #[test]
    fn head_observed_that_introduces_cycle_resets_to_initial_finalized_state() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let first_hash = BlockHash::with_last_byte(2);
        let second_hash = BlockHash::with_last_byte(3);
        let cycle_hash = BlockHash::with_last_byte(4);
        let mut state = empty_state_at(finalized_hash);

        state.canonical_tip = second_hash;
        state
            .blocks
            .0
            .insert(first_hash, block_with_parent(cycle_hash));
        state
            .blocks
            .0
            .insert(second_hash, block_with_parent(first_hash));

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: cycle_hash,
                parent_hash: second_hash,
            },
        );

        assert_chain_reset_at(&next_state, finalized_hash);
        assert!(next_state.pending_requests.is_empty_for_test());
        assert!(effects.is_empty());
        assert_state_invariants(&next_state);
    }

    // Observes the finalized hash as a new head.
    // This prevents finalized and recent storage from representing the same block.
    #[test]
    fn head_observed_with_finalized_hash_does_not_insert_finalized_block() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let parent_hash = BlockHash::with_last_byte(2);
        let state = empty_state_at(finalized_hash);

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: finalized_hash,
                parent_hash,
            },
        );

        assert_empty_initial_state_at(&next_state, finalized_hash);
        assert!(effects.is_empty());
        assert_state_invariants(&next_state);
    }

    // Observes a head directly on finalized.
    // This ensures connected canonical heads immediately start block-log discovery.
    #[test]
    fn connected_head_observed_requests_logs_for_head() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let head_hash = BlockHash::with_last_byte(2);
        let state = empty_state_at(finalized_hash);

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: head_hash,
                parent_hash: finalized_hash,
            },
        );

        assert_request_hashes(&effects, HashSet::new(), HashSet::from([head_hash]));
        assert_effects_are_well_formed(&next_state, &effects);
        assert_state_invariants(&next_state);
    }

    // Observes a head with an unknown parent.
    // This ensures the kernel fetches both head logs and the missing parent header.
    #[test]
    fn disconnected_head_observed_requests_head_logs_and_missing_parent_header() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let state = empty_state_at(finalized_hash);

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: head_hash,
                parent_hash: missing_parent_hash,
            },
        );

        assert_request_hashes(
            &effects,
            HashSet::from([missing_parent_hash]),
            HashSet::from([head_hash]),
        );
        assert_effects_are_well_formed(&next_state, &effects);
        assert_state_invariants(&next_state);
    }

    // Two heads that resolve to the same missing parent must fetch that parent once, not twice.
    // This pins the dedup guard that stops redundant ancestry header requests on sparse chains.
    #[test]
    fn second_head_sharing_missing_parent_does_not_reissue_header_request() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let shared_missing_parent = BlockHash::with_last_byte(2);
        let first_head = BlockHash::with_last_byte(3);
        let second_head = BlockHash::with_last_byte(4);
        let state = empty_state_at(finalized_hash);

        let (state, first_effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: first_head,
                parent_hash: shared_missing_parent,
            },
        );
        // First observation fetches the missing parent.
        assert_single_block_header_request_effect(&first_effects, shared_missing_parent);

        let (next_state, second_effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: second_head,
                parent_hash: shared_missing_parent,
            },
        );

        // Second observation reuses the in-flight request: it schedules the head's logs but no header.
        assert_request_hashes(
            &second_effects,
            HashSet::new(),
            HashSet::from([second_head]),
        );
        assert_eq!(
            pending_header_hashes(&next_state),
            HashSet::from([shared_missing_parent])
        );
        assert_eq!(
            next_state
                .pending_requests
                .pending_header_request_count_for_test(),
            1
        );
        assert_no_duplicate_pending_header_requests(&next_state);
        assert_effects_are_well_formed(&next_state, &second_effects);
        assert_state_invariants(&next_state);
    }

    // A distinct missing parent is still fetched even while another header request is in flight.
    // This proves the guard keys on the specific hash, not on "any header pending".
    #[test]
    fn head_with_distinct_missing_parent_still_requests_header_while_another_is_pending() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let first_missing_parent = BlockHash::with_last_byte(2);
        let first_head = BlockHash::with_last_byte(3);
        let second_missing_parent = BlockHash::with_last_byte(4);
        let second_head = BlockHash::with_last_byte(5);
        let state = empty_state_at(finalized_hash);

        let (state, _first_effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: first_head,
                parent_hash: first_missing_parent,
            },
        );

        let (next_state, second_effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: second_head,
                parent_hash: second_missing_parent,
            },
        );

        // A different missing parent is not suppressed by the unrelated in-flight header.
        assert_request_hashes(
            &second_effects,
            HashSet::from([second_missing_parent]),
            HashSet::from([second_head]),
        );
        assert_eq!(
            pending_header_hashes(&next_state),
            HashSet::from([first_missing_parent, second_missing_parent])
        );
        assert_no_duplicate_pending_header_requests(&next_state);
        assert_effects_are_well_formed(&next_state, &second_effects);
        assert_state_invariants(&next_state);
    }

    // Delivers the requested parent header for a disconnected head.
    // This ensures newly connected ancestors enter log discovery.
    #[test]
    fn missing_parent_header_received_requests_logs_for_parent() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: head_hash,
                parent_hash: missing_parent_hash,
            },
        );
        let header_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::BlockHeaderReceived {
                logs_bloom: bloom_matching_any(),
                request_id: header_request_id,
                hash: missing_parent_hash,
                parent_hash: finalized_hash,
            },
        );

        assert_request_hashes(
            &effects,
            HashSet::new(),
            HashSet::from([missing_parent_hash]),
        );
        assert_effects_are_well_formed(&next_state, &effects);
        assert_state_invariants(&next_state);
    }

    // Starts with a fully present canonical chain whose logs are unknown.
    // This ensures the scheduler backfills log requests for every present canonical block.
    #[test]
    fn present_canonical_prefix_schedules_logs_for_all_unknown_blocks() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let grandparent_hash = BlockHash::with_last_byte(2);
        let parent_hash = BlockHash::with_last_byte(3);
        let head_hash = BlockHash::with_last_byte(4);
        let mut state = empty_state_at(finalized_hash);

        state.canonical_tip = head_hash;
        state
            .blocks
            .0
            .insert(grandparent_hash, block_with_parent(finalized_hash));
        state
            .blocks
            .0
            .insert(parent_hash, block_with_parent(grandparent_hash));
        state
            .blocks
            .0
            .insert(head_hash, block_with_parent(parent_hash));

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: head_hash,
                parent_hash,
            },
        );

        assert_request_hashes(
            &effects,
            HashSet::new(),
            HashSet::from([head_hash, parent_hash, grandparent_hash]),
        );
        assert_effects_are_well_formed(&next_state, &effects);
        assert_state_invariants(&next_state);
    }

    // Delivers the same header response twice.
    // This prevents late or duplicate responses from multiplying pending log work.
    #[test]
    fn duplicate_header_response_does_not_duplicate_pending_log_requests() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: head_hash,
                parent_hash,
            },
        );
        let header_request_id = assert_single_block_header_request_effect(&effects, parent_hash);

        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::BlockHeaderReceived {
                logs_bloom: bloom_matching_any(),
                request_id: header_request_id,
                hash: parent_hash,
                parent_hash: finalized_hash,
            },
        );
        assert_request_hashes(&effects, HashSet::new(), HashSet::from([parent_hash]));

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::BlockHeaderReceived {
                logs_bloom: bloom_matching_any(),
                request_id: header_request_id,
                hash: parent_hash,
                parent_hash: finalized_hash,
            },
        );

        assert!(effects.is_empty());
        assert_state_invariants(&next_state);
    }

    // Schedules with an existing log request for the same block.
    // This ensures pending work suppresses duplicate log RPCs.
    #[test]
    fn pending_log_request_suppresses_duplicate_log_request() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let head_hash = BlockHash::with_last_byte(2);
        let mut state = empty_state_at(finalized_hash);
        state.canonical_tip = head_hash;
        state
            .blocks
            .0
            .insert(head_hash, block_with_parent(finalized_hash));
        let (pending_requests, _request_id) = state.pending_requests.with_new_request(
            GetBlockLogs {
                block_hash: head_hash,
            },
            state.tick,
        );
        state.pending_requests = pending_requests;

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: head_hash,
                parent_hash: finalized_hash,
            },
        );

        assert!(effects.is_empty());
        assert_state_invariants(&next_state);
    }

    // Schedules after a block's logs are already resolved.
    // This ensures completed log fetches are not repeated without new block data.
    #[test]
    fn resolved_log_status_suppresses_duplicate_log_request() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let head_hash = BlockHash::with_last_byte(2);
        let mut state = empty_state_at(finalized_hash);
        state.canonical_tip = head_hash;
        state.blocks.0.insert(
            head_hash,
            BlockNode {
                logs_bloom: None,
                parent_hash: finalized_hash,
                pool_logs: PoolLogsStatus::Resolved(HashSet::new()),
                pool_snapshots: HashMap::new(),
                pool_data_failures: HashMap::new(),
            },
        );

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: head_hash,
                parent_hash: finalized_hash,
            },
        );

        assert!(effects.is_empty());
        assert_state_invariants(&next_state);
    }

    // Delivers logs for an active block-log request.
    // This ties successful log application to request retirement and candidate validation scheduling.
    #[test]
    fn block_logs_received_for_matching_request_marks_logs_and_removes_pending_request() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let first_candidate = pool_candidate_address(3);
        let second_candidate = pool_candidate_address(4);
        let logs = HashSet::from([first_candidate, second_candidate]);
        let mut state = empty_state_at(finalized_hash);

        state.canonical_tip = block_hash;
        state
            .blocks
            .0
            .insert(block_hash, block_with_parent(finalized_hash));
        let (pending_requests, request_id) = state
            .pending_requests
            .with_new_request(GetBlockLogs { block_hash }, state.tick);
        state.pending_requests = pending_requests;

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::BlockLogsReceived {
                request_id,
                logs: pool_logs(&logs),
            },
        );

        assert_single_pool_metadata_request_effect(&effects, block_hash, &logs);
        assert!(!next_state.pending_requests.contains(&request_id));
        assert_resolved_pool_logs(&next_state, block_hash, finalized_hash, &logs);
        assert_state_invariants(&next_state);
    }

    // Delivers an empty log response for an active request.
    // This ensures no-log blocks become resolved instead of remaining unknown.
    #[test]
    fn block_logs_received_with_empty_logs_marks_block_resolved() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let logs = HashSet::new();
        let mut state = empty_state_at(finalized_hash);

        state.canonical_tip = block_hash;
        state
            .blocks
            .0
            .insert(block_hash, block_with_parent(finalized_hash));
        let (pending_requests, request_id) = state
            .pending_requests
            .with_new_request(GetBlockLogs { block_hash }, state.tick);
        state.pending_requests = pending_requests;

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::BlockLogsReceived {
                request_id,
                logs: pool_logs(&logs),
            },
        );

        assert!(effects.is_empty());
        assert!(!next_state.pending_requests.contains(&request_id));
        assert_resolved_pool_logs(&next_state, block_hash, finalized_hash, &logs);
        assert_state_invariants(&next_state);
    }

    // Delivers logs with no active matching request.
    // This protects state from stale or unsolicited RPC responses.
    #[test]
    fn block_logs_received_for_unknown_request_is_noop() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let logs = HashSet::from([pool_candidate_address(3)]);
        let mut state = empty_state_at(finalized_hash);

        state.canonical_tip = block_hash;
        state
            .blocks
            .0
            .insert(block_hash, block_with_parent(finalized_hash));

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::BlockLogsReceived {
                request_id: RequestId::from_raw_for_test(99),
                logs: pool_logs(&logs),
            },
        );

        assert!(effects.is_empty());
        assert!(next_state.pending_requests.is_empty_for_test());
        assert_single_unknown_block(&next_state, block_hash, finalized_hash);
        assert_state_invariants(&next_state);
    }

    // Delivers a log response after the target block was removed by reset.
    // This prevents late responses from recreating volatile block state.
    #[test]
    fn block_logs_received_for_missing_block_consumes_request_without_inserting_block() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_block_hash = BlockHash::with_last_byte(2);
        let logs = HashSet::from([pool_candidate_address(3)]);
        let mut state = empty_state_at(finalized_hash);
        let (pending_requests, request_id) = state.pending_requests.with_new_request(
            GetBlockLogs {
                block_hash: missing_block_hash,
            },
            state.tick,
        );
        state.pending_requests = pending_requests;

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::BlockLogsReceived {
                request_id,
                logs: pool_logs(&logs),
            },
        );

        assert!(effects.is_empty());
        assert_empty_initial_state_at(&next_state, finalized_hash);
        assert!(!next_state.blocks.0.contains_key(&missing_block_hash));
        assert_state_invariants(&next_state);
    }

    // Resolves logs containing an unknown candidate.
    // This connects raw log discovery to pool metadata validation instead of trusting emitters immediately.
    #[test]
    fn block_logs_received_stores_candidates_and_requests_metadata_for_unknown_candidates() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let first_candidate = pool_candidate_address(3);
        let second_candidate = pool_candidate_address(4);
        let logs = HashSet::from([first_candidate, second_candidate]);
        let mut state = empty_state_at(finalized_hash);

        state.canonical_tip = block_hash;
        state
            .blocks
            .0
            .insert(block_hash, block_with_parent(finalized_hash));
        let (pending_requests, request_id) = state
            .pending_requests
            .with_new_request(GetBlockLogs { block_hash }, state.tick);
        state.pending_requests = pending_requests;

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::BlockLogsReceived {
                request_id,
                logs: pool_logs(&logs),
            },
        );

        assert!(!next_state.pending_requests.contains(&request_id));
        assert_resolved_candidate_logs(&next_state, block_hash, finalized_hash, &logs);
        let metadata_request_id =
            assert_single_pool_metadata_request_effect(&effects, block_hash, &logs);
        assert!(next_state.pending_requests.contains(&metadata_request_id));
        assert_state_invariants(&next_state);
    }

    // Resolves logs for a candidate already verified in the pool registry.
    // This avoids redundant metadata RPC and lets trusted pools proceed directly.
    #[test]
    fn known_verified_candidates_do_not_request_metadata() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let candidate = pool_candidate_address(3);
        let logs = HashSet::from([candidate]);
        let mut state = empty_state_at(finalized_hash);

        state.pool_registry = TrustedPoolRegistry::new().with_metadata_results(
            ChainKey::Ethereum,
            HashMap::from([(candidate, Ok(pool_metadata(1, 2, UniswapV3Fee::Fee3000)))]),
        );
        state.canonical_tip = block_hash;
        state
            .blocks
            .0
            .insert(block_hash, block_with_parent(finalized_hash));
        let (pending_requests, request_id) = state
            .pending_requests
            .with_new_request(GetBlockLogs { block_hash }, state.tick);
        state.pending_requests = pending_requests;

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::BlockLogsReceived {
                request_id,
                logs: pool_logs(&logs),
            },
        );

        let token_request_id = assert_single_token_metadata_request_effect(
            &effects,
            block_hash,
            &HashSet::from([token_address(1), token_address(2)]),
        );
        assert!(next_state.pending_requests.contains(&token_request_id));
        assert_trusted_pool_logs_resolved(
            &next_state,
            block_hash,
            HashSet::from([PoolRef { key: candidate, chain: ChainKey::Ethereum }]),
        );
        assert_state_invariants(&next_state);
    }

    // Resolves logs for a candidate already rejected by the pool registry.
    // This prevents retrying validation for deterministic non-pool emitters.
    #[test]
    fn known_rejected_candidates_do_not_request_metadata() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let candidate = pool_candidate_address(3);
        let logs = HashSet::from([candidate]);
        let mut state = empty_state_at(finalized_hash);

        state.pool_registry = TrustedPoolRegistry::new().with_metadata_results(
            ChainKey::Ethereum,
            HashMap::from([(candidate, Err(PoolMetadataFailure::FactoryReturnedZero))]),
        );
        state.canonical_tip = block_hash;
        state
            .blocks
            .0
            .insert(block_hash, block_with_parent(finalized_hash));
        let (pending_requests, request_id) = state
            .pending_requests
            .with_new_request(GetBlockLogs { block_hash }, state.tick);
        state.pending_requests = pending_requests;

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::BlockLogsReceived {
                request_id,
                logs: pool_logs(&logs),
            },
        );

        assert!(effects.is_empty());
        assert_eq!(
            next_state.blocks.trusted_pool_logs(
                ChainKey::Ethereum,
                block_hash,
                &next_state.pool_registry
            ),
            Some(TrustedPoolLogs::Resolved(HashSet::new()))
        );
        assert_state_invariants(&next_state);
    }

    // Resolves logs for a trusted pool with known token metadata.
    // This ensures affected pools start pool-data fetching immediately.
    #[test]
    fn block_logs_received_for_verified_pool_requests_pool_data_at_block() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let candidate = pool_candidate_address(3);
        let pool = PoolRef { key: candidate, chain: ChainKey::Ethereum };
        let mut state = empty_state_at(finalized_hash);

        state.pool_registry = TrustedPoolRegistry::new().with_metadata_results(
            ChainKey::Ethereum,
            HashMap::from([(candidate, Ok(pool_metadata(1, 2, UniswapV3Fee::Fee3000)))]),
        );
        state.token_registry = TokenRegistry::new().with_metadata_results(HashMap::from([
            (token_address(1), Ok(token_metadata(6))),
            (token_address(2), Ok(token_metadata(18))),
        ]));
        state.canonical_tip = block_hash;
        state
            .blocks
            .0
            .insert(block_hash, block_with_parent(finalized_hash));
        let (pending_requests, request_id) = state
            .pending_requests
            .with_new_request(GetBlockLogs { block_hash }, state.tick);
        state.pending_requests = pending_requests;

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::BlockLogsReceived {
                request_id,
                logs: pool_logs(&HashSet::from([candidate])),
            },
        );

        let pool_data_request_id =
            assert_single_pool_data_request_effect(&effects, block_hash, &HashSet::from([pool]));
        assert!(next_state.pending_requests.contains(&pool_data_request_id));
        assert_effects_are_well_formed(&next_state, &effects);
        assert_state_invariants(&next_state);
    }

    // Resolves logs for an unvalidated candidate.
    // This prevents candidate emitters from driving pool-data RPC before registry trust is established.
    #[test]
    fn block_logs_received_does_not_request_pool_data_for_pending_candidate() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let candidate = pool_candidate_address(3);
        let mut state = empty_state_at(finalized_hash);

        state.canonical_tip = block_hash;
        state
            .blocks
            .0
            .insert(block_hash, block_with_parent(finalized_hash));
        let (pending_requests, request_id) = state
            .pending_requests
            .with_new_request(GetBlockLogs { block_hash }, state.tick);
        state.pending_requests = pending_requests;

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::BlockLogsReceived {
                request_id,
                logs: pool_logs(&HashSet::from([candidate])),
            },
        );

        assert_single_pool_metadata_request_effect(
            &effects,
            block_hash,
            &HashSet::from([candidate]),
        );
        assert_no_pool_data_request_effect(&effects);
        assert_state_invariants(&next_state);
    }

    // A Swap is an absolute snapshot, so a verified pool's state is derived straight from the log
    // and no pool-data read is scheduled for it.
    #[test]
    fn block_logs_received_derives_snapshot_from_swap_and_skips_pool_data() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let candidate = pool_candidate_address(3);
        let pool = PoolRef { key: candidate, chain: ChainKey::Ethereum };
        let mut state = empty_state_at(finalized_hash);

        state.pool_registry = registry_verifying(candidate);
        state.canonical_tip = block_hash;
        state
            .blocks
            .0
            .insert(block_hash, block_with_parent(finalized_hash));
        let (pending_requests, request_id) = state
            .pending_requests
            .with_new_request(GetBlockLogs { block_hash }, state.tick);
        state.pending_requests = pending_requests;

        let derived = pool_state(7);
        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::BlockLogsReceived {
                request_id,
                logs: vec![swap_log(candidate, 0, &derived)],
            },
        );

        assert_eq!(
            block_pool_snapshot(&next_state, block_hash, pool),
            Some(&derived)
        );
        assert_no_pool_data_request_effect(&effects);
        assert_state_invariants(&next_state);
    }

    // A Mint is a liquidity delta over the pool's prior snapshot: with a finalized base the new
    // snapshot is the base with its in-range liquidity adjusted, again skipping the pool-data read.
    #[test]
    fn block_logs_received_derives_mint_delta_over_finalized_base() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let candidate = pool_candidate_address(3);
        let pool = PoolRef { key: candidate, chain: ChainKey::Ethereum };
        let base = PoolState {
            sqrt_price_x96: U160::from(123u128),
            tick: I24::try_from(5).unwrap(),
            liquidity: 1_000,
        };

        let mut state = empty_state_at(finalized_hash);
        state.finalized_state = FinalizedState::with_pool_snapshots_for_test(
            finalized_hash,
            HashMap::from([(pool, base.clone())]),
        );
        state.pool_registry = registry_verifying(candidate);
        state.canonical_tip = block_hash;
        state
            .blocks
            .0
            .insert(block_hash, block_with_parent(finalized_hash));
        let (pending_requests, request_id) = state
            .pending_requests
            .with_new_request(GetBlockLogs { block_hash }, state.tick);
        state.pending_requests = pending_requests;

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::BlockLogsReceived {
                request_id,
                logs: vec![mint_log(candidate, 0, 0, 10, 500)],
            },
        );

        assert_eq!(
            block_pool_snapshot(&next_state, block_hash, pool),
            Some(&PoolState {
                liquidity: 1_500,
                ..base
            })
        );
        assert_no_pool_data_request_effect(&effects);
        assert_state_invariants(&next_state);
    }

    // A Mint for a verified pool with no prior snapshot cannot be derived (a delta needs a base),
    // so no snapshot is stored and the pool falls back to a pool-data read.
    #[test]
    fn block_logs_received_mint_without_base_falls_back_to_pool_data() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let candidate = pool_candidate_address(3);
        let pool = PoolRef { key: candidate, chain: ChainKey::Ethereum };
        let mut state = empty_state_at(finalized_hash);

        state.pool_registry = registry_verifying(candidate);
        state.token_registry = TokenRegistry::new().with_metadata_results(HashMap::from([
            (token_address(1), Ok(token_metadata(6))),
            (token_address(2), Ok(token_metadata(18))),
        ]));
        state.canonical_tip = block_hash;
        state
            .blocks
            .0
            .insert(block_hash, block_with_parent(finalized_hash));
        let (pending_requests, request_id) = state
            .pending_requests
            .with_new_request(GetBlockLogs { block_hash }, state.tick);
        state.pending_requests = pending_requests;

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::BlockLogsReceived {
                request_id,
                logs: vec![mint_log(candidate, 0, 0, 10, 500)],
            },
        );

        assert_eq!(block_pool_snapshot(&next_state, block_hash, pool), None);
        assert_single_pool_data_request_effect(&effects, block_hash, &HashSet::from([pool]));
        assert_state_invariants(&next_state);
    }

    // A `Swap` is an absolute snapshot, so it derives even when the pool's base is *uncertain* (an
    // ancestor whose logs are not yet `Resolved` might have changed the pool unseen). The swap pins
    // the exact post-block state regardless, so the block is snapshotted with no `GetPoolData` read.
    #[test]
    fn block_logs_received_derives_swap_with_unresolved_ancestor() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let ancestor_hash = BlockHash::with_last_byte(2);
        let block_hash = BlockHash::with_last_byte(3);
        let candidate = pool_candidate_address(4);
        let pool = PoolRef { key: candidate, chain: ChainKey::Ethereum };
        let mut state = empty_state_at(finalized_hash);

        state.pool_registry = registry_verifying(candidate);
        state.canonical_tip = block_hash;
        // The ancestor's logs are `Unknown`, so the base walk for `block_hash` is `Uncertain`.
        state
            .blocks
            .0
            .insert(ancestor_hash, block_with_parent(finalized_hash));
        state
            .blocks
            .0
            .insert(block_hash, block_with_parent(ancestor_hash));
        let (pending_requests, request_id) = state
            .pending_requests
            .with_new_request(GetBlockLogs { block_hash }, state.tick);
        state.pending_requests = pending_requests;

        let derived = pool_state(7);
        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::BlockLogsReceived {
                request_id,
                logs: vec![swap_log(candidate, 0, &derived)],
            },
        );

        assert_eq!(
            block_pool_snapshot(&next_state, block_hash, pool),
            Some(&derived)
        );
        assert_no_pool_data_request_effect(&effects);
        assert_state_invariants(&next_state);
    }

    // Within a block a leading `Mint` with no base has nothing to apply to and is skipped; a later
    // `Swap` seeds the run, so the derived snapshot is exactly the swap's absolute state.
    #[test]
    fn block_logs_received_derives_swap_after_leading_mint_without_base() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let candidate = pool_candidate_address(3);
        let pool = PoolRef { key: candidate, chain: ChainKey::Ethereum };
        let mut state = empty_state_at(finalized_hash);

        state.pool_registry = registry_verifying(candidate);
        state.canonical_tip = block_hash;
        state
            .blocks
            .0
            .insert(block_hash, block_with_parent(finalized_hash));
        let (pending_requests, request_id) = state
            .pending_requests
            .with_new_request(GetBlockLogs { block_hash }, state.tick);
        state.pending_requests = pending_requests;

        let derived = pool_state(7);
        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::BlockLogsReceived {
                request_id,
                logs: vec![
                    mint_log(candidate, 0, 0, 10, 500),
                    swap_log(candidate, 1, &derived),
                ],
            },
        );

        assert_eq!(
            block_pool_snapshot(&next_state, block_hash, pool),
            Some(&derived)
        );
        assert_no_pool_data_request_effect(&effects);
        assert_state_invariants(&next_state);
    }

    fn partial(
        parent_hash: BlockHash,
        candidates: HashSet<ProtocolPoolKey>,
        pool_snapshots: HashMap<PoolRef, PoolState>,
    ) -> BlockNode {
        BlockNode {
            logs_bloom: None,
            parent_hash,
            pool_logs: PoolLogsStatus::Partial(candidates),
            pool_snapshots,
            pool_data_failures: HashMap::new(),
        }
    }

    // A `Partial` block is provisional, so the authoritative log fetch is still scheduled for it
    // (a `Resolved` block, by contrast, is done).
    #[test]
    fn partial_logs_still_need_an_authoritative_log_fetch() {
        let finalized = BlockHash::with_last_byte(1);
        let partial_tip = BlockHash::with_last_byte(2);
        let resolved_tip = BlockHash::with_last_byte(3);
        let candidate = pool_candidate_address(4);

        let partial_graph = BlocksGraph(HashMap::from([(
            partial_tip,
            partial(finalized, HashSet::from([candidate]), HashMap::new()),
        )]));
        assert_eq!(
            partial_graph.unknown_present_canonical_log_hashes(
                partial_tip,
                finalized,
                &HashSet::new()
            ),
            vec![partial_tip]
        );

        let resolved_graph = BlocksGraph(HashMap::from([(
            resolved_tip,
            resolved_block_with_snapshots(finalized, HashSet::from([candidate]), HashMap::new()),
        )]));
        assert!(
            resolved_graph
                .unknown_present_canonical_log_hashes(resolved_tip, finalized, &HashSet::new())
                .is_empty()
        );
    }

    // The completeness scan must not advance through a `Partial` block even when it carries a
    // snapshot: provisional state cannot contribute to a "complete" overlay.
    #[test]
    fn complete_pool_state_scan_stops_at_partial_block() {
        let finalized = BlockHash::with_last_byte(1);
        let tip = BlockHash::with_last_byte(2);
        let candidate = pool_candidate_address(3);
        let pool = pool_address(3);
        let mut state = empty_state_at(finalized);

        state.pool_registry = registry_verifying(candidate);
        state.canonical_tip = tip;
        state.blocks.0.insert(
            tip,
            partial(
                finalized,
                HashSet::from([candidate]),
                HashMap::from([(pool, pool_state(3))]),
            ),
        );

        let update = state
            .latest_complete_pool_state_update_from(ChainKey::Ethereum, tip)
            .expect("overlay query");
        assert_eq!(update.block_hash, finalized);
        assert!(update.pool_snapshot_blocks.is_empty());
    }

    // The same block resolved (authoritative) now advances the overlay frontier and contributes its
    // snapshot.
    #[test]
    fn complete_pool_state_scan_advances_when_block_is_resolved() {
        let finalized = BlockHash::with_last_byte(1);
        let tip = BlockHash::with_last_byte(2);
        let candidate = pool_candidate_address(3);
        let pool = pool_address(3);
        let snapshot = pool_state(3);
        let mut state = empty_state_at(finalized);

        state.pool_registry = registry_verifying(candidate);
        state.canonical_tip = tip;
        state.blocks.0.insert(
            tip,
            resolved_block_with_snapshots(
                finalized,
                HashSet::from([candidate]),
                HashMap::from([(pool, snapshot.clone())]),
            ),
        );

        assert_eq!(
            resolved_complete_pool_state_update_from(&state, tip),
            Some(complete_pool_state_update(
                tip,
                HashMap::from([(pool, snapshot)])
            ))
        );
    }

    // A subscription log on a known block records a provisional (`Partial`) snapshot — enough to
    // skip the pool-data read — while still scheduling the authoritative `GetBlockLogs`.
    #[test]
    fn log_observed_on_known_block_is_provisional_and_still_fetches_authoritative_logs() {
        let finalized = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let candidate = pool_candidate_address(3);
        let pool = pool_address(3);
        let mut state = empty_state_at(finalized);

        state.pool_registry = registry_verifying(candidate);
        state.canonical_tip = block_hash;
        state
            .blocks
            .0
            .insert(block_hash, block_with_parent(finalized));

        let derived = pool_state(7);
        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::LogObserved {
                block_hash,
                logs: vec![swap_log(candidate, 0, &derived)],
            },
        );

        assert!(matches!(
            next_state.blocks.get(&block_hash).expect("block").pool_logs,
            PoolLogsStatus::Partial(_)
        ));
        assert_eq!(
            block_pool_snapshot(&next_state, block_hash, pool),
            Some(&derived)
        );
        assert_single_block_log_request_effect(&effects, block_hash);
        assert_no_pool_data_request_effect(&effects);
        assert_state_invariants(&next_state);
    }

    // A subscription log can arrive before the block's head: it is staged and applied once the head
    // brings the block into the graph.
    #[test]
    fn log_observed_for_unknown_block_buffers_until_head_arrives() {
        let finalized = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let candidate = pool_candidate_address(3);
        let pool = pool_address(3);
        let mut state = empty_state_at(finalized);

        state.pool_registry = registry_verifying(candidate);

        let derived = pool_state(7);
        let (buffered_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::LogObserved {
                block_hash,
                logs: vec![swap_log(candidate, 0, &derived)],
            },
        );

        // Buffering the log emits nothing; the verified-but-uncovered pool draws a background backfill.
        assert_no_priority_effects(&effects);
        assert!(buffered_state.blocks.get(&block_hash).is_none());

        let (next_state, _effects) = transition(
            ChainKey::Ethereum,
            buffered_state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: block_hash,
                parent_hash: finalized,
            },
        );

        assert!(matches!(
            next_state.blocks.get(&block_hash).expect("block").pool_logs,
            PoolLogsStatus::Partial(_)
        ));
        assert_eq!(
            block_pool_snapshot(&next_state, block_hash, pool),
            Some(&derived)
        );
        assert!(next_state.streamed_logs.is_empty());
        assert_state_invariants(&next_state);
    }

    // Resolves logs containing a rejected candidate.
    // This ensures rejected emitters cannot leak into pool-data scheduling.
    #[test]
    fn block_logs_received_excludes_rejected_candidates_from_pool_data_request() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let verified = pool_candidate_address(3);
        let rejected = pool_candidate_address(4);
        let verified_pool = PoolRef { key: verified, chain: ChainKey::Ethereum };
        let mut state = empty_state_at(finalized_hash);

        state.pool_registry = TrustedPoolRegistry::new().with_metadata_results(
            ChainKey::Ethereum,
            HashMap::from([
                (verified, Ok(pool_metadata(1, 2, UniswapV3Fee::Fee3000))),
                (rejected, Err(PoolMetadataFailure::FactoryReturnedZero)),
            ]),
        );
        state.token_registry = TokenRegistry::new().with_metadata_results(HashMap::from([
            (token_address(1), Ok(token_metadata(6))),
            (token_address(2), Ok(token_metadata(18))),
        ]));
        state.canonical_tip = block_hash;
        state
            .blocks
            .0
            .insert(block_hash, block_with_parent(finalized_hash));
        let (pending_requests, request_id) = state
            .pending_requests
            .with_new_request(GetBlockLogs { block_hash }, state.tick);
        state.pending_requests = pending_requests;

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::BlockLogsReceived {
                request_id,
                logs: pool_logs(&HashSet::from([verified, rejected])),
            },
        );

        assert_single_pool_data_request_effect(
            &effects,
            block_hash,
            &HashSet::from([verified_pool]),
        );
        assert_effects_are_well_formed(&next_state, &effects);
        assert_state_invariants(&next_state);
    }

    // Resolves the same unknown candidate on multiple blocks.
    // This suppresses duplicate validation RPC while one metadata request is pending.
    #[test]
    fn duplicate_unknown_candidates_across_blocks_do_not_create_duplicate_pending_metadata_requests()
     {
        let finalized_hash = BlockHash::with_last_byte(1);
        let parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let candidate = pool_candidate_address(4);
        let mut state = empty_state_at(finalized_hash);

        state.canonical_tip = head_hash;
        state
            .blocks
            .0
            .insert(parent_hash, block_with_parent(finalized_hash));
        state
            .blocks
            .0
            .insert(head_hash, block_with_parent(parent_hash));
        let (pending_requests, parent_request_id) = state.pending_requests.with_new_request(
            GetBlockLogs {
                block_hash: parent_hash,
            },
            state.tick,
        );
        let (pending_requests, head_request_id) = pending_requests.with_new_request(
            GetBlockLogs {
                block_hash: head_hash,
            },
            state.tick,
        );
        state.pending_requests = pending_requests;

        let (state, first_effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::BlockLogsReceived {
                request_id: parent_request_id,
                logs: pool_logs(&HashSet::from([candidate])),
            },
        );
        let (next_state, second_effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::BlockLogsReceived {
                request_id: head_request_id,
                logs: pool_logs(&HashSet::from([candidate])),
            },
        );

        assert_eq!(first_effects.len(), 1);
        assert!(second_effects.is_empty());
        assert_state_invariants(&next_state);
    }

    // Returns metadata for only part of a requested candidate batch.
    // This keeps omitted entries pending by rescheduling them instead of marking them done.
    #[test]
    fn pool_metadata_received_reschedules_missing_requested_candidates() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let first_candidate = pool_candidate_address(3);
        let second_candidate = pool_candidate_address(4);
        let candidates = HashSet::from([first_candidate, second_candidate]);
        let mut state = empty_state_at(finalized_hash);

        state.canonical_tip = block_hash;
        state.blocks.0.insert(
            block_hash,
            BlockNode {
                logs_bloom: None,
                parent_hash: finalized_hash,
                pool_logs: PoolLogsStatus::Resolved(candidates.clone()),
                pool_snapshots: HashMap::new(),
                pool_data_failures: HashMap::new(),
            },
        );
        let (pending_requests, request_id) = state.pending_requests.with_new_request(
            GetPoolMetadata {
                at: block_hash,
                candidates,
            },
            state.tick,
        );
        state.pending_requests = pending_requests;

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::PoolMetadataReceived {
                request_id,
                metadata: HashMap::from([(
                    first_candidate,
                    Ok(pool_metadata(1, 2, UniswapV3Fee::Fee500)),
                )]),
            },
        );

        assert!(!next_state.pending_requests.contains(&request_id));
        assert_eq!(
            next_state
                .pool_registry
                .verified_pool(ChainKey::Ethereum, first_candidate),
            Some(PoolRef { key: first_candidate, chain: ChainKey::Ethereum })
        );
        assert_eq!(
            next_state
                .pool_registry
                .verified_pool(ChainKey::Ethereum, second_candidate),
            None
        );
        let retry_request_id = assert_single_pool_metadata_request_effect(
            &effects,
            block_hash,
            &HashSet::from([second_candidate]),
        );
        assert!(next_state.pending_requests.contains(&retry_request_id));
        assert_canonical_resolved_candidates_are_known_or_pending(&next_state);
        assert_state_invariants(&next_state);
    }

    // Applies successful pool metadata for a resolved candidate.
    // This proves trusted-log projections update from registry state without rewriting block logs.
    #[test]
    fn pool_metadata_received_updates_registry_and_derived_trusted_logs() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let candidate = pool_candidate_address(3);
        let mut state = empty_state_at(finalized_hash);

        state.canonical_tip = block_hash;
        state.blocks.0.insert(
            block_hash,
            BlockNode {
                logs_bloom: None,
                parent_hash: finalized_hash,
                pool_logs: PoolLogsStatus::Resolved(HashSet::from([candidate])),
                pool_snapshots: HashMap::new(),
                pool_data_failures: HashMap::new(),
            },
        );
        let (pending_requests, request_id) = state.pending_requests.with_new_request(
            GetPoolMetadata {
                at: block_hash,
                candidates: HashSet::from([candidate]),
            },
            state.tick,
        );
        state.pending_requests = pending_requests;

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::PoolMetadataReceived {
                request_id,
                metadata: HashMap::from([(
                    candidate,
                    Ok(pool_metadata(1, 2, UniswapV3Fee::Fee500)),
                )]),
            },
        );

        let token_request_id = assert_single_token_metadata_request_effect(
            &effects,
            block_hash,
            &HashSet::from([token_address(1), token_address(2)]),
        );
        assert!(next_state.pending_requests.contains(&token_request_id));
        assert!(!next_state.pending_requests.contains(&request_id));
        assert_eq!(
            next_state
                .pool_registry
                .verified_metadata(PoolRef { key: candidate, chain: ChainKey::Ethereum }),
            Some(&pool_metadata(1, 2, UniswapV3Fee::Fee500))
        );
        assert_trusted_pool_logs_resolved(
            &next_state,
            block_hash,
            HashSet::from([PoolRef { key: candidate, chain: ChainKey::Ethereum }]),
        );
        assert_state_invariants(&next_state);
    }

    // Applies successful metadata for a candidate while token decimals are already available.
    // This ensures newly trusted affected pools flow into pool-data scheduling.
    #[test]
    fn pool_metadata_received_requests_pool_data_for_verified_candidate() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let candidate = pool_candidate_address(3);
        let pool = PoolRef { key: candidate, chain: ChainKey::Ethereum };
        let mut state = empty_state_at(finalized_hash);

        state.token_registry = TokenRegistry::new().with_metadata_results(HashMap::from([
            (token_address(1), Ok(token_metadata(6))),
            (token_address(2), Ok(token_metadata(18))),
        ]));
        state.canonical_tip = block_hash;
        state.blocks.0.insert(
            block_hash,
            BlockNode {
                logs_bloom: None,
                parent_hash: finalized_hash,
                pool_logs: PoolLogsStatus::Resolved(HashSet::from([candidate])),
                pool_snapshots: HashMap::new(),
                pool_data_failures: HashMap::new(),
            },
        );
        let (pending_requests, request_id) = state.pending_requests.with_new_request(
            GetPoolMetadata {
                at: block_hash,
                candidates: HashSet::from([candidate]),
            },
            state.tick,
        );
        state.pending_requests = pending_requests;

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::PoolMetadataReceived {
                request_id,
                metadata: HashMap::from([(
                    candidate,
                    Ok(pool_metadata(1, 2, UniswapV3Fee::Fee3000)),
                )]),
            },
        );

        let pool_data_request_id =
            assert_single_pool_data_request_effect(&effects, block_hash, &HashSet::from([pool]));
        assert!(next_state.pending_requests.contains(&pool_data_request_id));
        assert_effects_are_well_formed(&next_state, &effects);
        assert_state_invariants(&next_state);
    }

    // Builds a resolved block carrying per-pool data failures.
    // Coverage tests use it to place a failure marker on a specific block.
    fn resolved_block_with_failures(
        parent_hash: BlockHash,
        candidates: HashSet<ProtocolPoolKey>,
        pool_data_failures: HashMap<PoolRef, PoolDataFailure>,
    ) -> BlockNode {
        BlockNode {
            logs_bloom: None,
            parent_hash,
            pool_logs: PoolLogsStatus::Resolved(candidates),
            pool_snapshots: HashMap::new(),
            pool_data_failures,
        }
    }

    // Verifies a single candidate so its resolved logs project to a trusted pool.
    fn registry_verifying(candidate: ProtocolPoolKey) -> TrustedPoolRegistry {
        TrustedPoolRegistry::new().with_metadata_results(
            ChainKey::Ethereum,
            HashMap::from([(candidate, Ok(pool_metadata(1, 2, UniswapV3Fee::Fee500)))]),
        )
    }

    // A pool snapshotted at an earlier block must not be re-read as the tip advances without re-dirtying.
    // This is the redundant-read regression: coverage is checked per block, not only at the tip.
    #[test]
    fn snapshot_at_earlier_block_is_not_re_requested_as_tip_advances() {
        let finalized = BlockHash::with_last_byte(1);
        let dirty_block = BlockHash::with_last_byte(2);
        let middle = BlockHash::with_last_byte(3);
        let tip = BlockHash::with_last_byte(4);
        let candidate = pool_candidate_address(9);
        let pool = pool_address(9);

        let blocks = BlocksGraph(HashMap::from([
            (
                dirty_block,
                resolved_block_with_snapshots(
                    finalized,
                    HashSet::from([candidate]),
                    HashMap::from([(pool, pool_state(9))]),
                ),
            ),
            (
                middle,
                resolved_block_with_snapshots(dirty_block, HashSet::new(), HashMap::new()),
            ),
            (
                tip,
                resolved_block_with_snapshots(middle, HashSet::new(), HashMap::new()),
            ),
        ]));

        let request = blocks.unknown_present_canonical_pool_data_request(
            ChainKey::Ethereum,
            tip,
            finalized,
            &registry_verifying(candidate),
            &HashMap::new(),
        );

        assert_eq!(request, None);
    }

    // A pool with an in-flight read recorded at an earlier block must not be re-requested at the new tip.
    #[test]
    fn pending_pool_data_at_earlier_block_is_not_re_requested_as_tip_advances() {
        let finalized = BlockHash::with_last_byte(1);
        let dirty_block = BlockHash::with_last_byte(2);
        let tip = BlockHash::with_last_byte(3);
        let candidate = pool_candidate_address(9);
        let pool = pool_address(9);

        let blocks = BlocksGraph(HashMap::from([
            (
                dirty_block,
                resolved_block_with_snapshots(
                    finalized,
                    HashSet::from([candidate]),
                    HashMap::new(),
                ),
            ),
            (
                tip,
                resolved_block_with_snapshots(dirty_block, HashSet::new(), HashMap::new()),
            ),
        ]));
        let pending = HashMap::from([(dirty_block, HashSet::from([pool]))]);

        let request = blocks.unknown_present_canonical_pool_data_request(
            ChainKey::Ethereum,
            tip,
            finalized,
            &registry_verifying(candidate),
            &pending,
        );

        assert_eq!(request, None);
    }

    // A pool whose read failed at an earlier block must not be retried every head; retry waits for re-dirty.
    #[test]
    fn pool_data_failure_at_earlier_block_is_not_re_requested_as_tip_advances() {
        let finalized = BlockHash::with_last_byte(1);
        let dirty_block = BlockHash::with_last_byte(2);
        let tip = BlockHash::with_last_byte(3);
        let candidate = pool_candidate_address(9);
        let pool = pool_address(9);

        let blocks = BlocksGraph(HashMap::from([
            (
                dirty_block,
                resolved_block_with_failures(
                    finalized,
                    HashSet::from([candidate]),
                    HashMap::from([(pool, PoolDataFailure::CallFailed(PoolDataCall::Slot0))]),
                ),
            ),
            (
                tip,
                resolved_block_with_snapshots(dirty_block, HashSet::new(), HashMap::new()),
            ),
        ]));

        let request = blocks.unknown_present_canonical_pool_data_request(
            ChainKey::Ethereum,
            tip,
            finalized,
            &registry_verifying(candidate),
            &HashMap::new(),
        );

        assert_eq!(request, None);
    }

    // A pool re-dirtied by a block newer than its coverage must be requested again.
    #[test]
    fn pool_re_dirtied_after_coverage_is_requested_again() {
        let finalized = BlockHash::with_last_byte(1);
        let dirty_block = BlockHash::with_last_byte(2);
        let tip = BlockHash::with_last_byte(3);
        let candidate = pool_candidate_address(9);
        let pool = pool_address(9);

        let blocks = BlocksGraph(HashMap::from([
            (
                dirty_block,
                resolved_block_with_snapshots(
                    finalized,
                    HashSet::from([candidate]),
                    HashMap::from([(pool, pool_state(9))]),
                ),
            ),
            (
                tip,
                resolved_block_with_snapshots(
                    dirty_block,
                    HashSet::from([candidate]),
                    HashMap::new(),
                ),
            ),
        ]));

        let request = blocks.unknown_present_canonical_pool_data_request(
            ChainKey::Ethereum,
            tip,
            finalized,
            &registry_verifying(candidate),
            &HashMap::new(),
        );

        assert_eq!(request, Some((tip, HashSet::from([pool]))));
    }

    // A snapshot on the tip block still suppresses the read (existing behavior preserved).
    #[test]
    fn snapshot_at_tip_is_not_requested() {
        let finalized = BlockHash::with_last_byte(1);
        let tip = BlockHash::with_last_byte(2);
        let candidate = pool_candidate_address(9);
        let pool = pool_address(9);

        let blocks = BlocksGraph(HashMap::from([(
            tip,
            resolved_block_with_snapshots(
                finalized,
                HashSet::from([candidate]),
                HashMap::from([(pool, pool_state(9))]),
            ),
        )]));

        let request = blocks.unknown_present_canonical_pool_data_request(
            ChainKey::Ethereum,
            tip,
            finalized,
            &registry_verifying(candidate),
            &HashMap::new(),
        );

        assert_eq!(request, None);
    }

    // A dirty pool with no block-level coverage is requested (e.g. only a finalized snapshot exists).
    #[test]
    fn dirty_pool_without_block_coverage_is_requested() {
        let finalized = BlockHash::with_last_byte(1);
        let tip = BlockHash::with_last_byte(2);
        let candidate = pool_candidate_address(9);
        let pool = pool_address(9);

        let blocks = BlocksGraph(HashMap::from([(
            tip,
            resolved_block_with_snapshots(finalized, HashSet::from([candidate]), HashMap::new()),
        )]));

        let request = blocks.unknown_present_canonical_pool_data_request(
            ChainKey::Ethereum,
            tip,
            finalized,
            &registry_verifying(candidate),
            &HashMap::new(),
        );

        assert_eq!(request, Some((tip, HashSet::from([pool]))));
    }

    proptest! {
        // A pool is requested iff its latest dirty block is newer than its latest coverage
        // (snapshot, failure, or pending) across the whole canonical suffix — not just the tip.
        #[test]
        fn pool_requested_iff_dirtied_after_latest_coverage(
            tags in prop::collection::vec((any::<bool>(), 0u8..4), 1..7)
        ) {
            let finalized = BlockHash::with_last_byte(200);
            let candidate = pool_candidate_address(9);
            let pool = pool_address(9);

            let mut nodes = HashMap::new();
            let mut pending: HashMap<BlockHash, HashSet<PoolRef>> = HashMap::new();
            let mut latest_dirty: Option<usize> = None;
            let mut latest_cover: Option<usize> = None;

            for (idx, (dirty, coverage)) in tags.iter().copied().enumerate() {
                let block_hash = BlockHash::with_last_byte(idx as u8);
                let parent_hash = match idx.checked_sub(1) {
                    Some(prev) => BlockHash::with_last_byte(prev as u8),
                    None => finalized,
                };
                let candidates = if dirty {
                    HashSet::from([candidate])
                } else {
                    HashSet::new()
                };
                if dirty {
                    latest_dirty = Some(idx);
                }

                let mut snapshots = HashMap::new();
                let mut failures = HashMap::new();
                match coverage {
                    1 => {
                        snapshots.insert(pool, pool_state(9));
                        latest_cover = Some(idx);
                    }
                    2 => {
                        failures.insert(pool, PoolDataFailure::CallFailed(PoolDataCall::Slot0));
                        latest_cover = Some(idx);
                    }
                    3 => {
                        pending.insert(block_hash, HashSet::from([pool]));
                        latest_cover = Some(idx);
                    }
                    _ => {}
                }

                nodes.insert(
                    block_hash,
                    BlockNode {
                        logs_bloom: None,
                        parent_hash,
                        pool_logs: PoolLogsStatus::Resolved(candidates),
                        pool_snapshots: snapshots,
                        pool_data_failures: failures,
                    },
                );
            }

            let tip = BlockHash::with_last_byte((tags.len() - 1) as u8);
            let blocks = BlocksGraph(nodes);

            let request = blocks.unknown_present_canonical_pool_data_request(
                ChainKey::Ethereum,
                tip,
                finalized,
                &registry_verifying(candidate),
                &pending,
            );

            let should_request = match latest_dirty {
                Some(dirty_idx) => latest_cover.map_or(true, |cover_idx| cover_idx < dirty_idx),
                None => false,
            };

            if should_request {
                prop_assert_eq!(request, Some((tip, HashSet::from([pool]))));
            } else {
                prop_assert_eq!(request, None);
            }
        }
    }

    // Returns pool metadata for an address outside the request payload.
    // This protects the registry from stale or overbroad RPC responses.
    #[test]
    fn pool_metadata_received_ignores_unrequested_result_entries() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let requested = pool_candidate_address(3);
        let unrequested = pool_candidate_address(4);
        let mut state = empty_state_at(finalized_hash);
        let (pending_requests, request_id) = state.pending_requests.with_new_request(
            GetPoolMetadata {
                at: block_hash,
                candidates: HashSet::from([requested]),
            },
            state.tick,
        );
        state.pending_requests = pending_requests;

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::PoolMetadataReceived {
                request_id,
                metadata: HashMap::from([(
                    unrequested,
                    Ok(pool_metadata(1, 2, UniswapV3Fee::Fee500)),
                )]),
            },
        );

        assert!(effects.is_empty());
        assert_eq!(
            next_state
                .pool_registry
                .verified_metadata(PoolRef { key: unrequested, chain: ChainKey::Ethereum }),
            None
        );
        assert_eq!(
            next_state
                .pool_registry
                .verified_pool(ChainKey::Ethereum, unrequested),
            None
        );
        assert!(!next_state.pool_registry.is_rejected(unrequested));
        assert_state_invariants(&next_state);
    }

    // Returns token metadata with both requested and unrequested entries.
    // This ensures only requested token facts enter the registry.
    #[test]
    fn token_metadata_received_updates_registry_and_ignores_unrequested_entries() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let requested = token_address(3);
        let unrequested = token_address(4);
        let mut state = empty_state_at(finalized_hash);
        let (pending_requests, request_id) = state.pending_requests.with_new_request(
            GetTokenMetadata {
                at: block_hash,
                tokens: HashSet::from([requested]),
            },
            state.tick,
        );
        state.pending_requests = pending_requests;

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::TokenMetadataReceived {
                request_id,
                metadata: HashMap::from([
                    (requested, Ok(token_metadata(6))),
                    (unrequested, Ok(token_metadata(18))),
                ]),
            },
        );

        assert!(effects.is_empty());
        assert!(!next_state.pending_requests.contains(&request_id));
        assert_eq!(
            next_state.token_registry.verified_metadata(requested),
            Some(&token_metadata(6))
        );
        assert_eq!(
            next_state.token_registry.verified_metadata(unrequested),
            None
        );
        assert!(!next_state.token_registry.is_unsupported(unrequested));
        assert_state_invariants(&next_state);
    }

    // Schedules after verified pool tokens are already known or terminal.
    // This avoids repeated token metadata RPC before pool-data fetching.
    #[test]
    fn known_tokens_do_not_request_token_metadata() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let candidate = pool_candidate_address(3);
        let pool = PoolRef { key: candidate, chain: ChainKey::Ethereum };
        let logs = HashSet::from([candidate]);
        let mut state = empty_state_at(finalized_hash);

        state.pool_registry = TrustedPoolRegistry::new().with_metadata_results(
            ChainKey::Ethereum,
            HashMap::from([(candidate, Ok(pool_metadata(1, 2, UniswapV3Fee::Fee3000)))]),
        );
        state.token_registry = TokenRegistry::new().with_metadata_results(HashMap::from([
            (token_address(1), Ok(token_metadata(6))),
            (
                token_address(2),
                Err(TokenMetadataFailure::CallFailed(
                    TokenMetadataCall::Decimals,
                )),
            ),
        ]));
        state.canonical_tip = block_hash;
        state
            .blocks
            .0
            .insert(block_hash, block_with_parent(finalized_hash));
        let (pending_requests, request_id) = state
            .pending_requests
            .with_new_request(GetBlockLogs { block_hash }, state.tick);
        state.pending_requests = pending_requests;

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::BlockLogsReceived {
                request_id,
                logs: pool_logs(&logs),
            },
        );

        let pool_data_request_id =
            assert_single_pool_data_request_effect(&effects, block_hash, &HashSet::from([pool]));
        assert!(next_state.pending_requests.contains(&pool_data_request_id));
        assert_effects_are_well_formed(&next_state, &effects);
        assert_state_invariants(&next_state);
    }

    // Schedules pool data after multiple canonical blocks affected different trusted pools.
    // This batches accumulated dirty pools into a current-tip snapshot.
    #[test]
    fn pool_data_scheduler_accumulates_dirty_pools_at_canonical_tip() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let first_candidate = pool_candidate_address(4);
        let second_candidate = pool_candidate_address(5);
        let first_pool = PoolRef { key: first_candidate, chain: ChainKey::Ethereum };
        let second_pool = PoolRef { key: second_candidate, chain: ChainKey::Ethereum };
        let mut state = empty_state_at(finalized_hash);

        state.pool_registry = TrustedPoolRegistry::new().with_metadata_results(
            ChainKey::Ethereum,
            HashMap::from([
                (
                    first_candidate,
                    Ok(pool_metadata(1, 2, UniswapV3Fee::Fee3000)),
                ),
                (
                    second_candidate,
                    Ok(pool_metadata(3, 4, UniswapV3Fee::Fee500)),
                ),
            ]),
        );
        state.token_registry = TokenRegistry::new().with_metadata_results(HashMap::from([
            (token_address(1), Ok(token_metadata(6))),
            (token_address(2), Ok(token_metadata(18))),
            (token_address(3), Ok(token_metadata(6))),
            (token_address(4), Ok(token_metadata(18))),
        ]));
        state.canonical_tip = head_hash;
        state.blocks.0.insert(
            parent_hash,
            BlockNode {
                logs_bloom: None,
                parent_hash: finalized_hash,
                pool_logs: PoolLogsStatus::Resolved(HashSet::from([first_candidate])),
                pool_snapshots: HashMap::new(),
                pool_data_failures: HashMap::new(),
            },
        );
        state.blocks.0.insert(
            head_hash,
            BlockNode {
                logs_bloom: None,
                parent_hash,
                pool_logs: PoolLogsStatus::Resolved(HashSet::from([second_candidate])),
                pool_snapshots: HashMap::new(),
                pool_data_failures: HashMap::new(),
            },
        );

        let (next_state, effects) =
            schedule_unknown_canonical_requests(ChainKey::Ethereum, state, vec![]);

        let pool_data_request_id = assert_single_pool_data_request_effect(
            &effects,
            head_hash,
            &HashSet::from([first_pool, second_pool]),
        );
        assert!(next_state.pending_requests.contains(&pool_data_request_id));
        assert_effects_are_well_formed(&next_state, &effects);
        assert_state_invariants(&next_state);
    }

    // Schedules with one pool already pending at the target block.
    // This prevents duplicate same-block pool-data requests while allowing uncovered pools to proceed.
    #[test]
    fn pending_pool_data_request_suppresses_duplicate_pool_at_same_block() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let first_candidate = pool_candidate_address(3);
        let second_candidate = pool_candidate_address(4);
        let first_pool = PoolRef { key: first_candidate, chain: ChainKey::Ethereum };
        let second_pool = PoolRef { key: second_candidate, chain: ChainKey::Ethereum };
        let mut state = empty_state_at(finalized_hash);

        state.pool_registry = TrustedPoolRegistry::new().with_metadata_results(
            ChainKey::Ethereum,
            HashMap::from([
                (
                    first_candidate,
                    Ok(pool_metadata(1, 2, UniswapV3Fee::Fee3000)),
                ),
                (
                    second_candidate,
                    Ok(pool_metadata(3, 4, UniswapV3Fee::Fee500)),
                ),
            ]),
        );
        state.token_registry = TokenRegistry::new().with_metadata_results(HashMap::from([
            (token_address(1), Ok(token_metadata(6))),
            (token_address(2), Ok(token_metadata(18))),
            (token_address(3), Ok(token_metadata(6))),
            (token_address(4), Ok(token_metadata(18))),
        ]));
        state.canonical_tip = block_hash;
        state.blocks.0.insert(
            block_hash,
            BlockNode {
                logs_bloom: None,
                parent_hash: finalized_hash,
                pool_logs: PoolLogsStatus::Resolved(HashSet::from([
                    first_candidate,
                    second_candidate,
                ])),
                pool_snapshots: HashMap::new(),
                pool_data_failures: HashMap::new(),
            },
        );
        let (pending_requests, _) = state.pending_requests.with_new_request(
            GetPoolData {
                at: block_hash,
                pools: HashSet::from([first_pool]),
            },
            state.tick,
        );
        state.pending_requests = pending_requests;

        let (next_state, effects) =
            schedule_unknown_canonical_requests(ChainKey::Ethereum, state, vec![]);

        assert_single_pool_data_request_effect(&effects, block_hash, &HashSet::from([second_pool]));
        assert_effects_are_well_formed(&next_state, &effects);
        assert_state_invariants(&next_state);
    }

    /// Registers one verified pool and returns a state whose only canonical block (`block_hash`,
    /// child of the finalized anchor) carries `logs_bloom`, plus the verified pool's address. The
    /// scaffold for the bloom-gate tests below.
    fn state_with_one_verified_pool_and_block(
        finalized_hash: BlockHash,
        block_hash: BlockHash,
        logs_bloom: Option<Bloom>,
        pool_logs: PoolLogsStatus,
    ) -> (State, Address) {
        let candidate = pool_candidate_address(3);
        let mut state = empty_state_at(finalized_hash);
        state.pool_registry = TrustedPoolRegistry::new().with_metadata_results(
            ChainKey::Ethereum,
            HashMap::from([(candidate, Ok(pool_metadata(1, 2, UniswapV3Fee::Fee3000)))]),
        );
        state.canonical_tip = block_hash;
        state.blocks.0.insert(
            block_hash,
            BlockNode {
                parent_hash: finalized_hash,
                logs_bloom,
                pool_logs,
                pool_snapshots: HashMap::new(),
                pool_data_failures: HashMap::new(),
            },
        );
        (state, candidate.uniswap_v3_address().expect("v3 pool"))
    }

    // A block whose bloom contains none of the trusted pool addresses is promoted to `Resolved`
    // without an authoritative `GetBlockLogs`: the bloom proves no trusted pool emitted here.
    #[test]
    fn bloom_clear_block_resolves_empty_without_log_request() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let unrelated = Address::with_last_byte(9);
        let (state, _pool) = state_with_one_verified_pool_and_block(
            finalized_hash,
            block_hash,
            Some(bloom_containing(&[unrelated])),
            PoolLogsStatus::Unknown,
        );

        let (next_state, effects) =
            schedule_unknown_canonical_requests(ChainKey::Ethereum, state, vec![]);

        assert_no_block_log_request_effect(&effects, block_hash);
        assert!(matches!(
            next_state.blocks.get(&block_hash).map(|block| &block.pool_logs),
            Some(PoolLogsStatus::Resolved(candidates)) if candidates.is_empty()
        ));
        assert_effects_are_well_formed(&next_state, &effects);
        assert_state_invariants(&next_state);
    }

    // A block whose bloom contains a trusted pool address still gets the authoritative fetch, so
    // trusted-pool log completeness is unchanged.
    #[test]
    fn bloom_with_trusted_address_still_requests_logs() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let (mut state, pool_address) = state_with_one_verified_pool_and_block(
            finalized_hash,
            block_hash,
            // Seed the bloom with the trusted pool's address so the gate must fetch.
            None,
            PoolLogsStatus::Unknown,
        );
        state
            .blocks
            .0
            .get_mut(&block_hash)
            .expect("block present")
            .logs_bloom = Some(bloom_containing(&[pool_address]));

        let (next_state, effects) =
            schedule_unknown_canonical_requests(ChainKey::Ethereum, state, vec![]);

        let _ = assert_single_block_log_request_effect(&effects, block_hash);
        assert!(matches!(
            next_state
                .blocks
                .get(&block_hash)
                .map(|block| &block.pool_logs),
            Some(PoolLogsStatus::Unknown)
        ));
        assert_effects_are_well_formed(&next_state, &effects);
        assert_state_invariants(&next_state);
    }

    // A bloom-clear `Partial` block keeps its subscription-discovered candidates when promoted to
    // `Resolved`, so an unknown candidate still drives `PoolMetadata` validation (best-effort
    // discovery survives the skipped fetch).
    #[test]
    fn partial_block_bloom_clear_preserves_candidates_for_discovery() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let undiscovered = pool_candidate_address(7);
        let (state, _pool) = state_with_one_verified_pool_and_block(
            finalized_hash,
            block_hash,
            Some(bloom_containing(&[undiscovered.uniswap_v3_address().expect("v3 pool")])),
            PoolLogsStatus::Partial(HashSet::from([undiscovered])),
        );

        let (next_state, effects) =
            schedule_unknown_canonical_requests(ChainKey::Ethereum, state, vec![]);

        assert_no_block_log_request_effect(&effects, block_hash);
        assert!(matches!(
            next_state.blocks.get(&block_hash).map(|block| &block.pool_logs),
            Some(PoolLogsStatus::Resolved(candidates)) if *candidates == HashSet::from([undiscovered])
        ));
        // The retained candidate is still offered to metadata validation.
        let _ = assert_single_pool_metadata_request_effect(
            &effects,
            block_hash,
            &HashSet::from([undiscovered]),
        );
        assert_effects_are_well_formed(&next_state, &effects);
        assert_state_invariants(&next_state);
    }

    // A block whose bloom carries only the v4 PoolManager (no trusted v3 pool) still gets the
    // authoritative fetch: the PoolManager is the v4 discovery anchor, so v4-only blocks are never
    // bloom-skipped once the gate is active (a verified v3 pool is present here).
    #[test]
    fn bloom_with_v4_pool_manager_requests_logs() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let (state, _pool) = state_with_one_verified_pool_and_block(
            finalized_hash,
            block_hash,
            Some(bloom_containing(&[
                uniswap_v4::ETHEREUM_UNISWAP_V4_POOL_MANAGER_ADDRESS,
            ])),
            PoolLogsStatus::Unknown,
        );

        let (next_state, effects) =
            schedule_unknown_canonical_requests(ChainKey::Ethereum, state, vec![]);

        let _ = assert_single_block_log_request_effect(&effects, block_hash);
        assert!(matches!(
            next_state.blocks.get(&block_hash).map(|block| &block.pool_logs),
            Some(PoolLogsStatus::Unknown)
        ));
        assert_effects_are_well_formed(&next_state, &effects);
        assert_state_invariants(&next_state);
    }

    // A block with no recorded header bloom (bootstrap/finalized-anchor blocks) is fetched exactly
    // as before — the gate only acts on proof, never on absence of it.
    #[test]
    fn none_bloom_block_requests_logs() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let (state, _pool) = state_with_one_verified_pool_and_block(
            finalized_hash,
            block_hash,
            None,
            PoolLogsStatus::Unknown,
        );

        let (next_state, effects) =
            schedule_unknown_canonical_requests(ChainKey::Ethereum, state, vec![]);

        let _ = assert_single_block_log_request_effect(&effects, block_hash);
        assert_effects_are_well_formed(&next_state, &effects);
        assert_state_invariants(&next_state);
    }

    // With no verified pools there is nothing whose completeness to protect, but the per-block fetch
    // is still the discovery channel during warmup, so a bloom-bearing block is fetched, not skipped.
    #[test]
    fn bloom_clear_block_with_no_trusted_pools_still_requests_logs() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let mut state = empty_state_at(finalized_hash);
        state.canonical_tip = block_hash;
        state.blocks.0.insert(
            block_hash,
            BlockNode {
                parent_hash: finalized_hash,
                logs_bloom: Some(bloom_containing(&[Address::with_last_byte(9)])),
                pool_logs: PoolLogsStatus::Unknown,
                pool_snapshots: HashMap::new(),
                pool_data_failures: HashMap::new(),
            },
        );

        let (next_state, effects) =
            schedule_unknown_canonical_requests(ChainKey::Ethereum, state, vec![]);

        let _ = assert_single_block_log_request_effect(&effects, block_hash);
        assert_effects_are_well_formed(&next_state, &effects);
        assert_state_invariants(&next_state);
    }

    // Adds a newer canonical dirty block while an older snapshot request is pending.
    // This ensures fresh canonical changes can still schedule current-tip data.
    #[test]
    fn newer_dirty_block_schedules_fresh_pool_data_request_despite_older_pending_request() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let candidate = pool_candidate_address(4);
        let pool = PoolRef { key: candidate, chain: ChainKey::Ethereum };
        let mut state = empty_state_at(finalized_hash);

        state.pool_registry = TrustedPoolRegistry::new().with_metadata_results(
            ChainKey::Ethereum,
            HashMap::from([(candidate, Ok(pool_metadata(1, 2, UniswapV3Fee::Fee3000)))]),
        );
        state.token_registry = TokenRegistry::new().with_metadata_results(HashMap::from([
            (token_address(1), Ok(token_metadata(6))),
            (token_address(2), Ok(token_metadata(18))),
        ]));
        state.canonical_tip = head_hash;
        state.blocks.0.insert(
            parent_hash,
            BlockNode {
                logs_bloom: None,
                parent_hash: finalized_hash,
                pool_logs: PoolLogsStatus::Resolved(HashSet::from([candidate])),
                pool_snapshots: HashMap::new(),
                pool_data_failures: HashMap::new(),
            },
        );
        state.blocks.0.insert(
            head_hash,
            BlockNode {
                logs_bloom: None,
                parent_hash,
                pool_logs: PoolLogsStatus::Resolved(HashSet::from([candidate])),
                pool_snapshots: HashMap::new(),
                pool_data_failures: HashMap::new(),
            },
        );
        let (pending_requests, _) = state.pending_requests.with_new_request(
            GetPoolData {
                at: parent_hash,
                pools: HashSet::from([pool]),
            },
            state.tick,
        );
        state.pending_requests = pending_requests;

        let (next_state, effects) =
            schedule_unknown_canonical_requests(ChainKey::Ethereum, state, vec![]);

        assert_single_pool_data_request_effect(&effects, head_hash, &HashSet::from([pool]));
        assert_effects_are_well_formed(&next_state, &effects);
        assert_state_invariants(&next_state);
    }

    // Schedules after a deterministic pool-data failure is recorded at the target block.
    // This prevents automatic retry loops until a new dirty block appears.
    #[test]
    fn pool_data_scheduler_does_not_retry_pool_failure_at_same_block() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let candidate = pool_candidate_address(3);
        let pool = PoolRef { key: candidate, chain: ChainKey::Ethereum };
        let mut state = empty_state_at(finalized_hash);

        state.pool_registry = TrustedPoolRegistry::new().with_metadata_results(
            ChainKey::Ethereum,
            HashMap::from([(candidate, Ok(pool_metadata(1, 2, UniswapV3Fee::Fee3000)))]),
        );
        state.token_registry = TokenRegistry::new().with_metadata_results(HashMap::from([
            (token_address(1), Ok(token_metadata(6))),
            (token_address(2), Ok(token_metadata(18))),
        ]));
        state.canonical_tip = block_hash;
        state.blocks.0.insert(
            block_hash,
            BlockNode {
                logs_bloom: None,
                parent_hash: finalized_hash,
                pool_logs: PoolLogsStatus::Resolved(HashSet::from([candidate])),
                pool_snapshots: HashMap::new(),
                pool_data_failures: HashMap::from([(
                    pool,
                    PoolDataFailure::CallFailed(PoolDataCall::Slot0),
                )]),
            },
        );

        let (next_state, effects) =
            schedule_unknown_canonical_requests(ChainKey::Ethereum, state, vec![]);

        assert!(effects.is_empty());
        assert_state_invariants(&next_state);
    }

    // Schedules token metadata for multiple verified pools sharing token needs.
    // This batches missing token decimals across pools into one request.
    #[test]
    fn duplicate_tokens_across_verified_pools_share_one_token_metadata_request() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let first_candidate = pool_candidate_address(3);
        let second_candidate = pool_candidate_address(4);
        let logs = HashSet::from([first_candidate, second_candidate]);
        let mut state = empty_state_at(finalized_hash);

        state.pool_registry = TrustedPoolRegistry::new().with_metadata_results(
            ChainKey::Ethereum,
            HashMap::from([
                (
                    first_candidate,
                    Ok(pool_metadata(1, 2, UniswapV3Fee::Fee3000)),
                ),
                (
                    second_candidate,
                    Ok(pool_metadata(1, 2, UniswapV3Fee::Fee500)),
                ),
            ]),
        );
        state.canonical_tip = block_hash;
        state
            .blocks
            .0
            .insert(block_hash, block_with_parent(finalized_hash));
        let (pending_requests, request_id) = state
            .pending_requests
            .with_new_request(GetBlockLogs { block_hash }, state.tick);
        state.pending_requests = pending_requests;

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::BlockLogsReceived {
                request_id,
                logs: pool_logs(&logs),
            },
        );

        let token_request_id = assert_single_token_metadata_request_effect(
            &effects,
            block_hash,
            &HashSet::from([token_address(1), token_address(2)]),
        );
        assert!(next_state.pending_requests.contains(&token_request_id));
        assert_state_invariants(&next_state);
    }

    // Projects trusted logs from mixed verified and rejected candidates.
    // This proves derived trusted logs expose only registry-verified pools.
    #[test]
    fn rejected_candidates_never_appear_in_derived_trusted_pool_logs() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let verified = pool_candidate_address(3);
        let rejected = pool_candidate_address(4);
        let mut state = empty_state_at(finalized_hash);

        state.pool_registry = TrustedPoolRegistry::new().with_metadata_results(
            ChainKey::Ethereum,
            HashMap::from([
                (verified, Ok(pool_metadata(1, 2, UniswapV3Fee::Fee3000))),
                (
                    rejected,
                    Err(PoolMetadataFailure::FactoryMismatch {
                        returned: Address::with_last_byte(9),
                    }),
                ),
            ]),
        );
        state.canonical_tip = block_hash;
        state.blocks.0.insert(
            block_hash,
            BlockNode {
                logs_bloom: None,
                parent_hash: finalized_hash,
                pool_logs: PoolLogsStatus::Resolved(HashSet::from([verified, rejected])),
                pool_snapshots: HashMap::new(),
                pool_data_failures: HashMap::new(),
            },
        );

        assert_trusted_pool_logs_resolved(
            &state,
            block_hash,
            HashSet::from([PoolRef { key: verified, chain: ChainKey::Ethereum }]),
        );
        assert_state_invariants(&state);
    }

    // Projects trusted logs for unknown and resolved-empty blocks.
    // This preserves the distinction between unresolved logs and resolved logs with no trusted pools.
    #[test]
    fn trusted_pool_logs_for_unknown_block_logs_is_unknown() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let mut state = empty_state_at(finalized_hash);

        state.canonical_tip = block_hash;
        state
            .blocks
            .0
            .insert(block_hash, block_with_parent(finalized_hash));

        assert_eq!(
            state
                .blocks
                .trusted_pool_logs(ChainKey::Ethereum, block_hash, &state.pool_registry),
            Some(TrustedPoolLogs::Unknown)
        );
        assert_state_invariants(&state);
    }

    // Measures an empty canonical path at the finalized anchor.
    // This gives refresh policy callers a stable zero-length baseline.
    #[test]
    fn canonical_path_len_from_finalized_returns_zero_at_finalized_anchor() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let state = empty_state_at(finalized_hash);

        assert_eq!(state.canonical_path_len_from_finalized(), Some(0));
    }

    // Measures a fully connected canonical path from finalized to tip.
    // This keeps finalized-refresh scheduling tied to actual graph distance.
    #[test]
    fn canonical_path_len_from_finalized_returns_connected_path_length() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let first_hash = BlockHash::with_last_byte(2);
        let second_hash = BlockHash::with_last_byte(3);
        let third_hash = BlockHash::with_last_byte(4);
        let mut state = empty_state_at(finalized_hash);

        state.canonical_tip = third_hash;
        state
            .blocks
            .0
            .insert(first_hash, block_with_parent(finalized_hash));
        state
            .blocks
            .0
            .insert(second_hash, block_with_parent(first_hash));
        state
            .blocks
            .0
            .insert(third_hash, block_with_parent(second_hash));

        assert_eq!(state.canonical_path_len_from_finalized(), Some(3));
    }

    // Refuses to measure a canonical path with missing ancestry.
    // This prevents refresh policy from acting on a partial suffix.
    #[test]
    fn canonical_path_len_from_finalized_returns_none_for_disconnected_path() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let mut state = empty_state_at(finalized_hash);

        state.canonical_tip = head_hash;
        state
            .blocks
            .0
            .insert(head_hash, block_with_parent(missing_parent_hash));

        assert_eq!(state.canonical_path_len_from_finalized(), None);
    }

    // Refuses to measure cyclic canonical ancestry.
    // This keeps refresh policy from treating corrupt graph structure as mature depth.
    #[test]
    fn canonical_path_len_from_finalized_returns_none_for_cyclic_path() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let first_hash = BlockHash::with_last_byte(2);
        let second_hash = BlockHash::with_last_byte(3);
        let mut state = empty_state_at(finalized_hash);

        state.canonical_tip = second_hash;
        state
            .blocks
            .0
            .insert(first_hash, block_with_parent(second_hash));
        state
            .blocks
            .0
            .insert(second_hash, block_with_parent(first_hash));

        assert_eq!(state.canonical_path_len_from_finalized(), None);
    }

    // Measured against the finalized anchor with the tip sitting on it, the distance is zero.
    #[test]
    fn blocks_behind_returns_zero_when_reference_is_the_tip() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let state = empty_state_at(finalized_hash);

        assert_eq!(state.blocks_behind(finalized_hash), Some(0));
    }

    // Against the finalized anchor, the distance is the full connected path up to the tip.
    #[test]
    fn blocks_behind_returns_full_path_to_finalized_reference() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let first_hash = BlockHash::with_last_byte(2);
        let second_hash = BlockHash::with_last_byte(3);
        let third_hash = BlockHash::with_last_byte(4);
        let mut state = empty_state_at(finalized_hash);

        state.canonical_tip = third_hash;
        state
            .blocks
            .0
            .insert(first_hash, block_with_parent(finalized_hash));
        state
            .blocks
            .0
            .insert(second_hash, block_with_parent(first_hash));
        state
            .blocks
            .0
            .insert(third_hash, block_with_parent(second_hash));

        assert_eq!(state.blocks_behind(finalized_hash), Some(3));
    }

    // A mid-path reference measures only the blocks newer than it up to the tip.
    #[test]
    fn blocks_behind_measures_distance_from_mid_path_reference() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let first_hash = BlockHash::with_last_byte(2);
        let second_hash = BlockHash::with_last_byte(3);
        let third_hash = BlockHash::with_last_byte(4);
        let mut state = empty_state_at(finalized_hash);

        state.canonical_tip = third_hash;
        state
            .blocks
            .0
            .insert(first_hash, block_with_parent(finalized_hash));
        state
            .blocks
            .0
            .insert(second_hash, block_with_parent(first_hash));
        state
            .blocks
            .0
            .insert(third_hash, block_with_parent(second_hash));

        assert_eq!(state.blocks_behind(second_hash), Some(1));
    }

    // A reference off the tip's connected path is not measurable.
    #[test]
    fn blocks_behind_returns_none_for_disconnected_reference() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let mut state = empty_state_at(finalized_hash);

        state.canonical_tip = head_hash;
        state
            .blocks
            .0
            .insert(head_hash, block_with_parent(missing_parent_hash));

        assert_eq!(state.blocks_behind(finalized_hash), None);
    }

    // Surfaces the count of pools the registry has verified for read models.
    #[test]
    fn verified_pool_count_reflects_registry() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let mut state = empty_state_at(finalized_hash);
        state.pool_registry = TrustedPoolRegistry::new().with_metadata_results(
            ChainKey::Ethereum,
            HashMap::from([
                (
                    pool_candidate_address(2),
                    Ok(pool_metadata(1, 2, UniswapV3Fee::Fee500)),
                ),
                (
                    pool_candidate_address(3),
                    Ok(pool_metadata(3, 4, UniswapV3Fee::Fee3000)),
                ),
            ]),
        );

        assert_eq!(state.verified_pool_count(), 2);
    }

    // Checks finalized-refresh threshold behavior below the target.
    // This avoids early finalized-header fetches while the graph is still short.
    #[test]
    fn finalized_refresh_predicate_does_not_trigger_below_target() {
        assert!(!should_fetch_finalized_header(Some(70), Some(71), 72, 8));
    }

    // Checks the initial crossing into the refresh target.
    // This is the first point at which the graph should ask for a newer finalized anchor.
    #[test]
    fn finalized_refresh_predicate_triggers_when_crossing_target() {
        assert!(should_fetch_finalized_header(Some(71), Some(72), 72, 8));
    }

    // Checks duplicate suppression at a stable target length.
    // This prevents same-block events from dispatching repeated finalized fetches.
    #[test]
    fn finalized_refresh_predicate_does_not_trigger_without_length_change_at_target() {
        assert!(!should_fetch_finalized_header(Some(72), Some(72), 72, 8));
    }

    // Checks duplicate suppression within the first retry bucket.
    // This allows non-boundary head growth without repeated finalized fetches.
    #[test]
    fn finalized_refresh_predicate_does_not_trigger_within_same_retry_bucket() {
        assert!(!should_fetch_finalized_header(Some(72), Some(73), 72, 8));
        assert!(!should_fetch_finalized_header(Some(79), Some(79), 72, 8));
    }

    // Checks the next retry boundary above the target.
    // This provides bounded retries if compaction or finalized-header fetches lag.
    #[test]
    fn finalized_refresh_predicate_triggers_when_crossing_retry_bucket() {
        assert!(should_fetch_finalized_header(Some(79), Some(80), 72, 8));
        assert!(should_fetch_finalized_header(Some(87), Some(88), 72, 8));
    }

    // Checks duplicate suppression after a retry boundary.
    // This keeps every retry bucket to a single dispatch.
    #[test]
    fn finalized_refresh_predicate_does_not_trigger_inside_later_retry_bucket() {
        assert!(!should_fetch_finalized_header(Some(80), Some(87), 72, 8));
    }

    // Checks reconnecting ancestry directly into the target range.
    // This lets a completed parent chain request a finalized refresh once.
    #[test]
    fn finalized_refresh_predicate_triggers_when_disconnected_path_becomes_long_enough() {
        assert!(should_fetch_finalized_header(None, Some(72), 72, 8));
    }

    // Checks disconnected paths below the target.
    // This avoids refresh requests for paths that are newly connected but still short.
    #[test]
    fn finalized_refresh_predicate_does_not_trigger_when_reconnected_path_is_short() {
        assert!(!should_fetch_finalized_header(None, Some(71), 72, 8));
    }

    // Checks transitions from connected to disconnected ancestry.
    // This avoids treating missing ancestry as a retry boundary.
    #[test]
    fn finalized_refresh_predicate_does_not_trigger_when_path_becomes_disconnected() {
        assert!(!should_fetch_finalized_header(Some(80), None, 72, 8));
    }

    // Queries from the finalized anchor.
    // This establishes the empty overlay baseline for callers that already have the finalized snapshot.
    #[test]
    fn latest_complete_pool_state_update_from_finalized_returns_empty_update() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let state = empty_state_at(finalized_hash);

        let update = resolved_complete_pool_state_update_from(&state, finalized_hash);

        assert_eq!(
            update,
            Some(complete_pool_state_update(finalized_hash, HashMap::new()))
        );
    }

    // Anchors the tip-relative wrapper at the canonical tip.
    // This keeps optimization dispatch reading the same overlay as an explicit tip query.
    #[test]
    fn latest_complete_pool_state_update_anchors_at_canonical_tip() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let state = empty_state_at(finalized_hash);

        assert_eq!(
            resolved_complete_pool_state_update(&state),
            resolved_complete_pool_state_update_from(&state, state.canonical_tip)
        );
        assert_eq!(
            resolved_complete_pool_state_update(&state),
            Some(complete_pool_state_update(finalized_hash, HashMap::new()))
        );
    }

    // Queries an absent block hash.
    // This prevents callers from treating unknown ancestry as a usable optimization point.
    #[test]
    fn latest_complete_pool_state_update_from_absent_block_returns_none() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let state = empty_state_at(finalized_hash);

        assert_eq!(
            resolved_complete_pool_state_update_from(&state, BlockHash::with_last_byte(2)),
            None
        );
    }

    // Queries a block whose parent is not tracked and not finalized.
    // This keeps disconnected ancestry from producing partial overlays.
    #[test]
    fn latest_complete_pool_state_update_from_disconnected_path_returns_none() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let block_hash = BlockHash::with_last_byte(3);
        let mut state = empty_state_at(finalized_hash);

        state
            .blocks
            .0
            .insert(block_hash, block_with_parent(missing_parent_hash));

        assert_eq!(
            resolved_complete_pool_state_update_from(&state, block_hash),
            None
        );
    }

    // Queries a cyclic parent graph.
    // This protects the pure query from returning data from non-terminating ancestry.
    #[test]
    fn latest_complete_pool_state_update_from_cyclic_path_returns_none() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let first_hash = BlockHash::with_last_byte(2);
        let second_hash = BlockHash::with_last_byte(3);
        let mut state = empty_state_at(finalized_hash);

        state
            .blocks
            .0
            .insert(first_hash, block_with_parent(second_hash));
        state
            .blocks
            .0
            .insert(second_hash, block_with_parent(first_hash));

        assert_eq!(
            resolved_complete_pool_state_update_from(&state, first_hash),
            None
        );
    }

    // Stops at the last complete block before unknown logs.
    // This lets optimization use the freshest known-good prefix without waiting for later blocks.
    #[test]
    fn latest_complete_pool_state_update_stops_before_unknown_logs() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let complete_hash = BlockHash::with_last_byte(2);
        let unknown_hash = BlockHash::with_last_byte(3);
        let candidate = pool_candidate_address(4);
        let pool = PoolRef { key: candidate, chain: ChainKey::Ethereum };
        let snapshot = pool_state(5);
        let mut state = empty_state_at(finalized_hash);

        state.pool_registry = TrustedPoolRegistry::new().with_metadata_results(
            ChainKey::Ethereum,
            HashMap::from([(candidate, Ok(pool_metadata(1, 2, UniswapV3Fee::Fee3000)))]),
        );
        state.blocks.0.insert(
            complete_hash,
            resolved_block_with_snapshots(
                finalized_hash,
                HashSet::from([candidate]),
                HashMap::from([(pool, snapshot.clone())]),
            ),
        );
        state
            .blocks
            .0
            .insert(unknown_hash, block_with_parent(complete_hash));

        assert_eq!(
            resolved_complete_pool_state_update_from(&state, unknown_hash),
            Some(complete_pool_state_update(
                complete_hash,
                HashMap::from([(pool, snapshot)])
            ))
        );
    }

    // Stops at the last complete block before an unvalidated pool candidate.
    // This prevents a topic-matching emitter from being treated as a ready pool.
    #[test]
    fn latest_complete_pool_state_update_stops_before_pending_pool_validation() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let complete_hash = BlockHash::with_last_byte(2);
        let pending_hash = BlockHash::with_last_byte(3);
        let verified = pool_candidate_address(4);
        let pending = pool_candidate_address(5);
        let pool = PoolRef { key: verified, chain: ChainKey::Ethereum };
        let snapshot = pool_state(6);
        let mut state = empty_state_at(finalized_hash);

        state.pool_registry = TrustedPoolRegistry::new().with_metadata_results(
            ChainKey::Ethereum,
            HashMap::from([(verified, Ok(pool_metadata(1, 2, UniswapV3Fee::Fee3000)))]),
        );
        state.blocks.0.insert(
            complete_hash,
            resolved_block_with_snapshots(
                finalized_hash,
                HashSet::from([verified]),
                HashMap::from([(pool, snapshot.clone())]),
            ),
        );
        state.blocks.0.insert(
            pending_hash,
            resolved_block_with_snapshots(complete_hash, HashSet::from([pending]), HashMap::new()),
        );

        assert_eq!(
            resolved_complete_pool_state_update_from(&state, pending_hash),
            Some(complete_pool_state_update(
                complete_hash,
                HashMap::from([(pool, snapshot)])
            ))
        );
    }

    // Resolves rejected candidates without requiring pool data.
    // This keeps non-pool emitters from blocking the latest complete block.
    #[test]
    fn latest_complete_pool_state_update_accepts_rejected_candidates_without_snapshots() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let rejected = pool_candidate_address(3);
        let mut state = empty_state_at(finalized_hash);

        state.pool_registry = TrustedPoolRegistry::new().with_metadata_results(
            ChainKey::Ethereum,
            HashMap::from([(rejected, Err(PoolMetadataFailure::FactoryReturnedZero))]),
        );
        state.blocks.0.insert(
            block_hash,
            resolved_block_with_snapshots(
                finalized_hash,
                HashSet::from([rejected]),
                HashMap::new(),
            ),
        );

        assert_eq!(
            resolved_complete_pool_state_update_from(&state, block_hash),
            Some(complete_pool_state_update(block_hash, HashMap::new()))
        );
    }

    // Uses a snapshot from the same block that first marks the pool affected.
    // This is the common case for pool-data requests made at the affected block.
    #[test]
    fn latest_complete_pool_state_update_same_block_snapshot_satisfies_log() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let candidate = pool_candidate_address(3);
        let pool = PoolRef { key: candidate, chain: ChainKey::Ethereum };
        let snapshot = pool_state(4);
        let mut state = empty_state_at(finalized_hash);

        state.pool_registry = TrustedPoolRegistry::new().with_metadata_results(
            ChainKey::Ethereum,
            HashMap::from([(candidate, Ok(pool_metadata(1, 2, UniswapV3Fee::Fee3000)))]),
        );
        state.blocks.0.insert(
            block_hash,
            resolved_block_with_snapshots(
                finalized_hash,
                HashSet::from([candidate]),
                HashMap::from([(pool, snapshot.clone())]),
            ),
        );

        assert_eq!(
            resolved_complete_pool_state_update_from(&state, block_hash),
            Some(complete_pool_state_update(
                block_hash,
                HashMap::from([(pool, snapshot)])
            ))
        );
    }

    // Carries a valid snapshot forward across later blocks where the pool is not affected.
    // This lets callers use the newest complete block without refetching unchanged pools.
    #[test]
    fn latest_complete_pool_state_update_carries_snapshot_until_pool_is_affected_again() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let affected_hash = BlockHash::with_last_byte(2);
        let later_hash = BlockHash::with_last_byte(3);
        let candidate = pool_candidate_address(4);
        let pool = PoolRef { key: candidate, chain: ChainKey::Ethereum };
        let snapshot = pool_state(5);
        let mut state = empty_state_at(finalized_hash);

        state.pool_registry = TrustedPoolRegistry::new().with_metadata_results(
            ChainKey::Ethereum,
            HashMap::from([(candidate, Ok(pool_metadata(1, 2, UniswapV3Fee::Fee3000)))]),
        );
        state.blocks.0.insert(
            affected_hash,
            resolved_block_with_snapshots(
                finalized_hash,
                HashSet::from([candidate]),
                HashMap::from([(pool, snapshot.clone())]),
            ),
        );
        state.blocks.0.insert(
            later_hash,
            resolved_block_with_snapshots(affected_hash, HashSet::new(), HashMap::new()),
        );

        assert_eq!(
            resolved_complete_pool_state_update_from(&state, later_hash),
            Some(complete_pool_state_update(
                later_hash,
                HashMap::from([(pool, snapshot)])
            ))
        );
    }

    // Invalidates a prior snapshot when the pool is affected again.
    // This prevents stale pool states from being used for newer complete blocks.
    #[test]
    fn latest_complete_pool_state_update_later_log_requires_new_snapshot() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let first_hash = BlockHash::with_last_byte(2);
        let second_hash = BlockHash::with_last_byte(3);
        let third_hash = BlockHash::with_last_byte(4);
        let candidate = pool_candidate_address(5);
        let pool = PoolRef { key: candidate, chain: ChainKey::Ethereum };
        let first_snapshot = pool_state(6);
        let mut state = empty_state_at(finalized_hash);

        state.pool_registry = TrustedPoolRegistry::new().with_metadata_results(
            ChainKey::Ethereum,
            HashMap::from([(candidate, Ok(pool_metadata(1, 2, UniswapV3Fee::Fee3000)))]),
        );
        state.blocks.0.insert(
            first_hash,
            resolved_block_with_snapshots(
                finalized_hash,
                HashSet::from([candidate]),
                HashMap::from([(pool, first_snapshot.clone())]),
            ),
        );
        state.blocks.0.insert(
            second_hash,
            resolved_block_with_snapshots(first_hash, HashSet::from([candidate]), HashMap::new()),
        );
        state.blocks.0.insert(
            third_hash,
            resolved_block_with_snapshots(second_hash, HashSet::new(), HashMap::new()),
        );

        assert_eq!(
            resolved_complete_pool_state_update_from(&state, third_hash),
            Some(complete_pool_state_update(
                first_hash,
                HashMap::from([(pool, first_snapshot)])
            ))
        );
    }

    // Restores completeness once a newer snapshot appears after a later pool log.
    // This checks the overlay uses the latest usable state in the scanned prefix.
    #[test]
    fn latest_complete_pool_state_update_uses_new_snapshot_after_later_log() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let first_hash = BlockHash::with_last_byte(2);
        let second_hash = BlockHash::with_last_byte(3);
        let third_hash = BlockHash::with_last_byte(4);
        let candidate = pool_candidate_address(5);
        let pool = PoolRef { key: candidate, chain: ChainKey::Ethereum };
        let first_snapshot = pool_state(6);
        let third_snapshot = pool_state(7);
        let mut state = empty_state_at(finalized_hash);

        state.pool_registry = TrustedPoolRegistry::new().with_metadata_results(
            ChainKey::Ethereum,
            HashMap::from([(candidate, Ok(pool_metadata(1, 2, UniswapV3Fee::Fee3000)))]),
        );
        state.blocks.0.insert(
            first_hash,
            resolved_block_with_snapshots(
                finalized_hash,
                HashSet::from([candidate]),
                HashMap::from([(pool, first_snapshot)]),
            ),
        );
        state.blocks.0.insert(
            second_hash,
            resolved_block_with_snapshots(first_hash, HashSet::from([candidate]), HashMap::new()),
        );
        state.blocks.0.insert(
            third_hash,
            resolved_block_with_snapshots(
                second_hash,
                HashSet::new(),
                HashMap::from([(pool, third_snapshot.clone())]),
            ),
        );

        assert_eq!(
            resolved_complete_pool_state_update_from(&state, third_hash),
            Some(complete_pool_state_update(
                third_hash,
                HashMap::from([(pool, third_snapshot)])
            ))
        );
    }

    // Ignores pool-data failures when deciding whether a pool is covered.
    // This keeps failed snapshots from masquerading as usable pool state.
    #[test]
    fn latest_complete_pool_state_update_does_not_count_pool_data_failure_as_snapshot() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let candidate = pool_candidate_address(3);
        let pool = PoolRef { key: candidate, chain: ChainKey::Ethereum };
        let mut state = empty_state_at(finalized_hash);

        state.pool_registry = TrustedPoolRegistry::new().with_metadata_results(
            ChainKey::Ethereum,
            HashMap::from([(candidate, Ok(pool_metadata(1, 2, UniswapV3Fee::Fee3000)))]),
        );
        state.blocks.0.insert(
            block_hash,
            BlockNode {
                logs_bloom: None,
                parent_hash: finalized_hash,
                pool_logs: PoolLogsStatus::Resolved(HashSet::from([candidate])),
                pool_snapshots: HashMap::new(),
                pool_data_failures: HashMap::from([(
                    pool,
                    PoolDataFailure::CallFailed(PoolDataCall::Slot0),
                )]),
            },
        );

        assert_eq!(
            resolved_complete_pool_state_update_from(&state, block_hash),
            Some(complete_pool_state_update(finalized_hash, HashMap::new()))
        );
    }

    // Ignores snapshots for pools that were not affected in the scanned path.
    // This keeps the returned overlay limited to changes since the finalized snapshot.
    #[test]
    fn latest_complete_pool_state_update_excludes_unaffected_pool_snapshots() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let candidate = pool_candidate_address(3);
        let affected_pool = PoolRef { key: candidate, chain: ChainKey::Ethereum };
        let unaffected_pool = pool_address(4);
        let affected_snapshot = pool_state(5);
        let unaffected_snapshot = pool_state(6);
        let mut state = empty_state_at(finalized_hash);

        state.pool_registry = TrustedPoolRegistry::new().with_metadata_results(
            ChainKey::Ethereum,
            HashMap::from([(candidate, Ok(pool_metadata(1, 2, UniswapV3Fee::Fee3000)))]),
        );
        state.blocks.0.insert(
            block_hash,
            resolved_block_with_snapshots(
                finalized_hash,
                HashSet::from([candidate]),
                HashMap::from([
                    (affected_pool, affected_snapshot.clone()),
                    (unaffected_pool, unaffected_snapshot),
                ]),
            ),
        );

        assert_eq!(
            resolved_complete_pool_state_update_from(&state, block_hash),
            Some(complete_pool_state_update(
                block_hash,
                HashMap::from([(affected_pool, affected_snapshot)])
            ))
        );
    }

    // Starts the query from an older non-tip block.
    // This ensures newer descendant data is ignored when callers request an earlier point.
    #[test]
    fn latest_complete_pool_state_update_from_older_start_ignores_newer_descendants() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let first_hash = BlockHash::with_last_byte(2);
        let second_hash = BlockHash::with_last_byte(3);
        let candidate = pool_candidate_address(4);
        let pool = PoolRef { key: candidate, chain: ChainKey::Ethereum };
        let first_snapshot = pool_state(5);
        let second_snapshot = pool_state(6);
        let mut state = empty_state_at(finalized_hash);

        state.pool_registry = TrustedPoolRegistry::new().with_metadata_results(
            ChainKey::Ethereum,
            HashMap::from([(candidate, Ok(pool_metadata(1, 2, UniswapV3Fee::Fee3000)))]),
        );
        state.blocks.0.insert(
            first_hash,
            resolved_block_with_snapshots(
                finalized_hash,
                HashSet::from([candidate]),
                HashMap::from([(pool, first_snapshot.clone())]),
            ),
        );
        state.blocks.0.insert(
            second_hash,
            resolved_block_with_snapshots(
                first_hash,
                HashSet::from([candidate]),
                HashMap::from([(pool, second_snapshot)]),
            ),
        );

        assert_eq!(
            resolved_complete_pool_state_update_from(&state, first_hash),
            Some(complete_pool_state_update(
                first_hash,
                HashMap::from([(pool, first_snapshot)])
            ))
        );
    }

    // Compacts a complete finalized target into the finalized snapshot.
    // This keeps recent block storage bounded while preserving pool states needed after old blocks are removed.
    #[test]
    fn finalized_block_observed_compacts_complete_target_and_merges_pool_snapshots() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let target_hash = BlockHash::with_last_byte(2);
        let tip_hash = BlockHash::with_last_byte(3);
        let candidate = pool_candidate_address(4);
        let affected_pool = PoolRef { key: candidate, chain: ChainKey::Ethereum };
        let unaffected_pool = pool_address(5);
        let old_affected_snapshot = pool_state(6);
        let new_affected_snapshot = pool_state(7);
        let unaffected_snapshot = pool_state(8);
        let mut state = empty_state_at(finalized_hash);

        state.canonical_tip = tip_hash;
        state.finalized_state.pool_snapshots = HashMap::from([
            (affected_pool, old_affected_snapshot),
            (unaffected_pool, unaffected_snapshot.clone()),
        ]);
        state.pool_registry = TrustedPoolRegistry::new().with_metadata_results(
            ChainKey::Ethereum,
            HashMap::from([(candidate, Ok(pool_metadata(1, 2, UniswapV3Fee::Fee3000)))]),
        );
        state.blocks.0.insert(
            target_hash,
            resolved_block_with_snapshots(
                finalized_hash,
                HashSet::from([candidate]),
                HashMap::from([(affected_pool, new_affected_snapshot.clone())]),
            ),
        );
        state.blocks.0.insert(
            tip_hash,
            resolved_block_with_snapshots(target_hash, HashSet::new(), HashMap::new()),
        );

        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::FinalizedBlockObserved {
                block_hash: target_hash,
            },
        );

        assert!(effects.is_empty());
        assert_eq!(state.finalized_state.block_hash, target_hash);
        assert_eq!(
            &state.finalized_state.pool_snapshots,
            &HashMap::from([
                (affected_pool, new_affected_snapshot),
                (unaffected_pool, unaffected_snapshot),
            ])
        );
        assert!(!state.blocks.0.contains_key(&target_hash));
        assert!(state.blocks.0.contains_key(&tip_hash));
        assert_eq!(state.canonical_tip, tip_hash);
        assert_state_invariants(&state);
    }

    // Compacts to the newest complete prefix when the observed finalized target is still incomplete.
    // This lets finality make progress without waiting for later affected-pool snapshots.
    #[test]
    fn finalized_block_observed_compacts_to_latest_earlier_complete_block() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let complete_hash = BlockHash::with_last_byte(2);
        let incomplete_hash = BlockHash::with_last_byte(3);
        let target_hash = BlockHash::with_last_byte(4);
        let candidate = pool_candidate_address(5);
        let pool = PoolRef { key: candidate, chain: ChainKey::Ethereum };
        let snapshot = pool_state(6);
        let mut state = empty_state_at(finalized_hash);

        state.canonical_tip = target_hash;
        state.pool_registry = TrustedPoolRegistry::new().with_metadata_results(
            ChainKey::Ethereum,
            HashMap::from([(candidate, Ok(pool_metadata(1, 2, UniswapV3Fee::Fee3000)))]),
        );
        state.blocks.0.insert(
            complete_hash,
            resolved_block_with_snapshots(
                finalized_hash,
                HashSet::from([candidate]),
                HashMap::from([(pool, snapshot.clone())]),
            ),
        );
        state.blocks.0.insert(
            incomplete_hash,
            resolved_block_with_snapshots(
                complete_hash,
                HashSet::from([candidate]),
                HashMap::new(),
            ),
        );
        state.blocks.0.insert(
            target_hash,
            resolved_block_with_snapshots(incomplete_hash, HashSet::new(), HashMap::new()),
        );

        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::FinalizedBlockObserved {
                block_hash: target_hash,
            },
        );

        assert!(effects.is_empty());
        assert_eq!(state.finalized_state.block_hash, complete_hash);
        assert_eq!(
            &state.finalized_state.pool_snapshots,
            &HashMap::from([(pool, snapshot)])
        );
        assert!(!state.blocks.0.contains_key(&complete_hash));
        assert!(state.blocks.0.contains_key(&incomplete_hash));
        assert!(state.blocks.0.contains_key(&target_hash));
        assert_state_invariants(&state);
    }

    // Leaves state unchanged when no block past the current finalized anchor is complete.
    // This avoids treating failed compaction attempts as pending work.
    #[test]
    fn finalized_block_observed_with_only_finalized_complete_is_noop() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let target_hash = BlockHash::with_last_byte(2);
        let mut state = empty_state_at(finalized_hash);

        state.canonical_tip = target_hash;
        state
            .blocks
            .0
            .insert(target_hash, block_with_parent(finalized_hash));

        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::FinalizedBlockObserved {
                block_hash: target_hash,
            },
        );

        assert!(effects.is_empty());
        assert_eq!(state.finalized_state.block_hash, finalized_hash);
        assert!(state.blocks.0.contains_key(&target_hash));
        assert_eq!(state.canonical_tip, target_hash);
        assert_state_invariants(&state);
    }

    // Leaves state unchanged when the observed finalized hash is not connected to the finalized anchor.
    // This prevents partial or malformed ancestry from changing the immutable snapshot boundary.
    #[test]
    fn finalized_block_observed_with_disconnected_target_is_noop() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let target_hash = BlockHash::with_last_byte(3);
        let mut state = empty_state_at(finalized_hash);

        state.canonical_tip = target_hash;
        state
            .blocks
            .0
            .insert(target_hash, block_with_parent(missing_parent_hash));

        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::FinalizedBlockObserved {
                block_hash: target_hash,
            },
        );

        assert!(effects.is_empty());
        assert_eq!(state.finalized_state.block_hash, finalized_hash);
        assert!(state.blocks.0.contains_key(&target_hash));
        assert_state_invariants(&state);
    }

    // Leaves state unchanged when the observed finalized hash is on a non-canonical branch.
    // This keeps compaction aligned with the kernel's current canonical tip.
    #[test]
    fn finalized_block_observed_for_non_canonical_target_is_noop() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let canonical_hash = BlockHash::with_last_byte(2);
        let side_hash = BlockHash::with_last_byte(3);
        let mut state = empty_state_at(finalized_hash);

        state.canonical_tip = canonical_hash;
        state.blocks.0.insert(
            canonical_hash,
            resolved_block_with_snapshots(finalized_hash, HashSet::new(), HashMap::new()),
        );
        state.blocks.0.insert(
            side_hash,
            resolved_block_with_snapshots(finalized_hash, HashSet::new(), HashMap::new()),
        );

        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::FinalizedBlockObserved {
                block_hash: side_hash,
            },
        );

        assert!(effects.is_empty());
        assert_eq!(state.finalized_state.block_hash, finalized_hash);
        assert!(state.blocks.0.contains_key(&canonical_hash));
        assert!(state.blocks.0.contains_key(&side_hash));
        assert_state_invariants(&state);
    }

    // Prunes old blocks and their pending work after successful compaction.
    // This ensures stale responses for compacted-away blocks cannot mutate new finalized state.
    #[test]
    fn finalized_block_observed_prunes_removed_blocks_and_their_pending_requests() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let target_hash = BlockHash::with_last_byte(2);
        let retained_hash = BlockHash::with_last_byte(3);
        let side_hash = BlockHash::with_last_byte(4);
        let retained_token = token_address(5);
        let candidate = pool_candidate_address(6);
        let pool = PoolRef { key: candidate, chain: ChainKey::Ethereum };
        let mut state = empty_state_at(finalized_hash);

        state.canonical_tip = retained_hash;
        state.pool_registry = TrustedPoolRegistry::new().with_metadata_results(
            ChainKey::Ethereum,
            HashMap::from([(candidate, Ok(pool_metadata(1, 2, UniswapV3Fee::Fee3000)))]),
        );
        state.blocks.0.insert(
            target_hash,
            resolved_block_with_snapshots(
                finalized_hash,
                HashSet::from([candidate]),
                HashMap::from([(pool, pool_state(7))]),
            ),
        );
        state
            .blocks
            .0
            .insert(retained_hash, block_with_parent(target_hash));
        state
            .blocks
            .0
            .insert(side_hash, block_with_parent(finalized_hash));

        let (pending_requests, removed_logs_id) = state.pending_requests.with_new_request(
            GetBlockLogs {
                block_hash: target_hash,
            },
            tick(0),
        );
        let (pending_requests, removed_pool_data_id) = pending_requests.with_new_request(
            GetPoolData {
                at: target_hash,
                pools: HashSet::from([pool]),
            },
            tick(0),
        );
        let (pending_requests, removed_metadata_id) = pending_requests.with_new_request(
            GetPoolMetadata {
                at: side_hash,
                candidates: HashSet::from([candidate]),
            },
            tick(0),
        );
        let (pending_requests, retained_logs_id) = pending_requests.with_new_request(
            GetBlockLogs {
                block_hash: retained_hash,
            },
            tick(0),
        );
        let (pending_requests, retained_tokens_id) = pending_requests.with_new_request(
            GetTokenMetadata {
                at: retained_hash,
                tokens: HashSet::from([retained_token]),
            },
            tick(0),
        );
        state.pending_requests = pending_requests;

        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::FinalizedBlockObserved {
                block_hash: target_hash,
            },
        );

        assert!(effects.is_empty());
        assert!(!state.blocks.0.contains_key(&target_hash));
        assert!(!state.blocks.0.contains_key(&side_hash));
        assert!(state.blocks.0.contains_key(&retained_hash));
        assert!(!state.pending_requests.contains(&removed_logs_id));
        assert!(!state.pending_requests.contains(&removed_pool_data_id));
        assert!(!state.pending_requests.contains(&removed_metadata_id));
        assert!(state.pending_requests.contains(&retained_logs_id));
        assert!(state.pending_requests.contains(&retained_tokens_id));
        assert_state_invariants(&state);
    }

    // Does not retry a failed compaction from Tick after the target later becomes complete.
    // This documents that failed compaction attempts are intentionally forgotten.
    #[test]
    fn tick_does_not_retry_failed_finalized_compaction() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let target_hash = BlockHash::with_last_byte(2);
        let mut state = empty_state_at(finalized_hash);

        state.canonical_tip = target_hash;
        state
            .blocks
            .0
            .insert(target_hash, block_with_parent(finalized_hash));

        let (mut state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::FinalizedBlockObserved {
                block_hash: target_hash,
            },
        );

        assert!(effects.is_empty());
        state.blocks.0.insert(
            target_hash,
            resolved_block_with_snapshots(finalized_hash, HashSet::new(), HashMap::new()),
        );

        let (state, effects) = transition(ChainKey::Ethereum, state, Event::Tick);

        assert!(effects.is_empty());
        assert_eq!(state.finalized_state.block_hash, finalized_hash);
        assert!(state.blocks.0.contains_key(&target_hash));
        assert_state_invariants(&state);
    }

    // Delivers successful pool-data results for an active request.
    // This ensures snapshots are stored and the matching pending request is retired.
    #[test]
    fn pool_data_received_stores_snapshots_and_removes_pending_request() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let first_pool = pool_address(3);
        let second_pool = pool_address(4);
        let first_state = pool_state(5);
        let second_state = pool_state(6);
        let requested_pools = HashSet::from([first_pool, second_pool]);
        let mut state = empty_state_at(finalized_hash);

        state.canonical_tip = block_hash;
        state
            .blocks
            .0
            .insert(block_hash, block_with_parent(finalized_hash));
        let (pending_requests, request_id) = state.pending_requests.with_new_request(
            GetPoolData {
                at: block_hash,
                pools: requested_pools,
            },
            state.tick,
        );
        state.pending_requests = pending_requests;

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::PoolDataReceived {
                request_id,
                pools: HashMap::from([
                    (first_pool, Ok(first_state.clone())),
                    (second_pool, Ok(second_state.clone())),
                ]),
            },
        );

        assert!(effects.is_empty());
        assert!(!next_state.pending_requests.contains(&request_id));
        assert_pool_snapshot(&next_state, block_hash, first_pool, &first_state);
        assert_pool_snapshot(&next_state, block_hash, second_pool, &second_state);
        assert_no_pool_failure(&next_state, block_hash, first_pool);
        assert_no_pool_failure(&next_state, block_hash, second_pool);
        assert_state_invariants(&next_state);
    }

    // Delivers pool data for pools accumulated over multiple affected blocks.
    // This ensures all requested results apply to the target snapshot block.
    #[test]
    fn pool_data_received_applies_requested_snapshots_not_present_in_block_logs() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let logged_pool = pool_address(3);
        let logged_candidate =
            ProtocolPoolKey::UniswapV3(logged_pool.uniswap_v3_address().expect("v3 pool"));
        let requested_pool = pool_address(4);
        let requested_state = pool_state(5);
        let mut state = empty_state_at(finalized_hash);

        state.canonical_tip = block_hash;
        state.blocks.0.insert(
            block_hash,
            BlockNode {
                logs_bloom: None,
                parent_hash: finalized_hash,
                pool_logs: PoolLogsStatus::Resolved(HashSet::from([logged_candidate])),
                pool_snapshots: HashMap::new(),
                pool_data_failures: HashMap::new(),
            },
        );
        let (pending_requests, request_id) = state.pending_requests.with_new_request(
            GetPoolData {
                at: block_hash,
                pools: HashSet::from([requested_pool]),
            },
            state.tick,
        );
        state.pending_requests = pending_requests;

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::PoolDataReceived {
                request_id,
                pools: HashMap::from([(requested_pool, Ok(requested_state.clone()))]),
            },
        );

        assert!(effects.is_empty());
        assert_pool_snapshot(&next_state, block_hash, requested_pool, &requested_state);
        assert_no_pool_snapshot(&next_state, block_hash, logged_pool);
        assert_state_invariants(&next_state);
    }

    // Delivers pool-data results containing unrequested entries.
    // This protects block snapshots from stale or overbroad response data.
    #[test]
    fn pool_data_received_ignores_unrequested_result_entries() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let requested_pool = pool_address(3);
        let extra_pool = pool_address(4);
        let requested_state = pool_state(5);
        let extra_state = pool_state(6);
        let mut state = empty_state_at(finalized_hash);

        state.canonical_tip = block_hash;
        state
            .blocks
            .0
            .insert(block_hash, block_with_parent(finalized_hash));
        let (pending_requests, request_id) = state.pending_requests.with_new_request(
            GetPoolData {
                at: block_hash,
                pools: HashSet::from([requested_pool]),
            },
            state.tick,
        );
        state.pending_requests = pending_requests;

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::PoolDataReceived {
                request_id,
                pools: HashMap::from([
                    (requested_pool, Ok(requested_state.clone())),
                    (extra_pool, Ok(extra_state)),
                ]),
            },
        );

        assert!(effects.is_empty());
        assert_pool_snapshot(&next_state, block_hash, requested_pool, &requested_state);
        assert_no_pool_snapshot(&next_state, block_hash, extra_pool);
        assert_no_pool_failure(&next_state, block_hash, extra_pool);
        assert_state_invariants(&next_state);
    }

    // Delivers a requested per-pool failure.
    // This records deterministic failures while avoiding immediate retry loops.
    #[test]
    fn pool_data_received_stores_requested_failures_without_retry_effects() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let pool = pool_address(3);
        let failure = PoolDataFailure::CallFailed(PoolDataCall::Slot0);
        let mut state = empty_state_at(finalized_hash);

        state.canonical_tip = block_hash;
        state
            .blocks
            .0
            .insert(block_hash, block_with_parent(finalized_hash));
        let (pending_requests, request_id) = state.pending_requests.with_new_request(
            GetPoolData {
                at: block_hash,
                pools: HashSet::from([pool]),
            },
            state.tick,
        );
        state.pending_requests = pending_requests;

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::PoolDataReceived {
                request_id,
                pools: HashMap::from([(pool, Err(failure.clone()))]),
            },
        );

        assert!(effects.is_empty());
        assert!(!next_state.pending_requests.contains(&request_id));
        assert_no_pool_snapshot(&next_state, block_hash, pool);
        assert_pool_failure(&next_state, block_hash, pool, &failure);
        assert_state_invariants(&next_state);
    }

    // Records a failure on one block and then observes a newer dirty block.
    // This allows later current-tip requests to recover from earlier failures.
    #[test]
    fn pool_data_received_success_replaces_previous_failure() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let pool = pool_address(3);
        let state_snapshot = pool_state(4);
        let mut state = empty_state_at(finalized_hash);

        state.canonical_tip = block_hash;
        state
            .blocks
            .0
            .insert(block_hash, block_with_parent(finalized_hash));
        state
            .blocks
            .0
            .get_mut(&block_hash)
            .expect("block must be present")
            .pool_data_failures
            .insert(pool, PoolDataFailure::DecodeFailed(PoolDataCall::Liquidity));
        let (pending_requests, request_id) = state.pending_requests.with_new_request(
            GetPoolData {
                at: block_hash,
                pools: HashSet::from([pool]),
            },
            state.tick,
        );
        state.pending_requests = pending_requests;

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::PoolDataReceived {
                request_id,
                pools: HashMap::from([(pool, Ok(state_snapshot.clone()))]),
            },
        );

        assert!(effects.is_empty());
        assert_pool_snapshot(&next_state, block_hash, pool, &state_snapshot);
        assert_no_pool_failure(&next_state, block_hash, pool);
        assert_state_invariants(&next_state);
    }

    // Delivers a later failure for a pool that already has a snapshot.
    // This keeps the last successful snapshot instead of replacing it with failure state.
    #[test]
    fn pool_data_received_failure_does_not_overwrite_existing_success() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let pool = pool_address(3);
        let existing_state = pool_state(4);
        let mut state = empty_state_at(finalized_hash);

        state.canonical_tip = block_hash;
        state
            .blocks
            .0
            .insert(block_hash, block_with_parent(finalized_hash));
        state
            .blocks
            .0
            .get_mut(&block_hash)
            .expect("block must be present")
            .pool_snapshots
            .insert(pool, existing_state.clone());
        let (pending_requests, request_id) = state.pending_requests.with_new_request(
            GetPoolData {
                at: block_hash,
                pools: HashSet::from([pool]),
            },
            state.tick,
        );
        state.pending_requests = pending_requests;

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::PoolDataReceived {
                request_id,
                pools: HashMap::from([(
                    pool,
                    Err(PoolDataFailure::DecodeFailed(PoolDataCall::Slot0)),
                )]),
            },
        );

        assert!(effects.is_empty());
        assert_pool_snapshot(&next_state, block_hash, pool, &existing_state);
        assert_no_pool_failure(&next_state, block_hash, pool);
        assert_state_invariants(&next_state);
    }

    // Delivers pool-data results without an active request.
    // This protects snapshots from unsolicited or stale responses.
    #[test]
    fn pool_data_received_for_unknown_request_is_noop() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let pool = pool_address(3);
        let mut state = empty_state_at(finalized_hash);

        state.canonical_tip = block_hash;
        state
            .blocks
            .0
            .insert(block_hash, block_with_parent(finalized_hash));

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::PoolDataReceived {
                request_id: RequestId::from_raw_for_test(99),
                pools: HashMap::from([(pool, Ok(pool_state(4)))]),
            },
        );

        assert!(effects.is_empty());
        assert!(next_state.pending_requests.is_empty_for_test());
        assert_no_pool_snapshot(&next_state, block_hash, pool);
        assert_no_pool_failure(&next_state, block_hash, pool);
        assert_state_invariants(&next_state);
    }

    // Delivers pool-data after its target block was reset away.
    // This prevents late responses from recreating removed block snapshots.
    #[test]
    fn pool_data_received_for_missing_block_consumes_request_without_inserting_block() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_block_hash = BlockHash::with_last_byte(2);
        let pool = pool_address(3);
        let mut state = empty_state_at(finalized_hash);
        let (pending_requests, request_id) = state.pending_requests.with_new_request(
            GetPoolData {
                at: missing_block_hash,
                pools: HashSet::from([pool]),
            },
            state.tick,
        );
        state.pending_requests = pending_requests;

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::PoolDataReceived {
                request_id,
                pools: HashMap::from([(pool, Ok(pool_state(4)))]),
            },
        );

        assert!(effects.is_empty());
        assert_empty_initial_state_at(&next_state, finalized_hash);
        assert!(!next_state.blocks.0.contains_key(&missing_block_hash));
        assert_state_invariants(&next_state);
    }

    // Fails an active pool-data request at the transport level.
    // This keeps the same block and pool scope retryable with a fresh id.
    #[test]
    fn pool_data_request_failure_retries_original_request() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let pools = HashSet::from([pool_address(3), pool_address(4)]);
        let mut state = empty_state_at(finalized_hash);

        state.canonical_tip = block_hash;
        state
            .blocks
            .0
            .insert(block_hash, block_with_parent(finalized_hash));
        let (pending_requests, request_id) = state.pending_requests.with_new_request(
            GetPoolData {
                at: block_hash,
                pools: pools.clone(),
            },
            state.tick,
        );
        state.pending_requests = pending_requests;

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::RequestFailed {
                request_id: AnyRequestId::PoolData(request_id),
            },
        );

        let retry_request_id = assert_single_pool_data_request_effect(&effects, block_hash, &pools);
        assert_ne!(retry_request_id, request_id);
        assert!(!next_state.pending_requests.contains(&request_id));
        assert!(next_state.pending_requests.contains(&retry_request_id));
        assert_state_invariants(&next_state);
    }

    // Fails an active pool-metadata request at the transport level.
    // This preserves candidate validation scope across retry replacement.
    #[test]
    fn pool_metadata_request_failure_retries_original_request() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let candidates = HashSet::from([pool_candidate_address(3), pool_candidate_address(4)]);
        let mut state = empty_state_at(finalized_hash);

        state.canonical_tip = block_hash;
        state
            .blocks
            .0
            .insert(block_hash, block_with_parent(finalized_hash));
        let (pending_requests, request_id) = state.pending_requests.with_new_request(
            GetPoolMetadata {
                at: block_hash,
                candidates: candidates.clone(),
            },
            state.tick,
        );
        state.pending_requests = pending_requests;

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::RequestFailed {
                request_id: AnyRequestId::PoolMetadata(request_id),
            },
        );

        let retry_request_id =
            assert_single_pool_metadata_request_effect(&effects, block_hash, &candidates);
        assert_ne!(retry_request_id, request_id);
        assert!(!next_state.pending_requests.contains(&request_id));
        assert!(next_state.pending_requests.contains(&retry_request_id));
        assert_state_invariants(&next_state);
    }

    // Fails an active token-metadata request at the transport level.
    // This preserves token metadata scope across retry replacement.
    #[test]
    fn token_metadata_request_failure_retries_original_request() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let tokens = HashSet::from([token_address(3), token_address(4)]);
        let mut state = empty_state_at(finalized_hash);

        state.canonical_tip = block_hash;
        state
            .blocks
            .0
            .insert(block_hash, block_with_parent(finalized_hash));
        let (pending_requests, request_id) = state.pending_requests.with_new_request(
            GetTokenMetadata {
                at: block_hash,
                tokens: tokens.clone(),
            },
            state.tick,
        );
        state.pending_requests = pending_requests;

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::RequestFailed {
                request_id: AnyRequestId::TokenMetadata(request_id),
            },
        );

        let retry_request_id =
            assert_single_token_metadata_request_effect(&effects, block_hash, &tokens);
        assert_ne!(retry_request_id, request_id);
        assert!(!next_state.pending_requests.contains(&request_id));
        assert!(next_state.pending_requests.contains(&retry_request_id));
        assert_state_invariants(&next_state);
    }

    // Expires an active pool-metadata request.
    // This refreshes validation work without dropping candidates.
    #[test]
    fn pool_metadata_request_expiration_retries_original_request() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let candidates = HashSet::from([pool_candidate_address(3), pool_candidate_address(4)]);
        let mut state = empty_state_at(finalized_hash);

        state.canonical_tip = block_hash;
        state
            .blocks
            .0
            .insert(block_hash, block_with_parent(finalized_hash));
        let (pending_requests, request_id) = state.pending_requests.with_new_request(
            GetPoolMetadata {
                at: block_hash,
                candidates: candidates.clone(),
            },
            state.tick,
        );
        state.pending_requests = pending_requests;

        let (next_state, effects) = advance_ticks(state, REQUEST_TTL);

        let retry_request_id =
            assert_single_pool_metadata_request_effect(&effects, block_hash, &candidates);
        assert_ne!(retry_request_id, request_id);
        assert!(!next_state.pending_requests.contains(&request_id));
        assert!(next_state.pending_requests.contains(&retry_request_id));
        assert_no_active_request_is_expired(&next_state);
        assert_state_invariants(&next_state);
    }

    // Expires an active token-metadata request.
    // This refreshes token lookup work without dropping tokens.
    #[test]
    fn token_metadata_request_expiration_retries_original_request() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let tokens = HashSet::from([token_address(3), token_address(4)]);
        let mut state = empty_state_at(finalized_hash);

        state.canonical_tip = block_hash;
        state
            .blocks
            .0
            .insert(block_hash, block_with_parent(finalized_hash));
        let (pending_requests, request_id) = state.pending_requests.with_new_request(
            GetTokenMetadata {
                at: block_hash,
                tokens: tokens.clone(),
            },
            state.tick,
        );
        state.pending_requests = pending_requests;

        let (next_state, effects) = advance_ticks(state, REQUEST_TTL);

        let retry_request_id =
            assert_single_token_metadata_request_effect(&effects, block_hash, &tokens);
        assert_ne!(retry_request_id, request_id);
        assert!(!next_state.pending_requests.contains(&request_id));
        assert!(next_state.pending_requests.contains(&retry_request_id));
        assert_no_active_request_is_expired(&next_state);
        assert_state_invariants(&next_state);
    }

    // Sends a failure for an old pool-metadata id after retry replacement.
    // This prevents stale failures from creating duplicate validation retries.
    #[test]
    fn duplicate_pool_metadata_request_failure_for_old_id_is_ignored() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let candidates = HashSet::from([pool_candidate_address(3)]);
        let mut state = empty_state_at(finalized_hash);

        state.canonical_tip = block_hash;
        state
            .blocks
            .0
            .insert(block_hash, block_with_parent(finalized_hash));
        let (pending_requests, request_id) = state.pending_requests.with_new_request(
            GetPoolMetadata {
                at: block_hash,
                candidates: candidates.clone(),
            },
            state.tick,
        );
        state.pending_requests = pending_requests;

        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::RequestFailed {
                request_id: AnyRequestId::PoolMetadata(request_id),
            },
        );
        let retry_request_id =
            assert_single_pool_metadata_request_effect(&effects, block_hash, &candidates);

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::RequestFailed {
                request_id: AnyRequestId::PoolMetadata(request_id),
            },
        );

        assert!(effects.is_empty());
        assert_eq!(next_state.pending_requests.len_for_test(), 1);
        assert!(next_state.pending_requests.contains(&retry_request_id));
        assert_state_invariants(&next_state);
    }

    // Fails a log request after reset removed its block.
    // This keeps late request handling from mutating reset chain state.
    #[test]
    fn stale_block_logs_response_after_reset_does_not_resurrect_removed_block() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let head_hash = BlockHash::with_last_byte(2);
        let conflicting_parent_hash = BlockHash::with_last_byte(3);
        let logs = HashSet::from([pool_candidate_address(4)]);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: head_hash,
                parent_hash: finalized_hash,
            },
        );
        let log_request_id = assert_single_block_log_request_effect(&effects, head_hash);

        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: head_hash,
                parent_hash: conflicting_parent_hash,
            },
        );
        assert!(effects.is_empty());
        assert_chain_reset_at(&state, finalized_hash);
        assert!(state.pending_requests.contains(&log_request_id));

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::BlockLogsReceived {
                request_id: log_request_id,
                logs: pool_logs(&logs),
            },
        );

        assert!(effects.is_empty());
        assert_empty_initial_state_at(&next_state, finalized_hash);
        assert!(!next_state.blocks.0.contains_key(&head_hash));
        assert_state_invariants(&next_state);
    }

    // Delivers a matching header response for an active request.
    // This ensures success retires the header request that produced it.
    #[test]
    fn block_header_received_for_matching_request_removes_pending_request() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: head_hash,
                parent_hash: missing_parent_hash,
            },
        );
        let request_id = assert_single_block_header_request_effect(&effects, missing_parent_hash);

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::BlockHeaderReceived {
                logs_bloom: bloom_matching_any(),
                request_id,
                hash: missing_parent_hash,
                parent_hash: finalized_hash,
            },
        );

        assert_eq!(next_state.blocks.0.len(), 2);
        assert_single_block_with_parent(&next_state, head_hash, missing_parent_hash);
        assert_single_unknown_block(&next_state, missing_parent_hash, finalized_hash);
        assert!(!next_state.pending_requests.contains(&request_id));
        assert_request_hashes(
            &effects,
            HashSet::new(),
            HashSet::from([missing_parent_hash]),
        );
        assert_state_invariants(&next_state);
    }

    // Reports not-found for an active header request.
    // This treats missing canonical ancestry as a reset-worthy inconsistency.
    #[test]
    fn block_header_not_found_for_matching_request_resets_chain() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let unrelated_hash = BlockHash::with_last_byte(4);
        let mut state = empty_state_at(finalized_hash);
        state.tick = tick(7);

        let (mut state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: head_hash,
                parent_hash: missing_parent_hash,
            },
        );
        let missing_header_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);
        state = drain_block_log_effects(state, &effects);
        let unrelated_payload = GetBlockLogs {
            block_hash: unrelated_hash,
        };
        let (pending_requests, unrelated_request_id) = state
            .pending_requests
            .with_new_request(unrelated_payload.clone(), state.tick);
        state.pending_requests = pending_requests;
        let last_request_id = state.pending_requests.last_request_id_for_test();

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::BlockHeaderNotFound {
                request_id: missing_header_request_id,
            },
        );

        assert!(effects.is_empty());
        assert_chain_reset_at(&next_state, finalized_hash);
        assert!(next_state.tick == tick(7));
        assert!(next_state.pending_requests.last_request_id_for_test() == last_request_id);
        assert_eq!(next_state.pending_requests.len_for_test(), 1);
        assert!(
            !next_state
                .pending_requests
                .contains(&missing_header_request_id)
        );
        let retained_payload = next_state
            .pending_requests
            .get(&unrelated_request_id)
            .expect("unrelated request must remain pending");
        assert_eq!(
            retained_payload.payload.block_hash,
            unrelated_payload.block_hash
        );
        assert_state_invariants(&next_state);
    }

    // Reports not-found for an unknown request id.
    // This protects state from unsolicited failure notifications.
    #[test]
    fn block_header_not_found_for_unknown_request_is_noop() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: head_hash,
                parent_hash: missing_parent_hash,
            },
        );
        let pending_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);
        let state = drain_block_log_effects(state, &effects);
        let last_request_id = state.pending_requests.last_request_id_for_test();
        let dispatch_tick =
            active_request_dispatch_tick(&state.pending_requests, pending_request_id);

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::BlockHeaderNotFound {
                request_id: RequestId::from_raw_for_test(99),
            },
        );

        assert!(effects.is_empty());
        assert_eq!(next_state.canonical_tip, head_hash);
        assert_single_block_with_parent(&next_state, head_hash, missing_parent_hash);
        assert_eq!(next_state.pending_requests.len_for_test(), 1);
        assert!(next_state.pending_requests.contains(&pending_request_id));
        assert!(next_state.pending_requests.last_request_id_for_test() == last_request_id);
        assert!(
            active_request_dispatch_tick(&next_state.pending_requests, pending_request_id)
                == dispatch_tick
        );
        assert_state_invariants(&next_state);
        assert_missing_parents_for_known_blocks_are_pending(&next_state);
    }

    // Reports not-found after the request already completed successfully.
    // This avoids resetting chain state from a stale terminal response.
    #[test]
    fn late_block_header_not_found_after_success_is_ignored() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: head_hash,
                parent_hash: missing_parent_hash,
            },
        );
        let request_id = assert_single_block_header_request_effect(&effects, missing_parent_hash);
        let state = drain_block_log_effects(state, &effects);
        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::BlockHeaderReceived {
                logs_bloom: bloom_matching_any(),
                request_id,
                hash: missing_parent_hash,
                parent_hash: finalized_hash,
            },
        );
        assert_request_hashes(
            &effects,
            HashSet::new(),
            HashSet::from([missing_parent_hash]),
        );
        let state = drain_block_log_effects(state, &effects);

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::BlockHeaderNotFound { request_id },
        );

        assert!(effects.is_empty());
        assert_eq!(next_state.canonical_tip, head_hash);
        assert_single_block_with_parent(&next_state, head_hash, missing_parent_hash);
        assert_single_block_with_parent(&next_state, missing_parent_hash, finalized_hash);
        assert!(next_state.pending_requests.is_empty_for_test());
        assert_state_invariants(&next_state);
    }

    // Sends a second failure for a request already replaced by retry.
    // This avoids duplicate failure handling for obsolete ids.
    #[test]
    fn late_block_header_not_found_for_failed_request_is_ignored() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: head_hash,
                parent_hash: missing_parent_hash,
            },
        );
        let failed_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);
        let state = drain_block_log_effects(state, &effects);
        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            request_failed_for_header(failed_request_id),
        );
        let retry_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::BlockHeaderNotFound {
                request_id: failed_request_id,
            },
        );

        assert!(effects.is_empty());
        assert_eq!(next_state.canonical_tip, head_hash);
        assert_single_block_with_parent(&next_state, head_hash, missing_parent_hash);
        assert_eq!(next_state.pending_requests.len_for_test(), 1);
        assert!(next_state.pending_requests.contains(&retry_request_id));
        assert_state_invariants(&next_state);
        assert_missing_parents_for_known_blocks_are_pending(&next_state);
    }

    // Reports not-found for an expired request after retry created a newer id.
    // This prevents stale expiration-era responses from affecting active work.
    #[test]
    fn late_block_header_not_found_for_expired_request_is_ignored() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: head_hash,
                parent_hash: missing_parent_hash,
            },
        );
        let expired_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);
        let state = drain_block_log_effects(state, &effects);
        let (state, effects) = advance_ticks(state, REQUEST_TTL);
        let retry_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::BlockHeaderNotFound {
                request_id: expired_request_id,
            },
        );

        assert!(effects.is_empty());
        assert_eq!(next_state.canonical_tip, head_hash);
        assert_single_block_with_parent(&next_state, head_hash, missing_parent_hash);
        assert_eq!(next_state.pending_requests.len_for_test(), 1);
        assert!(next_state.pending_requests.contains(&retry_request_id));
        assert_state_invariants(&next_state);
        assert_missing_parents_for_known_blocks_are_pending(&next_state);
    }

    // Reports not-found for the active retry request.
    // This ensures the canonical reset signal still applies after retry replacement.
    #[test]
    fn block_header_not_found_for_current_retry_resets_chain() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: head_hash,
                parent_hash: missing_parent_hash,
            },
        );
        let failed_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);
        let state = drain_block_log_effects(state, &effects);
        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            request_failed_for_header(failed_request_id),
        );
        let retry_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::BlockHeaderNotFound {
                request_id: retry_request_id,
            },
        );

        assert!(effects.is_empty());
        assert_empty_initial_state_at(&next_state, finalized_hash);
        assert_state_invariants(&next_state);
    }

    // Reports not-found and then ticks past the old dispatch time.
    // This ensures consumed requests cannot retry from stale TTL bookkeeping.
    #[test]
    fn block_header_not_found_request_is_not_retried_at_original_expiration() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: head_hash,
                parent_hash: missing_parent_hash,
            },
        );
        let request_id = assert_single_block_header_request_effect(&effects, missing_parent_hash);
        let state = drain_block_log_effects(state, &effects);
        let (state, effects) = advance_ticks(state, REQUEST_TTL - 1);
        assert!(effects.is_empty());
        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::BlockHeaderNotFound { request_id },
        );
        assert!(effects.is_empty());
        assert_empty_initial_state_at(&state, finalized_hash);

        let (next_state, effects) = transition(ChainKey::Ethereum, state, Event::Tick);

        assert!(effects.is_empty());
        assert!(next_state.tick == tick(REQUEST_TTL));
        assert_empty_initial_state_at(&next_state, finalized_hash);
        assert_state_invariants(&next_state);
    }

    // Returns a different header than the one requested.
    // This keeps the original ancestry gap pending because the needed header was not supplied.
    #[test]
    fn mismatched_header_response_does_not_lose_missing_canonical_parent_request() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let unrelated_hash = BlockHash::with_last_byte(4);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: head_hash,
                parent_hash: missing_parent_hash,
            },
        );
        let request_id = assert_single_block_header_request_effect(&effects, missing_parent_hash);

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::BlockHeaderReceived {
                logs_bloom: bloom_matching_any(),
                request_id,
                hash: unrelated_hash,
                parent_hash: finalized_hash,
            },
        );

        assert_eq!(next_state.canonical_tip, head_hash);
        assert_single_block_with_parent(&next_state, head_hash, missing_parent_hash);
        assert!(
            next_state
                .pending_requests
                .pending_header_hashes_for_test()
                .contains(&missing_parent_hash),
            "missing canonical parent must remain requested"
        );
        let retry_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);
        assert!(next_state.pending_requests.contains(&retry_request_id));
        assert_state_invariants(&next_state);
    }

    // Returns an unexpected header that makes chain state unsafe.
    // This combines reset recovery with preserving the still-needed header request.
    #[test]
    fn conflicting_mismatched_header_response_resets_chain_and_retries_request() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let unrelated_hash = BlockHash::with_last_byte(4);
        let conflicting_parent_hash = BlockHash::with_last_byte(5);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: head_hash,
                parent_hash: missing_parent_hash,
            },
        );
        let request_id = assert_single_block_header_request_effect(&effects, missing_parent_hash);

        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::BlockHeaderReceived {
                logs_bloom: bloom_matching_any(),
                request_id: RequestId::from_raw_for_test(99),
                hash: unrelated_hash,
                parent_hash: finalized_hash,
            },
        );
        assert!(effects.is_empty());

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::BlockHeaderReceived {
                logs_bloom: bloom_matching_any(),
                request_id,
                hash: unrelated_hash,
                parent_hash: conflicting_parent_hash,
            },
        );

        assert_chain_reset_at(&next_state, finalized_hash);
        let retry_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);
        assert!(next_state.pending_requests.contains(&retry_request_id));
        assert!(!next_state.pending_requests.contains(&request_id));
        assert_state_invariants(&next_state);
    }

    // Processes late effects after a reset and fresh request issuance.
    // This prevents stale responses from colliding with new request ids.
    #[test]
    fn request_ids_are_not_reused_after_reset_while_old_effects_may_be_in_flight() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let first_missing_parent_hash = BlockHash::with_last_byte(2);
        let first_head_hash = BlockHash::with_last_byte(3);
        let conflicting_parent_hash = BlockHash::with_last_byte(4);
        let second_missing_parent_hash = BlockHash::with_last_byte(5);
        let second_head_hash = BlockHash::with_last_byte(6);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: first_head_hash,
                parent_hash: first_missing_parent_hash,
            },
        );
        let old_request_id =
            assert_single_block_header_request_effect(&effects, first_missing_parent_hash);

        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: first_head_hash,
                parent_hash: conflicting_parent_hash,
            },
        );
        assert_chain_reset_at(&state, finalized_hash);
        assert!(state.pending_requests.contains(&old_request_id));
        assert!(effects.is_empty());

        let (_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: second_head_hash,
                parent_hash: second_missing_parent_hash,
            },
        );
        let new_request_id =
            assert_single_block_header_request_effect(&effects, second_missing_parent_hash);

        assert!(
            new_request_id != old_request_id,
            "reset must not reuse IDs that may still identify in-flight effects"
        );
    }

    // Advances to just before request TTL.
    // This locks the lower bound so requests are not retried early.
    #[test]
    fn tick_before_ttl_does_not_retry_request() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: head_hash,
                parent_hash: missing_parent_hash,
            },
        );
        let request_id = assert_single_block_header_request_effect(&effects, missing_parent_hash);
        let state = drain_block_log_effects(state, &effects);

        let (next_state, effects) = advance_ticks(state, REQUEST_TTL - 1);

        assert!(effects.is_empty());
        assert!(next_state.tick == tick(REQUEST_TTL - 1));
        assert_eq!(next_state.pending_requests.len_for_test(), 1);
        assert!(next_state.pending_requests.contains(&request_id));
        assert_active_requests_have_exactly_one_dispatch_tick(&next_state);
    }

    // Advances exactly to the request TTL boundary.
    // This locks the first retry point and ensures it emits one replacement effect.
    #[test]
    fn tick_at_ttl_retries_request_exactly_once() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: head_hash,
                parent_hash: missing_parent_hash,
            },
        );
        let expired_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);
        let state = drain_block_log_effects(state, &effects);

        let (next_state, effects) = advance_ticks(state, REQUEST_TTL);

        let retry_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);
        assert!(retry_request_id != expired_request_id);
        assert!(next_state.tick == tick(REQUEST_TTL));
        assert_eq!(next_state.pending_requests.len_for_test(), 1);
        assert!(!next_state.pending_requests.contains(&expired_request_id));
        assert!(next_state.pending_requests.contains(&retry_request_id));
        assert_active_requests_have_exactly_one_dispatch_tick(&next_state);
        assert_no_active_request_is_expired(&next_state);
    }

    // Starts near tick overflow and advances past TTL.
    // This ensures wrapping tick arithmetic still expires requests correctly.
    #[test]
    fn expiration_works_across_tick_wraparound() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let dispatch_tick = tick(u64::MAX - 4);
        let mut state = empty_state_at(finalized_hash);
        state.tick = dispatch_tick;

        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: head_hash,
                parent_hash: missing_parent_hash,
            },
        );
        let expired_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);
        let state = drain_block_log_effects(state, &effects);

        let (state, effects) = advance_ticks(state, REQUEST_TTL - 1);
        assert!(effects.is_empty());
        assert!(state.tick == tick(dispatch_tick.raw_for_test().wrapping_add(REQUEST_TTL - 1)));
        assert!(state.pending_requests.contains(&expired_request_id));

        let (next_state, effects) = transition(ChainKey::Ethereum, state, Event::Tick);

        let retry_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);
        assert!(retry_request_id != expired_request_id);
        assert!(next_state.tick == tick(dispatch_tick.raw_for_test().wrapping_add(REQUEST_TTL)));
        assert_eq!(next_state.pending_requests.len_for_test(), 1);
        assert_no_active_request_is_expired(&next_state);
    }

    // Retries an expired request and then ticks again.
    // This ensures the replacement dispatch tick prevents immediate re-expiration.
    #[test]
    fn retry_does_not_expire_again_until_another_ttl() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: head_hash,
                parent_hash: missing_parent_hash,
            },
        );
        let first_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);
        let state = drain_block_log_effects(state, &effects);
        let (state, effects) = advance_ticks(state, REQUEST_TTL);
        let first_retry_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);
        assert!(first_retry_id != first_request_id);

        let (state, effects) = advance_ticks(state, REQUEST_TTL - 1);
        assert!(effects.is_empty());
        assert!(state.pending_requests.contains(&first_retry_id));

        let (next_state, effects) = transition(ChainKey::Ethereum, state, Event::Tick);

        let second_retry_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);
        assert!(second_retry_id != first_retry_id);
        assert_eq!(next_state.pending_requests.len_for_test(), 1);
        assert!(next_state.pending_requests.contains(&second_retry_id));
        assert_no_active_request_is_expired(&next_state);
    }

    // Creates requests dispatched at different ticks.
    // This ensures retry scheduling is based on each request's own age.
    #[test]
    fn requests_dispatched_at_different_ticks_expire_separately() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let first_hash = BlockHash::with_last_byte(2);
        let second_hash = BlockHash::with_last_byte(3);
        let mut state = empty_state_at(finalized_hash);
        let (pending_requests, first_request_id) = state.pending_requests.with_new_request(
            GetBlockHeader {
                block_hash: first_hash,
            },
            state.tick,
        );
        state.pending_requests = pending_requests;

        let (mut state, effects) = advance_ticks(state, 3);
        assert!(effects.is_empty());
        let (pending_requests, second_request_id) = state.pending_requests.with_new_request(
            GetBlockHeader {
                block_hash: second_hash,
            },
            state.tick,
        );
        state.pending_requests = pending_requests;

        let (state, effects) = advance_ticks(state, REQUEST_TTL - 3);

        assert_eq!(effects.len(), 1);
        assert_eq!(
            header_hashes_from_effects(&effects),
            HashSet::from([first_hash])
        );
        assert!(!state.pending_requests.contains(&first_request_id));
        assert!(state.pending_requests.contains(&second_request_id));

        let (next_state, effects) = advance_ticks(state, 3);

        assert_eq!(effects.len(), 1);
        assert_eq!(
            header_hashes_from_effects(&effects),
            HashSet::from([second_hash])
        );
        assert!(!next_state.pending_requests.contains(&second_request_id));
        assert_eq!(next_state.pending_requests.len_for_test(), 2);
        assert_eq!(
            pending_header_hashes(&next_state),
            HashSet::from([first_hash, second_hash])
        );
        assert_no_active_request_is_expired(&next_state);
    }

    // Expires multiple requests dispatched on the same tick.
    // This ensures each expired request is replaced exactly once.
    #[test]
    fn multiple_requests_dispatched_together_all_expire_once() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let requested_hashes = HashSet::from([
            BlockHash::with_last_byte(2),
            BlockHash::with_last_byte(3),
            BlockHash::with_last_byte(4),
        ]);
        let mut state = empty_state_at(finalized_hash);
        let mut original_request_ids = Vec::new();

        for block_hash in &requested_hashes {
            let (pending_requests, request_id) = state.pending_requests.with_new_request(
                GetBlockHeader {
                    block_hash: *block_hash,
                },
                state.tick,
            );
            state.pending_requests = pending_requests;
            original_request_ids.push(request_id);
        }

        let (next_state, effects) = advance_ticks(state, REQUEST_TTL);

        assert_eq!(effects.len(), requested_hashes.len());
        assert_eq!(header_hashes_from_effects(&effects), requested_hashes);
        assert_eq!(next_state.pending_requests.len_for_test(), 3);
        assert_eq!(pending_header_hashes(&next_state), requested_hashes);
        for original_request_id in original_request_ids {
            assert!(!next_state.pending_requests.contains(&original_request_id));
        }
        assert_active_requests_have_exactly_one_dispatch_tick(&next_state);
        assert_no_active_request_is_expired(&next_state);
    }

    // Ticks with no active requests.
    // This ensures time advancement does not emit spurious effects.
    #[test]
    fn empty_tick_only_advances_tick() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let known_hash = BlockHash::with_last_byte(2);
        let mut state = empty_state_at(finalized_hash);
        state.canonical_tip = known_hash;
        state
            .blocks
            .0
            .insert(known_hash, block_with_parent(finalized_hash));

        let (next_state, effects) = transition(ChainKey::Ethereum, state, Event::Tick);

        assert!(next_state.tick == tick(1));
        assert!(effects.is_empty());
        assert_eq!(next_state.canonical_tip, known_hash);
        assert_eq!(next_state.finalized_state.block_hash, finalized_hash);
        assert_eq!(next_state.blocks.0.len(), 1);
        assert_single_unknown_block(&next_state, known_hash, finalized_hash);
        assert!(next_state.pending_requests.is_empty_for_test());
        assert_state_invariants(&next_state);
    }

    // Fails an active request through the generic failure event.
    // This ensures transport errors preserve payload scope while replacing request identity.
    #[test]
    fn request_failed_retries_known_request_with_fresh_id() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: head_hash,
                parent_hash: missing_parent_hash,
            },
        );
        let failed_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);
        let state = drain_block_log_effects(state, &effects);

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            request_failed_for_header(failed_request_id),
        );

        let retry_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);
        assert!(retry_request_id != failed_request_id);
        assert_eq!(next_state.pending_requests.len_for_test(), 1);
        assert!(!next_state.pending_requests.contains(&failed_request_id));
        assert!(next_state.pending_requests.contains(&retry_request_id));
        assert_eq!(next_state.canonical_tip, head_hash);
        assert_single_block_with_parent(&next_state, head_hash, missing_parent_hash);
        assert_state_invariants(&next_state);
        assert_missing_parents_for_known_blocks_are_pending(&next_state);
    }

    // Sends a failure for an unknown request id.
    // This protects state and pending work from stale or unsolicited notifications.
    #[test]
    fn request_failed_for_unknown_id_is_noop() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: head_hash,
                parent_hash: missing_parent_hash,
            },
        );
        let pending_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);
        let state = drain_block_log_effects(state, &effects);
        let last_request_id = state.pending_requests.last_request_id_for_test();

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            request_failed_for_header(RequestId::from_raw_for_test(99)),
        );

        assert!(effects.is_empty());
        assert_eq!(next_state.pending_requests.len_for_test(), 1);
        assert!(next_state.pending_requests.contains(&pending_request_id));
        assert!(next_state.pending_requests.last_request_id_for_test() == last_request_id);
        assert_eq!(next_state.canonical_tip, head_hash);
        assert_single_block_with_parent(&next_state, head_hash, missing_parent_hash);
        assert_state_invariants(&next_state);
        assert_missing_parents_for_known_blocks_are_pending(&next_state);
    }

    // Sends a failure for an old id after retry already created a replacement.
    // This prevents duplicate active requests for the same original work.
    #[test]
    fn duplicate_request_failed_for_old_id_is_ignored() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: head_hash,
                parent_hash: missing_parent_hash,
            },
        );
        let failed_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);
        let state = drain_block_log_effects(state, &effects);
        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            request_failed_for_header(failed_request_id),
        );
        let retry_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            request_failed_for_header(failed_request_id),
        );

        assert!(effects.is_empty());
        assert_eq!(next_state.pending_requests.len_for_test(), 1);
        assert!(next_state.pending_requests.contains(&retry_request_id));
        assert!(
            next_state.pending_requests.last_request_id_for_test()
                == retry_request_id.raw_for_test()
        );
        assert_state_invariants(&next_state);
        assert_missing_parents_for_known_blocks_are_pending(&next_state);
    }

    // Repeatedly fails the active retry request.
    // This keeps retry replacement one-for-one under persistent transport errors.
    #[test]
    fn failed_retry_can_be_retried_again_without_growing_pending_requests() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: head_hash,
                parent_hash: missing_parent_hash,
            },
        );
        let mut failed_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);
        let mut state = drain_block_log_effects(state, &effects);

        for _ in 0..4 {
            let (next_state, effects) = transition(
                ChainKey::Ethereum,
                state,
                request_failed_for_header(failed_request_id),
            );
            let retry_request_id =
                assert_single_block_header_request_effect(&effects, missing_parent_hash);

            assert!(retry_request_id != failed_request_id);
            assert_eq!(next_state.pending_requests.len_for_test(), 1);
            assert!(!next_state.pending_requests.contains(&failed_request_id));
            assert!(next_state.pending_requests.contains(&retry_request_id));
            assert_state_invariants(&next_state);
            assert_missing_parents_for_known_blocks_are_pending(&next_state);

            state = next_state;
            failed_request_id = retry_request_id;
        }
    }

    // Resets chain state while a request is in flight, then fails that old id.
    // This preserves retry handling for request ids still tracked across reset.
    #[test]
    fn request_failed_after_chain_reset_retries_preserved_request() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let conflicting_parent_hash = BlockHash::with_last_byte(4);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: head_hash,
                parent_hash: missing_parent_hash,
            },
        );
        let failed_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);
        let state = drain_block_log_effects(state, &effects);
        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: head_hash,
                parent_hash: conflicting_parent_hash,
            },
        );
        assert!(effects.is_empty());
        assert_chain_reset_at(&state, finalized_hash);

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            request_failed_for_header(failed_request_id),
        );

        assert_chain_reset_at(&next_state, finalized_hash);
        let retry_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);
        assert!(retry_request_id != failed_request_id);
        assert_eq!(next_state.pending_requests.len_for_test(), 1);
        assert!(next_state.pending_requests.contains(&retry_request_id));
        assert_state_invariants(&next_state);
    }

    // Completes a request successfully and then receives a late failure for it.
    // This avoids retrying work already removed from pending state.
    #[test]
    fn successful_response_followed_by_failure_for_old_id_is_ignored() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: head_hash,
                parent_hash: missing_parent_hash,
            },
        );
        let request_id = assert_single_block_header_request_effect(&effects, missing_parent_hash);
        let state = drain_block_log_effects(state, &effects);
        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::BlockHeaderReceived {
                logs_bloom: bloom_matching_any(),
                request_id,
                hash: missing_parent_hash,
                parent_hash: finalized_hash,
            },
        );
        assert_request_hashes(
            &effects,
            HashSet::new(),
            HashSet::from([missing_parent_hash]),
        );
        let state = drain_block_log_effects(state, &effects);

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            request_failed_for_header(request_id),
        );

        assert!(effects.is_empty());
        assert!(next_state.pending_requests.is_empty_for_test());
        assert_eq!(next_state.canonical_tip, head_hash);
        assert_single_block_with_parent(&next_state, head_hash, missing_parent_hash);
        assert_single_block_with_parent(&next_state, missing_parent_hash, finalized_hash);
        assert_state_invariants(&next_state);
    }

    // Accepts a late successful response after a retry was already issued.
    // This uses useful data while keeping the newer request pending for safety.
    #[test]
    fn late_success_for_failed_request_is_accepted_while_retry_remains_pending() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: head_hash,
                parent_hash: missing_parent_hash,
            },
        );
        let failed_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);
        let state = drain_block_log_effects(state, &effects);
        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            request_failed_for_header(failed_request_id),
        );
        let retry_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::BlockHeaderReceived {
                logs_bloom: bloom_matching_any(),
                request_id: failed_request_id,
                hash: missing_parent_hash,
                parent_hash: finalized_hash,
            },
        );

        assert_request_hashes(
            &effects,
            HashSet::new(),
            HashSet::from([missing_parent_hash]),
        );
        let next_state = drain_block_log_effects(next_state, &effects);
        assert_eq!(next_state.pending_requests.len_for_test(), 1);
        assert!(next_state.pending_requests.contains(&retry_request_id));
        assert_eq!(next_state.canonical_tip, head_hash);
        assert_single_block_with_parent(&next_state, head_hash, missing_parent_hash);
        assert_single_block_with_parent(&next_state, missing_parent_hash, finalized_hash);
        assert_state_invariants(&next_state);
    }

    // Fails the original request after a retry response already succeeded.
    // This prevents obsolete failures from mutating state or issuing more work.
    #[test]
    fn successful_retry_followed_by_duplicate_failure_for_original_id_is_ignored() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: head_hash,
                parent_hash: missing_parent_hash,
            },
        );
        let failed_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);
        let state = drain_block_log_effects(state, &effects);
        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            request_failed_for_header(failed_request_id),
        );
        let retry_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);
        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::BlockHeaderReceived {
                logs_bloom: bloom_matching_any(),
                request_id: retry_request_id,
                hash: missing_parent_hash,
                parent_hash: finalized_hash,
            },
        );
        assert_request_hashes(
            &effects,
            HashSet::new(),
            HashSet::from([missing_parent_hash]),
        );
        let state = drain_block_log_effects(state, &effects);

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            request_failed_for_header(failed_request_id),
        );

        assert!(effects.is_empty());
        assert!(next_state.pending_requests.is_empty_for_test());
        assert_eq!(next_state.canonical_tip, head_hash);
        assert_single_block_with_parent(&next_state, head_hash, missing_parent_hash);
        assert_single_block_with_parent(&next_state, missing_parent_hash, finalized_hash);
        assert_state_invariants(&next_state);
    }

    // Completes a request successfully.
    // This ensures TTL bookkeeping is removed with the pending request.
    #[test]
    fn completed_request_is_not_retried_at_original_expiration() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: head_hash,
                parent_hash: missing_parent_hash,
            },
        );
        let request_id = assert_single_block_header_request_effect(&effects, missing_parent_hash);
        let state = drain_block_log_effects(state, &effects);
        let (state, effects) = advance_ticks(state, REQUEST_TTL - 1);
        assert!(effects.is_empty());
        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::BlockHeaderReceived {
                logs_bloom: bloom_matching_any(),
                request_id,
                hash: missing_parent_hash,
                parent_hash: finalized_hash,
            },
        );
        assert_request_hashes(
            &effects,
            HashSet::new(),
            HashSet::from([missing_parent_hash]),
        );
        let state = drain_block_log_effects(state, &effects);

        let (next_state, effects) = transition(ChainKey::Ethereum, state, Event::Tick);

        assert!(effects.is_empty());
        assert!(next_state.tick == tick(REQUEST_TTL));
        assert!(next_state.pending_requests.is_empty_for_test());
        assert_single_block_with_parent(&next_state, head_hash, missing_parent_hash);
        assert_single_block_with_parent(&next_state, missing_parent_hash, finalized_hash);
        assert_state_invariants(&next_state);
    }

    // Fails an active request and creates a retry.
    // This ensures the old request's dispatch tick is removed during replacement.
    #[test]
    fn failed_request_is_not_retried_at_original_expiration() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: head_hash,
                parent_hash: missing_parent_hash,
            },
        );
        let failed_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);
        let state = drain_block_log_effects(state, &effects);
        let (state, effects) = advance_ticks(state, REQUEST_TTL - 1);
        assert!(effects.is_empty());
        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            request_failed_for_header(failed_request_id),
        );
        let retry_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);

        let (state, effects) = transition(ChainKey::Ethereum, state, Event::Tick);

        assert!(effects.is_empty());
        assert!(state.tick == tick(REQUEST_TTL));
        assert!(state.pending_requests.contains(&retry_request_id));
        assert_no_active_request_is_expired(&state);

        let (next_state, effects) = advance_ticks(state, REQUEST_TTL - 1);

        let second_retry_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);
        assert!(second_retry_id != retry_request_id);
        assert_eq!(next_state.pending_requests.len_for_test(), 1);
        assert_no_active_request_is_expired(&next_state);
    }

    // Accepts a late header response for an old id after retry replacement.
    // This keeps useful chain data without discarding the active retry request.
    #[test]
    fn late_success_after_expiration_is_accepted_while_retry_remains_pending() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: head_hash,
                parent_hash: missing_parent_hash,
            },
        );
        let expired_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);
        let state = drain_block_log_effects(state, &effects);
        let (state, effects) = advance_ticks(state, REQUEST_TTL);
        let retry_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::BlockHeaderReceived {
                logs_bloom: bloom_matching_any(),
                request_id: expired_request_id,
                hash: missing_parent_hash,
                parent_hash: finalized_hash,
            },
        );

        assert_request_hashes(
            &effects,
            HashSet::new(),
            HashSet::from([missing_parent_hash]),
        );
        let next_state = drain_block_log_effects(next_state, &effects);
        assert_eq!(next_state.pending_requests.len_for_test(), 1);
        assert!(next_state.pending_requests.contains(&retry_request_id));
        assert_single_block_with_parent(&next_state, head_hash, missing_parent_hash);
        assert_single_block_with_parent(&next_state, missing_parent_hash, finalized_hash);
        assert_state_invariants(&next_state);
    }

    // Resets chain state while requests remain pending.
    // This keeps in-flight request TTL accounting stable across resets.
    #[test]
    fn chain_reset_preserves_request_expiration_age() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let conflicting_parent_hash = BlockHash::with_last_byte(4);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: head_hash,
                parent_hash: missing_parent_hash,
            },
        );
        let expired_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);
        let state = drain_block_log_effects(state, &effects);
        let (state, effects) = advance_ticks(state, REQUEST_TTL - 1);
        assert!(effects.is_empty());
        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                logs_bloom: bloom_matching_any(),
                hash: head_hash,
                parent_hash: conflicting_parent_hash,
            },
        );
        assert!(effects.is_empty());
        assert_chain_reset_at(&state, finalized_hash);
        assert!(state.pending_requests.contains(&expired_request_id));

        let (next_state, effects) = transition(ChainKey::Ethereum, state, Event::Tick);

        let retry_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);
        assert!(retry_request_id != expired_request_id);
        assert_chain_reset_at(&next_state, finalized_hash);
        assert_eq!(next_state.pending_requests.len_for_test(), 1);
        assert_no_active_request_is_expired(&next_state);
        assert_state_invariants(&next_state);
    }

    /// Applies emitted pool-metadata effects with deterministic validation outcomes.
    /// Properties use this to close the validation loop without depending on RPC decoding.
    fn apply_pool_metadata_effects_for_property(mut state: State, effects: Vec<Effect>) -> State {
        for effect in effects {
            if let Effect::Request(AnyIssuedRequest::PoolMetadata(IssuedRequest {
                request_id,
                request_payload,
            })) = effect
            {
                let metadata = request_payload
                    .candidates
                    .iter()
                    .copied()
                    .map(|candidate| {
                        let result = if candidate.uniswap_v3_address().expect("v3 pool").as_slice()[19] % 2 == 0 {
                            Ok(pool_metadata(1, 2, UniswapV3Fee::Fee3000))
                        } else {
                            Err(PoolMetadataFailure::FactoryReturnedZero)
                        };

                        (candidate, result)
                    })
                    .collect::<HashMap<_, _>>();
                let (next_state, effects) = transition(
                    ChainKey::Ethereum,
                    state,
                    Event::PoolMetadataReceived {
                        request_id,
                        metadata,
                    },
                );

                state = apply_token_metadata_effects_for_property(next_state, effects);
            }
        }

        state
    }

    /// Applies emitted token-metadata effects with deterministic successful decimals.
    /// Properties use this to make verified pools schedulable for pool-data without external calls.
    fn apply_token_metadata_effects_for_property(mut state: State, effects: Vec<Effect>) -> State {
        for effect in effects {
            if let Effect::Request(AnyIssuedRequest::TokenMetadata(IssuedRequest {
                request_id,
                request_payload,
            })) = effect
            {
                let metadata = request_payload
                    .tokens
                    .iter()
                    .copied()
                    .map(|token| (token, Ok(token_metadata(18))))
                    .collect::<HashMap<_, _>>();
                let (next_state, effects) = transition(
                    ChainKey::Ethereum,
                    state,
                    Event::TokenMetadataReceived {
                        request_id,
                        metadata,
                    },
                );

                state = next_state;
                // Token metadata is the last priority step; any follow-up is background backfill only.
                assert_no_priority_effects(&effects);
            }
        }

        state
    }

    // Property tests below document cross-event invariants where example tests would miss
    // shrinking-friendly edge cases.
    proptest! {
        // Generates dispatch ticks and elapsed durations across wrapping boundaries.
        // This keeps Tick expiration tied to elapsed time instead of raw numeric ordering.
        #[test]
        fn tick_expiration_matches_wrapping_elapsed_time(
            dispatch_tick in any::<u64>(),
            elapsed in 0u64..(REQUEST_TTL * 3),
        ) {
            let dispatch_tick = tick(dispatch_tick);
            let current_tick = tick(dispatch_tick.raw_for_test().wrapping_add(elapsed));

            prop_assert_eq!(
                current_tick.is_expired_since(dispatch_tick),
                elapsed >= REQUEST_TTL
            );
        }

        // Builds pending requests with arbitrary ages before a tick.
        // This ensures each expired request is replaced once and unexpired requests stay pending.
        #[test]
        fn tick_replaces_each_expired_request_once(
            tick_before in any::<u64>(),
            ages_after_tick in prop::collection::vec(1u64..(REQUEST_TTL * 3), 0..32),
        ) {
            let finalized_hash = hash_for_node(0);
            let mut state = empty_state_at(finalized_hash);
            state.tick = tick(tick_before);
            let tick_after = state.tick.next();
            let mut original_requests = Vec::new();
            let mut all_hashes = HashSet::new();
            let mut expired_hashes = HashSet::new();

            for (index, age_after_tick) in ages_after_tick.into_iter().enumerate() {
                let block_hash = hash_for_node(index + 1);
                let dispatch_tick = tick(tick_after.raw_for_test().wrapping_sub(age_after_tick));
                let (pending_requests, request_id) = state.pending_requests.with_new_request(
                    GetBlockHeader { block_hash },
                    dispatch_tick,
                );
                let should_expire = age_after_tick >= REQUEST_TTL;

                state.pending_requests = pending_requests;
                original_requests.push((request_id, should_expire));
                all_hashes.insert(block_hash);
                if should_expire {
                    expired_hashes.insert(block_hash);
                }
            }

            let expected_request_count = original_requests.len();
            let (next_state, effects) = transition(ChainKey::Ethereum, state, Event::Tick);

            prop_assert!(next_state.tick == tick_after);
            prop_assert_eq!(next_state.pending_requests.len_for_test(), expected_request_count);
            prop_assert_eq!(effects.len(), expired_hashes.len());
            prop_assert_eq!(header_hashes_from_effects(&effects), expired_hashes);
            prop_assert_eq!(pending_header_hashes(&next_state), all_hashes);

            for (original_request_id, should_expire) in original_requests {
                prop_assert_eq!(
                    next_state.pending_requests.contains(&original_request_id),
                    !should_expire
                );
            }

            assert_effects_are_well_formed(&next_state, &effects);
            assert_active_requests_have_exactly_one_dispatch_tick(&next_state);
            assert_no_active_request_is_expired(&next_state);
        }

        // Generates linear canonical prefixes and reobserves their tip.
        // This catches duplicate log scheduling across variable chain lengths.
        #[test]
        fn duplicate_head_observation_does_not_duplicate_pending_log_requests(
            chain_len in 1usize..16,
        ) {
            let finalized_hash = hash_for_node(0);
            let tip_hash = hash_for_node(chain_len);
            let tip_parent_hash = hash_for_node(chain_len - 1);
            let expected_log_hashes = (1..=chain_len)
                .map(hash_for_node)
                .collect::<HashSet<_>>();
            let mut state = empty_state_at(finalized_hash);

            state.canonical_tip = tip_hash;
            for node_index in 1..=chain_len {
                state
                    .blocks
                    .0
                    .insert(hash_for_node(node_index), block_with_parent(hash_for_node(node_index - 1)));
            }

            let (state, effects) = transition(ChainKey::Ethereum,
                state,
                Event::HeadObserved {
                    logs_bloom: bloom_matching_any(),
                    hash: tip_hash,
                    parent_hash: tip_parent_hash,
                },
            );

            prop_assert_eq!(
                block_log_request_hashes_from_effects(&effects),
                expected_log_hashes.clone()
            );
            prop_assert!(header_request_hashes_from_effects(&effects).is_empty());
            prop_assert_eq!(state.pending_requests.len_for_test(), chain_len);
            prop_assert_eq!(
                state.pending_requests.pending_block_log_hashes(),
                expected_log_hashes.clone()
            );
            assert_effects_are_well_formed(&state, &effects);
            assert_state_invariants(&state);

            let (next_state, effects) = transition(ChainKey::Ethereum,
                state,
                Event::HeadObserved {
                    logs_bloom: bloom_matching_any(),
                    hash: tip_hash,
                    parent_hash: tip_parent_hash,
                },
            );

            prop_assert!(effects.is_empty());
            prop_assert_eq!(next_state.pending_requests.len_for_test(), chain_len);
            prop_assert_eq!(
                next_state.pending_requests.pending_block_log_hashes(),
                expected_log_hashes
            );
            assert_state_invariants(&next_state);
        }

        // Replays generated head observations while draining header and log effects.
        // This ensures no present canonical block is left with unknown logs after emitted work is handled.
        #[test]
        fn drained_present_canonical_path_never_leaves_logs_unknown(
            chain in generated_chain_strategy(),
        ) {
            let finalized_hash = hash_for_node(0);
            let mut state = empty_state_at(finalized_hash);

            for head_index in &chain.observed_heads {
                let hash = hash_for_node(*head_index);
                let parent_hash = hash_for_node(parent_index(&chain, *head_index));

                state = apply_event_and_drain_block_headers(
                    state,
                    &chain,
                    Event::HeadObserved { hash, parent_hash , logs_bloom: bloom_matching_any() },
                );

                assert_present_canonical_logs_are_resolved(&state);
                assert_state_invariants(&state);
            }
        }

        // Generates per-block candidate sets and feeds them through log and metadata handling.
        // This keeps resolved canonical candidates either known or pending validation.
        #[test]
        fn canonical_resolved_pool_candidates_are_known_or_pending(
            block_candidate_bytes in proptest::collection::vec(
                proptest::collection::vec(any::<u8>(), 0..6),
                0..32,
            ),
        ) {
            let finalized_hash = hash_for_node(0);
            let mut state = empty_state_at(finalized_hash);
            let mut parent_hash = finalized_hash;

            for (block_index, candidate_bytes) in block_candidate_bytes.iter().enumerate() {
                let block_hash = hash_for_node(block_index + 1);
                let (next_state, effects) = transition(ChainKey::Ethereum,
                    state,
                    Event::HeadObserved {
                        logs_bloom: bloom_matching_any(),
                        hash: block_hash,
                        parent_hash,
                    },
                );

                state = next_state;
                assert_canonical_resolved_candidates_are_known_or_pending(&state);

                for effect in effects {
                    if let Effect::Request(AnyIssuedRequest::BlockLogs(IssuedRequest {
                        request_id,
                        ..
                    })) = effect
                    {
                        let candidates = candidate_bytes
                            .iter()
                            .copied()
                            .map(pool_candidate_address)
                            .collect::<HashSet<_>>();
                        let (next_state, effects) = transition(ChainKey::Ethereum,
                            state,
                            Event::BlockLogsReceived {
                                request_id,
                                logs: pool_logs(&candidates),
                            },
                        );

                        state = next_state;
                        assert_effects_are_well_formed(&state, &effects);
                        assert_canonical_resolved_candidates_are_known_or_pending(&state);

                        state = apply_pool_metadata_effects_for_property(state, effects);
                        assert_canonical_resolved_candidates_are_known_or_pending(&state);
                    }
                }

                parent_hash = block_hash;
            }
        }

        // Generates verified-pool candidates with deterministic metadata results.
        // This ensures token scheduling keeps verified canonical pool tokens known or pending.
        #[test]
        fn canonical_verified_pool_tokens_are_known_or_pending(
            block_candidate_bytes in proptest::collection::vec(
                proptest::collection::vec(any::<u8>(), 0..6),
                0..32,
            ),
        ) {
            let finalized_hash = hash_for_node(0);
            let mut state = empty_state_at(finalized_hash);
            let mut parent_hash = finalized_hash;
            let mut pool_metadata_results = HashMap::new();

            for (block_index, candidate_bytes) in block_candidate_bytes.iter().enumerate() {
                let block_hash = hash_for_node(block_index + 1);
                let candidates = candidate_bytes
                    .iter()
                    .copied()
                    .map(pool_candidate_address)
                    .collect::<HashSet<_>>();

                for candidate in &candidates {
                    let byte = candidate.uniswap_v3_address().expect("v3 pool").as_slice()[19];
                    pool_metadata_results.insert(
                        *candidate,
                        Ok(pool_metadata(
                            byte,
                            byte.wrapping_add(64),
                            UniswapV3Fee::Fee3000,
                        )),
                    );
                }

                state.blocks.0.insert(
                    block_hash,
                    BlockNode {
                        logs_bloom: None,
                        parent_hash,
                        pool_logs: PoolLogsStatus::Resolved(candidates),
                        pool_snapshots: HashMap::new(),
                        pool_data_failures: HashMap::new(),
                    },
                );
                state.canonical_tip = block_hash;
                parent_hash = block_hash;
            }

            state.pool_registry = state.pool_registry.with_metadata_results(ChainKey::Ethereum, pool_metadata_results);
            let (state, effects) = schedule_unknown_canonical_requests(ChainKey::Ethereum, state, vec![]);
            assert_effects_are_well_formed(&state, &effects);
            let pending_tokens = state.pending_requests.pending_token_metadata_tokens();

            let mut current_hash = state.canonical_tip;
            while current_hash != state.finalized_state.block_hash {
                let Some(block) = state.blocks.get(&current_hash) else {
                    break;
                };

                if let PoolLogsStatus::Resolved(candidates) = &block.pool_logs {
                    for candidate in candidates {
                        if let Some(pool) = state.pool_registry.verified_pool(ChainKey::Ethereum, *candidate) {
                            if let Some(metadata) = state.pool_registry.verified_metadata(pool) {
                                for token in [
                                    TokenAddress(metadata.token0, ChainKey::Ethereum),
                                    TokenAddress(metadata.token1, ChainKey::Ethereum),
                                ] {
                                    prop_assert!(
                                        state.token_registry.is_known(token)
                                            || pending_tokens.contains(&token)
                                    );
                                }
                            }
                        }
                    }
                }

                current_hash = block.parent_hash;
            }
        }

        // Generates registry, block, snapshot, failure, and pending-pool combinations.
        // This ensures pool-data scheduling asks only for verified pools still uncovered at the target block.
        #[test]
        fn scheduled_pool_data_requests_include_only_verified_uncovered_pools(
            verified_bytes in proptest::collection::hash_set(any::<u8>(), 0..16),
            rejected_bytes in proptest::collection::hash_set(any::<u8>(), 0..16),
            logged_bytes in proptest::collection::hash_set(any::<u8>(), 0..16),
            snapshot_bytes in proptest::collection::hash_set(any::<u8>(), 0..16),
            failure_bytes in proptest::collection::hash_set(any::<u8>(), 0..16),
            pending_bytes in proptest::collection::hash_set(any::<u8>(), 0..16),
        ) {
            let finalized_hash = hash_for_node(0);
            let block_hash = hash_for_node(1);
            let mut state = empty_state_at(finalized_hash);
            let mut pool_metadata_results = HashMap::new();
            let mut token_metadata_results = HashMap::new();

            for byte in &verified_bytes {
                let candidate = pool_candidate_address(*byte);
                pool_metadata_results.insert(
                    candidate,
                    Ok(pool_metadata(*byte, byte.wrapping_add(64), UniswapV3Fee::Fee3000)),
                );
                token_metadata_results.insert(token_address(*byte), Ok(token_metadata(18)));
                token_metadata_results.insert(token_address(byte.wrapping_add(64)), Ok(token_metadata(18)));
            }

            for byte in rejected_bytes.iter().filter(|byte| !verified_bytes.contains(byte)) {
                pool_metadata_results.insert(
                    pool_candidate_address(*byte),
                    Err(PoolMetadataFailure::FactoryReturnedZero),
                );
            }

            state.pool_registry = state.pool_registry.with_metadata_results(ChainKey::Ethereum, pool_metadata_results);
            state.token_registry = state.token_registry.with_metadata_results(token_metadata_results);
            state.canonical_tip = block_hash;
            state.blocks.0.insert(
                block_hash,
                BlockNode {
                    logs_bloom: None,
                    parent_hash: finalized_hash,
                    pool_logs: PoolLogsStatus::Resolved(
                        logged_bytes
                            .iter()
                            .copied()
                            .map(pool_candidate_address)
                            .collect(),
                    ),
                    pool_snapshots: snapshot_bytes
                        .iter()
                        .copied()
                        .map(|byte| (pool_address(byte), pool_state(byte)))
                        .collect(),
                    pool_data_failures: failure_bytes
                        .iter()
                        .filter(|byte| !snapshot_bytes.contains(byte))
                        .copied()
                        .map(|byte| {
                            (
                                pool_address(byte),
                                PoolDataFailure::CallFailed(PoolDataCall::Slot0),
                            )
                        })
                        .collect(),
                },
            );

            let pending_pools = pending_bytes
                .iter()
                .copied()
                .map(pool_address)
                .collect::<HashSet<_>>();
            if !pending_pools.is_empty() {
                let (pending_requests, _) = state.pending_requests.with_new_request(
                    GetPoolData {
                        at: block_hash,
                        pools: pending_pools.clone(),
                    },
                    state.tick,
                );
                state.pending_requests = pending_requests;
            }

            let has_unknown_logged_candidate = logged_bytes
                .iter()
                .copied()
                .map(pool_candidate_address)
                .any(|candidate| !state.pool_registry.is_known(ChainKey::Ethereum, candidate));
            let expected_pools = if has_unknown_logged_candidate {
                HashSet::new()
            } else {
                logged_bytes
                    .iter()
                    .filter(|byte| verified_bytes.contains(byte))
                    .filter(|byte| !snapshot_bytes.contains(byte))
                    .filter(|byte| !failure_bytes.contains(byte))
                    .map(|byte| pool_address(*byte))
                    .filter(|pool| !pending_pools.contains(pool))
                    .collect::<HashSet<_>>()
            };

            let (next_state, effects) = schedule_unknown_canonical_requests(ChainKey::Ethereum, state, vec![]);
            assert_effects_are_well_formed(&next_state, &effects);
            let pool_data_payloads = pool_data_request_payloads_from_effects(&effects);

            if expected_pools.is_empty() {
                prop_assert!(pool_data_payloads.is_empty());
            } else {
                prop_assert_eq!(pool_data_payloads, vec![(block_hash, expected_pools.clone())]);
                for pool in expected_pools {
                    prop_assert!(next_state.pool_registry.verified_metadata(pool).is_some());
                }
            }
        }

        // Generates connected paths with known pools and arbitrary snapshot placement.
        // This checks latest-complete overlays against an independent invalidation oracle.
        #[test]
        fn latest_complete_pool_state_update_matches_incremental_validity_oracle(
            blocks in proptest::collection::vec(
                (
                    proptest::collection::hash_set(0u8..16, 0..6),
                    proptest::collection::hash_set(0u8..16, 0..6),
                ),
                0..16,
            ),
        ) {
            let finalized_hash = hash_for_node(0);
            let mut state = empty_state_at(finalized_hash);
            let pool_metadata_results = (0u8..16)
                .map(|byte| {
                    (
                        pool_candidate_address(byte),
                        Ok(pool_metadata(
                            byte,
                            byte.wrapping_add(64),
                            UniswapV3Fee::Fee3000,
                        )),
                    )
                })
                .collect::<HashMap<_, _>>();
            let mut parent_hash = finalized_hash;

            state.pool_registry = state.pool_registry.with_metadata_results(ChainKey::Ethereum, pool_metadata_results);

            for (block_index, (candidate_bytes, snapshot_bytes)) in blocks.iter().enumerate() {
                let block_hash = hash_for_node(block_index + 1);
                state.blocks.0.insert(
                    block_hash,
                    BlockNode {
                        logs_bloom: None,
                        parent_hash,
                        pool_logs: PoolLogsStatus::Resolved(
                            candidate_bytes
                                .iter()
                                .copied()
                                .map(pool_candidate_address)
                                .collect(),
                        ),
                        pool_snapshots: snapshot_bytes
                            .iter()
                            .copied()
                            .map(|byte| (pool_address(byte), pool_state(byte)))
                            .collect(),
                        pool_data_failures: HashMap::new(),
                    },
                );
                parent_hash = block_hash;
            }

            let start_hash = if blocks.is_empty() {
                finalized_hash
            } else {
                hash_for_node(blocks.len())
            };
            let mut affected_pools = HashSet::new();
            let mut invalid_pools = HashSet::new();
            let mut latest_pool_states = HashMap::new();
            let mut expected_update = complete_pool_state_update(finalized_hash, HashMap::new());

            for (block_index, (candidate_bytes, snapshot_bytes)) in blocks.iter().enumerate() {
                let block_hash = hash_for_node(block_index + 1);

                for byte in candidate_bytes {
                    let pool = pool_address(*byte);
                    affected_pools.insert(pool);
                    invalid_pools.insert(pool);
                }

                for byte in snapshot_bytes {
                    let pool = pool_address(*byte);
                    if affected_pools.contains(&pool) {
                        latest_pool_states.insert(pool, pool_state(*byte));
                        invalid_pools.remove(&pool);
                    }
                }

                if invalid_pools.is_empty() {
                    expected_update =
                        complete_pool_state_update(block_hash, latest_pool_states.clone());
                }
            }

            prop_assert_eq!(
                resolved_complete_pool_state_update_from(&state,start_hash),
                Some(expected_update)
            );
        }

        // Generates complete/incomplete prefixes and compacts an observed finalized target on the canonical path.
        // This checks finalized snapshot advancement against the same invalidation rules used by the readiness query.
        #[test]
        fn finalized_compaction_matches_latest_complete_prefix_oracle(
            blocks in proptest::collection::vec(
                (
                    proptest::collection::hash_set(0u8..16, 0..6),
                    proptest::collection::hash_set(0u8..16, 0..6),
                ),
                0..16,
            ),
            target_choice in any::<usize>(),
        ) {
            let finalized_hash = hash_for_node(0);
            let baseline_pool = pool_address(250);
            let baseline_snapshot = pool_state(251);
            let mut state = empty_state_at(finalized_hash);
            let pool_metadata_results = (0u8..16)
                .map(|byte| {
                    (
                        pool_candidate_address(byte),
                        Ok(pool_metadata(
                            byte,
                            byte.wrapping_add(64),
                            UniswapV3Fee::Fee3000,
                        )),
                    )
                })
                .collect::<HashMap<_, _>>();
            let target_index = target_choice % (blocks.len() + 1);
            let target_hash = hash_for_node(target_index);
            let mut parent_hash = finalized_hash;

            state.finalized_state.pool_snapshots =
                HashMap::from([(baseline_pool, baseline_snapshot.clone())]);
            state.pool_registry = state.pool_registry.with_metadata_results(ChainKey::Ethereum, pool_metadata_results);

            for (block_index, (candidate_bytes, snapshot_bytes)) in blocks.iter().enumerate() {
                let block_hash = hash_for_node(block_index + 1);
                state.blocks.0.insert(
                    block_hash,
                    BlockNode {
                        logs_bloom: None,
                        parent_hash,
                        pool_logs: PoolLogsStatus::Resolved(
                            candidate_bytes
                                .iter()
                                .copied()
                                .map(pool_candidate_address)
                                .collect(),
                        ),
                        pool_snapshots: snapshot_bytes
                            .iter()
                            .copied()
                            .map(|byte| (pool_address(byte), pool_state(byte)))
                            .collect(),
                        pool_data_failures: HashMap::new(),
                    },
                );
                parent_hash = block_hash;
            }

            state.canonical_tip = if blocks.is_empty() {
                finalized_hash
            } else {
                hash_for_node(blocks.len())
            };

            let mut affected_pools = HashSet::new();
            let mut invalid_pools = HashSet::new();
            let mut latest_pool_states = HashMap::new();
            let mut expected_finalized_index = 0usize;
            let mut expected_snapshots = HashMap::from([(baseline_pool, baseline_snapshot.clone())]);

            for (block_index, (candidate_bytes, snapshot_bytes)) in
                blocks.iter().take(target_index).enumerate()
            {
                for byte in candidate_bytes {
                    let pool = pool_address(*byte);
                    affected_pools.insert(pool);
                    invalid_pools.insert(pool);
                }

                for byte in snapshot_bytes {
                    let pool = pool_address(*byte);
                    if affected_pools.contains(&pool) {
                        latest_pool_states.insert(pool, pool_state(*byte));
                        invalid_pools.remove(&pool);
                    }
                }

                if invalid_pools.is_empty() {
                    expected_finalized_index = block_index + 1;
                    expected_snapshots =
                        HashMap::from([(baseline_pool, baseline_snapshot.clone())]);
                    expected_snapshots.extend(latest_pool_states.clone());
                }
            }

            let (state, effects) = transition(ChainKey::Ethereum,
                state,
                Event::FinalizedBlockObserved {
                    block_hash: target_hash,
                },
            );

            // Compaction itself schedules nothing; with an idle pending tier the only follow-up is the
            // background backfill of uncovered verified pools.
            assert_no_priority_effects(&effects);
            prop_assert_eq!(
                state.finalized_state.block_hash,
                hash_for_node(expected_finalized_index)
            );
            prop_assert_eq!(&state.finalized_state.pool_snapshots, &expected_snapshots);

            for node_index in 1..=blocks.len() {
                prop_assert_eq!(
                    state.blocks.0.contains_key(&hash_for_node(node_index)),
                    node_index > expected_finalized_index
                );
            }

            assert_state_invariants(&state);
        }

        // Compaction must never drop a block at-or-after the observed block, on any branching graph.
        // This guards retention of newer blocks regardless of where the complete frontier lands.
        #[test]
        fn finalized_block_observed_never_drops_descendants_of_observed_block(
            chain in generated_chain_strategy(),
        ) {
            let finalized_hash = hash_for_node(0);
            let node_count = chain.parents.len();

            // Empty resolved logs make every block "complete" so compaction actually advances and
            // prunes (rather than no-opping), exercising the retention path on a branching graph.
            let build_state = || {
                let mut state = empty_state_at(finalized_hash);
                for node_index in 1..node_count {
                    state.blocks.0.insert(
                        hash_for_node(node_index),
                        BlockNode {
                            logs_bloom: None,
                            parent_hash: hash_for_node(parent_index(&chain, node_index)),
                            pool_logs: PoolLogsStatus::Resolved(HashSet::new()),
                            pool_snapshots: HashMap::new(),
                            pool_data_failures: HashMap::new(),
                        },
                    );
                }
                state.canonical_tip = hash_for_node(node_count - 1);
                state
            };

            // Hold the invariant for every node hash, the finalized anchor, and an absent hash.
            let mut observed_candidates: Vec<BlockHash> =
                (0..node_count).map(hash_for_node).collect();
            observed_candidates.push(hash_for_node(node_count + 7));

            for observed in observed_candidates {
                let state = build_state();
                let descendants: HashSet<BlockHash> = state
                    .blocks
                    .0
                    .keys()
                    .copied()
                    .filter(|hash| block_descends_from(&state.blocks.0, *hash, observed))
                    .collect();

                let result = state.with_finalized_block_observed(ChainKey::Ethereum, observed);

                for descendant in &descendants {
                    prop_assert!(
                        result.blocks.0.contains_key(descendant),
                        "compaction dropped a block at-or-after the observed block"
                    );
                }
            }
        }

        // Generates candidate logs with mixed registry outcomes.
        // This ensures trusted-log projection never exposes unverified candidates as pools.
        #[test]
        fn derived_resolved_trusted_pool_logs_never_include_unverified_pools(
            block_candidate_bytes in proptest::collection::vec(
                proptest::collection::vec(any::<u8>(), 0..6),
                0..32,
            ),
        ) {
            let finalized_hash = hash_for_node(0);
            let mut state = empty_state_at(finalized_hash);
            let mut parent_hash = finalized_hash;

            for (block_index, candidate_bytes) in block_candidate_bytes.iter().enumerate() {
                let block_hash = hash_for_node(block_index + 1);
                let (next_state, effects) = transition(ChainKey::Ethereum,
                    state,
                    Event::HeadObserved {
                        logs_bloom: bloom_matching_any(),
                        hash: block_hash,
                        parent_hash,
                    },
                );
                state = next_state;

                for effect in effects {
                    if let Effect::Request(AnyIssuedRequest::BlockLogs(IssuedRequest {
                        request_id,
                        ..
                    })) = effect
                    {
                        let candidates = candidate_bytes
                            .iter()
                            .copied()
                            .map(pool_candidate_address)
                            .collect::<HashSet<_>>();
                        let (next_state, effects) = transition(ChainKey::Ethereum,
                            state,
                            Event::BlockLogsReceived {
                                request_id,
                                logs: pool_logs(&candidates),
                            },
                        );
                        state = apply_pool_metadata_effects_for_property(next_state, effects);
                    }
                }

                parent_hash = block_hash;
            }

            for block_hash in state.blocks.0.keys() {
                if let Some(TrustedPoolLogs::Resolved(pools)) =
                    state.blocks.trusted_pool_logs(ChainKey::Ethereum, *block_hash, &state.pool_registry)
                {
                    for pool in pools {
                        prop_assert!(
                            state.pool_registry.verified_metadata(pool).is_some(),
                            "resolved trusted pool must be present in verified registry"
                        );
                    }
                }
            }
        }

        // Replays generated valid heads and immediately drains header/log work.
        // This proves the kernel reconstructs the observed ancestry closure with resolved logs.
        #[test]
        fn transition_reconstructs_observed_valid_chain(chain in generated_chain_strategy()) {
            let finalized_hash = hash_for_node(0);
            let mut state = empty_state_at(finalized_hash);

            for head_index in &chain.observed_heads {
                let hash = hash_for_node(*head_index);
                let parent_hash = hash_for_node(parent_index(&chain, *head_index));

                state = apply_event_and_drain_block_headers(
                    state,
                    &chain,
                    Event::HeadObserved { hash, parent_hash , logs_bloom: bloom_matching_any() },
                );
            }

            let expected_blocks = expected_observed_ancestor_closure(&chain);
            prop_assert_eq!(state.blocks.0.len(), expected_blocks.len());

            for (hash, parent_hash) in expected_blocks {
                let block = state
                    .blocks
                    .get(&hash)
                    .ok_or_else(|| TestCaseError::fail("expected block to be present"))?;

                prop_assert_eq!(block.parent_hash, parent_hash);
                prop_assert!(matches!(block.pool_logs, PoolLogsStatus::Resolved(_)));
                prop_assert!(block.pool_snapshots.is_empty());
            }

            let last_observed_head = chain
                .observed_heads
                .last()
                .map(|head_index| hash_for_node(*head_index))
                .unwrap_or(finalized_hash);

            prop_assert_eq!(state.canonical_tip, last_observed_head);
            prop_assert!(state.pending_requests.is_empty_for_test());
            prop_assert!(!state.blocks.0.contains_key(&finalized_hash));
        }

        // Replays generated heads while delaying emitted header/log responses.
        // This proves reconstruction does not depend on immediate RPC completion order.
        #[test]
        fn transition_reconstructs_valid_chain_with_delayed_header_responses(
            chain in generated_chain_strategy(),
        ) {
            let finalized_hash = hash_for_node(0);
            let mut state = empty_state_at(finalized_hash);
            let mut pending_effects = Vec::new();

            for head_index in &chain.observed_heads {
                let hash = hash_for_node(*head_index);
                let parent_hash = hash_for_node(parent_index(&chain, *head_index));
                let (next_state, effects) =
                    transition(
                        ChainKey::Ethereum,
                        state,
                        Event::HeadObserved {
                            hash,
                            parent_hash,
                            logs_bloom: bloom_matching_any(),
                        },
                    );

                state = next_state;
                pending_effects.extend(effects);
                // Concurrent heads may share a missing ancestor before its header resolves; the dedup
                // guard must keep at most one in-flight request per hash at every production-reached step.
                assert_no_duplicate_pending_header_requests(&state);
            }

            while let Some(effect) = pending_effects.pop() {
                match effect {
                    Effect::Request(AnyIssuedRequest::BlockHeader(IssuedRequest {
                        request_id,
                        request_payload: GetBlockHeader { block_hash },
                    })) => {
                        let block_index = node_index_for_hash(&chain, block_hash)
                            .ok_or_else(|| TestCaseError::fail("requested header must be in chain"))?;
                        let parent_hash = hash_for_node(parent_index(&chain, block_index));
                        let (next_state, effects) = transition(ChainKey::Ethereum,
                            state,
                            Event::BlockHeaderReceived {
                                logs_bloom: bloom_matching_any(),
                                request_id,
                                hash: block_hash,
                                parent_hash,
                            },
                        );

                        state = next_state;
                        pending_effects.extend(effects);
                        assert_no_duplicate_pending_header_requests(&state);
                    }
                    Effect::Request(AnyIssuedRequest::BlockLogs(IssuedRequest {
                        request_id, ..
                    })) => {
                        let (next_state, effects) = transition(ChainKey::Ethereum,
                            state,
                            Event::BlockLogsReceived {
                                request_id,
                                logs: Vec::new(),
                            },
                        );

                        state = next_state;
                        pending_effects.extend(effects);
                    }
                    Effect::Request(AnyIssuedRequest::PoolData(_)) => {}
                    Effect::Request(AnyIssuedRequest::PoolMetadata(_)) => {}
                    Effect::Request(AnyIssuedRequest::TokenMetadata(_)) => {}
                }
            }

            let expected_blocks = expected_observed_ancestor_closure(&chain);
            prop_assert_eq!(state.blocks.0.len(), expected_blocks.len());

            for (hash, parent_hash) in expected_blocks {
                let block = state
                    .blocks
                    .get(&hash)
                    .ok_or_else(|| TestCaseError::fail("expected block to be present"))?;

                prop_assert_eq!(block.parent_hash, parent_hash);
            }

            prop_assert!(state.pending_requests.is_empty_for_test());
            assert_state_invariants(&state);
        }

        // Replays generated heads with bounded failure and expiration plans before header success.
        // This proves retry churn does not prevent eventual ancestry reconstruction.
        #[test]
        fn transition_reconstructs_valid_chain_after_arbitrary_finite_retries(
            chain in generated_chain_strategy(),
            retry_plans in generated_retry_plans_strategy(),
        ) {
            let finalized_hash = hash_for_node(0);
            let mut state = empty_state_at(finalized_hash);

            for head_index in &chain.observed_heads {
                let hash = hash_for_node(*head_index);
                let parent_hash = hash_for_node(parent_index(&chain, *head_index));

                state = apply_event_and_drain_block_headers_with_retries(
                    state,
                    &chain,
                    Event::HeadObserved { hash, parent_hash , logs_bloom: bloom_matching_any() },
                    &retry_plans,
                );
            }

            let expected_blocks = expected_observed_ancestor_closure(&chain);
            prop_assert_eq!(state.blocks.0.len(), expected_blocks.len());

            for (hash, parent_hash) in expected_blocks {
                let block = state
                    .blocks
                    .get(&hash)
                    .ok_or_else(|| TestCaseError::fail("expected block to be present"))?;

                prop_assert_eq!(block.parent_hash, parent_hash);
            }

            let last_observed_head = chain
                .observed_heads
                .last()
                .map(|head_index| hash_for_node(*head_index))
                .unwrap_or(finalized_hash);

            prop_assert_eq!(state.canonical_tip, last_observed_head);
            prop_assert!(state.pending_requests.is_empty_for_test());
            prop_assert!(!state.blocks.0.contains_key(&finalized_hash));
            assert_state_invariants(&state);
        }

        // Generates unrelated pending requests around one missing-header request.
        // This ensures actionable not-found resets chain state while preserving unrelated pending work.
        #[test]
        fn actionable_block_header_not_found_resets_and_removes_only_matching_request(
            tick_value in any::<u64>(),
            generated_payloads in prop::collection::vec(
                generated_request_payload_strategy(),
                0..32,
            ),
        ) {
            let finalized_hash = hash_for_node(0);
            let known_hash = hash_for_node(1);
            let missing_hash = hash_for_node(32);
            let mut state = empty_state_at(finalized_hash);
            state.tick = tick(tick_value);
            state.canonical_tip = known_hash;
            state
                .blocks
                .0
                .insert(known_hash, block_with_parent(finalized_hash));
            let mut unrelated_requests = Vec::new();

            for generated_payload in generated_payloads {
                let request_payload = request_payload_from_generated(generated_payload);
                let expected_payload = request_payload.clone();
                let (pending_requests, request_id) =
                    issue_expected_request(state.pending_requests, request_payload, state.tick);

                state.pending_requests = pending_requests;
                unrelated_requests.push((request_id, expected_payload));
            }

            let target_payload = GetBlockHeader {
                block_hash: missing_hash,
            };
            let (pending_requests, target_request_id) = state
                .pending_requests
                .with_new_request(target_payload, state.tick);
            state.pending_requests = pending_requests;
            let last_request_id = state.pending_requests.last_request_id_for_test();

            let (next_state, effects) = transition(ChainKey::Ethereum,
                state,
                Event::BlockHeaderNotFound {
                    request_id: target_request_id,
                },
            );

            prop_assert!(effects.is_empty());
            prop_assert!(next_state.blocks.0.is_empty());
            prop_assert_eq!(next_state.canonical_tip, finalized_hash);
            prop_assert_eq!(next_state.finalized_state.block_hash, finalized_hash);
            prop_assert!(next_state.tick == tick(tick_value));
            prop_assert!(next_state.pending_requests.last_request_id_for_test() == last_request_id);
            prop_assert_eq!(
                next_state.pending_requests.len_for_test(),
                unrelated_requests.len()
            );
            prop_assert!(
                !next_state.pending_requests.contains(&target_request_id)
            );

            for (request_id, expected_payload) in unrelated_requests {
                prop_assert!(
                    pending_payload_matches(&next_state.pending_requests, request_id, &expected_payload),
                    "unrelated request must remain pending"
                );
            }

            assert_active_requests_have_exactly_one_dispatch_tick(&next_state);
            assert_state_invariants(&next_state);
        }

        // Generates arbitrary pending requests and reports not-found for an unused id.
        // This ensures unsolicited not-found responses leave state and pending work untouched.
        #[test]
        fn unknown_block_header_not_found_preserves_state(
            tick_value in any::<u64>(),
            generated_payloads in prop::collection::vec(
                generated_request_payload_strategy(),
                0..32,
            ),
        ) {
            let finalized_hash = hash_for_node(0);
            let known_hash = hash_for_node(1);
            let mut state = empty_state_at(finalized_hash);
            state.tick = tick(tick_value);
            state.canonical_tip = known_hash;
            state
                .blocks
                .0
                .insert(known_hash, block_with_parent(finalized_hash));
            let mut expected_requests = Vec::new();

            for generated_payload in generated_payloads {
                let request_payload = request_payload_from_generated(generated_payload);
                let expected_payload = request_payload.clone();
                let (pending_requests, request_id) =
                    issue_expected_request(state.pending_requests, request_payload, state.tick);

                state.pending_requests = pending_requests;
                expected_requests.push((request_id, expected_payload));
            }

            let last_request_id = state.pending_requests.last_request_id_for_test();
            let dispatch_ticks = state.pending_requests.dispatch_ticks_for_test();
            let unknown_request_id = RequestId::from_raw_for_test(last_request_id.wrapping_add(1));

            let (next_state, effects) = transition(ChainKey::Ethereum,
                state,
                Event::BlockHeaderNotFound {
                    request_id: unknown_request_id,
                },
            );

            prop_assert!(effects.is_empty());
            prop_assert_eq!(next_state.canonical_tip, known_hash);
            let known_block = next_state
                .blocks
                .get(&known_hash)
                .ok_or_else(|| TestCaseError::fail("known block must remain present"))?;
            prop_assert_eq!(known_block.parent_hash, finalized_hash);
            prop_assert_eq!(next_state.finalized_state.block_hash, finalized_hash);
            prop_assert!(next_state.tick == tick(tick_value));
            prop_assert!(next_state.pending_requests.last_request_id_for_test() == last_request_id);
            prop_assert!(next_state.pending_requests.dispatch_ticks_for_test() == dispatch_ticks);
            prop_assert_eq!(
                next_state.pending_requests.len_for_test(),
                expected_requests.len()
            );

            for (request_id, expected_payload) in expected_requests {
                prop_assert!(
                    pending_payload_matches(&next_state.pending_requests, request_id, &expected_payload),
                    "pending request must remain present"
                );
            }

            assert_state_invariants(&next_state);
        }

        // Generates a linear chain, forces a not-found reset, then reobserves heads.
        // This proves reconstruction can recover after canonical ancestry was cleared.
        #[test]
        fn chain_reconstructs_after_not_found_reset_and_reobservation(
            chain in generated_linear_chain_strategy(),
        ) {
            let finalized_hash = hash_for_node(0);
            let first_head_index = *chain
                .observed_heads
                .first()
                .ok_or_else(|| TestCaseError::fail("generated chain must have an observed head"))?;
            let first_head_hash = hash_for_node(first_head_index);
            let first_parent_hash = hash_for_node(parent_index(&chain, first_head_index));
            let state = empty_state_at(finalized_hash);
            let (state, effects) = transition(ChainKey::Ethereum,
                state,
                Event::HeadObserved {
                    logs_bloom: bloom_matching_any(),
                    hash: first_head_hash,
                    parent_hash: first_parent_hash,
                },
            );
            let request_id = assert_single_block_header_request_effect(&effects, first_parent_hash);
            let state = drain_block_log_effects(state, &effects);

            let (mut state, effects) =
                transition(ChainKey::Ethereum, state, Event::BlockHeaderNotFound { request_id });

            prop_assert!(effects.is_empty());
            prop_assert!(state.blocks.0.is_empty());
            prop_assert_eq!(state.canonical_tip, finalized_hash);
            prop_assert!(state.pending_requests.is_empty_for_test());

            for head_index in &chain.observed_heads {
                let hash = hash_for_node(*head_index);
                let parent_hash = hash_for_node(parent_index(&chain, *head_index));

                state = apply_event_and_drain_block_headers(
                    state,
                    &chain,
                    Event::HeadObserved { hash, parent_hash , logs_bloom: bloom_matching_any() },
                );
            }

            let expected_blocks = expected_observed_ancestor_closure(&chain);
            prop_assert_eq!(state.blocks.0.len(), expected_blocks.len());

            for (hash, parent_hash) in expected_blocks {
                let block = state
                    .blocks
                    .get(&hash)
                    .ok_or_else(|| TestCaseError::fail("expected block to be present"))?;

                prop_assert_eq!(block.parent_hash, parent_hash);
            }

            let last_observed_head = chain
                .observed_heads
                .last()
                .map(|head_index| hash_for_node(*head_index))
                .unwrap_or(finalized_hash);

            prop_assert_eq!(state.canonical_tip, last_observed_head);
            prop_assert!(state.pending_requests.is_empty_for_test());
            assert_state_invariants(&state);
        }

        // Generates requested and returned pool-data sets independently.
        // This ensures response handling applies only requested result entries.
        #[test]
        fn pool_data_received_never_applies_unrequested_results(
            requested_bytes in prop::collection::hash_set(any::<u8>(), 0..16),
            result_bytes in prop::collection::hash_set(any::<u8>(), 0..32),
        ) {
            let finalized_hash = hash_for_node(0);
            let block_hash = hash_for_node(1);
            let requested_pools = requested_bytes
                .into_iter()
                .map(pool_address)
                .collect::<HashSet<_>>();
            let result_pools = result_bytes
                .into_iter()
                .map(|last_byte| (pool_address(last_byte), pool_data_result_for_byte(last_byte)))
                .collect::<HashMap<_, _>>();
            let mut state = empty_state_at(finalized_hash);

            state.canonical_tip = block_hash;
            state
                .blocks
                .0
                .insert(block_hash, block_with_parent(finalized_hash));
            let (pending_requests, request_id) = state.pending_requests.with_new_request(
                GetPoolData {
                    at: block_hash,
                    pools: requested_pools.clone(),
                },
                state.tick,
            );
            state.pending_requests = pending_requests;

            let (next_state, effects) = transition(ChainKey::Ethereum,
                state,
                Event::PoolDataReceived {
                    request_id,
                    pools: result_pools.clone(),
                },
            );

            prop_assert!(effects.is_empty());
            let block = next_state
                .blocks
                .get(&block_hash)
                .ok_or_else(|| TestCaseError::fail("block must remain present"))?;

            for (pool, result) in &result_pools {
                if requested_pools.contains(pool) {
                    match result {
                        Ok(pool_state) => {
                            prop_assert_eq!(block.pool_snapshots.get(pool), Some(pool_state));
                            prop_assert!(!block.pool_data_failures.contains_key(pool));
                        }
                        Err(failure) => {
                            prop_assert!(!block.pool_snapshots.contains_key(pool));
                            prop_assert_eq!(block.pool_data_failures.get(pool), Some(failure));
                        }
                    }
                } else {
                    prop_assert!(!block.pool_snapshots.contains_key(pool));
                    prop_assert!(!block.pool_data_failures.contains_key(pool));
                }
            }

            for pool in requested_pools {
                if !result_pools.contains_key(&pool) {
                    prop_assert!(!block.pool_snapshots.contains_key(&pool));
                    prop_assert!(!block.pool_data_failures.contains_key(&pool));
                }
            }

            assert_state_invariants(&next_state);
        }

        // Generates request payloads and fails their active ids.
        // This ensures retry effects keep the exact original payload while issuing fresh ids.
        #[test]
        fn request_failure_preserves_any_request_payload(
            generated_payload in generated_request_payload_strategy(),
        ) {
            let finalized_hash = hash_for_node(0);
            let known_block_hash = hash_for_node(1);
            let request_payload = request_payload_from_generated(generated_payload);
            let expected_request_payload = request_payload.clone();
            let unrelated_request_payload = GetBlockHeader {
                block_hash: hash_for_node(2),
            };
            let mut state = empty_state_at(finalized_hash);
            state.canonical_tip = known_block_hash;
            state
                .blocks
                .0
                .insert(known_block_hash, block_with_parent(finalized_hash));

            let (pending_requests, failed_request_id) =
                issue_expected_request(state.pending_requests, request_payload, state.tick);
            let (pending_requests, unrelated_request_id) =
                pending_requests.with_new_request(unrelated_request_payload.clone(), state.tick);
            state.pending_requests = pending_requests;

            let (next_state, effects) = transition(ChainKey::Ethereum,
                state,
                Event::RequestFailed {
                    request_id: failed_request_id,
                },
            );

            let retry_request_id =
                assert_single_request_effect(&effects, &expected_request_payload);
            prop_assert!(retry_request_id != failed_request_id);
            prop_assert_eq!(next_state.pending_requests.len_for_test(), 2);
            prop_assert!(
                !next_state.pending_requests.contains_any_for_test(failed_request_id)
            );
            prop_assert!(
                next_state.pending_requests.contains_any_for_test(retry_request_id)
            );
            prop_assert!(
                next_state.pending_requests.contains(&unrelated_request_id)
            );

            prop_assert!(
                pending_payload_matches(
                    &next_state.pending_requests,
                    retry_request_id,
                    &expected_request_payload
                ),
                "retry request must remain pending"
            );

            let unrelated_payload = next_state
                .pending_requests
                .get(&unrelated_request_id)
                .ok_or_else(|| TestCaseError::fail("unrelated request must remain pending"))?;
            prop_assert_eq!(
                unrelated_payload.payload.block_hash,
                unrelated_request_payload.block_hash
            );

            prop_assert!(
                next_state.pending_requests.last_request_id_for_test()
                    == any_request_id_raw(retry_request_id)
            );
            prop_assert_eq!(next_state.canonical_tip, known_block_hash);
            assert_single_unknown_block(&next_state, known_block_hash, finalized_hash);
            assert_state_invariants(&next_state);
        }

        // Generates arbitrary event sequences.
        // This broad state-machine property keeps safety invariants true across unexpected event orderings.
        #[test]
        fn arbitrary_header_result_failure_not_found_and_tick_events_preserve_state_safety(
            generated_events in generated_event_sequence_strategy(),
        ) {
            let finalized_hash = hash_for_node(0);
            let mut state = empty_state_at(finalized_hash);

            for generated_event in generated_events {
                let previous_tick = state.tick;
                let is_tick = matches!(generated_event, GeneratedEvent::Tick);
                let (next_state, effects) =
                    transition(ChainKey::Ethereum, state, event_from_generated(generated_event));

                if is_tick {
                    prop_assert!(next_state.tick == previous_tick.next());
                    assert_no_active_request_is_expired(&next_state);
                } else {
                    prop_assert!(next_state.tick == previous_tick);
                }

                assert_state_invariants(&next_state);
                assert_canonical_unknown_logs_are_pending(&next_state);
                assert_effects_are_well_formed(&next_state, &effects);
                assert_missing_parents_for_known_blocks_are_pending(&next_state);
                assert_active_requests_have_exactly_one_dispatch_tick(&next_state);

                state = next_state;
            }
        }
    }
}
