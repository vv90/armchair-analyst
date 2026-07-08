use std::collections::{HashMap, HashSet};

use alloy::primitives::{BlockHash, Bloom};

pub(crate) mod pending_requests;
pub(crate) mod pool_registry;
pub(crate) mod token_registry;
pub(crate) mod blocks_graph;

use self::{pending_requests::*, pool_registry::*, token_registry::*};
use crate::{ChainKey, PoolLog, pool_state::*, tick::Tick, uniswap_v4};


/// The optimization read's result: for each pool the canonical unfinalized path touched, its
/// freshest folded state, plus the frontier block the overlay is valid at. Carries owned states,
/// so it stays valid across graph mutations. Merge over
/// [`State::finalized_pool_snapshots`] for the full per-pool view — an untouched pool is
/// absent from the overlay.
#[derive(Clone, Debug, PartialEq)]
pub struct OptimizationStateUpdate {
    pub block_hash: BlockHash,
    pub pool_states: HashMap<PoolRef, PoolState>,
}

pub struct State {
    pending_requests: PendingRequests,
    pool_registry: TrustedPoolRegistry,
    token_registry: TokenRegistry,
    tick: Tick,
    /// Subscription-observed logs for blocks not yet in the graph, keyed by block hash. Drained
    /// into the block when it enters via a head/header observation. Bounded by
    /// [`MAX_STREAMED_LOG_BLOCKS`]; raw input staging only, safe to drop.
    streamed_logs: HashMap<BlockHash, Vec<PoolLog>>,
    /// The log-sourced blocks graph and its finalized base — the sole chain-state authority since
    /// Increment 4 (the legacy graph is deleted). See [`Blocks`].
    blocks: Blocks,
}

/// The log-sourced [`blocks_graph::BlocksGraph`] and its advancing finalized base snapshot. The
/// graph's anchor is the finalized hash's sole home (invariant A1); the base holds the folded
/// finalized pool states. A verified pool absent from the base is absent, never stale (Blockers
/// 1b/1c); coverage grows via anchor-height seeding
/// ([`schedule_finalized_pool_seed_requests`]) plus the fold's own absolute-log self-seeding.
#[derive(Debug)]
struct Blocks {
    graph: blocks_graph::BlocksGraph,
    /// The finalized base the graph folds over — advances on each `finalized_to` re-root.
    finalized_snapshot: HashMap<PoolRef, PoolState>,
}

impl Blocks {
    /// A fresh empty graph anchored at `finalized_hash` with the given finalized base (empty recent
    /// region, anchor = finalized) — the `State::init` / test-constructor starting point.
    fn new(finalized_hash: BlockHash, finalized_snapshot: HashMap<PoolRef, PoolState>) -> Blocks {
        Blocks {
            graph: blocks_graph::BlocksGraph::new(finalized_hash),
            finalized_snapshot,
        }
    }

    /// Merges anchor-height pool-state reads into the finalized base, seeding pools that had no
    /// foldable absolute state yet (Blockers 1b/1c). The base only ever holds states *at the anchor*,
    /// so this is the one path — besides `finalized_to`'s fold — that mutates the snapshot.
    ///
    /// Stale-`at` guard (the correctness backstop; `retaining_block_targets` is the best-effort early
    /// prune): a response whose target is no longer the anchor — finalization advanced while it was in
    /// flight — is dropped wholesale rather than written at the wrong height. Failed per-pool reads are
    /// skipped: the pool stays uncovered and is re-requested on a later idle scheduling round.
    fn with_finalized_pool_seeds(
        mut self,
        at: BlockHash,
        results: HashMap<PoolRef, PoolDataResult>,
    ) -> Blocks {
        if at != self.graph.anchor_hash() {
            return self;
        }
        for (pool, result) in results {
            if let Ok(pool_state) = result {
                self.finalized_snapshot.insert(pool, pool_state);
            }
        }
        self
    }
}

/// Caps how many not-yet-known blocks can hold buffered subscription logs at once, bounding the
/// staging map when observed logs for a block arrive but its head never does (e.g. a reorg).
const MAX_STREAMED_LOG_BLOCKS: usize = 1024;

/// Caps how many uncovered pools one finalized-pool-seed request names, so a chain starting with a
/// large verified set (empty snapshot at bootstrap) drains over successive idle scheduling rounds
/// instead of one oversized multicall. The single tuning knob for anchor-height seed throughput.
const FINALIZED_POOL_SEED_CHUNK: usize = 100;

impl State {
    /// Creates kernel state anchored at a finalized hash with no pending requests, recent blocks,
    /// or finalized pool snapshots.
    /// Added as the pure state-machine entry point for runtimes that will feed events and execute effects.
    pub fn init(finalized_hash: BlockHash) -> State {
        State {
            blocks: Blocks::new(finalized_hash, HashMap::new()),
            pending_requests: PendingRequests::new(),
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
        seed_blocks: Vec<(BlockHash, BlockHash, u64, Vec<PoolLog>)>,
    ) -> (State, Vec<Effect>) {
        let tick = Tick::initial();

        // The graph starts warm: the seed window enters as header-less `Complete` blocks (the
        // ranged-getLogs payload is the full pool-vocabulary log set per block, and absence is
        // proof of emptiness — both registry-independent, since the range query is topics-only).
        // The first live head that lands on the seeded chain makes the whole window foldable at
        // once, so the optimization read serves recent state without a per-block header walk.
        let log_graph = blocks_graph::BlocksGraph::from_seed(finalized_hash, seed_blocks);

        // Seed blocks can reference ancestors that were not themselves seeded: a no-log block is
        // absent from the candidate window, so the segment above it floats. Request a header for
        // each distinct missing ancestor (the graph's dangling pending parents) so ancestry
        // reconstruction reconnects every seeded block down to the finalized anchor, the same way
        // the live scheduling chain does for one gap at a time.
        let (pending_requests, effects) = log_graph.missing_parents().into_iter().fold(
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
            blocks: Blocks {
                graph: log_graph,
                finalized_snapshot: finalized_pool_snapshots,
            },
            pending_requests,
            pool_registry,
            token_registry,
            tick,
            streamed_logs: HashMap::new(),
        };

        (state, effects)
    }

    /// Exposes the graph's finalized base to pure read models — the merge target for the
    /// optimization overlay ([`Self::optimization_update`]). A verified pool not yet seeded (by the
    /// anchor-height seed path or the fold's absolute-log self-seed) is absent, never stale
    /// (Blocker 1b/1c).
    pub(crate) fn finalized_pool_snapshots(&self) -> &HashMap<PoolRef, PoolState> {
        &self.blocks.finalized_snapshot
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

    /// The production optimization read: the blocks graph's best-effort fold over its finalized
    /// base ([`blocks_graph::BlocksGraph::optimization_pool_states`]), reading `Streamed` as well
    /// as `Complete` logs so the optimizer runs on the freshest state rather than stalling on the
    /// last fully-verified block. Total: an unconnected or unfoldable head yields an empty overlay
    /// at the anchor (the finalized state — never stale, at worst behind). The returned
    /// `block_hash` is the fold frontier, the height the reserves are valid at; the dispatch gate
    /// and progress reporting key on it as they did on the legacy complete-block hash.
    pub(crate) fn optimization_update(&self, chain: ChainKey) -> OptimizationStateUpdate {
        let v4_manager = uniswap_v4::pool_manager_address(chain);
        // Hand the fold the verified tracked set so it watches — and can absolute-seed — pools
        // discovered after bootstrap (the graph is registry-free; identity comes from here).
        let verified = self.pool_registry.verified_pools(chain);
        let (pool_states, block_hash) = self.blocks.graph.optimization_pool_states(
            &self.blocks.finalized_snapshot,
            &verified,
            v4_manager,
        );
        OptimizationStateUpdate {
            block_hash,
            pool_states,
        }
    }

    /// Counts canonical blocks the tip is ahead of `reference_hash` on a connected path.
    /// Added so read models can measure fetch progress from an already-known frontier block
    /// (such as the last dispatched optimization block) without rebuilding the complete
    /// pool-state overlay; `None` mirrors a reference that is off the tip's connected path.
    /// The reference is the optimization frontier, so the
    /// walk and the frontier now live in the same graph.
    pub(crate) fn blocks_behind(&self, reference_hash: BlockHash) -> Option<usize> {
        self.blocks.graph.distance_from_head(reference_hash)
    }

    #[cfg(test)]
    /// Builds kernel state from projection-relevant parts for tests.
    /// Added to keep projection tests focused on pure reserve generation instead of replaying unrelated RPC scheduling events.
    pub(crate) fn for_pool_reserve_projection_test(
        finalized_hash: BlockHash,
        finalized_pool_snapshots: HashMap<PoolRef, PoolState>,
        pool_registry: TrustedPoolRegistry,
        token_registry: TokenRegistry,
    ) -> State {
        State {
            blocks: Blocks::new(finalized_hash, finalized_pool_snapshots),
            pending_requests: PendingRequests::new(),
            pool_registry,
            token_registry,
            tick: Tick::initial(),
            streamed_logs: HashMap::new(),
        }
    }

    /// Measures the connected canonical path length from the finalized anchor to the current tip.
    /// Added so wrappers can trigger finalized-header refreshes from graph distance without inspecting graph internals.
    /// The graph anchor IS the finalized boundary (invariant A1).
    pub(crate) fn canonical_path_len_from_finalized(&self) -> Option<usize> {
        let graph = &self.blocks.graph;
        graph.distance_from_head(graph.anchor_hash())
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

    /// Applies and clears any staged subscription logs for a block that has just entered the
    /// graph. Draining is gated on actual presence so a refused admission (self-parent,
    /// conflicting parent, pending cap) leaves the staged logs buffered for a later attempt.
    fn with_streamed_logs_drained(mut self, block_hash: BlockHash) -> State {
        if !self.blocks.graph.contains(block_hash) {
            return self;
        }
        match self.streamed_logs.remove(&block_hash) {
            Some(logs) => self.with_log_streamed(block_hash, logs),
            None => self,
        }
    }

    // ---- Log-sourced graph feed (Increment 2; sole path since Increment 4) ---------------------

    /// Maps the blocks graph in place, leaving the finalized base untouched.
    fn map_graph(
        mut self,
        f: impl FnOnce(blocks_graph::BlocksGraph) -> blocks_graph::BlocksGraph,
    ) -> State {
        self.blocks.graph = f(self.blocks.graph);
        self
    }

    /// `HeadObserved` feed: admit the block (refuse-and-keep, decision (b)) and mark it the
    /// observed head, then drain any streamed logs staged before the head arrived so a
    /// log-before-head ordering does not lose data.
    fn with_log_head_observed(
        self,
        hash: BlockHash,
        parent_hash: BlockHash,
        number: u64,
        bloom: Bloom,
    ) -> State {
        self.map_graph(move |graph| {
            let graph = graph.admitted(hash, parent_hash, number, bloom);
            // A self-parent observation is garbage input and leaves the head unchanged; every
            // other outcome — admit, duplicate, anchor-readmit — advances the head, which
            // `with_observed_head` matches (it updates only for a present or anchor hash).
            if hash == parent_hash {
                graph
            } else {
                graph.with_observed_head(hash)
            }
        })
        .with_streamed_logs_drained(hash)
    }

    /// `BlockHeaderReceived` feed: admit without advancing the observed head (a backfilled header
    /// is ancestry, not a tip signal), draining any pre-staged streamed logs.
    fn with_log_header_received(
        self,
        hash: BlockHash,
        parent_hash: BlockHash,
        number: u64,
        bloom: Bloom,
    ) -> State {
        self.map_graph(move |graph| graph.admitted(hash, parent_hash, number, bloom))
            .with_streamed_logs_drained(hash)
    }

    /// `LogObserved` feed: best-effort streamed logs (a no-op if the block is not yet admitted;
    /// such pre-head logs are staged in `streamed_logs` and drained at admission above).
    fn with_log_streamed(self, block_hash: BlockHash, logs: Vec<PoolLog>) -> State {
        self.map_graph(move |graph| graph.with_streamed_logs(block_hash, logs))
    }

    /// `BlockLogsReceived` feed: authoritative complete logs (replaces any streamed set).
    fn with_log_complete(self, block_hash: BlockHash, logs: Vec<PoolLog>) -> State {
        self.map_graph(move |graph| graph.with_complete_logs(block_hash, logs))
    }

    /// `PoolDataReceived` feed: merge anchor-height pool reads into the finalized base
    /// ([`Blocks::with_finalized_pool_seeds`], which owns the stale-`at` guard).
    fn with_finalized_pool_seeds(
        mut self,
        at: BlockHash,
        results: HashMap<PoolRef, PoolDataResult>,
    ) -> State {
        self.blocks = self.blocks.with_finalized_pool_seeds(at, results);
        self
    }


    /// Advances the finalized anchor from an observed finality signal, self-gated by the blocks
    /// graph: `finalized_to` no-ops on an absent/pending/unconnected/off-canonical target and
    /// advances only over the foldable canonical prefix (fold-on-demand, reorg-safety decisions
    /// (c)/(d)), folding the now-final logs into the finalized snapshot and pruning everything
    /// that no longer descends from the new anchor.
    fn with_finalized_block_observed(mut self, chain: ChainKey, block_hash: BlockHash) -> State {
        let v4_manager = uniswap_v4::pool_manager_address(chain);
        // Hand the fold the verified tracked set so it watches — and can absolute-seed — pools
        // discovered after bootstrap (the graph is registry-free; identity comes from here).
        let verified = self.pool_registry.verified_pools(chain);
        let Blocks {
            graph,
            finalized_snapshot,
        } = self.blocks;
        let anchor_before = graph.anchor_hash();
        let (graph, finalized_snapshot) =
            graph.finalized_to(block_hash, &finalized_snapshot, &verified, v4_manager);

        if graph.anchor_hash() != anchor_before {
            // Retire block-scoped requests whose targets the re-root pruned (the anchor is not a
            // node and block-scoped requests never target it), and evict the staging buffer: any
            // logs whose head has not arrived by now are almost certainly orphaned, and a real
            // future block re-fetches authoritatively.
            self.pending_requests = self
                .pending_requests
                .retaining_block_targets(&graph.node_hashes());
            self.streamed_logs = HashMap::new();
        }

        self.blocks = Blocks {
            graph,
            finalized_snapshot,
        };
        self
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
        number: u64,
    },
    FinalizedBlockObserved {
        block_hash: BlockHash,
    },
    BlockHeaderReceived {
        request_id: RequestId<GetBlockHeader>,
        hash: BlockHash,
        parent_hash: BlockHash,
        logs_bloom: Bloom,
        number: u64,
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
    /// Anchor-height absolute pool reads seeding the finalized snapshot (Blockers 1b/1c coverage).
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


/// Emits log-fetch requests for present canonical blocks whose logs are unknown.
/// Added so header connectivity automatically drives pool-affecting log discovery.
fn schedule_unknown_canonical_log_requests(
    chain: ChainKey,
    mut state: State,
    mut effects: Vec<Effect>,
) -> (State, Vec<Effect>) {
    let pending_log_hashes = state.pending_requests.pending_block_log_hashes();
    // The bloom gate only fires once there is a verified pool whose log completeness to protect; with
    // none (`trusted: None`), the per-block fetch is still the discovery channel and every
    // bloom-bearing block is fetched. Capture that "gate active" decision from the verified set
    // *before* adding the v4 PoolManager discovery anchor, so warmup behavior is unchanged.
    let mut trusted_addresses = state.pool_registry.verified_addresses(chain);
    let gate_active = !trusted_addresses.is_empty();
    if let Some(manager) = uniswap_v4::pool_manager_address(chain) {
        // Anchor on the singleton PoolManager so a block carrying only v4 activity is never
        // bloom-skipped; otherwise, once any v3 pool is verified, new v4 pools would never be found.
        trusted_addresses.insert(manager);
    }

    // Blocks on the graph's present chain still lacking authoritative logs, bloom-gated with the
    // scheduler-owned trusted set above and deduped against in-flight fetches. The bloom has no
    // false negatives, so a trusted pool that did emit is never skipped, keeping trusted-pool log
    // completeness unchanged.
    for block_hash in state
        .blocks
        .graph
        .unresolved_log_request_hashes(gate_active.then_some(&trusted_addresses), &pending_log_hashes)
    {
        let request_payload = GetBlockLogs { block_hash };
        let (pending_requests, request_id) = state
            .pending_requests
            .with_new_request(request_payload.clone(), state.tick);

        state.pending_requests = pending_requests;
        effects.push(Effect::Request(AnyIssuedRequest::BlockLogs(IssuedRequest {
            request_id,
            request_payload,
        })));
    }

    (state, effects)
}

/// Emits pool metadata requests for unvalidated candidate addresses on the canonical path.
/// Added to turn log emitters into verified/rejected registry entries before using them as pools.
fn schedule_unknown_canonical_pool_metadata_requests(
    chain: ChainKey,
    candidates_from_head: &[(BlockHash, HashSet<ProtocolPoolKey>)],
    mut state: State,
    mut effects: Vec<Effect>,
) -> (State, Vec<Effect>) {
    // Candidates are derived from the graph's stored logs (`candidates_from_head`, one present-chain
    // walk shared with the token scheduler): one request per block naming pools the registry has not
    // yet validated, deduped across blocks newest-first and against in-flight validations — the same
    // filtering the legacy walk applied.
    let mut unavailable_candidates = state.pending_requests.pending_pool_metadata_candidates();
    for (block_hash, candidates) in candidates_from_head {
        let request_candidates = candidates
            .iter()
            .copied()
            .filter(|candidate| {
                !state.pool_registry.is_known(chain, *candidate)
                    && !unavailable_candidates.contains(candidate)
            })
            .collect::<HashSet<_>>();
        if request_candidates.is_empty() {
            continue;
        }
        unavailable_candidates.extend(request_candidates.iter().copied());

        let request_payload = GetPoolMetadata {
            at: *block_hash,
            candidates: request_candidates,
        };
        let (pending_requests, request_id) = state
            .pending_requests
            .with_new_request(request_payload.clone(), state.tick);

        state.pending_requests = pending_requests;
        effects.push(Effect::Request(AnyIssuedRequest::PoolMetadata(
            IssuedRequest {
                request_id,
                request_payload,
            },
        )));
    }

    (state, effects)
}

/// Emits token metadata requests for tokens referenced by verified canonical pools.
/// Added so reserve projection can use known decimals and avoid guessing token scale.
fn schedule_unknown_canonical_token_metadata_requests(
    chain: ChainKey,
    candidates_from_head: &[(BlockHash, HashSet<ProtocolPoolKey>)],
    mut state: State,
    mut effects: Vec<Effect>,
) -> (State, Vec<Effect>) {
    // Token needs are derived from the graph's stored logs (`candidates_from_head`, one present-chain
    // walk shared with the pool scheduler): each block's candidates map through the registry to their
    // verified metadata's token pair, keeping only tokens the token registry does not know and no
    // in-flight request already covers — the same mapping the legacy walk applied.
    let mut unavailable_tokens = state.pending_requests.pending_token_metadata_tokens();
    for (block_hash, candidates) in candidates_from_head {
        let tokens = candidates
            .iter()
            .filter_map(|candidate| state.pool_registry.verified_pool(chain, *candidate))
            .filter_map(|pool| state.pool_registry.verified_metadata(pool))
            .flat_map(|metadata| {
                [
                    TokenAddress(metadata.token0, chain),
                    TokenAddress(metadata.token1, chain),
                ]
            })
            .filter(|token| {
                !state.token_registry.is_known(*token) && !unavailable_tokens.contains(token)
            })
            .collect::<HashSet<_>>();
        if tokens.is_empty() {
            continue;
        }
        unavailable_tokens.extend(tokens.iter().copied());

        let request_payload = GetTokenMetadata {
            at: *block_hash,
            tokens,
        };
        let (pending_requests, request_id) = state
            .pending_requests
            .with_new_request(request_payload.clone(), state.tick);

        state.pending_requests = pending_requests;
        effects.push(Effect::Request(AnyIssuedRequest::TokenMetadata(
            IssuedRequest {
                request_id,
                request_payload,
            },
        )));
    }

    (state, effects)
}

/// Emits an anchor-height pool-data seed request for verified pools still absent from the finalized
/// snapshot. Fills the coverage hole (Blockers 1b/1c) where such a pool only materializes once its
/// canonical path carries an absolute Swap/Initialize (`derive_pool_state(None, run)`); a Mint/Burn-
/// only pool — and every pool at bootstrap, which starts with an empty snapshot — otherwise stays
/// invisible. The read targets the finalized anchor so the result seeds the fold base directly; a
/// tip-targeted read would have no sink (the graph stores no per-block snapshots, invariant L6).
///
/// Coverage/liveness only — per-block `eth_getLogs` stays the completeness authority. Idle-gated
/// behind block-graph backfill (header/log fetches are latency-critical, seeding is not) and chunked,
/// so a large uncovered set drains over successive scheduling rounds without starving the critical
/// path or issuing an oversized multicall.
fn schedule_finalized_pool_seed_requests(
    chain: ChainKey,
    mut state: State,
    mut effects: Vec<Effect>,
) -> (State, Vec<Effect>) {
    if state.pending_requests.has_pending_block_backfill() {
        return (state, effects);
    }

    // Verified pools with no finalized base yet, minus those an in-flight seed already covers.
    let pending = state.pending_requests.pending_pool_data_pools();
    let mut candidates = state
        .pool_registry
        .verified_pools(chain)
        .into_iter()
        .filter(|pool| {
            !state.blocks.finalized_snapshot.contains_key(pool) && !pending.contains(pool)
        })
        .collect::<Vec<_>>();
    if candidates.is_empty() {
        return (state, effects);
    }
    // Sort before truncating so the chunk boundary is stable across rounds (a partially-drained set
    // makes deterministic forward progress rather than re-rolling a random subset each time).
    candidates.sort_unstable();
    candidates.truncate(FINALIZED_POOL_SEED_CHUNK);

    let request_payload = GetPoolData {
        at: state.blocks.graph.anchor_hash(),
        pools: candidates.into_iter().collect(),
    };
    let (pending_requests, request_id) = state
        .pending_requests
        .with_new_request(request_payload.clone(), state.tick);

    state.pending_requests = pending_requests;
    effects.push(Effect::Request(AnyIssuedRequest::PoolData(IssuedRequest {
        request_id,
        request_payload,
    })));

    (state, effects)
}

/// Issues header requests for every dangling pending-ancestor gap root in the blocks graph
/// ([`blocks_graph::BlocksGraph::missing_parents`]). Scheduler-derived (Increment 4) rather than
/// admission-return-derived: any event that reaches the scheduling chain re-emits a dropped or
/// still-unserved backfill, so recovery no longer depends on the next disconnected head arriving.
/// Dedup against in-flight headers stays in [`request_missing_header`].
fn schedule_missing_header_requests(mut state: State, mut effects: Vec<Effect>) -> (State, Vec<Effect>) {
    for missing_hash in state.blocks.graph.missing_parents() {
        let (pending_requests, new_effects) =
            request_missing_header(state.pending_requests, state.tick, missing_hash);
        state.pending_requests = pending_requests;
        effects.extend(new_effects);
    }

    (state, effects)
}

/// Runs every canonical follow-up scheduler in the order needed for dependencies between requests.
/// Added so all transitions converge through one scheduling path instead of duplicating follow-up logic per event arm.
fn schedule_unknown_canonical_requests(
    chain: ChainKey,
    state: State,
    effects: Vec<Effect>,
) -> (State, Vec<Effect>) {
    // Headers first: they precede the log/metadata effects exactly as the admission-time
    // `request_missing_header` calls they replace did.
    let (state, effects) = schedule_missing_header_requests(state, effects);
    let (state, effects) = schedule_unknown_canonical_log_requests(chain, state, effects);
    // One present-chain candidate walk serves both metadata schedulers.
    let candidates_from_head = state.blocks.graph.pool_log_candidates_from_head();
    let (state, effects) = schedule_unknown_canonical_pool_metadata_requests(
        chain,
        &candidates_from_head,
        state,
        effects,
    );
    let (state, effects) = schedule_unknown_canonical_token_metadata_requests(
        chain,
        &candidates_from_head,
        state,
        effects,
    );
    // Lowest priority: anchor-height coverage seeding, self-gated to run only when the block graph
    // is not mid-backfill.
    schedule_finalized_pool_seed_requests(chain, state, effects)
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
            number,
        } => {
            // Admission is refuse-and-keep for every bad input (reorg-safety decisions (a)/(b)):
            // a conflicting parent refuses the one block, cycles are structurally absent from the
            // connected forest, and nothing resets. The scheduling chain then derives all
            // follow-up work — header backfill, log fetches, candidate/token validation — from
            // the graph itself.
            let state = state.with_log_head_observed(hash, parent_hash, number, logs_bloom);
            if hash == parent_hash {
                // A self-parent head is provably-garbage input (a hash commits to its parent);
                // nothing was admitted, so instead of the scheduling chain (which cannot see the
                // refused block) fetch the true header directly to continue from honest data.
                let (pending_requests, effects) =
                    request_missing_header(state.pending_requests, state.tick, hash);
                (
                    State {
                        pending_requests,
                        ..state
                    },
                    effects,
                )
            } else {
                schedule_unknown_canonical_requests(chain, state, vec![])
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
            number,
        } => {
            // Admit without advancing the head (a backfilled header is ancestry, not a tip
            // signal), refuse-and-keep on any bad input — see the `HeadObserved` arm.
            let state = state.with_log_header_received(hash, parent_hash, number, logs_bloom);
            let (pending_requests, request_payload) = state.pending_requests.take(&request_id);
            let mut state = State {
                pending_requests,
                ..state
            };

            let effects = match request_payload {
                // Mismatched response (shouldn't happen): something other than the requested block
                // came back, so the original request must still succeed — reissue it. A matching
                // response, or an unsolicited one with no pending payload, needs no retry.
                Some(PendingPayload {
                    payload: request_payload,
                    ..
                }) if request_payload.block_hash != hash => {
                    let (pending_requests, request_id) = state
                        .pending_requests
                        .with_new_request(request_payload.clone(), state.tick);
                    state.pending_requests = pending_requests;
                    vec![Effect::Request(AnyIssuedRequest::BlockHeader(
                        IssuedRequest {
                            request_id,
                            request_payload,
                        },
                    ))]
                }
                _ => vec![],
            };

            schedule_unknown_canonical_requests(chain, state, effects)
        }
        Event::BlockHeaderNotFound { request_id } => {
            // Reorg-safety decision (e): refuse-and-keep, no reset. Drop the request only —
            // transient provider lag self-heals when the missing-parents scheduler re-emits on a
            // later event, and a genuinely fabricated ancestry retries only until finalization
            // prunes its pending subtree (bounded, like the case-(b) poisoning window). Not
            // re-scheduling here is deliberate: an immediate re-emit would hammer a provider that
            // just said "unknown"; the next head observation is the natural retry cadence.
            let (pending_requests, _payload) = state.pending_requests.take(&request_id);
            (
                State {
                    pending_requests,
                    ..state
                },
                vec![],
            )
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
                    .with_log_complete(block_hash, logs);

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
            if state.blocks.graph.contains(block_hash) {
                // Known block: merge provisionally (`Streamed`), then schedule the authoritative
                // `GetBlockLogs` (a streamed block still needs the complete fetch).
                let state = state.with_log_streamed(block_hash, logs);
                schedule_unknown_canonical_requests(chain, state, vec![])
            } else {
                // The head has not arrived yet: stage the logs until the block enters the graph;
                // `with_log_head_observed`/`with_log_header_received` drain this staging map into
                // the graph at admission.
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
                    payload: GetPoolData { at, .. },
                    ..
                }) => {
                    let state = State {
                        pending_requests,
                        ..state
                    }
                    .with_finalized_pool_seeds(at, pools);

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

    (state, effects)
}

#[cfg(test)]
mod tests {
    use super::*;

    use alloy::primitives::{Address, BloomInput, U160, U256, aliases::I24};

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

    /// Asserts the core shape invariants for kernel state: the graph's anchor/head discipline.
    /// The deeper topology/log invariants (T2/T5/T6, I1, L5) are pinned by the blocks_graph
    /// unit and property tests; this catches kernel-level wiring that would violate them.
    fn assert_state_invariants(state: &State) {
        let graph = &state.blocks.graph;
        assert!(
            !graph.contains(graph.anchor_hash()),
            "finalized anchor must not be present as a node"
        );
        let head = graph.observed_head_hash();
        assert!(
            head == graph.anchor_hash() || graph.contains(head),
            "observed head must be the anchor or a present node"
        );
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

    /// Every present-chain block whose logs are not yet authoritative (`Complete`), regardless of
    /// bloom: the gate-inactive query form enumerates them all.
    fn unresolved_present_chain_hashes(state: &State) -> Vec<BlockHash> {
        state
            .blocks
            .graph
            .unresolved_log_request_hashes(None, &HashSet::new())
    }

    /// Asserts every present canonical block with unknown logs has an active log request.
    /// This protects the log scheduler from leaving canonical blocks permanently unqueried.
    fn assert_canonical_unknown_logs_are_pending(state: &State) {
        let pending_log_hashes = state.pending_requests.pending_block_log_hashes();
        for block_hash in unresolved_present_chain_hashes(state) {
            assert!(
                pending_log_hashes.contains(&block_hash),
                "canonical block without complete logs must have a pending log request"
            );
        }
    }

    /// Asserts observed canonical log candidates are either known by the registry or pending validation.
    /// This keeps candidate discovery connected to the trust boundary.
    fn assert_canonical_resolved_candidates_are_known_or_pending(state: &State) {
        let pending_candidates = state.pending_requests.pending_pool_metadata_candidates();
        for (_block_hash, candidates) in state.blocks.graph.pool_log_candidates_from_head() {
            for candidate in candidates {
                assert!(
                    state.pool_registry.is_known(ChainKey::Ethereum, candidate)
                        || pending_candidates.contains(&candidate),
                    "canonical observed log candidate must be known or pending metadata validation"
                );
            }
        }
    }

    /// Asserts every present canonical block has authoritative logs after log-draining helpers run.
    /// Properties use it to distinguish incomplete draining from scheduler bugs.
    fn assert_present_canonical_logs_are_resolved(state: &State) {
        assert!(
            unresolved_present_chain_hashes(state).is_empty(),
            "present canonical block logs must be complete"
        );
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
            }
        }
    }

    /// Asserts every dangling pending-ancestor gap root has a pending header request.
    /// This keeps disconnected canonical paths from stalling.
    fn assert_missing_parents_for_known_blocks_are_pending(state: &State) {
        let pending_headers = state.pending_requests.pending_header_hashes_for_test();
        for missing_hash in state.blocks.graph.missing_parents() {
            assert!(
                pending_headers.contains(&missing_hash),
                "missing canonical parent must have a pending header request"
            );
        }
    }

    /// Maps generated node indexes to deterministic block hashes.
    /// This gives properties stable, compact hash fixtures without arbitrary hash construction.
    fn hash_for_node(node_index: usize) -> BlockHash {
        BlockHash::with_last_byte((node_index + 1) as u8)
    }

    /// Test-only block number recovered from a block hash's trailing byte. These tests encode block
    /// identity/height in `BlockHash::with_last_byte(_)`, so this yields a per-hash-stable,
    /// height-ordered number for the log-sourced graph's block-admission entry. There is no
    /// production consumer of the plumbed `number` yet (see `Event::HeadObserved`).
    fn block_number_for(hash: BlockHash) -> u64 {
        hash.0[31] as u64
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
                number: hash_index as u64,
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
                number: hash_index as u64,
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


    // Exercises State::init from a finalized anchor.
    // This confirms initialization itself does not schedule work or create recent blocks.
    #[test]
    fn state_init_from_finalized_state_starts_with_empty_tracking() {
        let finalized_hash = BlockHash::with_last_byte(1);

        let state = State::init(finalized_hash);

        assert_empty_initial_state_at(&state, finalized_hash);
        assert!(state.blocks.finalized_snapshot.is_empty());
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
            AnyRequestId::PoolMetadata(request_id) => request_id.raw_for_test(),
            AnyRequestId::TokenMetadata(request_id) => request_id.raw_for_test(),
            AnyRequestId::PoolData(request_id) => request_id.raw_for_test(),
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
                            number: block_number_for(block_hash),
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
                Effect::Request(AnyIssuedRequest::PoolMetadata(_)) => {}
                Effect::Request(AnyIssuedRequest::TokenMetadata(_)) => {}
                Effect::Request(AnyIssuedRequest::PoolData(_)) => {}
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
                            number: block_number_for(block_hash),
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
                Effect::Request(AnyIssuedRequest::PoolMetadata(_)) => {}
                Effect::Request(AnyIssuedRequest::TokenMetadata(_)) => {}
                Effect::Request(AnyIssuedRequest::PoolData(_)) => {}
            }
        }

        state
    }

    /// Builds a clean kernel state anchored at a finalized hash.
    /// This gives tests empty registries, no pending work, and deterministic tick state.
    fn empty_state_at(finalized_hash: BlockHash) -> State {
        State {
            blocks: Blocks::new(finalized_hash, HashMap::new()),
            pending_requests: PendingRequests::new(),
            pool_registry: TrustedPoolRegistry::new(),
            token_registry: TokenRegistry::new(),
            tick: tick(0),
            streamed_logs: HashMap::new(),
        }
    }

    /// Replays a linear chain of heads rooted at the finalized anchor through `transition`, so both
    /// graphs observe identical ancestry — the event-driven replacement for planting blocks into a
    /// graph directly (Increment 4 test migration: direct construction dies with the legacy graph).
    fn state_with_observed_chain(finalized_hash: BlockHash, hashes: &[BlockHash]) -> State {
        let mut state = empty_state_at(finalized_hash);
        let mut parent_hash = finalized_hash;
        for &hash in hashes {
            let (next_state, _) = transition(
                ChainKey::Ethereum,
                state,
                Event::HeadObserved {
                    number: block_number_for(hash),
                    logs_bloom: bloom_matching_any(),
                    hash,
                    parent_hash,
                },
            );
            state = next_state;
            parent_hash = hash;
        }
        state
    }

    /// Mirrors a directly-planted linear chain into the blocks graph through its production
    /// admission API, so scheduler-input tests whose setup writes legacy blocks directly keep both
    /// graphs in step (Increment 4 migration; the legacy planting dies with the legacy graph).
    /// The observed head lands on the last planted hash (or stays at the anchor for an empty chain).
    fn plant_chain(state: &mut State, finalized_hash: BlockHash, hashes: &[BlockHash]) {
        let mut graph = Blocks::new(finalized_hash, HashMap::new()).graph;
        let mut parent_hash = finalized_hash;
        for (index, &hash) in hashes.iter().enumerate() {
            graph = graph.admitted(hash, parent_hash, (index + 1) as u64, bloom_matching_any());
            parent_hash = hash;
        }
        state.blocks.graph = graph.with_observed_head(parent_hash);
    }

    /// Mirrors a directly-planted resolved block into the blocks graph: admitted as the anchor's
    /// child with `Complete` swap logs naming each candidate, so the log-derived candidate walk
    /// sees what the legacy `Resolved(candidates)` planting expressed.
    fn plant_block_with_candidates(
        state: &mut State,
        finalized_hash: BlockHash,
        block_hash: BlockHash,
        candidates: &HashSet<ProtocolPoolKey>,
    ) {
        let logs: Vec<PoolLog> = candidates
            .iter()
            .enumerate()
            .map(|(index, candidate)| swap_log(*candidate, index as u64, &pool_state(9)))
            .collect();
        state.blocks.graph = Blocks::new(finalized_hash, HashMap::new())
            .graph
            .admitted(block_hash, finalized_hash, 1, bloom_matching_any())
            .with_complete_logs(block_hash, logs)
            .with_observed_head(block_hash);
    }

    /// Asserts the common empty-state baseline: an empty graph anchored at the finalized hash with
    /// the head at the anchor and no pending work.
    fn assert_empty_initial_state_at(state: &State, finalized_hash: BlockHash) {
        let graph = &state.blocks.graph;
        assert!(graph.node_hashes().is_empty());
        assert_eq!(graph.anchor_hash(), finalized_hash);
        assert_eq!(graph.observed_head_hash(), finalized_hash);
        assert!(state.pending_requests.is_empty_for_test());
    }

    /// Asserts a single admitted block whose logs are not yet authoritative.
    /// This verifies header ingestion before the log fetch enriches the block.
    fn assert_single_unknown_block(state: &State, hash: BlockHash, parent_hash: BlockHash) {
        assert_single_block_with_parent(state, hash, parent_hash);
        assert_eq!(
            state.blocks.graph.has_complete_logs_for_test(hash),
            Some(false),
            "block logs must not be authoritative yet"
        );
    }

    /// Asserts a tracked block is present with the expected parent reference.
    /// This lets ancestry tests ignore log status when it is scenario-specific.
    fn assert_single_block_with_parent(state: &State, hash: BlockHash, parent_hash: BlockHash) {
        assert_eq!(
            state.blocks.graph.parent_hash_for_test(hash),
            Some(parent_hash),
            "block must be present with the expected parent"
        );
    }

    /// Asserts authoritative log candidates were stored on one block.
    /// This catches decoded-log application bugs without depending on candidate ordering.
    fn assert_resolved_pool_logs(
        state: &State,
        hash: BlockHash,
        parent_hash: BlockHash,
        expected_logs: &HashSet<ProtocolPoolKey>,
    ) {
        assert_single_block_with_parent(state, hash, parent_hash);
        assert_eq!(
            state.blocks.graph.has_complete_logs_for_test(hash),
            Some(true),
            "block logs must be authoritative"
        );
        assert_eq!(
            state
                .blocks
                .graph
                .log_candidates_for_test(hash)
                .expect("block must be present"),
            *expected_logs
        );
    }

    /// Asserts the authoritative candidate set matches exactly (alias kept for call-site clarity).
    fn assert_resolved_candidate_logs(
        state: &State,
        hash: BlockHash,
        parent_hash: BlockHash,
        expected_logs: &HashSet<ProtocolPoolKey>,
    ) {
        assert_resolved_pool_logs(state, hash, parent_hash, expected_logs);
    }

    /// Asserts the block's stored candidates project through the registry to exactly the expected
    /// trusted pools. This keeps block storage from duplicating trust decisions.
    fn assert_trusted_pool_logs_resolved(
        state: &State,
        hash: BlockHash,
        expected_pools: HashSet<PoolRef>,
    ) {
        let candidates = state
            .blocks
            .graph
            .log_candidates_for_test(hash)
            .expect("block must be present");
        let trusted: HashSet<PoolRef> = candidates
            .iter()
            .filter_map(|candidate| {
                state
                    .pool_registry
                    .verified_pool(ChainKey::Ethereum, *candidate)
            })
            .collect();
        assert_eq!(trusted, expected_pools);
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
                number: block_number_for(head_hash),
                logs_bloom: bloom_matching_any(),
                hash: head_hash,
                parent_hash: head_hash,
            },
        );

        assert!(next_state.blocks.graph.node_hashes().is_empty());
        assert_eq!(next_state.blocks.graph.observed_head_hash(), finalized_hash);
        let request_id = assert_single_block_header_request_effect(&effects, head_hash);
        assert!(next_state.pending_requests.contains(&request_id));
        assert_state_invariants(&next_state);
    }

    // Observes a known block with a conflicting parent.
    // Reorg-safety decision (b): provably-wrong data is refused and the first-seen block kept —
    // one bad report can no longer wipe the recent graph (the legacy reset is gone).
    #[test]
    fn head_observed_with_conflicting_parent_is_refused_and_keeps_first_seen() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let head_hash = BlockHash::with_last_byte(2);
        let original_parent_hash = finalized_hash;
        let conflicting_parent_hash = BlockHash::with_last_byte(3);
        let state = state_with_observed_chain(finalized_hash, &[head_hash]);

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                number: block_number_for(head_hash),
                logs_bloom: bloom_matching_any(),
                hash: head_hash,
                parent_hash: conflicting_parent_hash,
            },
        );

        assert_single_block_with_parent(&next_state, head_hash, original_parent_hash);
        assert_eq!(
            next_state.blocks.graph.observed_head_hash(),
            head_hash
        );
        // The first observation's log request is still the only pending work; the refused
        // duplicate scheduled nothing new.
        assert!(effects.is_empty());
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
        assert_eq!(state.blocks.graph.anchor_hash(), finalized_hash);
        assert_eq!(
            state.finalized_pool_snapshots(),
            &HashMap::from([(pool, snapshot)])
        );
        assert_eq!(state.verified_pool_metadata(pool), Some(&metadata));
        assert_eq!(
            state.verified_token_metadata(token0),
            Some(&token_metadata(18))
        );
        assert_eq!(state.blocks.graph.observed_head_hash(), finalized_hash);
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
        let _pool = pool_address(4);
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
            vec![(
                seed_hash,
                finalized_hash,
                block_number_for(seed_hash),
                pool_logs(&HashSet::from([candidate])),
            )],
        );

        // The seed block's parent is the finalized anchor, so nothing needs reconnecting.
        assert!(activation_effects.is_empty());

        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                number: block_number_for(new_head),
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
        assert_eq!(state.blocks.graph.observed_head_hash(), new_head);
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
                (filler, finalized_hash, block_number_for(filler), Vec::new()),
                (
                    seed_hash,
                    filler,
                    block_number_for(seed_hash),
                    pool_logs(&HashSet::from([candidate])),
                ),
            ],
        );

        // The graph is fully connected, so no ancestor needs fetching.
        assert!(activation_effects.is_empty());

        let (state, _effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                number: block_number_for(new_head),
                logs_bloom: bloom_matching_any(),
                hash: new_head,
                parent_hash: seed_hash,
            },
        );

        assert_eq!(state.blocks.graph.observed_head_hash(), new_head);
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
                (filler, finalized_hash, block_number_for(filler), Vec::new()),
                (
                    seed_hash,
                    filler,
                    block_number_for(seed_hash),
                    pool_logs(&HashSet::from([candidate])),
                ),
            ],
        );
        let (state, _effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                number: block_number_for(old_head),
                logs_bloom: bloom_matching_any(),
                hash: old_head,
                parent_hash: seed_hash,
            },
        );
        assert_eq!(state.blocks.graph.observed_head_hash(), old_head);

        // Reorg: a competing head whose parent is a real block inside the formerly-bridged region.
        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                number: block_number_for(new_head),
                logs_bloom: bloom_matching_any(),
                hash: new_head,
                parent_hash: new_gap_block,
            },
        );
        // The new branch's missing real ancestor is fetched — the graph grew, it did not reset.
        let header_request_id = assert_single_block_header_request_effect(&effects, new_gap_block);
        assert_eq!(state.blocks.graph.observed_head_hash(), new_head);

        // The real gap block resolves, forking at the anchor: the new branch connects through.
        let (state, _effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::BlockHeaderReceived {
                number: block_number_for(new_gap_block),
                logs_bloom: bloom_matching_any(),
                request_id: header_request_id,
                hash: new_gap_block,
                parent_hash: finalized_hash,
            },
        );

        assert_eq!(state.blocks.graph.observed_head_hash(), new_head);
        // new_head -> new_gap_block -> anchor: the winning branch is real, with no filler.
        assert_eq!(state.canonical_path_len_from_finalized(), Some(2));
        // The filler and the old branch survive as an orphan (no reset would have kept them).
        assert!(state.blocks.graph.contains(filler));
        assert!(state.blocks.graph.contains(seed_hash));
        assert!(state.blocks.graph.contains(old_head));
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

    /// A one-block range-logs entry, as the bootstrap candidate scan would report it: one decoded
    /// delta-only log naming the block's candidate (derives no snapshot, like `pool_logs`).
    fn range_log_block(number: u64) -> RangeLogBlock {
        RangeLogBlock {
            number,
            hash: block_hash_for_number(number),
            logs: pool_logs(&HashSet::from([candidate_for_number(number)])),
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
            .map(|block| (block.hash, block.parent_hash, block.number, block.logs))
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

    /// Asserts the transition scheduled no priority (reconstruction/freshness) work. With the
    /// background pool-data backfill deleted (Increment 4), an idle pending tier emits nothing at
    /// all, so any effect indicates unexpected priority scheduling.
    fn assert_no_priority_effects(effects: &[Effect]) {
        assert!(effects.is_empty(), "expected no scheduled effects");
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
                number: block_number_for(fresh_head_hash()),
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
                number: block_number_for(fresh_head_hash()),
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

        // The bridge is a filler: a seeded block with no logs (real blocks each contribute one).
        let filler_count = outcome
            .seed_blocks
            .iter()
            .filter(|block| block.logs.is_empty())
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
                            number: block_number_for(fresh_head_hash()),
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
    // This keeps block graph updates idempotent: the duplicate changes nothing and schedules
    // nothing new (the first observation's log request is still pending).
    #[test]
    fn head_observed_with_duplicate_matching_block_does_not_change_state() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let head_hash = BlockHash::with_last_byte(2);
        let state = state_with_observed_chain(finalized_hash, &[head_hash]);

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                number: block_number_for(head_hash),
                logs_bloom: bloom_matching_any(),
                hash: head_hash,
                parent_hash: finalized_hash,
            },
        );

        assert_eq!(next_state.blocks.graph.node_hashes().len(), 1);
        assert_eq!(next_state.blocks.graph.observed_head_hash(), head_hash);
        assert_single_unknown_block(&next_state, head_hash, finalized_hash);
        assert!(effects.is_empty());
        assert_state_invariants(&next_state);
    }

    // Observes heads whose raw parents reference each other — a pending-pending cycle, provably
    // fabricated data. Reorg-safety decision (a): such a cycle is representable only among pending
    // nodes, is never followed by any walk, and is bounded — nothing resets, and no scheduling
    // work is derived from the cyclic branch.
    #[test]
    fn head_observed_pending_cycle_is_inert_and_does_not_reset() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let first_hash = BlockHash::with_last_byte(2);
        let second_hash = BlockHash::with_last_byte(3);
        let state = empty_state_at(finalized_hash);

        let (state, _effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                number: block_number_for(first_hash),
                logs_bloom: bloom_matching_any(),
                hash: first_hash,
                parent_hash: second_hash,
            },
        );
        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                number: block_number_for(second_hash),
                logs_bloom: bloom_matching_any(),
                hash: second_hash,
                parent_hash: first_hash,
            },
        );

        let graph = &next_state.blocks.graph;
        assert!(graph.contains(first_hash));
        assert!(graph.contains(second_hash));
        assert_eq!(graph.anchor_hash(), finalized_hash);
        // Both cyclic parents are present, so nothing is "missing" to fetch, and the cyclic
        // present-chain walk yields no log work.
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
                number: block_number_for(finalized_hash),
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
                number: block_number_for(head_hash),
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
                number: block_number_for(head_hash),
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
                number: block_number_for(first_head),
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
                number: block_number_for(second_head),
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
                number: block_number_for(first_head),
                logs_bloom: bloom_matching_any(),
                hash: first_head,
                parent_hash: first_missing_parent,
            },
        );

        let (next_state, second_effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                number: block_number_for(second_head),
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
                number: block_number_for(head_hash),
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
                number: block_number_for(missing_parent_hash),
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

        plant_chain(
            &mut state,
            finalized_hash,
            &[grandparent_hash, parent_hash, head_hash],
        );

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                number: block_number_for(head_hash),
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
                number: block_number_for(head_hash),
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
                number: block_number_for(parent_hash),
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
                number: block_number_for(parent_hash),
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
                number: block_number_for(head_hash),
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
        // A block whose authoritative logs are already complete must not be refetched.
        state.blocks.graph = Blocks::new(finalized_hash, HashMap::new())
            .graph
            .admitted(head_hash, finalized_hash, 1, bloom_matching_any())
            .with_complete_logs(head_hash, Vec::new())
            .with_observed_head(head_hash);

        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                number: block_number_for(head_hash),
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

        plant_chain(&mut state, finalized_hash, &[block_hash]);
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

        plant_chain(&mut state, finalized_hash, &[block_hash]);
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

        plant_chain(&mut state, finalized_hash, &[block_hash]);

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
        assert!(!next_state.blocks.graph.contains(missing_block_hash));
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

        plant_chain(&mut state, finalized_hash, &[block_hash]);
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
        plant_chain(&mut state, finalized_hash, &[block_hash]);
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
        plant_chain(&mut state, finalized_hash, &[block_hash]);
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
        assert_trusted_pool_logs_resolved(&next_state, block_hash, HashSet::new());
        assert_state_invariants(&next_state);
    }






    // A subscription log on a known block records a provisional (`Partial`) snapshot — enough to
    // skip the pool-data read — while still scheduling the authoritative `GetBlockLogs`.
    #[test]
    fn log_observed_on_known_block_is_provisional_and_still_fetches_authoritative_logs() {
        let finalized = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let candidate = pool_candidate_address(3);
        let _pool = pool_address(3);
        let mut state = empty_state_at(finalized);

        state.pool_registry = registry_verifying(candidate);
        plant_chain(&mut state, finalized, &[block_hash]);

        let derived = pool_state(7);
        let (next_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::LogObserved {
                block_hash,
                logs: vec![swap_log(candidate, 0, &derived)],
            },
        );

        // The streamed logs merged provisionally (not authoritative), so the authoritative fetch
        // is still scheduled and the candidate is visible for discovery.
        assert_eq!(
            next_state
                .blocks
                .graph
                .has_complete_logs_for_test(block_hash),
            Some(false)
        );
        assert_eq!(
            next_state
                .blocks
                .graph
                .log_candidates_for_test(block_hash),
            Some(HashSet::from([candidate]))
        );
        assert_single_block_log_request_effect(&effects, block_hash);
        assert_state_invariants(&next_state);
    }

    // A subscription log can arrive before the block's head: it is staged and applied once the head
    // brings the block into the graph.
    #[test]
    fn log_observed_for_unknown_block_buffers_until_head_arrives() {
        let finalized = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let candidate = pool_candidate_address(3);
        let _pool = pool_address(3);
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

        // Buffering the log emits nothing.
        assert_no_priority_effects(&effects);
        assert!(!buffered_state.blocks.graph.contains(block_hash));

        let (next_state, _effects) = transition(
            ChainKey::Ethereum,
            buffered_state,
            Event::HeadObserved {
                number: block_number_for(block_hash),
                logs_bloom: bloom_matching_any(),
                hash: block_hash,
                parent_hash: finalized,
            },
        );

        // Admission drained the staged logs into the block as a provisional (streamed) set.
        assert_eq!(
            next_state
                .blocks
                .graph
                .has_complete_logs_for_test(block_hash),
            Some(false)
        );
        assert_eq!(
            next_state
                .blocks
                .graph
                .log_candidates_for_test(block_hash),
            Some(HashSet::from([candidate]))
        );
        assert!(next_state.streamed_logs.is_empty());
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

        plant_chain(&mut state, finalized_hash, &[parent_hash, head_hash]);
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

        plant_block_with_candidates(&mut state, finalized_hash, block_hash, &candidates);
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

        plant_block_with_candidates(
            &mut state,
            finalized_hash,
            block_hash,
            &HashSet::from([candidate]),
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



    // Verifies a single candidate so its resolved logs project to a trusted pool.
    fn registry_verifying(candidate: ProtocolPoolKey) -> TrustedPoolRegistry {
        TrustedPoolRegistry::new().with_metadata_results(
            ChainKey::Ethereum,
            HashMap::from([(candidate, Ok(pool_metadata(1, 2, UniswapV3Fee::Fee500)))]),
        )
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
        let _pool = PoolRef { key: candidate, chain: ChainKey::Ethereum };
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
        plant_chain(&mut state, finalized_hash, &[block_hash]);
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

        assert_effects_are_well_formed(&next_state, &effects);
        assert_state_invariants(&next_state);
    }


    /// Registers one verified pool and returns a state whose only canonical block (`block_hash`,
    /// child of the finalized anchor) carries `logs_bloom`, plus the verified pool's address. The
    /// scaffold for the bloom-gate tests below. Mirrors the block into the blocks graph too — the
    /// request source since Increment 4. A `None` bloom is mirrored as all-ones: a header-less
    /// non-`Complete` node is unreachable through the graph's production feeds, and the all-ones
    /// bloom pins the same conservative-fetch behavior legacy's `None` did (true `None` handling
    /// is pinned by the blocks_graph gate tests).
    fn state_with_one_verified_pool_and_block(
        finalized_hash: BlockHash,
        block_hash: BlockHash,
        logs_bloom: Option<Bloom>,
        pool_logs: PlantedLogs,
    ) -> (State, Address) {
        let candidate = pool_candidate_address(3);
        let mut state = empty_state_at(finalized_hash);
        state.pool_registry = TrustedPoolRegistry::new().with_metadata_results(
            ChainKey::Ethereum,
            HashMap::from([(candidate, Ok(pool_metadata(1, 2, UniswapV3Fee::Fee3000)))]),
        );

        let candidate_logs = |candidates: &HashSet<ProtocolPoolKey>| -> Vec<PoolLog> {
            candidates
                .iter()
                .enumerate()
                .map(|(index, candidate)| swap_log(*candidate, index as u64, &pool_state(9)))
                .collect()
        };
        let graph = Blocks::new(finalized_hash, HashMap::new()).graph.admitted(
            block_hash,
            finalized_hash,
            1,
            logs_bloom.unwrap_or_else(bloom_matching_any),
        );
        let graph = match &pool_logs {
            PlantedLogs::Unknown => graph,
            PlantedLogs::Streamed(candidates) => {
                graph.with_streamed_logs(block_hash, candidate_logs(candidates))
            }
            PlantedLogs::Complete(candidates) => {
                graph.with_complete_logs(block_hash, candidate_logs(candidates))
            }
        };
        state.blocks.graph = graph.with_observed_head(block_hash);

        (state, candidate.uniswap_v3_address().expect("v3 pool"))
    }

    /// The log state a bloom-gate test plants on its single block, as candidate sets (the
    /// graph stores real logs; the helper synthesizes one swap per candidate).
    enum PlantedLogs {
        Unknown,
        Streamed(HashSet<ProtocolPoolKey>),
        #[allow(dead_code)]
        Complete(HashSet<ProtocolPoolKey>),
    }

    // A block whose bloom contains none of the trusted pool addresses is never fetched: the bloom
    // proves no trusted pool emitted here, and the fold skips bloom-clear blocks the same way, so
    // no promotion write is needed — the block simply stays non-authoritative and unfetched.
    #[test]
    fn bloom_clear_block_resolves_empty_without_log_request() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let unrelated = Address::with_last_byte(9);
        let (state, _pool) = state_with_one_verified_pool_and_block(
            finalized_hash,
            block_hash,
            Some(bloom_containing(&[unrelated])),
            PlantedLogs::Unknown,
        );

        let (next_state, effects) =
            schedule_unknown_canonical_requests(ChainKey::Ethereum, state, vec![]);

        assert_no_block_log_request_effect(&effects, block_hash);
        assert_eq!(
            next_state
                .blocks
                .graph
                .has_complete_logs_for_test(block_hash),
            Some(false)
        );
        assert_effects_are_well_formed(&next_state, &effects);
        assert_state_invariants(&next_state);
    }

    // A block whose bloom contains a trusted pool address still gets the authoritative fetch, so
    // trusted-pool log completeness is unchanged.
    #[test]
    fn bloom_with_trusted_address_still_requests_logs() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        // The helper's verified candidate is deterministic, so the trusted address is known up
        // front and the bloom can be seeded with it so the gate must fetch.
        let trusted_pool_address = pool_candidate_address(3)
            .uniswap_v3_address()
            .expect("v3 pool");
        let (state, pool_address) = state_with_one_verified_pool_and_block(
            finalized_hash,
            block_hash,
            Some(bloom_containing(&[trusted_pool_address])),
            PlantedLogs::Unknown,
        );
        assert_eq!(pool_address, trusted_pool_address);

        let (next_state, effects) =
            schedule_unknown_canonical_requests(ChainKey::Ethereum, state, vec![]);

        let _ = assert_single_block_log_request_effect(&effects, block_hash);
        assert_effects_are_well_formed(&next_state, &effects);
        assert_state_invariants(&next_state);
    }

    // A bloom-clear block with subscription-discovered (streamed) candidates keeps them visible:
    // an unknown candidate still drives `PoolMetadata` validation even though the bloom gate
    // skips the authoritative fetch (best-effort discovery survives the skipped fetch).
    #[test]
    fn partial_block_bloom_clear_preserves_candidates_for_discovery() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let undiscovered = pool_candidate_address(7);
        let (state, _pool) = state_with_one_verified_pool_and_block(
            finalized_hash,
            block_hash,
            Some(bloom_containing(&[undiscovered.uniswap_v3_address().expect("v3 pool")])),
            PlantedLogs::Streamed(HashSet::from([undiscovered])),
        );

        let (next_state, effects) =
            schedule_unknown_canonical_requests(ChainKey::Ethereum, state, vec![]);

        assert_no_block_log_request_effect(&effects, block_hash);
        assert_eq!(
            next_state
                .blocks
                .graph
                .log_candidates_for_test(block_hash),
            Some(HashSet::from([undiscovered]))
        );
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
            PlantedLogs::Unknown,
        );

        let (next_state, effects) =
            schedule_unknown_canonical_requests(ChainKey::Ethereum, state, vec![]);

        let _ = assert_single_block_log_request_effect(&effects, block_hash);
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
            PlantedLogs::Unknown,
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
        // A block with an unrelated-address bloom — with no verified pools the gate is inactive,
        // so it must be fetched regardless.
        state.blocks.graph = Blocks::new(finalized_hash, HashMap::new())
            .graph
            .admitted(
                block_hash,
                finalized_hash,
                1,
                bloom_containing(&[Address::with_last_byte(9)]),
            )
            .with_observed_head(block_hash);

        let (next_state, effects) =
            schedule_unknown_canonical_requests(ChainKey::Ethereum, state, vec![]);

        let _ = assert_single_block_log_request_effect(&effects, block_hash);
        assert_effects_are_well_formed(&next_state, &effects);
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
        plant_chain(&mut state, finalized_hash, &[block_hash]);
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
        let state = state_with_observed_chain(
            finalized_hash,
            &[
                BlockHash::with_last_byte(2),
                BlockHash::with_last_byte(3),
                BlockHash::with_last_byte(4),
            ],
        );

        assert_eq!(state.canonical_path_len_from_finalized(), Some(3));
    }

    // Refuses to measure a canonical path with missing ancestry.
    // This prevents refresh policy from acting on a partial suffix.
    #[test]
    fn canonical_path_len_from_finalized_returns_none_for_disconnected_path() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let state = empty_state_at(finalized_hash);

        let (state, _) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                number: block_number_for(head_hash),
                logs_bloom: bloom_matching_any(),
                hash: head_hash,
                parent_hash: missing_parent_hash,
            },
        );

        assert_eq!(state.canonical_path_len_from_finalized(), None);
    }

    // Refuses to measure cyclic canonical ancestry.
    // This keeps refresh policy from treating corrupt graph structure as mature depth.
    // A cycle is representable only among pending nodes (connected ancestry is acyclic by
    // construction), and cannot be reached through events while the legacy reset still fires on
    // it, so the graph is built directly through its own admission path.
    #[test]
    fn canonical_path_len_from_finalized_returns_none_for_cyclic_path() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let first_hash = BlockHash::with_last_byte(2);
        let second_hash = BlockHash::with_last_byte(3);
        let mut state = empty_state_at(finalized_hash);

        state.blocks.graph = Blocks::new(finalized_hash, HashMap::new())
            .graph
            .admitted(first_hash, second_hash, 2, bloom_matching_any())
            .admitted(second_hash, first_hash, 3, bloom_matching_any())
            .with_observed_head(second_hash);

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
        let state = state_with_observed_chain(
            finalized_hash,
            &[
                BlockHash::with_last_byte(2),
                BlockHash::with_last_byte(3),
                BlockHash::with_last_byte(4),
            ],
        );

        assert_eq!(state.blocks_behind(finalized_hash), Some(3));
    }

    // A mid-path reference measures only the blocks newer than it up to the tip.
    #[test]
    fn blocks_behind_measures_distance_from_mid_path_reference() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let second_hash = BlockHash::with_last_byte(3);
        let state = state_with_observed_chain(
            finalized_hash,
            &[
                BlockHash::with_last_byte(2),
                second_hash,
                BlockHash::with_last_byte(4),
            ],
        );

        assert_eq!(state.blocks_behind(second_hash), Some(1));
    }

    // A reference off the tip's connected path is not measurable.
    #[test]
    fn blocks_behind_returns_none_for_disconnected_reference() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let state = empty_state_at(finalized_hash);

        let (state, _) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                number: block_number_for(head_hash),
                logs_bloom: bloom_matching_any(),
                hash: head_hash,
                parent_hash: missing_parent_hash,
            },
        );

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


    /// Drives one head + its authoritative (possibly empty) log response through `transition`,
    /// returning the next state — the event-driven builder for finalization tests that need a
    /// fully-`Complete` block on the canonical chain.
    fn observe_complete_block(
        state: State,
        hash: BlockHash,
        parent_hash: BlockHash,
        logs: Vec<PoolLog>,
    ) -> State {
        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                number: block_number_for(hash),
                logs_bloom: bloom_matching_any(),
                hash,
                parent_hash,
            },
        );
        let request_id = assert_single_block_log_request_effect(&effects, hash);
        let (state, _effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::BlockLogsReceived { request_id, logs },
        );
        state
    }

    // Finalization advances the graph anchor to a fully-complete target and folds the now-final
    // logs into the finalized snapshot: an affected pool moves to its folded state while
    // untouched pools carry over unchanged.
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

        state.blocks.finalized_snapshot = HashMap::from([
            (affected_pool, old_affected_snapshot),
            (unaffected_pool, unaffected_snapshot.clone()),
        ]);
        state.pool_registry = TrustedPoolRegistry::new().with_metadata_results(
            ChainKey::Ethereum,
            HashMap::from([(candidate, Ok(pool_metadata(1, 2, UniswapV3Fee::Fee3000)))]),
        );

        let state = observe_complete_block(
            state,
            target_hash,
            finalized_hash,
            vec![swap_log(candidate, 0, &new_affected_snapshot)],
        );
        let (state, _effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                number: block_number_for(tip_hash),
                logs_bloom: bloom_matching_any(),
                hash: tip_hash,
                parent_hash: target_hash,
            },
        );

        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::FinalizedBlockObserved {
                block_hash: target_hash,
            },
        );

        assert!(effects.is_empty());
        let graph = &state.blocks.graph;
        assert_eq!(graph.anchor_hash(), target_hash);
        assert_eq!(
            state.blocks.finalized_snapshot,
            HashMap::from([
                (affected_pool, new_affected_snapshot),
                (unaffected_pool, unaffected_snapshot),
            ])
        );
        assert!(!graph.contains(target_hash));
        assert!(graph.contains(tip_hash));
        assert_eq!(graph.observed_head_hash(), tip_hash);
        assert_state_invariants(&state);
    }

    // Compacts to the newest complete prefix when the observed finalized target's path still has
    // an unresolved bloom-hit block (fold-on-demand partial compaction, decision (c)).
    #[test]
    fn finalized_block_observed_compacts_to_latest_earlier_complete_block() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let complete_hash = BlockHash::with_last_byte(2);
        let incomplete_hash = BlockHash::with_last_byte(3);
        let target_hash = BlockHash::with_last_byte(4);
        let candidate = pool_candidate_address(5);
        let pool = PoolRef { key: candidate, chain: ChainKey::Ethereum };
        let folded = pool_state(6);
        let mut state = empty_state_at(finalized_hash);

        state.blocks.finalized_snapshot = HashMap::from([(pool, pool_state(9))]);
        state.pool_registry = TrustedPoolRegistry::new().with_metadata_results(
            ChainKey::Ethereum,
            HashMap::from([(candidate, Ok(pool_metadata(1, 2, UniswapV3Fee::Fee3000)))]),
        );

        let state = observe_complete_block(
            state,
            complete_hash,
            finalized_hash,
            vec![swap_log(candidate, 0, &folded)],
        );
        // The two newer blocks arrive but their authoritative logs never do.
        let state = state_with_more_observed_heads(
            state,
            complete_hash,
            &[incomplete_hash, target_hash],
        );

        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::FinalizedBlockObserved {
                block_hash: target_hash,
            },
        );

        assert!(effects.is_empty());
        let graph = &state.blocks.graph;
        assert_eq!(graph.anchor_hash(), complete_hash);
        assert_eq!(
            state.blocks.finalized_snapshot,
            HashMap::from([(pool, folded)])
        );
        assert!(!graph.contains(complete_hash));
        assert!(graph.contains(incomplete_hash));
        assert!(graph.contains(target_hash));
        assert_state_invariants(&state);
    }

    /// Extends an already-built state with further observed heads (no log responses).
    fn state_with_more_observed_heads(
        mut state: State,
        mut parent_hash: BlockHash,
        hashes: &[BlockHash],
    ) -> State {
        for &hash in hashes {
            let (next_state, _effects) = transition(
                ChainKey::Ethereum,
                state,
                Event::HeadObserved {
                    number: block_number_for(hash),
                    logs_bloom: bloom_matching_any(),
                    hash,
                    parent_hash,
                },
            );
            state = next_state;
            parent_hash = hash;
        }
        state
    }

    // Leaves state unchanged when no block past the current finalized anchor is complete. A pool
    // must be verified for the unresolved block to be a fold blocker — with nothing watched, no
    // block's logs are ever needed and the anchor would advance freely.
    #[test]
    fn finalized_block_observed_with_only_finalized_complete_is_noop() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let target_hash = BlockHash::with_last_byte(2);
        let mut state = state_with_observed_chain(finalized_hash, &[target_hash]);
        state.pool_registry = registry_verifying(pool_candidate_address(3));

        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::FinalizedBlockObserved {
                block_hash: target_hash,
            },
        );

        assert!(effects.is_empty());
        let graph = &state.blocks.graph;
        assert_eq!(graph.anchor_hash(), finalized_hash);
        assert!(graph.contains(target_hash));
        assert_eq!(graph.observed_head_hash(), target_hash);
        assert_state_invariants(&state);
    }

    // Leaves state unchanged when the observed finalized hash is not connected to the finalized anchor.
    // This prevents partial or malformed ancestry from changing the immutable snapshot boundary.
    #[test]
    fn finalized_block_observed_with_disconnected_target_is_noop() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let target_hash = BlockHash::with_last_byte(3);
        let state = empty_state_at(finalized_hash);

        let (state, _effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                number: block_number_for(target_hash),
                logs_bloom: bloom_matching_any(),
                hash: target_hash,
                parent_hash: missing_parent_hash,
            },
        );

        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::FinalizedBlockObserved {
                block_hash: target_hash,
            },
        );

        assert!(effects.is_empty());
        let graph = &state.blocks.graph;
        assert_eq!(graph.anchor_hash(), finalized_hash);
        assert!(graph.contains(target_hash));
        assert_state_invariants(&state);
    }

    // Leaves state unchanged when the observed finalized hash is on a non-canonical branch
    // (reorg-safety decision (d): a connected side-fork target no-ops and waits for the head).
    #[test]
    fn finalized_block_observed_for_non_canonical_target_is_noop() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let canonical_hash = BlockHash::with_last_byte(2);
        let side_hash = BlockHash::with_last_byte(3);
        let state = empty_state_at(finalized_hash);

        // Observe the side fork first, then the canonical head, so both descend from the anchor
        // and the observed head rests on the canonical branch.
        let state = state_with_more_observed_heads(state, finalized_hash, &[side_hash]);
        let state = state_with_more_observed_heads(state, finalized_hash, &[canonical_hash]);

        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::FinalizedBlockObserved {
                block_hash: side_hash,
            },
        );

        assert!(effects.is_empty());
        let graph = &state.blocks.graph;
        assert_eq!(graph.anchor_hash(), finalized_hash);
        assert!(graph.contains(canonical_hash));
        assert!(graph.contains(side_hash));
        assert_eq!(graph.observed_head_hash(), canonical_hash);
        assert_state_invariants(&state);
    }

    // Prunes pending requests targeting blocks the finalization re-root removed, while requests
    // for retained blocks survive. This ensures stale responses for pruned blocks cannot mutate
    // new finalized state.
    #[test]
    fn finalized_block_observed_prunes_removed_blocks_and_their_pending_requests() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let target_hash = BlockHash::with_last_byte(2);
        let retained_hash = BlockHash::with_last_byte(3);
        let side_hash = BlockHash::with_last_byte(4);
        let retained_token = token_address(5);
        let candidate = pool_candidate_address(6);
        let state = empty_state_at(finalized_hash);

        // A complete target with a retained (still-unresolved) child on top.
        let state = observe_complete_block(state, target_hash, finalized_hash, Vec::new());
        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                number: block_number_for(retained_hash),
                logs_bloom: bloom_matching_any(),
                hash: retained_hash,
                parent_hash: target_hash,
            },
        );
        let retained_logs_id = assert_single_block_log_request_effect(&effects, retained_hash);
        let mut state = state;

        // Requests targeting the soon-pruned target and an absent side block, plus one more
        // request for the retained block.
        let (pending_requests, removed_logs_id) = state.pending_requests.with_new_request(
            GetBlockLogs {
                block_hash: target_hash,
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
        let graph = &state.blocks.graph;
        assert_eq!(graph.anchor_hash(), target_hash);
        assert!(!graph.contains(target_hash));
        assert!(graph.contains(retained_hash));
        assert!(!state.pending_requests.contains(&removed_logs_id));
        assert!(!state.pending_requests.contains(&removed_metadata_id));
        assert!(state.pending_requests.contains(&retained_logs_id));
        assert!(state.pending_requests.contains(&retained_tokens_id));
        assert_state_invariants(&state);
    }



    // Fails an active pool-metadata request at the transport level.
    // This preserves candidate validation scope across retry replacement.
    #[test]
    fn pool_metadata_request_failure_retries_original_request() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let candidates = HashSet::from([pool_candidate_address(3), pool_candidate_address(4)]);
        let mut state = empty_state_at(finalized_hash);

        plant_chain(&mut state, finalized_hash, &[block_hash]);
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

        plant_chain(&mut state, finalized_hash, &[block_hash]);
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

        plant_chain(&mut state, finalized_hash, &[block_hash]);
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

        plant_chain(&mut state, finalized_hash, &[block_hash]);
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

        plant_chain(&mut state, finalized_hash, &[block_hash]);
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
                number: block_number_for(head_hash),
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
                number: block_number_for(missing_parent_hash),
                logs_bloom: bloom_matching_any(),
                request_id,
                hash: missing_parent_hash,
                parent_hash: finalized_hash,
            },
        );

        assert_eq!(next_state.blocks.graph.node_hashes().len(), 2);
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
    // Reorg-safety decision (e): refuse-and-keep — the request is dropped, the pending block and
    // every unrelated request stay put, and a later event's scheduling pass re-emits the backfill.
    #[test]
    fn block_header_not_found_for_matching_request_drops_only_that_request() {
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
                number: block_number_for(head_hash),
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
        // The pending block survives; only the not-found request is gone.
        assert_single_block_with_parent(&next_state, head_hash, missing_parent_hash);
        assert_eq!(next_state.blocks.graph.anchor_hash(), finalized_hash);
        assert!(next_state.tick == tick(7));
        assert!(next_state.pending_requests.last_request_id_for_test() == last_request_id);
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
                number: block_number_for(head_hash),
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
        assert_eq!(next_state.blocks.graph.observed_head_hash(), head_hash);
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
                number: block_number_for(head_hash),
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
                number: block_number_for(missing_parent_hash),
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
        assert_eq!(next_state.blocks.graph.observed_head_hash(), head_hash);
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
                number: block_number_for(head_hash),
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
        assert_eq!(next_state.blocks.graph.observed_head_hash(), head_hash);
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
                number: block_number_for(head_hash),
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
        assert_eq!(next_state.blocks.graph.observed_head_hash(), head_hash);
        assert_single_block_with_parent(&next_state, head_hash, missing_parent_hash);
        assert_eq!(next_state.pending_requests.len_for_test(), 1);
        assert!(next_state.pending_requests.contains(&retry_request_id));
        assert_state_invariants(&next_state);
        assert_missing_parents_for_known_blocks_are_pending(&next_state);
    }

    // Reports not-found for the active retry request.
    // Decision (e) still applies after retry replacement: only the retry request is dropped.
    #[test]
    fn block_header_not_found_for_current_retry_drops_only_that_request() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::HeadObserved {
                number: block_number_for(head_hash),
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
        assert_single_block_with_parent(&next_state, head_hash, missing_parent_hash);
        assert!(next_state.pending_requests.is_empty_for_test());
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
                number: block_number_for(head_hash),
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
        assert_single_block_with_parent(&state, head_hash, missing_parent_hash);
        assert!(state.pending_requests.is_empty_for_test());

        let (next_state, effects) = transition(ChainKey::Ethereum, state, Event::Tick);

        assert!(effects.is_empty());
        assert!(next_state.tick == tick(REQUEST_TTL));
        assert!(next_state.pending_requests.is_empty_for_test());
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
                number: block_number_for(head_hash),
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
                number: block_number_for(unrelated_hash),
                logs_bloom: bloom_matching_any(),
                request_id,
                hash: unrelated_hash,
                parent_hash: finalized_hash,
            },
        );

        assert_eq!(next_state.blocks.graph.observed_head_hash(), head_hash);
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

    // Returns a mismatched header response that also contradicts a known block's parent.
    // Reorg-safety decision (b): the conflicting report is refused (first-seen wins, no reset)
    // while the still-needed original header request is retried with a fresh id.
    #[test]
    fn conflicting_mismatched_header_response_is_refused_and_retries_request() {
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
                number: block_number_for(head_hash),
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
                number: block_number_for(unrelated_hash),
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
                number: block_number_for(unrelated_hash),
                logs_bloom: bloom_matching_any(),
                request_id,
                hash: unrelated_hash,
                parent_hash: conflicting_parent_hash,
            },
        );

        // Refuse-and-keep: the pending head and the first-seen unrelated block both survive, the
        // conflicting parent claim is discarded.
        assert_single_block_with_parent(&next_state, head_hash, missing_parent_hash);
        assert_single_block_with_parent(&next_state, unrelated_hash, finalized_hash);
        assert_eq!(next_state.blocks.graph.anchor_hash(), finalized_hash);
        let retry_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);
        assert!(next_state.pending_requests.contains(&retry_request_id));
        assert!(!next_state.pending_requests.contains(&request_id));
        assert_state_invariants(&next_state);
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
                number: block_number_for(head_hash),
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
                number: block_number_for(head_hash),
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
                number: block_number_for(head_hash),
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
                number: block_number_for(head_hash),
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
    // This ensures time advancement does not emit spurious effects or mutate chain state.
    #[test]
    fn empty_tick_only_advances_tick() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let known_hash = BlockHash::with_last_byte(2);
        let mut state = empty_state_at(finalized_hash);
        plant_chain(&mut state, finalized_hash, &[known_hash]);

        let (next_state, effects) = transition(ChainKey::Ethereum, state, Event::Tick);

        assert!(next_state.tick == tick(1));
        assert!(effects.is_empty());
        assert_eq!(next_state.blocks.graph.observed_head_hash(), known_hash);
        assert_eq!(next_state.blocks.graph.anchor_hash(), finalized_hash);
        assert_eq!(next_state.blocks.graph.node_hashes().len(), 1);
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
                number: block_number_for(head_hash),
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
        assert_eq!(next_state.blocks.graph.observed_head_hash(), head_hash);
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
                number: block_number_for(head_hash),
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
        assert_eq!(next_state.blocks.graph.observed_head_hash(), head_hash);
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
                number: block_number_for(head_hash),
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
                number: block_number_for(head_hash),
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
                number: block_number_for(head_hash),
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
                number: block_number_for(missing_parent_hash),
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
        assert_eq!(next_state.blocks.graph.observed_head_hash(), head_hash);
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
                number: block_number_for(head_hash),
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
                number: block_number_for(missing_parent_hash),
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
        assert_eq!(next_state.blocks.graph.observed_head_hash(), head_hash);
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
                number: block_number_for(head_hash),
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
                number: block_number_for(missing_parent_hash),
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
        assert_eq!(next_state.blocks.graph.observed_head_hash(), head_hash);
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
                number: block_number_for(head_hash),
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
                number: block_number_for(missing_parent_hash),
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
                number: block_number_for(head_hash),
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
                number: block_number_for(head_hash),
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
                number: block_number_for(missing_parent_hash),
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

            let planted: Vec<BlockHash> = (1..=chain_len).map(hash_for_node).collect();
            plant_chain(&mut state, finalized_hash, &planted);

            let (state, effects) = transition(ChainKey::Ethereum,
                state,
                Event::HeadObserved {
                    number: block_number_for(tip_hash),
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
                    number: block_number_for(tip_hash),
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
                    Event::HeadObserved { hash, parent_hash , logs_bloom: bloom_matching_any(), number: block_number_for(hash) },
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
                        number: block_number_for(block_hash),
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
            let mut last_hash = finalized_hash;
            let mut pool_metadata_results = HashMap::new();
            let mut graph = Blocks::new(finalized_hash, HashMap::new()).graph;

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

                // Mirror the resolved block into the blocks graph — the candidate source the
                // token scheduler reads since Increment 4.
                let planted_logs: Vec<PoolLog> = candidates
                    .iter()
                    .enumerate()
                    .map(|(index, candidate)| swap_log(*candidate, index as u64, &pool_state(9)))
                    .collect();
                graph = graph
                    .admitted(
                        block_hash,
                        parent_hash,
                        (block_index + 1) as u64,
                        bloom_matching_any(),
                    )
                    .with_complete_logs(block_hash, planted_logs);

                last_hash = block_hash;
                parent_hash = block_hash;
            }
            state.blocks.graph = graph.with_observed_head(last_hash);

            state.pool_registry = state.pool_registry.with_metadata_results(ChainKey::Ethereum, pool_metadata_results);
            let (state, effects) = schedule_unknown_canonical_requests(ChainKey::Ethereum, state, vec![]);
            assert_effects_are_well_formed(&state, &effects);
            let pending_tokens = state.pending_requests.pending_token_metadata_tokens();

            for (_block_hash, candidates) in state.blocks.graph.pool_log_candidates_from_head() {
                for candidate in candidates {
                    if let Some(pool) = state.pool_registry.verified_pool(ChainKey::Ethereum, candidate) {
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
                    Event::HeadObserved { hash, parent_hash , logs_bloom: bloom_matching_any(), number: block_number_for(hash) },
                );
            }

            let expected_blocks = expected_observed_ancestor_closure(&chain);
            prop_assert_eq!(state.blocks.graph.node_hashes().len(), expected_blocks.len());

            for (hash, parent_hash) in expected_blocks {
                prop_assert_eq!(
                    state.blocks.graph.parent_hash_for_test(hash),
                    Some(parent_hash)
                );
                prop_assert_eq!(
                    state.blocks.graph.has_complete_logs_for_test(hash),
                    Some(true)
                );
            }

            let last_observed_head = chain
                .observed_heads
                .last()
                .map(|head_index| hash_for_node(*head_index))
                .unwrap_or(finalized_hash);

            prop_assert_eq!(state.blocks.graph.observed_head_hash(), last_observed_head);
            prop_assert!(state.pending_requests.is_empty_for_test());
            prop_assert!(!state.blocks.graph.contains(finalized_hash));
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
                            number: block_number_for(hash),
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
                                number: block_number_for(block_hash),
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
                    Effect::Request(AnyIssuedRequest::PoolMetadata(_)) => {}
                    Effect::Request(AnyIssuedRequest::TokenMetadata(_)) => {}
                    Effect::Request(AnyIssuedRequest::PoolData(_)) => {}
                }
            }

            let expected_blocks = expected_observed_ancestor_closure(&chain);
            prop_assert_eq!(state.blocks.graph.node_hashes().len(), expected_blocks.len());

            for (hash, parent_hash) in expected_blocks {
                prop_assert_eq!(
                    state.blocks.graph.parent_hash_for_test(hash),
                    Some(parent_hash)
                );
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
                    Event::HeadObserved { hash, parent_hash , logs_bloom: bloom_matching_any(), number: block_number_for(hash) },
                    &retry_plans,
                );
            }

            let expected_blocks = expected_observed_ancestor_closure(&chain);
            prop_assert_eq!(state.blocks.graph.node_hashes().len(), expected_blocks.len());

            for (hash, parent_hash) in expected_blocks {
                prop_assert_eq!(
                    state.blocks.graph.parent_hash_for_test(hash),
                    Some(parent_hash)
                );
            }

            let last_observed_head = chain
                .observed_heads
                .last()
                .map(|head_index| hash_for_node(*head_index))
                .unwrap_or(finalized_hash);

            prop_assert_eq!(state.blocks.graph.observed_head_hash(), last_observed_head);
            prop_assert!(state.pending_requests.is_empty_for_test());
            prop_assert!(!state.blocks.graph.contains(finalized_hash));
            assert_state_invariants(&state);
        }

        // Generates unrelated pending requests around one missing-header request.
        // This ensures actionable not-found resets chain state while preserving unrelated pending work.
        #[test]
        fn actionable_block_header_not_found_removes_only_matching_request(
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
            plant_chain(&mut state, finalized_hash, &[known_hash]);
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
            // Decision (e): refuse-and-keep — the known block and head survive; only the
            // not-found request is removed.
            prop_assert!(next_state.blocks.graph.contains(known_hash));
            prop_assert_eq!(next_state.blocks.graph.observed_head_hash(), known_hash);
            prop_assert_eq!(next_state.blocks.graph.anchor_hash(), finalized_hash);
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
            plant_chain(&mut state, finalized_hash, &[known_hash]);
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
            prop_assert_eq!(next_state.blocks.graph.observed_head_hash(), known_hash);
            prop_assert_eq!(
                next_state.blocks.graph.parent_hash_for_test(known_hash),
                Some(finalized_hash)
            );
            prop_assert_eq!(next_state.blocks.graph.anchor_hash(), finalized_hash);
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
        // This proves reconstruction proceeds after a not-found dropped the backfill request.
        #[test]
        fn chain_reconstructs_after_not_found_and_reobservation(
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
                    number: block_number_for(first_head_hash),
                    logs_bloom: bloom_matching_any(),
                    hash: first_head_hash,
                    parent_hash: first_parent_hash,
                },
            );
            let request_id = assert_single_block_header_request_effect(&effects, first_parent_hash);
            let state = drain_block_log_effects(state, &effects);

            let (mut state, effects) =
                transition(ChainKey::Ethereum, state, Event::BlockHeaderNotFound { request_id });

            // Decision (e): the pending head survives the not-found; only the request is gone.
            prop_assert!(effects.is_empty());
            prop_assert!(state.blocks.graph.contains(first_head_hash));
            prop_assert_eq!(state.blocks.graph.observed_head_hash(), first_head_hash);
            prop_assert!(state.pending_requests.is_empty_for_test());

            for head_index in &chain.observed_heads {
                let hash = hash_for_node(*head_index);
                let parent_hash = hash_for_node(parent_index(&chain, *head_index));

                state = apply_event_and_drain_block_headers(
                    state,
                    &chain,
                    Event::HeadObserved { hash, parent_hash , logs_bloom: bloom_matching_any(), number: block_number_for(hash) },
                );
            }

            let expected_blocks = expected_observed_ancestor_closure(&chain);
            prop_assert_eq!(state.blocks.graph.node_hashes().len(), expected_blocks.len());

            for (hash, parent_hash) in expected_blocks {
                prop_assert_eq!(
                    state.blocks.graph.parent_hash_for_test(hash),
                    Some(parent_hash)
                );
            }

            let last_observed_head = chain
                .observed_heads
                .last()
                .map(|head_index| hash_for_node(*head_index))
                .unwrap_or(finalized_hash);

            prop_assert_eq!(state.blocks.graph.observed_head_hash(), last_observed_head);
            prop_assert!(state.pending_requests.is_empty_for_test());
            assert_state_invariants(&state);
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
            plant_chain(&mut state, finalized_hash, &[known_block_hash]);

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
            prop_assert_eq!(next_state.blocks.graph.observed_head_hash(), known_block_hash);
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
            // Decision (e): a BlockHeaderNotFound drops the backfill request WITHOUT re-emitting
            // (deliberate — the next head observation is the retry cadence), so the
            // every-missing-parent-is-pending invariant is deferred until the next event that
            // runs the scheduling chain.
            let mut backfill_deferred = false;

            for generated_event in generated_events {
                let previous_tick = state.tick;
                let is_tick = matches!(generated_event, GeneratedEvent::Tick);
                match &generated_event {
                    GeneratedEvent::BlockHeaderNotFound { .. } => backfill_deferred = true,
                    // A self-parent head is garbage input handled without the scheduling chain,
                    // so it does not clear a deferred backfill.
                    GeneratedEvent::HeadObserved {
                        hash_index,
                        parent_index,
                    } if hash_index != parent_index => backfill_deferred = false,
                    GeneratedEvent::BlockHeaderReceived { .. } => backfill_deferred = false,
                    _ => {}
                }
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
                if !backfill_deferred {
                    assert_missing_parents_for_known_blocks_are_pending(&next_state);
                }
                assert_active_requests_have_exactly_one_dispatch_tick(&next_state);

                state = next_state;
            }
        }
    }

    // --- anchor-height pool-data seeding (Blockers 1b/1c coverage) ---

    /// Extracts the single expected pool-data seed request, returning its id, target anchor, and
    /// named pools. Keeps the seeding tests strict about there being exactly one chunked request.
    fn single_pool_data_request_effect(
        effects: &[Effect],
    ) -> (RequestId<GetPoolData>, BlockHash, HashSet<PoolRef>) {
        let requests = effects
            .iter()
            .filter_map(|effect| match effect {
                Effect::Request(AnyIssuedRequest::PoolData(IssuedRequest {
                    request_id,
                    request_payload: GetPoolData { at, pools },
                })) => Some((*request_id, *at, pools.clone())),
                _ => None,
            })
            .collect::<Vec<_>>();

        assert_eq!(requests.len(), 1, "expected exactly one pool-data request effect");
        requests.into_iter().next().expect("one pool-data request")
    }

    fn verified_registry(candidates: &[ProtocolPoolKey]) -> TrustedPoolRegistry {
        let results = candidates
            .iter()
            .map(|candidate| (*candidate, Ok(pool_metadata(6, 7, UniswapV3Fee::Fee3000))))
            .collect::<HashMap<_, _>>();
        TrustedPoolRegistry::new().with_metadata_results(ChainKey::Ethereum, results)
    }

    #[test]
    fn finalized_pool_seed_requests_target_verified_uncovered_pools_at_the_anchor() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let mut state = empty_state_at(finalized_hash);
        state.pool_registry =
            verified_registry(&[pool_candidate_address(4), pool_candidate_address(5)]);

        let (_state, effects) =
            schedule_finalized_pool_seed_requests(ChainKey::Ethereum, state, vec![]);

        let (_id, at, pools) = single_pool_data_request_effect(&effects);
        assert_eq!(at, finalized_hash);
        assert_eq!(pools, HashSet::from([pool_address(4), pool_address(5)]));
    }

    #[test]
    fn finalized_pool_seed_requests_skip_pools_already_in_the_snapshot() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let mut state = empty_state_at(finalized_hash);
        state.pool_registry =
            verified_registry(&[pool_candidate_address(4), pool_candidate_address(5)]);
        state.blocks.finalized_snapshot = HashMap::from([(pool_address(4), pool_state(9))]);

        let (_state, effects) =
            schedule_finalized_pool_seed_requests(ChainKey::Ethereum, state, vec![]);

        let (_id, _at, pools) = single_pool_data_request_effect(&effects);
        assert_eq!(pools, HashSet::from([pool_address(5)]));
    }

    #[test]
    fn finalized_pool_seed_requests_skip_pools_already_in_flight() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let mut state = empty_state_at(finalized_hash);
        state.pool_registry =
            verified_registry(&[pool_candidate_address(4), pool_candidate_address(5)]);
        // pool 4 already covered by an in-flight seed request.
        let (pending, _) = state.pending_requests.with_new_request(
            GetPoolData {
                at: finalized_hash,
                pools: HashSet::from([pool_address(4)]),
            },
            tick(0),
        );
        state.pending_requests = pending;

        let (_state, effects) =
            schedule_finalized_pool_seed_requests(ChainKey::Ethereum, state, vec![]);

        let (_id, _at, pools) = single_pool_data_request_effect(&effects);
        assert_eq!(pools, HashSet::from([pool_address(5)]));
    }

    #[test]
    fn finalized_pool_seed_requests_are_capped_at_the_chunk_size() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let candidates = (0u8..(FINALIZED_POOL_SEED_CHUNK as u8 + 30))
            .map(pool_candidate_address)
            .collect::<Vec<_>>();
        let mut state = empty_state_at(finalized_hash);
        state.pool_registry = verified_registry(&candidates);

        let (_state, effects) =
            schedule_finalized_pool_seed_requests(ChainKey::Ethereum, state, vec![]);

        let (_id, at, pools) = single_pool_data_request_effect(&effects);
        assert_eq!(at, finalized_hash);
        assert_eq!(pools.len(), FINALIZED_POOL_SEED_CHUNK);
    }

    #[test]
    fn finalized_pool_seed_requests_are_deferred_while_block_backfill_is_pending() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let mut state = empty_state_at(finalized_hash);
        state.pool_registry = verified_registry(&[pool_candidate_address(4)]);
        // A header backfill in flight closes the idle gate.
        let (pending, _) = state.pending_requests.with_new_request(
            GetBlockHeader {
                block_hash: BlockHash::with_last_byte(9),
            },
            tick(0),
        );
        state.pending_requests = pending;

        let (_state, effects) =
            schedule_finalized_pool_seed_requests(ChainKey::Ethereum, state, vec![]);

        assert!(
            effects.is_empty(),
            "seeding must defer to latency-critical block backfill"
        );
    }

    #[test]
    fn finalized_pool_seed_requests_are_empty_when_every_verified_pool_is_covered() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let mut state = empty_state_at(finalized_hash);
        state.pool_registry = verified_registry(&[pool_candidate_address(4)]);
        state.blocks.finalized_snapshot = HashMap::from([(pool_address(4), pool_state(9))]);

        let (_state, effects) =
            schedule_finalized_pool_seed_requests(ChainKey::Ethereum, state, vec![]);

        assert!(effects.is_empty());
    }

    #[test]
    fn pool_data_received_seeds_the_finalized_snapshot() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let seeded = pool_state(9);
        let mut state = empty_state_at(finalized_hash);
        state.pool_registry = verified_registry(&[pool_candidate_address(4)]);

        let (state, effects) =
            schedule_finalized_pool_seed_requests(ChainKey::Ethereum, state, vec![]);
        let (request_id, _at, _pools) = single_pool_data_request_effect(&effects);

        let (state, _effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::PoolDataReceived {
                request_id,
                pools: HashMap::from([(pool_address(4), Ok(seeded.clone()))]),
            },
        );

        assert_eq!(
            state.finalized_pool_snapshots(),
            &HashMap::from([(pool_address(4), seeded)])
        );
    }

    #[test]
    fn with_finalized_pool_seeds_merges_ok_results_at_the_anchor() {
        let anchor = BlockHash::with_last_byte(1);
        let seeded = pool_state(9);

        let blocks = Blocks::new(anchor, HashMap::new()).with_finalized_pool_seeds(
            anchor,
            HashMap::from([(pool_address(4), Ok(seeded.clone()))]),
        );

        assert_eq!(
            blocks.finalized_snapshot,
            HashMap::from([(pool_address(4), seeded)])
        );
    }

    #[test]
    fn with_finalized_pool_seeds_skips_failed_reads() {
        let anchor = BlockHash::with_last_byte(1);

        let blocks = Blocks::new(anchor, HashMap::new()).with_finalized_pool_seeds(
            anchor,
            HashMap::from([(
                pool_address(4),
                Err(PoolDataFailure::CallFailed(PoolDataCall::Slot0)),
            )]),
        );

        assert!(blocks.finalized_snapshot.is_empty());
    }

    #[test]
    fn with_finalized_pool_seeds_drops_a_stale_at_response() {
        let anchor = BlockHash::with_last_byte(1);
        let stale = BlockHash::with_last_byte(2);

        let blocks = Blocks::new(anchor, HashMap::new()).with_finalized_pool_seeds(
            stale,
            HashMap::from([(pool_address(4), Ok(pool_state(9)))]),
        );

        assert!(
            blocks.finalized_snapshot.is_empty(),
            "a response for a superseded anchor must not be written"
        );
    }

    proptest! {
        /// The finalized base only ever gains states at the current anchor: an on-anchor response
        /// merges exactly its `Ok` reads over the base (failures skipped); an off-anchor response is
        /// dropped wholesale, whatever it carries.
        #[test]
        fn with_finalized_pool_seeds_only_writes_ok_reads_at_the_anchor(
            anchor_byte in 0u8..8,
            at_byte in 0u8..8,
            base_pools in prop::collection::vec((0u8..32, 0u8..32), 0..6),
            seed_pools in prop::collection::vec((0u8..32, any::<bool>(), 0u8..32), 0..6),
        ) {
            let anchor = BlockHash::with_last_byte(anchor_byte);
            let at = BlockHash::with_last_byte(at_byte);
            let base = base_pools
                .into_iter()
                .map(|(pool, state)| (pool_address(pool), pool_state(state)))
                .collect::<HashMap<_, _>>();
            let results = seed_pools
                .into_iter()
                .map(|(pool, ok, state)| {
                    let result = if ok {
                        Ok(pool_state(state))
                    } else {
                        Err(PoolDataFailure::CallFailed(PoolDataCall::Slot0))
                    };
                    (pool_address(pool), result)
                })
                .collect::<HashMap<_, _>>();

            let blocks = Blocks::new(anchor, base.clone())
                .with_finalized_pool_seeds(at, results.clone());

            if at == anchor {
                let mut expected = base;
                for (pool, result) in &results {
                    if let Ok(state) = result {
                        expected.insert(*pool, state.clone());
                    }
                }
                prop_assert_eq!(blocks.finalized_snapshot, expected);
            } else {
                prop_assert_eq!(blocks.finalized_snapshot, base);
            }
        }
    }
}
