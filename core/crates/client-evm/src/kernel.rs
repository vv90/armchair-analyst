use std::collections::{HashMap, HashSet};

use alloy::primitives::BlockHash;

use crate::{pending_requests::*, pool_registry::*, pool_state::*, tick::Tick};

enum PoolLogsStatus {
    Unknown,
    Resolved(HashSet<PoolCandidateAddress>),
}

struct BlockNode {
    parent_hash: BlockHash,
    pool_logs: PoolLogsStatus,
    pool_snapshots: HashMap<PoolAddress, PoolState>,
    pool_data_failures: HashMap<PoolAddress, PoolDataFailure>,
}

enum NewBlockError {
    SelfParentBlock(BlockHash, BlocksGraph),
    ExistingBlock(BlocksGraph),
    ConflictingBlockParent,
    CycleDetected,
}

struct BlocksGraph(HashMap<BlockHash, BlockNode>);

impl BlocksGraph {
    /// Builds an empty in-memory graph of recent, non-finalized blocks.
    fn new() -> BlocksGraph {
        BlocksGraph(HashMap::new())
    }

    /// Looks up a recent block node by hash without changing graph state.
    fn get(&self, hash: &BlockHash) -> Option<&BlockNode> {
        self.0.get(hash)
    }

    /// Adds a block header and reports the first missing parent on its path to finality.
    fn with_new_block(
        self,
        hash: BlockHash,
        parent_hash: BlockHash,
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

    /// Records fetched log-derived pool candidates for an existing recent block.
    fn with_pool_logs(
        self,
        block_hash: BlockHash,
        logs: HashSet<PoolCandidateAddress>,
    ) -> BlocksGraph {
        let BlocksGraph(mut blocks) = self;

        if let Some(block) = blocks.get_mut(&block_hash) {
            block.pool_logs = PoolLogsStatus::Resolved(logs);
        }

        BlocksGraph(blocks)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    /// Derives the trusted pool-log view for a block from stored candidates and the registry.
    fn trusted_pool_logs(
        &self,
        block_hash: BlockHash,
        registry: &TrustedPoolRegistry,
    ) -> Option<TrustedPoolLogs> {
        self.get(&block_hash).map(|block| match &block.pool_logs {
            PoolLogsStatus::Unknown => TrustedPoolLogs::Unknown,
            PoolLogsStatus::Resolved(candidates) => registry.trusted_pool_logs(candidates),
        })
    }

    /// Applies requested pool state results to the target block snapshot.
    fn with_pool_data(
        self,
        block_hash: BlockHash,
        requested_pools: HashSet<PoolAddress>,
        pool_results: HashMap<PoolAddress, PoolDataResult>,
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

    /// Finds present canonical blocks whose log requests are neither resolved nor pending.
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

            if matches!(block.pool_logs, PoolLogsStatus::Unknown)
                && !pending_log_hashes.contains(&current_hash)
            {
                hashes.push(current_hash);
            }

            current_hash = block.parent_hash;
        }

        hashes
    }

    /// Groups unresolved canonical pool candidates into metadata validation requests.
    fn unknown_present_canonical_pool_metadata_requests(
        &self,
        tip_hash: BlockHash,
        finalized_hash: BlockHash,
        registry: &TrustedPoolRegistry,
        pending_candidates: &HashSet<PoolCandidateAddress>,
    ) -> Vec<(BlockHash, HashSet<PoolCandidateAddress>)> {
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

            if let PoolLogsStatus::Resolved(candidates) = &block.pool_logs {
                let request_candidates = candidates
                    .iter()
                    .copied()
                    .filter(|candidate| {
                        !registry.is_known(*candidate)
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
}

pub struct FinalizedState {
    pub block_hash: BlockHash,
    pool_snapshots: HashMap<PoolAddress, PoolState>,
}

impl FinalizedState {
    /// Creates an empty finalized snapshot anchored at the provided block hash.
    pub fn empty_at(block_hash: BlockHash) -> FinalizedState {
        FinalizedState {
            block_hash,
            pool_snapshots: HashMap::new(),
        }
    }
}

pub struct State {
    blocks: BlocksGraph,
    canonical_tip: BlockHash,
    pending_requests: PendingRequests,
    finalized_state: FinalizedState,
    pool_registry: TrustedPoolRegistry,
    tick: Tick,
}

impl State {
    /// Initializes kernel state from the finalized snapshot with no pending work.
    pub fn init(finalized_state: FinalizedState) -> State {
        State {
            blocks: BlocksGraph::new(),
            canonical_tip: finalized_state.block_hash,
            pending_requests: PendingRequests::new(),
            finalized_state,
            pool_registry: TrustedPoolRegistry::new(),
            tick: Tick::initial(),
        }
    }

    /// Rebuilds volatile chain state while preserving immutable pool registry facts.
    fn reset(
        finalized_state: FinalizedState,
        tick: Tick,
        pool_registry: TrustedPoolRegistry,
    ) -> State {
        State {
            blocks: BlocksGraph::new(),
            canonical_tip: finalized_state.block_hash,
            pending_requests: PendingRequests::new(),
            finalized_state,
            pool_registry,
            tick,
        }
    }
}

pub enum Event {
    HeadObserved {
        hash: BlockHash,
        parent_hash: BlockHash,
    },
    BlockHeaderReceived {
        request_id: RequestId<GetBlockHeader>,
        hash: BlockHash,
        parent_hash: BlockHash,
    },
    BlockHeaderNotFound {
        request_id: RequestId<GetBlockHeader>,
    },
    BlockLogsReceived {
        request_id: RequestId<GetBlockLogs>,
        logs: HashSet<PoolCandidateAddress>,
    },
    PoolMetadataReceived {
        request_id: RequestId<GetPoolMetadata>,
        metadata: HashMap<PoolCandidateAddress, PoolMetadataResult>,
    },
    PoolDataReceived {
        request_id: RequestId<GetPoolData>,
        pools: HashMap<PoolAddress, PoolDataResult>,
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

/// Schedules log fetches for canonical blocks that are present but still unknown.
fn schedule_unknown_canonical_log_requests(
    state: State,
    mut effects: Vec<Effect>,
) -> (State, Vec<Effect>) {
    let State {
        blocks,
        canonical_tip,
        pending_requests,
        finalized_state,
        pool_registry,
        tick,
    } = state;

    let pending_log_hashes = pending_requests.pending_block_log_hashes();
    let block_hashes = blocks.unknown_present_canonical_log_hashes(
        canonical_tip,
        finalized_state.block_hash,
        &pending_log_hashes,
    );
    let mut pending_requests = pending_requests;

    for block_hash in block_hashes {
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

    (
        State {
            blocks,
            canonical_tip,
            pending_requests,
            finalized_state,
            pool_registry,
            tick,
        },
        effects,
    )
}

/// Schedules metadata validation for canonical log candidates not known by the registry.
fn schedule_unknown_canonical_pool_metadata_requests(
    state: State,
    mut effects: Vec<Effect>,
) -> (State, Vec<Effect>) {
    let State {
        blocks,
        canonical_tip,
        pending_requests,
        finalized_state,
        pool_registry,
        tick,
    } = state;

    let pending_candidates = pending_requests.pending_pool_metadata_candidates();
    let requests = blocks.unknown_present_canonical_pool_metadata_requests(
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
            tick,
        },
        effects,
    )
}

/// Schedules all currently actionable canonical follow-up requests.
fn schedule_unknown_canonical_requests(state: State, effects: Vec<Effect>) -> (State, Vec<Effect>) {
    let (state, effects) = schedule_unknown_canonical_log_requests(state, effects);
    schedule_unknown_canonical_pool_metadata_requests(state, effects)
}

/// Advances the pure client kernel state machine by one event and emits required effects.
pub fn transition(state: State, event: Event) -> (State, Vec<Effect>) {
    match event {
        Event::HeadObserved { hash, parent_hash } => {
            match state
                .blocks
                .with_new_block(hash, parent_hash, state.finalized_state.block_hash)
            {
                Ok((blocks, None)) => schedule_unknown_canonical_requests(
                    State {
                        blocks,
                        canonical_tip: hash,
                        ..state
                    },
                    vec![],
                ),
                Ok((blocks, Some(missing_hash))) => {
                    let request_payload = GetBlockHeader {
                        block_hash: missing_hash,
                    };
                    let (pending_requests, request_id) = state
                        .pending_requests
                        .with_new_request(request_payload.clone(), state.tick);

                    schedule_unknown_canonical_requests(
                        State {
                            blocks,
                            canonical_tip: hash,
                            pending_requests,
                            ..state
                        },
                        vec![Effect::Request(AnyIssuedRequest::BlockHeader(
                            IssuedRequest {
                                request_id,
                                request_payload,
                            },
                        ))],
                    )
                }
                Err(NewBlockError::SelfParentBlock(missing_hash, blocks)) => {
                    let request_payload = GetBlockHeader {
                        block_hash: missing_hash,
                    };
                    let (pending_requests, request_id) = state
                        .pending_requests
                        .with_new_request(request_payload.clone(), state.tick);
                    (
                        State {
                            blocks,
                            pending_requests,
                            ..state
                        },
                        vec![Effect::Request(AnyIssuedRequest::BlockHeader(
                            IssuedRequest {
                                request_id,
                                request_payload,
                            },
                        ))],
                    )
                }
                Err(NewBlockError::ExistingBlock(blocks)) => schedule_unknown_canonical_requests(
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
                        ..State::reset(state.finalized_state, state.tick, state.pool_registry)
                    },
                    vec![],
                ),
                Err(NewBlockError::CycleDetected) => (
                    State {
                        pending_requests: state.pending_requests,
                        ..State::reset(state.finalized_state, state.tick, state.pool_registry)
                    },
                    vec![],
                ),
            }
        }
        Event::BlockHeaderReceived {
            request_id,
            hash,
            parent_hash,
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

            let (new_state, effects, should_schedule_log_requests) = match state
                .blocks
                .with_new_block(hash, parent_hash, state.finalized_state.block_hash)
            {
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
                    let request_payload = GetBlockHeader {
                        block_hash: missing_hash,
                    };
                    let (pending_requests, request_id) =
                        pending_requests.with_new_request(request_payload.clone(), state.tick);

                    (
                        State {
                            blocks,
                            pending_requests,
                            ..state
                        },
                        vec![Effect::Request(AnyIssuedRequest::BlockHeader(
                            IssuedRequest {
                                request_id,
                                request_payload,
                            },
                        ))],
                        true,
                    )
                }
                Err(NewBlockError::SelfParentBlock(missing_hash, blocks)) => {
                    let request_payload = GetBlockHeader {
                        block_hash: missing_hash,
                    };
                    let (pending_requests, request_id) =
                        pending_requests.with_new_request(request_payload.clone(), state.tick);
                    (
                        State {
                            blocks,
                            pending_requests,
                            ..state
                        },
                        vec![Effect::Request(AnyIssuedRequest::BlockHeader(
                            IssuedRequest {
                                request_id,
                                request_payload,
                            },
                        ))],
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
                        ..State::reset(state.finalized_state, state.tick, state.pool_registry)
                    },
                    vec![],
                    false,
                ),
                Err(NewBlockError::CycleDetected) => (
                    State {
                        pending_requests,
                        ..State::reset(state.finalized_state, state.tick, state.pool_registry)
                    },
                    vec![],
                    false,
                ),
            };

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
                schedule_unknown_canonical_requests(new_state, effects)
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
                        ..State::reset(state.finalized_state, state.tick, state.pool_registry)
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
                    let blocks = state.blocks.with_pool_logs(block_hash, logs);

                    schedule_unknown_canonical_pool_metadata_requests(
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
                    let pool_registry = state.pool_registry.with_metadata_results(metadata);

                    schedule_unknown_canonical_pool_metadata_requests(
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
    }
}

/// Walks from a tip toward finality, returning the first missing block hash if any.
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

#[cfg(test)]
mod tests {
    use super::*;

    use alloy::primitives::{Address, U160, aliases::I24};

    use crate::tick::REQUEST_TTL_FOR_TEST as REQUEST_TTL;
    use proptest::prelude::*;

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

    /// Generates arbitrary rooted block graphs with observed heads.
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

    /// Generates rooted linear chains for reset-and-reobserve properties.
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

    /// Generates arbitrary kernel events that should preserve safety invariants.
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

    /// Generates finite sequences of arbitrary kernel events.
    fn generated_event_sequence_strategy() -> impl Strategy<Value = Vec<GeneratedEvent>> {
        prop::collection::vec(generated_event_strategy(), 1..128)
    }

    /// Generates retry plans for delayed header-response properties.
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

    /// Generates request payload shapes used by retry and not-found properties.
    fn generated_request_payload_strategy() -> impl Strategy<Value = GeneratedRequestPayload> {
        prop_oneof![
            any::<u8>()
                .prop_map(|block_index| GeneratedRequestPayload::GetBlockLogs { block_index }),
            any::<u8>()
                .prop_map(|block_index| GeneratedRequestPayload::GetBlockHeader { block_index }),
        ]
    }

    /// Checks the global structural invariants expected after every transition.
    fn assert_state_invariants(state: &State) {
        assert_finalized_block_not_in_recent_blocks(state);
        assert_no_self_parent_blocks(state);
        assert_canonical_tip_is_known_or_finalized(state);
        assert_parent_walks_do_not_cycle(state);
        assert_pool_snapshots_and_failures_do_not_overlap(state);
    }

    /// Ensures unknown canonical logs are always backed by pending log requests.
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

            if matches!(block.pool_logs, PoolLogsStatus::Unknown) {
                assert!(
                    pending_log_hashes.contains(&current_hash),
                    "canonical block with unknown logs must have a pending log request"
                );
            }

            current_hash = block.parent_hash;
        }
    }

    /// Ensures resolved canonical log candidates are either known or pending validation.
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

            if let PoolLogsStatus::Resolved(candidates) = &block.pool_logs {
                for candidate in candidates {
                    assert!(
                        state.pool_registry.is_known(*candidate)
                            || pending_candidates.contains(candidate),
                        "canonical resolved log candidate must be known or pending metadata validation"
                    );
                }
            }

            current_hash = block.parent_hash;
        }
    }

    /// Ensures every present block on the canonical path has resolved log status.
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

    /// Verifies every emitted request effect is recorded in pending state.
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
            }
        }
    }

    /// Ensures a known tip's first missing parent is represented by a pending header request.
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

    /// Ensures every known block with a missing ancestor has that ancestor pending.
    fn assert_missing_parents_for_known_blocks_are_pending(state: &State) {
        for block_hash in state.blocks.0.keys() {
            assert_missing_parent_is_pending(state, *block_hash);
        }
    }

    /// Ensures finalized state is not duplicated in the recent block graph.
    fn assert_finalized_block_not_in_recent_blocks(state: &State) {
        assert!(
            !state
                .blocks
                .0
                .contains_key(&state.finalized_state.block_hash),
            "finalized block must not be present in recent blocks"
        );
    }

    /// Ensures no recent block directly points to itself.
    fn assert_no_self_parent_blocks(state: &State) {
        for (hash, block) in &state.blocks.0 {
            assert_ne!(
                hash, &block.parent_hash,
                "block must not reference itself as parent"
            );
        }
    }

    /// Ensures the canonical tip is either finalized or present in recent blocks.
    fn assert_canonical_tip_is_known_or_finalized(state: &State) {
        assert!(
            state.canonical_tip == state.finalized_state.block_hash
                || state.blocks.0.contains_key(&state.canonical_tip),
            "canonical tip must be finalized or present in recent blocks"
        );
    }

    /// Ensures no parent walk through recent blocks cycles.
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

    /// Ensures successful and failed pool snapshots do not coexist for the same block/pool.
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

    /// Maps compact generated node indexes into deterministic block hashes.
    fn hash_for_node(node_index: usize) -> BlockHash {
        BlockHash::with_last_byte((node_index + 1) as u8)
    }

    /// Converts generated events into concrete kernel events.
    fn event_from_generated(generated_event: GeneratedEvent) -> Event {
        match generated_event {
            GeneratedEvent::HeadObserved {
                hash_index,
                parent_index,
            } => Event::HeadObserved {
                hash: hash_for_node(hash_index),
                parent_hash: hash_for_node(parent_index),
            },
            GeneratedEvent::BlockHeaderReceived {
                request_id,
                hash_index,
                parent_index,
            } => Event::BlockHeaderReceived {
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

    /// Converts generated request descriptions into concrete expected request payloads.
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

    /// Builds test ticks from raw integer values.
    fn tick(value: u64) -> Tick {
        Tick::from_raw_for_test(value)
    }

    /// Wraps a header request id in a typed request-failed event.
    fn request_failed_for_header(request_id: RequestId<GetBlockHeader>) -> Event {
        Event::RequestFailed {
            request_id: AnyRequestId::BlockHeader(request_id),
        }
    }

    // Verifies finalized-state initialization keeps only the finalized hash.
    #[test]
    fn finalized_state_empty_at_stores_hash_with_empty_pool_snapshots() {
        let finalized_hash = BlockHash::with_last_byte(1);

        let finalized_state = FinalizedState::empty_at(finalized_hash);

        assert_eq!(finalized_state.block_hash, finalized_hash);
        assert!(finalized_state.pool_snapshots.is_empty());
    }

    // Verifies state initialization starts with empty volatile tracking.
    #[test]
    fn state_init_from_finalized_state_starts_with_empty_tracking() {
        let finalized_hash = BlockHash::with_last_byte(1);

        let state = State::init(FinalizedState::empty_at(finalized_hash));

        assert_empty_initial_state_at(&state, finalized_hash);
        assert!(state.blocks.0.is_empty());
        assert!(state.finalized_state.pool_snapshots.is_empty());
        assert_eq!(state.tick.raw_for_test(), Tick::initial().raw_for_test());
    }

    /// Inserts an expected request payload into pending state and returns its typed id.
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

    /// Checks whether a pending request id still contains the expected request payload.
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

    /// Extracts the raw numeric id from any typed request id.
    fn any_request_id_raw(request_id: AnyRequestId) -> u64 {
        match request_id {
            AnyRequestId::BlockHeader(request_id) => request_id.raw_for_test(),
            AnyRequestId::BlockLogs(request_id) => request_id.raw_for_test(),
            AnyRequestId::PoolData(request_id) => request_id.raw_for_test(),
            AnyRequestId::PoolMetadata(request_id) => request_id.raw_for_test(),
        }
    }

    /// Reads a generated node's parent index, defaulting missing indexes to finalized.
    fn parent_index(chain: &GeneratedChain, node_index: usize) -> usize {
        chain.parents.get(node_index).copied().unwrap_or_default()
    }

    /// Finds the generated node index represented by a concrete block hash.
    fn node_index_for_hash(chain: &GeneratedChain, hash: BlockHash) -> Option<usize> {
        (0..chain.parents.len()).find(|node_index| hash_for_node(*node_index) == hash)
    }

    /// Computes all observed heads and ancestors expected to be reconstructed.
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

    /// Applies one event and drains all header/log effects with successful generated responses.
    fn apply_event_and_drain_block_headers(
        mut state: State,
        chain: &GeneratedChain,
        event: Event,
    ) -> State {
        let (next_state, effects) = transition(state, event);
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
                        state,
                        Event::BlockHeaderReceived {
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
                        state,
                        Event::BlockLogsReceived {
                            request_id,
                            logs: HashSet::new(),
                        },
                    );

                    state = next_state;
                    assert_state_invariants(&state);
                    pending_effects.extend(effects);
                }
                Effect::Request(AnyIssuedRequest::PoolData(_)) => {}
                Effect::Request(AnyIssuedRequest::PoolMetadata(_)) => {}
            }
        }

        state
    }

    /// Applies one event while exercising generated failure and expiration retry plans.
    fn apply_event_and_drain_block_headers_with_retries(
        mut state: State,
        chain: &GeneratedChain,
        event: Event,
        retry_plans: &[Vec<GeneratedRetry>],
    ) -> State {
        let (next_state, effects) = transition(state, event);
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
                            GeneratedRetry::Failure => {
                                transition(state, request_failed_for_header(current_request_id))
                            }
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
                        state,
                        Event::BlockHeaderReceived {
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
                        state,
                        Event::BlockLogsReceived {
                            request_id,
                            logs: HashSet::new(),
                        },
                    );

                    state = next_state;
                    assert_state_invariants(&state);
                    assert_effects_are_well_formed(&state, &effects);
                    pending_effects.extend(effects);
                }
                Effect::Request(AnyIssuedRequest::PoolData(_)) => {}
                Effect::Request(AnyIssuedRequest::PoolMetadata(_)) => {}
            }
        }

        state
    }

    /// Builds a recent block fixture with unknown logs and empty pool state.
    fn block_with_parent(parent_hash: BlockHash) -> BlockNode {
        BlockNode {
            parent_hash,
            pool_logs: PoolLogsStatus::Unknown,
            pool_snapshots: HashMap::new(),
            pool_data_failures: HashMap::new(),
        }
    }

    /// Builds an initialized test state at a finalized hash.
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
            tick: tick(0),
        }
    }

    /// Checks that volatile chain state has reset back to a finalized anchor.
    fn assert_chain_reset_at(state: &State, finalized_hash: BlockHash) {
        assert!(state.blocks.0.is_empty());
        assert_eq!(state.canonical_tip, finalized_hash);
        assert_eq!(state.finalized_state.block_hash, finalized_hash);
    }

    /// Checks the empty state shape expected immediately after initialization.
    fn assert_empty_initial_state_at(state: &State, finalized_hash: BlockHash) {
        assert_chain_reset_at(state, finalized_hash);
        assert!(state.pending_requests.is_empty_for_test());
    }

    /// Checks a single block fixture is present with unknown logs.
    fn assert_single_unknown_block(state: &State, hash: BlockHash, parent_hash: BlockHash) {
        let block = state
            .blocks
            .get(&hash)
            .expect("expected block to be present");

        assert_eq!(block.parent_hash, parent_hash);
        assert!(matches!(block.pool_logs, PoolLogsStatus::Unknown));
        assert!(block.pool_snapshots.is_empty());
    }

    /// Checks a single block fixture is present with the expected parent.
    fn assert_single_block_with_parent(state: &State, hash: BlockHash, parent_hash: BlockHash) {
        let block = state
            .blocks
            .get(&hash)
            .expect("expected block to be present");

        assert_eq!(block.parent_hash, parent_hash);
        assert!(block.pool_snapshots.is_empty());
    }

    /// Checks a block contains exactly the expected resolved pool candidate logs.
    fn assert_resolved_pool_logs(
        state: &State,
        hash: BlockHash,
        parent_hash: BlockHash,
        expected_logs: &HashSet<PoolCandidateAddress>,
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

    /// Checks resolved candidate logs by direct set equality.
    fn assert_resolved_candidate_logs(
        state: &State,
        hash: BlockHash,
        parent_hash: BlockHash,
        expected_logs: &HashSet<PoolCandidateAddress>,
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

    /// Checks the trusted-pool projection for a block resolves to the expected pools.
    fn assert_trusted_pool_logs_resolved(
        state: &State,
        hash: BlockHash,
        expected_pools: HashSet<PoolAddress>,
    ) {
        assert_eq!(
            state
                .blocks
                .trusted_pool_logs(hash, &state.pool_registry)
                .expect("block must be present"),
            TrustedPoolLogs::Resolved(expected_pools)
        );
    }

    /// Extracts the only header request effect for a specific block hash.
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

    /// Extracts the only block-log request effect for a specific block hash.
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

    /// Extracts the only pool-data request effect for a specific block and pool set.
    fn assert_single_pool_data_request_effect(
        effects: &[Effect],
        at: BlockHash,
        pools: &HashSet<PoolAddress>,
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

    /// Extracts the only pool-metadata request effect for a specific block and candidate set.
    fn assert_single_pool_metadata_request_effect(
        effects: &[Effect],
        at: BlockHash,
        candidates: &HashSet<PoolCandidateAddress>,
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

    /// Extracts the only request effect matching an expected generic request payload.
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

    /// Builds deterministic test pool addresses from a byte.
    fn pool_address(last_byte: u8) -> PoolAddress {
        PoolAddress(Address::with_last_byte(last_byte))
    }

    /// Builds deterministic test pool candidate addresses from a byte.
    fn pool_candidate_address(last_byte: u8) -> PoolCandidateAddress {
        PoolCandidateAddress(Address::with_last_byte(last_byte))
    }

    /// Builds deterministic immutable pool metadata for tests.
    fn pool_metadata(token0: u8, token1: u8, fee: UniswapV3Fee) -> PoolMetadata {
        PoolMetadata {
            token0: Address::with_last_byte(token0),
            token1: Address::with_last_byte(token1),
            fee,
        }
    }

    /// Builds deterministic mutable pool state for tests.
    fn pool_state(last_byte: u8) -> PoolState {
        PoolState {
            sqrt_price_x96: U160::from(u64::from(last_byte) + 1),
            tick: I24::try_from(i32::from(last_byte)).expect("test tick must fit int24"),
            liquidity: u128::from(last_byte) + 10,
        }
    }

    /// Alternates deterministic pool-data successes and failures by byte parity.
    fn pool_data_result_for_byte(last_byte: u8) -> PoolDataResult {
        if last_byte % 2 == 0 {
            Ok(pool_state(last_byte))
        } else {
            Err(PoolDataFailure::CallFailed(PoolDataCall::Slot0))
        }
    }

    /// Checks that a block stores the expected pool snapshot.
    fn assert_pool_snapshot(
        state: &State,
        block_hash: BlockHash,
        pool: PoolAddress,
        expected: &PoolState,
    ) {
        let block = state
            .blocks
            .get(&block_hash)
            .expect("expected block to be present");

        assert_eq!(block.pool_snapshots.get(&pool), Some(expected));
    }

    /// Checks that a block does not store a snapshot for a pool.
    fn assert_no_pool_snapshot(state: &State, block_hash: BlockHash, pool: PoolAddress) {
        let block = state
            .blocks
            .get(&block_hash)
            .expect("expected block to be present");

        assert!(!block.pool_snapshots.contains_key(&pool));
    }

    /// Checks that a block stores the expected pool-data failure.
    fn assert_pool_failure(
        state: &State,
        block_hash: BlockHash,
        pool: PoolAddress,
        expected: &PoolDataFailure,
    ) {
        let block = state
            .blocks
            .get(&block_hash)
            .expect("expected block to be present");

        assert_eq!(block.pool_data_failures.get(&pool), Some(expected));
    }

    /// Checks that a block does not store a failure for a pool.
    fn assert_no_pool_failure(state: &State, block_hash: BlockHash, pool: PoolAddress) {
        let block = state
            .blocks
            .get(&block_hash)
            .expect("expected block to be present");

        assert!(!block.pool_data_failures.contains_key(&pool));
    }

    /// Advances the kernel by repeated tick events and accumulates emitted effects.
    fn advance_ticks(mut state: State, count: u64) -> (State, Vec<Effect>) {
        let mut effects = Vec::new();

        for _ in 0..count {
            let (next_state, tick_effects) = transition(state, Event::Tick);
            state = next_state;
            effects.extend(tick_effects);
        }

        (state, effects)
    }

    /// Responds to all block-log request effects with empty log sets.
    fn drain_block_log_effects(mut state: State, effects: &[Effect]) -> State {
        for effect in effects {
            let Effect::Request(AnyIssuedRequest::BlockLogs(IssuedRequest { request_id, .. })) =
                effect
            else {
                continue;
            };

            let (next_state, effects) = transition(
                state,
                Event::BlockLogsReceived {
                    request_id: *request_id,
                    logs: HashSet::new(),
                },
            );

            state = next_state;
            assert!(effects.is_empty());
            assert_state_invariants(&state);
        }

        state
    }

    /// Extracts block hashes from effects that are expected to all be header requests.
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

    /// Extracts all block hashes requested by header effects.
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

    /// Extracts all block hashes requested by block-log effects.
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

    /// Checks that emitted header and log requests target exactly the expected hashes.
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

    /// Returns pending header request hashes for tests.
    fn pending_header_hashes(state: &State) -> HashSet<BlockHash> {
        state.pending_requests.pending_header_hashes_for_test()
    }

    /// Reads the dispatch tick for an active header request.
    fn active_request_dispatch_tick(
        pending_requests: &PendingRequests,
        request_id: RequestId<GetBlockHeader>,
    ) -> Tick {
        pending_requests
            .header_dispatch_tick_for_test(&request_id)
            .expect("active request must have a dispatch tick")
    }

    /// Ensures request bookkeeping has one dispatch tick per active request.
    fn assert_active_requests_have_exactly_one_dispatch_tick(state: &State) {
        assert_eq!(
            state.pending_requests.dispatch_ticks_for_test().len(),
            state.pending_requests.len_for_test(),
            "active request must have exactly one dispatch tick"
        );
    }

    /// Ensures retry expiration cleanup leaves no active request already expired.
    fn assert_no_active_request_is_expired(state: &State) {
        for dispatch_tick in state.pending_requests.dispatch_ticks_for_test() {
            assert!(
                !state.tick.is_expired_since(dispatch_tick),
                "active request must not remain expired after a tick"
            );
        }
    }

    // Verifies the invariant guard rejects finalized-block duplication.
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

    // Verifies the invariant guard rejects direct self-parent blocks.
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

    // Verifies the invariant guard rejects unknown canonical tips.
    #[test]
    #[should_panic(expected = "canonical tip must be finalized or present in recent blocks")]
    fn state_invariants_reject_unknown_canonical_tip() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let mut state = empty_state_at(finalized_hash);

        state.canonical_tip = BlockHash::with_last_byte(2);

        assert_state_invariants(&state);
    }

    // Verifies the invariant guard rejects parent-link cycles.
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

    // Verifies self-parent head observations fall back to fetching the header.
    #[test]
    fn head_observed_with_self_parent_fetches_header_without_changing_state() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let head_hash = BlockHash::with_last_byte(2);
        let state = empty_state_at(finalized_hash);

        let (next_state, effects) = transition(
            state,
            Event::HeadObserved {
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

    // Verifies conflicting parent observations reset volatile chain state.
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
            state,
            Event::HeadObserved {
                hash: head_hash,
                parent_hash: conflicting_parent_hash,
            },
        );

        assert_chain_reset_at(&next_state, finalized_hash);
        assert!(next_state.pending_requests.is_empty_for_test());
        assert!(effects.is_empty());
        assert_state_invariants(&next_state);
    }

    // Verifies chain resets preserve immutable pool registry facts.
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

        state.pool_registry = TrustedPoolRegistry::new().with_metadata_results(HashMap::from([
            (verified_candidate, Ok(metadata.clone())),
            (
                rejected_candidate,
                Err(PoolMetadataFailure::FactoryReturnedZero),
            ),
        ]));
        state.canonical_tip = head_hash;
        state
            .blocks
            .0
            .insert(head_hash, block_with_parent(original_parent_hash));

        let (next_state, effects) = transition(
            state,
            Event::HeadObserved {
                hash: head_hash,
                parent_hash: conflicting_parent_hash,
            },
        );

        assert_chain_reset_at(&next_state, finalized_hash);
        assert!(effects.is_empty());
        assert_eq!(
            next_state
                .pool_registry
                .verified_metadata(PoolAddress(verified_candidate.0)),
            Some(&metadata)
        );
        assert!(next_state.pool_registry.is_rejected(rejected_candidate));
        assert_state_invariants(&next_state);
    }

    // Verifies duplicate matching head observations only schedule missing work.
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
            state,
            Event::HeadObserved {
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

    // Verifies newly introduced cycles reset volatile chain state.
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
            state,
            Event::HeadObserved {
                hash: cycle_hash,
                parent_hash: second_hash,
            },
        );

        assert_chain_reset_at(&next_state, finalized_hash);
        assert!(next_state.pending_requests.is_empty_for_test());
        assert!(effects.is_empty());
        assert_state_invariants(&next_state);
    }

    // Verifies finalized hashes are never inserted as recent blocks.
    #[test]
    fn head_observed_with_finalized_hash_does_not_insert_finalized_block() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let parent_hash = BlockHash::with_last_byte(2);
        let state = empty_state_at(finalized_hash);

        let (next_state, effects) = transition(
            state,
            Event::HeadObserved {
                hash: finalized_hash,
                parent_hash,
            },
        );

        assert_empty_initial_state_at(&next_state, finalized_hash);
        assert!(effects.is_empty());
        assert_state_invariants(&next_state);
    }

    // Verifies a connected new head schedules log fetching.
    #[test]
    fn connected_head_observed_requests_logs_for_head() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let head_hash = BlockHash::with_last_byte(2);
        let state = empty_state_at(finalized_hash);

        let (next_state, effects) = transition(
            state,
            Event::HeadObserved {
                hash: head_hash,
                parent_hash: finalized_hash,
            },
        );

        assert_request_hashes(&effects, HashSet::new(), HashSet::from([head_hash]));
        assert_effects_are_well_formed(&next_state, &effects);
        assert_state_invariants(&next_state);
    }

    // Verifies a disconnected new head schedules its logs and missing parent header.
    #[test]
    fn disconnected_head_observed_requests_head_logs_and_missing_parent_header() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let state = empty_state_at(finalized_hash);

        let (next_state, effects) = transition(
            state,
            Event::HeadObserved {
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

    // Verifies receiving a missing parent schedules logs for that parent.
    #[test]
    fn missing_parent_header_received_requests_logs_for_parent() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            state,
            Event::HeadObserved {
                hash: head_hash,
                parent_hash: missing_parent_hash,
            },
        );
        let header_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);

        let (next_state, effects) = transition(
            state,
            Event::BlockHeaderReceived {
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

    // Verifies all present canonical blocks with unknown logs get scheduled.
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
            state,
            Event::HeadObserved {
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

    // Verifies duplicate header responses do not duplicate pending log requests.
    #[test]
    fn duplicate_header_response_does_not_duplicate_pending_log_requests() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            state,
            Event::HeadObserved {
                hash: head_hash,
                parent_hash,
            },
        );
        let header_request_id = assert_single_block_header_request_effect(&effects, parent_hash);

        let (state, effects) = transition(
            state,
            Event::BlockHeaderReceived {
                request_id: header_request_id,
                hash: parent_hash,
                parent_hash: finalized_hash,
            },
        );
        assert_request_hashes(&effects, HashSet::new(), HashSet::from([parent_hash]));

        let (next_state, effects) = transition(
            state,
            Event::BlockHeaderReceived {
                request_id: header_request_id,
                hash: parent_hash,
                parent_hash: finalized_hash,
            },
        );

        assert!(effects.is_empty());
        assert_state_invariants(&next_state);
    }

    // Verifies pending log requests suppress duplicate scheduling.
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
            state,
            Event::HeadObserved {
                hash: head_hash,
                parent_hash: finalized_hash,
            },
        );

        assert!(effects.is_empty());
        assert_state_invariants(&next_state);
    }

    // Verifies resolved log status suppresses duplicate scheduling.
    #[test]
    fn resolved_log_status_suppresses_duplicate_log_request() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let head_hash = BlockHash::with_last_byte(2);
        let mut state = empty_state_at(finalized_hash);
        state.canonical_tip = head_hash;
        state.blocks.0.insert(
            head_hash,
            BlockNode {
                parent_hash: finalized_hash,
                pool_logs: PoolLogsStatus::Resolved(HashSet::new()),
                pool_snapshots: HashMap::new(),
                pool_data_failures: HashMap::new(),
            },
        );

        let (next_state, effects) = transition(
            state,
            Event::HeadObserved {
                hash: head_hash,
                parent_hash: finalized_hash,
            },
        );

        assert!(effects.is_empty());
        assert_state_invariants(&next_state);
    }

    // Verifies matching log responses mark logs and clear pending state.
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
            state,
            Event::BlockLogsReceived {
                request_id,
                logs: logs.clone(),
            },
        );

        assert_single_pool_metadata_request_effect(&effects, block_hash, &logs);
        assert!(!next_state.pending_requests.contains(&request_id));
        assert_resolved_pool_logs(&next_state, block_hash, finalized_hash, &logs);
        assert_state_invariants(&next_state);
    }

    // Verifies empty log responses still resolve the block log status.
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
            state,
            Event::BlockLogsReceived {
                request_id,
                logs: logs.clone(),
            },
        );

        assert!(effects.is_empty());
        assert!(!next_state.pending_requests.contains(&request_id));
        assert_resolved_pool_logs(&next_state, block_hash, finalized_hash, &logs);
        assert_state_invariants(&next_state);
    }

    // Verifies unsolicited log responses do not mutate block state.
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
            state,
            Event::BlockLogsReceived {
                request_id: RequestId::from_raw_for_test(99),
                logs,
            },
        );

        assert!(effects.is_empty());
        assert!(next_state.pending_requests.is_empty_for_test());
        assert_single_unknown_block(&next_state, block_hash, finalized_hash);
        assert_state_invariants(&next_state);
    }

    // Verifies stale log responses for missing blocks consume only the pending request.
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

        let (next_state, effects) =
            transition(state, Event::BlockLogsReceived { request_id, logs });

        assert!(effects.is_empty());
        assert_empty_initial_state_at(&next_state, finalized_hash);
        assert!(!next_state.blocks.0.contains_key(&missing_block_hash));
        assert_state_invariants(&next_state);
    }

    // Verifies log candidates are stored and scheduled for metadata validation.
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
            state,
            Event::BlockLogsReceived {
                request_id,
                logs: logs.clone(),
            },
        );

        assert!(!next_state.pending_requests.contains(&request_id));
        assert_resolved_candidate_logs(&next_state, block_hash, finalized_hash, &logs);
        let metadata_request_id =
            assert_single_pool_metadata_request_effect(&effects, block_hash, &logs);
        assert!(next_state.pending_requests.contains(&metadata_request_id));
        assert_state_invariants(&next_state);
    }

    // Verifies already verified pool candidates skip metadata requests.
    #[test]
    fn known_verified_candidates_do_not_request_metadata() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let candidate = pool_candidate_address(3);
        let logs = HashSet::from([candidate]);
        let mut state = empty_state_at(finalized_hash);

        state.pool_registry = TrustedPoolRegistry::new().with_metadata_results(HashMap::from([(
            candidate,
            Ok(pool_metadata(1, 2, UniswapV3Fee::Fee3000)),
        )]));
        state.canonical_tip = block_hash;
        state
            .blocks
            .0
            .insert(block_hash, block_with_parent(finalized_hash));
        let (pending_requests, request_id) = state
            .pending_requests
            .with_new_request(GetBlockLogs { block_hash }, state.tick);
        state.pending_requests = pending_requests;

        let (next_state, effects) =
            transition(state, Event::BlockLogsReceived { request_id, logs });

        assert!(effects.is_empty());
        assert_trusted_pool_logs_resolved(
            &next_state,
            block_hash,
            HashSet::from([PoolAddress(candidate.0)]),
        );
        assert_state_invariants(&next_state);
    }

    // Verifies already rejected pool candidates skip metadata requests.
    #[test]
    fn known_rejected_candidates_do_not_request_metadata() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let candidate = pool_candidate_address(3);
        let logs = HashSet::from([candidate]);
        let mut state = empty_state_at(finalized_hash);

        state.pool_registry = TrustedPoolRegistry::new().with_metadata_results(HashMap::from([(
            candidate,
            Err(PoolMetadataFailure::FactoryReturnedZero),
        )]));
        state.canonical_tip = block_hash;
        state
            .blocks
            .0
            .insert(block_hash, block_with_parent(finalized_hash));
        let (pending_requests, request_id) = state
            .pending_requests
            .with_new_request(GetBlockLogs { block_hash }, state.tick);
        state.pending_requests = pending_requests;

        let (next_state, effects) =
            transition(state, Event::BlockLogsReceived { request_id, logs });

        assert!(effects.is_empty());
        assert_eq!(
            next_state
                .blocks
                .trusted_pool_logs(block_hash, &next_state.pool_registry),
            Some(TrustedPoolLogs::Resolved(HashSet::new()))
        );
        assert_state_invariants(&next_state);
    }

    // Verifies one pending metadata request covers duplicate unknown candidates.
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
            state,
            Event::BlockLogsReceived {
                request_id: parent_request_id,
                logs: HashSet::from([candidate]),
            },
        );
        let (next_state, second_effects) = transition(
            state,
            Event::BlockLogsReceived {
                request_id: head_request_id,
                logs: HashSet::from([candidate]),
            },
        );

        assert_eq!(first_effects.len(), 1);
        assert!(second_effects.is_empty());
        assert_state_invariants(&next_state);
    }

    // Verifies partial metadata responses reschedule omitted requested candidates.
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
            next_state.pool_registry.verified_pool(first_candidate),
            Some(PoolAddress(first_candidate.0))
        );
        assert_eq!(
            next_state.pool_registry.verified_pool(second_candidate),
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

    // Verifies metadata responses update the registry and trusted-log projection.
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
            state,
            Event::PoolMetadataReceived {
                request_id,
                metadata: HashMap::from([(
                    candidate,
                    Ok(pool_metadata(1, 2, UniswapV3Fee::Fee500)),
                )]),
            },
        );

        assert!(effects.is_empty());
        assert!(!next_state.pending_requests.contains(&request_id));
        assert_eq!(
            next_state
                .pool_registry
                .verified_metadata(PoolAddress(candidate.0)),
            Some(&pool_metadata(1, 2, UniswapV3Fee::Fee500))
        );
        assert_trusted_pool_logs_resolved(
            &next_state,
            block_hash,
            HashSet::from([PoolAddress(candidate.0)]),
        );
        assert_state_invariants(&next_state);
    }

    // Verifies unrequested metadata response entries are ignored.
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
                .verified_metadata(PoolAddress(unrequested.0)),
            None
        );
        assert_eq!(next_state.pool_registry.verified_pool(unrequested), None);
        assert!(!next_state.pool_registry.is_rejected(unrequested));
        assert_state_invariants(&next_state);
    }

    // Verifies rejected candidates are excluded from trusted-log projection.
    #[test]
    fn rejected_candidates_never_appear_in_derived_trusted_pool_logs() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let verified = pool_candidate_address(3);
        let rejected = pool_candidate_address(4);
        let mut state = empty_state_at(finalized_hash);

        state.pool_registry = TrustedPoolRegistry::new().with_metadata_results(HashMap::from([
            (verified, Ok(pool_metadata(1, 2, UniswapV3Fee::Fee3000))),
            (
                rejected,
                Err(PoolMetadataFailure::FactoryMismatch {
                    returned: Address::with_last_byte(9),
                }),
            ),
        ]));
        state.canonical_tip = block_hash;
        state.blocks.0.insert(
            block_hash,
            BlockNode {
                parent_hash: finalized_hash,
                pool_logs: PoolLogsStatus::Resolved(HashSet::from([verified, rejected])),
                pool_snapshots: HashMap::new(),
                pool_data_failures: HashMap::new(),
            },
        );

        assert_trusted_pool_logs_resolved(
            &state,
            block_hash,
            HashSet::from([PoolAddress(verified.0)]),
        );
        assert_state_invariants(&state);
    }

    // Verifies unknown block logs derive an unknown trusted-log status.
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
                .trusted_pool_logs(block_hash, &state.pool_registry),
            Some(TrustedPoolLogs::Unknown)
        );
        assert_state_invariants(&state);
    }

    // Verifies matching pool-data responses store snapshots and clear pending state.
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

    // Verifies requested pool snapshots are not limited to pools logged in that block.
    #[test]
    fn pool_data_received_applies_requested_snapshots_not_present_in_block_logs() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let block_hash = BlockHash::with_last_byte(2);
        let logged_pool = pool_address(3);
        let logged_candidate = PoolCandidateAddress(logged_pool.0);
        let requested_pool = pool_address(4);
        let requested_state = pool_state(5);
        let mut state = empty_state_at(finalized_hash);

        state.canonical_tip = block_hash;
        state.blocks.0.insert(
            block_hash,
            BlockNode {
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

    // Verifies unrequested pool-data response entries are ignored.
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

    // Verifies requested pool-data failures are recorded without immediate retry effects.
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

    // Verifies a later successful pool-data result replaces a previous failure.
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

    // Verifies a later pool-data failure does not overwrite an existing snapshot.
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

    // Verifies unsolicited pool-data responses do not mutate block state.
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

    // Verifies stale pool-data responses for missing blocks consume only the request.
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

    // Verifies pool-data request failures retry the original request payload.
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

    // Verifies pool-metadata request failures retry the original request payload.
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

    // Verifies expired pool-metadata requests retry the original request payload.
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

    // Verifies duplicate failure for an old pool-metadata request id is ignored.
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
            state,
            Event::RequestFailed {
                request_id: AnyRequestId::PoolMetadata(request_id),
            },
        );
        let retry_request_id =
            assert_single_pool_metadata_request_effect(&effects, block_hash, &candidates);

        let (next_state, effects) = transition(
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

    // Verifies stale log responses after reset do not resurrect removed blocks.
    #[test]
    fn stale_block_logs_response_after_reset_does_not_resurrect_removed_block() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let head_hash = BlockHash::with_last_byte(2);
        let conflicting_parent_hash = BlockHash::with_last_byte(3);
        let logs = HashSet::from([pool_candidate_address(4)]);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            state,
            Event::HeadObserved {
                hash: head_hash,
                parent_hash: finalized_hash,
            },
        );
        let log_request_id = assert_single_block_log_request_effect(&effects, head_hash);

        let (state, effects) = transition(
            state,
            Event::HeadObserved {
                hash: head_hash,
                parent_hash: conflicting_parent_hash,
            },
        );
        assert!(effects.is_empty());
        assert_chain_reset_at(&state, finalized_hash);
        assert!(state.pending_requests.contains(&log_request_id));

        let (next_state, effects) = transition(
            state,
            Event::BlockLogsReceived {
                request_id: log_request_id,
                logs,
            },
        );

        assert!(effects.is_empty());
        assert_empty_initial_state_at(&next_state, finalized_hash);
        assert!(!next_state.blocks.0.contains_key(&head_hash));
        assert_state_invariants(&next_state);
    }

    // Verifies a matching header response clears its pending request.
    #[test]
    fn block_header_received_for_matching_request_removes_pending_request() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            state,
            Event::HeadObserved {
                hash: head_hash,
                parent_hash: missing_parent_hash,
            },
        );
        let request_id = assert_single_block_header_request_effect(&effects, missing_parent_hash);

        let (next_state, effects) = transition(
            state,
            Event::BlockHeaderReceived {
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

    // Verifies a matching header-not-found resets chain state and removes that request.
    #[test]
    fn block_header_not_found_for_matching_request_resets_chain() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let unrelated_hash = BlockHash::with_last_byte(4);
        let mut state = empty_state_at(finalized_hash);
        state.tick = tick(7);

        let (mut state, effects) = transition(
            state,
            Event::HeadObserved {
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

    // Verifies an unknown header-not-found response is ignored.
    #[test]
    fn block_header_not_found_for_unknown_request_is_noop() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            state,
            Event::HeadObserved {
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

    // Verifies late header-not-found after a successful response is ignored.
    #[test]
    fn late_block_header_not_found_after_success_is_ignored() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            state,
            Event::HeadObserved {
                hash: head_hash,
                parent_hash: missing_parent_hash,
            },
        );
        let request_id = assert_single_block_header_request_effect(&effects, missing_parent_hash);
        let state = drain_block_log_effects(state, &effects);
        let (state, effects) = transition(
            state,
            Event::BlockHeaderReceived {
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

        let (next_state, effects) = transition(state, Event::BlockHeaderNotFound { request_id });

        assert!(effects.is_empty());
        assert_eq!(next_state.canonical_tip, head_hash);
        assert_single_block_with_parent(&next_state, head_hash, missing_parent_hash);
        assert_single_block_with_parent(&next_state, missing_parent_hash, finalized_hash);
        assert!(next_state.pending_requests.is_empty_for_test());
        assert_state_invariants(&next_state);
    }

    // Verifies late header-not-found for a failed request is ignored.
    #[test]
    fn late_block_header_not_found_for_failed_request_is_ignored() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            state,
            Event::HeadObserved {
                hash: head_hash,
                parent_hash: missing_parent_hash,
            },
        );
        let failed_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);
        let state = drain_block_log_effects(state, &effects);
        let (state, effects) = transition(state, request_failed_for_header(failed_request_id));
        let retry_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);

        let (next_state, effects) = transition(
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

    // Verifies late header-not-found for an expired request is ignored.
    #[test]
    fn late_block_header_not_found_for_expired_request_is_ignored() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            state,
            Event::HeadObserved {
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

    // Verifies header-not-found for the current retry resets chain state.
    #[test]
    fn block_header_not_found_for_current_retry_resets_chain() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            state,
            Event::HeadObserved {
                hash: head_hash,
                parent_hash: missing_parent_hash,
            },
        );
        let failed_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);
        let state = drain_block_log_effects(state, &effects);
        let (state, effects) = transition(state, request_failed_for_header(failed_request_id));
        let retry_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);

        let (next_state, effects) = transition(
            state,
            Event::BlockHeaderNotFound {
                request_id: retry_request_id,
            },
        );

        assert!(effects.is_empty());
        assert_empty_initial_state_at(&next_state, finalized_hash);
        assert_state_invariants(&next_state);
    }

    // Verifies header-not-found requests are not retried at their original expiration.
    #[test]
    fn block_header_not_found_request_is_not_retried_at_original_expiration() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            state,
            Event::HeadObserved {
                hash: head_hash,
                parent_hash: missing_parent_hash,
            },
        );
        let request_id = assert_single_block_header_request_effect(&effects, missing_parent_hash);
        let state = drain_block_log_effects(state, &effects);
        let (state, effects) = advance_ticks(state, REQUEST_TTL - 1);
        assert!(effects.is_empty());
        let (state, effects) = transition(state, Event::BlockHeaderNotFound { request_id });
        assert!(effects.is_empty());
        assert_empty_initial_state_at(&state, finalized_hash);

        let (next_state, effects) = transition(state, Event::Tick);

        assert!(effects.is_empty());
        assert!(next_state.tick == tick(REQUEST_TTL));
        assert_empty_initial_state_at(&next_state, finalized_hash);
        assert_state_invariants(&next_state);
    }

    // Verifies mismatched header responses preserve the missing canonical parent request.
    #[test]
    fn mismatched_header_response_does_not_lose_missing_canonical_parent_request() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let unrelated_hash = BlockHash::with_last_byte(4);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            state,
            Event::HeadObserved {
                hash: head_hash,
                parent_hash: missing_parent_hash,
            },
        );
        let request_id = assert_single_block_header_request_effect(&effects, missing_parent_hash);

        let (next_state, effects) = transition(
            state,
            Event::BlockHeaderReceived {
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

    // Verifies conflicting mismatched header responses reset and retry the original request.
    #[test]
    fn conflicting_mismatched_header_response_resets_chain_and_retries_request() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let unrelated_hash = BlockHash::with_last_byte(4);
        let conflicting_parent_hash = BlockHash::with_last_byte(5);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            state,
            Event::HeadObserved {
                hash: head_hash,
                parent_hash: missing_parent_hash,
            },
        );
        let request_id = assert_single_block_header_request_effect(&effects, missing_parent_hash);

        let (state, effects) = transition(
            state,
            Event::BlockHeaderReceived {
                request_id: RequestId::from_raw_for_test(99),
                hash: unrelated_hash,
                parent_hash: finalized_hash,
            },
        );
        assert!(effects.is_empty());

        let (next_state, effects) = transition(
            state,
            Event::BlockHeaderReceived {
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

    // Verifies reset does not reuse request ids while old effects may still arrive.
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
            state,
            Event::HeadObserved {
                hash: first_head_hash,
                parent_hash: first_missing_parent_hash,
            },
        );
        let old_request_id =
            assert_single_block_header_request_effect(&effects, first_missing_parent_hash);

        let (state, effects) = transition(
            state,
            Event::HeadObserved {
                hash: first_head_hash,
                parent_hash: conflicting_parent_hash,
            },
        );
        assert_chain_reset_at(&state, finalized_hash);
        assert!(state.pending_requests.contains(&old_request_id));
        assert!(effects.is_empty());

        let (_state, effects) = transition(
            state,
            Event::HeadObserved {
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

    // Verifies ticks before TTL do not retry active requests.
    #[test]
    fn tick_before_ttl_does_not_retry_request() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            state,
            Event::HeadObserved {
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

    // Verifies a request is retried exactly once when it reaches TTL.
    #[test]
    fn tick_at_ttl_retries_request_exactly_once() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            state,
            Event::HeadObserved {
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

    // Verifies expiration arithmetic works across tick wraparound.
    #[test]
    fn expiration_works_across_tick_wraparound() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let dispatch_tick = tick(u64::MAX - 4);
        let mut state = empty_state_at(finalized_hash);
        state.tick = dispatch_tick;

        let (state, effects) = transition(
            state,
            Event::HeadObserved {
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

        let (next_state, effects) = transition(state, Event::Tick);

        let retry_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);
        assert!(retry_request_id != expired_request_id);
        assert!(next_state.tick == tick(dispatch_tick.raw_for_test().wrapping_add(REQUEST_TTL)));
        assert_eq!(next_state.pending_requests.len_for_test(), 1);
        assert_no_active_request_is_expired(&next_state);
    }

    // Verifies a retry uses a fresh TTL window.
    #[test]
    fn retry_does_not_expire_again_until_another_ttl() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            state,
            Event::HeadObserved {
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

        let (next_state, effects) = transition(state, Event::Tick);

        let second_retry_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);
        assert!(second_retry_id != first_retry_id);
        assert_eq!(next_state.pending_requests.len_for_test(), 1);
        assert!(next_state.pending_requests.contains(&second_retry_id));
        assert_no_active_request_is_expired(&next_state);
    }

    // Verifies requests dispatched at different ticks expire independently.
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

    // Verifies multiple requests dispatched together each expire once.
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

    // Verifies a tick with no expired requests only advances time.
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

        let (next_state, effects) = transition(state, Event::Tick);

        assert!(next_state.tick == tick(1));
        assert!(effects.is_empty());
        assert_eq!(next_state.canonical_tip, known_hash);
        assert_eq!(next_state.finalized_state.block_hash, finalized_hash);
        assert_eq!(next_state.blocks.0.len(), 1);
        assert_single_unknown_block(&next_state, known_hash, finalized_hash);
        assert!(next_state.pending_requests.is_empty_for_test());
        assert_state_invariants(&next_state);
    }

    // Verifies an explicit request failure retries a known request with a fresh id.
    #[test]
    fn request_failed_retries_known_request_with_fresh_id() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            state,
            Event::HeadObserved {
                hash: head_hash,
                parent_hash: missing_parent_hash,
            },
        );
        let failed_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);
        let state = drain_block_log_effects(state, &effects);

        let (next_state, effects) = transition(state, request_failed_for_header(failed_request_id));

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

    // Verifies explicit failure for an unknown request id is ignored.
    #[test]
    fn request_failed_for_unknown_id_is_noop() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            state,
            Event::HeadObserved {
                hash: head_hash,
                parent_hash: missing_parent_hash,
            },
        );
        let pending_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);
        let state = drain_block_log_effects(state, &effects);
        let last_request_id = state.pending_requests.last_request_id_for_test();

        let (next_state, effects) = transition(
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

    // Verifies duplicate failure for an old request id is ignored.
    #[test]
    fn duplicate_request_failed_for_old_id_is_ignored() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            state,
            Event::HeadObserved {
                hash: head_hash,
                parent_hash: missing_parent_hash,
            },
        );
        let failed_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);
        let state = drain_block_log_effects(state, &effects);
        let (state, effects) = transition(state, request_failed_for_header(failed_request_id));
        let retry_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);

        let (next_state, effects) = transition(state, request_failed_for_header(failed_request_id));

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

    // Verifies repeated retry failures do not grow active pending request count.
    #[test]
    fn failed_retry_can_be_retried_again_without_growing_pending_requests() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            state,
            Event::HeadObserved {
                hash: head_hash,
                parent_hash: missing_parent_hash,
            },
        );
        let mut failed_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);
        let mut state = drain_block_log_effects(state, &effects);

        for _ in 0..4 {
            let (next_state, effects) =
                transition(state, request_failed_for_header(failed_request_id));
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

    // Verifies failures for preserved requests still retry after chain reset.
    #[test]
    fn request_failed_after_chain_reset_retries_preserved_request() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let conflicting_parent_hash = BlockHash::with_last_byte(4);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            state,
            Event::HeadObserved {
                hash: head_hash,
                parent_hash: missing_parent_hash,
            },
        );
        let failed_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);
        let state = drain_block_log_effects(state, &effects);
        let (state, effects) = transition(
            state,
            Event::HeadObserved {
                hash: head_hash,
                parent_hash: conflicting_parent_hash,
            },
        );
        assert!(effects.is_empty());
        assert_chain_reset_at(&state, finalized_hash);

        let (next_state, effects) = transition(state, request_failed_for_header(failed_request_id));

        assert_chain_reset_at(&next_state, finalized_hash);
        let retry_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);
        assert!(retry_request_id != failed_request_id);
        assert_eq!(next_state.pending_requests.len_for_test(), 1);
        assert!(next_state.pending_requests.contains(&retry_request_id));
        assert_state_invariants(&next_state);
    }

    // Verifies failure after successful completion of the same id is ignored.
    #[test]
    fn successful_response_followed_by_failure_for_old_id_is_ignored() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            state,
            Event::HeadObserved {
                hash: head_hash,
                parent_hash: missing_parent_hash,
            },
        );
        let request_id = assert_single_block_header_request_effect(&effects, missing_parent_hash);
        let state = drain_block_log_effects(state, &effects);
        let (state, effects) = transition(
            state,
            Event::BlockHeaderReceived {
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

        let (next_state, effects) = transition(state, request_failed_for_header(request_id));

        assert!(effects.is_empty());
        assert!(next_state.pending_requests.is_empty_for_test());
        assert_eq!(next_state.canonical_tip, head_hash);
        assert_single_block_with_parent(&next_state, head_hash, missing_parent_hash);
        assert_single_block_with_parent(&next_state, missing_parent_hash, finalized_hash);
        assert_state_invariants(&next_state);
    }

    // Verifies late success for a failed original request is accepted.
    #[test]
    fn late_success_for_failed_request_is_accepted_while_retry_remains_pending() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            state,
            Event::HeadObserved {
                hash: head_hash,
                parent_hash: missing_parent_hash,
            },
        );
        let failed_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);
        let state = drain_block_log_effects(state, &effects);
        let (state, effects) = transition(state, request_failed_for_header(failed_request_id));
        let retry_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);

        let (next_state, effects) = transition(
            state,
            Event::BlockHeaderReceived {
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

    // Verifies duplicate failure after successful retry is ignored.
    #[test]
    fn successful_retry_followed_by_duplicate_failure_for_original_id_is_ignored() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            state,
            Event::HeadObserved {
                hash: head_hash,
                parent_hash: missing_parent_hash,
            },
        );
        let failed_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);
        let state = drain_block_log_effects(state, &effects);
        let (state, effects) = transition(state, request_failed_for_header(failed_request_id));
        let retry_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);
        let (state, effects) = transition(
            state,
            Event::BlockHeaderReceived {
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

        let (next_state, effects) = transition(state, request_failed_for_header(failed_request_id));

        assert!(effects.is_empty());
        assert!(next_state.pending_requests.is_empty_for_test());
        assert_eq!(next_state.canonical_tip, head_hash);
        assert_single_block_with_parent(&next_state, head_hash, missing_parent_hash);
        assert_single_block_with_parent(&next_state, missing_parent_hash, finalized_hash);
        assert_state_invariants(&next_state);
    }

    // Verifies completed requests are not retried at their original expiration.
    #[test]
    fn completed_request_is_not_retried_at_original_expiration() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            state,
            Event::HeadObserved {
                hash: head_hash,
                parent_hash: missing_parent_hash,
            },
        );
        let request_id = assert_single_block_header_request_effect(&effects, missing_parent_hash);
        let state = drain_block_log_effects(state, &effects);
        let (state, effects) = advance_ticks(state, REQUEST_TTL - 1);
        assert!(effects.is_empty());
        let (state, effects) = transition(
            state,
            Event::BlockHeaderReceived {
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

        let (next_state, effects) = transition(state, Event::Tick);

        assert!(effects.is_empty());
        assert!(next_state.tick == tick(REQUEST_TTL));
        assert!(next_state.pending_requests.is_empty_for_test());
        assert_single_block_with_parent(&next_state, head_hash, missing_parent_hash);
        assert_single_block_with_parent(&next_state, missing_parent_hash, finalized_hash);
        assert_state_invariants(&next_state);
    }

    // Verifies failed requests are not retried at their original expiration.
    #[test]
    fn failed_request_is_not_retried_at_original_expiration() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            state,
            Event::HeadObserved {
                hash: head_hash,
                parent_hash: missing_parent_hash,
            },
        );
        let failed_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);
        let state = drain_block_log_effects(state, &effects);
        let (state, effects) = advance_ticks(state, REQUEST_TTL - 1);
        assert!(effects.is_empty());
        let (state, effects) = transition(state, request_failed_for_header(failed_request_id));
        let retry_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);

        let (state, effects) = transition(state, Event::Tick);

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

    // Verifies late success after expiration is accepted while retry remains pending.
    #[test]
    fn late_success_after_expiration_is_accepted_while_retry_remains_pending() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            state,
            Event::HeadObserved {
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
            state,
            Event::BlockHeaderReceived {
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

    // Verifies chain reset preserves pending request expiration age.
    #[test]
    fn chain_reset_preserves_request_expiration_age() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let conflicting_parent_hash = BlockHash::with_last_byte(4);
        let state = empty_state_at(finalized_hash);

        let (state, effects) = transition(
            state,
            Event::HeadObserved {
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
            state,
            Event::HeadObserved {
                hash: head_hash,
                parent_hash: conflicting_parent_hash,
            },
        );
        assert!(effects.is_empty());
        assert_chain_reset_at(&state, finalized_hash);
        assert!(state.pending_requests.contains(&expired_request_id));

        let (next_state, effects) = transition(state, Event::Tick);

        let retry_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);
        assert!(retry_request_id != expired_request_id);
        assert_chain_reset_at(&next_state, finalized_hash);
        assert_eq!(next_state.pending_requests.len_for_test(), 1);
        assert_no_active_request_is_expired(&next_state);
        assert_state_invariants(&next_state);
    }

    /// Fulfills emitted pool-metadata effects with deterministic validation outcomes.
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
                        let result = if candidate.0.as_slice()[19] % 2 == 0 {
                            Ok(pool_metadata(1, 2, UniswapV3Fee::Fee3000))
                        } else {
                            Err(PoolMetadataFailure::FactoryReturnedZero)
                        };

                        (candidate, result)
                    })
                    .collect::<HashMap<_, _>>();
                let (next_state, effects) = transition(
                    state,
                    Event::PoolMetadataReceived {
                        request_id,
                        metadata,
                    },
                );

                state = next_state;
                assert!(effects.is_empty());
            }
        }

        state
    }

    // Property tests below use descriptive function names as their scenario annotations.
    proptest! {
        // Verifies wrapping tick arithmetic matches elapsed-time expiration.
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

        // Verifies each expired request is replaced once on tick.
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
            let (next_state, effects) = transition(state, Event::Tick);

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

        // Verifies duplicate head observations do not duplicate pending log work.
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

            let (state, effects) = transition(
                state,
                Event::HeadObserved {
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

            let (next_state, effects) = transition(
                state,
                Event::HeadObserved {
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

        // Verifies drained canonical paths never retain unknown log status.
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
                    Event::HeadObserved { hash, parent_hash },
                );

                assert_present_canonical_logs_are_resolved(&state);
                assert_state_invariants(&state);
            }
        }

        // Verifies canonical resolved candidates are always known or pending validation.
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
                let (next_state, effects) = transition(
                    state,
                    Event::HeadObserved {
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
                        let (next_state, effects) = transition(
                            state,
                            Event::BlockLogsReceived {
                                request_id,
                                logs: candidates,
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

        // Verifies trusted log projections never expose unverified pool candidates.
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
                let (next_state, effects) = transition(
                    state,
                    Event::HeadObserved {
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
                        let (next_state, effects) = transition(
                            state,
                            Event::BlockLogsReceived {
                                request_id,
                                logs: candidates,
                            },
                        );
                        state = apply_pool_metadata_effects_for_property(next_state, effects);
                    }
                }

                parent_hash = block_hash;
            }

            for block_hash in state.blocks.0.keys() {
                if let Some(TrustedPoolLogs::Resolved(pools)) =
                    state.blocks.trusted_pool_logs(*block_hash, &state.pool_registry)
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

        // Verifies transition reconstructs observed valid chain ancestry.
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
                    Event::HeadObserved { hash, parent_hash },
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

        // Verifies delayed header responses still reconstruct valid chain ancestry.
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
                    transition(state, Event::HeadObserved { hash, parent_hash });

                state = next_state;
                pending_effects.extend(effects);
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
                        let (next_state, effects) = transition(
                            state,
                            Event::BlockHeaderReceived {
                                request_id,
                                hash: block_hash,
                                parent_hash,
                            },
                        );

                        state = next_state;
                        pending_effects.extend(effects);
                    }
                    Effect::Request(AnyIssuedRequest::BlockLogs(IssuedRequest {
                        request_id, ..
                    })) => {
                        let (next_state, effects) = transition(
                            state,
                            Event::BlockLogsReceived {
                                request_id,
                                logs: HashSet::new(),
                            },
                        );

                        state = next_state;
                        pending_effects.extend(effects);
                    }
                    Effect::Request(AnyIssuedRequest::PoolData(_)) => {}
                    Effect::Request(AnyIssuedRequest::PoolMetadata(_)) => {}
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

        // Verifies finite retry plans still reconstruct valid chain ancestry.
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
                    Event::HeadObserved { hash, parent_hash },
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

        // Verifies actionable not-found responses reset only matching request state.
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

            let (next_state, effects) = transition(
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

        // Verifies unknown not-found responses preserve existing state.
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

            let (next_state, effects) = transition(
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

        // Verifies chain reconstruction after not-found reset and reobservation.
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
            let (state, effects) = transition(
                state,
                Event::HeadObserved {
                    hash: first_head_hash,
                    parent_hash: first_parent_hash,
                },
            );
            let request_id = assert_single_block_header_request_effect(&effects, first_parent_hash);
            let state = drain_block_log_effects(state, &effects);

            let (mut state, effects) =
                transition(state, Event::BlockHeaderNotFound { request_id });

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
                    Event::HeadObserved { hash, parent_hash },
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

        // Verifies pool-data responses never apply unrequested result entries.
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

            let (next_state, effects) = transition(
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

        // Verifies request failures preserve payload identity across retry.
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

            let (next_state, effects) = transition(
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

        // Verifies arbitrary events preserve core state-safety invariants.
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
                    transition(state, event_from_generated(generated_event));

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
