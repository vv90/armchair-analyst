use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
};

use alloy::primitives::{Address, BlockHash};

use crate::{pending_requests::*, tick::Tick};

// placeholder for pool state
struct PoolState {}

#[derive(Hash, PartialEq, Eq, Clone, Copy)]
struct PoolAddress(Address);

enum PoolLogsStatus {
    Unknown,
    Resolved(HashSet<PoolAddress>),
}

struct BlockNode {
    parent_hash: BlockHash,
    pool_logs: PoolLogsStatus,
    pool_snapshots: HashMap<PoolAddress, PoolState>,
}

enum NewBlockError {
    SelfParentBlock(BlockHash, BlocksGraph),
    ExistingBlock(BlocksGraph),
    ConflictingBlockParent,
    CycleDetected,
}

struct BlocksGraph(HashMap<BlockHash, BlockNode>);

impl BlocksGraph {
    fn new() -> BlocksGraph {
        BlocksGraph(HashMap::new())
    }

    fn get(&self, hash: &BlockHash) -> Option<&BlockNode> {
        self.0.get(hash)
    }

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
            },
        );

        match find_missing_block_hash(&blocks, hash, finalized_hash) {
            Ok(None) => Ok((BlocksGraph(blocks), None)),
            Ok(Some(missing_hash)) => Ok((BlocksGraph(blocks), Some(missing_hash))),
            Err(BlocksGraphCycleError) => Err(NewBlockError::CycleDetected),
        }
    }
}

pub struct FinalizedState {
    pub block_hash: BlockHash,
    pool_snapshots: HashMap<PoolAddress, PoolState>,
}

impl FinalizedState {
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
    tick: Tick,
}

impl State {
    pub fn init(finalized_state: FinalizedState) -> State {
        State {
            blocks: BlocksGraph::new(),
            canonical_tip: finalized_state.block_hash,
            pending_requests: PendingRequests::new(),
            finalized_state,
            tick: Tick::initial(),
        }
    }
    fn reset(finalized_state: FinalizedState, tick: Tick) -> State {
        State {
            blocks: BlocksGraph::new(),
            canonical_tip: finalized_state.block_hash,
            pending_requests: PendingRequests::new(),
            finalized_state,
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
        logs: HashSet<PoolAddress>,
    },
    PoolDataReceived {
        request_id: RequestId<GetPoolData>,
        pools: HashMap<PoolAddress, PoolState>,
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

pub fn transition(state: State, event: Event) -> (State, Vec<Effect>) {
    match event {
        Event::HeadObserved { hash, parent_hash } => {
            match state
                .blocks
                .with_new_block(hash, parent_hash, state.finalized_state.block_hash)
            {
                Ok((blocks, None)) => (
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

                    (
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
                Err(NewBlockError::ExistingBlock(blocks)) => (
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
                        ..State::reset(state.finalized_state, state.tick)
                    },
                    vec![],
                ),
                Err(NewBlockError::CycleDetected) => (
                    State {
                        pending_requests: state.pending_requests,
                        ..State::reset(state.finalized_state, state.tick)
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

            let (new_state, effects) = match state.blocks.with_new_block(
                hash,
                parent_hash,
                state.finalized_state.block_hash,
            ) {
                Ok((blocks, None)) => (
                    State {
                        blocks,
                        pending_requests,
                        ..state
                    },
                    vec![],
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
                    )
                }
                Err(NewBlockError::ExistingBlock(blocks)) => (
                    State {
                        blocks,
                        pending_requests,
                        ..state
                    },
                    vec![],
                ),
                Err(NewBlockError::ConflictingBlockParent) => (
                    State {
                        pending_requests,
                        ..State::reset(state.finalized_state, state.tick)
                    },
                    vec![],
                ),
                Err(NewBlockError::CycleDetected) => (
                    State {
                        pending_requests,
                        ..State::reset(state.finalized_state, state.tick)
                    },
                    vec![],
                ),
            };

            if let Some(PendingPayload {
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
            }
        }
        Event::BlockHeaderNotFound { request_id } => {
            let (pending_requests, payload) = state.pending_requests.take(&request_id);
            if let Some(_) = payload {
                (
                    State {
                        pending_requests,
                        ..State::reset(state.finalized_state, state.tick)
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
        Event::BlockLogsReceived { request_id, logs } => (state, vec![]),
        Event::PoolDataReceived { request_id, pools } => (state, vec![]),
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

    fn generated_event_sequence_strategy() -> impl Strategy<Value = Vec<GeneratedEvent>> {
        prop::collection::vec(generated_event_strategy(), 1..128)
    }

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

    fn generated_request_payload_strategy() -> impl Strategy<Value = GeneratedRequestPayload> {
        prop_oneof![
            any::<u8>()
                .prop_map(|block_index| GeneratedRequestPayload::GetBlockLogs { block_index }),
            any::<u8>()
                .prop_map(|block_index| GeneratedRequestPayload::GetBlockHeader { block_index }),
        ]
    }

    fn assert_state_invariants(state: &State) {
        assert_finalized_block_not_in_recent_blocks(state);
        assert_no_self_parent_blocks(state);
        assert_canonical_tip_is_known_or_finalized(state);
        assert_parent_walks_do_not_cycle(state);
    }

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
            }
        }
    }

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

    fn assert_missing_parents_for_known_blocks_are_pending(state: &State) {
        for block_hash in state.blocks.0.keys() {
            assert_missing_parent_is_pending(state, *block_hash);
        }
    }

    fn assert_finalized_block_not_in_recent_blocks(state: &State) {
        assert!(
            !state
                .blocks
                .0
                .contains_key(&state.finalized_state.block_hash),
            "finalized block must not be present in recent blocks"
        );
    }

    fn assert_no_self_parent_blocks(state: &State) {
        for (hash, block) in &state.blocks.0 {
            assert_ne!(
                hash, &block.parent_hash,
                "block must not reference itself as parent"
            );
        }
    }

    fn assert_canonical_tip_is_known_or_finalized(state: &State) {
        assert!(
            state.canonical_tip == state.finalized_state.block_hash
                || state.blocks.0.contains_key(&state.canonical_tip),
            "canonical tip must be finalized or present in recent blocks"
        );
    }

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

    fn hash_for_node(node_index: usize) -> BlockHash {
        BlockHash::with_last_byte((node_index + 1) as u8)
    }

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

    fn tick(value: u64) -> Tick {
        Tick::from_raw_for_test(value)
    }

    fn request_failed_for_header(request_id: RequestId<GetBlockHeader>) -> Event {
        Event::RequestFailed {
            request_id: AnyRequestId::BlockHeader(request_id),
        }
    }

    #[test]
    fn finalized_state_empty_at_stores_hash_with_empty_pool_snapshots() {
        let finalized_hash = BlockHash::with_last_byte(1);

        let finalized_state = FinalizedState::empty_at(finalized_hash);

        assert_eq!(finalized_state.block_hash, finalized_hash);
        assert!(finalized_state.pool_snapshots.is_empty());
    }

    #[test]
    fn state_init_from_finalized_state_starts_with_empty_tracking() {
        let finalized_hash = BlockHash::with_last_byte(1);

        let state = State::init(FinalizedState::empty_at(finalized_hash));

        assert_empty_initial_state_at(&state, finalized_hash);
        assert!(state.blocks.0.is_empty());
        assert!(state.finalized_state.pool_snapshots.is_empty());
        assert_eq!(state.tick.raw_for_test(), Tick::initial().raw_for_test());
    }

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

    fn any_request_id_raw(request_id: AnyRequestId) -> u64 {
        match request_id {
            AnyRequestId::BlockHeader(request_id) => request_id.raw_for_test(),
            AnyRequestId::BlockLogs(request_id) => request_id.raw_for_test(),
            AnyRequestId::PoolData(request_id) => request_id.raw_for_test(),
        }
    }

    fn parent_index(chain: &GeneratedChain, node_index: usize) -> usize {
        chain.parents.get(node_index).copied().unwrap_or_default()
    }

    fn node_index_for_hash(chain: &GeneratedChain, hash: BlockHash) -> Option<usize> {
        (0..chain.parents.len()).find(|node_index| hash_for_node(*node_index) == hash)
    }

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
            let Effect::Request(AnyIssuedRequest::BlockHeader(IssuedRequest {
                request_id,
                request_payload: GetBlockHeader { block_hash },
            })) = effect
            else {
                continue;
            };

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

        state
    }

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
            let Effect::Request(AnyIssuedRequest::BlockHeader(IssuedRequest {
                request_id,
                request_payload: GetBlockHeader { block_hash },
            })) = effect
            else {
                continue;
            };

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

        state
    }

    fn block_with_parent(parent_hash: BlockHash) -> BlockNode {
        BlockNode {
            parent_hash,
            pool_logs: PoolLogsStatus::Unknown,
            pool_snapshots: HashMap::new(),
        }
    }

    fn empty_state_at(finalized_hash: BlockHash) -> State {
        State {
            blocks: BlocksGraph::new(),
            canonical_tip: finalized_hash,
            pending_requests: PendingRequests::new(),
            finalized_state: FinalizedState {
                block_hash: finalized_hash,
                pool_snapshots: HashMap::new(),
            },
            tick: tick(0),
        }
    }

    fn assert_chain_reset_at(state: &State, finalized_hash: BlockHash) {
        assert!(state.blocks.0.is_empty());
        assert_eq!(state.canonical_tip, finalized_hash);
        assert_eq!(state.finalized_state.block_hash, finalized_hash);
    }

    fn assert_empty_initial_state_at(state: &State, finalized_hash: BlockHash) {
        assert_chain_reset_at(state, finalized_hash);
        assert!(state.pending_requests.is_empty_for_test());
    }

    fn assert_single_unknown_block(state: &State, hash: BlockHash, parent_hash: BlockHash) {
        let block = state
            .blocks
            .get(&hash)
            .expect("expected block to be present");

        assert_eq!(block.parent_hash, parent_hash);
        assert!(matches!(block.pool_logs, PoolLogsStatus::Unknown));
        assert!(block.pool_snapshots.is_empty());
    }

    fn assert_single_block_header_request_effect(
        effects: &[Effect],
        block_hash: BlockHash,
    ) -> RequestId<GetBlockHeader> {
        assert_eq!(effects.len(), 1);

        match &effects[0] {
            Effect::Request(AnyIssuedRequest::BlockHeader(IssuedRequest {
                request_id,
                request_payload:
                    GetBlockHeader {
                        block_hash: requested_hash,
                    },
            })) => {
                assert_eq!(*requested_hash, block_hash);
                *request_id
            }
            _ => panic!("expected single block header request effect"),
        }
    }

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

    fn advance_ticks(mut state: State, count: u64) -> (State, Vec<Effect>) {
        let mut effects = Vec::new();

        for _ in 0..count {
            let (next_state, tick_effects) = transition(state, Event::Tick);
            state = next_state;
            effects.extend(tick_effects);
        }

        (state, effects)
    }

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

    fn pending_header_hashes(state: &State) -> HashSet<BlockHash> {
        state.pending_requests.pending_header_hashes_for_test()
    }

    fn active_request_dispatch_tick(
        pending_requests: &PendingRequests,
        request_id: RequestId<GetBlockHeader>,
    ) -> Tick {
        pending_requests
            .header_dispatch_tick_for_test(&request_id)
            .expect("active request must have a dispatch tick")
    }

    fn assert_active_requests_have_exactly_one_dispatch_tick(state: &State) {
        assert_eq!(
            state.pending_requests.dispatch_ticks_for_test().len(),
            state.pending_requests.len_for_test(),
            "active request must have exactly one dispatch tick"
        );
    }

    fn assert_no_active_request_is_expired(state: &State) {
        for dispatch_tick in state.pending_requests.dispatch_ticks_for_test() {
            assert!(
                !state.tick.is_expired_since(dispatch_tick),
                "active request must not remain expired after a tick"
            );
        }
    }

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

    #[test]
    #[should_panic(expected = "canonical tip must be finalized or present in recent blocks")]
    fn state_invariants_reject_unknown_canonical_tip() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let mut state = empty_state_at(finalized_hash);

        state.canonical_tip = BlockHash::with_last_byte(2);

        assert_state_invariants(&state);
    }

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
        assert!(effects.is_empty());
        assert_state_invariants(&next_state);
    }

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
        assert_single_unknown_block(&next_state, head_hash, missing_parent_hash);
        assert_single_unknown_block(&next_state, missing_parent_hash, finalized_hash);
        assert!(!next_state.pending_requests.contains(&request_id));
        assert!(effects.is_empty());
        assert_state_invariants(&next_state);
    }

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
        assert_single_unknown_block(&next_state, head_hash, missing_parent_hash);
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
        let (state, effects) = transition(
            state,
            Event::BlockHeaderReceived {
                request_id,
                hash: missing_parent_hash,
                parent_hash: finalized_hash,
            },
        );
        assert!(effects.is_empty());

        let (next_state, effects) = transition(state, Event::BlockHeaderNotFound { request_id });

        assert!(effects.is_empty());
        assert_eq!(next_state.canonical_tip, head_hash);
        assert_single_unknown_block(&next_state, head_hash, missing_parent_hash);
        assert_single_unknown_block(&next_state, missing_parent_hash, finalized_hash);
        assert!(next_state.pending_requests.is_empty_for_test());
        assert_state_invariants(&next_state);
    }

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
        assert_single_unknown_block(&next_state, head_hash, missing_parent_hash);
        assert_eq!(next_state.pending_requests.len_for_test(), 1);
        assert!(next_state.pending_requests.contains(&retry_request_id));
        assert_state_invariants(&next_state);
        assert_missing_parents_for_known_blocks_are_pending(&next_state);
    }

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
        assert_single_unknown_block(&next_state, head_hash, missing_parent_hash);
        assert_eq!(next_state.pending_requests.len_for_test(), 1);
        assert!(next_state.pending_requests.contains(&retry_request_id));
        assert_state_invariants(&next_state);
        assert_missing_parents_for_known_blocks_are_pending(&next_state);
    }

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
        assert_single_unknown_block(&next_state, head_hash, missing_parent_hash);
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

        let (next_state, effects) = advance_ticks(state, REQUEST_TTL - 1);

        assert!(effects.is_empty());
        assert!(next_state.tick == tick(REQUEST_TTL - 1));
        assert_eq!(next_state.pending_requests.len_for_test(), 1);
        assert!(next_state.pending_requests.contains(&request_id));
        assert_active_requests_have_exactly_one_dispatch_tick(&next_state);
    }

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

        let (next_state, effects) = transition(state, request_failed_for_header(failed_request_id));

        let retry_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);
        assert!(retry_request_id != failed_request_id);
        assert_eq!(next_state.pending_requests.len_for_test(), 1);
        assert!(!next_state.pending_requests.contains(&failed_request_id));
        assert!(next_state.pending_requests.contains(&retry_request_id));
        assert_eq!(next_state.canonical_tip, head_hash);
        assert_single_unknown_block(&next_state, head_hash, missing_parent_hash);
        assert_state_invariants(&next_state);
        assert_missing_parents_for_known_blocks_are_pending(&next_state);
    }

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
        assert_single_unknown_block(&next_state, head_hash, missing_parent_hash);
        assert_state_invariants(&next_state);
        assert_missing_parents_for_known_blocks_are_pending(&next_state);
    }

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

    #[test]
    fn failed_retry_can_be_retried_again_without_growing_pending_requests() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let missing_parent_hash = BlockHash::with_last_byte(2);
        let head_hash = BlockHash::with_last_byte(3);
        let state = empty_state_at(finalized_hash);

        let (mut state, effects) = transition(
            state,
            Event::HeadObserved {
                hash: head_hash,
                parent_hash: missing_parent_hash,
            },
        );
        let mut failed_request_id =
            assert_single_block_header_request_effect(&effects, missing_parent_hash);

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
        let (state, effects) = transition(
            state,
            Event::BlockHeaderReceived {
                request_id,
                hash: missing_parent_hash,
                parent_hash: finalized_hash,
            },
        );
        assert!(effects.is_empty());

        let (next_state, effects) = transition(state, request_failed_for_header(request_id));

        assert!(effects.is_empty());
        assert!(next_state.pending_requests.is_empty_for_test());
        assert_eq!(next_state.canonical_tip, head_hash);
        assert_single_unknown_block(&next_state, head_hash, missing_parent_hash);
        assert_single_unknown_block(&next_state, missing_parent_hash, finalized_hash);
        assert_state_invariants(&next_state);
    }

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

        assert!(effects.is_empty());
        assert_eq!(next_state.pending_requests.len_for_test(), 1);
        assert!(next_state.pending_requests.contains(&retry_request_id));
        assert_eq!(next_state.canonical_tip, head_hash);
        assert_single_unknown_block(&next_state, head_hash, missing_parent_hash);
        assert_single_unknown_block(&next_state, missing_parent_hash, finalized_hash);
        assert_state_invariants(&next_state);
    }

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
        assert!(effects.is_empty());

        let (next_state, effects) = transition(state, request_failed_for_header(failed_request_id));

        assert!(effects.is_empty());
        assert!(next_state.pending_requests.is_empty_for_test());
        assert_eq!(next_state.canonical_tip, head_hash);
        assert_single_unknown_block(&next_state, head_hash, missing_parent_hash);
        assert_single_unknown_block(&next_state, missing_parent_hash, finalized_hash);
        assert_state_invariants(&next_state);
    }

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
        assert!(effects.is_empty());

        let (next_state, effects) = transition(state, Event::Tick);

        assert!(effects.is_empty());
        assert!(next_state.tick == tick(REQUEST_TTL));
        assert!(next_state.pending_requests.is_empty_for_test());
        assert_single_unknown_block(&next_state, head_hash, missing_parent_hash);
        assert_single_unknown_block(&next_state, missing_parent_hash, finalized_hash);
        assert_state_invariants(&next_state);
    }

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

        assert!(effects.is_empty());
        assert_eq!(next_state.pending_requests.len_for_test(), 1);
        assert!(next_state.pending_requests.contains(&retry_request_id));
        assert_single_unknown_block(&next_state, head_hash, missing_parent_hash);
        assert_single_unknown_block(&next_state, missing_parent_hash, finalized_hash);
        assert_state_invariants(&next_state);
    }

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

    proptest! {
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
                prop_assert!(matches!(block.pool_logs, PoolLogsStatus::Unknown));
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
                let Effect::Request(AnyIssuedRequest::BlockHeader(IssuedRequest {
                    request_id,
                    request_payload: GetBlockHeader { block_hash },
                })) = effect
                else {
                    continue;
                };

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
                assert_effects_are_well_formed(&next_state, &effects);
                assert_missing_parents_for_known_blocks_are_pending(&next_state);
                assert_active_requests_have_exactly_one_dispatch_tick(&next_state);

                state = next_state;
            }
        }
    }
}
