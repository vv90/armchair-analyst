use std::{
    collections::{HashMap, HashSet, hash_map::Entry},
    hash::Hash,
};

use alloy::primitives::{Address, BlockHash};

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

#[derive(Hash, PartialEq, Eq, Clone, Copy)]
struct RequestId(u64);

impl RequestId {
    fn next(self) -> RequestId {
        RequestId(self.0 + 1)
    }
}

#[derive(Clone)]
enum RequestPayload {
    GetBlockLogs {
        block_hash: BlockHash,
    },
    GetBlockHeader {
        block_hash: BlockHash,
    },
    GetPoolData {
        at: BlockHash,
        pools: HashSet<PoolAddress>,
    },
}

enum NewBlockError {
    CycleDetected,
    ConflictingParent,
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
    ) -> Result<(BlocksGraph, Option<BlockHash>), NewBlockError> {
        if hash == parent_hash {
            return Ok((self, None));
        }

        if let Some(block) = self.get(&hash) {
            if block.parent_hash == parent_hash {
                return Ok((self, None));
            }

            return Err(NewBlockError::ConflictingParent);
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

        match find_missing_block_hash(&blocks, hash, parent_hash) {
            Ok(None) => Ok((BlocksGraph(blocks), None)),
            Ok(Some(missing_hash)) => Ok((BlocksGraph(blocks), Some(missing_hash))),
            Err(BlocksGraphCycleError) => Err(NewBlockError::CycleDetected),
        }
    }
}

struct PendingRequests {
    requests: HashMap<RequestId, RequestPayload>,
    last_request_id: RequestId,
}

impl PendingRequests {
    fn new() -> PendingRequests {
        PendingRequests {
            requests: HashMap::new(),
            last_request_id: RequestId(0),
        }
    }

    fn get(&self, request_id: &RequestId) -> Option<&RequestPayload> {
        self.requests.get(request_id)
    }

    fn with_new_request(self, request_payload: RequestPayload) -> (PendingRequests, RequestId) {
        let PendingRequests {
            mut requests,
            last_request_id,
        } = self;
        let new_request_id = last_request_id.next();

        requests.insert(new_request_id, request_payload);

        (
            PendingRequests {
                requests,
                last_request_id: new_request_id,
            },
            new_request_id,
        )
    }
}

struct FinalizedState {
    block_hash: BlockHash,
    pool_snapshots: HashMap<PoolAddress, PoolState>,
}

struct State {
    blocks: BlocksGraph,
    canonical_tip: BlockHash,
    pending_requests: PendingRequests,
    finalized_state: FinalizedState,
}

impl State {
    fn init(finalized_state: FinalizedState) -> State {
        State {
            blocks: BlocksGraph::new(),
            canonical_tip: finalized_state.block_hash,
            pending_requests: PendingRequests::new(),
            finalized_state,
        }
    }
}

enum Event {
    HeadObserved {
        hash: BlockHash,
        parent_hash: BlockHash,
    },
    BlockHeaderReceived {
        hash: BlockHash,
        parent_hash: BlockHash,
    },
    BlockLogsReceived {
        request_id: RequestId,
        logs: HashSet<PoolAddress>,
    },
    PoolDataReceived {
        request_id: RequestId,
        pools: HashMap<PoolAddress, PoolState>,
    },
}

enum Effect {
    Request(RequestId, RequestPayload),
}

struct BlocksGraphCycleError;

fn transition(state: State, event: Event) -> (State, Vec<Effect>) {
    match event {
        Event::HeadObserved { hash, parent_hash } => {
            let State {
                blocks,
                pending_requests,
                finalized_state,
                ..
            } = state;

            match blocks.with_new_block(hash, parent_hash) {
                Ok((blocks, None)) => (
                    State {
                        blocks,
                        canonical_tip: hash,
                        pending_requests,
                        finalized_state,
                    },
                    vec![],
                ),
                Ok((blocks, Some(missing_hash))) => {
                    let request_payload = RequestPayload::GetBlockHeader {
                        block_hash: missing_hash,
                    };
                    let (pending_requests, request_id) =
                        pending_requests.with_new_request(request_payload.clone());

                    (
                        State {
                            blocks,
                            canonical_tip: hash,
                            pending_requests,
                            finalized_state,
                        },
                        vec![Effect::Request(
                            request_id,
                            RequestPayload::GetBlockHeader {
                                block_hash: missing_hash,
                            },
                        )],
                    )
                }
                Err(_) => (State::init(finalized_state), vec![]),
            }

            // match find_missing_block_hash(&blocks, hash, finalized_state.block_hash) {
            //     Ok(None) => (
            //         State {
            //             blocks,
            //             canonical_tip: hash,
            //             pending_requests,
            //             finalized_state,
            //             last_request_id,
            //         },
            //         vec![],
            //     ),
            //     Ok(Some(missing_hash)) => {
            //         let request_id = last_request_id.next();
            //         let request_payload = RequestPayload::GetBlockHeader {
            //             block_hash: missing_hash,
            //         };

            //         pending_requests.insert(request_id, request_payload.clone());

            //         (
            //             State {
            //                 blocks,
            //                 canonical_tip: hash,
            //                 pending_requests,
            //                 finalized_state,
            //                 last_request_id: request_id,
            //             },
            //             vec![Effect::Request(request_id, request_payload)],
            //         )
            //     }
            //     Err(BlocksGraphCycleError) => (State::init(finalized_state), vec![]),
            // }
        }
        Event::BlockHeaderReceived { hash, parent_hash } => {
            let State {
                blocks,
                pending_requests,
                finalized_state,
                canonical_tip,
            } = state;

            match blocks.with_new_block(hash, parent_hash) {
                Ok((blocks, None)) => (
                    State {
                        blocks,
                        canonical_tip,
                        pending_requests,
                        finalized_state,
                    },
                    vec![],
                ),
                Ok((blocks, Some(missing_hash))) => {
                    let request_payload = RequestPayload::GetBlockHeader {
                        block_hash: missing_hash,
                    };
                    let (pending_requests, request_id) =
                        pending_requests.with_new_request(request_payload.clone());

                    (
                        State {
                            blocks,
                            canonical_tip,
                            pending_requests,
                            finalized_state,
                        },
                        vec![Effect::Request(
                            request_id,
                            RequestPayload::GetBlockHeader {
                                block_hash: missing_hash,
                            },
                        )],
                    )
                }
                Err(_) => (State::init(finalized_state), vec![]),
            }
        }
        Event::BlockLogsReceived { request_id, logs } => (state, vec![]),
        Event::PoolDataReceived { request_id, pools } => (state, vec![]),
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

    fn assert_state_invariants(state: &State) {
        assert_finalized_block_not_in_recent_blocks(state);
        assert_no_self_parent_blocks(state);
        assert_canonical_tip_has_no_known_child(state);
        assert_canonical_tip_is_known_or_finalized(state);
        assert_parent_walks_do_not_cycle(state);
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

    fn assert_canonical_tip_has_no_known_child(state: &State) {
        assert!(
            state
                .blocks
                .0
                .values()
                .all(|block| block.parent_hash != state.canonical_tip),
            "canonical tip must not have a known child"
        );
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
        }
    }

    fn assert_empty_initial_state_at(state: &State, finalized_hash: BlockHash) {
        assert!(state.blocks.0.is_empty());
        assert_eq!(state.canonical_tip, finalized_hash);
        assert!(state.pending_requests.requests.is_empty());
        assert_eq!(state.finalized_state.block_hash, finalized_hash);
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
    #[should_panic(expected = "canonical tip must not have a known child")]
    fn state_invariants_reject_known_child_of_canonical_tip() {
        let finalized_hash = BlockHash::with_last_byte(1);
        let tip_hash = BlockHash::with_last_byte(2);
        let child_hash = BlockHash::with_last_byte(3);
        let mut state = empty_state_at(finalized_hash);

        state.canonical_tip = tip_hash;
        state
            .blocks
            .0
            .insert(tip_hash, block_with_parent(finalized_hash));
        state
            .blocks
            .0
            .insert(child_hash, block_with_parent(tip_hash));

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
    fn head_observed_with_self_parent_does_not_change_state() {
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

        assert_empty_initial_state_at(&next_state, finalized_hash);
        assert!(effects.is_empty());
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

        assert_empty_initial_state_at(&next_state, finalized_hash);
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

        assert_empty_initial_state_at(&next_state, finalized_hash);
        assert!(effects.is_empty());
        assert_state_invariants(&next_state);
    }
}
