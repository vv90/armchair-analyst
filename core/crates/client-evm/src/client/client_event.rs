use std::collections::HashSet;

use alloy::{primitives::BlockHash, rpc::types::Header, serde::WithOtherFields};

use crate::{PoolCandidateAddress, PoolLog};

pub type ClientHead = WithOtherFields<Header>;

/// Pool-event candidates discovered in a single block by a ranged `eth_getLogs` query.
/// Added so a one-shot range scan can attribute candidate pools to their block for bootstrap
/// graph and pool-log seeding without per-block log requests.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RangeLogBlock {
    pub number: u64,
    pub hash: BlockHash,
    pub candidates: HashSet<PoolCandidateAddress>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ClientEvent {
    Subscribed {
        subscription_id: String,
    },
    NewHead {
        subscription_id: String,
        header: ClientHead,
    },
    /// A single state-relevant pool log delivered by the live `logs` subscription, with the block
    /// it belongs to.
    PoolLogObserved {
        subscription_id: String,
        block_hash: BlockHash,
        log: PoolLog,
    },
    Closed {
        subscription_id: String,
    },
}
