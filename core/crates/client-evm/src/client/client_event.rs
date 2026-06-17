use std::collections::HashSet;

use alloy::{primitives::BlockHash, rpc::types::Header, serde::WithOtherFields};

use crate::PoolCandidateAddress;

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
    Closed {
        subscription_id: String,
    },
}
