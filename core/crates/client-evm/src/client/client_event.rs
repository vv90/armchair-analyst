use alloy::{primitives::BlockHash, rpc::types::Header, serde::WithOtherFields};

use crate::PoolLog;

pub type ClientHead = WithOtherFields<Header>;

/// The decoded pool logs of a single block, discovered by a ranged `eth_getLogs` query.
/// Added so a one-shot range scan can attribute pool logs to their block for bootstrap graph and
/// pool-log seeding without per-block log requests. Carries the full registry-free decoded set —
/// candidate keys are derived from it — so the seeded graph keeps every log a later-verified pool
/// may need (the range query is topics-only; completeness is independent of the pool set).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RangeLogBlock {
    pub number: u64,
    pub hash: BlockHash,
    pub logs: Vec<PoolLog>,
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
