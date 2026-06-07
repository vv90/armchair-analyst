use alloy::{rpc::types::Header, serde::WithOtherFields};

pub type ClientHead = WithOtherFields<Header>;

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
