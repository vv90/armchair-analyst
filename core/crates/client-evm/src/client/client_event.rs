use alloy::{
    rpc::types::{Header, Log},
    serde::WithOtherFields,
};

pub type ClientHead = WithOtherFields<Header>;

#[derive(Clone, Debug, PartialEq)]
pub enum ClientEvent {
    Subscribed {
        subscription_id: String,
    },
    Notification {
        subscription_id: String,
        result: Log,
    },
    NewHead {
        subscription_id: String,
        header: ClientHead,
    },
    Closed {
        subscription_id: String,
    },
}
