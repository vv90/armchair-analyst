use alloy::rpc::types::Log;

#[derive(Clone, Debug, PartialEq)]
pub enum ClientEvent {
    Subscribed {
        subscription_id: String,
    },
    Notification {
        subscription_id: String,
        result: Log,
    },
    Closed,
}
