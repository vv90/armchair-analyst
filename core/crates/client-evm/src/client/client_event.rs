use alloy::primitives::Log;
use serde_json::Value;

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
