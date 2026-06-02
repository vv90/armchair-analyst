use thiserror::Error;

#[derive(Debug, Error)]
pub enum ClientEvmError {
    #[error("invalid subscription config: {0}")]
    InvalidSubscriptionConfig(String),

    #[error("websocket error: {0}")]
    WebSocketError(tungstenite::Error),

    #[error("json error: {0}")]
    JsonError(serde_json::Error),

    #[error("json-rpc error: {0}")]
    JsonRpcError(String),

    #[error("malformed json-rpc response: {0}")]
    MalformedJsonRpcResponse(String),

    #[error("event receiver dropped")]
    EventReceiverDropped,
}
