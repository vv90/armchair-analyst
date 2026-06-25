use std::fmt;

use thiserror::Error;

/// Which RPC endpoint configuration failed validation. Distinguishes the otherwise-identical
/// "missing url / api key" messages between the websocket subscription and the http request paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigScope {
    Subscription,
    Http,
    Graph,
}

impl fmt::Display for ConfigScope {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            ConfigScope::Subscription => "subscription",
            ConfigScope::Http => "http",
            ConfigScope::Graph => "graph",
        };
        formatter.write_str(label)
    }
}

#[derive(Debug, Error)]
pub enum ClientEvmError {
    #[error("invalid {scope} config: {reason}")]
    InvalidConfig { scope: ConfigScope, reason: String },

    #[error("websocket error: {0}")]
    WebSocketError(tungstenite::Error),

    /// A transport-level HTTP failure with no usable response — connection refused, TLS, timeout,
    /// DNS. Carries `ureq::Error`'s own message. Non-2xx responses are `HttpStatus`, not this.
    #[error("http transport error: {0}")]
    HttpTransport(ureq::Error),

    /// A non-2xx HTTP response. `body` is the (sanitized, truncated) response body, which usually
    /// carries the provider's real reason (response-size cap, execution timeout, upstream down,
    /// rate limit) — the detail ureq otherwise discards.
    #[error("http status {status}: {body}")]
    HttpStatus { status: u16, body: String },

    /// The provider returned a well-formed JSON-RPC `error` object. Often transient/retryable
    /// (e.g. a provider's "temporary internal error, please retry"). `code`/`message` are the
    /// provider's own values.
    #[error("json-rpc error {code}: {message}")]
    JsonRpcError { code: String, message: String },

    /// A response that reached us but did not match what we expected: a bad JSON-RPC envelope, a
    /// failed decode, or a value that violates an invariant. `context` names the request the
    /// response was for (e.g. "block header", "multicall3 batch"); `detail` is the specific reason,
    /// including actual offending values where available.
    #[error("malformed {context} response: {detail}")]
    MalformedResponse { context: String, detail: String },

    #[error("event receiver dropped")]
    EventReceiverDropped,
}
