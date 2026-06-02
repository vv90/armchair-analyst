use serde_json::{Value, json};

use crate::{ClientEvmError, EvmNetwork, RpcConfig, uniswap_v3::pool_event_signature_hashes};

pub(crate) fn compose_ws_endpoint(config: &RpcConfig) -> Result<String, ClientEvmError> {
    let ws_url = config.ws_url.trim();
    if ws_url.is_empty() {
        return Err(ClientEvmError::InvalidSubscriptionConfig(
            "websocket url is required".to_owned(),
        ));
    }

    let api_key = config.api_key.trim();
    if api_key.is_empty() {
        return Err(ClientEvmError::InvalidSubscriptionConfig(
            "rpc api key is required".to_owned(),
        ));
    }

    Ok(format!(
        "{}/{}/{}",
        ws_url.trim_end_matches('/'),
        network_path(config.network),
        api_key
    ))
}

fn network_path(network: EvmNetwork) -> &'static str {
    match network {
        EvmNetwork::Ethereum => "ethereum",
    }
}

pub(crate) fn build_pool_events_subscribe_request(request_id: u64) -> Value {
    let event_topics = pool_event_signature_hashes()
        .into_iter()
        .map(|topic| topic.to_string())
        .collect::<Vec<_>>();

    json!({
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "eth_subscribe",
        "params": [
            "logs",
            {
                "topics": [event_topics]
            }
        ]
    })
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn build_unsubscribe_request(request_id: u64, subscription_id: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "eth_unsubscribe",
        "params": [subscription_id]
    })
}

pub(crate) fn json_rpc_error_message(error: &Value) -> String {
    let code = error
        .get("code")
        .map_or_else(|| "unknown".to_owned(), Value::to_string);
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("unknown error");

    format!("{code}: {message}")
}

pub(crate) fn parse_subscription_response(
    value: &Value,
    expected_request_id: u64,
) -> Result<Option<String>, ClientEvmError> {
    let Some(response_id) = value.get("id").and_then(Value::as_u64) else {
        return Ok(None);
    };

    if response_id != expected_request_id {
        return Ok(None);
    }

    if let Some(error) = value.get("error") {
        return Err(ClientEvmError::JsonRpcError(json_rpc_error_message(error)));
    }

    value
        .get("result")
        .and_then(Value::as_str)
        .filter(|subscription_id| !subscription_id.trim().is_empty())
        .map(|subscription_id| Some(subscription_id.to_owned()))
        .ok_or_else(|| {
            ClientEvmError::MalformedJsonRpcResponse(
                "subscription response result must be a non-empty string".to_owned(),
            )
        })
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use proptest::prelude::*;
    use serde_json::{Value, json};

    use super::*;

    #[test]
    fn compose_ws_endpoint_appends_network_and_key() {
        let config = rpc_config(" wss://lb.drpc.org/ ", " api-key ");

        let result = compose_ws_endpoint(&config);

        assert!(matches!(
            result.as_deref(),
            Ok("wss://lb.drpc.org/ethereum/api-key")
        ));
    }

    #[test]
    fn compose_ws_endpoint_rejects_empty_ws_url() {
        let config = rpc_config(" ", "api-key");

        assert!(matches!(
            compose_ws_endpoint(&config),
            Err(ClientEvmError::InvalidSubscriptionConfig(_))
        ));
    }

    #[test]
    fn compose_ws_endpoint_rejects_empty_api_key() {
        let config = rpc_config("wss://lb.drpc.org", "\t");

        assert!(matches!(
            compose_ws_endpoint(&config),
            Err(ClientEvmError::InvalidSubscriptionConfig(_))
        ));
    }

    #[test]
    fn subscribe_request_uses_topic_only_pool_event_filter() {
        let request = build_pool_events_subscribe_request(7);
        let expected_topics = pool_event_signature_hashes()
            .into_iter()
            .map(|topic| topic.to_string())
            .collect::<Vec<_>>();

        assert_eq!(request.get("jsonrpc"), Some(&json!("2.0")));
        assert_eq!(request.get("id"), Some(&json!(7)));
        assert_eq!(request.get("method"), Some(&json!("eth_subscribe")));
        assert_eq!(
            request.get("params"),
            Some(&json!([
                "logs",
                {
                    "topics": [expected_topics]
                }
            ]))
        );
    }

    #[test]
    fn subscribe_request_does_not_filter_by_address() {
        let request = build_pool_events_subscribe_request(7);
        let address_filter = request
            .get("params")
            .and_then(Value::as_array)
            .and_then(|params| params.get(1))
            .and_then(|filter| filter.get("address"));

        assert_eq!(address_filter, None);
    }

    #[test]
    fn subscribe_request_includes_unique_pool_event_topics() {
        let request = build_pool_events_subscribe_request(7);
        let unique_topic_count = request
            .get("params")
            .and_then(Value::as_array)
            .and_then(|params| params.get(1))
            .and_then(|filter| filter.get("topics"))
            .and_then(Value::as_array)
            .and_then(|topic_groups| topic_groups.first())
            .and_then(Value::as_array)
            .map(|topics| topics.iter().collect::<HashSet<_>>().len());

        assert_eq!(unique_topic_count, Some(9));
    }

    #[test]
    fn unsubscribe_request_uses_subscription_id_param() {
        let request = build_unsubscribe_request(8, "0xsubscription");

        assert_eq!(
            request,
            json!({
                "jsonrpc": "2.0",
                "id": 8,
                "method": "eth_unsubscribe",
                "params": ["0xsubscription"]
            })
        );
    }

    #[test]
    fn subscription_response_parses_json_rpc_result() {
        let response = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": "0xsubscription"
        });

        let result = parse_subscription_response(&response, 1);

        assert!(matches!(
            result,
            Ok(Some(ref subscription_id)) if subscription_id == "0xsubscription"
        ));
    }

    #[test]
    fn subscription_response_parses_json_rpc_error() {
        let response = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "error": {
                "code": -32000,
                "message": "subscription failed"
            }
        });

        let result = parse_subscription_response(&response, 1);

        assert!(matches!(
            result,
            Err(ClientEvmError::JsonRpcError(ref message))
                if message == "-32000: subscription failed"
        ));
    }

    #[test]
    fn subscription_response_rejects_missing_result_and_error() {
        let response = json!({
            "jsonrpc": "2.0",
            "id": 1
        });

        assert!(matches!(
            parse_subscription_response(&response, 1),
            Err(ClientEvmError::MalformedJsonRpcResponse(_))
        ));
    }

    #[test]
    fn subscription_response_rejects_non_string_result() {
        let response = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": 3
        });

        assert!(matches!(
            parse_subscription_response(&response, 1),
            Err(ClientEvmError::MalformedJsonRpcResponse(_))
        ));
    }

    #[test]
    fn subscription_response_rejects_blank_subscription_id() {
        for blank in ["", " ", "\t"] {
            let response = json!({
                "jsonrpc": "2.0",
                "id": 1,
                "result": blank
            });

            assert!(matches!(
                parse_subscription_response(&response, 1),
                Err(ClientEvmError::MalformedJsonRpcResponse(_))
            ));
        }
    }

    #[test]
    fn subscription_response_ignores_unexpected_request_id() {
        let response = json!({
            "jsonrpc": "2.0",
            "id": 2,
            "result": "0xsubscription"
        });

        let result = parse_subscription_response(&response, 1);

        assert!(matches!(result, Ok(None)));
    }

    proptest! {
        #[test]
        fn compose_ws_endpoint_preserves_non_empty_parts(
            ws_url in "[a-z]{3,10}://[a-z0-9.-]{1,30}/*",
            api_key in "[A-Za-z0-9_-]{1,40}",
        ) {
            let config = rpc_config(
                &format!(" {ws_url} "),
                &format!(" {api_key} "),
            );
            let expected = format!(
                "{}/ethereum/{}",
                ws_url.trim_end_matches('/'),
                api_key
            );

            prop_assert_eq!(compose_ws_endpoint(&config)?, expected);
        }

        #[test]
        fn compose_ws_endpoint_rejects_whitespace_ws_url(
            whitespace in "[ \\t\\n\\r]{1,12}",
            api_key in "[A-Za-z0-9_-]{1,40}",
        ) {
            let config = rpc_config(&whitespace, &api_key);

            prop_assert!(matches!(
                compose_ws_endpoint(&config),
                Err(ClientEvmError::InvalidSubscriptionConfig(_))
            ));
        }

        #[test]
        fn compose_ws_endpoint_rejects_whitespace_api_key(
            ws_url in "[a-z]{3,10}://[a-z0-9.-]{1,30}",
            whitespace in "[ \\t\\n\\r]{1,12}",
        ) {
            let config = rpc_config(&ws_url, &whitespace);

            prop_assert!(matches!(
                compose_ws_endpoint(&config),
                Err(ClientEvmError::InvalidSubscriptionConfig(_))
            ));
        }

        #[test]
        fn compose_ws_endpoint_ignores_http_url(
            http_url in "\\PC*",
            ws_url in "[a-z]{3,10}://[a-z0-9.-]{1,30}/*",
            api_key in "[A-Za-z0-9_-]{1,40}",
        ) {
            let mut config = rpc_config(&ws_url, &api_key);
            let expected = compose_ws_endpoint(&config)?;
            config.http_url = http_url;

            prop_assert_eq!(compose_ws_endpoint(&config)?, expected);
        }

        #[test]
        fn subscribe_request_preserves_request_id(request_id in any::<u64>()) {
            let request = build_pool_events_subscribe_request(request_id);

            prop_assert_eq!(request.get("id"), Some(&json!(request_id)));
        }

        #[test]
        fn subscribe_request_always_uses_topic_only_pool_event_filter(request_id in any::<u64>()) {
            let request = build_pool_events_subscribe_request(request_id);
            let expected_topics = pool_event_signature_hashes()
                .into_iter()
                .map(|topic| topic.to_string())
                .collect::<Vec<_>>();

            prop_assert_eq!(request.get("method"), Some(&json!("eth_subscribe")));
            prop_assert_eq!(
                request.get("params"),
                Some(&json!([
                    "logs",
                    {
                        "topics": [expected_topics]
                    }
                ]))
            );
            prop_assert_eq!(
                request
                    .get("params")
                    .and_then(Value::as_array)
                    .and_then(|params| params.get(1))
                    .and_then(|filter| filter.get("address")),
                None
            );
        }

        #[test]
        fn unsubscribe_request_preserves_id_and_subscription(
            request_id in any::<u64>(),
            subscription_id in "\\PC*",
        ) {
            let request = build_unsubscribe_request(request_id, &subscription_id);

            prop_assert_eq!(request.get("id"), Some(&json!(request_id)));
            prop_assert_eq!(request.get("method"), Some(&json!("eth_unsubscribe")));
            prop_assert_eq!(request.get("params"), Some(&json!([subscription_id])));
        }

    }

    fn rpc_config(ws_url: &str, api_key: &str) -> RpcConfig {
        RpcConfig {
            network: EvmNetwork::Ethereum,
            http_url: "https://lb.drpc.live/".to_owned(),
            ws_url: ws_url.to_owned(),
            api_key: api_key.to_owned(),
        }
    }
}
