use std::collections::HashSet;

use alloy::{primitives::BlockHash, rpc::types::Log};
use serde_json::{Value, json};

use crate::{
    ClientEvmError, ClientHead, PoolCandidateAddress, uniswap_v3::pool_event_signature_hashes,
};

fn pool_event_topic_filter() -> Vec<String> {
    pool_event_signature_hashes()
        .into_iter()
        .map(|topic| topic.to_string())
        .collect::<Vec<_>>()
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn build_pool_events_subscribe_request(request_id: u64) -> Value {
    let event_topics = pool_event_topic_filter();

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

pub(crate) fn build_new_heads_subscribe_request(request_id: u64) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "eth_subscribe",
        "params": ["newHeads"]
    })
}

pub(crate) fn build_block_header_request(request_id: u64, block_hash: BlockHash) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "eth_getBlockByHash",
        "params": [block_hash, false]
    })
}

pub(crate) fn build_block_logs_request(request_id: u64, block_hash: BlockHash) -> Value {
    let event_topics = pool_event_topic_filter();

    json!({
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "eth_getLogs",
        "params": [{
            "blockHash": block_hash,
            "topics": [event_topics]
        }]
    })
}

pub(crate) fn build_finalized_block_header_request(request_id: u64) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "eth_getBlockByNumber",
        "params": ["finalized", false]
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

pub(crate) fn parse_block_header_response(
    value: &Value,
    expected_request_id: u64,
    expected_block_hash: BlockHash,
) -> Result<Option<ClientHead>, ClientEvmError> {
    let header = parse_block_header_response_by_id(value, expected_request_id)?;

    if let Some(header) = &header {
        if header.inner.hash != expected_block_hash {
            return Err(ClientEvmError::MalformedJsonRpcResponse(
                "returned block hash does not match requested block hash".to_owned(),
            ));
        }
    }

    Ok(header)
}

pub(crate) fn parse_block_header_response_by_id(
    value: &Value,
    expected_request_id: u64,
) -> Result<Option<ClientHead>, ClientEvmError> {
    if value.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
        return Err(ClientEvmError::MalformedJsonRpcResponse(
            "block header response must use json-rpc 2.0".to_owned(),
        ));
    }

    let response_id = value.get("id").and_then(Value::as_u64).ok_or_else(|| {
        ClientEvmError::MalformedJsonRpcResponse(
            "block header response must contain a numeric request id".to_owned(),
        )
    })?;

    if response_id != expected_request_id {
        return Err(ClientEvmError::MalformedJsonRpcResponse(
            "block header response request id does not match request".to_owned(),
        ));
    }

    match (value.get("result"), value.get("error")) {
        (Some(_), Some(_)) => Err(ClientEvmError::MalformedJsonRpcResponse(
            "block header response must not contain both result and error".to_owned(),
        )),
        (None, None) => Err(ClientEvmError::MalformedJsonRpcResponse(
            "block header response must contain result or error".to_owned(),
        )),
        (None, Some(error)) => Err(ClientEvmError::JsonRpcError(json_rpc_error_message(error))),
        (Some(Value::Null), None) => Ok(None),
        (Some(result), None) => {
            let header = serde_json::from_value::<ClientHead>(result.clone())
                .map_err(ClientEvmError::JsonError)?;

            Ok(Some(header))
        }
    }
}

pub(crate) fn parse_block_logs_response(
    value: &Value,
    expected_request_id: u64,
    expected_block_hash: BlockHash,
) -> Result<HashSet<PoolCandidateAddress>, ClientEvmError> {
    if value.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
        return Err(ClientEvmError::MalformedJsonRpcResponse(
            "block logs response must use json-rpc 2.0".to_owned(),
        ));
    }

    let response_id = value.get("id").and_then(Value::as_u64).ok_or_else(|| {
        ClientEvmError::MalformedJsonRpcResponse(
            "block logs response must contain a numeric request id".to_owned(),
        )
    })?;

    if response_id != expected_request_id {
        return Err(ClientEvmError::MalformedJsonRpcResponse(
            "block logs response request id does not match request".to_owned(),
        ));
    }

    match (value.get("result"), value.get("error")) {
        (Some(_), Some(_)) => Err(ClientEvmError::MalformedJsonRpcResponse(
            "block logs response must not contain both result and error".to_owned(),
        )),
        (None, None) => Err(ClientEvmError::MalformedJsonRpcResponse(
            "block logs response must contain result or error".to_owned(),
        )),
        (None, Some(error)) => Err(ClientEvmError::JsonRpcError(json_rpc_error_message(error))),
        (Some(result), None) => {
            if !result.is_array() {
                return Err(ClientEvmError::MalformedJsonRpcResponse(
                    "block logs response result must be an array".to_owned(),
                ));
            }

            let logs = serde_json::from_value::<Vec<Log>>(result.clone())
                .map_err(ClientEvmError::JsonError)?;

            if logs
                .iter()
                .any(|log| log.block_hash != Some(expected_block_hash))
            {
                return Err(ClientEvmError::MalformedJsonRpcResponse(
                    "returned log block hash does not match requested block hash".to_owned(),
                ));
            }

            Ok(logs
                .into_iter()
                .map(|log| PoolCandidateAddress(log.address()))
                .collect())
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use alloy::primitives::{Address, B256};
    use proptest::prelude::*;
    use serde_json::{Value, json};

    use crate::PoolCandidateAddress;

    use super::*;

    #[test]
    fn block_header_request_uses_hash_without_full_transactions() {
        let block_hash = B256::with_last_byte(7);

        let request = build_block_header_request(9, block_hash);

        assert_eq!(
            request,
            json!({
                "jsonrpc": "2.0",
                "id": 9,
                "method": "eth_getBlockByHash",
                "params": [block_hash, false]
            })
        );
    }

    #[test]
    fn finalized_block_header_request_uses_finalized_tag_without_full_transactions() {
        let request = build_finalized_block_header_request(9);

        assert_eq!(
            request,
            json!({
                "jsonrpc": "2.0",
                "id": 9,
                "method": "eth_getBlockByNumber",
                "params": ["finalized", false]
            })
        );
    }

    #[test]
    fn block_logs_request_uses_block_hash_and_pool_event_topics() {
        let block_hash = B256::with_last_byte(7);
        let request = build_block_logs_request(9, block_hash);
        let expected_topics = pool_event_signature_hashes()
            .into_iter()
            .map(|topic| topic.to_string())
            .collect::<Vec<_>>();

        assert_eq!(
            request,
            json!({
                "jsonrpc": "2.0",
                "id": 9,
                "method": "eth_getLogs",
                "params": [{
                    "blockHash": block_hash,
                    "topics": [expected_topics]
                }]
            })
        );
    }

    #[test]
    fn block_logs_request_does_not_filter_by_address() {
        let request = build_block_logs_request(9, B256::with_last_byte(7));
        let address_filter = request
            .get("params")
            .and_then(Value::as_array)
            .and_then(|params| params.first())
            .and_then(|filter| filter.get("address"));

        assert_eq!(address_filter, None);
    }

    #[test]
    fn block_logs_response_decodes_pool_candidate_addresses() {
        let block_hash = B256::with_last_byte(7);
        let first_pool = Address::with_last_byte(1);
        let second_pool = Address::with_last_byte(2);
        let response = json!({
            "jsonrpc": "2.0",
            "id": 9,
            "result": [
                log_result(first_pool, block_hash),
                log_result(second_pool, block_hash),
                log_result(first_pool, block_hash)
            ]
        });

        let result = parse_block_logs_response(&response, 9, block_hash);

        assert!(matches!(
            result,
            Ok(ref pools)
                if pools.len() == 2
                    && pools.contains(&PoolCandidateAddress(first_pool))
                    && pools.contains(&PoolCandidateAddress(second_pool))
        ));
    }

    #[test]
    fn block_logs_response_returns_empty_set_for_empty_result() {
        let block_hash = B256::with_last_byte(7);
        let response = json!({
            "jsonrpc": "2.0",
            "id": 9,
            "result": []
        });

        let result = parse_block_logs_response(&response, 9, block_hash);

        assert!(matches!(result, Ok(ref pools) if pools.is_empty()));
    }

    #[test]
    fn block_logs_response_parses_json_rpc_error() {
        let block_hash = B256::with_last_byte(7);
        let response = json!({
            "jsonrpc": "2.0",
            "id": 9,
            "error": {
                "code": -32000,
                "message": "logs unavailable"
            }
        });

        assert!(matches!(
            parse_block_logs_response(&response, 9, block_hash),
            Err(ClientEvmError::JsonRpcError(ref message))
                if message == "-32000: logs unavailable"
        ));
    }

    #[test]
    fn block_logs_response_rejects_unexpected_request_id() {
        let block_hash = B256::with_last_byte(7);
        let response = json!({
            "jsonrpc": "2.0",
            "id": 10,
            "result": []
        });

        assert!(matches!(
            parse_block_logs_response(&response, 9, block_hash),
            Err(ClientEvmError::MalformedJsonRpcResponse(_))
        ));
    }

    #[test]
    fn block_logs_response_rejects_missing_result_and_error() {
        let block_hash = B256::with_last_byte(7);
        let response = json!({
            "jsonrpc": "2.0",
            "id": 9
        });

        assert!(matches!(
            parse_block_logs_response(&response, 9, block_hash),
            Err(ClientEvmError::MalformedJsonRpcResponse(_))
        ));
    }

    #[test]
    fn block_logs_response_rejects_result_and_error_together() {
        let block_hash = B256::with_last_byte(7);
        let response = json!({
            "jsonrpc": "2.0",
            "id": 9,
            "result": [],
            "error": {
                "code": -32000,
                "message": "conflicting response"
            }
        });

        assert!(matches!(
            parse_block_logs_response(&response, 9, block_hash),
            Err(ClientEvmError::MalformedJsonRpcResponse(_))
        ));
    }

    #[test]
    fn block_logs_response_rejects_malformed_log() {
        let block_hash = B256::with_last_byte(7);
        let response = json!({
            "jsonrpc": "2.0",
            "id": 9,
            "result": [
                {
                    "address": Address::with_last_byte(1)
                }
            ]
        });

        assert!(matches!(
            parse_block_logs_response(&response, 9, block_hash),
            Err(ClientEvmError::JsonError(_))
        ));
    }

    #[test]
    fn block_logs_response_rejects_mismatched_block_hash() {
        let requested_hash = B256::with_last_byte(7);
        let returned_hash = B256::with_last_byte(8);
        let response = json!({
            "jsonrpc": "2.0",
            "id": 9,
            "result": [
                log_result(Address::with_last_byte(1), returned_hash)
            ]
        });

        assert!(matches!(
            parse_block_logs_response(&response, 9, requested_hash),
            Err(ClientEvmError::MalformedJsonRpcResponse(_))
        ));
    }

    #[test]
    fn block_logs_response_rejects_invalid_json_rpc_version() {
        let block_hash = B256::with_last_byte(7);
        let response = json!({
            "jsonrpc": "1.0",
            "id": 9,
            "result": []
        });

        assert!(matches!(
            parse_block_logs_response(&response, 9, block_hash),
            Err(ClientEvmError::MalformedJsonRpcResponse(_))
        ));
    }

    #[test]
    fn block_header_response_decodes_matching_header_and_extra_fields() {
        let block_hash = B256::with_last_byte(1);
        let response = json!({
            "jsonrpc": "2.0",
            "id": 9,
            "result": block_header_result(block_hash)
        });

        let result = parse_block_header_response(&response, 9, block_hash);

        assert!(matches!(
            result,
            Ok(Some(header))
                if header.inner.hash == block_hash
                    && header.inner.inner.parent_hash == B256::with_last_byte(2)
                    && header.inner.inner.number == 9
                    && matches!(
                        header.other.get_deserialized::<String>("providerTag"),
                        Some(Ok(ref tag)) if tag == "observed"
                    )
        ));
    }

    #[test]
    fn block_header_response_by_id_decodes_header_without_expected_hash() {
        let block_hash = B256::with_last_byte(1);
        let response = json!({
            "jsonrpc": "2.0",
            "id": 9,
            "result": block_header_result(block_hash)
        });

        let result = parse_block_header_response_by_id(&response, 9);

        assert!(matches!(
            result,
            Ok(Some(header))
                if header.inner.hash == block_hash
                    && header.inner.inner.parent_hash == B256::with_last_byte(2)
                    && header.inner.inner.number == 9
        ));
    }

    #[test]
    fn block_header_response_by_id_returns_none_for_null_result() {
        let response = json!({
            "jsonrpc": "2.0",
            "id": 9,
            "result": null
        });

        assert!(matches!(
            parse_block_header_response_by_id(&response, 9),
            Ok(None)
        ));
    }

    #[test]
    fn block_header_response_by_id_parses_json_rpc_error() {
        let response = json!({
            "jsonrpc": "2.0",
            "id": 9,
            "error": {
                "code": -32000,
                "message": "block unavailable"
            }
        });

        assert!(matches!(
            parse_block_header_response_by_id(&response, 9),
            Err(ClientEvmError::JsonRpcError(ref message))
                if message == "-32000: block unavailable"
        ));
    }

    #[test]
    fn block_header_response_by_id_rejects_unexpected_request_id() {
        let response = json!({
            "jsonrpc": "2.0",
            "id": 10,
            "result": block_header_result(B256::with_last_byte(1))
        });

        assert!(matches!(
            parse_block_header_response_by_id(&response, 9),
            Err(ClientEvmError::MalformedJsonRpcResponse(_))
        ));
    }

    #[test]
    fn block_header_response_by_id_rejects_malformed_response() {
        let response = json!({
            "jsonrpc": "2.0",
            "id": 9
        });

        assert!(matches!(
            parse_block_header_response_by_id(&response, 9),
            Err(ClientEvmError::MalformedJsonRpcResponse(_))
        ));
    }

    #[test]
    fn block_header_response_returns_none_for_null_result() {
        let block_hash = B256::with_last_byte(1);
        let response = json!({
            "jsonrpc": "2.0",
            "id": 9,
            "result": null
        });

        assert!(matches!(
            parse_block_header_response(&response, 9, block_hash),
            Ok(None)
        ));
    }

    #[test]
    fn block_header_response_parses_json_rpc_error() {
        let block_hash = B256::with_last_byte(1);
        let response = json!({
            "jsonrpc": "2.0",
            "id": 9,
            "error": {
                "code": -32000,
                "message": "block unavailable"
            }
        });

        assert!(matches!(
            parse_block_header_response(&response, 9, block_hash),
            Err(ClientEvmError::JsonRpcError(ref message))
                if message == "-32000: block unavailable"
        ));
    }

    #[test]
    fn block_header_response_rejects_unexpected_request_id() {
        let block_hash = B256::with_last_byte(1);
        let response = json!({
            "jsonrpc": "2.0",
            "id": 10,
            "result": block_header_result(block_hash)
        });

        assert!(matches!(
            parse_block_header_response(&response, 9, block_hash),
            Err(ClientEvmError::MalformedJsonRpcResponse(_))
        ));
    }

    #[test]
    fn block_header_response_rejects_missing_result_and_error() {
        let block_hash = B256::with_last_byte(1);
        let response = json!({
            "jsonrpc": "2.0",
            "id": 9
        });

        assert!(matches!(
            parse_block_header_response(&response, 9, block_hash),
            Err(ClientEvmError::MalformedJsonRpcResponse(_))
        ));
    }

    #[test]
    fn block_header_response_rejects_result_and_error_together() {
        let block_hash = B256::with_last_byte(1);
        let response = json!({
            "jsonrpc": "2.0",
            "id": 9,
            "result": block_header_result(block_hash),
            "error": {
                "code": -32000,
                "message": "conflicting response"
            }
        });

        assert!(matches!(
            parse_block_header_response(&response, 9, block_hash),
            Err(ClientEvmError::MalformedJsonRpcResponse(_))
        ));
    }

    #[test]
    fn block_header_response_rejects_malformed_header() {
        let block_hash = B256::with_last_byte(1);
        let response = json!({
            "jsonrpc": "2.0",
            "id": 9,
            "result": {
                "hash": block_hash
            }
        });

        assert!(matches!(
            parse_block_header_response(&response, 9, block_hash),
            Err(ClientEvmError::JsonError(_))
        ));
    }

    #[test]
    fn block_header_response_rejects_mismatched_hash() {
        let requested_hash = B256::with_last_byte(1);
        let returned_hash = B256::with_last_byte(2);
        let response = json!({
            "jsonrpc": "2.0",
            "id": 9,
            "result": block_header_result(returned_hash)
        });

        assert!(matches!(
            parse_block_header_response(&response, 9, requested_hash),
            Err(ClientEvmError::MalformedJsonRpcResponse(_))
        ));
    }

    #[test]
    fn block_header_response_rejects_invalid_json_rpc_version() {
        let block_hash = B256::with_last_byte(1);
        let response = json!({
            "jsonrpc": "1.0",
            "id": 9,
            "result": block_header_result(block_hash)
        });

        assert!(matches!(
            parse_block_header_response(&response, 9, block_hash),
            Err(ClientEvmError::MalformedJsonRpcResponse(_))
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
    fn new_heads_subscribe_request_uses_new_heads_subscription() {
        let request = build_new_heads_subscribe_request(9);

        assert_eq!(
            request,
            json!({
                "jsonrpc": "2.0",
                "id": 9,
                "method": "eth_subscribe",
                "params": ["newHeads"]
            })
        );
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
        fn block_header_request_preserves_request_id_and_hash(
            request_id in any::<u64>(),
            block_hash in any::<[u8; 32]>(),
        ) {
            let block_hash = B256::from(block_hash);
            let request = build_block_header_request(request_id, block_hash);

            prop_assert_eq!(request.get("id"), Some(&json!(request_id)));
            prop_assert_eq!(request.get("method"), Some(&json!("eth_getBlockByHash")));
            prop_assert_eq!(request.get("params"), Some(&json!([block_hash, false])));
        }

        #[test]
        fn block_logs_request_preserves_request_id_and_hash(
            request_id in any::<u64>(),
            block_hash in any::<[u8; 32]>(),
        ) {
            let block_hash = B256::from(block_hash);
            let request = build_block_logs_request(request_id, block_hash);

            prop_assert_eq!(request.get("id"), Some(&json!(request_id)));
            prop_assert_eq!(request.get("method"), Some(&json!("eth_getLogs")));
            prop_assert_eq!(
                request
                    .get("params")
                    .and_then(Value::as_array)
                    .and_then(|params| params.first())
                    .and_then(|filter| filter.get("blockHash")),
                Some(&json!(block_hash))
            );
        }

        #[test]
        fn finalized_block_header_request_preserves_request_id(request_id in any::<u64>()) {
            let request = build_finalized_block_header_request(request_id);

            prop_assert_eq!(request.get("id"), Some(&json!(request_id)));
            prop_assert_eq!(request.get("method"), Some(&json!("eth_getBlockByNumber")));
            prop_assert_eq!(request.get("params"), Some(&json!(["finalized", false])));
        }

        #[test]
        fn subscribe_request_preserves_request_id(request_id in any::<u64>()) {
            let request = build_pool_events_subscribe_request(request_id);

            prop_assert_eq!(request.get("id"), Some(&json!(request_id)));
        }

        #[test]
        fn new_heads_subscribe_request_preserves_request_id(request_id in any::<u64>()) {
            let request = build_new_heads_subscribe_request(request_id);

            prop_assert_eq!(request.get("id"), Some(&json!(request_id)));
            prop_assert_eq!(request.get("method"), Some(&json!("eth_subscribe")));
            prop_assert_eq!(request.get("params"), Some(&json!(["newHeads"])));
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

    fn log_result(address: Address, block_hash: B256) -> Value {
        json!({
            "address": address,
            "topics": [
                pool_event_signature_hashes()[0]
            ],
            "data": "0x",
            "blockHash": block_hash,
            "blockNumber": "0x4",
            "transactionHash": B256::with_last_byte(5),
            "transactionIndex": "0x6",
            "logIndex": "0x7",
            "removed": false
        })
    }

    fn block_header_result(block_hash: B256) -> Value {
        json!({
            "hash": block_hash,
            "parentHash": B256::with_last_byte(2),
            "sha3Uncles": B256::with_last_byte(3),
            "miner": "0x0000000000000000000000000000000000000004",
            "stateRoot": B256::with_last_byte(5),
            "transactionsRoot": B256::with_last_byte(6),
            "receiptsRoot": B256::with_last_byte(7),
            "logsBloom": format!("0x{}", "00".repeat(256)),
            "difficulty": "0xd",
            "number": "0x9",
            "gasLimit": "0xb",
            "gasUsed": "0xa",
            "timestamp": "0xc",
            "extraData": "0x010203",
            "mixHash": B256::with_last_byte(14),
            "nonce": "0x000000000000000f",
            "providerTag": "observed"
        })
    }
}
