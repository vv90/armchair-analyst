use alloy::{
    primitives::{Address, BlockHash, Bytes, address},
    sol,
    sol_types::SolCall,
};
use serde_json::{Value, json};

use crate::ClientEvmError;

use super::client_utils::json_rpc_result;

pub(crate) const MULTICALL3_ADDRESS: Address = address!("cA11bde05977b3631167028862bE2a173976CA11");

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MulticallCall {
    pub(crate) target: Address,
    pub(crate) call_data: Bytes,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MulticallCallResult {
    pub(crate) success: bool,
    pub(crate) return_data: Bytes,
}

sol! {
    interface Multicall3 {
        #[derive(Debug, PartialEq, Eq)]
        struct Call3 {
            address target;
            bool allowFailure;
            bytes callData;
        }

        #[derive(Debug, PartialEq, Eq)]
        struct Result {
            bool success;
            bytes returnData;
        }

        function aggregate3(Call3[] calldata calls) external payable returns (Result[] memory returnData);
    }
}

pub(crate) fn build_multicall3_request(
    request_id: u64,
    at: BlockHash,
    calls: &[MulticallCall],
) -> Value {
    let call = Multicall3::aggregate3Call {
        calls: calls
            .iter()
            .map(|call| Multicall3::Call3 {
                target: call.target,
                allowFailure: true,
                callData: call.call_data.clone(),
            })
            .collect(),
    };

    json!({
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "eth_call",
        "params": [
            {
                "to": MULTICALL3_ADDRESS,
                "data": Bytes::from(call.abi_encode())
            },
            {
                "blockHash": at
            }
        ]
    })
}

pub(crate) fn parse_multicall3_response(
    value: &Value,
    expected_request_id: u64,
    expected_call_count: usize,
) -> Result<Vec<MulticallCallResult>, ClientEvmError> {
    let context = "multicall3";
    let result = json_rpc_result(value, expected_request_id, context)?;

    let return_data = serde_json::from_value::<Bytes>(result.clone()).map_err(|error| {
        ClientEvmError::MalformedResponse {
            context: context.to_owned(),
            detail: format!("result must decode as hex bytes: {error}"),
        }
    })?;
    let results =
        Multicall3::aggregate3Call::abi_decode_returns(return_data.as_ref()).map_err(|error| {
            ClientEvmError::MalformedResponse {
                context: context.to_owned(),
                detail: format!("result must decode as aggregate3 return data: {error}"),
            }
        })?;

    if results.len() != expected_call_count {
        return Err(ClientEvmError::MalformedResponse {
            context: context.to_owned(),
            detail: format!(
                "returned {} call results, expected {expected_call_count}",
                results.len()
            ),
        });
    }

    Ok(results
        .into_iter()
        .map(|result| MulticallCallResult {
            success: result.success,
            return_data: result.returnData,
        })
        .collect())
}

/// Builds a single JSON-RPC batch request carrying one `aggregate3` `eth_call` per chunk. Each
/// chunk is assigned a distinct id (`1..=chunks.len()`) so the response array can be matched back to
/// its chunk even if the provider reorders entries. Splitting a large call set into bounded chunks
/// keeps each individual `eth_call` under the node's response/gas limit; batching keeps the whole set
/// to one HTTP round-trip.
pub(crate) fn build_multicall3_batch_request(at: BlockHash, chunks: &[&[MulticallCall]]) -> Value {
    Value::Array(
        chunks
            .iter()
            .enumerate()
            .map(|(index, chunk)| build_multicall3_request(index as u64 + 1, at, chunk))
            .collect(),
    )
}

/// Parses a JSON-RPC batch response into the flat, in-chunk-order concatenation of every chunk's
/// call results. Entries are located by the deterministic per-chunk id (not array position, since
/// batch responses may be reordered), each delegating to `parse_multicall3_response` for validation.
/// A non-array body (whole-batch rejection), a wrong entry count, a missing id, or any entry-level
/// JSON-RPC error fails the whole parse — matching the all-or-nothing contract of the single call.
pub(crate) fn parse_multicall3_batch_response(
    value: &Value,
    expected_counts: &[usize],
) -> Result<Vec<MulticallCallResult>, ClientEvmError> {
    let entries = value
        .as_array()
        .ok_or_else(|| ClientEvmError::MalformedResponse {
            context: "multicall3 batch".to_owned(),
            detail: "response must be a json-rpc array".to_owned(),
        })?;

    if entries.len() != expected_counts.len() {
        return Err(ClientEvmError::MalformedResponse {
            context: "multicall3 batch".to_owned(),
            detail: format!(
                "returned {} entries, expected {}",
                entries.len(),
                expected_counts.len()
            ),
        });
    }

    let mut results = Vec::new();
    for (index, expected_count) in expected_counts.iter().enumerate() {
        let request_id = index as u64 + 1;
        let entry = entries
            .iter()
            .find(|entry| entry.get("id").and_then(Value::as_u64) == Some(request_id))
            .ok_or_else(|| ClientEvmError::MalformedResponse {
                context: "multicall3 batch".to_owned(),
                detail: format!("missing request id {request_id}"),
            })?;

        results.extend(parse_multicall3_response(
            entry,
            request_id,
            *expected_count,
        )?);
    }

    Ok(results)
}

#[cfg(test)]
pub(crate) fn aggregate3_return_data_for_test(results: &[MulticallCallResult]) -> Bytes {
    Bytes::from(Multicall3::aggregate3Call::abi_encode_returns(
        &results
            .iter()
            .map(|result| Multicall3::Result {
                success: result.success,
                returnData: result.return_data.clone(),
            })
            .collect::<Vec<_>>(),
    ))
}

#[cfg(test)]
pub(crate) fn decode_aggregate3_call_data_for_test(
    call_data: &Bytes,
) -> alloy::sol_types::Result<Vec<MulticallCall>> {
    let call = Multicall3::aggregate3Call::abi_decode(call_data.as_ref())?;

    Ok(call
        .calls
        .into_iter()
        .map(|call| MulticallCall {
            target: call.target,
            call_data: call.callData,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use alloy::primitives::B256;

    use super::*;

    #[test]
    fn multicall3_request_uses_single_eth_call_to_canonical_address_at_block_hash() {
        let at = B256::with_last_byte(7);
        let target = Address::with_last_byte(1);
        let call_data = Bytes::from(vec![0x12, 0x34]);
        let request = build_multicall3_request(
            9,
            at,
            &[MulticallCall {
                target,
                call_data: call_data.clone(),
            }],
        );

        assert_eq!(request.get("jsonrpc"), Some(&json!("2.0")));
        assert_eq!(request.get("id"), Some(&json!(9)));
        assert_eq!(request.get("method"), Some(&json!("eth_call")));
        assert_eq!(
            request
                .get("params")
                .and_then(Value::as_array)
                .and_then(|params| params.first())
                .and_then(|call| call.get("to")),
            Some(&json!(MULTICALL3_ADDRESS))
        );
        assert_eq!(
            request
                .get("params")
                .and_then(Value::as_array)
                .and_then(|params| params.get(1)),
            Some(&json!({ "blockHash": at }))
        );

        let encoded_call = request
            .get("params")
            .and_then(Value::as_array)
            .and_then(|params| params.first())
            .and_then(|call| call.get("data"))
            .cloned()
            .and_then(|data| serde_json::from_value::<Bytes>(data).ok())
            .expect("request must contain encoded multicall data");
        let decoded_call = Multicall3::aggregate3Call::abi_decode(encoded_call.as_ref())
            .expect("request data must decode as aggregate3 call");

        assert_eq!(
            decoded_call.calls,
            vec![Multicall3::Call3 {
                target,
                allowFailure: true,
                callData: call_data,
            }]
        );
    }

    #[test]
    fn multicall3_response_decodes_ordered_call_results() {
        let first_return_data = Bytes::from(vec![0xaa]);
        let second_return_data = Bytes::from(vec![0xbb, 0xcc]);
        let response = json!({
            "jsonrpc": "2.0",
            "id": 9,
            "result": aggregate3_return_data_for_test(&[
                MulticallCallResult {
                    success: true,
                    return_data: first_return_data.clone(),
                },
                MulticallCallResult {
                    success: false,
                    return_data: second_return_data.clone(),
                },
            ])
        });

        let result =
            parse_multicall3_response(&response, 9, 2).expect("multicall3 response must decode");

        assert_eq!(
            result,
            vec![
                MulticallCallResult {
                    success: true,
                    return_data: first_return_data,
                },
                MulticallCallResult {
                    success: false,
                    return_data: second_return_data,
                },
            ]
        );
    }

    #[test]
    fn multicall3_response_rejects_wrong_call_count() {
        let response = json!({
            "jsonrpc": "2.0",
            "id": 9,
            "result": aggregate3_return_data_for_test(&[])
        });

        assert!(matches!(
            parse_multicall3_response(&response, 9, 1),
            Err(ClientEvmError::MalformedResponse { .. })
        ));
    }

    #[test]
    fn multicall3_response_parses_json_rpc_error() {
        let response = json!({
            "jsonrpc": "2.0",
            "id": 9,
            "error": {
                "code": -32000,
                "message": "execution reverted"
            }
        });

        assert!(matches!(
            parse_multicall3_response(&response, 9, 1),
            Err(ClientEvmError::JsonRpcError { ref code, ref message })
                if code == "-32000" && message == "execution reverted"
        ));
    }

    #[test]
    fn multicall3_response_rejects_malformed_return_data() {
        let response = json!({
            "jsonrpc": "2.0",
            "id": 9,
            "result": Bytes::from(vec![0x12, 0x34])
        });

        assert!(matches!(
            parse_multicall3_response(&response, 9, 1),
            Err(ClientEvmError::MalformedResponse { .. })
        ));
    }

    fn call(last_byte: u8) -> MulticallCall {
        MulticallCall {
            target: Address::with_last_byte(last_byte),
            call_data: Bytes::from(vec![last_byte]),
        }
    }

    fn result(last_byte: u8) -> MulticallCallResult {
        MulticallCallResult {
            success: true,
            return_data: Bytes::from(vec![last_byte]),
        }
    }

    fn batch_entry(id: u64, results: &[MulticallCallResult]) -> Value {
        json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": aggregate3_return_data_for_test(results)
        })
    }

    #[test]
    fn batch_request_packs_one_eth_call_per_chunk_with_distinct_ids() {
        let at = B256::with_last_byte(7);
        let first = [call(1), call(2)];
        let second = [call(3)];
        let request = build_multicall3_batch_request(at, &[&first, &second]);

        let entries = request.as_array().expect("batch request must be an array");
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].get("id"), Some(&json!(1)));
        assert_eq!(entries[1].get("id"), Some(&json!(2)));

        for entry in entries {
            assert_eq!(entry.get("method"), Some(&json!("eth_call")));
            assert_eq!(
                entry
                    .get("params")
                    .and_then(Value::as_array)
                    .and_then(|params| params.first())
                    .and_then(|call| call.get("to")),
                Some(&json!(MULTICALL3_ADDRESS))
            );
            assert_eq!(
                entry
                    .get("params")
                    .and_then(Value::as_array)
                    .and_then(|params| params.get(1)),
                Some(&json!({ "blockHash": at }))
            );
        }

        let decoded_second = decode_aggregate3_call_data_for_test(
            &serde_json::from_value::<Bytes>(
                entries[1]
                    .get("params")
                    .and_then(Value::as_array)
                    .and_then(|params| params.first())
                    .and_then(|call| call.get("data"))
                    .cloned()
                    .expect("entry must carry call data"),
            )
            .expect("entry data must be bytes"),
        )
        .expect("entry data must decode as aggregate3");
        assert_eq!(decoded_second, second);
    }

    #[test]
    fn batch_response_flattens_chunk_results_in_chunk_order() {
        let response = json!([
            batch_entry(1, &[result(0xaa)]),
            batch_entry(2, &[result(0xbb), result(0xcc)]),
        ]);

        let results = parse_multicall3_batch_response(&response, &[1, 2])
            .expect("batch response must decode");

        assert_eq!(results, vec![result(0xaa), result(0xbb), result(0xcc)]);
    }

    #[test]
    fn batch_response_matches_entries_by_id_when_reordered() {
        let response = json!([
            batch_entry(2, &[result(0xbb), result(0xcc)]),
            batch_entry(1, &[result(0xaa)]),
        ]);

        let results = parse_multicall3_batch_response(&response, &[1, 2])
            .expect("reordered batch response must decode");

        assert_eq!(results, vec![result(0xaa), result(0xbb), result(0xcc)]);
    }

    #[test]
    fn batch_response_surfaces_entry_level_json_rpc_error() {
        let response = json!([
            batch_entry(1, &[result(0xaa)]),
            {
                "jsonrpc": "2.0",
                "id": 2,
                "error": { "code": -32000, "message": "execution reverted" }
            },
        ]);

        assert!(matches!(
            parse_multicall3_batch_response(&response, &[1, 1]),
            Err(ClientEvmError::JsonRpcError { ref code, ref message })
                if code == "-32000" && message == "execution reverted"
        ));
    }

    #[test]
    fn batch_response_rejects_missing_request_id() {
        let response = json!([
            batch_entry(1, &[result(0xaa)]),
            batch_entry(1, &[result(0xbb)]),
        ]);

        assert!(matches!(
            parse_multicall3_batch_response(&response, &[1, 1]),
            Err(ClientEvmError::MalformedResponse { .. })
        ));
    }

    #[test]
    fn batch_response_rejects_non_array_body() {
        let response = batch_entry(1, &[result(0xaa)]);

        assert!(matches!(
            parse_multicall3_batch_response(&response, &[1]),
            Err(ClientEvmError::MalformedResponse { .. })
        ));
    }

    #[test]
    fn batch_response_rejects_entry_count_mismatch() {
        let response = json!([batch_entry(1, &[result(0xaa)])]);

        assert!(matches!(
            parse_multicall3_batch_response(&response, &[1, 1]),
            Err(ClientEvmError::MalformedResponse { .. })
        ));
    }
}
