use alloy::{
    primitives::{Address, BlockHash, Bytes, address},
    sol,
    sol_types::SolCall,
};
use serde_json::{Value, json};

use crate::ClientEvmError;

use super::client_utils::json_rpc_error_message;

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
    if value.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
        return Err(ClientEvmError::MalformedJsonRpcResponse(
            "multicall3 response must use json-rpc 2.0".to_owned(),
        ));
    }

    let response_id = value.get("id").and_then(Value::as_u64).ok_or_else(|| {
        ClientEvmError::MalformedJsonRpcResponse(
            "multicall3 response must contain a numeric request id".to_owned(),
        )
    })?;

    if response_id != expected_request_id {
        return Err(ClientEvmError::MalformedJsonRpcResponse(
            "multicall3 response request id does not match request".to_owned(),
        ));
    }

    match (value.get("result"), value.get("error")) {
        (Some(_), Some(_)) => Err(ClientEvmError::MalformedJsonRpcResponse(
            "multicall3 response must not contain both result and error".to_owned(),
        )),
        (None, None) => Err(ClientEvmError::MalformedJsonRpcResponse(
            "multicall3 response must contain result or error".to_owned(),
        )),
        (None, Some(error)) => Err(ClientEvmError::JsonRpcError(json_rpc_error_message(error))),
        (Some(result), None) => {
            let return_data = serde_json::from_value::<Bytes>(result.clone())
                .map_err(ClientEvmError::JsonError)?;
            let results = Multicall3::aggregate3Call::abi_decode_returns(return_data.as_ref())
                .map_err(|error| {
                    ClientEvmError::MalformedJsonRpcResponse(format!(
                        "multicall3 response result must decode as aggregate3 return data: {error}"
                    ))
                })?;

            if results.len() != expected_call_count {
                return Err(ClientEvmError::MalformedJsonRpcResponse(format!(
                    "multicall3 response returned {} call results, expected {expected_call_count}",
                    results.len()
                )));
            }

            Ok(results
                .into_iter()
                .map(|result| MulticallCallResult {
                    success: result.success,
                    return_data: result.returnData,
                })
                .collect())
        }
    }
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
            Err(ClientEvmError::MalformedJsonRpcResponse(_))
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
            Err(ClientEvmError::JsonRpcError(ref message))
                if message == "-32000: execution reverted"
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
            Err(ClientEvmError::MalformedJsonRpcResponse(_))
        ));
    }
}
