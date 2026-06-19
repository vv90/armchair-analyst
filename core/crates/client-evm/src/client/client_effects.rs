use std::{
    collections::{HashMap, HashSet},
    net::TcpStream,
    str,
    sync::mpsc::Sender,
};

use alloy::{
    primitives::{Address, BlockHash, Bytes, U256, Uint},
    sol_types::SolCall,
};
use serde::Deserialize;
use serde_json::Value;
use tungstenite::{Message, WebSocket, connect, stream::MaybeTlsStream};

use crate::{
    ChainKey, ClientEvmError, PoolAddress, PoolCandidateAddress, PoolDataCall, PoolDataFailure,
    PoolDataResult, PoolMetadata, PoolMetadataCall, PoolMetadataFailure, PoolMetadataResult,
    PoolState, RangeLogBlock, RpcConfig, TokenAddress, TokenDecimals, TokenMetadata,
    TokenMetadataCall, TokenMetadataFailure, TokenMetadataResult, UniswapV3Fee,
    config::{compose_http_endpoint, compose_ws_endpoint},
};

use super::{
    ClientEvent, ClientHead,
    client_utils::{
        build_block_header_request, build_block_logs_request, build_finalized_block_header_request,
        build_new_heads_subscribe_request, build_pool_logs_range_request,
        parse_block_header_response, parse_block_header_response_by_id, parse_block_logs_response,
        parse_pool_logs_range_response, parse_subscription_response,
    },
    multicall3::{
        MulticallCall, MulticallCallResult, build_multicall3_request, parse_multicall3_response,
    },
};

const HTTP_REQUEST_ID: u64 = 1;
const SUBSCRIBE_REQUEST_ID: u64 = 1;

type BlockingWebSocket = WebSocket<MaybeTlsStream<TcpStream>>;

#[derive(Debug, Deserialize)]
struct SubscriptionDataParams<T> {
    subscription: String,
    result: T,
}

#[derive(Debug, Deserialize)]
struct SubscriptionNotification<T> {
    params: SubscriptionDataParams<T>,
}

pub fn fetch_block_header(
    agent: &ureq::Agent,
    config: &RpcConfig,
    chain: ChainKey,
    block_hash: BlockHash,
) -> Result<Option<ClientHead>, ClientEvmError> {
    let endpoint = compose_http_endpoint(config, chain)?;
    let request = build_block_header_request(HTTP_REQUEST_ID, block_hash);
    let mut response = agent
        .post(endpoint.as_str())
        .send_json(&request)
        .map_err(ClientEvmError::HttpError)?;
    let response_value = response
        .body_mut()
        .read_json::<Value>()
        .map_err(ClientEvmError::HttpError)?;

    parse_block_header_response(&response_value, HTTP_REQUEST_ID, block_hash)
}

pub fn fetch_block_logs(
    agent: &ureq::Agent,
    config: &RpcConfig,
    chain: ChainKey,
    block_hash: BlockHash,
) -> Result<HashSet<PoolCandidateAddress>, ClientEvmError> {
    let endpoint = compose_http_endpoint(config, chain)?;
    let request = build_block_logs_request(HTTP_REQUEST_ID, block_hash);
    let mut response = agent
        .post(endpoint.as_str())
        .send_json(&request)
        .map_err(ClientEvmError::HttpError)?;
    let response_value = response
        .body_mut()
        .read_json::<Value>()
        .map_err(ClientEvmError::HttpError)?;

    parse_block_logs_response(&response_value, HTTP_REQUEST_ID, block_hash)
}

pub fn fetch_pool_candidates_in_range(
    agent: &ureq::Agent,
    config: &RpcConfig,
    chain: ChainKey,
    from_block: u64,
) -> Result<Vec<RangeLogBlock>, ClientEvmError> {
    let endpoint = compose_http_endpoint(config, chain)?;
    let request = build_pool_logs_range_request(HTTP_REQUEST_ID, from_block);
    let mut response = agent
        .post(endpoint.as_str())
        .send_json(&request)
        .map_err(ClientEvmError::HttpError)?;
    let response_value = response
        .body_mut()
        .read_json::<Value>()
        .map_err(ClientEvmError::HttpError)?;

    parse_pool_logs_range_response(&response_value, HTTP_REQUEST_ID)
}

pub fn fetch_pool_data(
    agent: &ureq::Agent,
    config: &RpcConfig,
    chain: ChainKey,
    at: BlockHash,
    pools: HashSet<PoolAddress>,
) -> Result<HashMap<PoolAddress, PoolDataResult>, ClientEvmError> {
    if pools.is_empty() {
        return Ok(HashMap::new());
    }

    let endpoint = compose_http_endpoint(config, chain)?;
    let pools = sorted_pool_addresses(pools);
    let calls = pool_data_multicall_calls(&pools);
    let request = build_multicall3_request(HTTP_REQUEST_ID, at, &calls);
    let mut response = agent
        .post(endpoint.as_str())
        .send_json(&request)
        .map_err(ClientEvmError::HttpError)?;
    let response_value = response
        .body_mut()
        .read_json::<Value>()
        .map_err(ClientEvmError::HttpError)?;
    let results = parse_multicall3_response(&response_value, HTTP_REQUEST_ID, calls.len())?;

    Ok(decode_pool_data_results(&pools, &results))
}

pub fn fetch_pool_metadata(
    agent: &ureq::Agent,
    config: &RpcConfig,
    chain: ChainKey,
    at: BlockHash,
    candidates: HashSet<PoolCandidateAddress>,
) -> Result<HashMap<PoolCandidateAddress, PoolMetadataResult>, ClientEvmError> {
    if candidates.is_empty() {
        return Ok(HashMap::new());
    }

    let endpoint = compose_http_endpoint(config, chain)?;
    let candidates = sorted_pool_candidate_addresses(candidates);
    let calls = pool_metadata_candidate_multicall_calls(&candidates);
    let request = build_multicall3_request(HTTP_REQUEST_ID, at, &calls);
    let mut response = agent
        .post(endpoint.as_str())
        .send_json(&request)
        .map_err(ClientEvmError::HttpError)?;
    let response_value = response
        .body_mut()
        .read_json::<Value>()
        .map_err(ClientEvmError::HttpError)?;
    let results = parse_multicall3_response(&response_value, HTTP_REQUEST_ID, calls.len())?;
    let (mut metadata_results, factory_inputs) =
        decode_pool_metadata_candidate_results(&candidates, &results);

    if factory_inputs.is_empty() {
        return Ok(metadata_results);
    }

    let factory_calls = pool_metadata_factory_multicall_calls(&factory_inputs);
    let request = build_multicall3_request(HTTP_REQUEST_ID, at, &factory_calls);
    let mut response = agent
        .post(endpoint.as_str())
        .send_json(&request)
        .map_err(ClientEvmError::HttpError)?;
    let response_value = response
        .body_mut()
        .read_json::<Value>()
        .map_err(ClientEvmError::HttpError)?;
    let factory_results =
        parse_multicall3_response(&response_value, HTTP_REQUEST_ID, factory_calls.len())?;

    metadata_results.extend(decode_pool_metadata_factory_results(
        &factory_inputs,
        &factory_results,
    ));

    Ok(metadata_results)
}

pub fn fetch_token_metadata(
    agent: &ureq::Agent,
    config: &RpcConfig,
    chain: ChainKey,
    at: BlockHash,
    tokens: HashSet<TokenAddress>,
) -> Result<HashMap<TokenAddress, TokenMetadataResult>, ClientEvmError> {
    if tokens.is_empty() {
        return Ok(HashMap::new());
    }

    let endpoint = compose_http_endpoint(config, chain)?;
    let tokens = sorted_token_addresses(tokens);
    let calls = token_metadata_multicall_calls(&tokens);
    let request = build_multicall3_request(HTTP_REQUEST_ID, at, &calls);
    let mut response = agent
        .post(endpoint.as_str())
        .send_json(&request)
        .map_err(ClientEvmError::HttpError)?;
    let response_value = response
        .body_mut()
        .read_json::<Value>()
        .map_err(ClientEvmError::HttpError)?;
    let results = parse_multicall3_response(&response_value, HTTP_REQUEST_ID, calls.len())?;

    Ok(decode_token_metadata_results(&tokens, &results))
}

fn sorted_pool_candidate_addresses(
    candidates: HashSet<PoolCandidateAddress>,
) -> Vec<PoolCandidateAddress> {
    let mut candidates = candidates.into_iter().collect::<Vec<_>>();
    candidates.sort_by(|left, right| left.0.cmp(&right.0));
    candidates
}

fn sorted_token_addresses(tokens: HashSet<TokenAddress>) -> Vec<TokenAddress> {
    let mut tokens = tokens.into_iter().collect::<Vec<_>>();
    tokens.sort_by(|left, right| left.0.cmp(&right.0));
    tokens
}

fn token_metadata_multicall_calls(tokens: &[TokenAddress]) -> Vec<MulticallCall> {
    tokens
        .iter()
        .map(|token| MulticallCall {
            target: token.0,
            call_data: Bytes::from(crate::erc20::decimalsCall {}.abi_encode()),
        })
        .collect()
}

fn decode_token_metadata_results(
    tokens: &[TokenAddress],
    results: &[MulticallCallResult],
) -> HashMap<TokenAddress, TokenMetadataResult> {
    tokens
        .iter()
        .enumerate()
        .map(|(index, token)| (*token, decode_token_metadata_result(results.get(index))))
        .collect()
}

fn decode_token_metadata_result(result: Option<&MulticallCallResult>) -> TokenMetadataResult {
    let decimals = decode_token_decimals(result)?;
    let decimals = TokenDecimals::try_from_u256(decimals)?;

    Ok(TokenMetadata { decimals })
}

fn decode_token_decimals(
    result: Option<&MulticallCallResult>,
) -> Result<U256, TokenMetadataFailure> {
    decode_token_metadata_multicall_result(result, TokenMetadataCall::Decimals, |return_data| {
        crate::erc20::decimalsCall::abi_decode_returns(return_data)
    })
}

fn decode_token_metadata_multicall_result<T>(
    result: Option<&MulticallCallResult>,
    call: TokenMetadataCall,
    decode: impl FnOnce(&[u8]) -> alloy::sol_types::Result<T>,
) -> Result<T, TokenMetadataFailure> {
    match result {
        None => Err(TokenMetadataFailure::MissingResponse(call)),
        Some(result) if !result.success => Err(TokenMetadataFailure::CallFailed(call)),
        Some(result) => decode(result.return_data.as_ref())
            .map_err(|_| TokenMetadataFailure::DecodeFailed(call)),
    }
}

fn pool_metadata_candidate_multicall_calls(
    candidates: &[PoolCandidateAddress],
) -> Vec<MulticallCall> {
    candidates
        .iter()
        .flat_map(|candidate| {
            [
                MulticallCall {
                    target: candidate.0,
                    call_data: Bytes::from(crate::uniswap_v3::token0Call {}.abi_encode()),
                },
                MulticallCall {
                    target: candidate.0,
                    call_data: Bytes::from(crate::uniswap_v3::token1Call {}.abi_encode()),
                },
                MulticallCall {
                    target: candidate.0,
                    call_data: Bytes::from(crate::uniswap_v3::feeCall {}.abi_encode()),
                },
            ]
        })
        .collect()
}

fn pool_metadata_factory_multicall_calls(
    metadata: &[(PoolCandidateAddress, PoolMetadata)],
) -> Vec<MulticallCall> {
    metadata
        .iter()
        .map(|(_, metadata)| MulticallCall {
            target: crate::uniswap_v3::ETHEREUM_UNISWAP_V3_FACTORY_ADDRESS,
            call_data: Bytes::from(
                crate::uniswap_v3::getPoolCall {
                    tokenA: metadata.token0,
                    tokenB: metadata.token1,
                    fee: Uint::<24, 1>::from(metadata.fee.pips()),
                }
                .abi_encode(),
            ),
        })
        .collect()
}

fn decode_pool_metadata_candidate_results(
    candidates: &[PoolCandidateAddress],
    results: &[MulticallCallResult],
) -> (
    HashMap<PoolCandidateAddress, PoolMetadataResult>,
    Vec<(PoolCandidateAddress, PoolMetadata)>,
) {
    let mut result_chunks = results.chunks(3);
    let mut metadata_results = HashMap::new();
    let mut factory_inputs = Vec::new();

    for candidate in candidates {
        let result_chunk = result_chunks.next().unwrap_or(&[]);

        match decode_pool_metadata_candidate_result(
            result_chunk.first(),
            result_chunk.get(1),
            result_chunk.get(2),
        ) {
            Ok(metadata) => factory_inputs.push((*candidate, metadata)),
            Err(error) => {
                metadata_results.insert(*candidate, Err(error));
            }
        }
    }

    (metadata_results, factory_inputs)
}

fn decode_pool_metadata_candidate_result(
    token0: Option<&MulticallCallResult>,
    token1: Option<&MulticallCallResult>,
    fee: Option<&MulticallCallResult>,
) -> PoolMetadataResult {
    let token0 = decode_pool_metadata_token0(token0)?;
    let token1 = decode_pool_metadata_token1(token1)?;
    let fee_pips = decode_pool_metadata_fee(fee)?;
    let fee = UniswapV3Fee::try_from_pips(fee_pips)
        .ok_or(PoolMetadataFailure::UnsupportedFee(fee_pips))?;

    Ok(PoolMetadata {
        token0,
        token1,
        fee,
    })
}

fn decode_pool_metadata_factory_results(
    metadata: &[(PoolCandidateAddress, PoolMetadata)],
    results: &[MulticallCallResult],
) -> HashMap<PoolCandidateAddress, PoolMetadataResult> {
    metadata
        .iter()
        .enumerate()
        .map(|(index, (candidate, metadata))| {
            (
                *candidate,
                decode_pool_metadata_factory_result(*candidate, metadata, results.get(index)),
            )
        })
        .collect()
}

fn decode_pool_metadata_factory_result(
    candidate: PoolCandidateAddress,
    metadata: &PoolMetadata,
    result: Option<&MulticallCallResult>,
) -> PoolMetadataResult {
    let returned = decode_pool_metadata_factory_pool(result)?;

    if returned == Address::ZERO {
        Err(PoolMetadataFailure::FactoryReturnedZero)
    } else if returned != candidate.0 {
        Err(PoolMetadataFailure::FactoryMismatch { returned })
    } else {
        Ok(metadata.clone())
    }
}

fn decode_pool_metadata_token0(
    result: Option<&MulticallCallResult>,
) -> Result<Address, PoolMetadataFailure> {
    decode_pool_metadata_multicall_result(result, PoolMetadataCall::Token0, |return_data| {
        crate::uniswap_v3::token0Call::abi_decode_returns(return_data)
    })
}

fn decode_pool_metadata_token1(
    result: Option<&MulticallCallResult>,
) -> Result<Address, PoolMetadataFailure> {
    decode_pool_metadata_multicall_result(result, PoolMetadataCall::Token1, |return_data| {
        crate::uniswap_v3::token1Call::abi_decode_returns(return_data)
    })
}

fn decode_pool_metadata_fee(
    result: Option<&MulticallCallResult>,
) -> Result<u32, PoolMetadataFailure> {
    decode_pool_metadata_multicall_result(result, PoolMetadataCall::Fee, |return_data| {
        crate::uniswap_v3::feeCall::abi_decode_returns(return_data)
    })
}

fn decode_pool_metadata_factory_pool(
    result: Option<&MulticallCallResult>,
) -> Result<Address, PoolMetadataFailure> {
    decode_pool_metadata_multicall_result(result, PoolMetadataCall::FactoryGetPool, |return_data| {
        crate::uniswap_v3::getPoolCall::abi_decode_returns(return_data)
    })
}

fn decode_pool_metadata_multicall_result<T>(
    result: Option<&MulticallCallResult>,
    call: PoolMetadataCall,
    decode: impl FnOnce(&[u8]) -> alloy::sol_types::Result<T>,
) -> Result<T, PoolMetadataFailure> {
    match result {
        None => Err(PoolMetadataFailure::MissingResponse(call)),
        Some(result) if !result.success => Err(PoolMetadataFailure::CallFailed(call)),
        Some(result) => {
            decode(result.return_data.as_ref()).map_err(|_| PoolMetadataFailure::DecodeFailed(call))
        }
    }
}

fn sorted_pool_addresses(pools: HashSet<PoolAddress>) -> Vec<PoolAddress> {
    let mut pools = pools.into_iter().collect::<Vec<_>>();
    pools.sort_by(|left, right| left.0.cmp(&right.0));
    pools
}

fn pool_data_multicall_calls(pools: &[PoolAddress]) -> Vec<MulticallCall> {
    pools
        .iter()
        .flat_map(|pool| {
            [
                MulticallCall {
                    target: pool.0,
                    call_data: Bytes::from(crate::uniswap_v3::slot0Call {}.abi_encode()),
                },
                MulticallCall {
                    target: pool.0,
                    call_data: Bytes::from(crate::uniswap_v3::liquidityCall {}.abi_encode()),
                },
            ]
        })
        .collect()
}

fn decode_pool_data_results(
    pools: &[PoolAddress],
    results: &[MulticallCallResult],
) -> HashMap<PoolAddress, PoolDataResult> {
    let mut result_chunks = results.chunks(2);

    pools
        .iter()
        .map(|pool| {
            let result_chunk = result_chunks.next().unwrap_or(&[]);
            (
                *pool,
                decode_pool_data_result(result_chunk.first(), result_chunk.get(1)),
            )
        })
        .collect()
}

fn decode_pool_data_result(
    slot0: Option<&MulticallCallResult>,
    liquidity: Option<&MulticallCallResult>,
) -> PoolDataResult {
    let slot0 = decode_slot0(slot0)?;
    let liquidity = decode_liquidity(liquidity)?;

    Ok(PoolState {
        sqrt_price_x96: slot0.sqrtPriceX96,
        tick: slot0.tick,
        liquidity,
    })
}

fn decode_slot0(
    result: Option<&MulticallCallResult>,
) -> Result<crate::uniswap_v3::slot0Return, PoolDataFailure> {
    decode_multicall_result(result, PoolDataCall::Slot0, |return_data| {
        crate::uniswap_v3::slot0Call::abi_decode_returns(return_data)
    })
}

fn decode_liquidity(result: Option<&MulticallCallResult>) -> Result<u128, PoolDataFailure> {
    decode_multicall_result(result, PoolDataCall::Liquidity, |return_data| {
        crate::uniswap_v3::liquidityCall::abi_decode_returns(return_data)
    })
}

fn decode_multicall_result<T>(
    result: Option<&MulticallCallResult>,
    call: PoolDataCall,
    decode: impl FnOnce(&[u8]) -> alloy::sol_types::Result<T>,
) -> Result<T, PoolDataFailure> {
    match result {
        None => Err(PoolDataFailure::MissingResponse(call)),
        Some(result) if !result.success => Err(PoolDataFailure::CallFailed(call)),
        Some(result) => {
            decode(result.return_data.as_ref()).map_err(|_| PoolDataFailure::DecodeFailed(call))
        }
    }
}

pub fn fetch_finalized_block_header(
    agent: &ureq::Agent,
    config: &RpcConfig,
    chain: ChainKey,
) -> Result<Option<ClientHead>, ClientEvmError> {
    let endpoint = compose_http_endpoint(config, chain)?;
    let request = build_finalized_block_header_request(HTTP_REQUEST_ID);
    let mut response = agent
        .post(endpoint.as_str())
        .send_json(&request)
        .map_err(ClientEvmError::HttpError)?;
    let response_value = response
        .body_mut()
        .read_json::<Value>()
        .map_err(ClientEvmError::HttpError)?;

    parse_block_header_response_by_id(&response_value, HTTP_REQUEST_ID)
}

pub fn subscribe_new_heads<T, F>(
    config: &RpcConfig,
    chain: ChainKey,
    sender: &Sender<T>,
    map_event: F,
) -> Result<(), ClientEvmError>
where
    F: Fn(ClientEvent) -> Option<T>,
{
    let endpoint = compose_ws_endpoint(config, chain)?;
    let (mut socket, _) =
        connect(endpoint.as_str()).map_err(|ws_error| ClientEvmError::WebSocketError(ws_error))?;

    let subscribe_request = build_new_heads_subscribe_request(SUBSCRIBE_REQUEST_ID);
    socket
        .send(Message::text(subscribe_request.to_string()))
        .map_err(|ws_error| ClientEvmError::WebSocketError(ws_error))?;

    let subscription_id = read_subscription_id(&mut socket, SUBSCRIBE_REQUEST_ID)?;
    send_event(
        sender,
        ClientEvent::Subscribed {
            subscription_id: subscription_id.clone(),
        },
        &map_event,
    )?;

    read_new_head_events(&mut socket, &subscription_id, sender, &map_event)
}

fn read_subscription_id(
    socket: &mut BlockingWebSocket,
    expected_request_id: u64,
) -> Result<String, ClientEvmError> {
    loop {
        let Some(message) = read_json_rpc_message::<Value>(socket)? else {
            return Err(ClientEvmError::MalformedJsonRpcResponse(
                "websocket closed before subscription response".to_owned(),
            ));
        };

        if let Some(subscription_id) = parse_subscription_response(&message, expected_request_id)? {
            return Ok(subscription_id);
        }
    }
}

fn read_new_head_events<T, F>(
    socket: &mut BlockingWebSocket,
    subscription_id: &str,
    sender: &Sender<T>,
    map_event: &F,
) -> Result<(), ClientEvmError>
where
    F: Fn(ClientEvent) -> Option<T>,
{
    loop {
        let Some(sub_notification) =
            read_json_rpc_message::<SubscriptionNotification<ClientHead>>(socket)?
        else {
            send_event(
                sender,
                ClientEvent::Closed {
                    subscription_id: subscription_id.to_owned(),
                },
                map_event,
            )?;
            return Ok(());
        };

        if sub_notification.params.subscription != subscription_id {
            println!(
                "Received notification for unexpected subscription: {}. Expected: {}. Ignoring.",
                sub_notification.params.subscription, subscription_id
            );
            continue;
        }

        send_event(
            sender,
            ClientEvent::NewHead {
                subscription_id: sub_notification.params.subscription,
                header: sub_notification.params.result,
            },
            map_event,
        )?;
    }
}

fn read_json_rpc_message<T>(socket: &mut BlockingWebSocket) -> Result<Option<T>, ClientEvmError>
where
    T: for<'de> Deserialize<'de>,
{
    loop {
        match socket.read() {
            Ok(Message::Text(text)) => {
                let message = serde_json::from_slice::<T>(text.as_bytes())
                    .map_err(|error| ClientEvmError::JsonError(error))?;
                return Ok(Some(message));
            }
            Ok(Message::Binary(bytes)) => {
                let text = str::from_utf8(bytes.as_ref())
                    .map_err(|error| ClientEvmError::MalformedJsonRpcResponse(error.to_string()))?;
                let message = serde_json::from_str::<T>(text)
                    .map_err(|error| ClientEvmError::JsonError(error))?;
                return Ok(Some(message));
            }
            Ok(Message::Ping(_) | Message::Pong(_) | Message::Frame(_)) => {}
            Ok(Message::Close(_)) => return Ok(None),
            Err(tungstenite::Error::ConnectionClosed) => return Ok(None),
            Err(error) => return Err(ClientEvmError::WebSocketError(error)),
        }
    }
}

fn send_event<T, F>(
    sender: &Sender<T>,
    event: ClientEvent,
    map_event: &F,
) -> Result<(), ClientEvmError>
where
    F: Fn(ClientEvent) -> Option<T>,
{
    match map_event(event) {
        Some(e) => sender
            .send(e)
            .map_err(|_| ClientEvmError::EventReceiverDropped),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashSet,
        io::{Read, Write},
        net::TcpListener,
        sync::mpsc::{Receiver, channel},
        thread::{self, JoinHandle},
    };

    use alloy::{
        primitives::{Address, B256, Bytes, U160, U256, aliases::I24},
        sol_types::SolCall,
    };
    use serde_json::{Value, json};

    use crate::{
        PoolAddress, PoolCandidateAddress, PoolDataCall, PoolDataFailure, PoolMetadataCall,
        PoolMetadataFailure, PoolState, TokenAddress, TokenDecimals, TokenMetadata,
        TokenMetadataCall, TokenMetadataFailure, UniswapV3Fee,
        client::multicall3::{
            MULTICALL3_ADDRESS, MulticallCall, MulticallCallResult,
            aggregate3_return_data_for_test, decode_aggregate3_call_data_for_test,
        },
    };

    use super::*;

    #[test]
    fn fetch_block_header_posts_expected_request_and_decodes_response() {
        let block_hash = B256::with_last_byte(1);
        let response = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": block_header_result(block_hash)
        });
        let (http_url, received_request, server) = spawn_json_rpc_server(response);
        let config = rpc_config(&http_url);
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_block_header(&agent, &config, ChainKey::Ethereum, block_hash);

        assert!(matches!(
            result,
            Ok(Some(header))
                if header.inner.hash == block_hash
                    && header.inner.inner.parent_hash == B256::with_last_byte(2)
        ));

        let request = received_request
            .recv()
            .expect("server must report received request");
        assert_eq!(request.path, "/ethereum/api-key");
        assert_eq!(
            request.body,
            json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "eth_getBlockByHash",
                "params": [block_hash, false]
            })
        );
        server.join().expect("server thread must complete");
    }

    #[test]
    fn fetch_block_header_maps_transport_failure_to_http_error() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("test listener must bind");
        let address = listener
            .local_addr()
            .expect("test listener must have local address");
        drop(listener);
        let config = rpc_config(&format!("http://{address}"));
        let agent = ureq::Agent::new_with_defaults();

        let result =
            fetch_block_header(&agent, &config, ChainKey::Ethereum, B256::with_last_byte(1));

        assert!(matches!(result, Err(ClientEvmError::HttpError(_))));
    }

    #[test]
    fn fetch_block_logs_posts_expected_request_and_decodes_pool_candidate_addresses() {
        let block_hash = B256::with_last_byte(1);
        let first_pool = Address::with_last_byte(2);
        let second_pool = Address::with_last_byte(3);
        let response = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": [
                log_result(first_pool, block_hash),
                log_result(second_pool, block_hash),
                log_result(first_pool, block_hash)
            ]
        });
        let (http_url, received_request, server) = spawn_json_rpc_server(response);
        let config = rpc_config(&http_url);
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_block_logs(&agent, &config, ChainKey::Ethereum, block_hash);

        assert!(matches!(
            result,
            Ok(ref pools)
                if pools.len() == 2
                    && pools.contains(&PoolCandidateAddress(first_pool))
                    && pools.contains(&PoolCandidateAddress(second_pool))
        ));

        let request = received_request
            .recv()
            .expect("server must report received request");
        assert_eq!(request.path, "/ethereum/api-key");
        assert_eq!(
            request.body,
            json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "eth_getLogs",
                "params": [{
                    "blockHash": block_hash,
                    "topics": [crate::uniswap_v3::pool_event_signature_hashes()
                        .into_iter()
                        .map(|topic| topic.to_string())
                        .collect::<Vec<_>>()]
                }]
            })
        );
        server.join().expect("server thread must complete");
    }

    #[test]
    fn fetch_block_logs_maps_transport_failure_to_http_error() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("test listener must bind");
        let address = listener
            .local_addr()
            .expect("test listener must have local address");
        drop(listener);
        let config = rpc_config(&format!("http://{address}"));
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_block_logs(&agent, &config, ChainKey::Ethereum, B256::with_last_byte(1));

        assert!(matches!(result, Err(ClientEvmError::HttpError(_))));
    }

    #[test]
    fn fetch_pool_data_with_empty_pool_set_returns_empty_results_without_http() {
        let config = rpc_config("http://127.0.0.1:9");
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_pool_data(
            &agent,
            &config,
            ChainKey::Ethereum,
            B256::with_last_byte(1),
            HashSet::new(),
        );

        assert!(matches!(result, Ok(ref pools) if pools.is_empty()));
    }

    #[test]
    fn fetch_pool_data_posts_multicall3_request_and_decodes_pool_states() {
        let at = B256::with_last_byte(1);
        let first_pool = PoolAddress(Address::with_last_byte(2), ChainKey::Ethereum);
        let second_pool = PoolAddress(Address::with_last_byte(3), ChainKey::Ethereum);
        let first_state = pool_state(11, -12, 13);
        let second_state = pool_state(21, 22, 23);
        let response = multicall3_response([
            successful_multicall_result(slot0_return_data(&first_state)),
            successful_multicall_result(liquidity_return_data(first_state.liquidity)),
            successful_multicall_result(slot0_return_data(&second_state)),
            successful_multicall_result(liquidity_return_data(second_state.liquidity)),
        ]);
        let (http_url, received_request, server) = spawn_json_rpc_server(response);
        let config = rpc_config(&http_url);
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_pool_data(
            &agent,
            &config,
            ChainKey::Ethereum,
            at,
            HashSet::from([second_pool, first_pool]),
        );

        let pools = result.expect("pool data fetch must succeed");
        assert_eq!(pools.get(&first_pool), Some(&Ok(first_state.clone())));
        assert_eq!(pools.get(&second_pool), Some(&Ok(second_state.clone())));

        let request = received_request
            .recv()
            .expect("server must report received request");
        assert_eq!(request.path, "/ethereum/api-key");
        assert_eq!(request.body.get("method"), Some(&json!("eth_call")));
        assert_eq!(
            request
                .body
                .get("params")
                .and_then(Value::as_array)
                .and_then(|params| params.first())
                .and_then(|call| call.get("to")),
            Some(&json!(MULTICALL3_ADDRESS))
        );
        assert_eq!(
            request
                .body
                .get("params")
                .and_then(Value::as_array)
                .and_then(|params| params.get(1)),
            Some(&json!({ "blockHash": at }))
        );

        let multicall_data = request
            .body
            .get("params")
            .and_then(Value::as_array)
            .and_then(|params| params.first())
            .and_then(|call| call.get("data"))
            .cloned()
            .and_then(|data| serde_json::from_value::<Bytes>(data).ok())
            .expect("pool data request must contain multicall data");
        let calls = decode_aggregate3_call_data_for_test(&multicall_data)
            .expect("pool data request must decode as aggregate3");
        assert_eq!(
            calls
                .iter()
                .map(|call| (call.target, call.call_data.clone()))
                .collect::<Vec<_>>(),
            vec![
                (first_pool.0, slot0_call_data()),
                (first_pool.0, liquidity_call_data()),
                (second_pool.0, slot0_call_data()),
                (second_pool.0, liquidity_call_data()),
            ]
        );

        server.join().expect("server thread must complete");
    }

    #[test]
    fn fetch_pool_data_returns_per_pool_failure_for_failed_inner_call() {
        let at = B256::with_last_byte(1);
        let first_pool = PoolAddress(Address::with_last_byte(2), ChainKey::Ethereum);
        let second_pool = PoolAddress(Address::with_last_byte(3), ChainKey::Ethereum);
        let first_state = pool_state(11, -12, 13);
        let second_state = pool_state(21, 22, 23);
        let response = multicall3_response([
            successful_multicall_result(slot0_return_data(&first_state)),
            successful_multicall_result(liquidity_return_data(first_state.liquidity)),
            successful_multicall_result(slot0_return_data(&second_state)),
            failed_multicall_result(),
        ]);
        let (http_url, _received_request, server) = spawn_json_rpc_server(response);
        let config = rpc_config(&http_url);
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_pool_data(
            &agent,
            &config,
            ChainKey::Ethereum,
            at,
            HashSet::from([first_pool, second_pool]),
        );

        let pools = result.expect("outer multicall must succeed");
        assert_eq!(pools.get(&first_pool), Some(&Ok(first_state)));
        assert_eq!(
            pools.get(&second_pool),
            Some(&Err(PoolDataFailure::CallFailed(PoolDataCall::Liquidity)))
        );
        server.join().expect("server thread must complete");
    }

    #[test]
    fn fetch_pool_data_returns_per_pool_failure_for_malformed_inner_return_data() {
        let at = B256::with_last_byte(1);
        let pool = PoolAddress(Address::with_last_byte(2), ChainKey::Ethereum);
        let state = pool_state(11, -12, 13);
        let response = multicall3_response([
            successful_multicall_result(Bytes::from(vec![0x12])),
            successful_multicall_result(liquidity_return_data(state.liquidity)),
        ]);
        let (http_url, _received_request, server) = spawn_json_rpc_server(response);
        let config = rpc_config(&http_url);
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_pool_data(
            &agent,
            &config,
            ChainKey::Ethereum,
            at,
            HashSet::from([pool]),
        );

        let pools = result.expect("outer multicall must succeed");
        assert_eq!(
            pools.get(&pool),
            Some(&Err(PoolDataFailure::DecodeFailed(PoolDataCall::Slot0)))
        );
        server.join().expect("server thread must complete");
    }

    #[test]
    fn fetch_pool_data_maps_transport_failure_to_http_error() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("test listener must bind");
        let address = listener
            .local_addr()
            .expect("test listener must have local address");
        drop(listener);
        let config = rpc_config(&format!("http://{address}"));
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_pool_data(
            &agent,
            &config,
            ChainKey::Ethereum,
            B256::with_last_byte(1),
            HashSet::from([PoolAddress(Address::with_last_byte(2), ChainKey::Ethereum)]),
        );

        assert!(matches!(result, Err(ClientEvmError::HttpError(_))));
    }

    #[test]
    fn fetch_pool_metadata_with_empty_candidate_set_returns_empty_results_without_http() {
        let config = rpc_config("http://127.0.0.1:9");
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_pool_metadata(
            &agent,
            &config,
            ChainKey::Ethereum,
            B256::with_last_byte(1),
            HashSet::new(),
        );

        assert!(matches!(result, Ok(ref metadata) if metadata.is_empty()));
    }

    #[test]
    fn fetch_pool_metadata_posts_candidate_and_factory_multicalls_and_verifies_metadata() {
        let at = B256::with_last_byte(1);
        let first_candidate = PoolCandidateAddress(Address::with_last_byte(2));
        let second_candidate = PoolCandidateAddress(Address::with_last_byte(3));
        let first_token0 = Address::with_last_byte(4);
        let first_token1 = Address::with_last_byte(5);
        let second_token0 = Address::with_last_byte(6);
        let second_token1 = Address::with_last_byte(7);
        let returned_mismatch = Address::with_last_byte(8);
        let first_fee = UniswapV3Fee::Fee500;
        let second_fee = UniswapV3Fee::Fee3000;
        let first_response = multicall3_response([
            successful_multicall_result(address_return_data_for_token0(first_token0)),
            successful_multicall_result(address_return_data_for_token1(first_token1)),
            successful_multicall_result(fee_return_data(first_fee.pips())),
            successful_multicall_result(address_return_data_for_token0(second_token0)),
            successful_multicall_result(address_return_data_for_token1(second_token1)),
            successful_multicall_result(fee_return_data(second_fee.pips())),
        ]);
        let second_response = multicall3_response([
            successful_multicall_result(get_pool_return_data(first_candidate.0)),
            successful_multicall_result(get_pool_return_data(returned_mismatch)),
        ]);
        let (http_url, received_request, server) =
            spawn_json_rpc_server_sequence(vec![first_response, second_response]);
        let config = rpc_config(&http_url);
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_pool_metadata(
            &agent,
            &config,
            ChainKey::Ethereum,
            at,
            HashSet::from([second_candidate, first_candidate]),
        )
        .expect("metadata fetch must succeed");

        assert_eq!(
            result.get(&first_candidate),
            Some(&Ok(crate::PoolMetadata {
                token0: first_token0,
                token1: first_token1,
                fee: first_fee,
            }))
        );
        assert_eq!(
            result.get(&second_candidate),
            Some(&Err(PoolMetadataFailure::FactoryMismatch {
                returned: returned_mismatch,
            }))
        );

        let first_request = received_request
            .recv()
            .expect("server must report first request");
        assert_eq!(first_request.path, "/ethereum/api-key");
        assert_multicall_request_at(&first_request.body, at);
        let first_calls = multicall_calls_from_request(&first_request.body);
        assert_eq!(
            first_calls
                .iter()
                .map(|call| (call.target, call.call_data.clone()))
                .collect::<Vec<_>>(),
            vec![
                (first_candidate.0, token0_call_data()),
                (first_candidate.0, token1_call_data()),
                (first_candidate.0, fee_call_data()),
                (second_candidate.0, token0_call_data()),
                (second_candidate.0, token1_call_data()),
                (second_candidate.0, fee_call_data()),
            ]
        );

        let second_request = received_request
            .recv()
            .expect("server must report second request");
        assert_multicall_request_at(&second_request.body, at);
        let second_calls = multicall_calls_from_request(&second_request.body);
        assert_eq!(
            second_calls
                .iter()
                .map(|call| (call.target, call.call_data.clone()))
                .collect::<Vec<_>>(),
            vec![
                (
                    crate::uniswap_v3::ETHEREUM_UNISWAP_V3_FACTORY_ADDRESS,
                    get_pool_call_data(first_token0, first_token1, first_fee),
                ),
                (
                    crate::uniswap_v3::ETHEREUM_UNISWAP_V3_FACTORY_ADDRESS,
                    get_pool_call_data(second_token0, second_token1, second_fee),
                ),
            ]
        );
        server.join().expect("server thread must complete");
    }

    #[test]
    fn fetch_pool_metadata_rejects_unsupported_fee_without_factory_lookup() {
        let at = B256::with_last_byte(1);
        let candidate = PoolCandidateAddress(Address::with_last_byte(2));
        let response = multicall3_response([
            successful_multicall_result(address_return_data_for_token0(Address::with_last_byte(3))),
            successful_multicall_result(address_return_data_for_token1(Address::with_last_byte(4))),
            successful_multicall_result(fee_return_data(2500)),
        ]);
        let (http_url, received_request, server) = spawn_json_rpc_server_sequence(vec![response]);
        let config = rpc_config(&http_url);
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_pool_metadata(
            &agent,
            &config,
            ChainKey::Ethereum,
            at,
            HashSet::from([candidate]),
        )
        .expect("outer multicall must succeed");

        assert_eq!(
            result.get(&candidate),
            Some(&Err(PoolMetadataFailure::UnsupportedFee(2500)))
        );
        assert!(received_request.recv().is_ok());
        assert!(received_request.try_recv().is_err());
        server.join().expect("server thread must complete");
    }

    #[test]
    fn fetch_pool_metadata_rejects_round_one_call_and_decode_failures() {
        let at = B256::with_last_byte(1);
        let failed_candidate = PoolCandidateAddress(Address::with_last_byte(2));
        let malformed_candidate = PoolCandidateAddress(Address::with_last_byte(3));
        let response = multicall3_response([
            failed_multicall_result(),
            successful_multicall_result(address_return_data_for_token1(Address::with_last_byte(4))),
            successful_multicall_result(fee_return_data(UniswapV3Fee::Fee500.pips())),
            successful_multicall_result(Bytes::from(vec![0x12])),
            successful_multicall_result(address_return_data_for_token1(Address::with_last_byte(5))),
            successful_multicall_result(fee_return_data(UniswapV3Fee::Fee3000.pips())),
        ]);
        let (http_url, _received_request, server) = spawn_json_rpc_server_sequence(vec![response]);
        let config = rpc_config(&http_url);
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_pool_metadata(
            &agent,
            &config,
            ChainKey::Ethereum,
            at,
            HashSet::from([failed_candidate, malformed_candidate]),
        )
        .expect("outer multicall must succeed");

        assert_eq!(
            result.get(&failed_candidate),
            Some(&Err(PoolMetadataFailure::CallFailed(
                PoolMetadataCall::Token0
            )))
        );
        assert_eq!(
            result.get(&malformed_candidate),
            Some(&Err(PoolMetadataFailure::DecodeFailed(
                PoolMetadataCall::Token0
            )))
        );
        server.join().expect("server thread must complete");
    }

    #[test]
    fn fetch_pool_metadata_rejects_zero_factory_result() {
        let at = B256::with_last_byte(1);
        let candidate = PoolCandidateAddress(Address::with_last_byte(2));
        let token0 = Address::with_last_byte(3);
        let token1 = Address::with_last_byte(4);
        let response = multicall3_response([
            successful_multicall_result(address_return_data_for_token0(token0)),
            successful_multicall_result(address_return_data_for_token1(token1)),
            successful_multicall_result(fee_return_data(UniswapV3Fee::Fee100.pips())),
        ]);
        let factory_response = multicall3_response([successful_multicall_result(
            get_pool_return_data(Address::ZERO),
        )]);
        let (http_url, _received_request, server) =
            spawn_json_rpc_server_sequence(vec![response, factory_response]);
        let config = rpc_config(&http_url);
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_pool_metadata(
            &agent,
            &config,
            ChainKey::Ethereum,
            at,
            HashSet::from([candidate]),
        )
        .expect("outer multicall must succeed");

        assert_eq!(
            result.get(&candidate),
            Some(&Err(PoolMetadataFailure::FactoryReturnedZero))
        );
        server.join().expect("server thread must complete");
    }

    #[test]
    fn fetch_pool_metadata_rejects_factory_call_and_decode_failures() {
        let at = B256::with_last_byte(1);
        let failed_candidate = PoolCandidateAddress(Address::with_last_byte(2));
        let malformed_candidate = PoolCandidateAddress(Address::with_last_byte(3));
        let failed_token0 = Address::with_last_byte(4);
        let failed_token1 = Address::with_last_byte(5);
        let malformed_token0 = Address::with_last_byte(6);
        let malformed_token1 = Address::with_last_byte(7);
        let response = multicall3_response([
            successful_multicall_result(address_return_data_for_token0(failed_token0)),
            successful_multicall_result(address_return_data_for_token1(failed_token1)),
            successful_multicall_result(fee_return_data(UniswapV3Fee::Fee500.pips())),
            successful_multicall_result(address_return_data_for_token0(malformed_token0)),
            successful_multicall_result(address_return_data_for_token1(malformed_token1)),
            successful_multicall_result(fee_return_data(UniswapV3Fee::Fee3000.pips())),
        ]);
        let factory_response = multicall3_response([
            failed_multicall_result(),
            successful_multicall_result(Bytes::from(vec![0x12])),
        ]);
        let (http_url, _received_request, server) =
            spawn_json_rpc_server_sequence(vec![response, factory_response]);
        let config = rpc_config(&http_url);
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_pool_metadata(
            &agent,
            &config,
            ChainKey::Ethereum,
            at,
            HashSet::from([failed_candidate, malformed_candidate]),
        )
        .expect("outer multicalls must succeed");

        assert_eq!(
            result.get(&failed_candidate),
            Some(&Err(PoolMetadataFailure::CallFailed(
                PoolMetadataCall::FactoryGetPool
            )))
        );
        assert_eq!(
            result.get(&malformed_candidate),
            Some(&Err(PoolMetadataFailure::DecodeFailed(
                PoolMetadataCall::FactoryGetPool
            )))
        );
        server.join().expect("server thread must complete");
    }

    #[test]
    fn fetch_token_metadata_with_empty_token_set_returns_empty_results_without_http() {
        let config = rpc_config("http://127.0.0.1:9");
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_token_metadata(
            &agent,
            &config,
            ChainKey::Ethereum,
            B256::with_last_byte(1),
            HashSet::new(),
        );

        assert!(matches!(result, Ok(ref metadata) if metadata.is_empty()));
    }

    #[test]
    fn fetch_token_metadata_posts_multicall3_request_and_decodes_decimals() {
        let at = B256::with_last_byte(1);
        let first_token = TokenAddress(Address::with_last_byte(2), ChainKey::Ethereum);
        let second_token = TokenAddress(Address::with_last_byte(3), ChainKey::Ethereum);
        let response = multicall3_response([
            successful_multicall_result(decimals_return_data(6)),
            successful_multicall_result(decimals_return_data(18)),
        ]);
        let (http_url, received_request, server) = spawn_json_rpc_server(response);
        let config = rpc_config(&http_url);
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_token_metadata(
            &agent,
            &config,
            ChainKey::Ethereum,
            at,
            HashSet::from([second_token, first_token]),
        )
        .expect("token metadata fetch must succeed");

        assert_eq!(
            result.get(&first_token),
            Some(&Ok(TokenMetadata {
                decimals: TokenDecimals::try_from_u256(U256::from(6)).unwrap(),
            }))
        );
        assert_eq!(
            result.get(&second_token),
            Some(&Ok(TokenMetadata {
                decimals: TokenDecimals::try_from_u256(U256::from(18)).unwrap(),
            }))
        );

        let request = received_request
            .recv()
            .expect("server must report received request");
        assert_eq!(request.path, "/ethereum/api-key");
        assert_multicall_request_at(&request.body, at);
        let calls = multicall_calls_from_request(&request.body);
        assert_eq!(
            calls
                .iter()
                .map(|call| (call.target, call.call_data.clone()))
                .collect::<Vec<_>>(),
            vec![
                (first_token.0, decimals_call_data()),
                (second_token.0, decimals_call_data()),
            ]
        );
        server.join().expect("server thread must complete");
    }

    #[test]
    fn fetch_token_metadata_returns_per_token_failures() {
        let at = B256::with_last_byte(1);
        let failed_token = TokenAddress(Address::with_last_byte(2), ChainKey::Ethereum);
        let malformed_token = TokenAddress(Address::with_last_byte(3), ChainKey::Ethereum);
        let unsupported_token = TokenAddress(Address::with_last_byte(4), ChainKey::Ethereum);
        let response = multicall3_response([
            failed_multicall_result(),
            successful_multicall_result(Bytes::from(vec![0x12])),
            successful_multicall_result(decimals_return_data(37)),
        ]);
        let (http_url, _received_request, server) = spawn_json_rpc_server(response);
        let config = rpc_config(&http_url);
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_token_metadata(
            &agent,
            &config,
            ChainKey::Ethereum,
            at,
            HashSet::from([failed_token, malformed_token, unsupported_token]),
        )
        .expect("outer multicall must succeed");

        assert_eq!(
            result.get(&failed_token),
            Some(&Err(TokenMetadataFailure::CallFailed(
                TokenMetadataCall::Decimals
            )))
        );
        assert_eq!(
            result.get(&malformed_token),
            Some(&Err(TokenMetadataFailure::DecodeFailed(
                TokenMetadataCall::Decimals
            )))
        );
        assert_eq!(
            result.get(&unsupported_token),
            Some(&Err(TokenMetadataFailure::UnsupportedDecimals(U256::from(
                37
            ))))
        );
        server.join().expect("server thread must complete");
    }

    #[test]
    fn fetch_token_metadata_maps_transport_failure_to_http_error() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("test listener must bind");
        let address = listener
            .local_addr()
            .expect("test listener must have local address");
        drop(listener);
        let config = rpc_config(&format!("http://{address}"));
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_token_metadata(
            &agent,
            &config,
            ChainKey::Ethereum,
            B256::with_last_byte(1),
            HashSet::from([TokenAddress(Address::with_last_byte(2), ChainKey::Ethereum)]),
        );

        assert!(matches!(result, Err(ClientEvmError::HttpError(_))));
    }

    #[test]
    fn fetch_finalized_block_header_posts_expected_request_and_decodes_response() {
        let block_hash = B256::with_last_byte(1);
        let response = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": block_header_result(block_hash)
        });
        let (http_url, received_request, server) = spawn_json_rpc_server(response);
        let config = rpc_config(&http_url);
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_finalized_block_header(&agent, &config, ChainKey::Ethereum);

        assert!(matches!(
            result,
            Ok(Some(header))
                if header.inner.hash == block_hash
                    && header.inner.inner.parent_hash == B256::with_last_byte(2)
        ));

        let request = received_request
            .recv()
            .expect("server must report received request");
        assert_eq!(request.path, "/ethereum/api-key");
        assert_eq!(
            request.body,
            json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "eth_getBlockByNumber",
                "params": ["finalized", false]
            })
        );
        server.join().expect("server thread must complete");
    }

    #[test]
    fn fetch_finalized_block_header_maps_transport_failure_to_http_error() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("test listener must bind");
        let address = listener
            .local_addr()
            .expect("test listener must have local address");
        drop(listener);
        let config = rpc_config(&format!("http://{address}"));
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_finalized_block_header(&agent, &config, ChainKey::Ethereum);

        assert!(matches!(result, Err(ClientEvmError::HttpError(_))));
    }

    #[test]
    fn new_head_notification_preserves_header_fields() {
        let result =
            serde_json::from_value::<SubscriptionNotification<ClientHead>>(new_head_notification());

        assert!(matches!(
            result,
            Ok(SubscriptionNotification {
                params: SubscriptionDataParams {
                    subscription,
                    result,
                },
            }) if subscription == "0xsubscription"
                && result.inner.hash == B256::with_last_byte(1)
                && result.inner.inner.parent_hash == B256::with_last_byte(2)
                && result.inner.inner.number == 9
                && result.inner.inner.timestamp == 12
        ));
    }

    #[test]
    fn new_head_notification_preserves_extra_header_fields() {
        let result =
            serde_json::from_value::<SubscriptionNotification<ClientHead>>(new_head_notification());

        assert!(matches!(
            result,
            Ok(SubscriptionNotification {
                params: SubscriptionDataParams { result, .. },
            }) if matches!(
                result.other.get_deserialized::<String>("providerTag"),
                Some(Ok(ref tag)) if tag == "observed"
            )
        ));
    }

    #[test]
    fn client_new_head_event_carries_header() {
        let parsed =
            serde_json::from_value::<SubscriptionNotification<ClientHead>>(new_head_notification());

        assert!(matches!(
            parsed.map(|notification| ClientEvent::NewHead {
                subscription_id: notification.params.subscription,
                header: notification.params.result,
            }),
            Ok(ClientEvent::NewHead { header, .. })
                if header.inner.hash == B256::with_last_byte(1)
                    && header.inner.inner.number == 9
        ));
    }

    fn new_head_notification() -> Value {
        json!({
            "jsonrpc": "2.0",
            "method": "eth_subscription",
            "params": {
                "subscription": "0xsubscription",
                "result": {
                    "hash": "0x0000000000000000000000000000000000000000000000000000000000000001",
                    "parentHash": "0x0000000000000000000000000000000000000000000000000000000000000002",
                    "sha3Uncles": "0x0000000000000000000000000000000000000000000000000000000000000003",
                    "miner": "0x0000000000000000000000000000000000000004",
                    "stateRoot": "0x0000000000000000000000000000000000000000000000000000000000000005",
                    "transactionsRoot": "0x0000000000000000000000000000000000000000000000000000000000000006",
                    "receiptsRoot": "0x0000000000000000000000000000000000000000000000000000000000000007",
                    "logsBloom": zero_logs_bloom(),
                    "difficulty": "0xd",
                    "number": "0x9",
                    "gasLimit": "0xb",
                    "gasUsed": "0xa",
                    "timestamp": "0xc",
                    "extraData": "0x010203",
                    "mixHash": "0x000000000000000000000000000000000000000000000000000000000000000e",
                    "nonce": "0x000000000000000f",
                    "providerTag": "observed"
                }
            }
        })
    }

    fn zero_logs_bloom() -> String {
        format!("0x{}", "00".repeat(256))
    }

    fn multicall3_response<const N: usize>(results: [MulticallCallResult; N]) -> Value {
        json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": aggregate3_return_data_for_test(&results)
        })
    }

    fn successful_multicall_result(return_data: Bytes) -> MulticallCallResult {
        MulticallCallResult {
            success: true,
            return_data,
        }
    }

    fn failed_multicall_result() -> MulticallCallResult {
        MulticallCallResult {
            success: false,
            return_data: Bytes::from(Vec::<u8>::new()),
        }
    }

    fn pool_state(sqrt_price_x96: u64, tick: i32, liquidity: u128) -> PoolState {
        PoolState {
            sqrt_price_x96: U160::from(sqrt_price_x96),
            tick: I24::try_from(tick).expect("test tick must fit int24"),
            liquidity,
        }
    }

    fn slot0_call_data() -> Bytes {
        Bytes::from(crate::uniswap_v3::slot0Call {}.abi_encode())
    }

    fn liquidity_call_data() -> Bytes {
        Bytes::from(crate::uniswap_v3::liquidityCall {}.abi_encode())
    }

    fn slot0_return_data(pool_state: &PoolState) -> Bytes {
        Bytes::from(crate::uniswap_v3::slot0Call::abi_encode_returns(
            &crate::uniswap_v3::slot0Return {
                sqrtPriceX96: pool_state.sqrt_price_x96,
                tick: I24::try_from(pool_state.tick).expect("test tick must fit int24"),
                observationIndex: 0,
                observationCardinality: 0,
                observationCardinalityNext: 0,
                feeProtocol: 0,
                unlocked: true,
            },
        ))
    }

    fn liquidity_return_data(liquidity: u128) -> Bytes {
        Bytes::from(crate::uniswap_v3::liquidityCall::abi_encode_returns(
            &liquidity,
        ))
    }

    fn token0_call_data() -> Bytes {
        Bytes::from(crate::uniswap_v3::token0Call {}.abi_encode())
    }

    fn token1_call_data() -> Bytes {
        Bytes::from(crate::uniswap_v3::token1Call {}.abi_encode())
    }

    fn fee_call_data() -> Bytes {
        Bytes::from(crate::uniswap_v3::feeCall {}.abi_encode())
    }

    fn decimals_call_data() -> Bytes {
        Bytes::from(crate::erc20::decimalsCall {}.abi_encode())
    }

    fn decimals_return_data(decimals: u8) -> Bytes {
        Bytes::from(crate::erc20::decimalsCall::abi_encode_returns(&U256::from(
            decimals,
        )))
    }

    fn get_pool_call_data(token0: Address, token1: Address, fee: UniswapV3Fee) -> Bytes {
        Bytes::from(
            crate::uniswap_v3::getPoolCall {
                tokenA: token0,
                tokenB: token1,
                fee: alloy::primitives::Uint::<24, 1>::from(fee.pips()),
            }
            .abi_encode(),
        )
    }

    fn address_return_data_for_token0(address: Address) -> Bytes {
        Bytes::from(crate::uniswap_v3::token0Call::abi_encode_returns(&address))
    }

    fn address_return_data_for_token1(address: Address) -> Bytes {
        Bytes::from(crate::uniswap_v3::token1Call::abi_encode_returns(&address))
    }

    fn fee_return_data(fee: u32) -> Bytes {
        Bytes::from(crate::uniswap_v3::feeCall::abi_encode_returns(&fee))
    }

    fn get_pool_return_data(pool: Address) -> Bytes {
        Bytes::from(crate::uniswap_v3::getPoolCall::abi_encode_returns(&pool))
    }

    fn assert_multicall_request_at(request: &Value, at: B256) {
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
    }

    fn multicall_calls_from_request(request: &Value) -> Vec<MulticallCall> {
        let multicall_data = request
            .get("params")
            .and_then(Value::as_array)
            .and_then(|params| params.first())
            .and_then(|call| call.get("data"))
            .cloned()
            .and_then(|data| serde_json::from_value::<Bytes>(data).ok())
            .expect("request must contain multicall data");

        decode_aggregate3_call_data_for_test(&multicall_data)
            .expect("request data must decode as aggregate3")
    }

    struct ReceivedHttpRequest {
        path: String,
        body: Value,
    }

    fn spawn_json_rpc_server(
        response: Value,
    ) -> (String, Receiver<ReceivedHttpRequest>, JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("test server must bind");
        let address = listener
            .local_addr()
            .expect("test server must have local address");
        let (sender, receiver) = channel();

        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("test server must accept request");
            let request = read_http_request(&mut stream);
            sender
                .send(request)
                .expect("test server must report received request");

            let response_body =
                serde_json::to_vec(&response).expect("test response must serialize");
            let response_headers = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                response_body.len()
            );

            stream
                .write_all(response_headers.as_bytes())
                .expect("test server must write response headers");
            stream
                .write_all(&response_body)
                .expect("test server must write response body");
        });

        (format!("http://{address}"), receiver, handle)
    }

    fn spawn_json_rpc_server_sequence(
        responses: Vec<Value>,
    ) -> (String, Receiver<ReceivedHttpRequest>, JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("test server must bind");
        let address = listener
            .local_addr()
            .expect("test server must have local address");
        let (sender, receiver) = channel();

        let handle = thread::spawn(move || {
            for response in responses {
                let (mut stream, _) = listener.accept().expect("test server must accept request");
                let request = read_http_request(&mut stream);
                sender
                    .send(request)
                    .expect("test server must report received request");

                let response_body =
                    serde_json::to_vec(&response).expect("test response must serialize");
                let response_headers = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    response_body.len()
                );

                stream
                    .write_all(response_headers.as_bytes())
                    .expect("test server must write response headers");
                stream
                    .write_all(&response_body)
                    .expect("test server must write response body");
            }
        });

        (format!("http://{address}"), receiver, handle)
    }

    fn read_http_request(stream: &mut TcpStream) -> ReceivedHttpRequest {
        let mut request_bytes = Vec::new();
        let mut buffer = [0; 1024];
        let (body_start, content_length) = loop {
            let bytes_read = stream
                .read(&mut buffer)
                .expect("test server must read request");
            assert!(bytes_read > 0, "request must contain headers and body");
            request_bytes.extend_from_slice(&buffer[..bytes_read]);

            if let Some(header_end) = request_bytes
                .windows(4)
                .position(|window| window == b"\r\n\r\n")
            {
                let body_start = header_end + 4;
                let headers = str::from_utf8(&request_bytes[..header_end])
                    .expect("request headers must be utf-8");
                let content_length = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().ok())
                            .flatten()
                    })
                    .expect("request must contain content-length");

                if request_bytes.len() >= body_start + content_length {
                    break (body_start, content_length);
                }
            }
        };

        let headers =
            str::from_utf8(&request_bytes[..body_start]).expect("request headers must be utf-8");
        let path = headers
            .lines()
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
            .expect("request line must contain path")
            .to_owned();
        let body = serde_json::from_slice(&request_bytes[body_start..body_start + content_length])
            .expect("request body must be json");

        ReceivedHttpRequest { path, body }
    }

    fn rpc_config(http_url: &str) -> RpcConfig {
        RpcConfig {
            http_url: http_url.to_owned(),
            ws_url: "wss://example.invalid".to_owned(),
            api_key: "api-key".to_owned(),
        }
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
            "logsBloom": zero_logs_bloom(),
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

    fn log_result(address: Address, block_hash: B256) -> Value {
        json!({
            "address": address,
            "topics": [
                crate::uniswap_v3::pool_event_signature_hashes()[0]
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
}
