use std::{
    collections::{HashMap, HashSet},
    net::TcpStream,
    str,
    sync::mpsc::Sender,
    thread,
};

use alloy::{
    primitives::{Address, BlockHash, Bytes, U256, Uint},
    rpc::types::Log,
    sol_types::SolCall,
};
use serde::Deserialize;
use serde_json::Value;
use tungstenite::{Message, WebSocket, connect, stream::MaybeTlsStream};

use crate::{
    ChainKey, ClientEvmError, PoolFee, PoolRef, ProtocolPoolKey, PoolDataCall, PoolDataFailure,
    PoolDataResult, PoolLog, PoolMetadata, PoolMetadataCall, PoolMetadataFailure,
    PoolMetadataResult, PoolState, RangeLogBlock, TokenAddress, TokenDecimals,
    TokenMetadata, TokenMetadataCall, TokenMetadataFailure, TokenMetadataResult, UniswapV3Fee,
    decode_pool_log,
    endpoints::{ChainEndpoints, EndpointPool},
};

use super::{
    ClientEvent, ClientHead,
    client_utils::{
        build_block_header_request, build_block_logs_request, build_finalized_block_header_request,
        build_new_heads_subscribe_request, build_pool_events_subscribe_request,
        build_pool_logs_range_request, parse_block_header_response,
        parse_block_header_response_by_id, parse_block_logs_response,
        parse_pool_logs_range_response, parse_subscription_response,
    },
    multicall3::{
        MulticallBlock, MulticallCall, MulticallCallResult, build_multicall3_batch_request,
        parse_multicall3_batch_response,
    },
};

const HTTP_REQUEST_ID: u64 = 1;
const SUBSCRIBE_REQUEST_ID: u64 = 1;

/// Maximum `aggregate3` sub-calls packed into one `eth_call`. Bounds each call's response/gas so a
/// dense chain (e.g. Arbitrum) cannot produce a single multicall the node rejects.
const MULTICALL_CHUNK_SIZE: usize = 500;
/// Maximum `eth_call` entries packed into one JSON-RPC batch (one HTTP round-trip). Bounds the batch
/// payload; call sets larger than `MULTICALL_CHUNK_SIZE * MULTICALL_MAX_BATCH_ITEMS` span several
/// batches.
const MULTICALL_MAX_BATCH_ITEMS: usize = 3;
/// Maximum batch requests dispatched concurrently per fetch. Bounds simultaneous HTTP round-trips
/// (provider RPS / compute-unit limit) — distinct from [`MULTICALL_MAX_BATCH_ITEMS`], which bounds a
/// single request's response size.
const MULTICALL_MAX_CONCURRENT_BATCHES: usize = 4;

/// Maximum bytes of an HTTP error-response body retained in a [`ClientEvmError::HttpStatus`]. Bounds
/// the log line so a large HTML/JSON error page cannot flood it.
const MAX_ERROR_BODY_LEN: usize = 512;

/// Posts a JSON-RPC request and returns the parsed response value. Status-as-error is disabled per
/// request so a non-2xx response is not collapsed into an opaque `ureq` status error: its body —
/// which carries the provider's actual reason (response-size cap, execution timeout, upstream down,
/// rate limit) — is captured into [`ClientEvmError::HttpStatus`]. Genuine transport failures
/// (connect, TLS, timeout, DNS) surface as [`ClientEvmError::HttpTransport`].
fn send_rpc_request(
    agent: &ureq::Agent,
    endpoint: &str,
    request: &impl serde::Serialize,
) -> Result<Value, ClientEvmError> {
    let mut response = agent
        .post(endpoint)
        .config()
        .http_status_as_error(false)
        .build()
        .send_json(request)
        .map_err(ClientEvmError::HttpTransport)?;

    let status = response.status();
    if !status.is_success() {
        let body = response
            .body_mut()
            .read_to_string()
            .unwrap_or_else(|error| format!("<unreadable body: {error}>"));
        return Err(ClientEvmError::HttpStatus {
            status: status.as_u16(),
            body: sanitize_error_body(&body),
        });
    }

    response
        .body_mut()
        .read_json::<Value>()
        .map_err(|error| ClientEvmError::MalformedResponse {
            context: "http json".to_owned(),
            detail: error.to_string(),
        })
}

/// Collapses whitespace runs to single spaces and truncates to [`MAX_ERROR_BODY_LEN`] characters,
/// appending an ellipsis when cut, so a captured error body stays on a single log line.
fn sanitize_error_body(body: &str) -> String {
    let collapsed = body.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.chars().count() > MAX_ERROR_BODY_LEN {
        let truncated: String = collapsed.chars().take(MAX_ERROR_BODY_LEN).collect();
        format!("{truncated}…")
    } else {
        collapsed
    }
}

/// Dispatches one JSON-RPC batch (a group of `aggregate3` chunks) and returns its results flattened
/// in chunk order. Self-contained: the request carries its own local ids and is validated against its
/// own per-chunk counts, so batches share no state and can run concurrently.
fn run_multicall_batch(
    agent: &ureq::Agent,
    pool: &EndpointPool,
    block: MulticallBlock,
    batch: &[&[MulticallCall]],
) -> Result<Vec<MulticallCallResult>, ClientEvmError> {
    let request = build_multicall3_batch_request(block, batch);
    let expected_counts: Vec<usize> = batch.iter().map(|chunk| chunk.len()).collect();
    pool.with_failover(|endpoint| {
        let response_value = send_rpc_request(agent, endpoint, &request)?;
        parse_multicall3_batch_response(&response_value, &expected_counts)
    })
}

/// Executes a Multicall3 `aggregate3` over an arbitrary number of calls by chunking them into
/// bounded sub-multicalls and dispatching each group of chunks as a single JSON-RPC batch. Batches
/// are dispatched concurrently in windows of [`MULTICALL_MAX_CONCURRENT_BATCHES`] to overlap their
/// HTTP round-trips while bounding in-flight requests. Returns the call results flattened in input
/// order, exactly as a single `aggregate3` would — callers stay oblivious to the chunking.
fn aggregate3_batched(
    agent: &ureq::Agent,
    pool: &EndpointPool,
    block: MulticallBlock,
    calls: &[MulticallCall],
) -> Result<Vec<MulticallCallResult>, ClientEvmError> {
    if calls.is_empty() {
        return Ok(Vec::new());
    }

    let chunks: Vec<&[MulticallCall]> = calls.chunks(MULTICALL_CHUNK_SIZE).collect();
    let batches: Vec<&[&[MulticallCall]]> = chunks.chunks(MULTICALL_MAX_BATCH_ITEMS).collect();
    let mut results = Vec::with_capacity(calls.len());

    for window in batches.chunks(MULTICALL_MAX_CONCURRENT_BATCHES) {
        let window_results = thread::scope(|scope| {
            let handles: Vec<_> = window
                .iter()
                .map(|batch| scope.spawn(move || run_multicall_batch(agent, pool, block, batch)))
                .collect();
            handles
                .into_iter()
                .map(|handle| {
                    handle.join().unwrap_or_else(|_| {
                        Err(ClientEvmError::MalformedResponse {
                            context: "multicall3 batch".to_owned(),
                            detail: "batch worker thread panicked".to_owned(),
                        })
                    })
                })
                .collect::<Vec<Result<Vec<MulticallCallResult>, ClientEvmError>>>()
        });

        for batch_result in window_results {
            results.extend(batch_result?);
        }
    }

    Ok(results)
}

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
    endpoints: &ChainEndpoints,
    chain: ChainKey,
    block_hash: BlockHash,
) -> Result<Option<ClientHead>, ClientEvmError> {
    let pool = endpoints.pool(chain)?;
    let request = build_block_header_request(HTTP_REQUEST_ID, block_hash);
    pool.with_failover(|endpoint| {
        let response_value = send_rpc_request(agent, endpoint, &request)?;
        parse_block_header_response(&response_value, HTTP_REQUEST_ID, block_hash)
    })
}

pub fn fetch_block_logs(
    agent: &ureq::Agent,
    endpoints: &ChainEndpoints,
    chain: ChainKey,
    block_hash: BlockHash,
) -> Result<Vec<PoolLog>, ClientEvmError> {
    let pool = endpoints.pool(chain)?;
    let request = build_block_logs_request(HTTP_REQUEST_ID, block_hash);
    pool.with_failover(|endpoint| {
        let response_value = send_rpc_request(agent, endpoint, &request)?;
        parse_block_logs_response(&response_value, HTTP_REQUEST_ID, block_hash)
    })
}

pub fn fetch_pool_candidates_in_range(
    agent: &ureq::Agent,
    endpoints: &ChainEndpoints,
    chain: ChainKey,
    from_block: u64,
) -> Result<Vec<RangeLogBlock>, ClientEvmError> {
    let pool = endpoints.pool(chain)?;
    let request = build_pool_logs_range_request(HTTP_REQUEST_ID, from_block);
    pool.with_failover(|endpoint| {
        let response_value = send_rpc_request(agent, endpoint, &request)?;
        parse_pool_logs_range_response(&response_value, HTTP_REQUEST_ID)
    })
}

pub fn fetch_pool_data(
    agent: &ureq::Agent,
    endpoints: &ChainEndpoints,
    chain: ChainKey,
    at: BlockHash,
    pools: HashSet<PoolRef>,
) -> Result<HashMap<PoolRef, PoolDataResult>, ClientEvmError> {
    if pools.is_empty() {
        return Ok(HashMap::new());
    }

    let endpoint_pool = endpoints.pool(chain)?;
    let state_view = crate::uniswap_v4::state_view_address(chain);
    let plans = sorted_pool_data_call_plans(pools, state_view);
    let calls = pool_data_multicall_calls(&plans);
    let results = aggregate3_batched(agent, endpoint_pool, MulticallBlock::Hash(at), &calls)?;

    Ok(decode_pool_data_results(&plans, &results))
}

pub fn fetch_pool_metadata(
    agent: &ureq::Agent,
    endpoints: &ChainEndpoints,
    chain: ChainKey,
    // Pool metadata (token0/token1/fee/tickSpacing) is immutable, so it is read at `latest` rather
    // than the anchor block: this avoids historical-state execution that pruned free-tier upstreams
    // reject outright. The anchor `_at` is retained for request plumbing/symmetry with pool data.
    _at: BlockHash,
    candidates: HashSet<ProtocolPoolKey>,
) -> Result<HashMap<ProtocolPoolKey, PoolMetadataResult>, ClientEvmError> {
    if candidates.is_empty() {
        return Ok(HashMap::new());
    }

    let endpoint_pool = endpoints.pool(chain)?;
    let candidates = sorted_pool_candidate_addresses(candidates);
    let calls = pool_metadata_candidate_multicall_calls(&candidates);
    let results = aggregate3_batched(agent, endpoint_pool, MulticallBlock::Latest, &calls)?;
    let (mut metadata_results, factory_inputs) =
        decode_pool_metadata_candidate_results(&candidates, &results);

    if factory_inputs.is_empty() {
        return Ok(metadata_results);
    }

    let factory_calls = pool_metadata_factory_multicall_calls(chain, &factory_inputs);
    let factory_results =
        aggregate3_batched(agent, endpoint_pool, MulticallBlock::Latest, &factory_calls)?;

    metadata_results.extend(decode_pool_metadata_factory_results(
        &factory_inputs,
        &factory_results,
    ));

    Ok(metadata_results)
}

pub fn fetch_token_metadata(
    agent: &ureq::Agent,
    endpoints: &ChainEndpoints,
    chain: ChainKey,
    // Token metadata (decimals) is immutable, so it is read at `latest` rather than the anchor block
    // for the same reason as pool metadata. The anchor `_at` is retained for request plumbing.
    _at: BlockHash,
    tokens: HashSet<TokenAddress>,
) -> Result<HashMap<TokenAddress, TokenMetadataResult>, ClientEvmError> {
    if tokens.is_empty() {
        return Ok(HashMap::new());
    }

    let endpoint_pool = endpoints.pool(chain)?;
    let tokens = sorted_token_addresses(tokens);
    let calls = token_metadata_multicall_calls(&tokens);
    let results = aggregate3_batched(agent, endpoint_pool, MulticallBlock::Latest, &calls)?;

    Ok(decode_token_metadata_results(&tokens, &results))
}

fn sorted_pool_candidate_addresses(
    candidates: HashSet<ProtocolPoolKey>,
) -> Vec<ProtocolPoolKey> {
    // This RPC path validates pools via the v3 factory and per-pool `token0`/`token1`/`fee` reads,
    // so it applies only to v3 candidates; v4 metadata is event-sourced. Drop any non-v3 candidate
    // defensively (none reach here today).
    let mut candidates = candidates
        .into_iter()
        .filter(|candidate| candidate.uniswap_v3_address().is_some())
        .collect::<Vec<_>>();
    candidates.sort();
    candidates
}

/// The v3 pool contract address a candidate's metadata calls target. `sorted_pool_candidate_addresses`
/// has already dropped non-v3 candidates, so the zero-address fallback is unreachable; it keeps this
/// panic-free regardless.
fn candidate_target(candidate: ProtocolPoolKey) -> Address {
    candidate.uniswap_v3_address().unwrap_or(Address::ZERO)
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
    candidates: &[ProtocolPoolKey],
) -> Vec<MulticallCall> {
    candidates
        .iter()
        .flat_map(|candidate| {
            let target = candidate_target(*candidate);
            [
                MulticallCall {
                    target,
                    call_data: Bytes::from(crate::uniswap_v3::token0Call {}.abi_encode()),
                },
                MulticallCall {
                    target,
                    call_data: Bytes::from(crate::uniswap_v3::token1Call {}.abi_encode()),
                },
                MulticallCall {
                    target,
                    call_data: Bytes::from(crate::uniswap_v3::feeCall {}.abi_encode()),
                },
            ]
        })
        .collect()
}

fn pool_metadata_factory_multicall_calls(
    chain: ChainKey,
    metadata: &[(ProtocolPoolKey, PoolMetadata)],
) -> Vec<MulticallCall> {
    let factory = crate::uniswap_v3::v3_factory_address(chain);
    metadata
        .iter()
        .map(|(_, metadata)| MulticallCall {
            target: factory,
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
    candidates: &[ProtocolPoolKey],
    results: &[MulticallCallResult],
) -> (
    HashMap<ProtocolPoolKey, PoolMetadataResult>,
    Vec<(ProtocolPoolKey, PoolMetadata)>,
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
    let tier = UniswapV3Fee::try_from_pips(fee_pips)
        .ok_or(PoolMetadataFailure::UnsupportedFee(fee_pips))?;

    Ok(PoolMetadata {
        token0,
        token1,
        fee: PoolFee::Tiered(tier),
    })
}

fn decode_pool_metadata_factory_results(
    metadata: &[(ProtocolPoolKey, PoolMetadata)],
    results: &[MulticallCallResult],
) -> HashMap<ProtocolPoolKey, PoolMetadataResult> {
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
    candidate: ProtocolPoolKey,
    metadata: &PoolMetadata,
    result: Option<&MulticallCallResult>,
) -> PoolMetadataResult {
    let returned = decode_pool_metadata_factory_pool(result)?;

    if returned == Address::ZERO {
        Err(PoolMetadataFailure::FactoryReturnedZero)
    } else if returned != candidate_target(candidate) {
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

/// Each pool paired with the two multicall calls (state + liquidity) that read its live state,
/// sorted by `PoolRef` for deterministic multicall ordering. v3 pools read `slot0()`/`liquidity()`
/// from their own contract; v4 pools read `getSlot0(id)`/`getLiquidity(id)` from the chain's
/// `StateView`. `PoolRef`'s `Ord` keeps all v3 pools (ordered by address) ahead of v4, so v3 ordering
/// is unchanged. A v4 pool on a chain with no known `StateView` produces no plan and is skipped — it
/// has no contract to target; this cannot happen today since v4 is Ethereum-only.
fn sorted_pool_data_call_plans(
    pools: HashSet<PoolRef>,
    state_view: Option<Address>,
) -> Vec<(PoolRef, [MulticallCall; 2])> {
    let mut plans = pools
        .into_iter()
        .filter_map(|pool| pool_data_call_plan(pool, state_view).map(|calls| (pool, calls)))
        .collect::<Vec<_>>();
    plans.sort_by(|left, right| left.0.cmp(&right.0));
    plans
}

/// The state + liquidity calls for one pool, or `None` for a v4 pool whose chain has no `StateView`.
fn pool_data_call_plan(pool: PoolRef, state_view: Option<Address>) -> Option<[MulticallCall; 2]> {
    match pool.key {
        ProtocolPoolKey::UniswapV3(address) => Some([
            MulticallCall {
                target: address,
                call_data: Bytes::from(crate::uniswap_v3::slot0Call {}.abi_encode()),
            },
            MulticallCall {
                target: address,
                call_data: Bytes::from(crate::uniswap_v3::liquidityCall {}.abi_encode()),
            },
        ]),
        ProtocolPoolKey::UniswapV4(pool_id) => state_view.map(|target| {
            [
                MulticallCall {
                    target,
                    call_data: Bytes::from(
                        crate::uniswap_v4::getSlot0Call { poolId: pool_id.0 }.abi_encode(),
                    ),
                },
                MulticallCall {
                    target,
                    call_data: Bytes::from(
                        crate::uniswap_v4::getLiquidityCall { poolId: pool_id.0 }.abi_encode(),
                    ),
                },
            ]
        }),
    }
}

fn pool_data_multicall_calls(plans: &[(PoolRef, [MulticallCall; 2])]) -> Vec<MulticallCall> {
    plans
        .iter()
        .flat_map(|(_, calls)| calls.clone())
        .collect()
}

fn decode_pool_data_results(
    plans: &[(PoolRef, [MulticallCall; 2])],
    results: &[MulticallCallResult],
) -> HashMap<PoolRef, PoolDataResult> {
    let mut result_chunks = results.chunks(2);

    plans
        .iter()
        .map(|(pool, _)| {
            let result_chunk = result_chunks.next().unwrap_or(&[]);
            let state = result_chunk.first();
            let liquidity = result_chunk.get(1);
            // Both protocols yield the same `PoolState`; only the ABI of the state read differs, so
            // dispatch the decode on the pool's protocol.
            let pool_data = match pool.key {
                ProtocolPoolKey::UniswapV3(_) => decode_pool_data_result(state, liquidity),
                ProtocolPoolKey::UniswapV4(_) => decode_v4_pool_data_result(state, liquidity),
            };
            (*pool, pool_data)
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

fn decode_v4_pool_data_result(
    slot0: Option<&MulticallCallResult>,
    liquidity: Option<&MulticallCallResult>,
) -> PoolDataResult {
    let slot0 = decode_v4_slot0(slot0)?;
    let liquidity = decode_v4_liquidity(liquidity)?;

    Ok(PoolState {
        sqrt_price_x96: slot0.sqrtPriceX96,
        tick: slot0.tick,
        liquidity,
    })
}

fn decode_v4_slot0(
    result: Option<&MulticallCallResult>,
) -> Result<crate::uniswap_v4::getSlot0Return, PoolDataFailure> {
    decode_multicall_result(result, PoolDataCall::Slot0, |return_data| {
        crate::uniswap_v4::getSlot0Call::abi_decode_returns(return_data)
    })
}

fn decode_v4_liquidity(result: Option<&MulticallCallResult>) -> Result<u128, PoolDataFailure> {
    decode_multicall_result(result, PoolDataCall::Liquidity, |return_data| {
        crate::uniswap_v4::getLiquidityCall::abi_decode_returns(return_data)
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
    endpoints: &ChainEndpoints,
    chain: ChainKey,
) -> Result<Option<ClientHead>, ClientEvmError> {
    let pool = endpoints.pool(chain)?;
    let request = build_finalized_block_header_request(HTTP_REQUEST_ID);
    pool.with_failover(|endpoint| {
        let response_value = send_rpc_request(agent, endpoint, &request)?;
        parse_block_header_response_by_id(&response_value, HTTP_REQUEST_ID)
    })
}

pub fn subscribe_new_heads<T, F>(
    ws_url: &str,
    sender: &Sender<T>,
    map_event: F,
) -> Result<(), ClientEvmError>
where
    F: Fn(ClientEvent) -> Option<T>,
{
    let (mut socket, _) =
        connect(ws_url).map_err(|ws_error| ClientEvmError::WebSocketError(ws_error))?;

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
            return Err(ClientEvmError::MalformedResponse {
                context: "subscription".to_owned(),
                detail: "websocket closed before subscription response".to_owned(),
            });
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

/// Subscribes to the live `logs` stream for pool events, mirroring [`subscribe_new_heads`]. Each
/// state-relevant log is decoded and forwarded as a [`ClientEvent::PoolLogObserved`] for its block.
pub fn subscribe_pool_events<T, F>(
    ws_url: &str,
    sender: &Sender<T>,
    map_event: F,
) -> Result<(), ClientEvmError>
where
    F: Fn(ClientEvent) -> Option<T>,
{
    let (mut socket, _) =
        connect(ws_url).map_err(|ws_error| ClientEvmError::WebSocketError(ws_error))?;

    let subscribe_request = build_pool_events_subscribe_request(SUBSCRIBE_REQUEST_ID);
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

    read_pool_log_events(&mut socket, &subscription_id, sender, &map_event)
}

fn read_pool_log_events<T, F>(
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
            read_json_rpc_message::<SubscriptionNotification<Log>>(socket)?
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
            continue;
        }

        let log = sub_notification.params.result;
        // Drop logs we cannot attribute to a block or that are not state-relevant pool events.
        let (Some(block_hash), Some(pool_log)) = (log.block_hash, decode_pool_log(&log)) else {
            continue;
        };

        send_event(
            sender,
            ClientEvent::PoolLogObserved {
                subscription_id: sub_notification.params.subscription,
                block_hash,
                log: pool_log,
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
                let message = serde_json::from_slice::<T>(text.as_bytes()).map_err(|error| {
                    ClientEvmError::MalformedResponse {
                        context: "subscription".to_owned(),
                        detail: error.to_string(),
                    }
                })?;
                return Ok(Some(message));
            }
            Ok(Message::Binary(bytes)) => {
                let text = str::from_utf8(bytes.as_ref()).map_err(|error| {
                    ClientEvmError::MalformedResponse {
                        context: "subscription".to_owned(),
                        detail: error.to_string(),
                    }
                })?;
                let message = serde_json::from_str::<T>(text).map_err(|error| {
                    ClientEvmError::MalformedResponse {
                        context: "subscription".to_owned(),
                        detail: error.to_string(),
                    }
                })?;
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
        PoolRef, ProtocolPoolKey, PoolDataCall, PoolDataFailure, PoolMetadataCall,
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
        let endpoints = endpoints_for(&http_url);
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_block_header(&agent, &endpoints, ChainKey::Ethereum, block_hash);

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
    fn fetch_block_header_maps_transport_failure_to_http_transport_error() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("test listener must bind");
        let address = listener
            .local_addr()
            .expect("test listener must have local address");
        drop(listener);
        let endpoints = endpoints_for(&format!("http://{address}"));
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_block_header(
            &agent,
            &endpoints,
            ChainKey::Ethereum,
            B256::with_last_byte(1),
        );

        assert!(matches!(result, Err(ClientEvmError::HttpTransport(_))));
    }

    #[test]
    fn fetch_block_header_maps_non_2xx_status_to_http_status_with_body() {
        let (http_url, server) = spawn_http_status_server(
            "500 Internal Server Error",
            "{\"error\":\"response size exceeded\"}",
        );
        let endpoints = endpoints_for(&http_url);
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_block_header(
            &agent,
            &endpoints,
            ChainKey::Ethereum,
            B256::with_last_byte(1),
        );

        assert!(matches!(
            result,
            Err(ClientEvmError::HttpStatus { status: 500, ref body })
                if body.contains("response size exceeded")
        ));
        server.join().expect("server thread must complete");
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
        let endpoints = endpoints_for(&http_url);
        let agent = ureq::Agent::new_with_defaults();

        let candidates = fetch_block_logs(&agent, &endpoints, ChainKey::Ethereum, block_hash)
            .expect("logs fetch")
            .iter()
            .map(|log| ProtocolPoolKey::UniswapV3(log.pool.uniswap_v3_address().expect("v3 pool")))
            .collect::<HashSet<_>>();

        assert_eq!(
            candidates,
            HashSet::from([
                ProtocolPoolKey::UniswapV3(first_pool),
                ProtocolPoolKey::UniswapV3(second_pool),
            ])
        );

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
                        .chain(crate::uniswap_v4::pool_event_signature_hashes())
                        .map(|topic| topic.to_string())
                        .collect::<Vec<_>>()]
                }]
            })
        );
        server.join().expect("server thread must complete");
    }

    #[test]
    fn fetch_block_logs_maps_transport_failure_to_http_transport_error() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("test listener must bind");
        let address = listener
            .local_addr()
            .expect("test listener must have local address");
        drop(listener);
        let endpoints = endpoints_for(&format!("http://{address}"));
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_block_logs(
            &agent,
            &endpoints,
            ChainKey::Ethereum,
            B256::with_last_byte(1),
        );

        assert!(matches!(result, Err(ClientEvmError::HttpTransport(_))));
    }

    #[test]
    fn fetch_pool_data_with_empty_pool_set_returns_empty_results_without_http() {
        let endpoints = endpoints_for("http://127.0.0.1:9");
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_pool_data(
            &agent,
            &endpoints,
            ChainKey::Ethereum,
            B256::with_last_byte(1),
            HashSet::new(),
        );

        assert!(matches!(result, Ok(ref pools) if pools.is_empty()));
    }

    #[test]
    fn fetch_pool_data_posts_multicall3_request_and_decodes_pool_states() {
        let at = B256::with_last_byte(1);
        let first_pool = PoolRef::uniswap_v3(Address::with_last_byte(2), ChainKey::Ethereum);
        let second_pool = PoolRef::uniswap_v3(Address::with_last_byte(3), ChainKey::Ethereum);
        let first_state = pool_state(11, -12, 13);
        let second_state = pool_state(21, 22, 23);
        let response = multicall3_response([
            successful_multicall_result(slot0_return_data(&first_state)),
            successful_multicall_result(liquidity_return_data(first_state.liquidity)),
            successful_multicall_result(slot0_return_data(&second_state)),
            successful_multicall_result(liquidity_return_data(second_state.liquidity)),
        ]);
        let (http_url, received_request, server) = spawn_json_rpc_server(response);
        let endpoints = endpoints_for(&http_url);
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_pool_data(
            &agent,
            &endpoints,
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
        assert_multicall_request_at(&request.body, at);
        let calls = multicall_calls_from_request(&request.body);
        let first_target = first_pool.uniswap_v3_address().expect("v3 pool");
        let second_target = second_pool.uniswap_v3_address().expect("v3 pool");
        assert_eq!(
            calls
                .iter()
                .map(|call| (call.target, call.call_data.clone()))
                .collect::<Vec<_>>(),
            vec![
                (first_target, slot0_call_data()),
                (first_target, liquidity_call_data()),
                (second_target, slot0_call_data()),
                (second_target, liquidity_call_data()),
            ]
        );

        server.join().expect("server thread must complete");
    }

    #[test]
    fn fetch_pool_data_returns_per_pool_failure_for_failed_inner_call() {
        let at = B256::with_last_byte(1);
        let first_pool = PoolRef::uniswap_v3(Address::with_last_byte(2), ChainKey::Ethereum);
        let second_pool = PoolRef::uniswap_v3(Address::with_last_byte(3), ChainKey::Ethereum);
        let first_state = pool_state(11, -12, 13);
        let second_state = pool_state(21, 22, 23);
        let response = multicall3_response([
            successful_multicall_result(slot0_return_data(&first_state)),
            successful_multicall_result(liquidity_return_data(first_state.liquidity)),
            successful_multicall_result(slot0_return_data(&second_state)),
            failed_multicall_result(),
        ]);
        let (http_url, _received_request, server) = spawn_json_rpc_server(response);
        let endpoints = endpoints_for(&http_url);
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_pool_data(
            &agent,
            &endpoints,
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
        let pool = PoolRef::uniswap_v3(Address::with_last_byte(2), ChainKey::Ethereum);
        let state = pool_state(11, -12, 13);
        let response = multicall3_response([
            successful_multicall_result(Bytes::from(vec![0x12])),
            successful_multicall_result(liquidity_return_data(state.liquidity)),
        ]);
        let (http_url, _received_request, server) = spawn_json_rpc_server(response);
        let endpoints = endpoints_for(&http_url);
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_pool_data(
            &agent,
            &endpoints,
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
    fn fetch_pool_data_maps_transport_failure_to_http_transport_error() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("test listener must bind");
        let address = listener
            .local_addr()
            .expect("test listener must have local address");
        drop(listener);
        let endpoints = endpoints_for(&format!("http://{address}"));
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_pool_data(
            &agent,
            &endpoints,
            ChainKey::Ethereum,
            B256::with_last_byte(1),
            HashSet::from([PoolRef::uniswap_v3(Address::with_last_byte(2), ChainKey::Ethereum)]),
        );

        assert!(matches!(result, Err(ClientEvmError::HttpTransport(_))));
    }

    #[test]
    fn fetch_pool_data_reads_a_v4_pool_via_the_state_view() {
        let at = B256::with_last_byte(1);
        let pool_id = crate::uniswap_v4::PoolId(B256::with_last_byte(7));
        let pool = PoolRef::uniswap_v4(pool_id, ChainKey::Ethereum);
        let state = pool_state(31, -32, 33);
        let response = multicall3_response([
            successful_multicall_result(v4_get_slot0_return_data(&state)),
            successful_multicall_result(v4_get_liquidity_return_data(state.liquidity)),
        ]);
        let (http_url, received_request, server) = spawn_json_rpc_server(response);
        let endpoints = endpoints_for(&http_url);
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_pool_data(
            &agent,
            &endpoints,
            ChainKey::Ethereum,
            at,
            HashSet::from([pool]),
        );

        let pools = result.expect("pool data fetch must succeed");
        assert_eq!(pools.get(&pool), Some(&Ok(state)));

        let request = received_request
            .recv()
            .expect("server must report received request");
        assert_multicall_request_at(&request.body, at);
        // v4 pools have no per-pool address; their state is read from the singleton StateView.
        let state_view = crate::uniswap_v4::ETHEREUM_UNISWAP_V4_STATE_VIEW_ADDRESS;
        let calls = multicall_calls_from_request(&request.body);
        assert_eq!(
            calls
                .iter()
                .map(|call| (call.target, call.call_data.clone()))
                .collect::<Vec<_>>(),
            vec![
                (state_view, v4_get_slot0_call_data(pool_id)),
                (state_view, v4_get_liquidity_call_data(pool_id)),
            ]
        );

        server.join().expect("server thread must complete");
    }

    #[test]
    fn fetch_pool_data_decodes_mixed_v3_and_v4_pools_with_v3_ordered_first() {
        let at = B256::with_last_byte(1);
        let v3_pool = PoolRef::uniswap_v3(Address::with_last_byte(2), ChainKey::Ethereum);
        let v4_id = crate::uniswap_v4::PoolId(B256::with_last_byte(9));
        let v4_pool = PoolRef::uniswap_v4(v4_id, ChainKey::Ethereum);
        let v3_state = pool_state(11, -12, 13);
        let v4_state = pool_state(21, 22, 23);
        // `PoolRef`'s `Ord` places every v3 pool ahead of v4, so the multicall is ordered v3 then v4.
        let response = multicall3_response([
            successful_multicall_result(slot0_return_data(&v3_state)),
            successful_multicall_result(liquidity_return_data(v3_state.liquidity)),
            successful_multicall_result(v4_get_slot0_return_data(&v4_state)),
            successful_multicall_result(v4_get_liquidity_return_data(v4_state.liquidity)),
        ]);
        let (http_url, received_request, server) = spawn_json_rpc_server(response);
        let endpoints = endpoints_for(&http_url);
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_pool_data(
            &agent,
            &endpoints,
            ChainKey::Ethereum,
            at,
            HashSet::from([v4_pool, v3_pool]),
        );

        let pools = result.expect("pool data fetch must succeed");
        assert_eq!(pools.get(&v3_pool), Some(&Ok(v3_state)));
        assert_eq!(pools.get(&v4_pool), Some(&Ok(v4_state)));

        let request = received_request
            .recv()
            .expect("server must report received request");
        let v3_target = v3_pool.uniswap_v3_address().expect("v3 pool");
        let state_view = crate::uniswap_v4::ETHEREUM_UNISWAP_V4_STATE_VIEW_ADDRESS;
        let calls = multicall_calls_from_request(&request.body);
        assert_eq!(
            calls
                .iter()
                .map(|call| (call.target, call.call_data.clone()))
                .collect::<Vec<_>>(),
            vec![
                (v3_target, slot0_call_data()),
                (v3_target, liquidity_call_data()),
                (state_view, v4_get_slot0_call_data(v4_id)),
                (state_view, v4_get_liquidity_call_data(v4_id)),
            ]
        );

        server.join().expect("server thread must complete");
    }

    #[test]
    fn fetch_pool_data_returns_per_pool_failure_for_failed_v4_inner_call() {
        let at = B256::with_last_byte(1);
        let pool =
            PoolRef::uniswap_v4(crate::uniswap_v4::PoolId(B256::with_last_byte(7)), ChainKey::Ethereum);
        let state = pool_state(31, -32, 33);
        let response = multicall3_response([
            successful_multicall_result(v4_get_slot0_return_data(&state)),
            failed_multicall_result(),
        ]);
        let (http_url, _received_request, server) = spawn_json_rpc_server(response);
        let endpoints = endpoints_for(&http_url);
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_pool_data(
            &agent,
            &endpoints,
            ChainKey::Ethereum,
            at,
            HashSet::from([pool]),
        );

        let pools = result.expect("outer multicall must succeed");
        assert_eq!(
            pools.get(&pool),
            Some(&Err(PoolDataFailure::CallFailed(PoolDataCall::Liquidity)))
        );
        server.join().expect("server thread must complete");
    }

    #[test]
    fn pool_data_call_plans_skip_v4_pools_when_the_chain_has_no_state_view() {
        let v3_pool = PoolRef::uniswap_v3(Address::with_last_byte(2), ChainKey::Ethereum);
        let v4_pool =
            PoolRef::uniswap_v4(crate::uniswap_v4::PoolId(B256::with_last_byte(7)), ChainKey::Arbitrum);

        // With no StateView for the chain the v4 pool has no contract to target and is dropped, while
        // the v3 pool still produces its (address-targeted) plan.
        let plans = sorted_pool_data_call_plans(HashSet::from([v3_pool, v4_pool]), None);

        assert_eq!(
            plans.iter().map(|(pool, _)| *pool).collect::<Vec<_>>(),
            vec![v3_pool]
        );
    }

    #[test]
    fn fetch_pool_metadata_with_empty_candidate_set_returns_empty_results_without_http() {
        let endpoints = endpoints_for("http://127.0.0.1:9");
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_pool_metadata(
            &agent,
            &endpoints,
            ChainKey::Ethereum,
            B256::with_last_byte(1),
            HashSet::new(),
        );

        assert!(matches!(result, Ok(ref metadata) if metadata.is_empty()));
    }

    #[test]
    fn fetch_pool_metadata_posts_candidate_and_factory_multicalls_and_verifies_metadata() {
        let at = B256::with_last_byte(1);
        let first_candidate = ProtocolPoolKey::UniswapV3(Address::with_last_byte(2));
        let second_candidate = ProtocolPoolKey::UniswapV3(Address::with_last_byte(3));
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
            successful_multicall_result(get_pool_return_data(
                first_candidate.uniswap_v3_address().expect("v3 pool"),
            )),
            successful_multicall_result(get_pool_return_data(returned_mismatch)),
        ]);
        let (http_url, received_request, server) =
            spawn_json_rpc_server_sequence(vec![first_response, second_response]);
        let endpoints = endpoints_for(&http_url);
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_pool_metadata(
            &agent,
            &endpoints,
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
                fee: PoolFee::Tiered(first_fee),
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
        assert_multicall_request_at_latest(&first_request.body);
        let first_calls = multicall_calls_from_request(&first_request.body);
        let first_target = first_candidate.uniswap_v3_address().expect("v3 pool");
        let second_target = second_candidate.uniswap_v3_address().expect("v3 pool");
        assert_eq!(
            first_calls
                .iter()
                .map(|call| (call.target, call.call_data.clone()))
                .collect::<Vec<_>>(),
            vec![
                (first_target, token0_call_data()),
                (first_target, token1_call_data()),
                (first_target, fee_call_data()),
                (second_target, token0_call_data()),
                (second_target, token1_call_data()),
                (second_target, fee_call_data()),
            ]
        );

        let second_request = received_request
            .recv()
            .expect("server must report second request");
        assert_multicall_request_at_latest(&second_request.body);
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
        let candidate = ProtocolPoolKey::UniswapV3(Address::with_last_byte(2));
        let response = multicall3_response([
            successful_multicall_result(address_return_data_for_token0(Address::with_last_byte(3))),
            successful_multicall_result(address_return_data_for_token1(Address::with_last_byte(4))),
            successful_multicall_result(fee_return_data(2500)),
        ]);
        let (http_url, received_request, server) = spawn_json_rpc_server_sequence(vec![response]);
        let endpoints = endpoints_for(&http_url);
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_pool_metadata(
            &agent,
            &endpoints,
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
        let failed_candidate = ProtocolPoolKey::UniswapV3(Address::with_last_byte(2));
        let malformed_candidate = ProtocolPoolKey::UniswapV3(Address::with_last_byte(3));
        let response = multicall3_response([
            failed_multicall_result(),
            successful_multicall_result(address_return_data_for_token1(Address::with_last_byte(4))),
            successful_multicall_result(fee_return_data(UniswapV3Fee::Fee500.pips())),
            successful_multicall_result(Bytes::from(vec![0x12])),
            successful_multicall_result(address_return_data_for_token1(Address::with_last_byte(5))),
            successful_multicall_result(fee_return_data(UniswapV3Fee::Fee3000.pips())),
        ]);
        let (http_url, _received_request, server) = spawn_json_rpc_server_sequence(vec![response]);
        let endpoints = endpoints_for(&http_url);
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_pool_metadata(
            &agent,
            &endpoints,
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
        let candidate = ProtocolPoolKey::UniswapV3(Address::with_last_byte(2));
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
        let endpoints = endpoints_for(&http_url);
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_pool_metadata(
            &agent,
            &endpoints,
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
        let failed_candidate = ProtocolPoolKey::UniswapV3(Address::with_last_byte(2));
        let malformed_candidate = ProtocolPoolKey::UniswapV3(Address::with_last_byte(3));
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
        let endpoints = endpoints_for(&http_url);
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_pool_metadata(
            &agent,
            &endpoints,
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
        let endpoints = endpoints_for("http://127.0.0.1:9");
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_token_metadata(
            &agent,
            &endpoints,
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
        let endpoints = endpoints_for(&http_url);
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_token_metadata(
            &agent,
            &endpoints,
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
        assert_multicall_request_at_latest(&request.body);
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
        let endpoints = endpoints_for(&http_url);
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_token_metadata(
            &agent,
            &endpoints,
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
    fn fetch_token_metadata_chunks_oversized_call_set_into_one_batch() {
        // One more token than fits in a single chunk forces the call set to split into two
        // `eth_call`s dispatched as one JSON-RPC batch. The chunk boundary must not disturb
        // positional decoding: every token still resolves to its own decimals.
        let at = B256::with_last_byte(1);
        let token_count = MULTICALL_CHUNK_SIZE + 1;
        let tokens: Vec<TokenAddress> = (0..token_count).map(token_address_from_index).collect();
        let decimals_for = |index: usize| (index % 19) as u8;

        let first_chunk: Vec<MulticallCallResult> = (0..MULTICALL_CHUNK_SIZE)
            .map(|index| successful_multicall_result(decimals_return_data(decimals_for(index))))
            .collect();
        let second_chunk = [successful_multicall_result(decimals_return_data(
            decimals_for(MULTICALL_CHUNK_SIZE),
        ))];
        let response = json!([
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": aggregate3_return_data_for_test(&first_chunk)
            },
            {
                "jsonrpc": "2.0",
                "id": 2,
                "result": aggregate3_return_data_for_test(&second_chunk)
            }
        ]);
        let (http_url, received_request, server) = spawn_json_rpc_server(response);
        let endpoints = endpoints_for(&http_url);
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_token_metadata(
            &agent,
            &endpoints,
            ChainKey::Ethereum,
            at,
            tokens.iter().copied().collect(),
        )
        .expect("token metadata fetch must succeed");

        assert_eq!(result.len(), token_count);
        for (index, token) in tokens.iter().enumerate() {
            assert_eq!(
                result.get(token),
                Some(&Ok(TokenMetadata {
                    decimals: TokenDecimals::try_from_u256(U256::from(decimals_for(index)))
                        .expect("test decimals must be supported"),
                })),
                "token at sorted index {index} must decode to its own decimals"
            );
        }

        let request = received_request
            .recv()
            .expect("server must report received request");
        let entries = request
            .body
            .as_array()
            .expect("request must be a json-rpc batch array");
        assert_eq!(entries.len(), 2, "call set must split into two chunks");
        let calls = multicall_batch_calls_from_request(&request.body);
        assert_eq!(
            calls
                .iter()
                .map(|call| (call.target, call.call_data.clone()))
                .collect::<Vec<_>>(),
            tokens
                .iter()
                .map(|token| (token.0, decimals_call_data()))
                .collect::<Vec<_>>()
        );
        server.join().expect("server thread must complete");
    }

    #[test]
    fn aggregate3_batched_preserves_order_across_batches() {
        // A call set large enough to span several JSON-RPC batches (more than the concurrency
        // window), so batches complete out of order under parallel dispatch. The flattened results
        // must still match input order exactly — each result echoes its own call's data.
        let at = B256::with_last_byte(1);
        let call_count = MULTICALL_CHUNK_SIZE * MULTICALL_MAX_BATCH_ITEMS * 4 + 1;
        let calls: Vec<MulticallCall> = (0..call_count).map(multicall_call_from_index).collect();
        let chunk_count = call_count.div_ceil(MULTICALL_CHUNK_SIZE);
        let batch_count = chunk_count.div_ceil(MULTICALL_MAX_BATCH_ITEMS);
        let (http_url, server) = spawn_concurrent_multicall_echo_server(batch_count);
        let endpoint_pool = EndpointPool::single("drpc", format!("{http_url}/ethereum/api-key"));
        let agent = ureq::Agent::new_with_defaults();

        let results = aggregate3_batched(&agent, &endpoint_pool, MulticallBlock::Hash(at), &calls)
            .expect("batched aggregate must succeed");

        assert_eq!(results.len(), call_count);
        for (index, (call, result)) in calls.iter().zip(results.iter()).enumerate() {
            assert!(result.success, "result {index} must be successful");
            assert_eq!(
                result.return_data, call.call_data,
                "result at index {index} is out of input order"
            );
        }
        server.join().expect("server thread must complete");
    }

    #[test]
    fn aggregate3_batched_surfaces_a_single_batch_error() {
        // Span more than one batch and poison a call in the second batch. The all-or-nothing contract
        // means the whole aggregate fails with that batch's JSON-RPC error, even though the first
        // batch succeeds.
        let at = B256::with_last_byte(1);
        let call_count =
            MULTICALL_CHUNK_SIZE * MULTICALL_MAX_BATCH_ITEMS + MULTICALL_CHUNK_SIZE + 1;
        let mut calls: Vec<MulticallCall> =
            (0..call_count).map(multicall_call_from_index).collect();
        let poison_index = MULTICALL_CHUNK_SIZE * MULTICALL_MAX_BATCH_ITEMS + 1;
        if let Some(call) = calls.get_mut(poison_index) {
            call.target = poison_target();
        }
        let chunk_count = call_count.div_ceil(MULTICALL_CHUNK_SIZE);
        let batch_count = chunk_count.div_ceil(MULTICALL_MAX_BATCH_ITEMS);
        let (http_url, server) = spawn_concurrent_multicall_echo_server(batch_count);
        let endpoint_pool = EndpointPool::single("drpc", format!("{http_url}/ethereum/api-key"));
        let agent = ureq::Agent::new_with_defaults();

        let result = aggregate3_batched(&agent, &endpoint_pool, MulticallBlock::Hash(at), &calls);

        assert!(matches!(result, Err(ClientEvmError::JsonRpcError { .. })));
        server.join().expect("server thread must complete");
    }

    fn token_address_from_index(index: usize) -> TokenAddress {
        let mut bytes = [0u8; 20];
        let suffix = (index as u32 + 1).to_be_bytes();
        bytes[16..].copy_from_slice(&suffix);
        TokenAddress(Address::from(bytes), ChainKey::Ethereum)
    }

    #[test]
    fn fetch_token_metadata_maps_transport_failure_to_http_transport_error() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("test listener must bind");
        let address = listener
            .local_addr()
            .expect("test listener must have local address");
        drop(listener);
        let endpoints = endpoints_for(&format!("http://{address}"));
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_token_metadata(
            &agent,
            &endpoints,
            ChainKey::Ethereum,
            B256::with_last_byte(1),
            HashSet::from([TokenAddress(Address::with_last_byte(2), ChainKey::Ethereum)]),
        );

        assert!(matches!(result, Err(ClientEvmError::HttpTransport(_))));
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
        let endpoints = endpoints_for(&http_url);
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_finalized_block_header(&agent, &endpoints, ChainKey::Ethereum);

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
    fn fetch_finalized_block_header_maps_transport_failure_to_http_transport_error() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("test listener must bind");
        let address = listener
            .local_addr()
            .expect("test listener must have local address");
        drop(listener);
        let endpoints = endpoints_for(&format!("http://{address}"));
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_finalized_block_header(&agent, &endpoints, ChainKey::Ethereum);

        assert!(matches!(result, Err(ClientEvmError::HttpTransport(_))));
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
        // Multicall fetches now dispatch a JSON-RPC batch (one `eth_call` per chunk). A call set this
        // small fits in a single chunk, so the response is a one-entry batch array with id 1.
        json!([{
            "jsonrpc": "2.0",
            "id": 1,
            "result": aggregate3_return_data_for_test(&results)
        }])
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

    fn v4_get_slot0_call_data(pool_id: crate::uniswap_v4::PoolId) -> Bytes {
        Bytes::from(crate::uniswap_v4::getSlot0Call { poolId: pool_id.0 }.abi_encode())
    }

    fn v4_get_liquidity_call_data(pool_id: crate::uniswap_v4::PoolId) -> Bytes {
        Bytes::from(crate::uniswap_v4::getLiquidityCall { poolId: pool_id.0 }.abi_encode())
    }

    fn v4_get_slot0_return_data(pool_state: &PoolState) -> Bytes {
        Bytes::from(crate::uniswap_v4::getSlot0Call::abi_encode_returns(
            &crate::uniswap_v4::getSlot0Return {
                sqrtPriceX96: pool_state.sqrt_price_x96,
                tick: pool_state.tick,
                protocolFee: alloy::primitives::Uint::<24, 1>::ZERO,
                lpFee: alloy::primitives::Uint::<24, 1>::ZERO,
            },
        ))
    }

    fn v4_get_liquidity_return_data(liquidity: u128) -> Bytes {
        Bytes::from(crate::uniswap_v4::getLiquidityCall::abi_encode_returns(
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

    fn first_batch_entry(request: &Value) -> &Value {
        request
            .as_array()
            .and_then(|entries| entries.first())
            .expect("multicall request must be a json-rpc batch array")
    }

    fn assert_multicall_request_block(request: &Value, expected_block: Value) {
        let request = first_batch_entry(request);
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
            Some(&expected_block)
        );
    }

    /// State-sensitive reads (pool data) pin the anchor block hash.
    fn assert_multicall_request_at(request: &Value, at: B256) {
        assert_multicall_request_block(request, json!({ "blockHash": at }));
    }

    /// Immutable reads (pool/token metadata) are served at `latest`.
    fn assert_multicall_request_at_latest(request: &Value) {
        assert_multicall_request_block(request, json!("latest"));
    }

    fn multicall_batch_calls_from_request(request: &Value) -> Vec<MulticallCall> {
        request
            .as_array()
            .expect("multicall request must be a json-rpc batch array")
            .iter()
            .flat_map(multicall_calls_from_entry)
            .collect()
    }

    fn multicall_calls_from_entry(entry: &Value) -> Vec<MulticallCall> {
        let data = entry
            .get("params")
            .and_then(Value::as_array)
            .and_then(|params| params.first())
            .and_then(|call| call.get("data"))
            .cloned()
            .and_then(|data| serde_json::from_value::<Bytes>(data).ok())
            .expect("batch entry must contain multicall data");

        decode_aggregate3_call_data_for_test(&data)
            .expect("batch entry data must decode as aggregate3")
    }

    fn multicall_call_from_index(index: usize) -> MulticallCall {
        let mut bytes = [0u8; 20];
        let suffix = (index as u32 + 1).to_be_bytes();
        bytes[16..].copy_from_slice(&suffix);
        MulticallCall {
            target: Address::from(bytes),
            call_data: Bytes::from((index as u64).to_be_bytes().to_vec()),
        }
    }

    fn poison_target() -> Address {
        Address::from([0xeeu8; 20])
    }

    /// Builds a JSON-RPC batch response for a received multicall batch request: each entry echoes its
    /// chunk's call data back as the per-call return data, so callers can assert results land in input
    /// order. An entry whose chunk contains [`poison_target`] yields a JSON-RPC error instead, modelling
    /// a single failing batch.
    fn multicall_echo_response(request: &Value) -> Value {
        let entries = request
            .as_array()
            .expect("multicall request must be a json-rpc batch array");
        Value::Array(
            entries
                .iter()
                .map(|entry| {
                    let id = entry
                        .get("id")
                        .and_then(Value::as_u64)
                        .expect("batch entry must carry a numeric id");
                    let calls = multicall_calls_from_entry(entry);
                    if calls.iter().any(|call| call.target == poison_target()) {
                        json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "error": { "code": -32000, "message": "poison batch" }
                        })
                    } else {
                        let results: Vec<MulticallCallResult> = calls
                            .iter()
                            .map(|call| successful_multicall_result(call.call_data.clone()))
                            .collect();
                        json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "result": aggregate3_return_data_for_test(&results)
                        })
                    }
                })
                .collect(),
        )
    }

    fn write_json_response<W: Write>(stream: &mut W, response: &Value) {
        let response_body = serde_json::to_vec(response).expect("test response must serialize");
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

    /// Test server that handles each of `connection_count` connections on its own thread, replying via
    /// [`multicall_echo_response`]. Unlike `spawn_json_rpc_server_sequence` (serial accept, response per
    /// accept order), it pairs each response with its own request, so it stays correct when batches are
    /// dispatched concurrently and complete out of order.
    fn spawn_concurrent_multicall_echo_server(connection_count: usize) -> (String, JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("test server must bind");
        let address = listener
            .local_addr()
            .expect("test server must have local address");

        let handle = thread::spawn(move || {
            let mut handlers = Vec::new();
            for _ in 0..connection_count {
                let (mut stream, _) = listener.accept().expect("test server must accept request");
                handlers.push(thread::spawn(move || {
                    let request = read_http_request(&mut stream);
                    let response = multicall_echo_response(&request.body);
                    write_json_response(&mut stream, &response);
                }));
            }
            for handler in handlers {
                handler.join().expect("connection handler must complete");
            }
        });

        (format!("http://{address}"), handle)
    }

    fn multicall_calls_from_request(request: &Value) -> Vec<MulticallCall> {
        let request = first_batch_entry(request);
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

    fn spawn_http_status_server(
        status_line: &'static str,
        body: &'static str,
    ) -> (String, JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("test server must bind");
        let address = listener
            .local_addr()
            .expect("test server must have local address");

        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("test server must accept request");
            let _ = read_http_request(&mut stream);

            let response_headers = format!(
                "HTTP/1.1 {status_line}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );

            stream
                .write_all(response_headers.as_bytes())
                .expect("test server must write response headers");
            stream
                .write_all(body.as_bytes())
                .expect("test server must write response body");
        });

        (format!("http://{address}"), handle)
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

    // Builds a single-endpoint pool whose URL mirrors the dRPC composition (`{http}/ethereum/api-key`),
    // so the fetch tests' path assertions are unchanged by the move to a multi-provider pool.
    fn endpoints_for(http_url: &str) -> ChainEndpoints {
        ChainEndpoints::single(
            ChainKey::Ethereum,
            "drpc",
            format!("{}/ethereum/api-key", http_url.trim_end_matches('/')),
        )
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
        use alloy::primitives::{I256, U160, aliases::I24};
        use alloy::sol_types::SolEvent;

        use crate::uniswap_v3::Swap;

        // A real, decodable Swap log: block-logs parsing now decodes events, so the fixture must
        // carry valid topics and data rather than an empty placeholder.
        let event = Swap {
            sender: Address::with_last_byte(9),
            recipient: Address::with_last_byte(10),
            amount0: I256::ZERO,
            amount1: I256::ZERO,
            sqrtPriceX96: U160::from(1u128),
            liquidity: 1,
            tick: I24::ZERO,
        };
        let log = alloy::rpc::types::Log {
            inner: alloy::primitives::Log {
                address,
                data: event.encode_log_data(),
            },
            block_hash: Some(block_hash),
            block_number: Some(4),
            log_index: Some(7),
            ..Default::default()
        };

        serde_json::to_value(&log).expect("log serializes to json")
    }
}
