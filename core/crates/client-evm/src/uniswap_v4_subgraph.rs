//! Uniswap v4 pool-metadata resolution via a subgraph/indexer (The Graph).
//!
//! A v4 [`PoolId`](crate::uniswap_v4::PoolId) is a one-way hash of its `PoolKey`, and the chain reveals
//! the key only once (in the `Initialize` event), which scrolls out of log retention. An indexer keyed
//! by id has already captured every historical `Initialize`, so it restores the v3 shape: discover a
//! pool from any of its logs, then resolve its metadata here. The indexer is trusted only for
//! *existence* — every returned key is re-hashed and any `PoolId` mismatch is rejected by
//! [`pool_metadata_from_pool_key`], alongside the existing hooked/dynamic-fee/tick-spacing rejections.

use std::collections::{HashMap, HashSet};

use alloy::primitives::{
    Address, B256,
    aliases::{I24, U24},
};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    ChainKey, ClientEvmError, GraphEndpoints, PoolMetadataResult, ProtocolPoolKey,
    uniswap_v4::{PoolId, PoolKey, pool_metadata_from_pool_key},
};

/// Largest value a `uint24` fee can hold; a `feeTier` past this cannot be a real on-chain fee.
const MAX_UINT24: u32 = 0x00_FF_FF_FF;

/// Maximum characters of a captured HTTP error-response body, so a large error page stays on one log
/// line. Mirrors the bound used by the JSON-RPC client.
const MAX_ERROR_BODY_LEN: usize = 512;

/// GraphQL query for the canonical Uniswap v4 subgraph: the `Pool` entities for a set of ids, with the
/// fields needed to reconstruct each `PoolKey`. Same-schema mirrors must serve these field names.
const POOLS_QUERY: &str = "query Pools($ids: [ID!]!) { \
pools(where: { id_in: $ids }) { id token0 { id } token1 { id } feeTier tickSpacing hooks } }";

/// Resolves Uniswap v4 pool metadata for the v4 members of `candidates` from the configured subgraph.
///
/// Non-v4 candidates are ignored. With no v4 candidates — or no subgraph configured for `chain` — this
/// returns an empty map without any request (v4 metadata is simply skipped). Candidates the indexer
/// does not return (indexing lag / not yet indexed) are **omitted** from the map, *not* recorded as
/// errors, so the kernel re-requests them on a later cycle. Per-pool rejections
/// (hooked/dynamic-fee/tick-spacing/id-mismatch) are kept as `Err` entries. A transport, HTTP, or
/// GraphQL-level failure fails the whole call (and fails over across same-schema mirrors first).
pub fn fetch_v4_pool_metadata(
    agent: &ureq::Agent,
    graph_endpoints: &GraphEndpoints,
    chain: ChainKey,
    candidates: HashSet<ProtocolPoolKey>,
) -> Result<HashMap<ProtocolPoolKey, PoolMetadataResult>, ClientEvmError> {
    let requested: HashSet<PoolId> = candidates
        .iter()
        .filter_map(ProtocolPoolKey::uniswap_v4_pool_id)
        .collect();
    if requested.is_empty() {
        return Ok(HashMap::new());
    }

    let Some(pool) = graph_endpoints.pool(chain) else {
        return Ok(HashMap::new());
    };

    let mut ids: Vec<String> = requested.iter().map(pool_id_query_string).collect();
    // Deterministic order so a captured request body is stable to assert against.
    ids.sort();
    let request = json!({ "query": POOLS_QUERY, "variables": { "ids": ids } });

    pool.with_failover(|endpoint| {
        let value = send_graphql_request(agent, endpoint, &request)?;
        decode_v4_pools_response(value, &requested)
    })
}

/// The lowercase, `0x`-prefixed hex of a [`PoolId`], matching how the subgraph keys its `Pool` entities.
fn pool_id_query_string(id: &PoolId) -> String {
    format!("{:#x}", id.0)
}

/// Posts a GraphQL request and returns the parsed response value. Status-as-error is disabled so a
/// non-2xx response is captured as [`ClientEvmError::HttpStatus`] (carrying the body's actual reason —
/// rate limit, gateway down) rather than an opaque transport error; genuine transport faults surface as
/// [`ClientEvmError::HttpTransport`]. Mirrors the JSON-RPC client's `send_rpc_request`.
///
/// Public for the offline `aa-token-vetting` tool, which runs its own queries against the same
/// subgraph endpoints (composed with [`crate::EndpointPool::with_failover`] like
/// [`fetch_v4_pool_metadata`] does).
pub fn send_graphql_request(
    agent: &ureq::Agent,
    endpoint: &str,
    request: &Value,
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
            context: "graph response".to_owned(),
            detail: error.to_string(),
        })
}

/// Collapses whitespace runs and truncates to [`MAX_ERROR_BODY_LEN`] so a captured error body stays on
/// one log line.
fn sanitize_error_body(body: &str) -> String {
    let collapsed = body.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.chars().count() > MAX_ERROR_BODY_LEN {
        let truncated: String = collapsed.chars().take(MAX_ERROR_BODY_LEN).collect();
        format!("{truncated}…")
    } else {
        collapsed
    }
}

#[derive(Debug, Deserialize)]
struct SubgraphResponse {
    data: Option<SubgraphData>,
    errors: Option<Vec<SubgraphError>>,
}

#[derive(Debug, Deserialize)]
struct SubgraphData {
    pools: Vec<SubgraphPool>,
}

#[derive(Debug, Deserialize)]
struct SubgraphError {
    message: String,
}

#[derive(Debug, Deserialize)]
struct SubgraphToken {
    id: Address,
}

/// One `Pool` entity from the subgraph. `fee_tier`/`tick_spacing` are BigInt fields, serialized as
/// decimal strings by the canonical schema.
#[derive(Debug, Deserialize)]
struct SubgraphPool {
    id: B256,
    token0: SubgraphToken,
    token1: SubgraphToken,
    #[serde(rename = "feeTier")]
    fee_tier: String,
    #[serde(rename = "tickSpacing")]
    tick_spacing: String,
    hooks: Address,
}

/// Decodes a subgraph response into per-candidate metadata results. A GraphQL `errors` payload, a
/// response carrying neither `data` nor `errors`, or an unparseable pool field is a (retryable)
/// response-level error; only pools whose id was actually requested are kept, and each is verified by
/// re-hashing its key.
fn decode_v4_pools_response(
    value: Value,
    requested: &HashSet<PoolId>,
) -> Result<HashMap<ProtocolPoolKey, PoolMetadataResult>, ClientEvmError> {
    let response: SubgraphResponse =
        serde_json::from_value(value).map_err(|error| malformed_graph_pools(error.to_string()))?;

    if let Some(errors) = response.errors {
        if !errors.is_empty() {
            let detail = errors
                .into_iter()
                .map(|error| error.message)
                .collect::<Vec<_>>()
                .join("; ");
            return Err(malformed_graph_pools(detail));
        }
    }

    let data = response
        .data
        .ok_or_else(|| malformed_graph_pools("response has neither data nor errors".to_owned()))?;

    let mut metadata = HashMap::new();
    for entry in data.pools {
        let id = PoolId(entry.id);
        // Defensive: keep only pools we asked about, so a mirror that ignores the filter cannot inject
        // unrequested ids.
        if !requested.contains(&id) {
            continue;
        }

        let key = subgraph_pool_key(&entry)?;
        metadata.insert(
            ProtocolPoolKey::UniswapV4(id),
            pool_metadata_from_pool_key(id, &key),
        );
    }

    Ok(metadata)
}

/// Rebuilds a [`PoolKey`] from a subgraph entry, parsing its decimal-string numeric fields. A value
/// that cannot be a real `uint24` fee or `int24` tick spacing is treated as a (retryable) malformed
/// response rather than a per-pool rejection — the indexer returned data that cannot describe any pool.
fn subgraph_pool_key(entry: &SubgraphPool) -> Result<PoolKey, ClientEvmError> {
    let fee_pips: u32 = entry
        .fee_tier
        .parse()
        .map_err(|_| malformed_graph_pools(format!("non-numeric feeTier {}", entry.fee_tier)))?;
    if fee_pips > MAX_UINT24 {
        return Err(malformed_graph_pools(format!(
            "feeTier {fee_pips} exceeds uint24"
        )));
    }

    let tick_spacing_i32: i32 = entry.tick_spacing.parse().map_err(|_| {
        malformed_graph_pools(format!("non-numeric tickSpacing {}", entry.tick_spacing))
    })?;
    let tick_spacing = I24::try_from(tick_spacing_i32).map_err(|_| {
        malformed_graph_pools(format!("tickSpacing {tick_spacing_i32} exceeds int24"))
    })?;

    Ok(PoolKey {
        currency0: entry.token0.id,
        currency1: entry.token1.id,
        fee: U24::from(fee_pips),
        tickSpacing: tick_spacing,
        hooks: entry.hooks,
    })
}

fn malformed_graph_pools(detail: String) -> ClientEvmError {
    ClientEvmError::MalformedResponse {
        context: "graph pools".to_owned(),
        detail,
    }
}

#[cfg(test)]
mod tests {
    use std::{
        io::{Read, Write},
        net::{TcpListener, TcpStream},
        str,
        sync::mpsc::{Receiver, channel},
        thread::{self, JoinHandle},
    };

    use alloy::primitives::{address, b256};
    use serde_json::json;

    use super::*;
    use crate::{PoolFee, PoolMetadata, PoolMetadataFailure, endpoints::is_retryable};

    // The flagship mainnet v4 pool: native ETH / USDC, fee 500, tick spacing 10, no hooks. The id is
    // recomputed from the key, so the vector is self-checking offline.
    const ETH_USDC_POOL_ID: B256 =
        b256!("21c67e77068de97969ba93d4aab21826d33ca12bb9f565d8496e8fda8a82ca27");
    const USDC: Address = address!("a0b86991c6218b36c1d19d4a2e9eb0ce3606eb48");

    fn v4_candidate(id: B256) -> ProtocolPoolKey {
        ProtocolPoolKey::UniswapV4(PoolId(id))
    }

    fn eth_usdc_pool_json(id: B256) -> Value {
        json!({
            "id": format!("{id:#x}"),
            "token0": { "id": format!("{:#x}", Address::ZERO) },
            "token1": { "id": format!("{USDC:#x}") },
            "feeTier": "500",
            "tickSpacing": "10",
            "hooks": format!("{:#x}", Address::ZERO),
        })
    }

    #[test]
    fn resolves_metadata_for_returned_pools() {
        let response = json!({ "data": { "pools": [eth_usdc_pool_json(ETH_USDC_POOL_ID)] } });
        let (url, _requests, server) = spawn_graph_server(vec![response]);
        let graph = GraphEndpoints::single(ChainKey::Ethereum, "thegraph", url);
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_v4_pool_metadata(
            &agent,
            &graph,
            ChainKey::Ethereum,
            HashSet::from([v4_candidate(ETH_USDC_POOL_ID)]),
        )
        .expect("fetch succeeds");

        assert_eq!(
            result.get(&v4_candidate(ETH_USDC_POOL_ID)),
            Some(&Ok(PoolMetadata {
                token0: Address::ZERO,
                token1: USDC,
                fee: PoolFee::Static {
                    pips: 500,
                    tick_spacing: 10,
                },
            }))
        );
        server.join().expect("server thread completes");
    }

    #[test]
    fn sends_the_pools_query_with_requested_ids() {
        let response = json!({ "data": { "pools": [] } });
        let (url, requests, server) = spawn_graph_server(vec![response]);
        let graph = GraphEndpoints::single(ChainKey::Ethereum, "thegraph", url);
        let agent = ureq::Agent::new_with_defaults();

        let _ = fetch_v4_pool_metadata(
            &agent,
            &graph,
            ChainKey::Ethereum,
            HashSet::from([v4_candidate(ETH_USDC_POOL_ID)]),
        )
        .expect("fetch succeeds");

        let body = requests.recv().expect("server reports the request");
        assert_eq!(body["query"], json!(POOLS_QUERY));
        assert_eq!(
            body["variables"]["ids"],
            json!([format!("{ETH_USDC_POOL_ID:#x}")])
        );
        server.join().expect("server thread completes");
    }

    #[test]
    fn rejects_a_dynamic_fee_pool_as_a_per_pool_error() {
        // feeTier with the dynamic-fee high bit set (0x800000 = 8388608); the key is otherwise valid.
        let pool = json!({
            "id": format!("{ETH_USDC_POOL_ID:#x}"),
            "token0": { "id": format!("{:#x}", Address::ZERO) },
            "token1": { "id": format!("{USDC:#x}") },
            "feeTier": "8388608",
            "tickSpacing": "10",
            "hooks": format!("{:#x}", Address::ZERO),
        });
        let (url, _requests, server) =
            spawn_graph_server(vec![json!({ "data": { "pools": [pool] } })]);
        let graph = GraphEndpoints::single(ChainKey::Ethereum, "thegraph", url);
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_v4_pool_metadata(
            &agent,
            &graph,
            ChainKey::Ethereum,
            HashSet::from([v4_candidate(ETH_USDC_POOL_ID)]),
        )
        .expect("fetch succeeds");

        assert_eq!(
            result.get(&v4_candidate(ETH_USDC_POOL_ID)),
            Some(&Err(PoolMetadataFailure::DynamicFee))
        );
        server.join().expect("server thread completes");
    }

    #[test]
    fn omits_candidates_absent_from_the_response() {
        let present = ETH_USDC_POOL_ID;
        let missing = B256::with_last_byte(9);
        let response = json!({ "data": { "pools": [eth_usdc_pool_json(present)] } });
        let (url, _requests, server) = spawn_graph_server(vec![response]);
        let graph = GraphEndpoints::single(ChainKey::Ethereum, "thegraph", url);
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_v4_pool_metadata(
            &agent,
            &graph,
            ChainKey::Ethereum,
            HashSet::from([v4_candidate(present), v4_candidate(missing)]),
        )
        .expect("fetch succeeds");

        assert!(result.contains_key(&v4_candidate(present)));
        // The indexer did not return the second pool: it is omitted, not recorded as an error.
        assert!(!result.contains_key(&v4_candidate(missing)));
        server.join().expect("server thread completes");
    }

    #[test]
    fn graphql_errors_map_to_a_retryable_error() {
        let response = json!({ "errors": [{ "message": "bad indexers" }] });
        let (url, _requests, server) = spawn_graph_server(vec![response]);
        let graph = GraphEndpoints::single(ChainKey::Ethereum, "thegraph", url);
        let agent = ureq::Agent::new_with_defaults();

        let error = fetch_v4_pool_metadata(
            &agent,
            &graph,
            ChainKey::Ethereum,
            HashSet::from([v4_candidate(ETH_USDC_POOL_ID)]),
        )
        .expect_err("graphql errors fail the call");

        assert!(matches!(error, ClientEvmError::MalformedResponse { .. }));
        assert!(is_retryable(&error));
        server.join().expect("server thread completes");
    }

    #[test]
    fn ignores_non_v4_candidates_without_a_request() {
        // No subgraph configured and only a v3 candidate: returns empty without touching the network.
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_v4_pool_metadata(
            &agent,
            &GraphEndpoints::empty(),
            ChainKey::Ethereum,
            HashSet::from([ProtocolPoolKey::UniswapV3(Address::with_last_byte(1))]),
        )
        .expect("fetch succeeds");

        assert!(result.is_empty());
    }

    #[test]
    fn returns_empty_when_no_subgraph_for_chain() {
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_v4_pool_metadata(
            &agent,
            &GraphEndpoints::empty(),
            ChainKey::Ethereum,
            HashSet::from([v4_candidate(ETH_USDC_POOL_ID)]),
        )
        .expect("fetch succeeds");

        assert!(result.is_empty());
    }

    /// Spawns a one-shot-per-response HTTP server returning canned JSON, reporting each request's parsed
    /// JSON body over the channel.
    fn spawn_graph_server(responses: Vec<Value>) -> (String, Receiver<Value>, JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("test server binds");
        let address = listener.local_addr().expect("test server has an address");
        let (sender, receiver) = channel();

        let handle = thread::spawn(move || {
            for response in responses {
                let (mut stream, _) = listener.accept().expect("test server accepts");
                let body = read_request_body(&mut stream);
                sender.send(body).expect("test server reports the request");

                let response_body = serde_json::to_vec(&response).expect("response serializes");
                let headers = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    response_body.len()
                );
                stream.write_all(headers.as_bytes()).expect("write headers");
                stream.write_all(&response_body).expect("write body");
            }
        });

        (format!("http://{address}"), receiver, handle)
    }

    fn read_request_body(stream: &mut TcpStream) -> Value {
        let mut request_bytes = Vec::new();
        let mut buffer = [0; 1024];
        let (body_start, content_length) = loop {
            let bytes_read = stream.read(&mut buffer).expect("read request");
            assert!(bytes_read > 0, "request must contain headers and body");
            request_bytes.extend_from_slice(&buffer[..bytes_read]);

            if let Some(header_end) = request_bytes
                .windows(4)
                .position(|window| window == b"\r\n\r\n")
            {
                let body_start = header_end + 4;
                let headers = str::from_utf8(&request_bytes[..header_end])
                    .expect("request headers are utf-8");
                let content_length = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().ok())
                            .flatten()
                    })
                    .expect("request has content-length");

                if request_bytes.len() >= body_start + content_length {
                    break (body_start, content_length);
                }
            }
        };

        serde_json::from_slice(&request_bytes[body_start..body_start + content_length])
            .expect("request body is json")
    }
}
