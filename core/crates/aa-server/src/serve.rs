//! The pure serving surface: types and functions that turn [`ServerState`] into the response
//! payloads the data plane exposes. No I/O, no transport crate — [`crate::runtime`] binds a blocking
//! HTTP server and calls [`http_response`]. Keeping this pure means the whole response surface is
//! unit-tested before any network dependency enters the crate.
//!
//! The server is a pure data plane: it ships raw chain evidence (blocks, hashes, freshness, and
//! current-tick pool state) and never derived conclusions. `/health` is the freshness snapshot;
//! `POST /slice` returns each requested pool's raw state at the latest snapshot — `sqrt_price_x96`,
//! `tick`, `liquidity` (the three fields the client's optimizer derives reserves and swap caps
//! from), with everything else (reserves, fees, decimals) left to the client.

use std::collections::HashMap;
use std::str::FromStr;

use client_evm::{Address, BlockHash, PoolRef, PoolState, uniswap_v4::PoolId};

use crate::core::{CHAIN, ServerState};

/// A point-in-time projection of the server's servable facts, mirroring [`ServerState`]'s two cases
/// so "no anchor yet" cannot be confused with a real reading. The single projection published by
/// the runtime and read by every request: its `Running` arm carries the fold's full pool overlay so
/// the serve thread (which cannot touch kernel state) can answer both `/health` and `/slice`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ServerSnapshot {
    /// No anchor yet — the initial finalized probe has not landed.
    AwaitingAnchor,
    /// Anchored and warming/serving.
    Running {
        /// Finalized anchor `(hash, number)`.
        finalized: (BlockHash, u64),
        /// Observed canonical tip.
        canonical: BlockHash,
        /// The fold frontier — the block the pool overlay is valid at (may lag `canonical`).
        frontier: BlockHash,
        /// Count of verified (tracked) pools.
        verified_pool_count: usize,
        /// In-flight RPC requests.
        in_flight: usize,
        /// Cumulative WS-miss backstop count.
        ws_miss: u64,
        /// How far the fold frontier lags the observed tip, if derivable.
        behind: Option<usize>,
        /// The projected current-tick pool overlay at `frontier`, keyed by pool identity. A pool
        /// absent here has no resolved state at the frontier (untracked or unresolved) — a `/slice`
        /// request reports it `Incomplete`.
        pools: HashMap<PoolRef, PoolState>,
    },
}

/// A transport-agnostic HTTP response. A future server adapter maps this onto its own response type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HttpResponse {
    pub status: u16,
    pub body: String,
}

/// A `POST /slice` request body: the pool set the client wants current-tick state for.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize)]
pub struct SliceRequest {
    pub pools: Vec<PoolQuery>,
}

/// One pool's on-chain identity on the wire. The chain is implicit (the server is single-chain), so
/// only the protocol-specific key travels. Deserialized from a request and serialized back into each
/// [`PoolSlice`] so a client can match responses to what it asked for.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "protocol", rename_all = "snake_case")]
pub enum PoolQuery {
    /// A Uniswap v3 pool, identified by its own contract address (`0x`-hex).
    UniswapV3 { address: String },
    /// A Uniswap v4 pool, identified by its `PoolId` (`0x`-hex, 32 bytes).
    UniswapV4 { pool_id: String },
}

/// Raw current-tick pool state on the wire. `sqrt_price_x96` (U160) and `liquidity` (u128) exceed a
/// JSON number's safe integer range, so they ride as `0x`-hex strings; `tick` (I24) fits an `i32`.
/// These are the only per-block fields the client's optimizer needs — it derives reserves, swap
/// caps, and fees itself from these plus static metadata.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct WirePoolState {
    pub sqrt_price_x96: String,
    pub tick: i32,
    pub liquidity: String,
}

/// One pool's slice result: its echoed identity flattened with its completeness/state.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct PoolSlice {
    #[serde(flatten)]
    pub key: PoolQuery,
    #[serde(flatten)]
    pub state: PoolCompleteness,
}

/// Whether a requested pool had resolved state at the frontier. `Incomplete` is a raw fact, not a
/// verdict — the server never says *why* (untracked vs unresolved); the client interprets it.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "completeness", rename_all = "snake_case")]
pub enum PoolCompleteness {
    Complete { state: WirePoolState },
    Incomplete,
}

/// The `POST /slice` response: the frontier the state is valid at, its depth below the observed tip,
/// and one entry per requested pool.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct SliceResponse {
    /// The frontier block hash (`0x`-hex) the pool states are valid at.
    pub block_hash: String,
    /// Canonical blocks the observed tip is ahead of the frontier (the freshness/reorg-depth fact).
    pub confirmations: u64,
    pub pools: Vec<PoolSlice>,
}

/// Projects the server state into a [`ServerSnapshot`]. Pure. The `Running` arm runs the single
/// `projected_pool_states` fold (the per-event hotspot) once and keeps both its overlay and frontier
/// so the published snapshot feeds `/health` (scalars) and `/slice` (the overlay) from one fold.
pub fn server_snapshot(state: &ServerState) -> ServerSnapshot {
    match state {
        ServerState::AwaitingAnchor => ServerSnapshot::AwaitingAnchor,
        ServerState::Running(kernel_state) => {
            let (pools, frontier) = kernel_state.projected_pool_states(CHAIN);
            ServerSnapshot::Running {
                finalized: kernel_state.finalized_head(),
                canonical: kernel_state.canonical_head(),
                frontier,
                verified_pool_count: kernel_state.verified_pool_count(),
                in_flight: kernel_state.in_flight_request_count(),
                ws_miss: kernel_state.ws_miss_count(),
                behind: kernel_state.blocks_behind(frontier),
                pools,
            }
        }
    }
}

/// Serializes the health snapshot to JSON. Hand-rolled over a fixed, flat shape — every field is a
/// number, a `0x`-hex block hash, or a fixed status literal, so there is nothing to escape. The
/// `frontier`/`pools` fields carry `/slice`'s payload and are not part of the health view.
fn to_json(snapshot: &ServerSnapshot) -> String {
    match snapshot {
        ServerSnapshot::AwaitingAnchor => r#"{"status":"awaiting_anchor"}"#.to_owned(),
        ServerSnapshot::Running {
            finalized: (finalized_hash, finalized_number),
            canonical,
            verified_pool_count,
            in_flight,
            ws_miss,
            behind,
            ..
        } => {
            let behind = match behind {
                Some(blocks) => blocks.to_string(),
                None => "null".to_owned(),
            };
            format!(
                r#"{{"status":"running","finalized":{{"number":{finalized_number},"hash":"{finalized_hash}"}},"canonical":"{canonical}","pools":{verified_pool_count},"in_flight":{in_flight},"ws_miss":{ws_miss},"behind":{behind}}}"#
            )
        }
    }
}

/// Returns the path portion of a request URL, dropping any `?query` suffix. A transport's request
/// URL carries the query string (e.g. `/health?probe=1`), but [`http_response`] matches exact
/// paths, so the query is stripped before routing. Pure — a thin string split, unit-tested here.
pub fn strip_query(url: &str) -> &str {
    match url.split_once('?') {
        Some((path, _query)) => path,
        None => url,
    }
}

/// Resolves a wire pool identity into a [`PoolRef`] on this server's chain. Fails (a `400`) on
/// malformed hex — a v3 address or a v4 `PoolId` that will not parse.
fn pool_ref_of(query: &PoolQuery) -> Result<PoolRef, ()> {
    match query {
        PoolQuery::UniswapV3 { address } => {
            let address = Address::from_str(address).map_err(|_| ())?;
            Ok(PoolRef::uniswap_v3(address, CHAIN))
        }
        PoolQuery::UniswapV4 { pool_id } => {
            // `BlockHash` is alloy's `B256`, the same 32-byte type a `PoolId` wraps.
            let pool_id = BlockHash::from_str(pool_id).map_err(|_| ())?;
            Ok(PoolRef::uniswap_v4(PoolId(pool_id), CHAIN))
        }
    }
}

/// Renders raw pool state onto the wire: the two big integers as `0x`-hex, the tick as a number.
/// `i32::try_from` is infallible for an `I24` (it is a strict subset); the `unwrap_or` keeps the
/// path panic-free regardless.
fn wire_pool_state(state: &PoolState) -> WirePoolState {
    WirePoolState {
        sqrt_price_x96: format!("{:#x}", state.sqrt_price_x96),
        tick: i32::try_from(state.tick).unwrap_or(0),
        liquidity: format!("{:#x}", state.liquidity),
    }
}

fn bad_request() -> HttpResponse {
    HttpResponse {
        status: 400,
        body: String::new(),
    }
}

/// The pure `POST /slice` handler: parse the pool set, look each up in the published overlay, and
/// answer with raw state or `Incomplete`. `503` before the anchor lands, `400` on a malformed body
/// or pool id.
pub fn slice_response(body: &str, snapshot: &ServerSnapshot) -> HttpResponse {
    let ServerSnapshot::Running {
        frontier,
        behind,
        pools,
        ..
    } = snapshot
    else {
        return HttpResponse {
            status: 503,
            body: r#"{"status":"awaiting_anchor"}"#.to_owned(),
        };
    };

    let request: SliceRequest = match serde_json::from_str(body) {
        Ok(request) => request,
        Err(_) => return bad_request(),
    };

    let mut slices = Vec::with_capacity(request.pools.len());
    for query in request.pools {
        let pool_ref = match pool_ref_of(&query) {
            Ok(pool_ref) => pool_ref,
            Err(()) => return bad_request(),
        };
        let state = match pools.get(&pool_ref) {
            Some(state) => PoolCompleteness::Complete {
                state: wire_pool_state(state),
            },
            None => PoolCompleteness::Incomplete,
        };
        slices.push(PoolSlice { key: query, state });
    }

    let response = SliceResponse {
        block_hash: format!("{frontier:#x}"),
        confirmations: (*behind).unwrap_or(0) as u64,
        pools: slices,
    };
    match serde_json::to_string(&response) {
        Ok(body) => HttpResponse { status: 200, body },
        Err(_) => HttpResponse {
            status: 500,
            body: String::new(),
        },
    }
}

/// The pure request→response decision. `GET /health` returns the snapshot; `POST /slice` returns the
/// pool slice; a wrong method on either is `405`; any other path is `404`. `method` is the HTTP
/// method token (e.g. `"GET"`); `body` is the request body (ignored except by `POST /slice`).
pub fn http_response(
    method: &str,
    path: &str,
    body: &str,
    snapshot: &ServerSnapshot,
) -> HttpResponse {
    match (method, path) {
        ("GET", "/health") => HttpResponse {
            status: 200,
            body: to_json(snapshot),
        },
        ("POST", "/slice") => slice_response(body, snapshot),
        (_, "/health") | (_, "/slice") => HttpResponse {
            status: 405,
            body: String::new(),
        },
        _ => HttpResponse {
            status: 404,
            body: String::new(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{AnchorHeader, ServerInput, server_transition};
    use client_evm::{Bloom, I24, U160};

    fn hash(byte: u8) -> BlockHash {
        BlockHash::with_last_byte(byte)
    }

    /// Empty-inits a `Running` state anchored at `number` (hash derived from `number`).
    fn running_at_anchor(number: u64) -> ServerState {
        let (state, _) = server_transition(
            ServerState::AwaitingAnchor,
            ServerInput::FinalizedHeader(Some(AnchorHeader {
                hash: hash(number as u8),
                number,
            })),
        );
        state
    }

    fn sample_pool_state() -> PoolState {
        PoolState {
            sqrt_price_x96: U160::from(123_456_789_u64),
            tick: I24::try_from(60_i32).expect("60 fits int24"),
            liquidity: 1_000_000_u128,
        }
    }

    /// A `Running` snapshot carrying one seeded pool at `frontier` = hash(101).
    fn running_with_pool(pool_ref: PoolRef, state: PoolState) -> ServerSnapshot {
        let mut pools = HashMap::new();
        pools.insert(pool_ref, state);
        ServerSnapshot::Running {
            finalized: (hash(100), 100),
            canonical: hash(101),
            frontier: hash(101),
            verified_pool_count: 1,
            in_flight: 0,
            ws_miss: 0,
            behind: Some(2),
            pools,
        }
    }

    #[test]
    fn server_snapshot_of_awaiting_is_awaiting() {
        assert_eq!(
            server_snapshot(&ServerState::AwaitingAnchor),
            ServerSnapshot::AwaitingAnchor
        );
    }

    #[test]
    fn server_snapshot_of_empty_running_reports_the_anchor_and_no_pools() {
        let state = running_at_anchor(100);

        assert_eq!(
            server_snapshot(&state),
            ServerSnapshot::Running {
                finalized: (hash(100), 100),
                canonical: hash(100),
                frontier: hash(100),
                verified_pool_count: 0,
                in_flight: 0,
                ws_miss: 0,
                behind: Some(0),
                pools: HashMap::new(),
            }
        );
    }

    #[test]
    fn server_snapshot_wires_each_field_to_its_accessor() {
        // Warm one connected head so at least one field is non-default, then assert the projection
        // reads each field from the matching kernel accessor (no accidental cross-wiring).
        let (state, _) = server_transition(
            running_at_anchor(100),
            ServerInput::Kernel(client_evm::kernel::Event::HeadObserved {
                hash: hash(101),
                parent_hash: hash(100),
                logs_bloom: Bloom::ZERO,
                number: 101,
            }),
        );

        let snapshot = server_snapshot(&state);
        let ServerState::Running(kernel_state) = &state else {
            panic!("expected Running");
        };
        let (pools, frontier) = kernel_state.projected_pool_states(CHAIN);

        assert_eq!(
            snapshot,
            ServerSnapshot::Running {
                finalized: kernel_state.finalized_head(),
                canonical: kernel_state.canonical_head(),
                frontier,
                verified_pool_count: kernel_state.verified_pool_count(),
                in_flight: kernel_state.in_flight_request_count(),
                ws_miss: kernel_state.ws_miss_count(),
                behind: kernel_state.blocks_behind(frontier),
                pools,
            }
        );
    }

    #[test]
    fn to_json_of_awaiting_is_the_fixed_literal() {
        assert_eq!(
            to_json(&ServerSnapshot::AwaitingAnchor),
            r#"{"status":"awaiting_anchor"}"#
        );
    }

    #[test]
    fn to_json_of_running_emits_every_health_field() {
        let snapshot = ServerSnapshot::Running {
            finalized: (hash(100), 100),
            canonical: hash(101),
            frontier: hash(101),
            verified_pool_count: 3,
            in_flight: 2,
            ws_miss: 7,
            behind: Some(5),
            pools: HashMap::new(),
        };

        let expected = format!(
            r#"{{"status":"running","finalized":{{"number":100,"hash":"{}"}},"canonical":"{}","pools":3,"in_flight":2,"ws_miss":7,"behind":5}}"#,
            hash(100),
            hash(101),
        );

        assert_eq!(to_json(&snapshot), expected);
    }

    #[test]
    fn to_json_renders_absent_lag_as_null() {
        let snapshot = ServerSnapshot::Running {
            finalized: (hash(1), 1),
            canonical: hash(1),
            frontier: hash(1),
            verified_pool_count: 0,
            in_flight: 0,
            ws_miss: 0,
            behind: None,
            pools: HashMap::new(),
        };

        assert!(to_json(&snapshot).ends_with(r#""behind":null}"#));
    }

    #[test]
    fn get_health_returns_the_snapshot_json() {
        let snapshot = server_snapshot(&running_at_anchor(100));

        let response = http_response("GET", "/health", "", &snapshot);

        assert_eq!(response.status, 200);
        assert_eq!(response.body, to_json(&snapshot));
    }

    #[test]
    fn get_health_while_awaiting_returns_awaiting_json() {
        let response = http_response("GET", "/health", "", &ServerSnapshot::AwaitingAnchor);

        assert_eq!(response.status, 200);
        assert_eq!(response.body, r#"{"status":"awaiting_anchor"}"#);
    }

    #[test]
    fn unknown_path_is_not_found() {
        let response = http_response("GET", "/pools", "", &ServerSnapshot::AwaitingAnchor);

        assert_eq!(response.status, 404);
        assert!(response.body.is_empty());
    }

    #[test]
    fn wrong_method_on_health_is_method_not_allowed() {
        let response = http_response("POST", "/health", "", &ServerSnapshot::AwaitingAnchor);

        assert_eq!(response.status, 405);
        assert!(response.body.is_empty());
    }

    #[test]
    fn wrong_method_on_slice_is_method_not_allowed() {
        let response = http_response("GET", "/slice", "", &ServerSnapshot::AwaitingAnchor);

        assert_eq!(response.status, 405);
        assert!(response.body.is_empty());
    }

    #[test]
    fn strip_query_drops_the_query_string() {
        assert_eq!(strip_query("/health?a=1&b=2"), "/health");
    }

    #[test]
    fn strip_query_leaves_a_bare_path_unchanged() {
        assert_eq!(strip_query("/health"), "/health");
        assert_eq!(strip_query("/"), "/");
    }

    #[test]
    fn strip_query_handles_an_empty_query() {
        assert_eq!(strip_query("/health?"), "/health");
    }

    #[test]
    fn pool_ref_of_parses_a_v3_address() {
        let query = PoolQuery::UniswapV3 {
            address: "0x1111111111111111111111111111111111111111".to_owned(),
        };

        let pool_ref = pool_ref_of(&query).expect("valid v3 address");
        assert_eq!(
            pool_ref,
            PoolRef::uniswap_v3(
                Address::from_str("0x1111111111111111111111111111111111111111").expect("addr"),
                CHAIN,
            )
        );
    }

    #[test]
    fn pool_ref_of_parses_a_v4_pool_id() {
        let raw = "0x2222222222222222222222222222222222222222222222222222222222222222";
        let query = PoolQuery::UniswapV4 {
            pool_id: raw.to_owned(),
        };

        let pool_ref = pool_ref_of(&query).expect("valid v4 pool_id");
        assert_eq!(
            pool_ref,
            PoolRef::uniswap_v4(PoolId(BlockHash::from_str(raw).expect("b256")), CHAIN)
        );
    }

    #[test]
    fn pool_ref_of_rejects_malformed_hex() {
        assert!(
            pool_ref_of(&PoolQuery::UniswapV3 {
                address: "not-hex".to_owned(),
            })
            .is_err()
        );
        assert!(
            pool_ref_of(&PoolQuery::UniswapV4 {
                pool_id: "0xdeadbeef".to_owned(),
            })
            .is_err()
        );
    }

    #[test]
    fn wire_pool_state_hex_encodes_the_big_integers() {
        let wire = wire_pool_state(&sample_pool_state());

        assert_eq!(
            wire.sqrt_price_x96,
            format!("{:#x}", U160::from(123_456_789_u64))
        );
        assert!(wire.sqrt_price_x96.starts_with("0x"));
        assert_eq!(wire.liquidity, format!("{:#x}", 1_000_000_u128));
        assert!(wire.liquidity.starts_with("0x"));
        assert_eq!(wire.tick, 60);
    }

    #[test]
    fn slice_of_a_present_pool_is_complete_with_hex_state() {
        let pool_ref = PoolRef::uniswap_v3(
            Address::from_str("0x1111111111111111111111111111111111111111").expect("addr"),
            CHAIN,
        );
        let snapshot = running_with_pool(pool_ref, sample_pool_state());
        let body = r#"{"pools":[{"protocol":"uniswap_v3","address":"0x1111111111111111111111111111111111111111"}]}"#;

        let response = slice_response(body, &snapshot);

        assert_eq!(response.status, 200);
        let parsed: serde_json::Value = serde_json::from_str(&response.body).expect("json");
        assert_eq!(parsed["block_hash"], format!("{:#x}", hash(101)));
        assert_eq!(parsed["confirmations"], 2);
        assert_eq!(parsed["pools"][0]["protocol"], "uniswap_v3");
        assert_eq!(parsed["pools"][0]["completeness"], "complete");
        assert_eq!(
            parsed["pools"][0]["state"]["sqrt_price_x96"],
            format!("{:#x}", U160::from(123_456_789_u64))
        );
        assert_eq!(parsed["pools"][0]["state"]["tick"], 60);
    }

    #[test]
    fn slice_of_an_absent_pool_is_incomplete() {
        // A snapshot whose overlay is empty: any requested pool is unresolved at the frontier.
        let snapshot = ServerSnapshot::Running {
            finalized: (hash(100), 100),
            canonical: hash(101),
            frontier: hash(101),
            verified_pool_count: 0,
            in_flight: 0,
            ws_miss: 0,
            behind: Some(0),
            pools: HashMap::new(),
        };
        let body = r#"{"pools":[{"protocol":"uniswap_v3","address":"0x1111111111111111111111111111111111111111"}]}"#;

        let response = slice_response(body, &snapshot);

        assert_eq!(response.status, 200);
        let parsed: serde_json::Value = serde_json::from_str(&response.body).expect("json");
        assert_eq!(parsed["pools"][0]["completeness"], "incomplete");
        assert!(parsed["pools"][0].get("state").is_none());
    }

    #[test]
    fn slice_of_an_empty_pool_set_is_an_empty_array() {
        let snapshot = running_with_pool(
            PoolRef::uniswap_v3(
                Address::from_str("0x1111111111111111111111111111111111111111").expect("addr"),
                CHAIN,
            ),
            sample_pool_state(),
        );

        let response = slice_response(r#"{"pools":[]}"#, &snapshot);

        assert_eq!(response.status, 200);
        let parsed: serde_json::Value = serde_json::from_str(&response.body).expect("json");
        assert_eq!(parsed["pools"].as_array().expect("array").len(), 0);
    }

    #[test]
    fn slice_of_a_malformed_body_is_bad_request() {
        let snapshot = running_with_pool(
            PoolRef::uniswap_v3(
                Address::from_str("0x1111111111111111111111111111111111111111").expect("addr"),
                CHAIN,
            ),
            sample_pool_state(),
        );

        assert_eq!(slice_response("not json", &snapshot).status, 400);
    }

    #[test]
    fn slice_of_a_malformed_pool_id_is_bad_request() {
        let snapshot = running_with_pool(
            PoolRef::uniswap_v3(
                Address::from_str("0x1111111111111111111111111111111111111111").expect("addr"),
                CHAIN,
            ),
            sample_pool_state(),
        );
        let body = r#"{"pools":[{"protocol":"uniswap_v3","address":"nope"}]}"#;

        assert_eq!(slice_response(body, &snapshot).status, 400);
    }

    #[test]
    fn slice_before_the_anchor_is_service_unavailable() {
        let response = slice_response(r#"{"pools":[]}"#, &ServerSnapshot::AwaitingAnchor);

        assert_eq!(response.status, 503);
        assert_eq!(response.body, r#"{"status":"awaiting_anchor"}"#);
    }

    #[test]
    fn post_slice_routes_through_http_response() {
        let snapshot = running_with_pool(
            PoolRef::uniswap_v3(
                Address::from_str("0x1111111111111111111111111111111111111111").expect("addr"),
                CHAIN,
            ),
            sample_pool_state(),
        );
        let body = r#"{"pools":[]}"#;

        assert_eq!(
            http_response("POST", "/slice", body, &snapshot),
            slice_response(body, &snapshot)
        );
    }
}
