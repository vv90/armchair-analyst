//! The pure serving surface: types and functions that turn [`ServerState`] into the response
//! payloads the data plane exposes. No I/O, no transport crate — a later increment binds a blocking
//! HTTP server and calls [`http_response`]. Keeping this pure means the whole response surface is
//! unit-tested before any network dependency enters the crate.
//!
//! The server is a pure data plane: it ships raw chain evidence (blocks, hashes, freshness) and never
//! derived conclusions. This first payload is the health/freshness snapshot — the same facts the
//! runtime smoke log reads.

use client_evm::BlockHash;

use crate::core::{CHAIN, ServerState};

/// A point-in-time projection of the server's freshness facts, mirroring [`ServerState`]'s two cases
/// so "no anchor yet" cannot be confused with a real reading.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HealthSnapshot {
    /// No anchor yet — the initial finalized probe has not landed.
    AwaitingAnchor,
    /// Anchored and warming/serving.
    Running {
        /// Finalized anchor `(hash, number)`.
        finalized: (BlockHash, u64),
        /// Observed canonical tip.
        canonical: BlockHash,
        /// Count of verified (tracked) pools.
        verified_pool_count: usize,
        /// In-flight RPC requests.
        in_flight: usize,
        /// Cumulative WS-miss backstop count.
        ws_miss: u64,
        /// How far the fold frontier lags the observed tip, if derivable.
        behind: Option<usize>,
    },
}

/// A transport-agnostic HTTP response. A future server adapter maps this onto its own response type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HttpResponse {
    pub status: u16,
    pub body: String,
}

/// Projects the server state into a [`HealthSnapshot`]. Pure: the `Running` arm reads the same
/// accessors as the runtime smoke log (`runtime.rs` `observe_state`).
pub fn health_snapshot(state: &ServerState) -> HealthSnapshot {
    match state {
        ServerState::AwaitingAnchor => HealthSnapshot::AwaitingAnchor,
        ServerState::Running(kernel_state) => {
            // The fold frontier the lag is measured against — the block the projected overlay is
            // valid at (its pool overlay is discarded here; only the frontier hash is needed).
            let (_, frontier) = kernel_state.projected_pool_states(CHAIN);
            HealthSnapshot::Running {
                finalized: kernel_state.finalized_head(),
                canonical: kernel_state.canonical_head(),
                verified_pool_count: kernel_state.verified_pool_count(),
                in_flight: kernel_state.in_flight_request_count(),
                ws_miss: kernel_state.ws_miss_count(),
                behind: kernel_state.blocks_behind(frontier),
            }
        }
    }
}

/// Serializes a snapshot to JSON. Hand-rolled over a fixed, flat shape — every field is a number, a
/// `0x`-hex block hash, or a fixed status literal, so there is nothing to escape. A structured
/// payload (the slice surface) will introduce a real serializer; this one deliberately adds no
/// dependency.
fn to_json(snapshot: &HealthSnapshot) -> String {
    match snapshot {
        HealthSnapshot::AwaitingAnchor => r#"{"status":"awaiting_anchor"}"#.to_owned(),
        HealthSnapshot::Running {
            finalized: (finalized_hash, finalized_number),
            canonical,
            verified_pool_count,
            in_flight,
            ws_miss,
            behind,
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

/// The pure request→response decision. `GET /health` returns the snapshot; a wrong method on
/// `/health` is `405`; any other path is `404`. `method` is the HTTP method token (e.g. `"GET"`).
pub fn http_response(method: &str, path: &str, snapshot: &HealthSnapshot) -> HttpResponse {
    match (method, path) {
        ("GET", "/health") => HttpResponse {
            status: 200,
            body: to_json(snapshot),
        },
        (_, "/health") => HttpResponse {
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
    use client_evm::Bloom;

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

    #[test]
    fn health_snapshot_of_awaiting_is_awaiting() {
        assert_eq!(
            health_snapshot(&ServerState::AwaitingAnchor),
            HealthSnapshot::AwaitingAnchor
        );
    }

    #[test]
    fn health_snapshot_of_empty_running_reports_the_anchor_and_no_pools() {
        let state = running_at_anchor(100);

        assert_eq!(
            health_snapshot(&state),
            HealthSnapshot::Running {
                finalized: (hash(100), 100),
                canonical: hash(100),
                verified_pool_count: 0,
                in_flight: 0,
                ws_miss: 0,
                behind: Some(0),
            }
        );
    }

    #[test]
    fn health_snapshot_wires_each_field_to_its_accessor() {
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

        let snapshot = health_snapshot(&state);
        let ServerState::Running(kernel_state) = &state else {
            panic!("expected Running");
        };
        let (_, frontier) = kernel_state.projected_pool_states(CHAIN);

        assert_eq!(
            snapshot,
            HealthSnapshot::Running {
                finalized: kernel_state.finalized_head(),
                canonical: kernel_state.canonical_head(),
                verified_pool_count: kernel_state.verified_pool_count(),
                in_flight: kernel_state.in_flight_request_count(),
                ws_miss: kernel_state.ws_miss_count(),
                behind: kernel_state.blocks_behind(frontier),
            }
        );
    }

    #[test]
    fn to_json_of_awaiting_is_the_fixed_literal() {
        assert_eq!(
            to_json(&HealthSnapshot::AwaitingAnchor),
            r#"{"status":"awaiting_anchor"}"#
        );
    }

    #[test]
    fn to_json_of_running_emits_every_field() {
        let snapshot = HealthSnapshot::Running {
            finalized: (hash(100), 100),
            canonical: hash(101),
            verified_pool_count: 3,
            in_flight: 2,
            ws_miss: 7,
            behind: Some(5),
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
        let snapshot = HealthSnapshot::Running {
            finalized: (hash(1), 1),
            canonical: hash(1),
            verified_pool_count: 0,
            in_flight: 0,
            ws_miss: 0,
            behind: None,
        };

        assert!(to_json(&snapshot).ends_with(r#""behind":null}"#));
    }

    #[test]
    fn get_health_returns_the_snapshot_json() {
        let snapshot = health_snapshot(&running_at_anchor(100));

        let response = http_response("GET", "/health", &snapshot);

        assert_eq!(response.status, 200);
        assert_eq!(response.body, to_json(&snapshot));
    }

    #[test]
    fn get_health_while_awaiting_returns_awaiting_json() {
        let response = http_response("GET", "/health", &HealthSnapshot::AwaitingAnchor);

        assert_eq!(response.status, 200);
        assert_eq!(response.body, r#"{"status":"awaiting_anchor"}"#);
    }

    #[test]
    fn unknown_path_is_not_found() {
        let response = http_response("GET", "/pools", &HealthSnapshot::AwaitingAnchor);

        assert_eq!(response.status, 404);
        assert!(response.body.is_empty());
    }

    #[test]
    fn wrong_method_on_health_is_method_not_allowed() {
        let response = http_response("POST", "/health", &HealthSnapshot::AwaitingAnchor);

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
}
