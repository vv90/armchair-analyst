//! The executor for the reducer's three fetch effects (`FetchMeta`/`FetchHealth`/`FetchSlice`): the
//! HTTP data-plane adapter. It is the one effect edge that is network I/O — an outbound `ureq` client
//! against `aa-server`'s `GET /pools/meta`, `GET /health`, and `POST /slice` — turning the reducer's
//! [`FetchRequest`]s into [`Event::MetaFetched`]/[`Event::HealthFetched`]/[`Event::SliceFetched`],
//! closing the `FetchX → XFetched` half of the loop the [`crate::OptimizerWorker`] closes for `Optimize`.
//!
//! The decision logic (when to poll, which pools to ask for) lives in the reducer, so this adapter is
//! a pure *command executor*: it performs exactly the one request it is told and reports the outcome.
//! [`DataPlaneClient::handle`] is synchronous and total — every transport, HTTP-status, or JSON fault
//! degrades to [`Event::EffectFailed`]`(`[`EffectError::Fetch`]`)` rather than a panic — so it is
//! testable against a loopback socket without a driver. [`run`] is the only threaded part: a thin
//! channel loop around `handle`.

use std::sync::mpsc::{Receiver, Sender};

use aa_wire::{HealthResponse, PoolsMetaResponse, SliceRequest, SliceResponse};
use serde::de::DeserializeOwned;

use crate::pending::FetchId;
use crate::state::{Event, FetchKind};

/// The three data-plane fetches the reducer can ask for — a narrow mirror of [`crate::Effect`]'s fetch
/// variants so the adapter is total over exactly its domain (the driver maps `Effect::FetchMeta { id }
/// -> FetchRequest::Meta { id }`, and so on). Each carries the [`FetchId`] to echo back on the outcome
/// event so a superseded response can be rejected. `Optimize` cannot reach here.
#[derive(Clone, Debug)]
pub enum FetchRequest {
    /// `GET /pools/meta`.
    Meta {
        /// The id to echo on the outcome event.
        id: FetchId,
    },
    /// `GET /health`.
    Health {
        /// The id to echo on the outcome event.
        id: FetchId,
    },
    /// `POST /slice` for the given pool set.
    Slice {
        /// The id to echo on the outcome event.
        id: FetchId,
        /// The pools to request state for.
        request: SliceRequest,
    },
}

/// Outbound HTTP client bound to one `aa-server` base URL (e.g. `http://127.0.0.1:8080`). Owns a
/// shared [`ureq::Agent`]. Pure in the sense the reducer cares about: [`DataPlaneClient::handle`]
/// performs one request and returns exactly one [`Event`], and no fault can escape as a panic.
pub struct DataPlaneClient {
    agent: ureq::Agent,
    base_url: String,
}

impl DataPlaneClient {
    /// A client for the given base URL (scheme + host + port, no trailing slash), e.g.
    /// `http://127.0.0.1:8080`. Paths are appended verbatim.
    pub fn new(base_url: String) -> DataPlaneClient {
        DataPlaneClient {
            agent: ureq::Agent::new_with_defaults(),
            base_url,
        }
    }

    /// Execute one fetch, returning the [`Event`] the reducer must be fed. Total: any transport,
    /// non-2xx status, body-read, or JSON-parse fault becomes [`Event::FetchFailed`], so the adapter
    /// can never take down the loop. The driver reports *what failed and why*; turning that into a
    /// typed [`crate::EffectError`] is the reducer's job, so the request kind is written once here.
    pub fn handle(&self, request: FetchRequest) -> Event {
        match request {
            FetchRequest::Meta { id } => match self.get::<PoolsMetaResponse>("/pools/meta") {
                Ok(response) => Event::MetaFetched { id, response },
                Err(message) => Event::FetchFailed {
                    id,
                    kind: FetchKind::Meta,
                    message,
                },
            },
            FetchRequest::Health { id } => match self.get::<HealthResponse>("/health") {
                Ok(response) => Event::HealthFetched { id, response },
                Err(message) => Event::FetchFailed {
                    id,
                    kind: FetchKind::Health,
                    message,
                },
            },
            FetchRequest::Slice { id, request } => match self.post_slice(&request) {
                Ok(response) => Event::SliceFetched { id, response },
                Err(message) => Event::FetchFailed {
                    id,
                    kind: FetchKind::Slice,
                    message,
                },
            },
        }
    }

    /// One GET + JSON-decode against `path`, yielding the failure's diagnostic text on any fault. A
    /// `ureq::Error` covers both transport faults and non-2xx statuses (ureq surfaces the latter as
    /// `Error::StatusCode`), so every failure funnels through the same `Err(String)`.
    fn get<T: DeserializeOwned>(&self, path: &str) -> Result<T, String> {
        let url = format!("{}{path}", self.base_url);
        let mut response = self
            .agent
            .get(&url)
            .call()
            .map_err(|error| error.to_string())?;
        let body = response
            .body_mut()
            .read_to_string()
            .map_err(|error| error.to_string())?;
        serde_json::from_str::<T>(&body).map_err(|error| error.to_string())
    }

    /// The `POST /slice` counterpart of [`DataPlaneClient::get`]: serialize the request body, POST it,
    /// then read + JSON-decode the response. Same uniform `Err(String)` path.
    fn post_slice(&self, request: &SliceRequest) -> Result<SliceResponse, String> {
        let url = format!("{}/slice", self.base_url);
        let body = serde_json::to_string(request).map_err(|error| error.to_string())?;
        let mut response = self
            .agent
            .post(&url)
            .send(body.as_str())
            .map_err(|error| error.to_string())?;
        let body = response
            .body_mut()
            .read_to_string()
            .map_err(|error| error.to_string())?;
        serde_json::from_str::<SliceResponse>(&body).map_err(|error| error.to_string())
    }
}

/// The threaded shell: execute each incoming fetch on the given client and forward the resulting
/// event. Returns (ending the thread) when either channel closes — the request sender dropped
/// (`recv` errs) or the event receiver dropped (`send` errs) — so a torn-down driver stops the adapter
/// cleanly. Panic-free: both channel ends are handled as `Result`. Mirror of [`crate::run_optimizer`].
pub fn run(client: DataPlaneClient, requests: Receiver<FetchRequest>, events: Sender<Event>) {
    while let Ok(request) = requests.recv() {
        if events.send(client.handle(request)).is_err() {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::thread::{self, JoinHandle};

    use aa_wire::{
        FinalizedHead, PoolCompleteness, PoolMetaEntry, PoolQuery, PoolSlice, TokenMetaEntry,
        WirePoolState,
    };

    use super::*;

    /// Spawns a `tiny_http` server on an ephemeral loopback port that answers every request from
    /// `handler` — a `(method, path, body) -> (status, body)` closure — and returns the port plus its
    /// join handle. Mirrors aa-server's runtime loopback test. The server ends when the last client
    /// drops (the incoming iterator is finite only on shutdown, so tests join detached threads by port
    /// reuse — here each test owns its own server and lets the thread outlive the assertions).
    fn loopback(
        handler: impl Fn(&str, &str, &str) -> (u16, String) + Send + 'static,
    ) -> (u16, JoinHandle<()>) {
        let server = tiny_http::Server::http("127.0.0.1:0").expect("bind ephemeral loopback port");
        let port = server.server_addr().to_ip().expect("ip listen addr").port();
        let handle = thread::spawn(move || {
            for mut request in server.incoming_requests() {
                let method = request.method().as_str().to_owned();
                let path = request.url().to_owned();
                let mut body = String::new();
                let _ = request.as_reader().read_to_string(&mut body);
                let (status, response_body) = handler(&method, &path, &body);
                let response =
                    tiny_http::Response::from_string(response_body).with_status_code(status);
                let _ = request.respond(response);
            }
        });
        (port, handle)
    }

    fn client_for(port: u16) -> DataPlaneClient {
        DataPlaneClient::new(format!("http://127.0.0.1:{port}"))
    }

    /// A fixed fetch id; the adapter only echoes it back, so any value serves the round-trip.
    fn id() -> FetchId {
        FetchId::from_raw_for_test(1)
    }

    fn sample_meta() -> PoolsMetaResponse {
        PoolsMetaResponse {
            pools: vec![PoolMetaEntry {
                key: PoolQuery::UniswapV3 {
                    address: "0x1111111111111111111111111111111111111111".to_owned(),
                },
                token0: "0x0000000000000000000000000000000000000001".to_owned(),
                token1: "0x0000000000000000000000000000000000000002".to_owned(),
                fee_pips: 3000,
                tick_spacing: 60,
            }],
            tokens: vec![
                TokenMetaEntry {
                    address: "0x0000000000000000000000000000000000000001".to_owned(),
                    decimals: 18,
                },
                TokenMetaEntry {
                    address: "0x0000000000000000000000000000000000000002".to_owned(),
                    decimals: 6,
                },
            ],
        }
    }

    fn sample_health() -> HealthResponse {
        HealthResponse::Running {
            finalized: FinalizedHead {
                number: 100,
                hash: "0x0000000000000000000000000000000000000000000000000000000000000064"
                    .to_owned(),
            },
            canonical: "0x0000000000000000000000000000000000000000000000000000000000000065"
                .to_owned(),
            pools: 3,
            in_flight: 2,
            ws_miss: 7,
            behind: Some(5),
        }
    }

    fn sample_slice() -> SliceResponse {
        SliceResponse {
            block_hash: "0x00000000000000000000000000000000000000000000000000000000000000aa"
                .to_owned(),
            confirmations: 2,
            pools: vec![PoolSlice {
                key: PoolQuery::UniswapV3 {
                    address: "0x1111111111111111111111111111111111111111".to_owned(),
                },
                state: PoolCompleteness::Complete {
                    state: WirePoolState {
                        sqrt_price_x96: "0x75bcd15".to_owned(),
                        tick: 60,
                        liquidity: "0xf4240".to_owned(),
                    },
                },
            }],
        }
    }

    fn slice_request() -> SliceRequest {
        SliceRequest {
            pools: vec![PoolQuery::UniswapV3 {
                address: "0x1111111111111111111111111111111111111111".to_owned(),
            }],
        }
    }

    #[test]
    fn meta_success_yields_meta_fetched() {
        let expected = sample_meta();
        let body = serde_json::to_string(&expected).expect("serialize meta");
        let (port, _server) = loopback(move |method, path, _| {
            assert_eq!((method, path), ("GET", "/pools/meta"));
            (200, body.clone())
        });

        let event = client_for(port).handle(FetchRequest::Meta { id: id() });

        assert_eq!(
            event,
            Event::MetaFetched {
                id: id(),
                response: expected
            }
        );
    }

    #[test]
    fn health_success_yields_health_fetched() {
        let expected = sample_health();
        let body = serde_json::to_string(&expected).expect("serialize health");
        let (port, _server) = loopback(move |method, path, _| {
            assert_eq!((method, path), ("GET", "/health"));
            (200, body.clone())
        });

        let event = client_for(port).handle(FetchRequest::Health { id: id() });

        assert_eq!(
            event,
            Event::HealthFetched {
                id: id(),
                response: expected
            }
        );
    }

    #[test]
    fn slice_success_posts_the_request_and_yields_slice_fetched() {
        let expected = sample_slice();
        let request = slice_request();
        let request_for_assert = request.clone();
        let body = serde_json::to_string(&expected).expect("serialize slice");
        let (port, _server) = loopback(move |method, path, request_body| {
            assert_eq!((method, path), ("POST", "/slice"));
            // The client serialized and POSTed exactly the request it was handed.
            let received: SliceRequest =
                serde_json::from_str(request_body).expect("deserialize posted body");
            assert_eq!(received, request_for_assert);
            (200, body.clone())
        });

        let event = client_for(port).handle(FetchRequest::Slice { id: id(), request });

        assert_eq!(
            event,
            Event::SliceFetched {
                id: id(),
                response: expected
            }
        );
    }

    #[test]
    fn non_2xx_status_is_a_recorded_fetch_error() {
        let (port, _server) = loopback(|_, _, _| (500, "boom".to_owned()));

        let event = client_for(port).handle(FetchRequest::Meta { id: id() });

        let Event::FetchFailed { kind, message, .. } = event else {
            panic!("a non-2xx status must be a recorded fetch failure, got {event:?}");
        };
        assert_eq!(kind, FetchKind::Meta);
        assert!(!message.is_empty(), "the failure must carry a diagnostic");
    }

    #[test]
    fn malformed_body_is_a_recorded_fetch_error() {
        let (port, _server) = loopback(|_, _, _| (200, "not json".to_owned()));

        let event = client_for(port).handle(FetchRequest::Health { id: id() });

        let Event::FetchFailed { kind, message, .. } = event else {
            panic!("an undecodable body must be a recorded fetch failure, got {event:?}");
        };
        assert_eq!(kind, FetchKind::Health);
        assert!(!message.is_empty(), "the failure must carry a diagnostic");
    }

    #[test]
    fn connection_failure_is_a_recorded_fetch_error() {
        // Port 1 on loopback has nothing listening: the connection is refused, not a status error.
        let event = DataPlaneClient::new("http://127.0.0.1:1".to_owned())
            .handle(FetchRequest::Meta { id: id() });

        let Event::FetchFailed { kind, message, .. } = event else {
            panic!("a refused connection must be a recorded fetch failure, got {event:?}");
        };
        assert_eq!(kind, FetchKind::Meta);
        assert!(!message.is_empty(), "the failure must carry a diagnostic");
    }

    #[test]
    fn run_loop_processes_requests_in_order_then_exits_on_close() {
        let meta_body = serde_json::to_string(&sample_meta()).expect("serialize meta");
        let health_body = serde_json::to_string(&sample_health()).expect("serialize health");
        let (port, _server) = loopback(move |_, path, _| match path {
            "/pools/meta" => (200, meta_body.clone()),
            "/health" => (200, health_body.clone()),
            _ => (404, String::new()),
        });

        let (request_tx, request_rx) = mpsc::channel();
        let (event_tx, event_rx) = mpsc::channel();
        let worker = thread::spawn(move || run(client_for(port), request_rx, event_tx));

        request_tx
            .send(FetchRequest::Meta { id: id() })
            .expect("send meta");
        request_tx
            .send(FetchRequest::Health { id: id() })
            .expect("send health");

        assert_eq!(
            event_rx.recv().expect("meta event"),
            Event::MetaFetched {
                id: id(),
                response: sample_meta()
            }
        );
        assert_eq!(
            event_rx.recv().expect("health event"),
            Event::HealthFetched {
                id: id(),
                response: sample_health()
            }
        );

        // Dropping the request sender closes the channel, so the worker loop returns and the thread joins.
        drop(request_tx);
        worker.join().expect("adapter thread joins");
    }
}
