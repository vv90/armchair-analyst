//! `aa-wire` — the data-plane HTTP wire contract shared by `aa-server` (serialize) and the GUI
//! client (deserialize), giving the request/response shapes a single owner so the two sides can't
//! drift.
//!
//! It owns the DTOs for `POST /slice`, `GET /pools/meta`, and `GET /health`. The types are
//! primitive-only (`String`/`i32`/`u64`/…) — every big integer rides as a `0x`-hex string and every
//! address/hash as its `0x`-hex form — so this crate stays a leaf depending only on `serde`, with no
//! chain-domain dependency. `aa-server` maps its domain types onto these; the client maps them back.
//! Both directions derive `Serialize + Deserialize` so either side can produce or consume any shape.

use serde::{Deserialize, Serialize};

/// A `POST /slice` request body: the pool set the client wants current-tick state for.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SliceRequest {
    pub pools: Vec<PoolQuery>,
}

/// One pool's on-chain identity on the wire. The chain is implicit (the server is single-chain), so
/// only the protocol-specific key travels. Deserialized from a request and serialized back into each
/// [`PoolSlice`] so a client can match responses to what it asked for.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
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
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WirePoolState {
    pub sqrt_price_x96: String,
    pub tick: i32,
    pub liquidity: String,
}

/// One pool's slice result: its echoed identity flattened with its completeness/state.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PoolSlice {
    #[serde(flatten)]
    pub key: PoolQuery,
    #[serde(flatten)]
    pub state: PoolCompleteness,
}

/// Whether a requested pool had resolved state at the frontier. `Incomplete` is a raw fact, not a
/// verdict — the server never says *why* (untracked vs unresolved); the client interprets it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "completeness", rename_all = "snake_case")]
pub enum PoolCompleteness {
    Complete { state: WirePoolState },
    Incomplete,
}

/// The `POST /slice` response: the frontier the state is valid at, its depth below the observed tip,
/// and one entry per requested pool.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SliceResponse {
    /// The frontier block hash (`0x`-hex) the pool states are valid at.
    pub block_hash: String,
    /// Canonical blocks the observed tip is ahead of the frontier (the freshness/reorg-depth fact).
    pub confirmations: u64,
    pub pools: Vec<PoolSlice>,
}

/// One verified pool's static metadata on the wire: its echoed identity flattened with its token
/// pair (`0x`-hex addresses) and fee. `fee_pips`/`tick_spacing` are the protocol-agnostic fee facts
/// (a v3 tier and a v4 static fee both reduce to these), so the client reconstructs swap math
/// without needing to know which protocol produced them (the protocol still travels in `key`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PoolMetaEntry {
    #[serde(flatten)]
    pub key: PoolQuery,
    pub token0: String,
    pub token1: String,
    pub fee_pips: u32,
    pub tick_spacing: u16,
}

/// One token's static metadata on the wire: its `0x`-hex address and validated decimals.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenMetaEntry {
    pub address: String,
    pub decimals: u8,
}

/// The `GET /pools/meta` response: every verified pool's static metadata plus the decimals of every
/// token those pools reference — a self-contained catalog from which a client derives reserves and
/// swap caps (paired with `/slice` state) without any metadata RPC of its own.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PoolsMetaResponse {
    pub pools: Vec<PoolMetaEntry>,
    pub tokens: Vec<TokenMetaEntry>,
}

/// The `GET /health` response: the server's freshness snapshot, internally tagged on `status`.
/// `AwaitingAnchor` is the pre-anchor startup state (no reading yet); `Running` carries the anchor,
/// observed tip, tracked-pool count, and fold-lag facts. Field order matches the emitted JSON.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum HealthResponse {
    /// No anchor yet — the initial finalized probe has not landed.
    AwaitingAnchor,
    /// Anchored and warming/serving.
    Running {
        /// Finalized anchor `(number, hash)`.
        finalized: FinalizedHead,
        /// Observed canonical tip block hash (`0x`-hex).
        canonical: String,
        /// Count of verified (tracked) pools.
        pools: usize,
        /// In-flight RPC requests.
        in_flight: usize,
        /// Cumulative WS-miss backstop count.
        ws_miss: u64,
        /// How far the fold frontier lags the observed tip, if derivable (`null` when not).
        behind: Option<usize>,
    },
}

/// The finalized anchor on the wire: its block `number` and `0x`-hex `hash`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FinalizedHead {
    pub number: u64,
    pub hash: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A serialize → deserialize round trip must recover the original value. This exercises the
    /// deserialize half of every DTO — the direction the client relies on but the server never runs,
    /// including the `#[serde(flatten)]` + internally-tagged combinations (`PoolSlice`,
    /// `PoolMetaEntry`) that are the only non-obvious paths.
    fn round_trips<T>(value: &T)
    where
        T: Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
    {
        let json = serde_json::to_string(value).expect("serialize");
        let back: T = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(&back, value);
    }

    #[test]
    fn slice_request_round_trips() {
        round_trips(&SliceRequest {
            pools: vec![
                PoolQuery::UniswapV3 {
                    address: "0x1111111111111111111111111111111111111111".to_owned(),
                },
                PoolQuery::UniswapV4 {
                    pool_id: "0x2222222222222222222222222222222222222222222222222222222222222222"
                        .to_owned(),
                },
            ],
        });
    }

    #[test]
    fn slice_response_with_complete_and_incomplete_round_trips() {
        round_trips(&SliceResponse {
            block_hash: "0x00000000000000000000000000000000000000000000000000000000000000aa"
                .to_owned(),
            confirmations: 2,
            pools: vec![
                PoolSlice {
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
                },
                PoolSlice {
                    key: PoolQuery::UniswapV4 {
                        pool_id:
                            "0x2222222222222222222222222222222222222222222222222222222222222222"
                                .to_owned(),
                    },
                    state: PoolCompleteness::Incomplete,
                },
            ],
        });
    }

    #[test]
    fn pools_meta_response_round_trips() {
        round_trips(&PoolsMetaResponse {
            pools: vec![
                PoolMetaEntry {
                    key: PoolQuery::UniswapV3 {
                        address: "0x1111111111111111111111111111111111111111".to_owned(),
                    },
                    token0: "0x0000000000000000000000000000000000000001".to_owned(),
                    token1: "0x0000000000000000000000000000000000000002".to_owned(),
                    fee_pips: 3000,
                    tick_spacing: 60,
                },
                PoolMetaEntry {
                    key: PoolQuery::UniswapV4 {
                        pool_id:
                            "0x2222222222222222222222222222222222222222222222222222222222222222"
                                .to_owned(),
                    },
                    token0: "0x0000000000000000000000000000000000000001".to_owned(),
                    token1: "0x0000000000000000000000000000000000000002".to_owned(),
                    fee_pips: 500,
                    tick_spacing: 10,
                },
            ],
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
        });
    }

    #[test]
    fn health_response_awaiting_round_trips() {
        round_trips(&HealthResponse::AwaitingAnchor);
    }

    #[test]
    fn health_response_running_round_trips() {
        round_trips(&HealthResponse::Running {
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
        });
    }

    #[test]
    fn health_response_running_with_absent_lag_round_trips() {
        round_trips(&HealthResponse::Running {
            finalized: FinalizedHead {
                number: 1,
                hash: "0x0000000000000000000000000000000000000000000000000000000000000001"
                    .to_owned(),
            },
            canonical: "0x0000000000000000000000000000000000000000000000000000000000000001"
                .to_owned(),
            pools: 0,
            in_flight: 0,
            ws_miss: 0,
            behind: None,
        });
    }
}
