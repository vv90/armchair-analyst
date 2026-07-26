//! `aa-client-core` — the headless engine for the GUI client. Owns *all* application logic so the
//! UI stays thin: polling the `aa-server` data plane, the `aa-wire` → `optimization::PoolReserves`
//! adapter, client-side session config (init asset + bridges), the reconcile/optimize loop,
//! candidate tracking, and the `AppState` it projects into an `aa-client-api::ViewModel`.
//!
//! Transport-agnostic: it consumes `AppCommand`s and produces `ViewModel`s and knows nothing about
//! the binding (FFI/`cdylib`) or the UI framework. So far this crate holds the wire→reserves adapter
//! (below), the pure `state` reducer that drives the poll/optimize loop, the `optimizer` worker that
//! executes its `Optimize` effect, the `http` data-plane adapter that executes its fetch effects, and
//! the `runtime` composition root that runs the whole engine on `aa-framework`; the UI `ViewModel`
//! contract lands in a later increment.

mod http;
mod optimizer;
mod pending;
mod runtime;
mod state;

pub use http::{DataPlaneClient, FetchRequest, run as run_data_plane};
pub use optimizer::{OptimizerWorker, run as run_optimizer};
pub use pending::{FetchId, PendingFetches};
pub use runtime::{ClientEngineApp, ClientEngineRuntime, Subscription, run as run_engine};
pub use state::{
    AppState, AwaitStatus, Effect, EffectError, Event, FetchKind, OptimizeCommand, OptimizeStage,
    Phase, SessionConfig, SliceProvenance, slice_request_for, transition,
};

use std::collections::HashMap;
use std::str::FromStr;

use aa_wire::{PoolCompleteness, PoolQuery, PoolsMetaResponse, SliceResponse};
use client_evm::multi_chain_kernel::{PoolReserveProjectionError, pool_reserve_values};
use client_evm::uniswap_v4::PoolId;
use client_evm::{
    Address, BlockHash, ChainKey, PoolFee, PoolRef, PoolState, TokenAddress, TokenDecimals, U160,
    U256,
};
use optimization::{Invertible, PoolReserves};

/// Why a wire slice could not be projected into optimizer reserves. Every variant is a data fault in
/// the received payload (or a downstream math failure), never a panic — the adapter is total over
/// arbitrary input so a malformed server response degrades to a typed error, not a crash.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum WireAdapterError {
    /// A `0x`-hex field (address, pool id, `sqrt_price_x96`, `liquidity`, or `tick`) did not parse.
    #[error("failed to parse wire field `{field}` value {value:?}")]
    HexParse { field: &'static str, value: String },
    /// A token's decimals fell outside the supported range.
    #[error("unsupported token decimals for {address:?}")]
    Decimals { address: String },
    /// A pool appeared in the `/slice` response but not in the `/pools/meta` catalog, so its token
    /// pair and fee are unknown and it cannot be projected.
    #[error("slice pool {pool:?} has no metadata entry")]
    UnknownPool { pool: PoolRef },
    /// A pool references a token whose decimals are absent from the `/pools/meta` token catalog.
    #[error("pool {pool:?} references token {token:?} with no decimals entry")]
    UnknownToken { pool: PoolRef, token: Address },
    /// The reused domain projection math (virtual reserves / swap caps / fee) failed for a pool.
    /// Boxed because `PoolReserveProjectionError` embeds a `PoolState` clone, which would otherwise
    /// make every `Result<_, WireAdapterError>` in this module oversized (`clippy::result_large_err`).
    #[error(transparent)]
    Projection(Box<PoolReserveProjectionError>),
}

/// Maps a deserialized `/slice` state response plus its `/pools/meta` catalog into the optimizer's
/// input, `Vec<PoolReserves<PoolRef, TokenAddress>>` — the first consumer of the `aa-wire` deserialize
/// contract and the client-side mirror of the server's domain projection.
///
/// The wire is single-chain (it carries no chain tag), so the caller supplies the `chain` the server
/// is bound to; every parsed `PoolRef`/`TokenAddress` is stamped with it. The reserve/swap-cap/fee
/// assembly is not re-implemented here: it delegates to `client_evm::pool_reserve_values`, the same
/// function the server-side domain path (`pool_reserves_for_optimization`) uses, so the two paths
/// cannot drift.
///
/// Behaviour mirrors the domain projection: `Incomplete` pools (no frontier state) are skipped; each
/// projected pool is emitted **twice** — forward and `.inverse()` — so the optimizer sees both swap
/// directions; and output is sorted by `PoolRef` for deterministic, domain-matching order. The
/// response's `block_hash`/`confirmations` freshness facts are intentionally not consumed yet.
pub fn slice_to_reserves(
    slice: &SliceResponse,
    meta: &PoolsMetaResponse,
    chain: ChainKey,
) -> Result<Vec<PoolReserves<PoolRef, TokenAddress>>, WireAdapterError> {
    // Token address -> validated decimals, from the catalog's token list.
    let mut decimals = HashMap::with_capacity(meta.tokens.len());
    for token in &meta.tokens {
        let address = parse_address(&token.address, "token.address")?;
        let value = TokenDecimals::try_from_u256(U256::from(token.decimals)).map_err(|_| {
            WireAdapterError::Decimals {
                address: token.address.clone(),
            }
        })?;
        decimals.insert(address, value);
    }

    // Domain PoolRef -> (token0, token1, fee), from the catalog's pool list. Keying by the parsed
    // domain ref (rather than the wire `PoolQuery`) avoids widening the wire DTOs with `Hash`.
    let mut pool_meta = HashMap::with_capacity(meta.pools.len());
    for entry in &meta.pools {
        let pool = pool_ref_of(&entry.key, chain)?;
        let token0 = TokenAddress(parse_address(&entry.token0, "pool.token0")?, chain);
        let token1 = TokenAddress(parse_address(&entry.token1, "pool.token1")?, chain);
        let fee = PoolFee::Static {
            pips: entry.fee_pips,
            tick_spacing: entry.tick_spacing,
        };
        pool_meta.insert(pool, (token0, token1, fee));
    }

    // Project every pool that resolved to a frontier state; skip the rest.
    let mut projected = Vec::new();
    for slice_pool in &slice.pools {
        let PoolCompleteness::Complete { state } = &slice_pool.state else {
            continue;
        };
        let pool = pool_ref_of(&slice_pool.key, chain)?;
        let Some(&(token0, token1, fee)) = pool_meta.get(&pool) else {
            return Err(WireAdapterError::UnknownPool { pool });
        };
        let token0_decimals = *decimals
            .get(&token0.0)
            .ok_or(WireAdapterError::UnknownToken {
                pool,
                token: token0.0,
            })?;
        let token1_decimals = *decimals
            .get(&token1.0)
            .ok_or(WireAdapterError::UnknownToken {
                pool,
                token: token1.0,
            })?;

        let pool_state = parse_pool_state(state)?;
        let value = pool_reserve_values(
            pool,
            &pool_state,
            fee,
            token0,
            token1,
            token0_decimals,
            token1_decimals,
        )
        .map_err(|source| WireAdapterError::Projection(Box::new(source)))?;

        projected.push(PoolReserves {
            token0,
            token1,
            pool_id: pool,
            value,
        });
    }

    // Deterministic, domain-matching order: sort by pool, then emit forward + inverse per pool.
    projected.sort_by_key(|reserve| reserve.pool_id);
    let mut reserves = Vec::with_capacity(projected.len() * 2);
    for reserve in projected {
        reserves.extend([reserve, reserve.inverse()]);
    }
    Ok(reserves)
}

/// Parses a wire `PoolQuery` into its domain `PoolRef` on the given chain.
fn pool_ref_of(key: &PoolQuery, chain: ChainKey) -> Result<PoolRef, WireAdapterError> {
    match key {
        PoolQuery::UniswapV3 { address } => Ok(PoolRef::uniswap_v3(
            parse_address(address, "pool.address")?,
            chain,
        )),
        PoolQuery::UniswapV4 { pool_id } => {
            let hash = BlockHash::from_str(pool_id).map_err(|_| WireAdapterError::HexParse {
                field: "pool.pool_id",
                value: pool_id.clone(),
            })?;
            Ok(PoolRef::uniswap_v4(PoolId(hash), chain))
        }
    }
}

/// Parses a wire `WirePoolState` into the domain `PoolState`. The two big integers ride as
/// lowercase, minimal-width `0x`-hex (matching the server's `{:#x}` encoding); `tick` is a plain
/// `i32` that must fit an `I24`.
fn parse_pool_state(state: &aa_wire::WirePoolState) -> Result<PoolState, WireAdapterError> {
    let sqrt_price_x96 =
        U160::from_str_radix(strip_0x(&state.sqrt_price_x96), 16).map_err(|_| {
            WireAdapterError::HexParse {
                field: "sqrt_price_x96",
                value: state.sqrt_price_x96.clone(),
            }
        })?;
    let liquidity = u128::from_str_radix(strip_0x(&state.liquidity), 16).map_err(|_| {
        WireAdapterError::HexParse {
            field: "liquidity",
            value: state.liquidity.clone(),
        }
    })?;
    let tick = client_evm::I24::try_from(state.tick).map_err(|_| WireAdapterError::HexParse {
        field: "tick",
        value: state.tick.to_string(),
    })?;
    Ok(PoolState {
        sqrt_price_x96,
        tick,
        liquidity,
    })
}

/// Parses a `0x`-hex 20-byte address, tagging failures with the wire field name for diagnostics.
fn parse_address(value: &str, field: &'static str) -> Result<Address, WireAdapterError> {
    Address::from_str(value).map_err(|_| WireAdapterError::HexParse {
        field,
        value: value.to_owned(),
    })
}

/// Strips a leading `0x`/`0X` if present; a bare hex string is returned unchanged.
fn strip_0x(value: &str) -> &str {
    value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
        .unwrap_or(value)
}

#[cfg(test)]
mod tests {
    use aa_wire::{PoolMetaEntry, PoolSlice, TokenMetaEntry, WirePoolState};

    use super::*;

    // Tick-0 price (`2^96`), so `swap_limit_x/y` stay non-underflowing with any tick spacing.
    const SQRT_PRICE_TICK_0: u128 = 79_228_162_514_264_337_593_543_950_336;
    const CHAIN: ChainKey = ChainKey::Ethereum;

    fn addr(byte: u8) -> Address {
        Address::from([byte; 20])
    }

    fn pool_id(byte: u8) -> BlockHash {
        BlockHash::from([byte; 32])
    }

    /// Serializes a domain `PoolState` to the wire exactly as the server does (`{:#x}` big ints,
    /// plain `i32` tick), so parsing it back must recover the original.
    fn wire_state(state: &PoolState) -> WirePoolState {
        WirePoolState {
            sqrt_price_x96: format!("{:#x}", state.sqrt_price_x96),
            tick: i32::try_from(state.tick).expect("test tick fits i32"),
            liquidity: format!("{:#x}", state.liquidity),
        }
    }

    fn sample_state() -> PoolState {
        PoolState {
            sqrt_price_x96: U160::from(SQRT_PRICE_TICK_0),
            tick: client_evm::I24::try_from(0).expect("0 fits I24"),
            liquidity: 1_000_000_000_000_000_000u128,
        }
    }

    #[test]
    fn projects_v3_and_v4_pools_parity_sorted_with_inverses() {
        let state = sample_state();
        let token0 = addr(1);
        let token1 = addr(2);

        // A v3 pool and a v4 pool over the same token pair; both complete.
        let v3_key = PoolQuery::UniswapV3 {
            address: format!("{:#x}", addr(9)),
        };
        let v4_key = PoolQuery::UniswapV4 {
            pool_id: format!("{:#x}", pool_id(0xaa)),
        };
        let slice = SliceResponse {
            block_hash: format!("{:#x}", pool_id(0xbb)),
            confirmations: 2,
            pools: vec![
                // Deliberately v4-first on the wire to prove the adapter sorts (v3 must come out first).
                PoolSlice {
                    key: v4_key.clone(),
                    state: PoolCompleteness::Complete {
                        state: wire_state(&state),
                    },
                },
                PoolSlice {
                    key: v3_key.clone(),
                    state: PoolCompleteness::Complete {
                        state: wire_state(&state),
                    },
                },
            ],
        };
        let meta = PoolsMetaResponse {
            pools: vec![
                PoolMetaEntry {
                    key: v3_key,
                    token0: format!("{:#x}", token0),
                    token1: format!("{:#x}", token1),
                    fee_pips: 3000,
                    tick_spacing: 60,
                },
                PoolMetaEntry {
                    key: v4_key,
                    token0: format!("{:#x}", token0),
                    token1: format!("{:#x}", token1),
                    fee_pips: 500,
                    tick_spacing: 10,
                },
            ],
            tokens: vec![
                TokenMetaEntry {
                    address: format!("{:#x}", token0),
                    decimals: 18,
                },
                TokenMetaEntry {
                    address: format!("{:#x}", token1),
                    decimals: 6,
                },
            ],
        };

        let out = slice_to_reserves(&slice, &meta, CHAIN).expect("projection");

        // Four entries: each of the two pools forward + inverse, v3 sorted ahead of v4.
        assert_eq!(out.len(), 4);
        assert_eq!(out[0].pool_id, PoolRef::uniswap_v3(addr(9), CHAIN));
        assert_eq!(
            out[2].pool_id,
            PoolRef::uniswap_v4(PoolId(pool_id(0xaa)), CHAIN)
        );
        assert_eq!(out[1], out[0].inverse());
        assert_eq!(out[3], out[2].inverse());

        // Parity: the v3 entry must equal the domain projection of the same state, proving parsing
        // recovered the exact input and the adapter composed the reused math correctly.
        let expected_value = pool_reserve_values(
            PoolRef::uniswap_v3(addr(9), CHAIN),
            &state,
            PoolFee::Static {
                pips: 3000,
                tick_spacing: 60,
            },
            TokenAddress(token0, CHAIN),
            TokenAddress(token1, CHAIN),
            TokenDecimals::try_from_u256(U256::from(18u8)).expect("decimals"),
            TokenDecimals::try_from_u256(U256::from(6u8)).expect("decimals"),
        )
        .expect("domain projection");
        assert_eq!(out[0].value, expected_value);
        assert_eq!(out[0].token0, TokenAddress(token0, CHAIN));
        assert_eq!(out[0].token1, TokenAddress(token1, CHAIN));
    }

    #[test]
    fn skips_incomplete_pools() {
        let key = PoolQuery::UniswapV3 {
            address: format!("{:#x}", addr(9)),
        };
        let slice = SliceResponse {
            block_hash: format!("{:#x}", pool_id(0xbb)),
            confirmations: 0,
            pools: vec![PoolSlice {
                key: key.clone(),
                state: PoolCompleteness::Incomplete,
            }],
        };
        let meta = PoolsMetaResponse {
            pools: vec![PoolMetaEntry {
                key,
                token0: format!("{:#x}", addr(1)),
                token1: format!("{:#x}", addr(2)),
                fee_pips: 3000,
                tick_spacing: 60,
            }],
            tokens: vec![
                TokenMetaEntry {
                    address: format!("{:#x}", addr(1)),
                    decimals: 18,
                },
                TokenMetaEntry {
                    address: format!("{:#x}", addr(2)),
                    decimals: 6,
                },
            ],
        };

        assert_eq!(slice_to_reserves(&slice, &meta, CHAIN), Ok(vec![]));
    }

    #[test]
    fn errors_when_slice_pool_missing_from_meta() {
        let key = PoolQuery::UniswapV3 {
            address: format!("{:#x}", addr(9)),
        };
        let slice = SliceResponse {
            block_hash: format!("{:#x}", pool_id(0xbb)),
            confirmations: 0,
            pools: vec![PoolSlice {
                key,
                state: PoolCompleteness::Complete {
                    state: wire_state(&sample_state()),
                },
            }],
        };
        let meta = PoolsMetaResponse {
            pools: vec![],
            tokens: vec![],
        };

        assert_eq!(
            slice_to_reserves(&slice, &meta, CHAIN),
            Err(WireAdapterError::UnknownPool {
                pool: PoolRef::uniswap_v3(addr(9), CHAIN),
            })
        );
    }

    #[test]
    fn errors_when_pool_token_decimals_absent() {
        let key = PoolQuery::UniswapV3 {
            address: format!("{:#x}", addr(9)),
        };
        let slice = SliceResponse {
            block_hash: format!("{:#x}", pool_id(0xbb)),
            confirmations: 0,
            pools: vec![PoolSlice {
                key: key.clone(),
                state: PoolCompleteness::Complete {
                    state: wire_state(&sample_state()),
                },
            }],
        };
        let meta = PoolsMetaResponse {
            pools: vec![PoolMetaEntry {
                key,
                token0: format!("{:#x}", addr(1)),
                token1: format!("{:#x}", addr(2)),
                fee_pips: 3000,
                tick_spacing: 60,
            }],
            // token1 (addr(2)) omitted on purpose.
            tokens: vec![TokenMetaEntry {
                address: format!("{:#x}", addr(1)),
                decimals: 18,
            }],
        };

        assert_eq!(
            slice_to_reserves(&slice, &meta, CHAIN),
            Err(WireAdapterError::UnknownToken {
                pool: PoolRef::uniswap_v3(addr(9), CHAIN),
                token: addr(2),
            })
        );
    }

    #[test]
    fn errors_on_bad_hex() {
        let key = PoolQuery::UniswapV3 {
            address: "0xnot-hex".to_owned(),
        };
        let slice = SliceResponse {
            block_hash: format!("{:#x}", pool_id(0xbb)),
            confirmations: 0,
            pools: vec![PoolSlice {
                key,
                state: PoolCompleteness::Complete {
                    state: wire_state(&sample_state()),
                },
            }],
        };
        let meta = PoolsMetaResponse {
            pools: vec![],
            tokens: vec![],
        };

        assert_eq!(
            slice_to_reserves(&slice, &meta, CHAIN),
            Err(WireAdapterError::HexParse {
                field: "pool.address",
                value: "0xnot-hex".to_owned(),
            })
        );
    }
}
