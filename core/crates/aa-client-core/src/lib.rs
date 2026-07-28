//! `aa-client-core` — the headless engine for the GUI client. Owns *all* application logic so the
//! UI stays thin: polling the `aa-server` data plane, the `aa-wire` → `optimization::PoolReserves`
//! adapter, client-side session config (route + bridges), the reconcile/optimize loop,
//! candidate tracking, and the `AppState` it projects into an `aa-client-api::ViewModel`.
//!
//! Transport-agnostic: it consumes `AppCommand`s and produces `ViewModel`s and knows nothing about
//! the binding (FFI/`cdylib`) or the UI framework. So far this crate holds the wire→reserves adapter
//! (below), the pure `state` reducer that drives the poll/optimize loop, the self-clocked `optimizer`
//! worker fed fresh reserves through a coalescing `latest_slot`, the `http` data-plane adapter, and
//! the `runtime` composition root that runs the whole engine on `aa-framework`; the UI `ViewModel`
//! contract lands in a later increment.

mod http;
mod latest_slot;
mod optimizer;
mod pending;
mod runtime;
mod state;

pub use http::{DataPlaneClient, FetchRequest, run as run_data_plane};
pub use pending::{FetchId, PendingFetches};
pub use runtime::{
    ClientConfig, ClientEngineApp, ClientEngineRuntime, Subscription, run as run_engine,
};
pub use state::{
    AppState, AwaitReason, Effect, EffectError, Event, FetchKind, OptimizeStage, Route, Session,
    SessionConfig, SliceProvenance, Work, transition,
};

use std::collections::HashMap;
use std::str::FromStr;

use aa_wire::{PoolCompleteness, PoolQuery, PoolsMetaResponse, SliceRequest, SliceResponse};
use client_evm::multi_chain_kernel::{PoolReserveProjectionError, pool_reserve_values};
use client_evm::uniswap_v4::PoolId;
use client_evm::{
    Address, BlockHash, ChainKey, PoolFee, PoolRef, PoolState, TokenAddress, TokenDecimals, U160,
    U256,
};
use optimization::{Invertible, PoolReserves};

/// Why a `/slice` payload could not be projected into optimizer reserves. Both variants are faults in
/// the *fresh* payload — the static catalog's own faults are settled once, at its boundary, and are
/// [`CatalogFault`]s instead. Never a panic: [`Catalog::project`] is total over arbitrary input, so a
/// malformed or version-skewed response degrades to a typed error rather than a crash.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum WireAdapterError {
    /// A `0x`-hex slice field (pool address, pool id, `sqrt_price_x96`, `liquidity`, or `tick`) did
    /// not parse.
    #[error("failed to parse wire field `{field}` value {value:?}")]
    HexParse { field: &'static str, value: String },
    /// The reused domain projection math (virtual reserves / swap caps / fee) failed for a pool.
    /// Boxed because `PoolReserveProjectionError` embeds a `PoolState` clone, which would otherwise
    /// make every `Result<_, WireAdapterError>` in this module oversized (`clippy::result_large_err`).
    #[error(transparent)]
    Projection(Box<PoolReserveProjectionError>),
}

/// Why one `/pools/meta` entry was excluded from a [`Catalog`]. An excluded pool is one this client
/// cannot project, so it is never requested and never re-diagnosed — the reason is recorded here
/// once, instead of being rediscovered on every slice for the rest of the process.
///
/// A malformed *token* entry has no variant of its own: it simply fails to enter the decimals map,
/// and surfaces as [`CatalogFault::MissingDecimals`] on each pool that references it — which is the
/// only way an unreferenced token entry could ever matter.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum CatalogFault {
    /// A `0x`-hex field of a catalog pool entry (its key, `token0`, or `token1`) did not parse.
    #[error("failed to parse catalog field `{field}` value {value:?}")]
    HexParse { field: &'static str, value: String },
    /// The pool references a token with no usable decimals in the catalog's token list — either
    /// absent (the server lists a pool as soon as its pool registry holds it, before its token
    /// metadata has necessarily resolved) or present with an out-of-range value.
    #[error("pool {pool:?} references token {token:?} with no usable decimals entry")]
    MissingDecimals { pool: PoolRef, token: Address },
}

/// One pool's fully resolved catalog metadata: exactly the arguments `pool_reserve_values` takes
/// beyond the pool ref and its state, so projecting a slice does no assembly and no re-parsing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PoolEntry {
    token0: TokenAddress,
    token1: TokenAddress,
    fee: PoolFee,
    token0_decimals: TokenDecimals,
    token1_decimals: TokenDecimals,
}

/// A validated `/pools/meta` payload: the pools this client can actually project, the `/slice`
/// request that asks for exactly those, and why every other entry was excluded.
///
/// Holding this instead of the raw `PoolsMetaResponse` moves a decision that was being re-derived on
/// every single slice — parse each token address, validate its decimals, parse each pool key and both
/// of its token addresses, build two maps — to one pass at the boundary. Projecting a slice then
/// costs one key parse and one lookup per pool. It also makes "a catalog is held" mean "a usable
/// catalog is held": an entry that cannot be resolved is excluded *here*, with a reason, rather than
/// failing later against payloads that are not at fault.
#[derive(Clone, Debug, PartialEq)]
pub struct Catalog {
    /// The chain every parsed `PoolRef`/`TokenAddress` is stamped with. Owned by the catalog rather
    /// than passed per call, so a slice cannot be projected against a different chain than the
    /// catalog was built for.
    chain: ChainKey,
    pools: HashMap<PoolRef, PoolEntry>,
    request: SliceRequest,
    excluded: Vec<CatalogFault>,
}

impl Catalog {
    /// Resolves a `/pools/meta` payload. **Total**: every pool entry either lands in the admissible
    /// set or in [`Catalog::excluded`] with its reason, so an entry can never be silently dropped and
    /// a corrupt catalog can never panic the engine thread.
    pub fn parse(meta: &PoolsMetaResponse, chain: ChainKey) -> Catalog {
        // Token address -> validated decimals. Entries that don't parse or don't validate are simply
        // absent; the pools that reference them are excluded below, where the pool can be named.
        let mut decimals = HashMap::with_capacity(meta.tokens.len());
        for token in &meta.tokens {
            let (Ok(address), Ok(value)) = (
                Address::from_str(&token.address),
                TokenDecimals::try_from_u256(U256::from(token.decimals)),
            ) else {
                continue;
            };
            decimals.insert(address, value);
        }

        let mut pools = HashMap::with_capacity(meta.pools.len());
        let mut request = Vec::with_capacity(meta.pools.len());
        let mut excluded = Vec::new();
        for entry in &meta.pools {
            // Keyed by the parsed domain ref (rather than the wire `PoolQuery`) so the slice's own
            // keys are normalized through the same parse before they are looked up.
            let resolved = pool_ref_of(&entry.key, chain).and_then(|pool| {
                let token0 = parse_address(&entry.token0, "pool.token0")?;
                let token1 = parse_address(&entry.token1, "pool.token1")?;
                Ok((pool, token0, token1))
            });
            let (pool, token0, token1) = match resolved {
                Ok(resolved) => resolved,
                Err(fault) => {
                    excluded.push(fault.into());
                    continue;
                }
            };
            let (Some(&token0_decimals), Some(&token1_decimals)) =
                (decimals.get(&token0), decimals.get(&token1))
            else {
                let token = if decimals.contains_key(&token0) {
                    token1
                } else {
                    token0
                };
                excluded.push(CatalogFault::MissingDecimals { pool, token });
                continue;
            };

            pools.insert(
                pool,
                PoolEntry {
                    token0: TokenAddress(token0, chain),
                    token1: TokenAddress(token1, chain),
                    fee: PoolFee::Static {
                        pips: entry.fee_pips,
                        tick_spacing: entry.tick_spacing,
                    },
                    token0_decimals,
                    token1_decimals,
                },
            );
            request.push(entry.key.clone());
        }

        Catalog {
            chain,
            pools,
            request: SliceRequest { pools: request },
            excluded,
        }
    }

    /// The prebuilt `POST /slice` request: every admissible pool, and nothing else. Asking only for
    /// what can be projected is what makes an unprojectable *requested* pool unreachable rather than
    /// merely untested.
    pub fn slice_request(&self) -> &SliceRequest {
        &self.request
    }

    /// The number of pools this client can project.
    pub fn len(&self) -> usize {
        self.pools.len()
    }

    /// Whether no catalog pool resolved (an empty or wholly unusable payload).
    pub fn is_empty(&self) -> bool {
        self.pools.is_empty()
    }

    /// The catalog entries this client cannot use, and why — a durable, view-renderable record of
    /// what the server offered but the client dropped.
    pub fn excluded(&self) -> &[CatalogFault] {
        &self.excluded
    }

    /// Maps a deserialized `/slice` state response into the optimizer's input,
    /// `Vec<PoolReserves<PoolRef, TokenAddress>>` — the client-side mirror of the server's domain
    /// projection. The reserve/swap-cap/fee assembly is not re-implemented here: it delegates to
    /// `client_evm::pool_reserve_values`, the same function the server-side domain path
    /// (`pool_reserves_for_optimization`) uses, so the two paths cannot drift.
    ///
    /// A slice pool is projected iff it is `Complete` **and** admissible in this catalog; anything
    /// else is skipped, costing exactly that pool. A pool absent from the catalog is one the client
    /// never asked for — the server volunteered it, or the catalog was replaced while the slice was
    /// in flight — and voiding the whole slice over it would strand every other pool.
    ///
    /// Otherwise behaviour mirrors the domain projection: each projected pool is emitted **twice** —
    /// forward and `.inverse()` — so the optimizer sees both swap directions; and output is sorted by
    /// `PoolRef` for deterministic, domain-matching order. The response's `block_hash`/`confirmations`
    /// freshness facts are consumed by the reducer, not here.
    pub fn project(
        &self,
        slice: &SliceResponse,
    ) -> Result<Vec<PoolReserves<PoolRef, TokenAddress>>, WireAdapterError> {
        let mut projected = Vec::new();
        for slice_pool in &slice.pools {
            let PoolCompleteness::Complete { state } = &slice_pool.state else {
                continue;
            };
            let pool = pool_ref_of(&slice_pool.key, self.chain)?;
            let Some(entry) = self.pools.get(&pool) else {
                continue;
            };

            let pool_state = parse_pool_state(state)?;
            let value = pool_reserve_values(
                pool,
                &pool_state,
                entry.fee,
                entry.token0,
                entry.token1,
                entry.token0_decimals,
                entry.token1_decimals,
            )
            .map_err(|source| WireAdapterError::Projection(Box::new(source)))?;

            projected.push(PoolReserves {
                token0: entry.token0,
                token1: entry.token1,
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
}

/// A `0x`-hex wire field that did not parse — the *only* thing the shared parse helpers can fail
/// with. Returning this rather than one of the two error enums is what lets the catalog and the slice
/// share a single parse of a pool key while each names the failure in its own vocabulary, with no
/// unreachable variant to invent on either side.
#[derive(Clone, Debug, PartialEq, Eq)]
struct HexFault {
    field: &'static str,
    value: String,
}

impl From<HexFault> for WireAdapterError {
    fn from(fault: HexFault) -> WireAdapterError {
        WireAdapterError::HexParse {
            field: fault.field,
            value: fault.value,
        }
    }
}

impl From<HexFault> for CatalogFault {
    fn from(fault: HexFault) -> CatalogFault {
        CatalogFault::HexParse {
            field: fault.field,
            value: fault.value,
        }
    }
}

/// Parses a wire `PoolQuery` into its domain `PoolRef` on the given chain.
fn pool_ref_of(key: &PoolQuery, chain: ChainKey) -> Result<PoolRef, HexFault> {
    match key {
        PoolQuery::UniswapV3 { address } => Ok(PoolRef::uniswap_v3(
            parse_address(address, "pool.address")?,
            chain,
        )),
        PoolQuery::UniswapV4 { pool_id } => {
            let hash = BlockHash::from_str(pool_id).map_err(|_| HexFault {
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
fn parse_pool_state(state: &aa_wire::WirePoolState) -> Result<PoolState, HexFault> {
    let sqrt_price_x96 =
        U160::from_str_radix(strip_0x(&state.sqrt_price_x96), 16).map_err(|_| HexFault {
            field: "sqrt_price_x96",
            value: state.sqrt_price_x96.clone(),
        })?;
    let liquidity = u128::from_str_radix(strip_0x(&state.liquidity), 16).map_err(|_| HexFault {
        field: "liquidity",
        value: state.liquidity.clone(),
    })?;
    let tick = client_evm::I24::try_from(state.tick).map_err(|_| HexFault {
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
fn parse_address(value: &str, field: &'static str) -> Result<Address, HexFault> {
    Address::from_str(value).map_err(|_| HexFault {
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
    use proptest::prelude::*;

    use super::*;

    // Tick-0 price (`2^96`), so `swap_limit_x/y` stay non-underflowing with any tick spacing.
    const SQRT_PRICE_TICK_0: u128 = 79_228_162_514_264_337_593_543_950_336;
    const CHAIN: ChainKey = ChainKey::Ethereum;

    /// Parse-then-project in one call. The projection properties below are about what comes out of a
    /// slice, not about how the catalog was resolved — the boundary has its own properties — so they
    /// state the whole wire→reserves path in the one shape they care about.
    fn slice_to_reserves(
        slice: &SliceResponse,
        meta: &PoolsMetaResponse,
        chain: ChainKey,
    ) -> Result<Vec<PoolReserves<PoolRef, TokenAddress>>, WireAdapterError> {
        Catalog::parse(meta, chain).project(slice)
    }

    /// A pool to put on the wire: protocol, identity byte, its token pair, fee facts, liquidity, and
    /// whether `/slice` reports it `Complete`. Deliberately symbolic (small byte indices rather than
    /// raw addresses) so proptest shrinks to a readable counterexample — the kernel's
    /// `GeneratedEvent` uses the same trick.
    ///
    /// The price is pinned at the tick-0 boundary (`2^96`) and only *liquidity* varies, exactly as
    /// the server-side `pool_reserves_projection_emits_two_finite_directional_entries_per_pool`
    /// property does with `balanced_pool_state`: an arbitrary `(sqrt_price, tick)` pair is not
    /// generally self-consistent, and `sqrt_price_at_tick` is not exported from `client-evm`.
    #[derive(Clone, Copy, Debug)]
    struct GeneratedPool {
        uniswap_v4: bool,
        key_byte: u8,
        token0: u8,
        token1: u8,
        fee_pips: u32,
        tick_spacing: u16,
        liquidity: u128,
        complete: bool,
    }

    impl GeneratedPool {
        fn key(&self) -> PoolQuery {
            if self.uniswap_v4 {
                PoolQuery::UniswapV4 {
                    pool_id: format!("{:#x}", pool_id(self.key_byte)),
                }
            } else {
                PoolQuery::UniswapV3 {
                    address: format!("{:#x}", addr(self.key_byte)),
                }
            }
        }

        fn pool_ref(&self) -> PoolRef {
            if self.uniswap_v4 {
                PoolRef::uniswap_v4(PoolId(pool_id(self.key_byte)), CHAIN)
            } else {
                PoolRef::uniswap_v3(addr(self.key_byte), CHAIN)
            }
        }

        fn state(&self) -> PoolState {
            PoolState {
                sqrt_price_x96: U160::from(SQRT_PRICE_TICK_0),
                tick: client_evm::I24::ZERO,
                liquidity: self.liquidity,
            }
        }

        fn fee(&self) -> PoolFee {
            PoolFee::Static {
                pips: self.fee_pips,
                tick_spacing: self.tick_spacing,
            }
        }
    }

    fn generated_pool() -> impl Strategy<Value = GeneratedPool> {
        (
            any::<bool>(),
            0u8..12,
            0u8..6,
            0u8..6,
            prop_oneof![Just(100u32), Just(500), Just(3000), Just(10_000)],
            prop_oneof![Just(1u16), Just(10), Just(60), Just(200)],
            1u128..1_000_000_000_000_000_000_000,
            any::<bool>(),
        )
            .prop_map(
                |(
                    uniswap_v4,
                    key_byte,
                    token0,
                    token1,
                    fee_pips,
                    tick_spacing,
                    liquidity,
                    complete,
                )| GeneratedPool {
                    uniswap_v4,
                    key_byte,
                    // A pool over one token twice is a degenerate pair the catalog never emits.
                    token0,
                    token1: if token1 == token0 { token0 + 1 } else { token1 },
                    fee_pips,
                    tick_spacing,
                    liquidity,
                    complete,
                },
            )
    }

    /// A catalog and a matching slice for a set of pools whose `(protocol, key_byte)` identities are
    /// unique — the catalog is a map keyed by pool, so duplicates are not a shape the server emits.
    fn generated_catalog() -> impl Strategy<
        Value = (
            Vec<GeneratedPool>,
            Vec<u8>,
            SliceResponse,
            PoolsMetaResponse,
        ),
    > {
        (
            prop::collection::vec(generated_pool(), 0..10),
            // Decimals for token bytes 0..7; `TokenDecimals` accepts the usual ERC-20 range.
            prop::collection::vec(0u8..=18, 8),
            any::<u8>(),
            any::<u64>(),
        )
            .prop_map(|(pools, decimals, block, confirmations)| {
                let mut seen = std::collections::HashSet::new();
                let pools = pools
                    .into_iter()
                    .filter(|pool| seen.insert((pool.uniswap_v4, pool.key_byte)))
                    .collect::<Vec<_>>();

                let slice = SliceResponse {
                    block_hash: format!("{:#x}", pool_id(block)),
                    confirmations,
                    pools: pools
                        .iter()
                        .map(|pool| PoolSlice {
                            key: pool.key(),
                            state: if pool.complete {
                                PoolCompleteness::Complete {
                                    state: wire_state(&pool.state()),
                                }
                            } else {
                                PoolCompleteness::Incomplete
                            },
                        })
                        .collect(),
                };
                let meta = PoolsMetaResponse {
                    pools: pools
                        .iter()
                        .map(|pool| PoolMetaEntry {
                            key: pool.key(),
                            token0: format!("{:#x}", addr(pool.token0)),
                            token1: format!("{:#x}", addr(pool.token1)),
                            fee_pips: pool.fee_pips,
                            tick_spacing: pool.tick_spacing,
                        })
                        .collect(),
                    // Every token byte a pool can reference (0..=6, after the `token0 + 1` bump).
                    tokens: (0u8..8)
                        .map(|byte| TokenMetaEntry {
                            address: format!("{:#x}", addr(byte)),
                            decimals: decimals.get(usize::from(byte)).copied().unwrap_or(18),
                        })
                        .collect(),
                };

                (pools, decimals, slice, meta)
            })
    }

    proptest! {
        /// The projection's shape contract, over any catalog: exactly two entries per *complete*
        /// pool (incomplete ones contribute nothing), ordered by `PoolRef`, and each adjacent pair
        /// is a pool and its own inverse. The optimizer relies on all three — it indexes reserves
        /// positionally and needs both swap directions present for every pool.
        ///
        /// This is the client-side twin of `client-evm`'s
        /// `pool_reserves_projection_emits_two_finite_directional_entries_per_pool`.
        #[test]
        fn projection_emits_two_sorted_directional_entries_per_complete_pool(
            (pools, _decimals, slice, meta) in generated_catalog(),
        ) {
            let reserves = slice_to_reserves(&slice, &meta, CHAIN).expect("a valid catalog projects");

            let complete = pools.iter().filter(|pool| pool.complete).count();
            prop_assert_eq!(reserves.len(), complete * 2);

            // Sorted by pool, with each pool's inverse immediately after it.
            for pair in reserves.chunks(2) {
                let [forward, inverse] = pair else {
                    prop_assert!(false, "reserves must come in directional pairs");
                    return Ok(());
                };
                prop_assert_eq!(inverse, &forward.inverse());
                prop_assert_eq!(forward.pool_id, inverse.pool_id);
            }
            let order = reserves
                .iter()
                .step_by(2)
                .map(|reserve| reserve.pool_id)
                .collect::<Vec<_>>();
            let mut sorted = order.clone();
            sorted.sort();
            prop_assert_eq!(order, sorted, "pools must be emitted in PoolRef order");
        }

        /// Every projected value is finite and in range — no NaN, no infinity, no negative reserve
        /// or swap cap, and a fee multiplier in `(0, 1]`. The optimizer packs these straight into
        /// tensors, where a NaN would silently poison the whole model rather than fail loudly.
        /// Same value-domain pin as the server-side projection property.
        #[test]
        fn projected_values_are_finite_and_in_range(
            (_pools, _decimals, slice, meta) in generated_catalog(),
        ) {
            let reserves = slice_to_reserves(&slice, &meta, CHAIN).expect("a valid catalog projects");

            for reserve in &reserves {
                let value = reserve.value;
                prop_assert!(value.token_0.is_finite() && value.token_0 >= 0.0);
                prop_assert!(value.token_1.is_finite() && value.token_1 >= 0.0);
                prop_assert!(value.max_swap_0.is_finite() && value.max_swap_0 >= 0.0);
                prop_assert!(value.max_swap_1.is_finite() && value.max_swap_1 >= 0.0);
                prop_assert!(value.fee_multiplier > 0.0 && value.fee_multiplier <= 1.0);
            }
        }

        /// Parity with the domain projection, over any catalog. `slice_to_reserves`' contract is
        /// that it re-uses `client_evm::pool_reserve_values` rather than re-implementing the math,
        /// "so the two paths cannot drift" — this is that claim as an executable oracle, covering
        /// every generated fee tier, decimals pair, liquidity, and protocol rather than the single
        /// hand-written fixture. It also proves the hex round-trip recovers the exact input state.
        #[test]
        fn projection_matches_the_domain_math_pool_for_pool(
            (pools, decimals, slice, meta) in generated_catalog(),
        ) {
            let reserves = slice_to_reserves(&slice, &meta, CHAIN).expect("a valid catalog projects");

            for pool in pools.iter().filter(|pool| pool.complete) {
                let entry = reserves
                    .iter()
                    .find(|reserve| {
                        reserve.pool_id == pool.pool_ref()
                            && reserve.token0 == TokenAddress(addr(pool.token0), CHAIN)
                    })
                    .expect("every complete pool has a forward entry");

                let decimals_of = |byte: u8| {
                    TokenDecimals::try_from_u256(U256::from(
                        decimals.get(usize::from(byte)).copied().unwrap_or(18),
                    ))
                    .expect("generated decimals are in range")
                };
                let expected = pool_reserve_values(
                    pool.pool_ref(),
                    &pool.state(),
                    pool.fee(),
                    TokenAddress(addr(pool.token0), CHAIN),
                    TokenAddress(addr(pool.token1), CHAIN),
                    decimals_of(pool.token0),
                    decimals_of(pool.token1),
                )
                .expect("the domain projection succeeds on the same inputs");

                prop_assert_eq!(entry.value, expected);
            }
        }

        /// Every catalog entry is accounted for: admitted into the request, or excluded with a
        /// reason. This is what makes exclusion a *record* rather than a drop — the client silently
        /// losing pools the server offered would be indistinguishable from the server not offering
        /// them, and the catalog is fetched once, so the loss would be permanent.
        ///
        /// Both exclusion paths are exercised: a corrupted `token0` cannot parse, and a removed
        /// token entry leaves the pools referencing it with no decimals.
        #[test]
        fn every_catalog_entry_is_either_requested_or_recorded(
            (_pools, _decimals, _slice, mut meta) in generated_catalog(),
            corrupted in prop::collection::vec(any::<bool>(), 0..10),
            dropped in 0u8..8,
        ) {
            for (entry, corrupt) in meta.pools.iter_mut().zip(&corrupted) {
                if *corrupt {
                    entry.token0 = "0xnot-hex".to_owned();
                }
            }
            meta.tokens.retain(|token| token.address != format!("{:#x}", addr(dropped)));

            let catalog = Catalog::parse(&meta, CHAIN);

            prop_assert_eq!(
                catalog.slice_request().pools.len() + catalog.excluded().len(),
                meta.pools.len()
            );
        }

        /// Everything the catalog asks the server for is something it can project, and every pool it
        /// admitted is asked for exactly once. Together these are why a requested-but-unprojectable
        /// pool is unreachable rather than merely unobserved: the request is built from the resolved
        /// set, so the reserve set can have no hole where a pool was fetched and then dropped.
        #[test]
        fn every_requested_pool_is_one_the_catalog_can_project(
            (_pools, _decimals, _slice, meta) in generated_catalog(),
        ) {
            let catalog = Catalog::parse(&meta, CHAIN);
            // Answer the catalog's own request, with every pool complete.
            let slice = SliceResponse {
                block_hash: format!("{:#x}", pool_id(0)),
                confirmations: 0,
                pools: catalog
                    .slice_request()
                    .pools
                    .iter()
                    .cloned()
                    .map(|key| PoolSlice {
                        key,
                        state: PoolCompleteness::Complete { state: wire_state(&sample_state()) },
                    })
                    .collect(),
            };

            let reserves = catalog.project(&slice).expect("the catalog's own pools project");

            prop_assert_eq!(
                catalog.slice_request().pools.len(),
                catalog.len(),
                "an admissible pool must be requested exactly once"
            );
            prop_assert_eq!(
                reserves.len(),
                catalog.len() * 2,
                "a requested pool that does not project would be a hole in the reserve set"
            );
        }

        /// A catalog entry the client cannot resolve costs exactly that pool, never the slice. The
        /// server lists a pool as soon as its pool registry holds it but renders `tokens` from the
        /// tokens whose metadata it has, so a pool that outran its token metadata is an ordinary
        /// catalog — and the client fetches `/pools/meta` exactly once. Failing the whole slice over
        /// one such pool would mean recording a fault every poll and never optimizing at all.
        #[test]
        fn a_pool_with_unresolvable_metadata_does_not_void_the_others(
            (pools, _decimals, slice, mut meta) in generated_catalog(),
            dropped in 0u8..8,
        ) {
            // Exactly the shape a warming-up server emits: the pool stays listed, its token doesn't.
            meta.tokens.retain(|token| token.address != format!("{:#x}", addr(dropped)));

            let reserves = slice_to_reserves(&slice, &meta, CHAIN)
                .expect("a partial catalog still projects");

            let projectable = pools
                .iter()
                .filter(|pool| pool.complete && pool.token0 != dropped && pool.token1 != dropped)
                .count();
            prop_assert_eq!(reserves.len(), projectable * 2);
        }

        /// **Both stages are total over arbitrary wire input**: any combination of malformed hex,
        /// pools missing from the catalog, tokens missing decimals, and out-of-range values yields a
        /// `Catalog` (never a failure — every entry is admitted or recorded) and then `Ok` or a typed
        /// `WireAdapterError` — never a panic. The server is a separate process, so a corrupted or
        /// version-skewed response is untrusted input; a panic here would take down the whole engine
        /// thread. The accounting law is asserted here too, in the regime where nearly every entry is
        /// excluded, as the counterpart to the mixed regime above.
        #[test]
        fn parse_and_projection_are_total_over_arbitrary_wire_input(
            block_hash in "[0-9a-fx]{0,70}",
            pool_keys in prop::collection::vec("(0x)?[0-9a-zA-Z]{0,66}", 0..6),
            meta_keys in prop::collection::vec("(0x)?[0-9a-zA-Z]{0,66}", 0..6),
            sqrt_prices in prop::collection::vec("(0x)?[0-9a-zA-Z]{0,50}", 0..6),
            liquidities in prop::collection::vec("(0x)?[0-9a-zA-Z]{0,40}", 0..6),
            ticks in prop::collection::vec(any::<i32>(), 0..6),
            token_decimals in prop::collection::vec(any::<u8>(), 0..6),
            confirmations in any::<u64>(),
        ) {
            let slice = SliceResponse {
                block_hash,
                confirmations,
                pools: pool_keys
                    .iter()
                    .enumerate()
                    .map(|(index, address)| PoolSlice {
                        key: PoolQuery::UniswapV3 { address: address.clone() },
                        state: PoolCompleteness::Complete {
                            state: WirePoolState {
                                sqrt_price_x96: sqrt_prices.get(index).cloned().unwrap_or_default(),
                                tick: ticks.get(index).copied().unwrap_or_default(),
                                liquidity: liquidities.get(index).cloned().unwrap_or_default(),
                            },
                        },
                    })
                    .collect(),
            };
            let meta = PoolsMetaResponse {
                pools: meta_keys
                    .iter()
                    .map(|address| PoolMetaEntry {
                        key: PoolQuery::UniswapV3 { address: address.clone() },
                        token0: address.clone(),
                        token1: address.clone(),
                        fee_pips: 3000,
                        tick_spacing: 60,
                    })
                    .collect(),
                tokens: token_decimals
                    .iter()
                    .enumerate()
                    .map(|(index, decimals)| TokenMetaEntry {
                        address: format!("{:#x}", addr(index as u8)),
                        decimals: *decimals,
                    })
                    .collect(),
            };

            // The assertion *is* that these return rather than unwinding.
            let catalog = Catalog::parse(&meta, CHAIN);
            prop_assert_eq!(
                catalog.slice_request().pools.len() + catalog.excluded().len(),
                meta.pools.len()
            );
            let _ = catalog.project(&slice);
        }
    }

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
    fn skips_slice_pool_missing_from_meta() {
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

        assert_eq!(slice_to_reserves(&slice, &meta, CHAIN), Ok(vec![]));
    }

    #[test]
    fn skips_pool_whose_token_decimals_are_absent() {
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

        assert_eq!(slice_to_reserves(&slice, &meta, CHAIN), Ok(vec![]));
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
