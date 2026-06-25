//! Uniswap v4 integration helpers.
//!
//! Unlike v3, every v4 pool lives inside a single `PoolManager` contract and is identified by a
//! [`PoolId`] — `keccak256(abi.encode(PoolKey))` — rather than by its own contract address. Pool
//! state is read through the `StateView` periphery contract, and pool events are emitted by the
//! `PoolManager` keyed on the indexed `PoolId`.

use alloy::{
    primitives::{Address, B256, address, keccak256},
    sol,
    sol_types::{SolEvent, SolValue},
};

use crate::{ChainKey, PoolFee, PoolMetadata, PoolMetadataFailure, PoolMetadataResult};

/// Ethereum mainnet `PoolManager` singleton. All v4 pools live here and emit their events from this
/// address, so v4 discovery filters on this target plus the per-pool `PoolId` topic rather than on a
/// per-pool address the way v3 does.
pub const ETHEREUM_UNISWAP_V4_POOL_MANAGER_ADDRESS: Address =
    address!("000000000004444c5dc75cB358380D2e3dE08A90");

/// Ethereum mainnet `StateView` periphery contract. v4 pools expose no per-pool `slot0()`/
/// `liquidity()`; their state is read here via `getSlot0(id)`/`getLiquidity(id)`.
pub const ETHEREUM_UNISWAP_V4_STATE_VIEW_ADDRESS: Address =
    address!("7fFE42C4a5DEeA5b0feC41C94C136Cf115597227");

/// The high bit of a `uint24` fee marks a pool whose fee is set dynamically by its hook rather than
/// being a fixed value. Such pools fall outside the constant-fee swap math, so the fixed-fee path
/// must reject them.
pub const DYNAMIC_FEE_FLAG: u32 = 0x80_00_00;

/// A Uniswap v4 pool identity: `keccak256(abi.encode(PoolKey))`. This is the indexed `id` topic on
/// every `PoolManager` pool event and the key accepted by `StateView` reads.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct PoolId(pub B256);

sol! {
    /// The full identity of a v4 pool. `currency0`/`currency1` are sorted token addresses
    /// (`currency0 < currency1`), with the zero address denoting native ETH. `abi.encode` of this
    /// struct hashes to the pool's [`PoolId`].
    #[derive(Debug, PartialEq, Eq)]
    struct PoolKey {
        address currency0;
        address currency1;
        uint24 fee;
        int24 tickSpacing;
        address hooks;
    }

    function getSlot0(bytes32 poolId) external view returns (
        uint160 sqrtPriceX96,
        int24 tick,
        uint24 protocolFee,
        uint24 lpFee
    );

    function getLiquidity(bytes32 poolId) external view returns (uint128 liquidity);

    #[derive(Debug, PartialEq, Eq)]
    event Initialize(
        bytes32 indexed id,
        address indexed currency0,
        address indexed currency1,
        uint24 fee,
        int24 tickSpacing,
        address hooks,
        uint160 sqrtPriceX96,
        int24 tick
    );

    #[derive(Debug, PartialEq, Eq)]
    event Swap(
        bytes32 indexed id,
        address indexed sender,
        int128 amount0,
        int128 amount1,
        uint160 sqrtPriceX96,
        uint128 liquidity,
        int24 tick,
        uint24 fee
    );

    #[derive(Debug, PartialEq, Eq)]
    event ModifyLiquidity(
        bytes32 indexed id,
        address indexed sender,
        int24 tickLower,
        int24 tickUpper,
        int256 liquidityDelta,
        bytes32 salt
    );
}

/// Derives a pool's [`PoolId`] from its [`PoolKey`], matching the on-chain
/// `keccak256(abi.encode(key))`.
pub fn pool_id(key: &PoolKey) -> PoolId {
    PoolId(keccak256(key.abi_encode()))
}

/// Whether a `uint24` fee value denotes a hook-controlled dynamic fee rather than a fixed fee.
pub fn is_dynamic_fee(fee: u32) -> bool {
    fee & DYNAMIC_FEE_FLAG != 0
}

/// Validates a v4 [`PoolKey`] against its claimed [`PoolId`] and projects it to [`PoolMetadata`].
///
/// Because [`PoolId`] is a one-way hash of the [`PoolKey`], the key fields can only come from a
/// source that carried them — the on-chain `Initialize` event or an off-chain indexer keyed by id.
/// Either way the source is trusted only for *existence*: we recompute `keccak256(abi.encode(key))`
/// and reject any `key`/`id` mismatch, so a malformed event or a buggy/dishonest indexer cannot
/// register the wrong pool. Pure and panic-free; pools the fixed-fee concentrated-liquidity model
/// cannot represent are rejected:
///
/// * dynamic-fee pools (the hook sets the fee per swap),
/// * hooked pools (hooks can break constant-product swap math),
/// * keys whose recomputed [`PoolId`] does not match `id` (malformed / unsorted currencies),
/// * non-positive or out-of-`u16` tick spacings.
///
/// Native-ETH pools (`currency0 == address(0)`) are accepted; `token0` is stored as the zero
/// address and resolved to 18 decimals intrinsically by the token registry.
pub fn pool_metadata_from_pool_key(id: PoolId, key: &PoolKey) -> PoolMetadataResult {
    let [hi, mid, lo] = key.fee.to_be_bytes::<3>();
    let pips = u32::from_be_bytes([0, hi, mid, lo]);
    if is_dynamic_fee(pips) {
        return Err(PoolMetadataFailure::DynamicFee);
    }
    if key.hooks != Address::ZERO {
        return Err(PoolMetadataFailure::HookedPool { hooks: key.hooks });
    }

    if pool_id(key) != id {
        return Err(PoolMetadataFailure::PoolIdMismatch);
    }

    let tick_spacing_i32 = i32::try_from(key.tickSpacing)
        .map_err(|_| PoolMetadataFailure::InvalidTickSpacing { tick_spacing: i32::MIN })?;
    let tick_spacing = u16::try_from(tick_spacing_i32)
        .ok()
        .filter(|spacing| *spacing > 0)
        .ok_or(PoolMetadataFailure::InvalidTickSpacing {
            tick_spacing: tick_spacing_i32,
        })?;

    Ok(PoolMetadata {
        token0: key.currency0,
        token1: key.currency1,
        fee: PoolFee::Static { pips, tick_spacing },
    })
}

/// The `StateView` periphery contract for a chain, through which v4 pool state is read
/// (`getSlot0`/`getLiquidity` keyed by [`PoolId`]). `None` for chains where v4 is not deployed/known;
/// callers skip v4 state reads for such chains rather than targeting a missing contract.
pub fn state_view_address(chain: ChainKey) -> Option<Address> {
    match chain {
        ChainKey::Ethereum => Some(ETHEREUM_UNISWAP_V4_STATE_VIEW_ADDRESS),
        _ => None,
    }
}

/// The `PoolManager` singleton for a chain, the address every v4 pool event is emitted from. `None`
/// for chains where v4 is not deployed/known. Used as the live-discovery bloom anchor so a block
/// carrying only v4 activity is never bloom-skipped.
pub fn pool_manager_address(chain: ChainKey) -> Option<Address> {
    match chain {
        ChainKey::Ethereum => Some(ETHEREUM_UNISWAP_V4_POOL_MANAGER_ADDRESS),
        _ => None,
    }
}

/// Whether `address` is a known v4 `PoolManager`. Chain-agnostic (the manager address is globally
/// unique and [`decode_pool_log`](crate::decode_pool_log) carries no chain), so v4 log decoding can
/// reject events spoofed from a non-manager contract that merely reuses the matching `topic0`.
pub fn is_v4_pool_manager(address: Address) -> bool {
    address == ETHEREUM_UNISWAP_V4_POOL_MANAGER_ADDRESS
}

/// The `topic0` signature hashes of the state-relevant v4 pool events, mirroring v3's
/// `pool_event_signature_hashes`.
pub fn pool_event_signature_hashes() -> [B256; 3] {
    [
        Initialize::SIGNATURE_HASH,
        Swap::SIGNATURE_HASH,
        ModifyLiquidity::SIGNATURE_HASH,
    ]
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use alloy::primitives::{Uint, address, aliases::I24, b256};

    use super::*;
    use crate::uniswap_v3;

    fn fee(pips: u32) -> Uint<24, 1> {
        Uint::<24, 1>::from(pips)
    }

    fn tick_spacing(value: i32) -> I24 {
        I24::try_from(value).expect("tick spacing fixture in range")
    }

    // Real Ethereum-mainnet v4 pool: native ETH / USDC, fee 500, tick spacing 10, no hooks.
    // PoolId taken from the live pool; the test recomputes it from the key so the vector is checked
    // offline without any live data.
    const ETH_USDC_POOL_ID: B256 =
        b256!("21c67e77068de97969ba93d4aab21826d33ca12bb9f565d8496e8fda8a82ca27");

    fn eth_usdc_pool_key() -> PoolKey {
        PoolKey {
            currency0: Address::ZERO,
            currency1: address!("a0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"),
            fee: Uint::<24, 1>::from(500u32),
            tickSpacing: I24::try_from(10).expect("tick spacing in range"),
            hooks: Address::ZERO,
        }
    }

    #[test]
    fn pool_id_matches_known_mainnet_eth_usdc_pool() {
        assert_eq!(pool_id(&eth_usdc_pool_key()), PoolId(ETH_USDC_POOL_ID));
    }

    #[test]
    fn state_view_address_is_known_for_ethereum_only() {
        assert_eq!(
            state_view_address(ChainKey::Ethereum),
            Some(ETHEREUM_UNISWAP_V4_STATE_VIEW_ADDRESS)
        );
        assert_eq!(state_view_address(ChainKey::Arbitrum), None);
    }

    #[test]
    fn pool_manager_address_is_known_for_ethereum_only() {
        assert_eq!(
            pool_manager_address(ChainKey::Ethereum),
            Some(ETHEREUM_UNISWAP_V4_POOL_MANAGER_ADDRESS)
        );
        assert_eq!(pool_manager_address(ChainKey::Arbitrum), None);
    }

    #[test]
    fn is_v4_pool_manager_recognizes_only_the_known_manager() {
        assert!(is_v4_pool_manager(ETHEREUM_UNISWAP_V4_POOL_MANAGER_ADDRESS));
        assert!(!is_v4_pool_manager(Address::ZERO));
        assert!(!is_v4_pool_manager(address!(
            "00000000000000000000000000000000deadbeef"
        )));
    }

    #[test]
    fn pool_event_signature_hashes_are_unique() {
        let hashes = pool_event_signature_hashes();
        let unique = hashes.into_iter().collect::<HashSet<_>>();

        assert_eq!(unique.len(), hashes.len());
    }

    #[test]
    fn pool_event_signature_hashes_are_disjoint_from_v3() {
        let v3 = uniswap_v3::pool_event_signature_hashes()
            .into_iter()
            .collect::<HashSet<_>>();

        for hash in pool_event_signature_hashes() {
            assert!(!v3.contains(&hash));
        }
    }

    #[test]
    fn dynamic_fee_flag_is_detected_independently_of_fixed_fees() {
        assert!(is_dynamic_fee(DYNAMIC_FEE_FLAG));
        assert!(is_dynamic_fee(DYNAMIC_FEE_FLAG | 3000));
        assert!(!is_dynamic_fee(500));
        assert!(!is_dynamic_fee(3000));
        assert!(!is_dynamic_fee(0));
    }

    #[test]
    fn builds_static_metadata_for_a_valid_native_eth_pool() {
        // The flagship ETH/USDC pool: currency0 is native ETH (zero address), accepted as token0.
        let key = eth_usdc_pool_key();

        assert_eq!(
            pool_metadata_from_pool_key(pool_id(&key), &key),
            Ok(PoolMetadata {
                token0: Address::ZERO,
                token1: key.currency1,
                fee: PoolFee::Static {
                    pips: 500,
                    tick_spacing: 10,
                },
            })
        );
    }

    #[test]
    fn rejects_a_dynamic_fee_pool() {
        let key = PoolKey {
            fee: fee(DYNAMIC_FEE_FLAG | 500),
            ..eth_usdc_pool_key()
        };

        assert_eq!(
            pool_metadata_from_pool_key(pool_id(&key), &key),
            Err(PoolMetadataFailure::DynamicFee)
        );
    }

    #[test]
    fn rejects_a_hooked_pool() {
        let hooks = address!("00000000000000000000000000000000deadbeef");
        let key = PoolKey {
            hooks,
            ..eth_usdc_pool_key()
        };

        assert_eq!(
            pool_metadata_from_pool_key(pool_id(&key), &key),
            Err(PoolMetadataFailure::HookedPool { hooks })
        );
    }

    #[test]
    fn rejects_a_key_whose_id_does_not_match() {
        // A well-formed key paired with an id that does not hash from it.
        let key = eth_usdc_pool_key();

        assert_eq!(
            pool_metadata_from_pool_key(PoolId(B256::ZERO), &key),
            Err(PoolMetadataFailure::PoolIdMismatch)
        );
    }

    #[test]
    fn rejects_a_non_positive_tick_spacing() {
        let key = PoolKey {
            tickSpacing: tick_spacing(-1),
            ..eth_usdc_pool_key()
        };

        assert_eq!(
            pool_metadata_from_pool_key(pool_id(&key), &key),
            Err(PoolMetadataFailure::InvalidTickSpacing { tick_spacing: -1 })
        );
    }
}
