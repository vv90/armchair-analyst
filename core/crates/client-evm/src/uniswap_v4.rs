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
}
