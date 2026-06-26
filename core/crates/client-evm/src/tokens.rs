use alloy::primitives::{Address, address};

use crate::{ChainKey, TokenAddress};

pub const ETHEREUM_USDC_TOKEN_ADDRESS: TokenAddress = TokenAddress(
    address!("A0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"),
    ChainKey::Ethereum,
);

/// Native ETH on Ethereum, denoted by the zero address. v4 native-currency pools store this as their
/// `token0`; the token registry resolves it intrinsically to 18 decimals (it is not an ERC20).
pub const ETHEREUM_NATIVE_TOKEN_ADDRESS: TokenAddress = TokenAddress(Address::ZERO, ChainKey::Ethereum);

/// Canonical Wrapped Ether (WETH) on Ethereum. Wrapping is 1:1 with native ETH, so the optimizer
/// bridges this to [`ETHEREUM_NATIVE_TOKEN_ADDRESS`] to unify v4 native-ETH liquidity with v3 WETH.
pub const ETHEREUM_WETH_TOKEN_ADDRESS: TokenAddress = TokenAddress(
    address!("C02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"),
    ChainKey::Ethereum,
);

/// Circle native USDC on Arbitrum One (not the bridged USDC.e at 0xFF970…).
pub const ARBITRUM_USDC_TOKEN_ADDRESS: TokenAddress = TokenAddress(
    address!("af88d065e77c8cC2239327C5EDb3A432268e5831"),
    ChainKey::Arbitrum,
);

/// Native ETH on Arbitrum One (the chain's gas token), denoted by the zero address. v4 native-currency
/// pools store this as their `token0`; the token registry resolves it intrinsically to 18 decimals.
pub const ARBITRUM_NATIVE_TOKEN_ADDRESS: TokenAddress = TokenAddress(Address::ZERO, ChainKey::Arbitrum);

/// Canonical Wrapped Ether (WETH) on Arbitrum One. Wrapping is 1:1 with native ETH, so the optimizer
/// bridges this to [`ARBITRUM_NATIVE_TOKEN_ADDRESS`] to unify v4 native-ETH liquidity with v3 WETH.
pub const ARBITRUM_WETH_TOKEN_ADDRESS: TokenAddress = TokenAddress(
    address!("82aF49447D8a07e3bd95BD0d56f35241523fBab1"),
    ChainKey::Arbitrum,
);

// --- Base ---------------------------------------------------------------------------------------

/// Circle native USDC on Base.
pub const BASE_USDC_TOKEN_ADDRESS: TokenAddress = TokenAddress(
    address!("833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"),
    ChainKey::Base,
);
/// Native ETH on Base (the chain's gas token), denoted by the zero address; 18 decimals intrinsically.
pub const BASE_NATIVE_TOKEN_ADDRESS: TokenAddress = TokenAddress(Address::ZERO, ChainKey::Base);
/// Canonical Wrapped Ether (WETH) on Base, bridged 1:1 to [`BASE_NATIVE_TOKEN_ADDRESS`].
pub const BASE_WETH_TOKEN_ADDRESS: TokenAddress = TokenAddress(
    address!("4200000000000000000000000000000000000006"),
    ChainKey::Base,
);

// --- Optimism -----------------------------------------------------------------------------------

/// Circle native USDC on Optimism (not the bridged USDC.e).
pub const OPTIMISM_USDC_TOKEN_ADDRESS: TokenAddress = TokenAddress(
    address!("0b2C639c533813f4Aa9D7837CAf62653d097Ff85"),
    ChainKey::Optimism,
);
/// Native ETH on Optimism, denoted by the zero address; 18 decimals intrinsically.
pub const OPTIMISM_NATIVE_TOKEN_ADDRESS: TokenAddress = TokenAddress(Address::ZERO, ChainKey::Optimism);
/// Canonical Wrapped Ether (WETH) on Optimism, bridged 1:1 to [`OPTIMISM_NATIVE_TOKEN_ADDRESS`].
pub const OPTIMISM_WETH_TOKEN_ADDRESS: TokenAddress = TokenAddress(
    address!("4200000000000000000000000000000000000006"),
    ChainKey::Optimism,
);

// --- Polygon ------------------------------------------------------------------------------------

/// Circle native USDC on Polygon (not the bridged USDC.e at 0x2791…).
pub const POLYGON_USDC_TOKEN_ADDRESS: TokenAddress = TokenAddress(
    address!("3c499c542cEF5E3811e1192ce70d8cC03d5c3359"),
    ChainKey::Polygon,
);
/// Polygon's native gas token is POL (zero address, 18 decimals) — *not* ETH. WETH below is a bridged
/// ERC20, so unlike the rollups it is not 1:1 with the native token; it is the wrapped-ETH anchor only.
pub const POLYGON_NATIVE_TOKEN_ADDRESS: TokenAddress = TokenAddress(Address::ZERO, ChainKey::Polygon);
/// Bridged Wrapped Ether (WETH) on Polygon.
pub const POLYGON_WETH_TOKEN_ADDRESS: TokenAddress = TokenAddress(
    address!("7ceB23fD6bC0adD59E62ac25578270cFf1b9f619"),
    ChainKey::Polygon,
);

// --- BNB Chain ----------------------------------------------------------------------------------

/// USDC on BNB Chain. Note: 18 decimals here (not 6) — decimals are resolved via on-chain metadata,
/// so this address constant carries no decimals assumption.
pub const BNB_USDC_TOKEN_ADDRESS: TokenAddress = TokenAddress(
    address!("8AC76a51cc950d9822D68b83fE1Ad97B32Cd580d"),
    ChainKey::Bnb,
);
/// BNB Chain's native gas token is BNB (zero address, 18 decimals) — *not* ETH. WETH below is a bridged
/// ERC20 (Binance-Peg ETH), so it is not 1:1 with the native token.
pub const BNB_NATIVE_TOKEN_ADDRESS: TokenAddress = TokenAddress(Address::ZERO, ChainKey::Bnb);
/// Bridged Wrapped Ether (Binance-Peg ETH) on BNB Chain.
pub const BNB_WETH_TOKEN_ADDRESS: TokenAddress = TokenAddress(
    address!("2170Ed0880ac9A755fd29B2688956BD959F933F8"),
    ChainKey::Bnb,
);

// --- Avalanche ----------------------------------------------------------------------------------

/// Circle native USDC on Avalanche C-Chain.
pub const AVALANCHE_USDC_TOKEN_ADDRESS: TokenAddress = TokenAddress(
    address!("B97EF9Ef8734C71904D8002F8b6Bc66Dd9c48a6E"),
    ChainKey::Avalanche,
);
/// Avalanche's native gas token is AVAX (zero address, 18 decimals) — *not* ETH. WETH.e below is a
/// bridged ERC20, so it is not 1:1 with the native token.
pub const AVALANCHE_NATIVE_TOKEN_ADDRESS: TokenAddress =
    TokenAddress(Address::ZERO, ChainKey::Avalanche);
/// Bridged Wrapped Ether (WETH.e) on Avalanche C-Chain.
pub const AVALANCHE_WETH_TOKEN_ADDRESS: TokenAddress = TokenAddress(
    address!("49D5c2BdFfac6CE2BFdB6640F4F80f226bc10bAB"),
    ChainKey::Avalanche,
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ethereum_usdc_token_address_is_exported() {
        assert_eq!(
            ETHEREUM_USDC_TOKEN_ADDRESS,
            TokenAddress(
                address!("A0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"),
                ChainKey::Ethereum
            )
        );
    }

    #[test]
    fn ethereum_native_token_address_is_the_zero_address() {
        assert_eq!(
            ETHEREUM_NATIVE_TOKEN_ADDRESS,
            TokenAddress(Address::ZERO, ChainKey::Ethereum)
        );
    }

    #[test]
    fn ethereum_weth_token_address_is_exported() {
        assert_eq!(
            ETHEREUM_WETH_TOKEN_ADDRESS,
            TokenAddress(
                address!("C02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"),
                ChainKey::Ethereum
            )
        );
    }

    #[test]
    fn arbitrum_usdc_token_address_is_exported() {
        assert_eq!(
            ARBITRUM_USDC_TOKEN_ADDRESS,
            TokenAddress(
                address!("af88d065e77c8cC2239327C5EDb3A432268e5831"),
                ChainKey::Arbitrum
            )
        );
    }

    #[test]
    fn arbitrum_native_token_address_is_the_zero_address() {
        assert_eq!(
            ARBITRUM_NATIVE_TOKEN_ADDRESS,
            TokenAddress(Address::ZERO, ChainKey::Arbitrum)
        );
    }

    #[test]
    fn arbitrum_weth_token_address_is_exported() {
        assert_eq!(
            ARBITRUM_WETH_TOKEN_ADDRESS,
            TokenAddress(
                address!("82aF49447D8a07e3bd95BD0d56f35241523fBab1"),
                ChainKey::Arbitrum
            )
        );
    }

    #[test]
    fn added_chain_token_anchors_are_tagged_to_their_chain() {
        for (usdc, native, weth, chain) in [
            (
                BASE_USDC_TOKEN_ADDRESS,
                BASE_NATIVE_TOKEN_ADDRESS,
                BASE_WETH_TOKEN_ADDRESS,
                ChainKey::Base,
            ),
            (
                OPTIMISM_USDC_TOKEN_ADDRESS,
                OPTIMISM_NATIVE_TOKEN_ADDRESS,
                OPTIMISM_WETH_TOKEN_ADDRESS,
                ChainKey::Optimism,
            ),
            (
                POLYGON_USDC_TOKEN_ADDRESS,
                POLYGON_NATIVE_TOKEN_ADDRESS,
                POLYGON_WETH_TOKEN_ADDRESS,
                ChainKey::Polygon,
            ),
            (
                BNB_USDC_TOKEN_ADDRESS,
                BNB_NATIVE_TOKEN_ADDRESS,
                BNB_WETH_TOKEN_ADDRESS,
                ChainKey::Bnb,
            ),
            (
                AVALANCHE_USDC_TOKEN_ADDRESS,
                AVALANCHE_NATIVE_TOKEN_ADDRESS,
                AVALANCHE_WETH_TOKEN_ADDRESS,
                ChainKey::Avalanche,
            ),
        ] {
            assert_eq!(usdc.1, chain);
            assert_eq!(weth.1, chain);
            assert_eq!(native, TokenAddress(Address::ZERO, chain));
            assert_ne!(usdc.0, Address::ZERO);
            assert_ne!(weth.0, Address::ZERO);
        }
    }
}
