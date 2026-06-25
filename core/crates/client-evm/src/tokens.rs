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
}
