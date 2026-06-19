use alloy::primitives::address;

use crate::TokenAddress;

pub const ETHEREUM_USDC_TOKEN_ADDRESS: TokenAddress =
    TokenAddress(address!("A0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"));

/// Circle native USDC on Arbitrum One (not the bridged USDC.e at 0xFF970…).
pub const ARBITRUM_USDC_TOKEN_ADDRESS: TokenAddress =
    TokenAddress(address!("af88d065e77c8cC2239327C5EDb3A432268e5831"));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ethereum_usdc_token_address_is_exported() {
        assert_eq!(
            ETHEREUM_USDC_TOKEN_ADDRESS,
            TokenAddress(address!("A0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"))
        );
    }

    #[test]
    fn arbitrum_usdc_token_address_is_exported() {
        assert_eq!(
            ARBITRUM_USDC_TOKEN_ADDRESS,
            TokenAddress(address!("af88d065e77c8cC2239327C5EDb3A432268e5831"))
        );
    }
}
