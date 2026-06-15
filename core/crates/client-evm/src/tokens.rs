use alloy::primitives::address;

use crate::TokenAddress;

pub const ETHEREUM_USDC_TOKEN_ADDRESS: TokenAddress =
    TokenAddress(address!("A0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"));

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
}
