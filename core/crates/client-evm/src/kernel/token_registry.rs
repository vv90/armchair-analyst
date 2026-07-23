use std::collections::{HashMap, HashSet};

use alloy::primitives::{Address, U256};
use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error as _};

use crate::ChainKey;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct TokenAddress(pub Address, pub ChainKey);

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct TokenDecimals(u8);

impl TokenDecimals {
    const MAX_SUPPORTED: u8 = 36;

    pub fn try_from_u256(value: U256) -> Result<TokenDecimals, TokenMetadataFailure> {
        if value <= U256::from(Self::MAX_SUPPORTED) {
            Ok(TokenDecimals(value.to::<u8>()))
        } else {
            Err(TokenMetadataFailure::UnsupportedDecimals(value))
        }
    }

    pub fn value(self) -> u8 {
        self.0
    }
}

// Serialize as the bare decimal count, and re-validate the `MAX_SUPPORTED` bound on the way back in so
// a corrupted metadata-cache entry can never reintroduce an unrepresentable decimals value.
impl Serialize for TokenDecimals {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u8(self.0)
    }
}

impl<'de> Deserialize<'de> for TokenDecimals {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<TokenDecimals, D::Error> {
        let value = u8::deserialize(deserializer)?;
        if value <= Self::MAX_SUPPORTED {
            Ok(TokenDecimals(value))
        } else {
            Err(D::Error::custom(format!(
                "token decimals {value} exceeds supported maximum {}",
                Self::MAX_SUPPORTED
            )))
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenMetadata {
    pub decimals: TokenDecimals,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenMetadataCall {
    Decimals,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TokenMetadataFailure {
    CallFailed(TokenMetadataCall),
    DecodeFailed(TokenMetadataCall),
    MissingResponse(TokenMetadataCall),
    UnsupportedDecimals(U256),
}

pub type TokenMetadataResult = Result<TokenMetadata, TokenMetadataFailure>;

// The native currency (`Address::ZERO`) is not an ERC20, so its `decimals()` cannot be fetched; it is
// a fixed protocol fact. Every chain currently in scope (Ethereum, Arbitrum) uses 18-decimal native
// ETH, including v4 native-ETH pools that store `token0 = address(0)`. Treating the native currency as
// intrinsically known keeps it out of the RPC token-metadata path while still resolving its decimals.
static NATIVE_TOKEN_METADATA: TokenMetadata = TokenMetadata {
    decimals: TokenDecimals(18),
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenRegistry {
    verified: HashMap<TokenAddress, TokenMetadata>,
    unsupported: HashMap<TokenAddress, TokenMetadataFailure>,
}

impl TokenRegistry {
    pub fn new() -> TokenRegistry {
        TokenRegistry {
            verified: HashMap::new(),
            unsupported: HashMap::new(),
        }
    }

    pub fn with_metadata_results(
        self,
        results: HashMap<TokenAddress, TokenMetadataResult>,
    ) -> TokenRegistry {
        let mut registry = self;

        for (token, result) in results {
            match result {
                Ok(metadata) => {
                    registry.unsupported.remove(&token);
                    registry.verified.insert(token, metadata);
                }
                Err(failure) if !registry.verified.contains_key(&token) => {
                    registry.unsupported.insert(token, failure);
                }
                Err(_) => {}
            }
        }

        registry
    }

    pub fn verified_metadata(&self, token: TokenAddress) -> Option<&TokenMetadata> {
        if token.0 == Address::ZERO {
            return Some(&NATIVE_TOKEN_METADATA);
        }
        self.verified.get(&token)
    }

    pub fn unsupported_failure(&self, token: TokenAddress) -> Option<&TokenMetadataFailure> {
        self.unsupported.get(&token)
    }

    pub fn is_unsupported(&self, token: TokenAddress) -> bool {
        self.unsupported.contains_key(&token)
    }

    pub fn is_known(&self, token: TokenAddress) -> bool {
        // `verified_metadata` resolves the native currency intrinsically, so this also reports the
        // native token as known — keeping `address(0)` out of `unknown_tokens` and the RPC path.
        self.verified_metadata(token).is_some() || self.is_unsupported(token)
    }

    pub fn unknown_tokens(&self, tokens: &HashSet<TokenAddress>) -> HashSet<TokenAddress> {
        tokens
            .iter()
            .copied()
            .filter(|token| !self.is_known(*token))
            .collect()
    }
}

#[cfg(test)]
impl TokenRegistry {
    pub(crate) fn verified_tokens_for_test(&self) -> HashSet<TokenAddress> {
        self.verified.keys().copied().collect()
    }
}

impl Default for TokenRegistry {
    fn default() -> Self {
        TokenRegistry::new()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use alloy::primitives::{Address, U256};
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn same_address_on_different_chains_is_a_distinct_token() {
        let address = Address::with_last_byte(7);
        let ethereum = TokenAddress(address, ChainKey::Ethereum);
        let arbitrum = TokenAddress(address, ChainKey::Arbitrum);

        // Cross-chain token identity must not collapse: the optimizer keys reserves and the model
        // layout by token, so a shared address across chains has to remain two distinct columns.
        assert_ne!(ethereum, arbitrum);
        assert_eq!(HashSet::from([ethereum, arbitrum]).len(), 2);
    }

    #[test]
    fn token_decimals_accepts_supported_range() {
        assert_eq!(
            TokenDecimals::try_from_u256(U256::from(0)).unwrap().value(),
            0
        );
        assert_eq!(
            TokenDecimals::try_from_u256(U256::from(36))
                .unwrap()
                .value(),
            36
        );
    }

    #[test]
    fn token_decimals_rejects_unsupported_values() {
        assert_eq!(
            TokenDecimals::try_from_u256(U256::from(37)),
            Err(TokenMetadataFailure::UnsupportedDecimals(U256::from(37)))
        );
    }

    #[test]
    fn registry_verifies_token_metadata() {
        let token = token(1);
        let metadata = token_metadata(6);

        let registry = TokenRegistry::new()
            .with_metadata_results(HashMap::from([(token, Ok(metadata.clone()))]));

        assert_eq!(registry.verified_metadata(token), Some(&metadata));
        assert!(!registry.is_unsupported(token));
        assert!(registry.is_known(token));
    }

    #[test]
    fn failed_validation_marks_unknown_token_unsupported() {
        let token = token(1);
        let failure = TokenMetadataFailure::CallFailed(TokenMetadataCall::Decimals);

        let registry = TokenRegistry::new()
            .with_metadata_results(HashMap::from([(token, Err(failure.clone()))]));

        assert_eq!(registry.verified_metadata(token), None);
        assert_eq!(registry.unsupported_failure(token), Some(&failure));
        assert!(registry.is_known(token));
    }

    #[test]
    fn successful_validation_removes_previous_unsupported_state() {
        let token = token(1);
        let metadata = token_metadata(18);

        let registry = TokenRegistry::new()
            .with_metadata_results(HashMap::from([(
                token,
                Err(TokenMetadataFailure::CallFailed(
                    TokenMetadataCall::Decimals,
                )),
            )]))
            .with_metadata_results(HashMap::from([(token, Ok(metadata.clone()))]));

        assert_eq!(registry.verified_metadata(token), Some(&metadata));
        assert!(!registry.is_unsupported(token));
    }

    #[test]
    fn failed_validation_does_not_overwrite_verified_token() {
        let token = token(1);
        let metadata = token_metadata(18);

        let registry = TokenRegistry::new()
            .with_metadata_results(HashMap::from([(token, Ok(metadata.clone()))]))
            .with_metadata_results(HashMap::from([(
                token,
                Err(TokenMetadataFailure::DecodeFailed(
                    TokenMetadataCall::Decimals,
                )),
            )]));

        assert_eq!(registry.verified_metadata(token), Some(&metadata));
        assert!(!registry.is_unsupported(token));
    }

    #[test]
    fn unknown_tokens_excludes_verified_and_unsupported_tokens() {
        let verified = token(1);
        let unsupported = token(2);
        let unknown = token(3);
        let registry = TokenRegistry::new()
            .with_metadata_results(HashMap::from([(verified, Ok(token_metadata(6)))]))
            .with_metadata_results(HashMap::from([(
                unsupported,
                Err(TokenMetadataFailure::MissingResponse(
                    TokenMetadataCall::Decimals,
                )),
            )]));

        assert_eq!(
            registry.unknown_tokens(&HashSet::from([verified, unsupported, unknown])),
            HashSet::from([unknown])
        );
    }

    proptest! {
        #[test]
        fn verified_and_unsupported_sets_never_overlap_after_applying_results(
            result_bytes in proptest::collection::vec((any::<u8>(), any::<bool>()), 0..64),
        ) {
            let results = result_bytes
                .into_iter()
                .map(|(address_byte, should_verify)| {
                    let token = token(address_byte);
                    let result = if should_verify {
                        Ok(token_metadata(address_byte % 37))
                    } else {
                        Err(TokenMetadataFailure::CallFailed(TokenMetadataCall::Decimals))
                    };
                    (token, result)
                })
                .collect::<HashMap<_, _>>();

            let registry = TokenRegistry::new().with_metadata_results(results);

            for token in registry.verified_tokens_for_test() {
                prop_assert!(!registry.is_unsupported(token));
            }
        }
    }

    #[test]
    fn native_currency_is_intrinsically_known_with_eighteen_decimals() {
        let native = TokenAddress(Address::ZERO, ChainKey::Ethereum);
        // A fresh registry has fetched nothing, yet native ETH resolves: it is a fixed protocol fact,
        // not an RPC-fetched value (the zero address is not an ERC20).
        let registry = TokenRegistry::new();

        assert_eq!(
            registry.verified_metadata(native).unwrap().decimals.value(),
            18
        );
        assert!(registry.is_known(native));
        assert!(!registry.is_unsupported(native));
    }

    #[test]
    fn unknown_tokens_excludes_the_native_currency() {
        let native = TokenAddress(Address::ZERO, ChainKey::Ethereum);
        let erc20 = token(1);
        let registry = TokenRegistry::new();

        // The native currency must never be requested over RPC, so it is never an unknown token.
        assert_eq!(
            registry.unknown_tokens(&HashSet::from([native, erc20])),
            HashSet::from([erc20])
        );
    }

    #[test]
    fn native_currency_is_known_on_every_chain() {
        // Native ETH on any supported chain is 18 decimals; the intrinsic lookup is chain-agnostic.
        for chain in [ChainKey::Ethereum, ChainKey::Arbitrum] {
            let native = TokenAddress(Address::ZERO, chain);
            assert_eq!(
                TokenRegistry::new()
                    .verified_metadata(native)
                    .unwrap()
                    .decimals
                    .value(),
                18
            );
        }
    }

    fn token(last_byte: u8) -> TokenAddress {
        TokenAddress(Address::with_last_byte(last_byte), ChainKey::Ethereum)
    }

    fn token_metadata(decimals: u8) -> TokenMetadata {
        TokenMetadata {
            decimals: TokenDecimals::try_from_u256(U256::from(decimals)).unwrap(),
        }
    }
}
