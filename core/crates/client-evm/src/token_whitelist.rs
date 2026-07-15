//! Per-chain static token whitelist: the shared artifact schema and its validated form.
//!
//! The whitelist is produced offline (by the `aa-token-vetting` tool, which will eventually examine
//! each token's contract before approving it) and consumed by the runtime to constrain which tokens
//! the optimizer may route through. This module owns the schema so the writer and the reader agree
//! by construction; the TOML codec itself stays in the leaf crates — they (de)serialize
//! [`TokenWhitelistFile`] and convert through [`TokenWhitelistFile::into_whitelist`].
//!
//! Semantics of a *present* whitelist are deny-by-default: a chain with no section in the file
//! allows no tokens on that chain. "No whitelist configured at all" (allow everything) is
//! represented by the absence of a [`TokenWhitelist`], not by a permissive value of one.

use std::collections::{BTreeMap, HashMap, HashSet};

use alloy::primitives::Address;
use serde::{Deserialize, Serialize};

use crate::chain::{ChainKey, chain_key_for_network_path};
use crate::kernel::pool_registry::{PoolMetadataFailure, PoolMetadataResult};
use crate::kernel::token_registry::TokenAddress;
use crate::pool_state::ProtocolPoolKey;

/// The on-disk artifact, exactly as written by the vetting tool. Chains are keyed by the same
/// network slugs as the endpoints config file (see [`crate::chain::drpc_network_path`]); token
/// addresses are hex strings via alloy's `Address` serde.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TokenWhitelistFile {
    /// Provenance only (RFC 3339 generation time); ignored by the loader.
    pub generated_at: Option<String>,
    /// Provenance only (examiner name/version that approved the tokens); ignored by the loader.
    pub examiner: Option<String>,
    #[serde(default)]
    pub chains: BTreeMap<String, ChainTokens>,
}

/// One chain's approved tokens.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChainTokens {
    pub tokens: Vec<TokenEntry>,
}

/// One approved token. Everything besides the address is provenance for human review of the
/// artifact; the runtime keys on the address alone.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TokenEntry {
    pub address: Address,
    pub symbol: Option<String>,
    pub decimals: Option<u8>,
    pub examined_at: Option<String>,
    pub tvl_usd: Option<f64>,
}

/// A validated whitelist. Constructible only via [`TokenWhitelistFile::into_whitelist`], so a
/// whitelist with an unknown chain slug is unrepresentable past the parse boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenWhitelist {
    tokens: HashSet<TokenAddress>,
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum TokenWhitelistError {
    #[error("unknown chain slug in token whitelist: {0}")]
    UnknownChainSlug(String),
}

impl TokenWhitelistFile {
    /// Validates the raw file into a queryable whitelist. A chain slug that does not name an
    /// active chain is an error rather than a skip: a typo'd slug silently dropping a whole
    /// chain's tokens would deny-all that chain in production.
    pub fn into_whitelist(self) -> Result<TokenWhitelist, TokenWhitelistError> {
        let mut tokens = HashSet::new();

        for (slug, chain_tokens) in self.chains {
            let chain = chain_key_for_network_path(&slug)
                .ok_or(TokenWhitelistError::UnknownChainSlug(slug))?;

            for entry in chain_tokens.tokens {
                tokens.insert(TokenAddress(entry.address, chain));
            }
        }

        Ok(TokenWhitelist { tokens })
    }
}

impl TokenWhitelist {
    pub fn allows(&self, token: TokenAddress) -> bool {
        self.tokens.contains(&token)
    }

    /// The full allowed set, for seeding the optimizer's session config.
    pub fn token_set(&self) -> &HashSet<TokenAddress> {
        &self.tokens
    }

    /// Whether the chain's native pseudo-token (the zero address, as used by Uniswap v4
    /// native-currency pools) is whitelisted. Omitting it silently excludes every native pool on
    /// the chain, so the runtime warns at startup when a chain's section lacks it.
    pub fn allows_native(&self, chain: ChainKey) -> bool {
        self.tokens.contains(&TokenAddress(Address::ZERO, chain))
    }

    /// The discovery gate: turns each successfully resolved pool whose `token0`/`token1` (as
    /// [`TokenAddress`]es on `chain`) is not whitelisted into
    /// [`PoolMetadataFailure::TokenNotWhitelisted`], so the kernel's registry rejects it exactly
    /// like a metadata failure — never verified, never watched, never folded, re-requests
    /// suppressed. Failures pass through untouched. Runs *after* the metadata cache stores the
    /// true `Ok` results, so widening the whitelist on a later run re-admits gated pools from
    /// cache without refetching.
    pub fn gate_pool_metadata_results(
        &self,
        chain: ChainKey,
        results: HashMap<ProtocolPoolKey, PoolMetadataResult>,
    ) -> HashMap<ProtocolPoolKey, PoolMetadataResult> {
        results
            .into_iter()
            .map(|(candidate, result)| {
                let gated = match result {
                    Ok(metadata) => {
                        match [metadata.token0, metadata.token1]
                            .into_iter()
                            .find(|&token| !self.allows(TokenAddress(token, chain)))
                        {
                            Some(token) => Err(PoolMetadataFailure::TokenNotWhitelisted { token }),
                            None => Ok(metadata),
                        }
                    }
                    Err(failure) => Err(failure),
                };
                (candidate, gated)
            })
            .collect()
    }

    /// Whitelisted-token count per chain, for the startup log line. Chains with no whitelisted
    /// tokens are absent — the caller compares against the active-chain set to warn about them.
    pub fn chain_counts(&self) -> BTreeMap<ChainKey, usize> {
        let mut counts = BTreeMap::new();

        for token in &self.tokens {
            *counts.entry(token.1).or_insert(0) += 1;
        }

        counts
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain::drpc_network_path;
    use alloy::primitives::address;
    use serde_json::json;

    fn entry(address: Address) -> TokenEntry {
        TokenEntry {
            address,
            symbol: None,
            decimals: None,
            examined_at: None,
            tvl_usd: None,
        }
    }

    const USDC: Address = address!("a0b86991c6218b36c1d19d4a2e9eb0ce3606eb48");

    #[test]
    fn file_round_trips_through_serde_and_validates() {
        let file = TokenWhitelistFile {
            generated_at: Some("2026-07-15T12:00:00Z".to_string()),
            examiner: Some("approve-all/0.1.0".to_string()),
            chains: BTreeMap::from([(
                "ethereum".to_string(),
                ChainTokens {
                    tokens: vec![entry(USDC)],
                },
            )]),
        };

        let value = serde_json::to_value(&file).unwrap();
        let parsed: TokenWhitelistFile = serde_json::from_value(value).unwrap();
        assert_eq!(parsed, file);

        let whitelist = parsed.into_whitelist().unwrap();
        assert!(whitelist.allows(TokenAddress(USDC, ChainKey::Ethereum)));
    }

    #[test]
    fn addresses_deserialize_from_hex_strings() {
        let value = json!({
            "chains": {
                "ethereum": {
                    "tokens": [{ "address": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48" }]
                }
            }
        });

        let file: TokenWhitelistFile = serde_json::from_value(value).unwrap();
        let whitelist = file.into_whitelist().unwrap();

        assert!(whitelist.allows(TokenAddress(USDC, ChainKey::Ethereum)));
    }

    #[test]
    fn unknown_chain_slug_is_rejected() {
        let file = TokenWhitelistFile {
            generated_at: None,
            examiner: None,
            chains: BTreeMap::from([(
                "etherium".to_string(),
                ChainTokens {
                    tokens: vec![entry(USDC)],
                },
            )]),
        };

        assert_eq!(
            file.into_whitelist(),
            Err(TokenWhitelistError::UnknownChainSlug(
                "etherium".to_string()
            ))
        );
    }

    #[test]
    fn allows_distinguishes_the_same_address_on_different_chains() {
        let file = TokenWhitelistFile {
            generated_at: None,
            examiner: None,
            chains: BTreeMap::from([(
                "ethereum".to_string(),
                ChainTokens {
                    tokens: vec![entry(USDC)],
                },
            )]),
        };
        let whitelist = file.into_whitelist().unwrap();

        assert!(whitelist.allows(TokenAddress(USDC, ChainKey::Ethereum)));
        assert!(!whitelist.allows(TokenAddress(USDC, ChainKey::Arbitrum)));
    }

    #[test]
    fn absent_chain_denies_all_tokens_on_that_chain() {
        let file = TokenWhitelistFile {
            generated_at: None,
            examiner: None,
            chains: BTreeMap::from([(
                "ethereum".to_string(),
                ChainTokens {
                    tokens: vec![entry(USDC)],
                },
            )]),
        };
        let whitelist = file.into_whitelist().unwrap();

        assert!(!whitelist.allows(TokenAddress(Address::ZERO, ChainKey::Base)));
        assert!(!whitelist.chain_counts().contains_key(&ChainKey::Base));
    }

    #[test]
    fn chain_counts_deduplicate_repeated_entries() {
        let file = TokenWhitelistFile {
            generated_at: None,
            examiner: None,
            chains: BTreeMap::from([
                (
                    "ethereum".to_string(),
                    ChainTokens {
                        tokens: vec![entry(USDC), entry(USDC), entry(Address::ZERO)],
                    },
                ),
                (
                    "base".to_string(),
                    ChainTokens {
                        tokens: vec![entry(Address::ZERO)],
                    },
                ),
            ]),
        };
        let whitelist = file.into_whitelist().unwrap();

        assert_eq!(
            whitelist.chain_counts(),
            BTreeMap::from([(ChainKey::Ethereum, 2), (ChainKey::Base, 1)])
        );
    }

    mod gate {
        use super::*;
        use crate::kernel::pool_registry::{
            PoolFee, PoolMetadata, TrustedPoolRegistry, UniswapV3Fee,
        };
        use crate::uniswap_v4::PoolId;
        use alloy::primitives::B256;

        const WETH: Address = address!("c02aaa39b223fe8d0a0e5c4f27ead9083c756cc2");
        const UNLISTED: Address = address!("1111111111111111111111111111111111111111");

        fn whitelist() -> TokenWhitelist {
            TokenWhitelistFile {
                generated_at: None,
                examiner: None,
                chains: BTreeMap::from([(
                    "ethereum".to_string(),
                    ChainTokens {
                        tokens: vec![entry(USDC), entry(WETH)],
                    },
                )]),
            }
            .into_whitelist()
            .unwrap()
        }

        fn metadata(token0: Address, token1: Address) -> PoolMetadata {
            PoolMetadata {
                token0,
                token1,
                fee: PoolFee::Tiered(UniswapV3Fee::Fee500),
            }
        }

        fn v3_key(byte: u8) -> ProtocolPoolKey {
            ProtocolPoolKey::UniswapV3(Address::repeat_byte(byte))
        }

        #[test]
        fn passes_a_pool_with_both_tokens_whitelisted() {
            let key = v3_key(1);
            let results = HashMap::from([(key, Ok(metadata(USDC, WETH)))]);

            let gated = whitelist().gate_pool_metadata_results(ChainKey::Ethereum, results);

            assert_eq!(gated.get(&key), Some(&Ok(metadata(USDC, WETH))));
        }

        #[test]
        fn rejects_a_pool_with_a_non_whitelisted_token() {
            let key = v3_key(1);
            let results = HashMap::from([(key, Ok(metadata(USDC, UNLISTED)))]);

            let gated = whitelist().gate_pool_metadata_results(ChainKey::Ethereum, results);

            assert_eq!(
                gated.get(&key),
                Some(&Err(PoolMetadataFailure::TokenNotWhitelisted {
                    token: UNLISTED
                }))
            );
        }

        #[test]
        fn whitelisting_is_chain_scoped() {
            // The same addresses on a chain with no whitelist section must be rejected.
            let key = v3_key(1);
            let results = HashMap::from([(key, Ok(metadata(USDC, WETH)))]);

            let gated = whitelist().gate_pool_metadata_results(ChainKey::Arbitrum, results);

            assert_eq!(
                gated.get(&key),
                Some(&Err(PoolMetadataFailure::TokenNotWhitelisted {
                    token: USDC
                }))
            );
        }

        #[test]
        fn leaves_existing_failures_untouched() {
            let key = v3_key(1);
            let results = HashMap::from([(key, Err(PoolMetadataFailure::FactoryReturnedZero))]);

            let gated = whitelist().gate_pool_metadata_results(ChainKey::Ethereum, results);

            assert_eq!(
                gated.get(&key),
                Some(&Err(PoolMetadataFailure::FactoryReturnedZero))
            );
        }

        #[test]
        fn gates_v4_keys_like_v3_keys() {
            let key = ProtocolPoolKey::UniswapV4(PoolId(B256::repeat_byte(7)));
            let results = HashMap::from([(key, Ok(metadata(UNLISTED, WETH)))]);

            let gated = whitelist().gate_pool_metadata_results(ChainKey::Ethereum, results);

            assert_eq!(
                gated.get(&key),
                Some(&Err(PoolMetadataFailure::TokenNotWhitelisted {
                    token: UNLISTED
                }))
            );
        }

        #[test]
        fn gated_pools_land_in_the_registry_rejected_set() {
            // Composition pin: a gated result must behave exactly like a metadata failure in the
            // kernel registry — rejected (never verified) and known (re-requests suppressed).
            let key = v3_key(1);
            let results = HashMap::from([(key, Ok(metadata(USDC, UNLISTED)))]);

            let gated = whitelist().gate_pool_metadata_results(ChainKey::Ethereum, results);
            let registry =
                TrustedPoolRegistry::new().with_metadata_results(ChainKey::Ethereum, gated);

            assert!(registry.is_known(ChainKey::Ethereum, key));
            assert!(
                registry.verified_pool(ChainKey::Ethereum, key).is_none(),
                "a gated pool must never be verified"
            );
        }
    }

    #[test]
    fn every_active_chain_slug_is_accepted() {
        let chains = crate::chain::ACTIVE_CHAINS
            .iter()
            .map(|&chain| {
                (
                    drpc_network_path(chain).to_string(),
                    ChainTokens {
                        tokens: vec![entry(Address::ZERO)],
                    },
                )
            })
            .collect();

        let file = TokenWhitelistFile {
            generated_at: None,
            examiner: None,
            chains,
        };
        let whitelist = file.into_whitelist().unwrap();

        assert_eq!(
            whitelist.chain_counts().len(),
            crate::chain::ACTIVE_CHAINS.len()
        );
    }
}
