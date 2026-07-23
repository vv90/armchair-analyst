use std::collections::{HashMap, HashSet};

use alloy::primitives::Address;
use serde::{Deserialize, Serialize};

use crate::ChainKey;
use crate::pool_state::{PoolRef, ProtocolPoolKey};

// `PoolMetadata` is immutable for a given pool address, so it is persisted to the metadata cache
// (`client::metadata_cache`) and reused across runs. Serde is the cache's on-disk representation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PoolMetadata {
    pub token0: Address,
    pub token1: Address,
    pub fee: PoolFee,
}

/// A pool's fee and tick spacing. Uniswap v3 pools use one of four fee tiers, whose tick spacing is
/// derived from the tier and so is never stored. Uniswap v4 pools set fee and tick spacing
/// independently, so both are stored; dynamic-fee pools (whose fee is set per-swap by a hook) fall
/// outside constant-fee swap math and are rejected at the v4 decode boundary rather than represented
/// here.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PoolFee {
    Tiered(UniswapV3Fee),
    Static { pips: u32, tick_spacing: u16 },
}

impl PoolFee {
    pub fn pips(self) -> u32 {
        match self {
            PoolFee::Tiered(tier) => tier.pips(),
            PoolFee::Static { pips, .. } => pips,
        }
    }

    pub fn tick_spacing(self) -> u16 {
        match self {
            // v3 tier spacings (1/10/60/200) are always in u16 range.
            PoolFee::Tiered(tier) => tier.tick_spacing() as u16,
            PoolFee::Static { tick_spacing, .. } => tick_spacing,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum UniswapV3Fee {
    Fee100,
    Fee500,
    Fee3000,
    Fee10000,
}

impl UniswapV3Fee {
    pub fn try_from_pips(fee: u32) -> Option<UniswapV3Fee> {
        match fee {
            100 => Some(UniswapV3Fee::Fee100),
            500 => Some(UniswapV3Fee::Fee500),
            3000 => Some(UniswapV3Fee::Fee3000),
            10000 => Some(UniswapV3Fee::Fee10000),
            _ => None,
        }
    }

    pub fn pips(self) -> u32 {
        match self {
            UniswapV3Fee::Fee100 => 100,
            UniswapV3Fee::Fee500 => 500,
            UniswapV3Fee::Fee3000 => 3000,
            UniswapV3Fee::Fee10000 => 10000,
        }
    }

    pub fn tick_spacing(self) -> i32 {
        match self {
            UniswapV3Fee::Fee100 => 1,
            UniswapV3Fee::Fee500 => 10,
            UniswapV3Fee::Fee3000 => 60,
            UniswapV3Fee::Fee10000 => 200,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoolMetadataCall {
    Token0,
    Token1,
    Fee,
    FactoryGetPool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PoolMetadataFailure {
    // Uniswap v3 (RPC multicall + factory validation) failures.
    CallFailed(PoolMetadataCall),
    DecodeFailed(PoolMetadataCall),
    MissingResponse(PoolMetadataCall),
    UnsupportedFee(u32),
    FactoryReturnedZero,
    FactoryMismatch { returned: Address },
    // Uniswap v4 key-validation rejections, raised by
    // `crate::uniswap_v4::pool_metadata_from_pool_key`.
    DynamicFee,
    HookedPool { hooks: Address },
    PoolIdMismatch,
    InvalidTickSpacing { tick_spacing: i32 },
    // Policy rejection: the pool's metadata resolved fine, but a token is outside the configured
    // whitelist (`crate::token_whitelist::TokenWhitelist::gate_pool_metadata_results`).
    TokenNotWhitelisted { token: Address },
}

pub type PoolMetadataResult = Result<PoolMetadata, PoolMetadataFailure>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrustedPoolRegistry {
    verified: HashMap<PoolRef, PoolMetadata>,
    rejected: HashSet<ProtocolPoolKey>,
}

impl TrustedPoolRegistry {
    pub fn new() -> TrustedPoolRegistry {
        TrustedPoolRegistry {
            verified: HashMap::new(),
            rejected: HashSet::new(),
        }
    }

    pub fn with_metadata_results(
        self,
        chain: ChainKey,
        results: HashMap<ProtocolPoolKey, PoolMetadataResult>,
    ) -> TrustedPoolRegistry {
        let mut registry = self;

        for (candidate, result) in results {
            let pool = PoolRef {
                key: candidate,
                chain,
            };

            match result {
                Ok(metadata) => {
                    registry.rejected.remove(&candidate);
                    registry.verified.insert(pool, metadata);
                }
                Err(_) if !registry.verified.contains_key(&pool) => {
                    registry.rejected.insert(candidate);
                }
                Err(_) => {}
            }
        }

        registry
    }

    pub fn verified_metadata(&self, pool: PoolRef) -> Option<&PoolMetadata> {
        self.verified.get(&pool)
    }

    /// Counts the pools the registry has verified.
    /// Added so read models can surface tracked-pool progress without exposing the backing map.
    pub fn verified_size(&self) -> usize {
        self.verified.len()
    }

    /// The verified pools tracked on `chain` — the identity set the graph fold watches and may seed.
    /// Added so the log-sourced graph (registry-free by design) can be handed the tracked-pool set to
    /// widen its bloom-watch and seed absolute-state logs for pools discovered after bootstrap.
    pub fn verified_pools(&self, chain: ChainKey) -> HashSet<PoolRef> {
        self.verified
            .keys()
            .copied()
            .filter(|pool| pool.chain == chain)
            .collect()
    }

    /// Returns the verified pool addresses on `chain`.
    /// Added so the log-fetch gate can test a block's `logsBloom` against the trusted-pool set and
    /// skip the authoritative fetch for blocks that provably touch none of them.
    pub fn verified_addresses(&self, chain: ChainKey) -> HashSet<Address> {
        self.verified
            .keys()
            .filter(|pool| pool.chain == chain)
            .filter_map(|pool| pool.uniswap_v3_address())
            .collect()
    }

    pub fn verified_pool(&self, chain: ChainKey, candidate: ProtocolPoolKey) -> Option<PoolRef> {
        let pool = PoolRef {
            key: candidate,
            chain,
        };
        self.verified.contains_key(&pool).then_some(pool)
    }

    pub fn is_rejected(&self, candidate: ProtocolPoolKey) -> bool {
        self.rejected.contains(&candidate)
    }

    pub fn is_known(&self, chain: ChainKey, candidate: ProtocolPoolKey) -> bool {
        self.verified_pool(chain, candidate).is_some() || self.is_rejected(candidate)
    }
}

#[cfg(test)]
impl TrustedPoolRegistry {
    pub(crate) fn verified_pools_for_test(&self) -> HashSet<PoolRef> {
        self.verified.keys().copied().collect()
    }
}

impl Default for TrustedPoolRegistry {
    fn default() -> Self {
        TrustedPoolRegistry::new()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use alloy::primitives::{Address, B256};
    use proptest::prelude::*;

    use super::*;
    use crate::PoolRef;
    use crate::uniswap_v4::PoolId;

    #[test]
    fn fee_tiers_derive_tick_spacing_without_storing_it_in_metadata() {
        assert_eq!(UniswapV3Fee::try_from_pips(100), Some(UniswapV3Fee::Fee100));
        assert_eq!(UniswapV3Fee::try_from_pips(500), Some(UniswapV3Fee::Fee500));
        assert_eq!(
            UniswapV3Fee::try_from_pips(3000),
            Some(UniswapV3Fee::Fee3000)
        );
        assert_eq!(
            UniswapV3Fee::try_from_pips(10000),
            Some(UniswapV3Fee::Fee10000)
        );
        assert_eq!(UniswapV3Fee::Fee100.tick_spacing(), 1);
        assert_eq!(UniswapV3Fee::Fee500.tick_spacing(), 10);
        assert_eq!(UniswapV3Fee::Fee3000.tick_spacing(), 60);
        assert_eq!(UniswapV3Fee::Fee10000.tick_spacing(), 200);

        let metadata = pool_metadata(1, 2, UniswapV3Fee::Fee3000);
        assert_eq!(metadata.fee.tick_spacing(), 60);
    }

    #[test]
    fn pool_fee_tiered_delegates_to_the_tier() {
        let fee = PoolFee::Tiered(UniswapV3Fee::Fee3000);
        assert_eq!(fee.pips(), 3000);
        assert_eq!(fee.tick_spacing(), 60);
    }

    #[test]
    fn pool_fee_static_returns_its_stored_pips_and_tick_spacing() {
        let fee = PoolFee::Static {
            pips: 450,
            tick_spacing: 7,
        };
        assert_eq!(fee.pips(), 450);
        assert_eq!(fee.tick_spacing(), 7);
    }

    #[test]
    fn registry_verifies_by_pool_address_and_keeps_metadata_address_free() {
        let candidate = candidate(3);
        let metadata = pool_metadata(1, 2, UniswapV3Fee::Fee500);

        let registry = TrustedPoolRegistry::new().with_metadata_results(
            ChainKey::Ethereum,
            HashMap::from([(candidate, Ok(metadata.clone()))]),
        );

        assert_eq!(
            registry.verified_metadata(PoolRef {
                key: candidate,
                chain: ChainKey::Ethereum
            }),
            Some(&metadata)
        );
        assert_eq!(
            registry.verified_pool(ChainKey::Ethereum, candidate),
            Some(PoolRef {
                key: candidate,
                chain: ChainKey::Ethereum
            })
        );
        assert!(!registry.is_rejected(candidate));
    }

    #[test]
    fn registry_verifies_a_v4_pool_id_candidate_alongside_v3() {
        let v3 = candidate(3);
        let v4 = v4_candidate(7);
        let v3_metadata = pool_metadata(1, 2, UniswapV3Fee::Fee500);
        let v4_metadata = static_metadata(4, 5);

        let registry = TrustedPoolRegistry::new().with_metadata_results(
            ChainKey::Ethereum,
            HashMap::from([(v3, Ok(v3_metadata.clone())), (v4, Ok(v4_metadata.clone()))]),
        );

        assert_eq!(
            registry.verified_metadata(PoolRef {
                key: v4,
                chain: ChainKey::Ethereum,
            }),
            Some(&v4_metadata)
        );
        assert_eq!(
            registry.verified_pool(ChainKey::Ethereum, v4),
            Some(PoolRef {
                key: v4,
                chain: ChainKey::Ethereum,
            })
        );
        // The two protocols coexist without colliding.
        assert_eq!(
            registry.verified_metadata(PoolRef {
                key: v3,
                chain: ChainKey::Ethereum,
            }),
            Some(&v3_metadata)
        );
        assert_eq!(registry.verified_size(), 2);
    }

    #[test]
    fn registry_rejects_a_failed_v4_candidate() {
        let v4 = v4_candidate(7);

        let registry = TrustedPoolRegistry::new().with_metadata_results(
            ChainKey::Ethereum,
            HashMap::from([(v4, Err(PoolMetadataFailure::DynamicFee))]),
        );

        assert!(registry.is_rejected(v4));
        assert!(registry.verified_pool(ChainKey::Ethereum, v4).is_none());
    }

    #[test]
    fn successful_validation_removes_previous_rejection_for_same_address() {
        let candidate = candidate(3);
        let metadata = pool_metadata(1, 2, UniswapV3Fee::Fee500);

        let registry = TrustedPoolRegistry::new()
            .with_metadata_results(
                ChainKey::Ethereum,
                HashMap::from([(candidate, Err(PoolMetadataFailure::FactoryReturnedZero))]),
            )
            .with_metadata_results(
                ChainKey::Ethereum,
                HashMap::from([(candidate, Ok(metadata))]),
            );

        assert_eq!(
            registry.verified_pool(ChainKey::Ethereum, candidate),
            Some(PoolRef {
                key: candidate,
                chain: ChainKey::Ethereum
            })
        );
        assert!(!registry.is_rejected(candidate));
    }

    #[test]
    fn verified_size_counts_verified_pools_only() {
        let registry = TrustedPoolRegistry::new().with_metadata_results(
            ChainKey::Ethereum,
            HashMap::from([
                (candidate(1), Ok(pool_metadata(1, 2, UniswapV3Fee::Fee500))),
                (candidate(2), Ok(pool_metadata(3, 4, UniswapV3Fee::Fee3000))),
                (candidate(3), Err(PoolMetadataFailure::FactoryReturnedZero)),
            ]),
        );

        assert_eq!(registry.verified_size(), 2);
    }

    #[test]
    fn failed_validation_does_not_reject_already_verified_pool() {
        let candidate = candidate(3);
        let metadata = pool_metadata(1, 2, UniswapV3Fee::Fee500);

        let registry = TrustedPoolRegistry::new()
            .with_metadata_results(
                ChainKey::Ethereum,
                HashMap::from([(candidate, Ok(metadata))]),
            )
            .with_metadata_results(
                ChainKey::Ethereum,
                HashMap::from([(
                    candidate,
                    Err(PoolMetadataFailure::FactoryMismatch {
                        returned: Address::with_last_byte(9),
                    }),
                )]),
            );

        assert_eq!(
            registry.verified_pool(ChainKey::Ethereum, candidate),
            Some(PoolRef {
                key: candidate,
                chain: ChainKey::Ethereum
            })
        );
        assert!(!registry.is_rejected(candidate));
    }

    #[test]
    fn verified_addresses_returns_only_the_requested_chains_pools() {
        let ethereum_pool = candidate(3);
        let arbitrum_pool = candidate(4);
        let registry = TrustedPoolRegistry::new()
            .with_metadata_results(
                ChainKey::Ethereum,
                HashMap::from([(ethereum_pool, Ok(pool_metadata(1, 2, UniswapV3Fee::Fee500)))]),
            )
            .with_metadata_results(
                ChainKey::Arbitrum,
                HashMap::from([(arbitrum_pool, Ok(pool_metadata(1, 2, UniswapV3Fee::Fee500)))]),
            );

        assert_eq!(
            registry.verified_addresses(ChainKey::Ethereum),
            HashSet::from([ethereum_pool.uniswap_v3_address().expect("v3 pool")])
        );
        assert_eq!(
            registry.verified_addresses(ChainKey::Arbitrum),
            HashSet::from([arbitrum_pool.uniswap_v3_address().expect("v3 pool")])
        );
    }

    // `trusted_pool_logs`/`TrustedPoolLogs`/`unknown_candidates` were deleted as dead API
    // (no production consumer since the Increment-4 swap; the kernel derives trust directly via
    // `verified_pool`/`is_known`); their tests — a unit pin and two proptest sections — went
    // with them. The remaining property below still pins every registry query production uses.
    proptest! {
        #[test]
        fn verified_and_rejected_sets_never_overlap_after_applying_results(
            result_bytes in proptest::collection::vec((any::<u8>(), any::<bool>()), 0..64),
        ) {
            let results = result_bytes
                .into_iter()
                .map(|(address_byte, should_verify)| {
                    let candidate = candidate(address_byte);
                    let result = if should_verify {
                        Ok(pool_metadata(1, 2, UniswapV3Fee::Fee3000))
                    } else {
                        Err(PoolMetadataFailure::CallFailed(PoolMetadataCall::Token0))
                    };
                    (candidate, result)
                })
                .collect::<HashMap<_, _>>();

            let registry = TrustedPoolRegistry::new().with_metadata_results(ChainKey::Ethereum, results);

            for pool in registry.verified_pools_for_test() {
                prop_assert!(!registry.is_rejected(pool.key));
            }
        }

        #[test]
        fn registry_sequential_results_preserve_known_state(
            batches in proptest::collection::vec(
                proptest::collection::hash_map(any::<u8>(), any::<bool>(), 0..16),
                0..32,
            ),
        ) {
            let mut registry = TrustedPoolRegistry::new();
            let mut expected_verified = HashSet::new();
            let mut expected_rejected = HashSet::new();
            let mut observed_candidates = HashSet::new();

            for batch in batches {
                let results = batch
                    .iter()
                    .map(|(address_byte, should_verify)| {
                        let candidate = candidate(*address_byte);
                        let result = if *should_verify {
                            Ok(pool_metadata(
                                *address_byte,
                                address_byte.wrapping_add(1),
                                UniswapV3Fee::Fee3000,
                            ))
                        } else {
                            Err(PoolMetadataFailure::CallFailed(PoolMetadataCall::Token0))
                        };

                        (candidate, result)
                    })
                    .collect::<HashMap<_, _>>();

                for (address_byte, should_verify) in batch {
                    let candidate = candidate(address_byte);
                    observed_candidates.insert(candidate);

                    if should_verify {
                        expected_rejected.remove(&candidate);
                        expected_verified.insert(candidate);
                    } else if !expected_verified.contains(&candidate) {
                        expected_rejected.insert(candidate);
                    }
                }

                registry = registry.with_metadata_results(ChainKey::Ethereum, results);

                for candidate in &observed_candidates {
                    prop_assert_eq!(
                        registry.verified_pool(ChainKey::Ethereum, *candidate).is_some(),
                        expected_verified.contains(candidate)
                    );
                    prop_assert_eq!(registry.is_rejected(*candidate), expected_rejected.contains(candidate));
                    prop_assert_eq!(
                        registry.is_known(ChainKey::Ethereum, *candidate),
                        expected_verified.contains(candidate) || expected_rejected.contains(candidate)
                    );
                }
            }
        }
    }

    fn candidate(last_byte: u8) -> ProtocolPoolKey {
        ProtocolPoolKey::UniswapV3(Address::with_last_byte(last_byte))
    }

    fn v4_candidate(last_byte: u8) -> ProtocolPoolKey {
        ProtocolPoolKey::UniswapV4(PoolId(B256::with_last_byte(last_byte)))
    }

    fn static_metadata(token0: u8, token1: u8) -> PoolMetadata {
        PoolMetadata {
            token0: Address::with_last_byte(token0),
            token1: Address::with_last_byte(token1),
            fee: PoolFee::Static {
                pips: 500,
                tick_spacing: 10,
            },
        }
    }

    fn pool_metadata(token0: u8, token1: u8, fee: UniswapV3Fee) -> PoolMetadata {
        PoolMetadata {
            token0: Address::with_last_byte(token0),
            token1: Address::with_last_byte(token1),
            fee: PoolFee::Tiered(fee),
        }
    }
}
