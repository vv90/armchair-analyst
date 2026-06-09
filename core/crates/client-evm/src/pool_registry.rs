use std::collections::{HashMap, HashSet};

use alloy::primitives::Address;

use crate::PoolAddress;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct PoolCandidateAddress(pub Address);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PoolMetadata {
    pub token0: Address,
    pub token1: Address,
    pub fee: UniswapV3Fee,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
    CallFailed(PoolMetadataCall),
    DecodeFailed(PoolMetadataCall),
    MissingResponse(PoolMetadataCall),
    UnsupportedFee(u32),
    FactoryReturnedZero,
    FactoryMismatch { returned: Address },
}

pub type PoolMetadataResult = Result<PoolMetadata, PoolMetadataFailure>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TrustedPoolLogs {
    Unknown,
    PendingValidation,
    Resolved(HashSet<PoolAddress>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrustedPoolRegistry {
    verified: HashMap<PoolAddress, PoolMetadata>,
    rejected: HashSet<PoolCandidateAddress>,
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
        results: HashMap<PoolCandidateAddress, PoolMetadataResult>,
    ) -> TrustedPoolRegistry {
        let mut registry = self;

        for (candidate, result) in results {
            let pool = PoolAddress(candidate.0);

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

    pub fn verified_metadata(&self, pool: PoolAddress) -> Option<&PoolMetadata> {
        self.verified.get(&pool)
    }

    pub fn verified_pool(&self, candidate: PoolCandidateAddress) -> Option<PoolAddress> {
        let pool = PoolAddress(candidate.0);
        self.verified.contains_key(&pool).then_some(pool)
    }

    pub fn is_rejected(&self, candidate: PoolCandidateAddress) -> bool {
        self.rejected.contains(&candidate)
    }

    pub fn is_known(&self, candidate: PoolCandidateAddress) -> bool {
        self.verified_pool(candidate).is_some() || self.is_rejected(candidate)
    }

    pub fn unknown_candidates(
        &self,
        candidates: &HashSet<PoolCandidateAddress>,
    ) -> HashSet<PoolCandidateAddress> {
        candidates
            .iter()
            .copied()
            .filter(|candidate| !self.is_known(*candidate))
            .collect()
    }

    pub fn trusted_pool_logs(&self, candidates: &HashSet<PoolCandidateAddress>) -> TrustedPoolLogs {
        let mut pools = HashSet::new();
        for candidate in candidates {
            if let Some(pool) = self.verified_pool(*candidate) {
                pools.insert(pool);
            } else if !self.is_rejected(*candidate) {
                return TrustedPoolLogs::PendingValidation;
            }
        }

        TrustedPoolLogs::Resolved(pools)
    }
}

#[cfg(test)]
impl TrustedPoolRegistry {
    pub(crate) fn verified_pools_for_test(&self) -> HashSet<PoolAddress> {
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

    use alloy::primitives::Address;
    use proptest::prelude::*;

    use super::*;
    use crate::PoolAddress;

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
    fn registry_verifies_by_pool_address_and_keeps_metadata_address_free() {
        let candidate = candidate(3);
        let metadata = pool_metadata(1, 2, UniswapV3Fee::Fee500);

        let registry = TrustedPoolRegistry::new()
            .with_metadata_results(HashMap::from([(candidate, Ok(metadata.clone()))]));

        assert_eq!(
            registry.verified_metadata(PoolAddress(candidate.0)),
            Some(&metadata)
        );
        assert_eq!(
            registry.verified_pool(candidate),
            Some(PoolAddress(candidate.0))
        );
        assert!(!registry.is_rejected(candidate));
    }

    #[test]
    fn successful_validation_removes_previous_rejection_for_same_address() {
        let candidate = candidate(3);
        let metadata = pool_metadata(1, 2, UniswapV3Fee::Fee500);

        let registry = TrustedPoolRegistry::new()
            .with_metadata_results(HashMap::from([(
                candidate,
                Err(PoolMetadataFailure::FactoryReturnedZero),
            )]))
            .with_metadata_results(HashMap::from([(candidate, Ok(metadata))]));

        assert_eq!(
            registry.verified_pool(candidate),
            Some(PoolAddress(candidate.0))
        );
        assert!(!registry.is_rejected(candidate));
    }

    #[test]
    fn failed_validation_does_not_reject_already_verified_pool() {
        let candidate = candidate(3);
        let metadata = pool_metadata(1, 2, UniswapV3Fee::Fee500);

        let registry = TrustedPoolRegistry::new()
            .with_metadata_results(HashMap::from([(candidate, Ok(metadata))]))
            .with_metadata_results(HashMap::from([(
                candidate,
                Err(PoolMetadataFailure::FactoryMismatch {
                    returned: Address::with_last_byte(9),
                }),
            )]));

        assert_eq!(
            registry.verified_pool(candidate),
            Some(PoolAddress(candidate.0))
        );
        assert!(!registry.is_rejected(candidate));
    }

    #[test]
    fn trusted_pool_logs_are_pending_until_every_candidate_is_known() {
        let verified = candidate(3);
        let rejected = candidate(4);
        let pending = candidate(5);
        let registry = TrustedPoolRegistry::new()
            .with_metadata_results(HashMap::from([(
                verified,
                Ok(pool_metadata(1, 2, UniswapV3Fee::Fee3000)),
            )]))
            .with_metadata_results(HashMap::from([(
                rejected,
                Err(PoolMetadataFailure::FactoryReturnedZero),
            )]));

        assert_eq!(
            registry.trusted_pool_logs(&HashSet::from([verified, rejected, pending])),
            TrustedPoolLogs::PendingValidation
        );
        assert_eq!(
            registry.trusted_pool_logs(&HashSet::from([verified, rejected])),
            TrustedPoolLogs::Resolved(HashSet::from([PoolAddress(verified.0)]))
        );
    }

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

            let registry = TrustedPoolRegistry::new().with_metadata_results(results);

            for pool in registry.verified_pools_for_test() {
                prop_assert!(!registry.is_rejected(PoolCandidateAddress(pool.0)));
            }
        }

        #[test]
        fn registry_sequential_results_preserve_known_state(
            batches in proptest::collection::vec(
                proptest::collection::hash_map(any::<u8>(), any::<bool>(), 0..16),
                0..32,
            ),
            query_bytes in proptest::collection::hash_set(any::<u8>(), 0..32),
        ) {
            let mut registry = TrustedPoolRegistry::new();
            let mut expected_verified = HashSet::new();
            let mut expected_rejected = HashSet::new();
            let mut observed_candidates = HashSet::new();
            let query = query_bytes
                .into_iter()
                .map(candidate)
                .collect::<HashSet<_>>();

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

                registry = registry.with_metadata_results(results);

                for candidate in &observed_candidates {
                    prop_assert_eq!(
                        registry.verified_pool(*candidate).is_some(),
                        expected_verified.contains(candidate)
                    );
                    prop_assert_eq!(registry.is_rejected(*candidate), expected_rejected.contains(candidate));
                    prop_assert_eq!(
                        registry.is_known(*candidate),
                        expected_verified.contains(candidate) || expected_rejected.contains(candidate)
                    );
                }

                let expected_unknown = query
                    .iter()
                    .copied()
                    .filter(|candidate| !registry.is_known(*candidate))
                    .collect::<HashSet<_>>();
                prop_assert_eq!(registry.unknown_candidates(&query), expected_unknown);

                match registry.trusted_pool_logs(&query) {
                    TrustedPoolLogs::PendingValidation => {
                        prop_assert!(query.iter().any(|candidate| !registry.is_known(*candidate)));
                    }
                    TrustedPoolLogs::Resolved(pools) => {
                        let expected_pools = query
                            .iter()
                            .filter_map(|candidate| registry.verified_pool(*candidate))
                            .collect::<HashSet<_>>();

                        prop_assert!(query.iter().all(|candidate| registry.is_known(*candidate)));
                        prop_assert_eq!(pools, expected_pools);
                    }
                    TrustedPoolLogs::Unknown => {
                        prop_assert!(false, "registry-level trusted logs should not be unknown");
                    }
                }
            }
        }
    }

    fn candidate(last_byte: u8) -> PoolCandidateAddress {
        PoolCandidateAddress(Address::with_last_byte(last_byte))
    }

    fn pool_metadata(token0: u8, token1: u8, fee: UniswapV3Fee) -> PoolMetadata {
        PoolMetadata {
            token0: Address::with_last_byte(token0),
            token1: Address::with_last_byte(token1),
            fee,
        }
    }
}
