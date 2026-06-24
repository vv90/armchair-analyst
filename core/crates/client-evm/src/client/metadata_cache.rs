//! Persistent, address-keyed cache of immutable pool/token metadata, backed by redb.
//!
//! Pool metadata (`token0`/`token1`/`fee`) and token decimals never change once validated, so they
//! are stored once and reused across runs to avoid re-validating them via RPC on every bootstrap.
//! Pool *state* (reserves) is intentionally **not** cached — it is per-block and always fetched fresh.
//!
//! Address uniqueness is a property of the store, not of application logic: each row is keyed by
//! `(chain, raw-address)`, so there is exactly one entry per address per chain. `alloy::Address` is
//! already canonical 20-byte data (checksum casing is a display concern only), so the raw bytes are
//! the canonical key with no normalization step.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use redb::{Database, TableDefinition, backends::InMemoryBackend};

use crate::{
    ChainKey, PoolCandidateAddress, PoolMetadata, PoolMetadataResult, TokenAddress, TokenMetadata,
    TokenMetadataResult,
};

/// Composite key length: one chain tag byte followed by the 20-byte address.
const KEY_LEN: usize = 21;

const POOLS_TABLE: TableDefinition<&[u8], &[u8]> = TableDefinition::new("pool_metadata");
const TOKENS_TABLE: TableDefinition<&[u8], &[u8]> = TableDefinition::new("token_metadata");

#[derive(Debug, thiserror::Error)]
pub enum MetadataCacheError {
    #[error("metadata cache database error: {0}")]
    Database(String),
    #[error("metadata cache serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

fn database_error(error: impl std::fmt::Display) -> MetadataCacheError {
    MetadataCacheError::Database(error.to_string())
}

pub struct MetadataCache {
    database: Database,
}

impl MetadataCache {
    /// Opens (creating if absent) the on-disk metadata cache at `path`.
    pub fn open(path: &Path) -> Result<MetadataCache, MetadataCacheError> {
        let database = Database::create(path).map_err(database_error)?;
        MetadataCache::with_tables_created(database)
    }

    /// An in-memory cache with no on-disk backing, used by tests.
    pub fn in_memory() -> Result<MetadataCache, MetadataCacheError> {
        let database = Database::builder()
            .create_with_backend(InMemoryBackend::new())
            .map_err(database_error)?;
        MetadataCache::with_tables_created(database)
    }

    /// Materializes both tables up front so first-run reads never fail on a missing table.
    fn with_tables_created(database: Database) -> Result<MetadataCache, MetadataCacheError> {
        let write = database.begin_write().map_err(database_error)?;
        {
            write.open_table(POOLS_TABLE).map_err(database_error)?;
            write.open_table(TOKENS_TABLE).map_err(database_error)?;
        }
        write.commit().map_err(database_error)?;
        Ok(MetadataCache { database })
    }

    /// Returns the cached metadata for whichever requested candidates are already known.
    pub fn load_pool_metadata(
        &self,
        chain: ChainKey,
        candidates: &HashSet<PoolCandidateAddress>,
    ) -> Result<HashMap<PoolCandidateAddress, PoolMetadata>, MetadataCacheError> {
        let read = self.database.begin_read().map_err(database_error)?;
        let table = read.open_table(POOLS_TABLE).map_err(database_error)?;

        let mut hits = HashMap::new();
        for candidate in candidates {
            let key = composite_key(chain, candidate.0);
            if let Some(value) = table.get(key.as_slice()).map_err(database_error)? {
                hits.insert(*candidate, serde_json::from_slice(value.value())?);
            }
        }

        Ok(hits)
    }

    /// Persists every successfully validated pool metadata; failures and absent results are ignored.
    pub fn store_pool_metadata(
        &self,
        chain: ChainKey,
        results: &HashMap<PoolCandidateAddress, PoolMetadataResult>,
    ) -> Result<(), MetadataCacheError> {
        let entries = results
            .iter()
            .filter_map(|(candidate, result)| {
                result.as_ref().ok().map(|metadata| (candidate, metadata))
            })
            .collect::<Vec<_>>();
        if entries.is_empty() {
            return Ok(());
        }

        let write = self.database.begin_write().map_err(database_error)?;
        {
            let mut table = write.open_table(POOLS_TABLE).map_err(database_error)?;
            for (candidate, metadata) in entries {
                let key = composite_key(chain, candidate.0);
                let value = serde_json::to_vec(metadata)?;
                table
                    .insert(key.as_slice(), value.as_slice())
                    .map_err(database_error)?;
            }
        }
        write.commit().map_err(database_error)?;

        Ok(())
    }

    /// Returns every pool address stored for `chain` via a prefix range scan over the chain tag.
    ///
    /// Bootstrap uses this to widen its candidate set with the full known pool set, so a narrowed
    /// `finalized..tip` scan still re-activates every previously-validated pool (those resolve as
    /// cache hits; only genuinely new pools hit RPC).
    pub fn load_pool_addresses(
        &self,
        chain: ChainKey,
    ) -> Result<HashSet<PoolCandidateAddress>, MetadataCacheError> {
        let read = self.database.begin_read().map_err(database_error)?;
        let table = read.open_table(POOLS_TABLE).map_err(database_error)?;

        // The composite key sorts by chain tag first, so a chain's rows are one contiguous range
        // `[tag, 0x00..] ..= [tag, 0xFF..]` — no tag arithmetic, no risk of spanning another chain.
        let tag = chain_tag(chain);
        let mut lower = [0u8; KEY_LEN];
        lower[0] = tag;
        let mut upper = [0xFFu8; KEY_LEN];
        upper[0] = tag;
        let range = table
            .range::<&[u8]>(lower.as_slice()..=upper.as_slice())
            .map_err(database_error)?;

        let mut addresses = HashSet::new();
        for entry in range {
            let (key, _value) = entry.map_err(database_error)?;
            if let Some(address_bytes) = key.value().get(1..) {
                if let Ok(address) = alloy::primitives::Address::try_from(address_bytes) {
                    addresses.insert(PoolCandidateAddress(address));
                }
            }
        }

        Ok(addresses)
    }

    /// Returns the cached metadata for whichever requested tokens are already known.
    pub fn load_token_metadata(
        &self,
        tokens: &HashSet<TokenAddress>,
    ) -> Result<HashMap<TokenAddress, TokenMetadata>, MetadataCacheError> {
        let read = self.database.begin_read().map_err(database_error)?;
        let table = read.open_table(TOKENS_TABLE).map_err(database_error)?;

        let mut hits = HashMap::new();
        for token in tokens {
            let key = composite_key(token.1, token.0);
            if let Some(value) = table.get(key.as_slice()).map_err(database_error)? {
                hits.insert(*token, serde_json::from_slice(value.value())?);
            }
        }

        Ok(hits)
    }

    /// Persists every successfully validated token metadata; failures and absent results are ignored.
    pub fn store_token_metadata(
        &self,
        results: &HashMap<TokenAddress, TokenMetadataResult>,
    ) -> Result<(), MetadataCacheError> {
        let entries = results
            .iter()
            .filter_map(|(token, result)| result.as_ref().ok().map(|metadata| (token, metadata)))
            .collect::<Vec<_>>();
        if entries.is_empty() {
            return Ok(());
        }

        let write = self.database.begin_write().map_err(database_error)?;
        {
            let mut table = write.open_table(TOKENS_TABLE).map_err(database_error)?;
            for (token, metadata) in entries {
                let key = composite_key(token.1, token.0);
                let value = serde_json::to_vec(metadata)?;
                table
                    .insert(key.as_slice(), value.as_slice())
                    .map_err(database_error)?;
            }
        }
        write.commit().map_err(database_error)?;

        Ok(())
    }
}

/// One chain tag byte followed by the address bytes — unique per `(chain, address)`.
fn composite_key(chain: ChainKey, address: alloy::primitives::Address) -> [u8; KEY_LEN] {
    let mut key = [0u8; KEY_LEN];
    key[0] = chain_tag(chain);
    key[1..].copy_from_slice(address.as_slice());
    key
}

/// Stable, explicit chain discriminant for the key — not derived from enum ordering, so reordering
/// `ChainKey` can never silently repartition an existing cache.
fn chain_tag(chain: ChainKey) -> u8 {
    match chain {
        ChainKey::Ethereum => 0,
        ChainKey::Arbitrum => 1,
    }
}

#[cfg(test)]
mod tests {
    use alloy::primitives::{Address, U256};

    use super::*;
    use crate::{PoolFee, PoolMetadataCall, PoolMetadataFailure, TokenDecimals, UniswapV3Fee};

    fn pool_candidate(byte: u8) -> PoolCandidateAddress {
        PoolCandidateAddress(Address::with_last_byte(byte))
    }

    fn pool_metadata(byte: u8) -> PoolMetadata {
        PoolMetadata {
            token0: Address::with_last_byte(byte),
            token1: Address::with_last_byte(byte.wrapping_add(1)),
            fee: PoolFee::Tiered(UniswapV3Fee::Fee3000),
        }
    }

    fn token(byte: u8, chain: ChainKey) -> TokenAddress {
        TokenAddress(Address::with_last_byte(byte), chain)
    }

    fn token_metadata(decimals: u8) -> TokenMetadata {
        TokenMetadata {
            decimals: TokenDecimals::try_from_u256(U256::from(decimals)).unwrap(),
        }
    }

    #[test]
    fn pool_metadata_round_trips_through_the_cache() {
        let cache = MetadataCache::in_memory().unwrap();
        let candidate = pool_candidate(1);
        let metadata = pool_metadata(1);

        cache
            .store_pool_metadata(
                ChainKey::Ethereum,
                &HashMap::from([(candidate, Ok(metadata.clone()))]),
            )
            .unwrap();

        let loaded = cache
            .load_pool_metadata(ChainKey::Ethereum, &HashSet::from([candidate]))
            .unwrap();

        assert_eq!(loaded, HashMap::from([(candidate, metadata)]));
    }

    #[test]
    fn token_metadata_round_trips_through_the_cache() {
        let cache = MetadataCache::in_memory().unwrap();
        let token = token(2, ChainKey::Ethereum);
        let metadata = token_metadata(6);

        cache
            .store_token_metadata(&HashMap::from([(token, Ok(metadata.clone()))]))
            .unwrap();

        let loaded = cache.load_token_metadata(&HashSet::from([token])).unwrap();

        assert_eq!(loaded, HashMap::from([(token, metadata)]));
    }

    #[test]
    fn re_storing_the_same_address_is_idempotent() {
        let cache = MetadataCache::in_memory().unwrap();
        let candidate = pool_candidate(1);
        let metadata = pool_metadata(1);

        for _ in 0..3 {
            cache
                .store_pool_metadata(
                    ChainKey::Ethereum,
                    &HashMap::from([(candidate, Ok(metadata.clone()))]),
                )
                .unwrap();
        }

        let loaded = cache
            .load_pool_metadata(ChainKey::Ethereum, &HashSet::from([candidate]))
            .unwrap();

        assert_eq!(loaded, HashMap::from([(candidate, metadata)]));
    }

    #[test]
    fn same_address_on_different_chains_is_stored_separately() {
        let cache = MetadataCache::in_memory().unwrap();
        let ethereum = token(7, ChainKey::Ethereum);
        let arbitrum = token(7, ChainKey::Arbitrum);

        cache
            .store_token_metadata(&HashMap::from([
                (ethereum, Ok(token_metadata(6))),
                (arbitrum, Ok(token_metadata(18))),
            ]))
            .unwrap();

        let loaded = cache
            .load_token_metadata(&HashSet::from([ethereum, arbitrum]))
            .unwrap();

        assert_eq!(loaded.get(&ethereum), Some(&token_metadata(6)));
        assert_eq!(loaded.get(&arbitrum), Some(&token_metadata(18)));
    }

    #[test]
    fn failed_results_are_not_persisted() {
        let cache = MetadataCache::in_memory().unwrap();
        let candidate = pool_candidate(9);

        cache
            .store_pool_metadata(
                ChainKey::Ethereum,
                &HashMap::from([(
                    candidate,
                    Err(PoolMetadataFailure::CallFailed(PoolMetadataCall::Fee)),
                )]),
            )
            .unwrap();

        let loaded = cache
            .load_pool_metadata(ChainKey::Ethereum, &HashSet::from([candidate]))
            .unwrap();

        assert!(loaded.is_empty());
    }

    #[test]
    fn loading_an_absent_address_is_a_miss_not_an_error() {
        let cache = MetadataCache::in_memory().unwrap();

        let loaded = cache
            .load_pool_metadata(ChainKey::Ethereum, &HashSet::from([pool_candidate(3)]))
            .unwrap();

        assert!(loaded.is_empty());
    }

    #[test]
    fn load_pool_addresses_enumerates_only_the_requested_chain() {
        let cache = MetadataCache::in_memory().unwrap();
        let ethereum_a = pool_candidate(1);
        let ethereum_b = pool_candidate(2);
        let arbitrum = pool_candidate(1);

        cache
            .store_pool_metadata(
                ChainKey::Ethereum,
                &HashMap::from([
                    (ethereum_a, Ok(pool_metadata(1))),
                    (ethereum_b, Ok(pool_metadata(2))),
                ]),
            )
            .unwrap();
        cache
            .store_pool_metadata(
                ChainKey::Arbitrum,
                &HashMap::from([(arbitrum, Ok(pool_metadata(1)))]),
            )
            .unwrap();

        let ethereum = cache.load_pool_addresses(ChainKey::Ethereum).unwrap();

        // Same last byte exists on Arbitrum, but the chain-tagged prefix scan must not include it.
        assert_eq!(ethereum, HashSet::from([ethereum_a, ethereum_b]));
    }

    #[test]
    fn load_pool_addresses_is_empty_on_a_cold_table() {
        let cache = MetadataCache::in_memory().unwrap();

        let loaded = cache.load_pool_addresses(ChainKey::Ethereum).unwrap();

        assert!(loaded.is_empty());
    }

    #[test]
    fn corrupt_decimals_bytes_are_rejected_on_load() {
        // Defense in depth: a tampered value above the supported decimals bound must fail to
        // deserialize rather than reconstruct an invalid `TokenDecimals`.
        let result = serde_json::from_str::<TokenMetadata>("{\"decimals\":200}");

        assert!(result.is_err());
    }
}
