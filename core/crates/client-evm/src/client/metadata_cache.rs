//! Persistent cache of immutable pool/token metadata, backed by redb.
//!
//! Pool metadata (`token0`/`token1`/`fee`) and token decimals never change once validated, so they
//! are stored once and reused across runs to avoid re-validating them via RPC on every bootstrap.
//! Pool *state* (reserves) is intentionally **not** cached — it is per-block and always fetched fresh.
//!
//! Identity uniqueness is a property of the store, not of application logic. Tokens are keyed by
//! `(chain, raw-address)`. Pools are keyed by `(chain, protocol, identity)` — a v3 pool's identity
//! is its 20-byte address, a v4 pool's is its 32-byte `PoolId`; the explicit protocol tag keeps the
//! two from ever colliding even when their bytes overlap.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use alloy::primitives::{Address, B256};
use redb::{Database, TableDefinition, backends::InMemoryBackend};

use crate::uniswap_v4::PoolId;
use crate::{
    ChainKey, PoolMetadata, PoolMetadataResult, ProtocolPoolKey, TokenAddress, TokenMetadata,
    TokenMetadataResult,
};

/// Token composite key length: one chain tag byte followed by the 20-byte address.
const KEY_LEN: usize = 21;

/// Stable, explicit protocol discriminant for pool keys — like `chain_tag`, not derived from enum
/// ordering, so the on-disk format is decoupled from `ProtocolPoolKey`'s declaration order.
const PROTO_TAG_V3: u8 = 0;
const PROTO_TAG_V4: u8 = 1;

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
        candidates: &HashSet<ProtocolPoolKey>,
    ) -> Result<HashMap<ProtocolPoolKey, PoolMetadata>, MetadataCacheError> {
        let read = self.database.begin_read().map_err(database_error)?;
        let table = read.open_table(POOLS_TABLE).map_err(database_error)?;

        let mut hits = HashMap::new();
        for candidate in candidates {
            let key = pool_key(chain, *candidate);
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
        results: &HashMap<ProtocolPoolKey, PoolMetadataResult>,
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
                let key = pool_key(chain, *candidate);
                let value = serde_json::to_vec(metadata)?;
                table
                    .insert(key.as_slice(), value.as_slice())
                    .map_err(database_error)?;
            }
        }
        write.commit().map_err(database_error)?;

        Ok(())
    }

    /// Returns every pool candidate stored for `chain` (both protocols) via a prefix range scan over
    /// the chain tag.
    ///
    /// Bootstrap uses this to widen its candidate set with the full known pool set, so a narrowed
    /// `finalized..tip` scan still re-activates every previously-validated pool (those resolve as
    /// cache hits; only genuinely new pools hit RPC).
    pub fn load_pool_candidates(
        &self,
        chain: ChainKey,
    ) -> Result<HashSet<ProtocolPoolKey>, MetadataCacheError> {
        let read = self.database.begin_read().map_err(database_error)?;
        let table = read.open_table(POOLS_TABLE).map_err(database_error)?;

        // Pool keys sort by chain tag first, so a chain's rows (of either protocol and any length)
        // are exactly the half-open range `[tag] .. [tag + 1]`. The chain tag is 0 or 1, so the
        // `tag + 1` upper bound never overflows.
        let tag = chain_tag(chain);
        let lower = [tag];
        let upper = [tag + 1];
        let range = table
            .range::<&[u8]>(lower.as_slice()..upper.as_slice())
            .map_err(database_error)?;

        let mut candidates = HashSet::new();
        for entry in range {
            let (key, _value) = entry.map_err(database_error)?;
            if let Some(candidate) = key.value().get(1..).and_then(decode_pool_candidate) {
                candidates.insert(candidate);
            }
        }

        Ok(candidates)
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

/// One chain tag byte followed by the token address bytes — unique per `(chain, address)`.
fn composite_key(chain: ChainKey, address: alloy::primitives::Address) -> [u8; KEY_LEN] {
    let mut key = [0u8; KEY_LEN];
    key[0] = chain_tag(chain);
    key[1..].copy_from_slice(address.as_slice());
    key
}

/// Pool key: chain tag, protocol tag, then the protocol identity bytes (20 for a v3 address, 32 for
/// a v4 `PoolId`). The protocol tag prevents a v3 address and a v4 `PoolId` from colliding.
fn pool_key(chain: ChainKey, candidate: ProtocolPoolKey) -> Vec<u8> {
    let mut key = vec![chain_tag(chain)];
    match candidate {
        ProtocolPoolKey::UniswapV3(address) => {
            key.push(PROTO_TAG_V3);
            key.extend_from_slice(address.as_slice());
        }
        ProtocolPoolKey::UniswapV4(pool_id) => {
            key.push(PROTO_TAG_V4);
            key.extend_from_slice(pool_id.0.as_slice());
        }
    }
    key
}

/// Decodes a pool key's identity portion (the bytes after the chain tag) back into a
/// `ProtocolPoolKey`. Returns `None` for an unknown protocol tag or an identity of the wrong length.
fn decode_pool_candidate(identity: &[u8]) -> Option<ProtocolPoolKey> {
    let (proto_tag, id_bytes) = identity.split_first()?;
    match *proto_tag {
        PROTO_TAG_V3 => Address::try_from(id_bytes)
            .ok()
            .map(ProtocolPoolKey::UniswapV3),
        PROTO_TAG_V4 => B256::try_from(id_bytes)
            .ok()
            .map(|bytes| ProtocolPoolKey::UniswapV4(PoolId(bytes))),
        _ => None,
    }
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
    use alloy::primitives::{Address, B256, U256};

    use super::*;
    use crate::{PoolFee, PoolMetadataCall, PoolMetadataFailure, TokenDecimals, UniswapV3Fee};

    fn pool_candidate(byte: u8) -> ProtocolPoolKey {
        ProtocolPoolKey::UniswapV3(Address::with_last_byte(byte))
    }

    fn v4_pool_candidate(byte: u8) -> ProtocolPoolKey {
        ProtocolPoolKey::UniswapV4(PoolId(B256::with_last_byte(byte)))
    }

    fn pool_metadata(byte: u8) -> PoolMetadata {
        PoolMetadata {
            token0: Address::with_last_byte(byte),
            token1: Address::with_last_byte(byte.wrapping_add(1)),
            fee: PoolFee::Tiered(UniswapV3Fee::Fee3000),
        }
    }

    fn v4_pool_metadata(byte: u8) -> PoolMetadata {
        PoolMetadata {
            token0: Address::with_last_byte(byte),
            token1: Address::with_last_byte(byte.wrapping_add(1)),
            fee: PoolFee::Static {
                pips: 500,
                tick_spacing: 10,
            },
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
    fn load_pool_candidates_enumerates_only_the_requested_chain() {
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

        let ethereum = cache.load_pool_candidates(ChainKey::Ethereum).unwrap();

        // Same last byte exists on Arbitrum, but the chain-tagged prefix scan must not include it.
        assert_eq!(ethereum, HashSet::from([ethereum_a, ethereum_b]));
    }

    #[test]
    fn load_pool_candidates_returns_both_protocols_for_a_chain() {
        let cache = MetadataCache::in_memory().unwrap();
        let v3 = pool_candidate(1);
        let v4 = v4_pool_candidate(2);

        cache
            .store_pool_metadata(
                ChainKey::Ethereum,
                &HashMap::from([(v3, Ok(pool_metadata(1))), (v4, Ok(v4_pool_metadata(2)))]),
            )
            .unwrap();

        let loaded = cache.load_pool_candidates(ChainKey::Ethereum).unwrap();

        assert_eq!(loaded, HashSet::from([v3, v4]));
    }

    #[test]
    fn v4_pool_metadata_round_trips_through_the_cache() {
        let cache = MetadataCache::in_memory().unwrap();
        let candidate = v4_pool_candidate(9);
        let metadata = v4_pool_metadata(9);

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
    fn v3_address_and_v4_pool_id_with_shared_bytes_do_not_collide() {
        let cache = MetadataCache::in_memory().unwrap();
        // Both identities share the last byte; the protocol tag must keep their keys distinct.
        let v3 = ProtocolPoolKey::UniswapV3(Address::with_last_byte(0xAB));
        let v4 = ProtocolPoolKey::UniswapV4(PoolId(B256::with_last_byte(0xAB)));

        cache
            .store_pool_metadata(
                ChainKey::Ethereum,
                &HashMap::from([(v3, Ok(pool_metadata(1))), (v4, Ok(v4_pool_metadata(2)))]),
            )
            .unwrap();

        let loaded = cache
            .load_pool_metadata(ChainKey::Ethereum, &HashSet::from([v3, v4]))
            .unwrap();

        assert_eq!(loaded.get(&v3), Some(&pool_metadata(1)));
        assert_eq!(loaded.get(&v4), Some(&v4_pool_metadata(2)));
    }

    #[test]
    fn load_pool_candidates_is_empty_on_a_cold_table() {
        let cache = MetadataCache::in_memory().unwrap();

        let loaded = cache.load_pool_candidates(ChainKey::Ethereum).unwrap();

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
