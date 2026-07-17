#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ChainKey {
    Ethereum,
    Arbitrum,
    Base,
    Optimism,
    Polygon,
    Bnb,
    Avalanche,
}

/// Every chain the client knows how to speak to, active or not. Slug resolution
/// ([`chain_key_for_network_path`]) stays total over this set so config and whitelist sections for
/// temporarily deactivated chains remain valid rather than becoming startup errors.
const ALL_CHAINS: &[ChainKey] = &[
    ChainKey::Ethereum,
    ChainKey::Arbitrum,
    ChainKey::Base,
    ChainKey::Optimism,
    ChainKey::Polygon,
    ChainKey::Bnb,
    ChainKey::Avalanche,
];

/// The chains the runtime tracks. The canonical, unique source of the active-chain set: the runtime
/// seeds one bootstrapping chain and one new-heads subscription per entry, so adding a chain here is
/// the single switch that activates it.
pub const ACTIVE_CHAINS: &[ChainKey] = &[
    ChainKey::Ethereum,
    // Arbitrum is temporarily deactivated: its high-rate, frequently-duplicated head stream
    // saturated the transition thread on every mainnet run. Re-enable once the fold/walk cost per
    // head is bounded (see the run-diagnostics work).
    // ChainKey::Arbitrum,
    ChainKey::Base,
    ChainKey::Optimism,
    // Polygon and Bnb are temporarily deactivated: every mainnet run 2026-07-10..16 showed the
    // same chronic stall on both (frontier frozen, behind climbing, provider getLogs refused or
    // silently dropped). Re-enable once per-provider backoff / archive-capable getLogs lands.
    // ChainKey::Polygon,
    // ChainKey::Bnb,
    ChainKey::Avalanche,
];

/// The network slug a chain is keyed by in the endpoints config file (the `[[rpc]]`/`[[subgraph]]`
/// per-chain table key). Despite the name this is purely the config-side identity — the full RPC URLs
/// are written out per chain in the config — so the slug is chosen for readability (e.g. `bnb`, not
/// dRPC's `bsc` path segment).
pub fn drpc_network_path(chain: ChainKey) -> &'static str {
    match chain {
        ChainKey::Ethereum => "ethereum",
        ChainKey::Arbitrum => "arbitrum",
        ChainKey::Base => "base",
        ChainKey::Optimism => "optimism",
        ChainKey::Polygon => "polygon",
        ChainKey::Bnb => "bnb",
        ChainKey::Avalanche => "avalanche",
    }
}

/// Inverse of [`drpc_network_path`]: resolves a known chain from its network slug. Used to map the
/// per-chain keys of the endpoints config file onto [`ChainKey`]. Total over [`ALL_CHAINS`] (not
/// just the active set) so deactivating a chain never invalidates existing config/whitelist files;
/// unknown slugs return `None`.
pub fn chain_key_for_network_path(path: &str) -> Option<ChainKey> {
    ALL_CHAINS
        .iter()
        .copied()
        .find(|&chain| drpc_network_path(chain) == path)
}

/// Block span of a single pool-candidate `eth_getLogs` window during bootstrap. Sized per chain so a
/// window stays under the free-tier providers' result caps (binding one: infura ~10000 results).
/// Derived from observed log density: Base ~76 logs/block, Optimism ~29; the rest are sparse enough
/// (or have a small finalized→tip gap) that one whole-gap window already serves them, so their value
/// only trades call count for headroom and cannot break their bootstrap.
pub fn pool_candidate_block_range_chunk(chain: ChainKey) -> u64 {
    match chain {
        ChainKey::Base => 100,    // ~76 logs/blk → ~7.6k logs/window
        ChainKey::Optimism => 250, // ~29 logs/blk → ~7.3k logs/window
        ChainKey::Ethereum
        | ChainKey::Arbitrum
        | ChainKey::Polygon
        | ChainKey::Bnb
        | ChainKey::Avalanche => 5000, // sparse / small gap; bounded, ~1–few calls
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn ethereum_maps_to_drpc_network_path() {
        assert_eq!(drpc_network_path(ChainKey::Ethereum), "ethereum");
    }

    #[test]
    fn arbitrum_maps_to_drpc_network_path() {
        assert_eq!(drpc_network_path(ChainKey::Arbitrum), "arbitrum");
    }

    #[test]
    fn every_known_chain_has_a_unique_slug_that_roundtrips() {
        let mut slugs = std::collections::HashSet::new();
        for &chain in ALL_CHAINS {
            let slug = drpc_network_path(chain);
            assert!(slugs.insert(slug), "duplicate network slug: {slug}");
            assert_eq!(chain_key_for_network_path(slug), Some(chain));
        }
    }

    #[test]
    fn pool_candidate_chunk_is_tight_for_dense_chains_and_large_otherwise() {
        assert_eq!(pool_candidate_block_range_chunk(ChainKey::Base), 100);
        assert_eq!(pool_candidate_block_range_chunk(ChainKey::Optimism), 250);
        assert_eq!(pool_candidate_block_range_chunk(ChainKey::Ethereum), 5000);
    }

    #[test]
    fn every_active_chain_has_a_non_zero_pool_candidate_chunk() {
        for &chain in ACTIVE_CHAINS {
            assert!(
                pool_candidate_block_range_chunk(chain) > 0,
                "chain {chain:?} has a zero pool-candidate chunk"
            );
        }
    }

    #[test]
    fn chain_key_can_be_used_as_btree_map_key() {
        let mut chains = BTreeMap::new();

        chains.insert(ChainKey::Ethereum, "active");

        assert_eq!(chains.get(&ChainKey::Ethereum), Some(&"active"));
    }
}
