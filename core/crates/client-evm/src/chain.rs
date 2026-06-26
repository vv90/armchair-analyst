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

/// The chains the runtime tracks. The canonical, unique source of the active-chain set: the runtime
/// seeds one bootstrapping chain and one new-heads subscription per entry, so adding a chain here is
/// the single switch that activates it.
pub const ACTIVE_CHAINS: &[ChainKey] = &[
    ChainKey::Ethereum,
    ChainKey::Arbitrum,
    ChainKey::Base,
    ChainKey::Optimism,
    ChainKey::Polygon,
    ChainKey::Bnb,
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

/// Inverse of [`drpc_network_path`]: resolves an active chain from its network slug. Used to map the
/// per-chain keys of the endpoints config file onto [`ChainKey`]. Unknown slugs return `None`.
pub fn chain_key_for_network_path(path: &str) -> Option<ChainKey> {
    ACTIVE_CHAINS
        .iter()
        .copied()
        .find(|&chain| drpc_network_path(chain) == path)
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
    fn every_active_chain_has_a_unique_slug_that_roundtrips() {
        let mut slugs = std::collections::HashSet::new();
        for &chain in ACTIVE_CHAINS {
            let slug = drpc_network_path(chain);
            assert!(slugs.insert(slug), "duplicate network slug: {slug}");
            assert_eq!(chain_key_for_network_path(slug), Some(chain));
        }
    }

    #[test]
    fn chain_key_can_be_used_as_btree_map_key() {
        let mut chains = BTreeMap::new();

        chains.insert(ChainKey::Ethereum, "active");

        assert_eq!(chains.get(&ChainKey::Ethereum), Some(&"active"));
    }
}
