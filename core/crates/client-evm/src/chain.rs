#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ChainKey {
    Ethereum,
}

/// The chains the runtime tracks. The canonical, unique source of the active-chain set: the runtime
/// seeds one bootstrapping chain and one new-heads subscription per entry, so adding a chain here is
/// the single switch that activates it.
pub const ACTIVE_CHAINS: &[ChainKey] = &[ChainKey::Ethereum];

pub fn drpc_network_path(chain: ChainKey) -> &'static str {
    match chain {
        ChainKey::Ethereum => "ethereum",
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
    fn chain_key_can_be_used_as_btree_map_key() {
        let mut chains = BTreeMap::new();

        chains.insert(ChainKey::Ethereum, "active");

        assert_eq!(chains.get(&ChainKey::Ethereum), Some(&"active"));
    }
}
