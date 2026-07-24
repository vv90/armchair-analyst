//! [`MetadataCatalog`] — an O(1)-cloneable snapshot of the kernel's verified pool and token
//! metadata, built from the two persistent registry maps ([`super::pool_registry`],
//! [`super::token_registry`]). Cloning it shares the maps' roots rather than copying entries, so a
//! reader thread (e.g. the aa-server `/pools/meta` endpoint) can be handed the whole tracked set
//! without duplicating it. The `imbl` backing type stays private — external callers see only
//! `MetadataCatalog`, never the persistent-map type.

use imbl::HashMap as ImHashMap;

use super::pool_registry::PoolMetadata;
use super::token_registry::{TokenAddress, TokenMetadata};
use crate::pool_state::PoolRef;

/// A shared snapshot of the verified pool + token metadata. `Clone` is O(1) (each field is a
/// persistent-map handle whose clone shares the root); `Default` is the empty catalog.
///
/// **Chain-unfiltered.** A per-chain kernel's registry only ever holds that chain's entries, so this
/// view equals the chain set. A multi-chain caller would have to filter downstream (an O(n) walk),
/// which is why no chain filter lives here.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct MetadataCatalog {
    pools: ImHashMap<PoolRef, PoolMetadata>,
    tokens: ImHashMap<TokenAddress, TokenMetadata>,
}

impl MetadataCatalog {
    /// Builds a catalog from the registries' O(1) persistent-map views. `pub(crate)` because it takes
    /// the internal `imbl` handles; external callers construct one only via
    /// [`crate::kernel::State::metadata_catalog`].
    pub(crate) fn from_views(
        pools: ImHashMap<PoolRef, PoolMetadata>,
        tokens: ImHashMap<TokenAddress, TokenMetadata>,
    ) -> MetadataCatalog {
        MetadataCatalog { pools, tokens }
    }

    /// Iterates the verified pools as `(PoolRef, &PoolMetadata)`. Order is unspecified.
    pub fn iter_pools(&self) -> impl Iterator<Item = (PoolRef, &PoolMetadata)> {
        self.pools.iter().map(|(pool, metadata)| (*pool, metadata))
    }

    /// Verified metadata for `token`, if the registry holds it. Unlike the token registry's own
    /// lookup, this does not resolve the native currency intrinsically — the catalog surfaces only
    /// what was validated and cached (aa-server, its sole consumer, tracks no native-token pools).
    pub fn token_metadata(&self, token: TokenAddress) -> Option<&TokenMetadata> {
        self.tokens.get(&token)
    }

    /// The number of verified pools in the catalog.
    pub fn pool_count(&self) -> usize {
        self.pools.len()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use alloy::primitives::{Address, B256, U256};

    use super::*;
    use crate::kernel::State;
    use crate::kernel::pool_registry::{PoolFee, TrustedPoolRegistry, UniswapV3Fee};
    use crate::kernel::token_registry::{TokenDecimals, TokenRegistry};
    use crate::pool_state::ProtocolPoolKey;
    use crate::uniswap_v4::PoolId;
    use crate::{BlockHash, ChainKey};

    const CHAIN: ChainKey = ChainKey::Ethereum;

    fn token(byte: u8) -> Address {
        Address::with_last_byte(byte)
    }

    fn pool_metadata(token0: u8, token1: u8) -> PoolMetadata {
        PoolMetadata {
            token0: token(token0),
            token1: token(token1),
            fee: PoolFee::Tiered(UniswapV3Fee::Fee3000),
        }
    }

    fn decimals(value: u8) -> TokenMetadata {
        TokenMetadata {
            decimals: TokenDecimals::try_from_u256(U256::from(value)).expect("in range"),
        }
    }

    fn registries() -> (TrustedPoolRegistry, TokenRegistry) {
        let v3 = ProtocolPoolKey::UniswapV3(token(0x11));
        let v4 = ProtocolPoolKey::UniswapV4(PoolId(B256::with_last_byte(0x22)));
        let pool_registry = TrustedPoolRegistry::new().with_metadata_results(
            CHAIN,
            HashMap::from([(v3, Ok(pool_metadata(1, 2))), (v4, Ok(pool_metadata(1, 2)))]),
        );
        let token_registry = TokenRegistry::new().with_metadata_results(HashMap::from([
            (TokenAddress(token(1), CHAIN), Ok(decimals(18))),
            (TokenAddress(token(2), CHAIN), Ok(decimals(6))),
        ]));
        (pool_registry, token_registry)
    }

    #[test]
    fn catalog_surfaces_the_verified_pools_and_tokens() {
        let (pool_registry, token_registry) = registries();
        let catalog = MetadataCatalog::from_views(
            pool_registry.verified_view(),
            token_registry.verified_view(),
        );

        assert_eq!(catalog.pool_count(), 2);
        assert_eq!(catalog.iter_pools().count(), 2);
        assert_eq!(
            catalog.token_metadata(TokenAddress(token(1), CHAIN)),
            Some(&decimals(18))
        );
        assert_eq!(
            catalog.token_metadata(TokenAddress(token(2), CHAIN)),
            Some(&decimals(6))
        );
        // A token no pool referenced (and that was never validated) is simply absent.
        assert_eq!(catalog.token_metadata(TokenAddress(token(9), CHAIN)), None);
    }

    #[test]
    fn from_views_shares_the_map_roots_instead_of_copying_entries() {
        let (pool_registry, token_registry) = registries();
        let pool_view = pool_registry.verified_view();
        let token_view = token_registry.verified_view();

        let catalog = MetadataCatalog::from_views(pool_view.clone(), token_view.clone());

        // Building the catalog (and every subsequent clone of it) copies persistent-map roots, not
        // the entries — the property that lets `/pools/meta` be served without duplicating the
        // registry. `ptr_eq` holds only when both handles point at the same root node.
        assert!(catalog.pools.ptr_eq(&pool_view));
        assert!(catalog.tokens.ptr_eq(&token_view));

        let cloned = catalog.clone();
        assert!(cloned.pools.ptr_eq(&catalog.pools));
        assert!(cloned.tokens.ptr_eq(&catalog.tokens));
    }

    #[test]
    fn state_metadata_catalog_reflects_the_seeded_registries() {
        let (pool_registry, token_registry) = registries();
        let (state, _) = State::activate_from_seed(
            BlockHash::with_last_byte(100),
            100,
            HashMap::new(),
            pool_registry,
            token_registry,
            Vec::new(),
        );

        let catalog = state.metadata_catalog();

        assert_eq!(catalog.pool_count(), 2);
        assert_eq!(
            catalog.token_metadata(TokenAddress(token(1), CHAIN)),
            Some(&decimals(18))
        );
    }

    #[test]
    fn default_catalog_is_empty() {
        let empty = MetadataCatalog::default();
        assert_eq!(empty.pool_count(), 0);
        assert_eq!(empty.iter_pools().count(), 0);
        assert_eq!(empty.token_metadata(TokenAddress(token(1), CHAIN)), None);
    }
}
