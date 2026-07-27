//! Topological pre-filter that drops pools which cannot lie on a route from the
//! source asset to the output asset.
//!
//! The optimizer routes a fixed-depth flow from the `source_asset` to the
//! `output_asset`; a pool is usable only if it lies on some source→output path. We
//! reduce that to a **2-core** question with a single trick: add a virtual "demand"
//! edge `(source, output)` (representing the flow returning from sink to source),
//! then keep the 2-core of the source's connected component. Any simple
//! source→output path plus that edge is a cycle, so the 2-core keeps exactly the
//! path tokens. When `output == source` no edge is added and this reduces, *by
//! construction*, to the plain cycle filter (the closed-arbitrage case): every token
//! on the route must touch at least two distinct routes and connect back to the source.
//!
//! The filter is a conservative *necessary* condition — it never drops a pool that
//! could carry flow on a source→output route. It is hop-count agnostic w.r.t. the
//! model's layer depth and ignores liquidity/profitability, which remain the
//! optimizer's concern.

use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use crate::execution::OptimizationSessionConfig;
use crate::pool_reserves::PoolReserves;

/// The full pre-optimization admission funnel: the session's token whitelist (a hard constraint,
/// applied first so disallowed tokens also vanish from connectivity), then the topological
/// [`routable_reserves`] prune. All reserve snapshots enter the optimizer through this function.
pub(crate) fn admissible_reserves<TPool, TToken>(
    reserves: Vec<PoolReserves<TPool, TToken>>,
    session_config: &OptimizationSessionConfig<TToken>,
) -> Vec<PoolReserves<TPool, TToken>>
where
    TPool: Copy + Eq + Hash,
    TToken: Copy + Eq + Hash,
{
    let reserves = whitelisted_reserves(reserves, session_config.whitelist.as_ref());
    routable_reserves(
        reserves,
        session_config.source_asset,
        session_config.output_asset,
        &session_config.bridges,
    )
}

/// Keeps only the reserves whose both tokens are whitelisted; `None` disables filtering entirely.
///
/// Unlike [`routable_reserves`] there is no fall-back-to-input behavior: the whitelist is a hard
/// constraint, so a snapshot it empties stays empty (downstream validation reports it as
/// empty/not-ready rather than optimizing over disallowed tokens).
pub(crate) fn whitelisted_reserves<TPool, TToken>(
    reserves: Vec<PoolReserves<TPool, TToken>>,
    whitelist: Option<&HashSet<TToken>>,
) -> Vec<PoolReserves<TPool, TToken>>
where
    TPool: Copy + Eq + Hash,
    TToken: Copy + Eq + Hash,
{
    let Some(allowed) = whitelist else {
        return reserves;
    };

    reserves
        .into_iter()
        .filter(|reserve| allowed.contains(&reserve.token0) && allowed.contains(&reserve.token1))
        .collect()
}

/// Keeps only the reserves whose pools can lie on a route from `source_asset` to
/// `output_asset` — the 2-core of the source's connected component after adding a
/// virtual `(source, output)` demand edge. When `output == source` this is exactly
/// the 2-core cycle filter (closed arbitrage).
///
/// `bridges` are treated as real (zero-cost) routing edges and counted toward
/// connectivity and degree — without them an otherwise-reachable subgraph (e.g. a
/// second chain linked only by a USDC bridge) would be pruned as unreachable.
///
/// If the source asset itself is not routable (fewer than two distinct routes) or the
/// prune would otherwise empty the snapshot, the original `reserves` are returned
/// unchanged so downstream validation keeps its existing behavior.
pub(crate) fn routable_reserves<TPool, TToken>(
    reserves: Vec<PoolReserves<TPool, TToken>>,
    source_asset: TToken,
    output_asset: TToken,
    bridges: &HashSet<(TToken, TToken)>,
) -> Vec<PoolReserves<TPool, TToken>>
where
    TPool: Copy + Eq + Hash,
    TToken: Copy + Eq + Hash,
{
    let keep = routable_tokens(&reserves, source_asset, output_asset, bridges);

    if !keep.contains(&source_asset) {
        return reserves;
    }

    let pruned: Vec<PoolReserves<TPool, TToken>> = reserves
        .iter()
        .copied()
        .filter(|reserve| keep.contains(&reserve.token0) && keep.contains(&reserve.token1))
        .collect();

    if pruned.is_empty() { reserves } else { pruned }
}

/// Computes the set of tokens that survive the 2-core stripping and remain connected
/// to `source_asset`, after adding a virtual `(source, output)` demand edge (only when
/// they differ) so a source→output path is treated as a cycle.
fn routable_tokens<TPool, TToken>(
    reserves: &[PoolReserves<TPool, TToken>],
    source_asset: TToken,
    output_asset: TToken,
    bridges: &HashSet<(TToken, TToken)>,
) -> HashSet<TToken>
where
    TPool: Copy + Eq + Hash,
    TToken: Copy + Eq + Hash,
{
    // Undirected multigraph of tokens. Each unique pool contributes one edge (the two
    // directional reserves of a pool share a `pool_id` and collapse to a single edge);
    // parallel pools have distinct ids and each add an edge, so a 2-cycle across
    // parallel pools survives. Each bridge pair adds one edge (mirrored entries are
    // deduped).
    let mut adjacency: HashMap<TToken, Vec<TToken>> = HashMap::new();

    let mut seen_pools: HashSet<TPool> = HashSet::new();
    for reserve in reserves {
        if reserve.token0 == reserve.token1 {
            continue;
        }
        if seen_pools.insert(reserve.pool_id) {
            adjacency
                .entry(reserve.token0)
                .or_default()
                .push(reserve.token1);
            adjacency
                .entry(reserve.token1)
                .or_default()
                .push(reserve.token0);
        }
    }

    let mut seen_bridges: HashSet<(TToken, TToken)> = HashSet::new();
    for &(a, b) in bridges {
        if a == b || seen_bridges.contains(&(b, a)) || !seen_bridges.insert((a, b)) {
            continue;
        }
        adjacency.entry(a).or_default().push(b);
        adjacency.entry(b).or_default().push(a);
    }

    // Virtual demand edge closing an open source→output route into a cycle. Added only when the
    // assets differ, so the `output == source` case is byte-for-byte the original cycle filter.
    if source_asset != output_asset {
        adjacency
            .entry(source_asset)
            .or_default()
            .push(output_asset);
        adjacency
            .entry(output_asset)
            .or_default()
            .push(source_asset);
    }

    // 2-core: iteratively strip every token with multigraph degree < 2.
    let mut degree: HashMap<TToken, usize> = adjacency
        .iter()
        .map(|(token, neighbors)| (*token, neighbors.len()))
        .collect();
    let mut removed: HashSet<TToken> = HashSet::new();
    let mut stack: Vec<TToken> = degree
        .iter()
        .filter(|(_, d)| **d < 2)
        .map(|(token, _)| *token)
        .collect();

    while let Some(token) = stack.pop() {
        if removed.contains(&token) || degree.get(&token).copied().unwrap_or(0) >= 2 {
            continue;
        }
        removed.insert(token);
        if let Some(neighbors) = adjacency.get(&token) {
            for &neighbor in neighbors {
                if removed.contains(&neighbor) {
                    continue;
                }
                if let Some(d) = degree.get_mut(&neighbor) {
                    *d = d.saturating_sub(1);
                    if *d < 2 {
                        stack.push(neighbor);
                    }
                }
            }
        }
    }

    if removed.contains(&source_asset) || !adjacency.contains_key(&source_asset) {
        return HashSet::new();
    }

    // Restrict the surviving graph to the component containing the source asset.
    let mut keep: HashSet<TToken> = HashSet::new();
    keep.insert(source_asset);
    let mut frontier = vec![source_asset];
    while let Some(token) = frontier.pop() {
        if let Some(neighbors) = adjacency.get(&token) {
            for &neighbor in neighbors {
                if !removed.contains(&neighbor) && keep.insert(neighbor) {
                    frontier.push(neighbor);
                }
            }
        }
    }

    keep
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use proptest::prelude::*;

    use crate::pool_reserves::{PoolReserves, VirtualReserveValues};
    use crate::tokens::test::{self as tokens, TokenAddress};

    use super::*;

    fn value() -> VirtualReserveValues {
        VirtualReserveValues {
            token_0: 1.0,
            token_1: 1.0,
            fee_multiplier: 1.0,
            max_swap_0: 1.0,
            max_swap_1: 1.0,
        }
    }

    /// Builds both directional reserves of a single pool (they share `pool_id`).
    fn pool(
        token0: TokenAddress,
        token1: TokenAddress,
        pool_id: u32,
    ) -> Vec<PoolReserves<u32, TokenAddress>> {
        vec![
            PoolReserves {
                token0,
                token1,
                pool_id,
                value: value(),
            },
            PoolReserves {
                token0: token1,
                token1: token0,
                pool_id,
                value: value(),
            },
        ]
    }

    fn no_bridges() -> HashSet<(TokenAddress, TokenAddress)> {
        HashSet::new()
    }

    fn has_pool(reserves: &[PoolReserves<u32, TokenAddress>], pool_id: u32) -> bool {
        reserves.iter().any(|reserve| reserve.pool_id == pool_id)
    }

    #[test]
    fn drops_degree_one_leaf_pool() {
        let usdc = tokens::USDC.address;
        let weth = tokens::WETH.address;
        let wbtc = tokens::WBTC.address;
        let uni = tokens::UNI.address;

        // Triangle (all degree 2) plus a leaf pool hanging off USDC.
        let mut reserves = Vec::new();
        reserves.extend(pool(usdc, weth, 1));
        reserves.extend(pool(weth, wbtc, 2));
        reserves.extend(pool(wbtc, usdc, 3));
        reserves.extend(pool(usdc, uni, 4)); // uni is a leaf

        let pruned = routable_reserves(reserves, usdc, usdc, &no_bridges());

        assert!(has_pool(&pruned, 1));
        assert!(has_pool(&pruned, 2));
        assert!(has_pool(&pruned, 3));
        assert!(!has_pool(&pruned, 4), "leaf pool must be dropped");
    }

    #[test]
    fn keeps_parallel_pools_forming_a_two_cycle() {
        let usdc = tokens::USDC.address;
        let weth = tokens::WETH.address;

        // Two parallel pools between the same pair: a valid 2-cycle. A naive
        // distinct-neighbor degree would wrongly prune these.
        let mut reserves = Vec::new();
        reserves.extend(pool(usdc, weth, 1));
        reserves.extend(pool(usdc, weth, 2));

        let pruned = routable_reserves(reserves, usdc, usdc, &no_bridges());

        assert!(has_pool(&pruned, 1));
        assert!(has_pool(&pruned, 2));
    }

    #[test]
    fn keeps_remote_component_reachable_via_bridge() {
        let usdc = tokens::USDC.address;
        let weth = tokens::WETH.address;
        let wbtc = tokens::WBTC.address;
        // Stand-ins for a second chain's tokens.
        let arb_usdc = tokens::USDT.address;
        let arb_a = tokens::LINK.address;

        let mut reserves = Vec::new();
        // Ethereum triangle.
        reserves.extend(pool(usdc, weth, 1));
        reserves.extend(pool(weth, wbtc, 2));
        reserves.extend(pool(wbtc, usdc, 3));
        // Arbitrum side: parallel pools form their own 2-core.
        reserves.extend(pool(arb_usdc, arb_a, 4));
        reserves.extend(pool(arb_usdc, arb_a, 5));

        let bridges = HashSet::from([(usdc, arb_usdc), (arb_usdc, usdc)]);

        let with_bridge = routable_reserves(reserves.clone(), usdc, usdc, &bridges);
        assert!(has_pool(&with_bridge, 4), "bridged component must be kept");
        assert!(has_pool(&with_bridge, 5));

        // Without the bridge the Arbitrum component is unreachable from init.
        let without_bridge = routable_reserves(reserves, usdc, usdc, &no_bridges());
        assert!(
            !has_pool(&without_bridge, 4),
            "unbridged component must be dropped"
        );
        assert!(!has_pool(&without_bridge, 5));
    }

    #[test]
    fn drops_chain_hanging_off_a_bridge_leaf() {
        let usdc = tokens::USDC.address;
        let weth = tokens::WETH.address;
        let wbtc = tokens::WBTC.address;
        let x = tokens::AAVE.address;
        let y = tokens::SOL.address;

        let mut reserves = Vec::new();
        reserves.extend(pool(usdc, weth, 1));
        reserves.extend(pool(weth, wbtc, 2));
        reserves.extend(pool(wbtc, usdc, 3));
        // x is reached only by the bridge and a single dead-end pool to y.
        reserves.extend(pool(x, y, 4));

        let bridges = HashSet::from([(usdc, x), (x, usdc)]);
        let pruned = routable_reserves(reserves, usdc, usdc, &bridges);

        assert!(has_pool(&pruned, 1));
        assert!(
            !has_pool(&pruned, 4),
            "dead-end chain behind a bridge must be dropped"
        );
    }

    #[test]
    fn drops_disconnected_component() {
        let usdc = tokens::USDC.address;
        let weth = tokens::WETH.address;
        let wbtc = tokens::WBTC.address;
        let a = tokens::LINK.address;
        let b = tokens::AAVE.address;
        let c = tokens::SOL.address;

        let mut reserves = Vec::new();
        // Init triangle.
        reserves.extend(pool(usdc, weth, 1));
        reserves.extend(pool(weth, wbtc, 2));
        reserves.extend(pool(wbtc, usdc, 3));
        // Separate triangle, no link to init.
        reserves.extend(pool(a, b, 4));
        reserves.extend(pool(b, c, 5));
        reserves.extend(pool(c, a, 6));

        let pruned = routable_reserves(reserves, usdc, usdc, &no_bridges());

        assert!(has_pool(&pruned, 1));
        assert!(!has_pool(&pruned, 4));
        assert!(!has_pool(&pruned, 5));
        assert!(!has_pool(&pruned, 6));
    }

    #[test]
    fn returns_original_when_init_asset_is_isolated() {
        let usdc = tokens::USDC.address;
        let weth = tokens::WETH.address;

        // Single pool: both tokens are degree 1, the whole graph collapses.
        let reserves = pool(usdc, weth, 1);
        let pruned = routable_reserves(reserves.clone(), usdc, usdc, &no_bridges());

        assert_eq!(
            pruned, reserves,
            "fallback must leave the snapshot untouched"
        );
    }

    #[test]
    fn keeps_both_directions_of_a_kept_pool() {
        let usdc = tokens::USDC.address;
        let weth = tokens::WETH.address;
        let wbtc = tokens::WBTC.address;

        let mut reserves = Vec::new();
        reserves.extend(pool(usdc, weth, 1));
        reserves.extend(pool(weth, wbtc, 2));
        reserves.extend(pool(wbtc, usdc, 3));

        let pruned = routable_reserves(reserves.clone(), usdc, usdc, &no_bridges());

        assert_eq!(pruned.len(), reserves.len());
        let directions = pruned.iter().filter(|r| r.pool_id == 1).count();
        assert_eq!(directions, 2, "both directional reserves must be retained");
    }

    fn config_with_whitelist(
        source_asset: TokenAddress,
        whitelist: Option<HashSet<TokenAddress>>,
    ) -> OptimizationSessionConfig<TokenAddress> {
        OptimizationSessionConfig {
            source_asset,
            output_asset: source_asset,
            bridges: no_bridges(),
            whitelist,
        }
    }

    #[test]
    fn no_whitelist_keeps_every_reserve() {
        let usdc = tokens::USDC.address;
        let weth = tokens::WETH.address;

        let reserves = pool(usdc, weth, 1);
        let filtered = whitelisted_reserves(reserves.clone(), None);

        assert_eq!(filtered, reserves);
    }

    #[test]
    fn drops_pool_with_one_non_whitelisted_token() {
        let usdc = tokens::USDC.address;
        let weth = tokens::WETH.address;
        let wbtc = tokens::WBTC.address;

        let mut reserves = Vec::new();
        reserves.extend(pool(usdc, weth, 1));
        reserves.extend(pool(usdc, wbtc, 2)); // wbtc not whitelisted

        let whitelist = HashSet::from([usdc, weth]);
        let filtered = whitelisted_reserves(reserves, Some(&whitelist));

        assert!(has_pool(&filtered, 1));
        assert!(!has_pool(&filtered, 2));
    }

    #[test]
    fn keeps_both_directions_of_a_whitelisted_pool() {
        let usdc = tokens::USDC.address;
        let weth = tokens::WETH.address;

        let whitelist = HashSet::from([usdc, weth]);
        let filtered = whitelisted_reserves(pool(usdc, weth, 1), Some(&whitelist));

        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn empty_whitelist_empties_the_snapshot_without_fallback() {
        let usdc = tokens::USDC.address;
        let weth = tokens::WETH.address;

        let whitelist = HashSet::new();
        let filtered = whitelisted_reserves(pool(usdc, weth, 1), Some(&whitelist));

        assert!(
            filtered.is_empty(),
            "a whitelist is a hard constraint — no fallback to the input"
        );
    }

    #[test]
    fn admissible_reserves_applies_whitelist_before_routing() {
        let usdc = tokens::USDC.address;
        let weth = tokens::WETH.address;
        let wbtc = tokens::WBTC.address;
        let link = tokens::LINK.address;

        // Two triangles sharing the init asset. Removing wbtc from the whitelist breaks the
        // first triangle, and the routing filter must then prune its surviving weth leg too.
        let mut reserves = Vec::new();
        reserves.extend(pool(usdc, weth, 1));
        reserves.extend(pool(weth, wbtc, 2));
        reserves.extend(pool(wbtc, usdc, 3));
        reserves.extend(pool(usdc, link, 4));
        reserves.extend(pool(link, usdc, 5));

        let whitelist = HashSet::from([usdc, weth, link]);
        let config = config_with_whitelist(usdc, Some(whitelist));
        let admitted = admissible_reserves(reserves, &config);

        assert!(!has_pool(&admitted, 2), "whitelist drops the wbtc pools");
        assert!(!has_pool(&admitted, 3));
        assert!(
            !has_pool(&admitted, 1),
            "routing must prune the weth leg orphaned by the whitelist"
        );
        assert!(has_pool(&admitted, 4));
        assert!(has_pool(&admitted, 5));
    }

    // A small fixed token palette for property tests.
    fn palette() -> [TokenAddress; 6] {
        [
            tokens::USDC.address,
            tokens::WETH.address,
            tokens::WBTC.address,
            tokens::LINK.address,
            tokens::AAVE.address,
            tokens::SOL.address,
        ]
    }

    prop_compose! {
        fn arbitrary_reserves()(
            edges in prop::collection::vec((0usize..6, 0usize..6), 0..24)
        ) -> Vec<PoolReserves<u32, TokenAddress>> {
            let palette = palette();
            let mut reserves = Vec::new();
            for (pool_id, (i, j)) in edges.into_iter().enumerate() {
                reserves.extend(pool(palette[i], palette[j], pool_id as u32));
            }
            reserves
        }
    }

    prop_compose! {
        fn arbitrary_whitelist()(
            included in prop::collection::vec(any::<bool>(), 6)
        ) -> HashSet<TokenAddress> {
            palette()
                .into_iter()
                .zip(included)
                .filter_map(|(token, keep)| keep.then_some(token))
                .collect()
        }
    }

    proptest! {
        #[test]
        fn output_is_a_subset_of_input(reserves in arbitrary_reserves()) {
            let init = palette()[0];
            let input_keys: HashSet<(u32, TokenAddress, TokenAddress)> = reserves
                .iter()
                .map(|r| (r.pool_id, r.token0, r.token1))
                .collect();

            let pruned = routable_reserves(reserves, init, init, &no_bridges());

            for reserve in &pruned {
                prop_assert!(input_keys.contains(&(reserve.pool_id, reserve.token0, reserve.token1)));
            }
        }

        #[test]
        fn pruning_is_idempotent(reserves in arbitrary_reserves()) {
            let init = palette()[0];
            let once = routable_reserves(reserves, init, init, &no_bridges());
            let twice = routable_reserves(once.clone(), init, init, &no_bridges());
            prop_assert_eq!(once, twice);
        }

        #[test]
        fn pruning_never_grows_the_snapshot(reserves in arbitrary_reserves()) {
            let init = palette()[0];
            let len_before = reserves.len();
            let pruned = routable_reserves(reserves, init, init, &no_bridges());
            prop_assert!(pruned.len() <= len_before);
        }

        #[test]
        fn whitelisting_keeps_only_fully_whitelisted_entries(
            reserves in arbitrary_reserves(),
            whitelist in arbitrary_whitelist(),
        ) {
            let input_keys: HashSet<(u32, TokenAddress, TokenAddress)> = reserves
                .iter()
                .map(|r| (r.pool_id, r.token0, r.token1))
                .collect();

            let filtered = whitelisted_reserves(reserves, Some(&whitelist));

            for reserve in &filtered {
                prop_assert!(input_keys.contains(&(reserve.pool_id, reserve.token0, reserve.token1)));
                prop_assert!(whitelist.contains(&reserve.token0));
                prop_assert!(whitelist.contains(&reserve.token1));
            }
        }

        #[test]
        fn whitelisting_is_idempotent(
            reserves in arbitrary_reserves(),
            whitelist in arbitrary_whitelist(),
        ) {
            let once = whitelisted_reserves(reserves, Some(&whitelist));
            let twice = whitelisted_reserves(once.clone(), Some(&whitelist));
            prop_assert_eq!(once, twice);
        }

        #[test]
        fn whitelisting_never_grows_the_snapshot(
            reserves in arbitrary_reserves(),
            whitelist in arbitrary_whitelist(),
        ) {
            let len_before = reserves.len();
            let filtered = whitelisted_reserves(reserves, Some(&whitelist));
            prop_assert!(filtered.len() <= len_before);
        }

        #[test]
        fn admissible_without_whitelist_matches_routable(reserves in arbitrary_reserves()) {
            let init = palette()[0];
            let config = config_with_whitelist(init, None);
            let admitted = admissible_reserves(reserves.clone(), &config);
            let routed = routable_reserves(reserves, init, init, &no_bridges());
            prop_assert_eq!(admitted, routed);
        }
    }

    // --- Increment 3: source→output path prune (open-ended output asset) ---

    /// A frozen, verbatim copy of the *pre-Increment-3* cycle filter (the 2-core of the source's
    /// connected component). It is the permanent differential oracle: `routable_reserves` with
    /// `output == source` must equal this forever, guaranteeing the degenerate case never drifts
    /// from the arbitrage behavior it replaced.
    fn cycle_core_reference(
        reserves: Vec<PoolReserves<u32, TokenAddress>>,
        init_asset: TokenAddress,
        bridges: &HashSet<(TokenAddress, TokenAddress)>,
    ) -> Vec<PoolReserves<u32, TokenAddress>> {
        let keep = cycle_core_tokens_reference(&reserves, init_asset, bridges);

        if !keep.contains(&init_asset) {
            return reserves;
        }

        let pruned: Vec<PoolReserves<u32, TokenAddress>> = reserves
            .iter()
            .copied()
            .filter(|reserve| keep.contains(&reserve.token0) && keep.contains(&reserve.token1))
            .collect();

        if pruned.is_empty() { reserves } else { pruned }
    }

    fn cycle_core_tokens_reference(
        reserves: &[PoolReserves<u32, TokenAddress>],
        init_asset: TokenAddress,
        bridges: &HashSet<(TokenAddress, TokenAddress)>,
    ) -> HashSet<TokenAddress> {
        let mut adjacency: HashMap<TokenAddress, Vec<TokenAddress>> = HashMap::new();

        let mut seen_pools: HashSet<u32> = HashSet::new();
        for reserve in reserves {
            if reserve.token0 == reserve.token1 {
                continue;
            }
            if seen_pools.insert(reserve.pool_id) {
                adjacency
                    .entry(reserve.token0)
                    .or_default()
                    .push(reserve.token1);
                adjacency
                    .entry(reserve.token1)
                    .or_default()
                    .push(reserve.token0);
            }
        }

        let mut seen_bridges: HashSet<(TokenAddress, TokenAddress)> = HashSet::new();
        for &(a, b) in bridges {
            if a == b || seen_bridges.contains(&(b, a)) || !seen_bridges.insert((a, b)) {
                continue;
            }
            adjacency.entry(a).or_default().push(b);
            adjacency.entry(b).or_default().push(a);
        }

        let mut degree: HashMap<TokenAddress, usize> = adjacency
            .iter()
            .map(|(token, neighbors)| (*token, neighbors.len()))
            .collect();
        let mut removed: HashSet<TokenAddress> = HashSet::new();
        let mut stack: Vec<TokenAddress> = degree
            .iter()
            .filter(|(_, d)| **d < 2)
            .map(|(token, _)| *token)
            .collect();

        while let Some(token) = stack.pop() {
            if removed.contains(&token) || degree.get(&token).copied().unwrap_or(0) >= 2 {
                continue;
            }
            removed.insert(token);
            if let Some(neighbors) = adjacency.get(&token) {
                for &neighbor in neighbors {
                    if removed.contains(&neighbor) {
                        continue;
                    }
                    if let Some(d) = degree.get_mut(&neighbor) {
                        *d = d.saturating_sub(1);
                        if *d < 2 {
                            stack.push(neighbor);
                        }
                    }
                }
            }
        }

        if removed.contains(&init_asset) || !adjacency.contains_key(&init_asset) {
            return HashSet::new();
        }

        let mut keep: HashSet<TokenAddress> = HashSet::new();
        keep.insert(init_asset);
        let mut frontier = vec![init_asset];
        while let Some(token) = frontier.pop() {
            if let Some(neighbors) = adjacency.get(&token) {
                for &neighbor in neighbors {
                    if !removed.contains(&neighbor) && keep.insert(neighbor) {
                        frontier.push(neighbor);
                    }
                }
            }
        }

        keep
    }

    prop_compose! {
        fn arbitrary_bridges()(
            pairs in prop::collection::vec((0usize..6, 0usize..6), 0..6)
        ) -> HashSet<(TokenAddress, TokenAddress)> {
            let palette = palette();
            pairs
                .into_iter()
                .map(|(i, j)| (palette[i], palette[j]))
                .collect()
        }
    }

    #[test]
    fn open_acyclic_chain_is_kept_and_dangling_leaf_dropped() {
        // A simple source→output chain with NO cycle: usdc—weth—wbtc. Every vertex is degree ≤ 2 and
        // the two endpoints are degree 1, so the pre-Increment-3 cycle filter strips the whole graph
        // and (finding the init asset gone) falls back to returning the input untouched — it cannot
        // express an acyclic best-execution route. With `output = wbtc` the virtual demand edge
        // (usdc, wbtc) closes the chain into a cycle, so all three chain pools are kept, while a leaf
        // (wbtc—uni) hanging off the output is still dropped.
        let usdc = tokens::USDC.address;
        let weth = tokens::WETH.address;
        let wbtc = tokens::WBTC.address;
        let uni = tokens::UNI.address;

        let mut reserves = Vec::new();
        reserves.extend(pool(usdc, weth, 1));
        reserves.extend(pool(weth, wbtc, 2));
        reserves.extend(pool(wbtc, uni, 3)); // dead-end leaf off the output

        let pruned = routable_reserves(reserves, usdc, wbtc, &no_bridges());

        assert!(
            has_pool(&pruned, 1),
            "source leg of the s→o chain must be kept"
        );
        assert!(
            has_pool(&pruned, 2),
            "middle leg of the s→o chain must be kept"
        );
        assert!(
            !has_pool(&pruned, 3),
            "leaf hanging off the output must be dropped"
        );
    }

    #[test]
    fn open_path_reduces_to_cycle_when_output_equals_source() {
        // Spot check of the degenerate reduction on a concrete graph (the proptest below is the
        // exhaustive guard): passing the source as the output reproduces the leaf-drop behavior.
        let usdc = tokens::USDC.address;
        let weth = tokens::WETH.address;
        let wbtc = tokens::WBTC.address;
        let uni = tokens::UNI.address;

        let mut reserves = Vec::new();
        reserves.extend(pool(usdc, weth, 1));
        reserves.extend(pool(weth, wbtc, 2));
        reserves.extend(pool(wbtc, usdc, 3));
        reserves.extend(pool(usdc, uni, 4)); // leaf

        let pruned = routable_reserves(reserves.clone(), usdc, usdc, &no_bridges());
        let reference = cycle_core_reference(reserves, usdc, &no_bridges());
        assert_eq!(pruned, reference);
    }

    proptest! {
        #[test]
        fn output_source_equal_matches_frozen_cycle_reference(
            reserves in arbitrary_reserves(),
            bridges in arbitrary_bridges(),
        ) {
            // The permanent differential oracle: with `output == source`, the new path prune is
            // bit-identical to the frozen pre-Increment-3 cycle filter.
            let source = palette()[0];
            let via_path = routable_reserves(reserves.clone(), source, source, &bridges);
            let via_cycle = cycle_core_reference(reserves, source, &bridges);
            prop_assert_eq!(via_path, via_cycle);
        }

        #[test]
        fn open_path_output_is_a_subset_of_input(reserves in arbitrary_reserves()) {
            let source = palette()[0];
            let output = palette()[1];
            let input_keys: HashSet<(u32, TokenAddress, TokenAddress)> = reserves
                .iter()
                .map(|r| (r.pool_id, r.token0, r.token1))
                .collect();

            let pruned = routable_reserves(reserves, source, output, &no_bridges());

            for reserve in &pruned {
                prop_assert!(input_keys.contains(&(reserve.pool_id, reserve.token0, reserve.token1)));
            }
        }

        #[test]
        fn open_path_pruning_is_idempotent(reserves in arbitrary_reserves()) {
            let source = palette()[0];
            let output = palette()[1];
            let once = routable_reserves(reserves, source, output, &no_bridges());
            let twice = routable_reserves(once.clone(), source, output, &no_bridges());
            prop_assert_eq!(once, twice);
        }
    }
}
