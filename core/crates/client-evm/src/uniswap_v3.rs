//! Uniswap v3 integration helpers.

use alloy::{
    primitives::{Address, B256, address},
    sol,
    sol_types::SolEvent,
};

pub const ETHEREUM_UNISWAP_V3_FACTORY_ADDRESS: Address =
    address!("1F98431c8aD98523631AE4a59f267346ea31F984");

sol! {
    function token0() external view returns (address);

    function token1() external view returns (address);

    function fee() external view returns (uint32);

    function getPool(address tokenA, address tokenB, uint24 fee) external view returns (address pool);

    function slot0() external view returns (
        uint160 sqrtPriceX96,
        int24 tick,
        uint16 observationIndex,
        uint16 observationCardinality,
        uint16 observationCardinalityNext,
        uint8 feeProtocol,
        bool unlocked
    );

    function liquidity() external view returns (uint128);

    #[derive(Debug, PartialEq, Eq)]
    event Initialize(uint160 sqrtPriceX96, int24 tick);

    #[derive(Debug, PartialEq, Eq)]
    event Mint(
        address sender,
        address indexed owner,
        int24 indexed tickLower,
        int24 indexed tickUpper,
        uint128 amount,
        uint256 amount0,
        uint256 amount1
    );

    #[derive(Debug, PartialEq, Eq)]
    event Collect(
        address indexed owner,
        address recipient,
        int24 indexed tickLower,
        int24 indexed tickUpper,
        uint128 amount0,
        uint128 amount1
    );

    #[derive(Debug, PartialEq, Eq)]
    event Burn(
        address indexed owner,
        int24 indexed tickLower,
        int24 indexed tickUpper,
        uint128 amount,
        uint256 amount0,
        uint256 amount1
    );

    #[derive(Debug, PartialEq, Eq)]
    event Swap(
        address indexed sender,
        address indexed recipient,
        int256 amount0,
        int256 amount1,
        uint160 sqrtPriceX96,
        uint128 liquidity,
        int24 tick
    );

    #[derive(Debug, PartialEq, Eq)]
    event Flash(
        address indexed sender,
        address indexed recipient,
        uint256 amount0,
        uint256 amount1,
        uint256 paid0,
        uint256 paid1
    );

    #[derive(Debug, PartialEq, Eq)]
    event IncreaseObservationCardinalityNext(
        uint16 observationCardinalityNextOld,
        uint16 observationCardinalityNextNew
    );

    #[derive(Debug, PartialEq, Eq)]
    event SetFeeProtocol(
        uint8 feeProtocol0Old,
        uint8 feeProtocol1Old,
        uint8 feeProtocol0New,
        uint8 feeProtocol1New
    );

    #[derive(Debug, PartialEq, Eq)]
    event CollectProtocol(
        address indexed sender,
        address indexed recipient,
        uint128 amount0,
        uint128 amount1
    );
}

pub fn pool_event_signature_hashes() -> [B256; 9] {
    [
        Initialize::SIGNATURE_HASH,
        Mint::SIGNATURE_HASH,
        Collect::SIGNATURE_HASH,
        Burn::SIGNATURE_HASH,
        Swap::SIGNATURE_HASH,
        Flash::SIGNATURE_HASH,
        IncreaseObservationCardinalityNext::SIGNATURE_HASH,
        SetFeeProtocol::SIGNATURE_HASH,
        CollectProtocol::SIGNATURE_HASH,
    ]
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn pool_event_signature_hashes_contains_all_pool_events() {
        assert_eq!(pool_event_signature_hashes().len(), 9);
    }

    #[test]
    fn pool_event_signature_hashes_are_unique() {
        let hashes = pool_event_signature_hashes();
        let unique_hashes = hashes.into_iter().collect::<HashSet<_>>();

        assert_eq!(unique_hashes.len(), hashes.len());
    }
}
