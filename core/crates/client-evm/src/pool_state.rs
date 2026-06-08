use alloy::primitives::{Address, U160};

#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy)]
pub struct PoolAddress(pub Address);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PoolState {
    pub sqrt_price_x96: U160,
    pub tick: i32,
    pub liquidity: u128,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoolDataCall {
    Slot0,
    Liquidity,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PoolDataFailure {
    CallFailed(PoolDataCall),
    DecodeFailed(PoolDataCall),
    MissingResponse(PoolDataCall),
}

pub type PoolDataResult = Result<PoolState, PoolDataFailure>;
