use alloy::primitives::{
    Address, U160, U256, U512,
    aliases::{I24, U24},
};
use thiserror::Error;

use crate::tick_math::{self, TickMathError};

#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy)]
pub struct PoolAddress(pub Address);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PoolState {
    pub sqrt_price_x96: U160,
    pub tick: I24,
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

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum PoolStateError {
    #[error(transparent)]
    TickMath(#[from] TickMathError),

    #[error("sqrt_price_at_tick failed: {source}\n{pool_state:?}")]
    SqrtPriceAtTickFailed {
        source: TickMathError,
        pool_state: PoolState,
    },

    #[error(
        "Failed to calculate swap limit x (subtraction caused overflow): reserve_min: {reserve_min}\n reserve_current: {reserve_current}\n tick: {tick}\n tick_spacing: {tick_spacing}\n sqrt_price_current: {sqrt_price_current}\n sqrt_price_min: {sqrt_price_min}"
    )]
    SwapLimitXSubtractionOverflow {
        reserve_min: U256,
        reserve_current: U256,
        tick: I24,
        tick_spacing: u16,
        sqrt_price_current: U160,
        sqrt_price_min: U160,
    },

    #[error(
        "Failed to calculate swap limit y (subtraction caused overflow): reserve_max: {reserve_max}\n reserve_current: {reserve_current}\n tick: {tick}\n tick_spacing: {tick_spacing}\n sqrt_price_current: {sqrt_price_current}\n sqrt_price_max_x96: {sqrt_price_max_x96}"
    )]
    SwapLimitYSubtractionOverflow {
        reserve_max: U256,
        reserve_current: U256,
        tick: I24,
        tick_spacing: u16,
        sqrt_price_current: U160,
        sqrt_price_max_x96: U160,
    },
}

fn virtual_reserve_x(sqrt_price_x96: U160, liquidity: u128) -> U256 {
    // For reserve_0: we need L * 2^96 / sqrtP (to account for the Q64.96 format)
    // This produces a number in Q0.0 format (regular integer)
    let liquidity_x96: U256 = U256::from(liquidity) << 96;

    // logically, zero sqrt_price_x96 is an undefined behavior and supposed to be represented as error
    // but in practice, the same case in virtual_reserve_y would return zero because it does not involve division by sqrt_price_x96
    // also, sqrt_price_x96 = 0 is no different from liquidity = 0 from the practical perspective
    // both mean empty reserves and the calling code should be able to handle empty reserves even if sqrt_price_x96=0 is represented as error
    // so zero is returned in case of sqrt_price_x96 = 0 to be consistent with virtual_reserve_y
    let reserve_x96 = liquidity_x96
        .checked_div(U256::from(sqrt_price_x96))
        .unwrap_or(U256::ZERO);
    reserve_x96
}

fn virtual_reserve_y(sqrt_price_x96: U160, liquidity: u128) -> U256 {
    let q_96 = U512::from(1u128 << 96);
    // For reserve_1: we need L * sqrtP / 2^96
    // This also produces a number in Q0.0 format
    let liquidity_x96: U512 = U512::from(liquidity);
    let reserve_x96 = U256::from(liquidity_x96 * U512::from(sqrt_price_x96) / q_96);
    reserve_x96
}

impl PoolState {
    pub fn virtual_reserve_x(&self) -> U256 {
        virtual_reserve_x(self.sqrt_price_x96, self.liquidity)
    }

    pub fn virtual_reserve_y(&self) -> U256 {
        virtual_reserve_y(self.sqrt_price_x96, self.liquidity)
    }

    pub fn swap_limit_x(&self, tick_spacing: u16) -> Result<U256, PoolStateError> {
        let spacing = U24::from(tick_spacing);
        let tick_low = tick_math::tick_low(self.tick, spacing)?;
        let sqrt_price_min_x96 = tick_math::sqrt_price_at_tick(tick_low).map_err(|source| {
            PoolStateError::SqrtPriceAtTickFailed {
                source,
                pool_state: self.clone(),
            }
        })?;

        let reserve_current = virtual_reserve_x(self.sqrt_price_x96, self.liquidity);
        let reserve_min = virtual_reserve_x(sqrt_price_min_x96, self.liquidity);
        // let liquidity_x96: U256 = U256::from(self.liquidity) << 96;
        // let reserve_current = liquidity_x96 / U256::from(self.sqrt_price_x96);
        // let reserve_min = liquidity_x96 / U256::from(sqrt_price_min_x96);

        reserve_min.checked_sub(reserve_current).ok_or(
            PoolStateError::SwapLimitXSubtractionOverflow {
                reserve_min,
                reserve_current,
                tick: self.tick,
                tick_spacing,
                sqrt_price_current: self.sqrt_price_x96,
                sqrt_price_min: sqrt_price_min_x96,
            },
        )
    }

    pub fn swap_limit_y(&self, tick_spacing: u16) -> Result<U256, PoolStateError> {
        let spacing = U24::from(tick_spacing);
        let tick_high = tick_math::tick_high(self.tick, spacing)?;
        let sqrt_price_max_x96 = tick_math::sqrt_price_at_tick(tick_high).map_err(|source| {
            PoolStateError::SqrtPriceAtTickFailed {
                source,
                pool_state: self.clone(),
            }
        })?;

        let reserve_current = virtual_reserve_y(self.sqrt_price_x96, self.liquidity);
        let reserve_max = virtual_reserve_y(sqrt_price_max_x96, self.liquidity);

        reserve_max.checked_sub(reserve_current).ok_or(
            PoolStateError::SwapLimitYSubtractionOverflow {
                reserve_max,
                reserve_current,
                tick: self.tick,
                tick_spacing,
                sqrt_price_current: self.sqrt_price_x96,
                sqrt_price_max_x96,
            },
        )
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    const POOL_STATE_WBTC_USDC: PoolState = PoolState {
        sqrt_price_x96: U160::from_limbs([17134602959287796597, 139272449984, 0]),
        liquidity: 50170120777514,
        tick: I24::from_limbs([69583]),
    };

    #[test]
    fn test_max_swap_x() {
        let pool_state = POOL_STATE_WBTC_USDC;

        let max_swap_x = pool_state.swap_limit_x(60).unwrap();
        let virtual_reserve_x = pool_state.virtual_reserve_x(); //.unwrap();

        // println!("Max Swap X: {:?}", max_swap_x);
        // println!("Virtual Reserve X: {:?}", virtual_reserve_x);

        assert!(
            max_swap_x < virtual_reserve_x,
            "Max swap X exceeds virtual reserve X"
        );
        assert!(max_swap_x > 0, "Max swap X should be greater than zero");
    }

    #[test]
    fn test_max_swap_x_1() {
        let pool_state = PoolState {
            sqrt_price_x96: U160::from_limbs([5277553418330626170, 83406331270155, 0]),
            liquidity: 4844714101140627498,
            tick: I24::from_limbs([197490]),
        };

        let max_swap_x = pool_state.swap_limit_x(60);

        assert!(
            max_swap_x.is_ok(),
            "Failed to calculate max swap X: {:?}",
            max_swap_x.err()
        );
    }

    #[test]
    fn test_max_swap_y() {
        let pool_state = POOL_STATE_WBTC_USDC;

        let max_swap_y = pool_state.swap_limit_y(60).unwrap();
        let virtual_reserve_y = pool_state.virtual_reserve_y(); // .unwrap();

        println!("Max Swap Y: {:?}", max_swap_y);
        println!("Virtual Reserve Y: {:?}", virtual_reserve_y);

        assert!(
            max_swap_y < virtual_reserve_y,
            "Max swap Y exceeds virtual reserve Y"
        );
    }

    #[test]
    fn swap_limit_x_returns_typed_error_for_inconsistent_price() {
        let pool_state = PoolState {
            sqrt_price_x96: U160::from(79228162514264337593543950336_u128),
            liquidity: 1_000_000,
            tick: I24::from_limbs([60]),
        };

        let error = pool_state.swap_limit_x(60).unwrap_err();

        assert!(matches!(
            error,
            PoolStateError::SwapLimitXSubtractionOverflow { .. }
        ));
        assert!(
            error
                .to_string()
                .starts_with("Failed to calculate swap limit x (subtraction caused overflow)")
        );
    }
}
