use alloy::primitives::{
    U128, U160, U256,
    aliases::{I24, U24, U136},
    fixed_bytes,
};
use thiserror::Error;

pub const MAX_TICK: I24 = I24::from_limbs([887272]);

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum TickMathError {
    #[error("Tick out of bounds: {abs_tick}")]
    SqrtPriceTickOutOfBounds { abs_tick: U256 },

    #[error("Tick out of bounds: {tick}")]
    TickOutOfBounds { tick: I24 },

    #[error("Tick spacing must be positive: {tick_spacing}")]
    NonPositiveTickSpacing { tick_spacing: U24 },

    #[error(
        "checked_mul Failed to calculate tick low for tick: {tick}, tick_spacing: {tick_spacing}"
    )]
    TickLowMultiplicationOverflow { tick: I24, tick_spacing: I24 },

    #[error(
        "checked_add Failed to calculate tick high for tick: {tick}, tick_spacing: {tick_spacing}"
    )]
    TickHighAdditionOverflow { tick: I24, tick_spacing: U24 },
}

const VAL_0: U136 = U136::from_be_bytes(fixed_bytes!("0x0100000000000000000000000000000000").0);
const VAL_1: U128 = U128::from_be_bytes(fixed_bytes!("0xfffcb933bd6fad37aa2d162d1a594001").0);
const VAL_2: U128 = U128::from_be_bytes(fixed_bytes!("0xfff97272373d413259a46990580e213a").0);
const VAL_4: U128 = U128::from_be_bytes(fixed_bytes!("0xfff2e50f5f656932ef12357cf3c7fdcc").0);
const VAL_8: U128 = U128::from_be_bytes(fixed_bytes!("0xffe5caca7e10e4e61c3624eaa0941cd0").0);
const VAL_10: U128 = U128::from_be_bytes(fixed_bytes!("0xffcb9843d60f6159c9db58835c926644").0);
const VAL_20: U128 = U128::from_be_bytes(fixed_bytes!("0xff973b41fa98c081472e6896dfb254c0").0);
const VAL_40: U128 = U128::from_be_bytes(fixed_bytes!("0xff2ea16466c96a3843ec78b326b52861").0);
const VAL_80: U128 = U128::from_be_bytes(fixed_bytes!("0xfe5dee046a99a2a811c461f1969c3053").0);
const VAL_100: U128 = U128::from_be_bytes(fixed_bytes!("0xfcbe86c7900a88aedcffc83b479aa3a4").0);
const VAL_200: U128 = U128::from_be_bytes(fixed_bytes!("0xf987a7253ac413176f2b074cf7815e54").0);
const VAL_400: U128 = U128::from_be_bytes(fixed_bytes!("0xf3392b0822b70005940c7a398e4b70f3").0);
const VAL_800: U128 = U128::from_be_bytes(fixed_bytes!("0xe7159475a2c29b7443b29c7fa6e889d9").0);
const VAL_1000: U128 = U128::from_be_bytes(fixed_bytes!("0xd097f3bdfd2022b8845ad8f792aa5825").0);
const VAL_2000: U128 = U128::from_be_bytes(fixed_bytes!("0xa9f746462d870fdf8a65dc1f90e061e5").0);
const VAL_4000: U128 = U128::from_be_bytes(fixed_bytes!("0x70d869a156d2a1b890bb3df62baf32f7").0);
const VAL_8000: U128 = U128::from_be_bytes(fixed_bytes!("0x31be135f97d08fd981231505542fcfa6").0);
const VAL_10000: U128 = U128::from_be_bytes(fixed_bytes!("0x09aa508b5b7a84e1c677de54f3e99bc9").0);
const VAL_20000: U128 = U128::from_be_bytes(fixed_bytes!("0x005d6af8dedb81196699c329225ee604").0);
const VAL_40000: U128 = U128::from_be_bytes(fixed_bytes!("0x00002216e584f5fa1ea926041bedfe98").0);
const VAL_80000: U128 = U128::from_be_bytes(fixed_bytes!("0x00000000048a170391f7dc42444e8fa2").0);

const FACTOR_1: U24 = U24::from_be_bytes(fixed_bytes!("0x000001").0);
const FACTOR_2: U24 = U24::from_be_bytes(fixed_bytes!("0x000002").0);
const FACTOR_4: U24 = U24::from_be_bytes(fixed_bytes!("0x000004").0);
const FACTOR_8: U24 = U24::from_be_bytes(fixed_bytes!("0x000008").0);
const FACTOR_10: U24 = U24::from_be_bytes(fixed_bytes!("0x000010").0);
const FACTOR_20: U24 = U24::from_be_bytes(fixed_bytes!("0x000020").0);
const FACTOR_40: U24 = U24::from_be_bytes(fixed_bytes!("0x000040").0);
const FACTOR_80: U24 = U24::from_be_bytes(fixed_bytes!("0x000080").0);
const FACTOR_100: U24 = U24::from_be_bytes(fixed_bytes!("0x000100").0);
const FACTOR_200: U24 = U24::from_be_bytes(fixed_bytes!("0x000200").0);
const FACTOR_400: U24 = U24::from_be_bytes(fixed_bytes!("0x000400").0);
const FACTOR_800: U24 = U24::from_be_bytes(fixed_bytes!("0x000800").0);
const FACTOR_1000: U24 = U24::from_be_bytes(fixed_bytes!("0x001000").0);
const FACTOR_2000: U24 = U24::from_be_bytes(fixed_bytes!("0x002000").0);
const FACTOR_4000: U24 = U24::from_be_bytes(fixed_bytes!("0x004000").0);
const FACTOR_8000: U24 = U24::from_be_bytes(fixed_bytes!("0x008000").0);
const FACTOR_10000: U24 = U24::from_be_bytes(fixed_bytes!("0x010000").0);
const FACTOR_20000: U24 = U24::from_be_bytes(fixed_bytes!("0x020000").0);
const FACTOR_40000: U24 = U24::from_be_bytes(fixed_bytes!("0x040000").0);
const FACTOR_80000: U24 = U24::from_be_bytes(fixed_bytes!("0x080000").0);

pub fn sqrt_price_at_tick(tick: I24) -> Result<U160, TickMathError> {
    // original solidity code for reference:

    //uint256 absTick = tick < 0 ? uint256(-int256(tick)) : uint256(int256(tick));
    //require(absTick <= uint256(MAX_TICK), 'T');

    // uint256 ratio = abs_tick & 0x1 != 0 ? 0xfffcb933bd6fad37aa2d162d1a594001 : 0x100000000000000000000000000000000;
    // if (abs_tick & 0x2 != 0) ratio = (ratio * 0xfff97272373d413259a46990580e213a) >> 128;
    // if (abs_tick & 0x4 != 0) ratio = (ratio * 0xfff2e50f5f656932ef12357cf3c7fdcc) >> 128;
    // if (abs_tick & 0x8 != 0) ratio = (ratio * 0xffe5caca7e10e4e61c3624eaa0941cd0) >> 128;
    // if (abs_tick & 0x10 != 0) ratio = (ratio * 0xffcb9843d60f6159c9db58835c926644) >> 128;
    // if (abs_tick & 0x20 != 0) ratio = (ratio * 0xff973b41fa98c081472e6896dfb254c0) >> 128;
    // if (abs_tick & 0x40 != 0) ratio = (ratio * 0xff2ea16466c96a3843ec78b326b52861) >> 128;
    // if (abs_tick & 0x80 != 0) ratio = (ratio * 0xfe5dee046a99a2a811c461f1969c3053) >> 128;
    // if (abs_tick & 0x100 != 0) ratio = (ratio * 0xfcbe86c7900a88aedcffc83b479aa3a4) >> 128;
    // if (abs_tick & 0x200 != 0) ratio = (ratio * 0xf987a7253ac413176f2b074cf7815e54) >> 128;
    // if (abs_tick & 0x400 != 0) ratio = (ratio * 0xf3392b0822b70005940c7a398e4b70f3) >> 128;
    // if (abs_tick & 0x800 != 0) ratio = (ratio * 0xe7159475a2c29b7443b29c7fa6e889d9) >> 128;
    // if (abs_tick & 0x1000 != 0) ratio = (ratio * 0xd097f3bdfd2022b8845ad8f792aa5825) >> 128;
    // if (abs_tick & 0x2000 != 0) ratio = (ratio * 0xa9f746462d870fdf8a65dc1f90e061e5) >> 128;
    // if (abs_tick & 0x4000 != 0) ratio = (ratio * 0x70d869a156d2a1b890bb3df62baf32f7) >> 128;
    // if (abs_tick & 0x8000 != 0) ratio = (ratio * 0x31be135f97d08fd981231505542fcfa6) >> 128;
    // if (abs_tick & 0x10000 != 0) ratio = (ratio * 0x9aa508b5b7a84e1c677de54f3e99bc9) >> 128;
    // if (abs_tick & 0x20000 != 0) ratio = (ratio * 0x5d6af8dedb81196699c329225ee604) >> 128;
    // if (abs_tick & 0x40000 != 0) ratio = (ratio * 0x2216e584f5fa1ea926041bedfe98) >> 128;
    // if (abs_tick & 0x80000 != 0) ratio = (ratio * 0x48a170391f7dc42444e8fa2) >> 128;
    // if (tick > 0) ratio = type(uint256).max / ratio;

    // this divides by 1<<32 rounding up to go from a Q128.128 to a Q128.96.
    // we then downcast because we know the result always fits within 160 bits due to our tick input constraint
    // we round up in the division so getTickAtSqrtRatio of the output price is always consistent
    // sqrtPriceX96 = uint160((ratio >> 32) + (ratio % (1 << 32) == 0 ? 0 : 1));

    let abs_tick = if tick < I24::ZERO {
        U256::from(-tick)
    } else {
        U256::from(tick)
    };

    if abs_tick > U256::from(MAX_TICK) {
        return Err(TickMathError::SqrtPriceTickOutOfBounds { abs_tick });
    }

    let mut ratio: U256 = if abs_tick & U256::from(FACTOR_1) != U256::ZERO {
        U256::from(VAL_1)
    } else {
        U256::from(VAL_0)
    };
    if abs_tick & U256::from(FACTOR_2) != U256::ZERO {
        ratio = (ratio * U256::from(VAL_2)) >> 128;
    }
    if abs_tick & U256::from(FACTOR_4) != U256::ZERO {
        ratio = (ratio * U256::from(VAL_4)) >> 128;
    }
    if abs_tick & U256::from(FACTOR_8) != U256::ZERO {
        ratio = (ratio * U256::from(VAL_8)) >> 128;
    }
    if abs_tick & U256::from(FACTOR_10) != U256::ZERO {
        ratio = (ratio * U256::from(VAL_10)) >> 128;
    }
    if abs_tick & U256::from(FACTOR_20) != U256::ZERO {
        ratio = (ratio * U256::from(VAL_20)) >> 128;
    }
    if abs_tick & U256::from(FACTOR_40) != U256::ZERO {
        ratio = (ratio * U256::from(VAL_40)) >> 128;
    }
    if abs_tick & U256::from(FACTOR_80) != U256::ZERO {
        ratio = (ratio * U256::from(VAL_80)) >> 128;
    }
    if abs_tick & U256::from(FACTOR_100) != U256::ZERO {
        ratio = (ratio * U256::from(VAL_100)) >> 128;
    }
    if abs_tick & U256::from(FACTOR_200) != U256::ZERO {
        ratio = (ratio * U256::from(VAL_200)) >> 128;
    }
    if abs_tick & U256::from(FACTOR_400) != U256::ZERO {
        ratio = (ratio * U256::from(VAL_400)) >> 128;
    }
    if abs_tick & U256::from(FACTOR_800) != U256::ZERO {
        ratio = (ratio * U256::from(VAL_800)) >> 128;
    }
    if abs_tick & U256::from(FACTOR_1000) != U256::ZERO {
        ratio = (ratio * U256::from(VAL_1000)) >> 128;
    }
    if abs_tick & U256::from(FACTOR_2000) != U256::ZERO {
        ratio = (ratio * U256::from(VAL_2000)) >> 128;
    }
    if abs_tick & U256::from(FACTOR_4000) != U256::ZERO {
        ratio = (ratio * U256::from(VAL_4000)) >> 128;
    }
    if abs_tick & U256::from(FACTOR_8000) != U256::ZERO {
        ratio = (ratio * U256::from(VAL_8000)) >> 128;
    }
    if abs_tick & U256::from(FACTOR_10000) != U256::ZERO {
        ratio = (ratio * U256::from(VAL_10000)) >> 128;
    }
    if abs_tick & U256::from(FACTOR_20000) != U256::ZERO {
        ratio = (ratio * U256::from(VAL_20000)) >> 128;
    }
    if abs_tick & U256::from(FACTOR_40000) != U256::ZERO {
        ratio = (ratio * U256::from(VAL_40000)) >> 128;
    }
    if abs_tick & U256::from(FACTOR_80000) != U256::ZERO {
        ratio = (ratio * U256::from(VAL_80000)) >> 128;
    }

    if tick.is_positive() {
        ratio = U256::MAX / ratio;
    }

    let sqrt_price_x96 = (ratio >> 32)
        + if ratio % (U256::from(1) << 32) == U256::ZERO {
            U256::ZERO
        } else {
            U256::from(1)
        };

    Ok(U160::from(sqrt_price_x96))
}

pub fn tick_low(tick: I24, tick_spacing: U24) -> Result<I24, TickMathError> {
    if tick < -MAX_TICK || tick > MAX_TICK {
        return Err(TickMathError::TickOutOfBounds { tick });
    }

    if tick_spacing == U24::ZERO {
        return Err(TickMathError::NonPositiveTickSpacing { tick_spacing });
    }

    let i_tick_spacing = I24::from(tick_spacing);

    tick.div_euclid(i_tick_spacing)
        .checked_mul(i_tick_spacing)
        .ok_or(TickMathError::TickLowMultiplicationOverflow {
            tick,
            tick_spacing: i_tick_spacing,
        })
        .map(|t| t.max(-MAX_TICK))
}

pub fn tick_high(tick: I24, tick_spacing: U24) -> Result<I24, TickMathError> {
    let tick_low = tick_low(tick, tick_spacing)?;
    let i_tick_spacing = I24::from(tick_spacing);
    tick_low
        .checked_add(i_tick_spacing)
        .ok_or(TickMathError::TickHighAdditionOverflow { tick, tick_spacing })
        .map(|t| t.min(MAX_TICK))
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_tick_low() {
        let tick = I24::from_limbs([1000]);
        let tick_spacing = U24::from_limbs([60]);
        let expected_tick_low = I24::from_limbs([960]);

        let result = tick_low(tick, tick_spacing).unwrap();
        assert_eq!(result, expected_tick_low);
    }

    #[test]
    fn test_tick_low_negative() {
        let tick = -I24::from_limbs([1000]);
        let tick_spacing = U24::from_limbs([60]);
        let expected_tick_low = -I24::from_limbs([1020]);

        let result = tick_low(tick, tick_spacing).unwrap();
        assert_eq!(result, expected_tick_low);
    }

    #[test]
    fn test_tick_low_returns_within_bounds() {
        let tick = -MAX_TICK;
        let tick_spacing = U24::from_limbs([60]);
        let result = tick_low(tick, tick_spacing).unwrap();
        assert!(result >= -MAX_TICK && result <= MAX_TICK);
    }

    #[test]
    fn test_tick_high() {
        let tick = I24::from_limbs([1000]);
        let tick_spacing = U24::from_limbs([60]);
        let expected_tick_high = I24::from_limbs([1020]);

        let result = tick_high(tick, tick_spacing).unwrap();
        assert_eq!(result, expected_tick_high);
    }

    #[test]
    fn test_tick_high_negative() {
        let tick = -I24::from_limbs([1000]);
        let tick_spacing = U24::from_limbs([60]);
        let expected_tick_high = -I24::from_limbs([960]);

        let result = tick_high(tick, tick_spacing).unwrap();
        assert_eq!(result, expected_tick_high);
    }

    #[test]
    fn test_tick_high_returns_within_bounds() {
        let tick = MAX_TICK;
        let tick_spacing = U24::from_limbs([60]);
        let result = tick_high(tick, tick_spacing).unwrap();
        assert!(result >= -MAX_TICK && result <= MAX_TICK);
    }

    #[test]
    fn sqrt_price_at_tick_returns_typed_out_of_bounds_error() {
        let tick = I24::from_limbs([887273]);

        let error = sqrt_price_at_tick(tick).unwrap_err();

        assert!(matches!(
            error,
            TickMathError::SqrtPriceTickOutOfBounds { .. }
        ));
        assert_eq!(error.to_string(), "Tick out of bounds: 887273");
    }

    #[test]
    fn tick_low_returns_typed_out_of_bounds_error() {
        let tick = I24::from_limbs([887273]);

        let error = tick_low(tick, U24::from_limbs([60])).unwrap_err();

        assert!(matches!(error, TickMathError::TickOutOfBounds { .. }));
        assert_eq!(error.to_string(), "Tick out of bounds: 887273");
    }

    #[test]
    fn sqrt_price_at_tick_matches_reference_values() {
        assert_eq!(
            sqrt_price_at_tick(-I24::from_limbs([1])).unwrap(),
            U160::from(79224201403219477170569942574_u128)
        );
        assert_eq!(
            sqrt_price_at_tick(I24::ZERO).unwrap(),
            U160::from(79228162514264337593543950336_u128)
        );
        assert_eq!(
            sqrt_price_at_tick(I24::from_limbs([1])).unwrap(),
            U160::from(79232123823359799118286999568_u128)
        );
    }

    proptest! {
        #[test]
        fn sqrt_price_at_tick_is_strictly_increasing(tick in -887272_i32..887272_i32) {
            let tick = I24::try_from(tick).unwrap();
            let next_tick = tick + I24::from_limbs([1]);

            prop_assert!(sqrt_price_at_tick(tick).unwrap() < sqrt_price_at_tick(next_tick).unwrap());
        }
    }
}
