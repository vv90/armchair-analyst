use alloy::primitives::U256;
use thiserror::Error;

use crate::TokenDecimals;

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum TokenAmountConversionError {
    #[error("token amount conversion produced non-finite f64: {amount}")]
    NonFiniteF64 { amount: U256 },

    #[error("token amount exceeds f32 range after scaling: raw={raw}, decimals={decimals}")]
    F32Overflow { raw: U256, decimals: u8 },
}

pub fn u256_token_amount_to_f32(
    raw: U256,
    decimals: TokenDecimals,
) -> Result<f32, TokenAmountConversionError> {
    let value = u256_token_amount_to_f64(raw, decimals)?;

    if value > f64::from(f32::MAX) {
        return Err(TokenAmountConversionError::F32Overflow {
            raw,
            decimals: decimals.value(),
        });
    }

    let value = value as f32;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(TokenAmountConversionError::F32Overflow {
            raw,
            decimals: decimals.value(),
        })
    }
}

fn u256_token_amount_to_f64(
    raw: U256,
    decimals: TokenDecimals,
) -> Result<f64, TokenAmountConversionError> {
    let scale = decimal_scale(decimals);
    let whole = raw / scale;
    let remainder = raw % scale;
    let scale = u256_to_f64(scale)?;

    Ok(u256_to_f64(whole)? + u256_to_f64(remainder)? / scale)
}

fn decimal_scale(decimals: TokenDecimals) -> U256 {
    U256::from(10u8).pow(U256::from(decimals.value()))
}

fn u256_to_f64(amount: U256) -> Result<f64, TokenAmountConversionError> {
    let value = f64::from(amount);

    if value.is_finite() {
        Ok(value)
    } else {
        Err(TokenAmountConversionError::NonFiniteF64 { amount })
    }
}

#[cfg(test)]
mod tests {
    use alloy::primitives::{U256, U512};
    use proptest::prelude::*;

    use crate::TokenDecimals;

    use super::*;

    #[test]
    fn token_amount_to_f32_has_no_loss_for_small_exact_integer_amount() {
        let raw = U256::from(123_000_000u64);
        let decimals = token_decimals(6);

        let converted = u256_token_amount_to_f32(raw, decimals).unwrap();
        let loss = conversion_loss(raw, decimals, converted).unwrap();

        assert_eq!(converted, 123.0);
        assert_eq!(loss.absolute, 0.0);
        assert_eq!(loss.relative, 0.0);
    }

    #[test]
    fn token_amount_to_f32_quantifies_fractional_weth_sized_loss() {
        let raw = U256::from(1_234_567_890_123_456_789u128);
        let decimals = token_decimals(18);

        let converted = u256_token_amount_to_f32(raw, decimals).unwrap();
        let loss = conversion_loss(raw, decimals, converted).unwrap();

        assert_eq!(converted, 1.2345679);
        assert!(loss.absolute > 9.0e-9);
        assert!(loss.absolute < 1.0e-8);
        assert!(loss.relative > 7.0e-9);
        assert!(loss.relative < 8.0e-9);
    }

    #[test]
    fn token_amount_to_f32_quantifies_large_six_decimal_reserve_loss() {
        let raw = U256::from(123_456_789_123_456u128);
        let decimals = token_decimals(6);

        let converted = u256_token_amount_to_f32(raw, decimals).unwrap();
        let loss = conversion_loss(raw, decimals, converted).unwrap();

        assert_eq!(converted, 123_456_790.0);
        assert!(loss.absolute > 2.8);
        assert!(loss.absolute < 2.9);
        assert!(loss.relative > 2.3e-8);
        assert!(loss.relative < 2.4e-8);
    }

    #[test]
    fn token_amount_to_f32_quantifies_trillion_token_reserve_loss() {
        let raw = pow10(30);
        let decimals = token_decimals(18);

        let converted = u256_token_amount_to_f32(raw, decimals).unwrap();
        let loss = conversion_loss(raw, decimals, converted).unwrap();

        assert_eq!(converted, 1_000_000_000_000.0);
        assert!(loss.absolute > 4_000.0);
        assert!(loss.absolute < 4_100.0);
        assert!(loss.relative > 4.0e-9);
        assert!(loss.relative < 4.1e-9);
    }

    #[test]
    fn token_amount_to_f32_quantifies_tiny_eighteen_decimal_amount_loss() {
        let raw = U256::from(1u8);
        let decimals = token_decimals(18);

        let converted = u256_token_amount_to_f32(raw, decimals).unwrap();
        let loss = conversion_loss(raw, decimals, converted).unwrap();

        assert!(converted > 0.0);
        assert!(loss.absolute < 1.0e-25);
        assert!(loss.relative < 5.0e-8);
    }

    #[test]
    fn token_amount_to_f32_rejects_values_larger_than_f32_range() {
        let result = u256_token_amount_to_f32(U256::MAX, token_decimals(0));

        assert_eq!(
            result,
            Err(TokenAmountConversionError::F32Overflow {
                raw: U256::MAX,
                decimals: 0,
            })
        );
    }

    proptest! {
        #[test]
        fn finite_u128_amounts_convert_to_non_negative_finite_values(
            raw in any::<u128>(),
            decimals in 0u8..=36,
        ) {
            let result = u256_token_amount_to_f32(U256::from(raw), token_decimals(decimals));

            if let Ok(value) = result {
                prop_assert!(value.is_finite());
                prop_assert!(value >= 0.0);
            }
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct ConversionLoss {
        absolute: f64,
        relative: f64,
    }

    fn conversion_loss(
        raw: U256,
        decimals: TokenDecimals,
        converted: f32,
    ) -> Result<ConversionLoss, TokenAmountConversionError> {
        let absolute = exact_absolute_loss(raw, decimals, converted).as_f64();
        let reference = u256_token_amount_to_f64(raw, decimals)?;
        let relative = if reference == 0.0 {
            0.0
        } else {
            absolute / reference.abs()
        };

        Ok(ConversionLoss { absolute, relative })
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct ExactAbsoluteLoss {
        numerator: U512,
        denominator: U512,
    }

    impl ExactAbsoluteLoss {
        fn as_f64(&self) -> f64 {
            u512_to_f64(self.numerator) / u512_to_f64(self.denominator)
        }
    }

    fn exact_absolute_loss(
        raw: U256,
        decimals: TokenDecimals,
        converted: f32,
    ) -> ExactAbsoluteLoss {
        let scale = U512::from(pow10(decimals.value()));
        let raw = U512::from(raw);
        let (significand, exponent) = positive_f32_rational(converted);

        if exponent >= 0 {
            let converted_atoms = (U512::from(significand) << exponent as usize) * scale;
            ExactAbsoluteLoss {
                numerator: abs_diff(raw, converted_atoms),
                denominator: scale,
            }
        } else {
            let denominator_power = U512::from(1u8) << (-exponent) as usize;
            let raw_scaled = raw * denominator_power;
            let converted_scaled = U512::from(significand) * scale;

            ExactAbsoluteLoss {
                numerator: abs_diff(raw_scaled, converted_scaled),
                denominator: scale * denominator_power,
            }
        }
    }

    fn positive_f32_rational(value: f32) -> (u32, i16) {
        let bits = value.to_bits();
        let exponent = ((bits >> 23) & 0xff) as i16;
        let fraction = bits & 0x7f_ffff;

        if exponent == 0 {
            (fraction, -149)
        } else {
            ((1u32 << 23) | fraction, exponent - 150)
        }
    }

    fn abs_diff(left: U512, right: U512) -> U512 {
        if left >= right {
            left - right
        } else {
            right - left
        }
    }

    fn u512_to_f64(value: U512) -> f64 {
        value.to_string().parse::<f64>().unwrap()
    }

    fn token_decimals(decimals: u8) -> TokenDecimals {
        TokenDecimals::try_from_u256(U256::from(decimals)).unwrap()
    }

    fn pow10(exponent: u8) -> U256 {
        U256::from(10u8).pow(U256::from(exponent))
    }
}
