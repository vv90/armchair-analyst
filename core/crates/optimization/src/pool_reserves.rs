use crate::utils::Invertible;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PoolReserves<T: Copy, I: Copy> {
    pub token0: I,
    pub token1: I,
    pub pool_id: T,
    pub value: VirtualReserveValues,
}

impl<T: Copy, I: Copy> Invertible for PoolReserves<T, I> {
    fn inverse(self) -> PoolReserves<T, I> {
        PoolReserves {
            token0: self.token1,
            token1: self.token0,
            pool_id: self.pool_id,
            value: self.value.inverse(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VirtualReserveValues {
    pub token_0: f32,
    pub token_1: f32,
    pub fee_multiplier: f32,
    pub max_swap_0: f32,
    pub max_swap_1: f32,
}

impl Invertible for VirtualReserveValues {
    fn inverse(self) -> VirtualReserveValues {
        VirtualReserveValues {
            token_0: self.token_1,
            token_1: self.token_0,
            fee_multiplier: self.fee_multiplier,
            max_swap_0: self.max_swap_1,
            max_swap_1: self.max_swap_0,
        }
    }
}

#[cfg(test)]
pub mod test {
    use std::collections::HashMap;

    use crate::tokens::test as tokens;
    use crate::tokens::test::TokenAddress;

    use super::*;

    fn calculate_quote(
        reserves: &VirtualReserveValues,
        amount_in: f32,
        is_reverse_swap: bool,
    ) -> f32 {
        let amount_in_with_fee = amount_in * reserves.fee_multiplier;
        if is_reverse_swap {
            reserves.token_0 * amount_in_with_fee / (reserves.token_1 + amount_in_with_fee)
        } else {
            reserves.token_1 * amount_in_with_fee / (reserves.token_0 + amount_in_with_fee)
        }
    }

    pub fn plant_arbitrage(
        mut reserves_map: HashMap<(TokenAddress, TokenAddress, i32), VirtualReserveValues>,
    ) -> (
        HashMap<(TokenAddress, TokenAddress, i32), VirtualReserveValues>,
        f32,
    ) {
        // Plant arbitrage by modifying reserves
        // Increase USDC in WETH pool and decrease in WBTC pool
        // {
        //     let usdc_weth = reserves_map
        //         .get_mut(&(tokens::USDC.address, tokens::WETH.address, Fee::Medium))
        //         .unwrap();

        //     usdc_weth.token_0 += 1000000.0;
        // }
        {
            let weth_wbtc = reserves_map
                .get_mut(&(tokens::WETH.address, tokens::WBTC.address, 3000))
                .unwrap();
            println!("WETH/WBTC: {:?}", weth_wbtc);
            weth_wbtc.token_1 += 100.0;
        }
        {
            let wbtc_weth = reserves_map
                .get_mut(&(tokens::WBTC.address, tokens::WETH.address, 3000))
                .unwrap();
            println!("WBTC/WETH: {:?}", wbtc_weth);
            wbtc_weth.token_0 += 100.0;
        }
        // {
        //     let wbtc_usdc = reserves_map
        //         .get_mut(&(tokens::WBTC.address, tokens::USDC.address, Fee::Medium))
        //         .unwrap();
        // }

        let usdc_weth = reserves_map
            .get(&(tokens::USDC.address, tokens::WETH.address, 3000))
            .unwrap();
        let weth_wbtc = reserves_map
            .get(&(tokens::WETH.address, tokens::WBTC.address, 3000))
            .unwrap();
        let wbtc_usdc = reserves_map
            .get(&(tokens::WBTC.address, tokens::USDC.address, 3000))
            .unwrap();
        let wbtc_weth = reserves_map
            .get(&(tokens::WBTC.address, tokens::WETH.address, 3000))
            .unwrap();

        println!("WETH/WBTC: {:?}", weth_wbtc);
        println!("WBTC/WETH: {:?}", wbtc_weth);
        let usdc_amount = 1000.0;
        let weth_amount = calculate_quote(usdc_weth, usdc_amount, false);
        let wbtc_amount = calculate_quote(weth_wbtc, weth_amount, false);
        let usdc_amount_final = calculate_quote(wbtc_usdc, wbtc_amount, false);

        println!(
            "Planted arbitrage: USDC -> WETH -> WBTC -> USDC: {} -> {} -> {} -> {}",
            usdc_amount, weth_amount, wbtc_amount, usdc_amount_final
        );

        (reserves_map, usdc_amount_final - usdc_amount)
    }
}
// #[cfg(test)]
// pub mod tests {

//     use crate::{
//         ethereum::tokens,
//         uniswap::v3::{
//             pool::{Fee, PoolAddress},
//             pool_state::PoolState,
//         },
//     };
//     use alloy::primitives::{Address, FixedBytes, U160, aliases::I24};
//     use rust_decimal::prelude::*;
//     use rust_decimal_macros::dec;
//     use tokens::TokenInfo;

//     use super::*;

//     const POOL_STATE_WBTC_USDC: PoolState = PoolState {
//         sqrt_price_x96: U160::from_limbs([17134602959287796597, 139272449984, 0]),
//         liquidity: U160::from_limbs([50170120777514, 0, 0]),
//         tick: I24::from_limbs([69583]),
//     };

//     #[test]
//     fn test_calculate_quote() {
//         let fee = Fee::Medium;

//         let token_0 = tokens::WBTC.clone();
//         let token_1 = tokens::USDC.clone();
//         let reserves = POOL_STATE_WBTC_USDC
//             .pool_virtual_reserves(
//                 tokens::WBTC.decimals,
//                 tokens::USDC.decimals,
//                 fee as u32,
//                 fee.tick_spacing(),
//             )
//             .unwrap();

//         let reference_amount_in = 10u128.pow(token_0.decimals() - 3);
//         let reference_quote =
//             POOL_STATE_WBTC_USDC.calculate_quote(10u128.pow(token_0.decimals()), fee, false);
//         println!(
//             "Reference quote: {} -> {}",
//             Decimal::from_i128_with_scale(reference_amount_in as i128, token_0.decimals()),
//             Decimal::from_i128_with_scale(reference_quote as i128, token_1.decimals())
//         );
//         println!("{} - {}", token_0.symbol(), token_1.symbol());

//         let expected_quote = POOL_STATE_WBTC_USDC.calculate_quote_dec(
//             dec!(1),
//             false,
//             token_0.decimals(),
//             token_1.decimals(),
//         );

//         let expected_reverse_quote = POOL_INFO_WBTC_USDC.calculate_quote_dec(
//             dec!(100.0),
//             true,
//             token_0.decimals(),
//             token_1.decimals(),
//         );

//         println!("Expected quote: {} -> {}", dec!(1), expected_quote);

//         println!(
//             "Expected reverse quote: {} -> {}",
//             dec!(100.0),
//             expected_reverse_quote
//         );

//         let quote = calculate_quote(&reserves, 1.0, false);
//         let reverse_quote = calculate_quote(&reserves, 100.0, true);

//         assert!(
//             (quote - expected_quote.to_f64().unwrap()).abs() < f64::EPSILON,
//             "Quote mismatch. Expected: {}, Got: {}",
//             expected_quote,
//             quote
//         );
//         assert!(
//             (reverse_quote - expected_reverse_quote.to_f64().unwrap()).abs() < f64::EPSILON,
//             "Reverse quote mismatch. Expected: {}, Got: {}",
//             expected_reverse_quote,
//             reverse_quote
//         );
//     }
// }
