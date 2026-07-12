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

    /// Plants a genuinely profitable USDC -> WETH -> WBTC -> USDC cycle through the 3000-fee
    /// pools by skewing the WETH/WBTC price ~10% (multiplicative — an additive nudge would
    /// vanish against the ~1e22-scale reserves). Both directional entries mirror the same pool,
    /// so `token_1` forward and `token_0` reverse grow together. Returns the reserves and the
    /// cycle's after-fee profit for a 1000-unit USDC input, verified with `calculate_quote`.
    pub fn plant_arbitrage(
        mut reserves_map: HashMap<(TokenAddress, TokenAddress, i32), VirtualReserveValues>,
    ) -> (
        HashMap<(TokenAddress, TokenAddress, i32), VirtualReserveValues>,
        f32,
    ) {
        let price_skew = 1.10;
        {
            let weth_wbtc = reserves_map
                .get_mut(&(tokens::WETH.address, tokens::WBTC.address, 3000))
                .expect("WETH/WBTC 3000 pool missing from fixture");
            weth_wbtc.token_1 *= price_skew;
        }
        {
            let wbtc_weth = reserves_map
                .get_mut(&(tokens::WBTC.address, tokens::WETH.address, 3000))
                .expect("WBTC/WETH 3000 pool missing from fixture");
            wbtc_weth.token_0 *= price_skew;
        }

        let usdc_weth = reserves_map
            .get(&(tokens::USDC.address, tokens::WETH.address, 3000))
            .expect("USDC/WETH 3000 pool missing from fixture");
        let weth_wbtc = reserves_map
            .get(&(tokens::WETH.address, tokens::WBTC.address, 3000))
            .expect("WETH/WBTC 3000 pool missing from fixture");
        let wbtc_usdc = reserves_map
            .get(&(tokens::WBTC.address, tokens::USDC.address, 3000))
            .expect("WBTC/USDC 3000 pool missing from fixture");

        let usdc_amount = 1000.0;
        let weth_amount = calculate_quote(usdc_weth, usdc_amount, false);
        let wbtc_amount = calculate_quote(weth_wbtc, weth_amount, false);
        let usdc_amount_final = calculate_quote(wbtc_usdc, wbtc_amount, false);

        (reserves_map, usdc_amount_final - usdc_amount)
    }
}
