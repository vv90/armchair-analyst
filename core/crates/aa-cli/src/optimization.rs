use client_evm::{PoolAddress, TokenAddress, multi_chain_kernel::OptimizationPoolReserves};
use optimization::{
    OptimizationBackendSelection, OptimizationInitError, OptimizationRunner,
    OptimizationSessionConfig, OptimizationStepConfig, OptimizationStepError,
    OptimizationStepUpdate,
};

use crate::latest_slot::{LatestReceiveError, LatestReceiver};

const OPTIMIZATION_LAYERS: usize = 1;

#[derive(Clone, Debug, PartialEq)]
pub enum RunOptimizationError {
    Receive(LatestReceiveError),
    Init(OptimizationInitError),
    Step(OptimizationStepError),
}

pub fn run_optimization(
    receiver: LatestReceiver<OptimizationPoolReserves>,
    backend: OptimizationBackendSelection,
    session_config: OptimizationSessionConfig<TokenAddress>,
    step_config: OptimizationStepConfig,
) -> Result<(), RunOptimizationError> {
    let initial_snapshot = receiver
        .wait_take()
        .map_err(RunOptimizationError::Receive)?;
    let (mut runner, _result) =
        OptimizationRunner::<PoolAddress, TokenAddress, OPTIMIZATION_LAYERS>::init(
            backend,
            initial_snapshot.reserves,
            session_config,
            step_config,
        )
        .map_err(RunOptimizationError::Init)?;

    loop {
        let update = match receiver.try_take().map_err(RunOptimizationError::Receive)? {
            Some(snapshot) => OptimizationStepUpdate::NewReserves(snapshot.reserves),
            None => OptimizationStepUpdate::Continue,
        };
        let (next_runner, _result) = runner.run(update).map_err(RunOptimizationError::Step)?;

        runner = next_runner;
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use client_evm::{BlockHash, PoolAddress, TokenAddress};
    use optimization::{PoolReserves, VirtualReserveValues};

    use super::*;
    use crate::latest_slot::latest_slot;

    #[test]
    fn closed_before_initial_snapshot_is_returned_as_error() {
        let (sender, receiver) = latest_slot();

        sender.close().unwrap();

        let error = run_optimization(
            receiver,
            OptimizationBackendSelection::Cpu,
            session_config(),
            step_config(),
        )
        .unwrap_err();

        assert_eq!(
            error,
            RunOptimizationError::Receive(LatestReceiveError::Closed)
        );
    }

    #[test]
    fn closed_after_initialization_is_returned_as_error() {
        let (sender, receiver) = latest_slot();

        sender.send(snapshot(1)).unwrap();
        sender.close().unwrap();

        let error = run_optimization(
            receiver,
            OptimizationBackendSelection::Cpu,
            session_config(),
            step_config(),
        )
        .unwrap_err();

        assert_eq!(
            error,
            RunOptimizationError::Receive(LatestReceiveError::Closed)
        );
    }

    fn snapshot(last_byte: u8) -> OptimizationPoolReserves {
        OptimizationPoolReserves {
            block_hash: BlockHash::with_last_byte(last_byte),
            reserves: base_reserves(),
        }
    }

    fn session_config() -> OptimizationSessionConfig<TokenAddress> {
        OptimizationSessionConfig {
            init_asset: token(),
            bridges: HashSet::new(),
        }
    }

    fn step_config() -> OptimizationStepConfig {
        OptimizationStepConfig {
            input_amount: 1.0,
            iterations: 0,
        }
    }

    fn base_reserves() -> Vec<PoolReserves<PoolAddress, TokenAddress>> {
        vec![PoolReserves {
            pool_id: pool(),
            token0: token(),
            token1: token(),
            value: VirtualReserveValues {
                token_0: 2.0,
                token_1: 3.0,
                fee_multiplier: 0.997,
                max_swap_0: 1.0,
                max_swap_1: 1.0,
            },
        }]
    }

    fn pool() -> PoolAddress {
        PoolAddress(Default::default())
    }

    fn token() -> TokenAddress {
        TokenAddress(Default::default())
    }
}
