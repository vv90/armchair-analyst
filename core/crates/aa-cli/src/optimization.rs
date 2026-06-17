use std::sync::mpsc::Sender;

use client_evm::{PoolAddress, TokenAddress, multi_chain_kernel::OptimizationPoolReserves};
use optimization::{
    OptimizationBackendSelection, OptimizationInitError, OptimizationRunner,
    OptimizationSessionConfig, OptimizationStepConfig, OptimizationStepError, OptimizationStepResult,
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

pub fn run_optimization<T>(
    receiver: LatestReceiver<OptimizationPoolReserves>,
    backend: OptimizationBackendSelection,
    session_config: OptimizationSessionConfig<TokenAddress>,
    step_config: OptimizationStepConfig,
    result_sender: Sender<T>,
    map_result: impl Fn(OptimizationStepResult) -> T,
) -> Result<(), RunOptimizationError> {
    let initial_snapshot = receiver
        .wait_take()
        .map_err(RunOptimizationError::Receive)?;
    let (mut runner, result) =
        OptimizationRunner::<PoolAddress, TokenAddress, OPTIMIZATION_LAYERS>::init(
            backend,
            initial_snapshot.reserves,
            session_config,
            step_config,
        )
        .map_err(RunOptimizationError::Init)?;

    if result_sender.send(map_result(result)).is_err() {
        return Ok(());
    }

    loop {
        let update = match receiver.try_take().map_err(RunOptimizationError::Receive)? {
            Some(snapshot) => OptimizationStepUpdate::NewReserves(snapshot.reserves),
            None => OptimizationStepUpdate::Continue,
        };
        let (next_runner, result) = runner.run(update).map_err(RunOptimizationError::Step)?;

        if result_sender.send(map_result(result)).is_err() {
            return Ok(());
        }

        runner = next_runner;
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, sync::mpsc};

    use client_evm::{BlockHash, PoolAddress, TokenAddress};
    use optimization::{OptimizationStepStatus, PoolReserves, VirtualReserveValues};

    use super::*;
    use crate::latest_slot::latest_slot;

    #[test]
    fn closed_before_initial_snapshot_is_returned_as_error() {
        let (sender, receiver) = latest_slot();

        sender.close().unwrap();

        let (result_sender, _result_receiver) = mpsc::channel();
        let error = run_optimization(
            receiver,
            OptimizationBackendSelection::Cpu,
            session_config(),
            step_config(),
            result_sender,
            |result| result,
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

        let (result_sender, result_receiver) = mpsc::channel();
        let error = run_optimization(
            receiver,
            OptimizationBackendSelection::Cpu,
            session_config(),
            step_config(),
            result_sender,
            |result| result,
        )
        .unwrap_err();

        assert_eq!(
            error,
            RunOptimizationError::Receive(LatestReceiveError::Closed)
        );
        assert_eq!(
            result_receiver.recv().map(|result| result.status),
            Ok(OptimizationStepStatus::Initialized)
        );
    }

    #[test]
    fn exits_cleanly_when_result_receiver_is_dropped() {
        let (sender, receiver) = latest_slot();

        sender.send(snapshot(1)).unwrap();

        let (result_sender, result_receiver) = mpsc::channel::<OptimizationStepResult>();
        drop(result_receiver);

        let outcome = run_optimization(
            receiver,
            OptimizationBackendSelection::Cpu,
            session_config(),
            step_config(),
            result_sender,
            |result| result,
        );

        assert_eq!(outcome, Ok(()));
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
