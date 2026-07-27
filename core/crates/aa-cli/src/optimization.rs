use std::collections::HashSet;
use std::sync::mpsc::Sender;

use client_evm::{PoolRef, TokenAddress, multi_chain_kernel::OptimizationPoolReserves};
use optimization::{
    ExecutionPlan, OptimizationBackendSelection, OptimizationInitError, OptimizationRunner,
    OptimizationSessionConfig, OptimizationStepConfig, OptimizationStepError,
    OptimizationStepResult, OptimizationStepUpdate, reserves_reach_init_asset,
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
    map_result: impl Fn(OptimizationStepResult, Option<ExecutionPlan<PoolRef, TokenAddress>>) -> T,
) -> Result<(), RunOptimizationError> {
    // Wait for the first snapshot that can actually initialize. With cross-chain merging, the first
    // snapshot may arrive from a faster chain before the chain that owns the `init_asset` (quote
    // token) has reported, so the init asset is absent and init fails with `InitAssetOutputNotFound`.
    // That is a transient "not ready yet" condition, not a fatal error: drop the snapshot and wait
    // for the next one. Every other init error stays fatal.
    let (mut runner, result, plan) = loop {
        let snapshot = receiver
            .wait_take()
            .map_err(RunOptimizationError::Receive)?;
        match OptimizationRunner::<PoolRef, TokenAddress, OPTIMIZATION_LAYERS>::init(
            backend,
            snapshot.reserves,
            session_config.clone(),
            step_config,
        ) {
            Ok(initialized) => break initialized,
            Err(OptimizationInitError::Step(OptimizationStepError::InitAssetOutputNotFound)) => {
                continue;
            }
            Err(error) => return Err(RunOptimizationError::Init(error)),
        }
    };

    if result_sender.send(map_result(result, plan)).is_err() {
        return Ok(());
    }

    loop {
        let update = match receiver.try_take().map_err(RunOptimizationError::Receive)? {
            // Same transient guard as init: a merged cross-chain snapshot can momentarily fail to
            // reach the init asset (a chain bootstrapping, a refresh gap). Feeding it to the session
            // would abort the worker with `InitAssetOutputNotFound`, permanently closing the
            // optimization channel. Skip the snapshot and keep stepping the live session instead.
            Some(snapshot) if !reserves_reach_init_asset(&snapshot.reserves, &session_config) => {
                OptimizationStepUpdate::Continue
            }
            // No lagging-pool gate wired in yet: every reported pool stays active. The kernel-side
            // staleness gate that populates `disabled` is a separate follow-up.
            Some(snapshot) => OptimizationStepUpdate::NewReserves {
                reserves: snapshot.reserves,
                disabled: HashSet::new(),
            },
            None => OptimizationStepUpdate::Continue,
        };
        let (next_runner, result, plan) = runner.run(update).map_err(RunOptimizationError::Step)?;

        if result_sender.send(map_result(result, plan)).is_err() {
            return Ok(());
        }

        runner = next_runner;
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::{BTreeMap, HashSet},
        sync::mpsc,
    };

    use client_evm::{BlockHash, ChainKey, PoolRef, TokenAddress};
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
            |result, _plan| result,
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
            |result, _plan| result,
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
            |result, _plan| result,
        );

        assert_eq!(outcome, Ok(()));
    }

    #[test]
    fn first_snapshot_missing_init_asset_is_skipped_not_fatal() {
        let (sender, receiver) = latest_slot();

        // A snapshot whose reserves never output the init_asset — e.g. a faster chain reported before
        // the chain that owns the quote token. Init must treat this as "not ready yet": skip it and
        // keep waiting, not abort the worker with an Init error.
        sender.send(snapshot_without_init_asset(1)).unwrap();
        sender.close().unwrap();

        let (result_sender, result_receiver) = mpsc::channel();
        let outcome = run_optimization(
            receiver,
            OptimizationBackendSelection::Cpu,
            session_config(),
            step_config(),
            result_sender,
            |result, _plan| result,
        );

        // The unready snapshot was skipped (no fatal Init error); the wait then ends cleanly on close.
        // Initialization never completed, so no step result was ever produced.
        assert_eq!(
            outcome,
            Err(RunOptimizationError::Receive(LatestReceiveError::Closed))
        );
        assert!(result_receiver.recv().is_err());
    }

    #[test]
    fn map_result_receives_the_emitted_plan() {
        let (sender, receiver) = latest_slot();

        sender.send(snapshot(1)).unwrap();
        sender.close().unwrap();

        let (result_sender, result_receiver) = mpsc::channel();
        let _ = run_optimization(
            receiver,
            OptimizationBackendSelection::Cpu,
            session_config(),
            step_config(),
            result_sender,
            |result, plan| (result, plan),
        );

        let (result, plan) = result_receiver.recv().unwrap();
        assert_eq!(result.status, OptimizationStepStatus::Initialized);
        let plan = plan.expect("a completed step must emit a plan");
        assert_eq!(plan.init_asset, session_config().source_asset);
    }

    fn snapshot(last_byte: u8) -> OptimizationPoolReserves {
        OptimizationPoolReserves {
            block_hashes: BTreeMap::from([(
                ChainKey::Ethereum,
                BlockHash::with_last_byte(last_byte),
            )]),
            reserves: base_reserves(),
        }
    }

    fn snapshot_without_init_asset(last_byte: u8) -> OptimizationPoolReserves {
        // The session's init_asset is `token()` (Ethereum-stamped default). An Arbitrum-stamped
        // default token is a distinct key, so no reserve here ever outputs the init_asset.
        let other = TokenAddress(Default::default(), ChainKey::Arbitrum);
        OptimizationPoolReserves {
            block_hashes: BTreeMap::from([(
                ChainKey::Arbitrum,
                BlockHash::with_last_byte(last_byte),
            )]),
            reserves: vec![PoolReserves {
                pool_id: PoolRef::uniswap_v3(Default::default(), ChainKey::Arbitrum),
                token0: other,
                token1: other,
                value: VirtualReserveValues {
                    token_0: 2.0,
                    token_1: 3.0,
                    fee_multiplier: 0.997,
                    max_swap_0: 1.0,
                    max_swap_1: 1.0,
                },
            }],
        }
    }

    fn session_config() -> OptimizationSessionConfig<TokenAddress> {
        OptimizationSessionConfig {
            source_asset: token(),
            output_asset: token(),
            bridges: HashSet::new(),
            whitelist: None,
        }
    }

    fn step_config() -> OptimizationStepConfig {
        OptimizationStepConfig {
            input_amount: 1.0,
            iterations: 0,
        }
    }

    fn base_reserves() -> Vec<PoolReserves<PoolRef, TokenAddress>> {
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

    fn pool() -> PoolRef {
        PoolRef::uniswap_v3(Default::default(), ChainKey::Ethereum)
    }

    fn token() -> TokenAddress {
        TokenAddress(Default::default(), ChainKey::Ethereum)
    }
}
