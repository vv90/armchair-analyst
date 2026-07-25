//! The executor for [`crate::Effect::Optimize`]: the optimizer worker. It owns the heavy mutable
//! state the pure reducer deliberately does not — `optimization::OptimizationRunner`, the move-based
//! state machine holding the `Model`/optimizer/session tensors — and turns the reducer's
//! [`OptimizeCommand`]s into [`Event::OptimizerStepped`] outcomes, closing the `Optimize → stepped`
//! ping-pong.
//!
//! The decision logic (init vs. run vs. continue, whether a slice is productive) lives entirely in the
//! reducer, so this worker is a pure *command executor*: it runs exactly what it is told and reports
//! the result. [`OptimizerWorker::handle`] is synchronous and does no I/O, so a command sequence is
//! testable by asserting the returned events (deterministic on the Cpu backend). [`run`] is the only
//! threaded part — a thin channel loop around `handle`.

use std::sync::mpsc::{Receiver, Sender};

use client_evm::{PoolRef, TokenAddress};
use optimization::OptimizationRunner;

use crate::state::{EffectError, Event, OptimizeCommand, OptimizeStage};

/// One optimizer session's worth of layers. Matches aa-cli's `OPTIMIZATION_LAYERS`.
const LAYERS: usize = 1;

/// Owns the move-based runner at the effect edge. `None` until the first successful `Init`. Pure:
/// [`OptimizerWorker::handle`] performs no I/O and spawns nothing.
#[derive(Default)]
pub struct OptimizerWorker {
    runner: Option<OptimizationRunner<PoolRef, TokenAddress, LAYERS>>,
}

impl OptimizerWorker {
    /// A worker with no runner yet: the first `Init` command creates one.
    pub fn new() -> OptimizerWorker {
        OptimizerWorker { runner: None }
    }

    /// Execute one command, returning the [`Event`] the reducer must be fed. Total: every optimizer
    /// error — and a `Run` received before any `Init` — degrades to
    /// [`Event::EffectFailed`]`(`[`EffectError::Optimize`]`)` rather than a panic, so the worker can
    /// never take down the loop.
    pub fn handle(&mut self, command: OptimizeCommand) -> Event {
        match command {
            OptimizeCommand::Init {
                reserves,
                session_config,
                step_config,
                backend,
            } => match OptimizationRunner::<PoolRef, TokenAddress, LAYERS>::init(
                backend,
                reserves,
                session_config,
                step_config,
            ) {
                Ok((runner, result, plan)) => {
                    self.runner = Some(runner);
                    Event::OptimizerStepped { result, plan }
                }
                Err(error) => Event::EffectFailed(EffectError::Optimize {
                    stage: OptimizeStage::Init,
                    message: error.to_string(),
                }),
            },
            OptimizeCommand::Run(update) => match self.runner.take() {
                Some(runner) => match runner.run(update) {
                    Ok((runner, result, plan)) => {
                        self.runner = Some(runner);
                        Event::OptimizerStepped { result, plan }
                    }
                    Err(error) => Event::EffectFailed(EffectError::Optimize {
                        stage: OptimizeStage::Run,
                        message: error.to_string(),
                    }),
                },
                // A `Run` before `Init`: the reducer only emits this after it has emitted an `Init`,
                // and the command channel is FIFO, so it should not happen — but stay total.
                None => Event::EffectFailed(EffectError::Optimize {
                    stage: OptimizeStage::Run,
                    message: "run before init".to_owned(),
                }),
            },
        }
    }
}

/// The threaded shell: own a fresh [`OptimizerWorker`], execute each incoming command, and forward the
/// resulting event. Returns (ending the thread) when either channel closes — the command sender
/// dropped (`recv` errs) or the event receiver dropped (`send` errs) — so a torn-down driver stops
/// the worker cleanly. Panic-free: both channel ends are handled as `Result`.
pub fn run(commands: Receiver<OptimizeCommand>, events: Sender<Event>) {
    let mut worker = OptimizerWorker::new();
    while let Ok(command) = commands.recv() {
        if events.send(worker.handle(command)).is_err() {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::sync::mpsc;
    use std::thread;

    use client_evm::{ChainKey, PoolRef, TokenAddress};
    use optimization::{
        OptimizationBackendSelection, OptimizationSessionConfig, OptimizationStepConfig,
        OptimizationStepStatus, OptimizationStepUpdate, PoolReserves, VirtualReserveValues,
    };

    use super::*;

    const CHAIN: ChainKey = ChainKey::Ethereum;

    fn token() -> TokenAddress {
        TokenAddress(Default::default(), CHAIN)
    }

    fn pool() -> PoolRef {
        PoolRef::uniswap_v3(Default::default(), CHAIN)
    }

    fn session_config() -> OptimizationSessionConfig<TokenAddress> {
        OptimizationSessionConfig {
            init_asset: token(),
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

    /// A single self-token pool whose output is the init asset — the minimal snapshot that initializes
    /// (mirrors aa-cli's optimizer-test fixture).
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

    fn init_command(reserves: Vec<PoolReserves<PoolRef, TokenAddress>>) -> OptimizeCommand {
        OptimizeCommand::Init {
            reserves,
            session_config: session_config(),
            step_config: step_config(),
            backend: OptimizationBackendSelection::Cpu,
        }
    }

    /// Asserts an event is an `OptimizerStepped` of the given status and returns whether it carried a
    /// plan.
    fn expect_step(event: Event, status: OptimizationStepStatus) -> bool {
        match event {
            Event::OptimizerStepped { result, plan } => {
                assert_eq!(result.status, status);
                plan.is_some()
            }
            other => panic!("expected OptimizerStepped, got {other:?}"),
        }
    }

    #[test]
    fn init_produces_an_initialized_step_and_retains_the_runner() {
        let mut worker = OptimizerWorker::new();
        let has_plan = expect_step(
            worker.handle(init_command(base_reserves())),
            OptimizationStepStatus::Initialized,
        );
        assert!(has_plan, "a completed step must emit a plan");
        assert!(worker.runner.is_some(), "the runner must be retained");
    }

    #[test]
    fn continue_after_init_advances_the_session() {
        let mut worker = OptimizerWorker::new();
        worker.handle(init_command(base_reserves()));
        expect_step(
            worker.handle(OptimizeCommand::Run(OptimizationStepUpdate::Continue)),
            OptimizationStepStatus::Continued,
        );
    }

    #[test]
    fn new_reserves_with_same_keys_updates_in_place() {
        let mut worker = OptimizerWorker::new();
        worker.handle(init_command(base_reserves()));
        expect_step(
            worker.handle(OptimizeCommand::Run(OptimizationStepUpdate::NewReserves {
                reserves: base_reserves(),
                disabled: HashSet::new(),
            })),
            OptimizationStepStatus::Updated,
        );
    }

    #[test]
    fn run_before_init_is_a_recorded_error() {
        let mut worker = OptimizerWorker::new();
        let event = worker.handle(OptimizeCommand::Run(OptimizationStepUpdate::Continue));
        assert!(matches!(
            event,
            Event::EffectFailed(EffectError::Optimize {
                stage: OptimizeStage::Run,
                ..
            })
        ));
        assert!(worker.runner.is_none());
    }

    #[test]
    fn init_failure_is_a_recorded_error() {
        // Empty reserves abort init with `EmptyReserves`; the reducer's `is_productive` gate keeps
        // this from happening in practice, but the worker must stay total.
        let mut worker = OptimizerWorker::new();
        let event = worker.handle(init_command(Vec::new()));
        assert!(matches!(
            event,
            Event::EffectFailed(EffectError::Optimize {
                stage: OptimizeStage::Init,
                ..
            })
        ));
        assert!(worker.runner.is_none());
    }

    #[test]
    fn run_loop_processes_commands_in_order_then_exits_on_close() {
        let (command_tx, command_rx) = mpsc::channel();
        let (event_tx, event_rx) = mpsc::channel();
        let handle = thread::spawn(move || run(command_rx, event_tx));

        command_tx
            .send(init_command(base_reserves()))
            .expect("send init");
        command_tx
            .send(OptimizeCommand::Run(OptimizationStepUpdate::Continue))
            .expect("send continue");

        expect_step(
            event_rx.recv().expect("init event"),
            OptimizationStepStatus::Initialized,
        );
        expect_step(
            event_rx.recv().expect("continue event"),
            OptimizationStepStatus::Continued,
        );

        // Dropping the command sender closes the channel, so the worker loop returns and the thread
        // joins.
        drop(command_tx);
        handle.join().expect("worker thread joins");
    }
}
