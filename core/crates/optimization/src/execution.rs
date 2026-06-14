use std::{
    any::Any,
    collections::HashSet,
    hash::Hash,
    panic::{AssertUnwindSafe, catch_unwind},
};

use burn::tensor::{ElementConversion, backend::AutodiffBackend, cast::ToElement};
use thiserror::Error;

use crate::{OptimizationError, PoolReserves, model::Model};

const OPTIMIZATION_LAYERS: usize = 1;

type CpuBackend = burn::backend::Autodiff<burn::backend::NdArray<f32>>;
type WgpuBackend = burn::backend::Autodiff<burn::backend::Wgpu<f32>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptimizationBackendSelection {
    Auto,
    Wgpu,
    Cpu,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptimizationBackendUsed {
    Wgpu,
    Cpu,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OptimizationExecutionConfig<TToken> {
    pub backend: OptimizationBackendSelection,
    pub init_asset: TToken,
    pub input_amount: f32,
    pub iterations: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OptimizationExecutionResult {
    pub backend: OptimizationBackendUsed,
    pub input_amount: f32,
    pub output_amount: f32,
    pub profit_amount: f32,
    pub reserves_count: usize,
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum OptimizationExecutionError {
    #[error("optimization reserves are empty")]
    EmptyReserves,

    #[error("optimization input amount must be finite and greater than zero: {input_amount}")]
    InvalidInputAmount { input_amount: f32 },

    #[error("optimization iterations must be greater than zero")]
    ZeroIterations,

    #[error("init asset output not found")]
    InitAssetOutputNotFound,

    #[error("optimization model init failed: {source}")]
    ModelInit { source: OptimizationError },

    #[error("optimization backend {backend:?} failed: {message}")]
    BackendFailed {
        backend: OptimizationBackendUsed,
        message: String,
    },
}

pub fn execute_optimization<TPool, TToken>(
    reserves: Vec<PoolReserves<TPool, TToken>>,
    config: &OptimizationExecutionConfig<TToken>,
) -> Result<OptimizationExecutionResult, OptimizationExecutionError>
where
    TPool: Copy + PartialEq,
    TToken: Clone + Copy + PartialEq + Eq + Hash,
{
    execute_optimization_with_backend_runners(
        reserves,
        config,
        execute_wgpu::<TPool, TToken>,
        execute_cpu::<TPool, TToken>,
    )
}

fn execute_optimization_with_backend_runners<TPool, TToken, WgpuRunner, CpuRunner>(
    reserves: Vec<PoolReserves<TPool, TToken>>,
    config: &OptimizationExecutionConfig<TToken>,
    run_wgpu: WgpuRunner,
    run_cpu: CpuRunner,
) -> Result<OptimizationExecutionResult, OptimizationExecutionError>
where
    TPool: Copy + PartialEq,
    TToken: Clone + Copy + PartialEq + Eq + Hash,
    WgpuRunner: FnOnce(
        Vec<PoolReserves<TPool, TToken>>,
        &OptimizationExecutionConfig<TToken>,
    ) -> Result<OptimizationExecutionResult, OptimizationExecutionError>,
    CpuRunner: FnOnce(
        Vec<PoolReserves<TPool, TToken>>,
        &OptimizationExecutionConfig<TToken>,
    ) -> Result<OptimizationExecutionResult, OptimizationExecutionError>,
{
    validate_execution_input(&reserves, config)?;

    match config.backend {
        OptimizationBackendSelection::Auto => match run_wgpu(reserves.clone(), config) {
            Err(OptimizationExecutionError::BackendFailed { .. }) => run_cpu(reserves, config),
            result => result,
        },
        OptimizationBackendSelection::Wgpu => run_wgpu(reserves, config),
        OptimizationBackendSelection::Cpu => run_cpu(reserves, config),
    }
}

fn validate_execution_input<TPool, TToken>(
    reserves: &[PoolReserves<TPool, TToken>],
    config: &OptimizationExecutionConfig<TToken>,
) -> Result<(), OptimizationExecutionError>
where
    TPool: Copy,
    TToken: Copy + PartialEq,
{
    if reserves.is_empty() {
        return Err(OptimizationExecutionError::EmptyReserves);
    }

    if !config.input_amount.is_finite() || config.input_amount <= 0.0 {
        return Err(OptimizationExecutionError::InvalidInputAmount {
            input_amount: config.input_amount,
        });
    }

    if config.iterations == 0 {
        return Err(OptimizationExecutionError::ZeroIterations);
    }

    if reserves
        .iter()
        .all(|reserve| reserve.token1 != config.init_asset)
    {
        return Err(OptimizationExecutionError::InitAssetOutputNotFound);
    }

    Ok(())
}

fn execute_cpu<TPool, TToken>(
    reserves: Vec<PoolReserves<TPool, TToken>>,
    config: &OptimizationExecutionConfig<TToken>,
) -> Result<OptimizationExecutionResult, OptimizationExecutionError>
where
    TPool: Copy + PartialEq,
    TToken: Clone + Copy + PartialEq + Eq + Hash,
{
    execute_backend_catching_panics(OptimizationBackendUsed::Cpu, || {
        execute_with_backend::<CpuBackend, TPool, TToken>(
            reserves,
            config,
            OptimizationBackendUsed::Cpu,
        )
    })
}

fn execute_wgpu<TPool, TToken>(
    reserves: Vec<PoolReserves<TPool, TToken>>,
    config: &OptimizationExecutionConfig<TToken>,
) -> Result<OptimizationExecutionResult, OptimizationExecutionError>
where
    TPool: Copy + PartialEq,
    TToken: Clone + Copy + PartialEq + Eq + Hash,
{
    execute_backend_catching_panics(OptimizationBackendUsed::Wgpu, || {
        execute_with_backend::<WgpuBackend, TPool, TToken>(
            reserves,
            config,
            OptimizationBackendUsed::Wgpu,
        )
    })
}

fn execute_backend_catching_panics<F>(
    backend: OptimizationBackendUsed,
    run: F,
) -> Result<OptimizationExecutionResult, OptimizationExecutionError>
where
    F: FnOnce() -> Result<OptimizationExecutionResult, OptimizationExecutionError>,
{
    match catch_unwind(AssertUnwindSafe(run)) {
        Ok(result) => result,
        Err(payload) => Err(OptimizationExecutionError::BackendFailed {
            backend,
            message: panic_message(payload),
        }),
    }
}

fn execute_with_backend<B, TPool, TToken>(
    reserves: Vec<PoolReserves<TPool, TToken>>,
    config: &OptimizationExecutionConfig<TToken>,
    backend: OptimizationBackendUsed,
) -> Result<OptimizationExecutionResult, OptimizationExecutionError>
where
    B: AutodiffBackend,
    TPool: Copy + PartialEq,
    TToken: Clone + Copy + PartialEq + Eq + Hash,
{
    let reserves_count = reserves.len();
    let model = Model::<B, TPool, TToken, OPTIMIZATION_LAYERS>::init(
        config.init_asset,
        reserves,
        &HashSet::new(),
    )
    .map_err(model_init_error)?;
    let input = B::FloatElem::from_elem(config.input_amount);
    let model = model.optimize(input, config.iterations);
    let output_amount = model
        .evaluate(B::FloatElem::from_elem(config.input_amount))
        .to_f32();

    Ok(OptimizationExecutionResult {
        backend,
        input_amount: config.input_amount,
        output_amount,
        profit_amount: output_amount - config.input_amount,
        reserves_count,
    })
}

fn model_init_error(source: OptimizationError) -> OptimizationExecutionError {
    OptimizationExecutionError::ModelInit { source }
}

fn panic_message(payload: Box<dyn Any + Send>) -> String {
    match payload.downcast_ref::<&str>() {
        Some(message) => (*message).to_owned(),
        None => match payload.downcast_ref::<String>() {
            Some(message) => message.clone(),
            None => "backend panicked".to_owned(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Invertible, OptimizationError, PoolReserves, VirtualReserveValues};

    const TOKEN_A: u8 = 1;
    const TOKEN_B: u8 = 2;
    const TOKEN_C: u8 = 3;

    #[test]
    fn cpu_execution_is_returned_as_result_for_tiny_directional_reserve_set() {
        let config = cpu_config(TOKEN_A);

        let result = execute_optimization(reserves(), &config);

        match result {
            Ok(result) => {
                assert_eq!(result.backend, OptimizationBackendUsed::Cpu);
                assert_eq!(result.input_amount, config.input_amount);
                assert_eq!(result.reserves_count, 2);
                assert!(result.output_amount.is_finite());
                assert!(result.profit_amount.is_finite());
            }
            Err(OptimizationExecutionError::BackendFailed { backend, message }) => {
                assert_eq!(backend, OptimizationBackendUsed::Cpu);
                assert!(!message.is_empty());
            }
            Err(error) => panic!("unexpected optimization execution error: {error}"),
        }
    }

    #[test]
    fn empty_reserves_return_typed_error() {
        let error = execute_optimization::<u8, u8>(Vec::new(), &cpu_config(TOKEN_A)).unwrap_err();

        assert_eq!(error, OptimizationExecutionError::EmptyReserves);
    }

    #[test]
    fn missing_init_asset_returns_typed_error_before_model_init() {
        let error = execute_optimization(reserves(), &cpu_config(TOKEN_C)).unwrap_err();

        assert_eq!(error, OptimizationExecutionError::InitAssetOutputNotFound);
    }

    #[test]
    fn invalid_input_amounts_return_typed_errors() {
        for input_amount in [0.0, -1.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
            let mut config = cpu_config(TOKEN_A);
            config.input_amount = input_amount;

            let error = execute_optimization(reserves(), &config).unwrap_err();

            assert!(matches!(
                error,
                OptimizationExecutionError::InvalidInputAmount { .. }
            ));
        }
    }

    #[test]
    fn zero_iterations_returns_typed_error() {
        let mut config = cpu_config(TOKEN_A);
        config.iterations = 0;

        let error = execute_optimization(reserves(), &config).unwrap_err();

        assert_eq!(error, OptimizationExecutionError::ZeroIterations);
    }

    #[test]
    fn model_init_errors_are_mapped_to_execution_errors() {
        let error = model_init_error(OptimizationError::InitAssetNotFound);

        assert_eq!(
            error,
            OptimizationExecutionError::ModelInit {
                source: OptimizationError::InitAssetNotFound,
            }
        );
    }

    #[test]
    fn auto_backend_falls_back_to_cpu_after_injected_wgpu_backend_failure() {
        let mut config = cpu_config(TOKEN_A);
        config.backend = OptimizationBackendSelection::Auto;

        let result = execute_optimization_with_backend_runners(
            reserves(),
            &config,
            |_reserves, _config| {
                Err(OptimizationExecutionError::BackendFailed {
                    backend: OptimizationBackendUsed::Wgpu,
                    message: "no adapter".to_owned(),
                })
            },
            |_reserves, config| Ok(fake_result(OptimizationBackendUsed::Cpu, config)),
        )
        .unwrap();

        assert_eq!(result.backend, OptimizationBackendUsed::Cpu);
    }

    #[test]
    fn explicit_wgpu_backend_returns_injected_backend_failure_without_cpu_fallback() {
        let mut config = cpu_config(TOKEN_A);
        config.backend = OptimizationBackendSelection::Wgpu;

        let error = execute_optimization_with_backend_runners(
            reserves(),
            &config,
            |_reserves, _config| {
                Err(OptimizationExecutionError::BackendFailed {
                    backend: OptimizationBackendUsed::Wgpu,
                    message: "no adapter".to_owned(),
                })
            },
            |_reserves, config| Ok(fake_result(OptimizationBackendUsed::Cpu, config)),
        )
        .unwrap_err();

        assert_eq!(
            error,
            OptimizationExecutionError::BackendFailed {
                backend: OptimizationBackendUsed::Wgpu,
                message: "no adapter".to_owned(),
            }
        );
    }

    fn cpu_config(init_asset: u8) -> OptimizationExecutionConfig<u8> {
        OptimizationExecutionConfig {
            backend: OptimizationBackendSelection::Cpu,
            init_asset,
            input_amount: 10.0,
            iterations: 1,
        }
    }

    fn fake_result(
        backend: OptimizationBackendUsed,
        config: &OptimizationExecutionConfig<u8>,
    ) -> OptimizationExecutionResult {
        OptimizationExecutionResult {
            backend,
            input_amount: config.input_amount,
            output_amount: config.input_amount,
            profit_amount: 0.0,
            reserves_count: 2,
        }
    }

    fn reserves() -> Vec<PoolReserves<u8, u8>> {
        let reserve = PoolReserves {
            token0: TOKEN_A,
            token1: TOKEN_B,
            pool_id: 1,
            value: VirtualReserveValues {
                token_0: 1_000.0,
                token_1: 1_000.0,
                fee_multiplier: 1.0,
                max_swap_0: 500.0,
                max_swap_1: 500.0,
            },
        };

        vec![reserve, reserve.inverse()]
    }
}
