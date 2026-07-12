use std::{collections::HashSet, hash::Hash};

use burn::tensor::{ElementConversion, backend::AutodiffBackend, cast::ToElement};
use thiserror::Error;

use crate::{OptimizationError, PoolReserves, model::Model, model::ModelOptimizer};

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

type CpuBackend = burn::backend::Autodiff<burn::backend::NdArray<f32>>;
type WgpuBackend = burn::backend::Autodiff<burn::backend::Wgpu<f32>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ReserveKey<TPool: Copy, TToken: Copy> {
    pool_id: TPool,
    token0: TToken,
    token1: TToken,
}

pub struct OptimizationRunner<
    TPool: Copy + PartialEq + Eq + Hash,
    TToken: Clone + Copy + PartialEq + Eq + Hash,
    const LAYERS: usize,
> {
    inner: OptimizationRunnerInner<TPool, TToken, LAYERS>,
}

enum OptimizationRunnerInner<
    TPool: Copy + PartialEq + Eq + Hash,
    TToken: Clone + Copy + PartialEq + Eq + Hash,
    const LAYERS: usize,
> {
    // Boxed to keep the two backend variants close in size (a `TypedOptimizationRunner` holds the
    // full tensor block) and to shrink the value moved on every `run` step.
    Wgpu(Box<TypedOptimizationRunner<WgpuBackend, TPool, TToken, LAYERS>>),
    Cpu(Box<TypedOptimizationRunner<CpuBackend, TPool, TToken, LAYERS>>),
}

struct TypedOptimizationRunner<
    B: AutodiffBackend,
    TPool: Copy + PartialEq + Eq + Hash,
    TToken: Clone + Copy + PartialEq + Eq + Hash,
    const LAYERS: usize,
> {
    session: OptimizationSession<B, TPool, TToken, LAYERS>,
    step_config: OptimizationStepConfig,
}

struct OptimizationSession<
    B: AutodiffBackend,
    TPool: Copy,
    TToken: Clone + Copy + PartialEq + Eq + Hash,
    const LAYERS: usize,
> {
    model: Model<B, TPool, TToken, LAYERS>,
    optimizer: ModelOptimizer<B, LAYERS>,
    session_config: OptimizationSessionConfig<TToken>,
    reserve_keys: HashSet<ReserveKey<TPool, TToken>>,
    // Pools currently masked out of routing, echoed on `Continued` steps for diagnostics. Set in
    // lockstep with the model's disabled mask, so it never drifts from the tensors.
    disabled_count: usize,
}

#[derive(Clone, Debug)]
pub struct OptimizationSessionConfig<TToken> {
    pub init_asset: TToken,
    pub bridges: HashSet<(TToken, TToken)>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OptimizationStepConfig {
    pub input_amount: f32,
    pub iterations: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub enum OptimizationStepUpdate<TPool: Copy + Eq + Hash, TToken: Copy> {
    /// A fresh reserve snapshot plus the set of pool ids to temporarily mask out of routing
    /// (e.g. pools on a chain that has fallen behind). Disabling a pool keeps it in the snapshot
    /// and the key set, so the model updates in place instead of reinitializing and its trained
    /// weights survive. Ids not present in the snapshot are ignored.
    NewReserves {
        reserves: Vec<PoolReserves<TPool, TToken>>,
        disabled: HashSet<TPool>,
    },
    Continue,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptimizationStepStatus {
    Initialized,
    Updated,
    Reinitialized,
    Continued,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OptimizationStepResult {
    pub status: OptimizationStepStatus,
    pub input_amount: f32,
    pub output_amount: f32,
    pub profit_amount: f32,
    pub reserves_count: usize,
    pub disabled_count: usize,
    pub iterations_completed: usize,
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum OptimizationInitError {
    #[error("wgpu backend unavailable")]
    WgpuUnavailable,

    #[error("{0}")]
    Step(OptimizationStepError),
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum OptimizationStepError {
    #[error("optimization reserves are empty")]
    EmptyReserves,

    #[error("duplicate reserve key")]
    DuplicateReserveKey,

    #[error("optimization input amount must be finite and greater than zero: {input_amount}")]
    InvalidInputAmount { input_amount: f32 },

    #[error("init asset output not found")]
    InitAssetOutputNotFound,

    #[error("optimization model init failed: {source}")]
    ModelInit { source: OptimizationError },
}

impl<TPool, TToken, const LAYERS: usize> OptimizationRunner<TPool, TToken, LAYERS>
where
    TPool: Copy + PartialEq + Eq + Hash,
    TToken: Clone + Copy + PartialEq + Eq + Hash,
{
    pub fn init(
        backend: OptimizationBackendSelection,
        reserves: Vec<PoolReserves<TPool, TToken>>,
        session_config: OptimizationSessionConfig<TToken>,
        step_config: OptimizationStepConfig,
    ) -> Result<
        (
            OptimizationRunner<TPool, TToken, LAYERS>,
            OptimizationStepResult,
        ),
        OptimizationInitError,
    > {
        let backend = match backend {
            OptimizationBackendSelection::Cpu => OptimizationBackendUsed::Cpu,
            selection => select_backend(selection, wgpu_backend_available())?,
        };

        match backend {
            OptimizationBackendUsed::Wgpu => {
                let (runner, result) =
                    TypedOptimizationRunner::<WgpuBackend, TPool, TToken, LAYERS>::init(
                        reserves,
                        session_config,
                        step_config,
                    )
                    .map_err(OptimizationInitError::Step)?;

                Ok((
                    OptimizationRunner {
                        inner: OptimizationRunnerInner::Wgpu(Box::new(runner)),
                    },
                    result,
                ))
            }
            OptimizationBackendUsed::Cpu => {
                let (runner, result) =
                    TypedOptimizationRunner::<CpuBackend, TPool, TToken, LAYERS>::init(
                        reserves,
                        session_config,
                        step_config,
                    )
                    .map_err(OptimizationInitError::Step)?;

                Ok((
                    OptimizationRunner {
                        inner: OptimizationRunnerInner::Cpu(Box::new(runner)),
                    },
                    result,
                ))
            }
        }
    }

    pub fn run(
        self,
        update: OptimizationStepUpdate<TPool, TToken>,
    ) -> Result<
        (
            OptimizationRunner<TPool, TToken, LAYERS>,
            OptimizationStepResult,
        ),
        OptimizationStepError,
    > {
        match self.inner {
            OptimizationRunnerInner::Wgpu(runner) => {
                let (runner, result) = runner.run(update)?;

                Ok((
                    OptimizationRunner {
                        inner: OptimizationRunnerInner::Wgpu(Box::new(runner)),
                    },
                    result,
                ))
            }
            OptimizationRunnerInner::Cpu(runner) => {
                let (runner, result) = runner.run(update)?;

                Ok((
                    OptimizationRunner {
                        inner: OptimizationRunnerInner::Cpu(Box::new(runner)),
                    },
                    result,
                ))
            }
        }
    }

    pub fn backend_used(&self) -> OptimizationBackendUsed {
        match self.inner {
            OptimizationRunnerInner::Wgpu(_) => OptimizationBackendUsed::Wgpu,
            OptimizationRunnerInner::Cpu(_) => OptimizationBackendUsed::Cpu,
        }
    }
}

impl<B, TPool, TToken, const LAYERS: usize> TypedOptimizationRunner<B, TPool, TToken, LAYERS>
where
    B: AutodiffBackend,
    TPool: Copy + PartialEq + Eq + Hash,
    TToken: Clone + Copy + PartialEq + Eq + Hash,
{
    fn init(
        reserves: Vec<PoolReserves<TPool, TToken>>,
        session_config: OptimizationSessionConfig<TToken>,
        step_config: OptimizationStepConfig,
    ) -> Result<
        (
            TypedOptimizationRunner<B, TPool, TToken, LAYERS>,
            OptimizationStepResult,
        ),
        OptimizationStepError,
    > {
        let (session, result) = initialize_optimization_session::<B, TPool, TToken, LAYERS>(
            reserves,
            session_config,
            &step_config,
        )?;

        Ok((
            TypedOptimizationRunner {
                session,
                step_config,
            },
            result,
        ))
    }

    fn run(
        self,
        update: OptimizationStepUpdate<TPool, TToken>,
    ) -> Result<
        (
            TypedOptimizationRunner<B, TPool, TToken, LAYERS>,
            OptimizationStepResult,
        ),
        OptimizationStepError,
    > {
        let (session, result) = run_optimization_step(self.session, update, &self.step_config)?;

        Ok((
            TypedOptimizationRunner {
                session,
                step_config: self.step_config,
            },
            result,
        ))
    }
}

fn select_backend(
    selection: OptimizationBackendSelection,
    wgpu_available: bool,
) -> Result<OptimizationBackendUsed, OptimizationInitError> {
    match selection {
        OptimizationBackendSelection::Auto if wgpu_available => Ok(OptimizationBackendUsed::Wgpu),
        OptimizationBackendSelection::Auto => Ok(OptimizationBackendUsed::Cpu),
        OptimizationBackendSelection::Wgpu if wgpu_available => Ok(OptimizationBackendUsed::Wgpu),
        OptimizationBackendSelection::Wgpu => Err(OptimizationInitError::WgpuUnavailable),
        OptimizationBackendSelection::Cpu => Ok(OptimizationBackendUsed::Cpu),
    }
}

fn wgpu_backend_available() -> bool {
    static AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        // cubecl raises a panic when no GPU adapter is present. Silence the default hook for the
        // duration of the probe so the expected, handled "No possible adapter available" message
        // does not pollute the log, then restore it. Memoized in a `OnceLock` so the global
        // hook swap happens exactly once per process and cannot race other threads.
        let previous_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let available = std::panic::catch_unwind(|| {
            let device = burn::backend::wgpu::WgpuDevice::default();
            let _tensor = burn::tensor::Tensor::<WgpuBackend, 1>::zeros([1], &device);
        })
        .is_ok();
        std::panic::set_hook(previous_hook);
        available
    })
}

fn initialize_optimization_session<B, TPool, TToken, const LAYERS: usize>(
    reserves: Vec<PoolReserves<TPool, TToken>>,
    session_config: OptimizationSessionConfig<TToken>,
    step_config: &OptimizationStepConfig,
) -> Result<
    (
        OptimizationSession<B, TPool, TToken, LAYERS>,
        OptimizationStepResult,
    ),
    OptimizationStepError,
>
where
    B: AutodiffBackend,
    TPool: Copy + PartialEq + Eq + Hash,
    TToken: Clone + Copy + PartialEq + Eq + Hash,
{
    let reserves = crate::routing_filter::routable_reserves(
        reserves,
        session_config.init_asset,
        &session_config.bridges,
    );
    // First init has no disable signal — that arrives with `NewReserves` updates once the session
    // is running — so it starts with every pool active.
    initialize_optimization_session_with_status(
        reserves,
        session_config,
        step_config,
        OptimizationStepStatus::Initialized,
        &HashSet::new(),
    )
}

fn run_optimization_step<B, TPool, TToken, const LAYERS: usize>(
    session: OptimizationSession<B, TPool, TToken, LAYERS>,
    update: OptimizationStepUpdate<TPool, TToken>,
    step_config: &OptimizationStepConfig,
) -> Result<
    (
        OptimizationSession<B, TPool, TToken, LAYERS>,
        OptimizationStepResult,
    ),
    OptimizationStepError,
>
where
    B: AutodiffBackend,
    TPool: Copy + PartialEq + Eq + Hash,
    TToken: Clone + Copy + PartialEq + Eq + Hash,
{
    match update {
        OptimizationStepUpdate::Continue => {
            run_optimization_chunk(session, step_config, OptimizationStepStatus::Continued)
        }
        OptimizationStepUpdate::NewReserves { reserves, disabled } => {
            run_optimization_step_with_reserves(session, reserves, &disabled, step_config)
        }
    }
}

fn run_optimization_step_with_reserves<B, TPool, TToken, const LAYERS: usize>(
    session: OptimizationSession<B, TPool, TToken, LAYERS>,
    reserves: Vec<PoolReserves<TPool, TToken>>,
    disabled: &HashSet<TPool>,
    step_config: &OptimizationStepConfig,
) -> Result<
    (
        OptimizationSession<B, TPool, TToken, LAYERS>,
        OptimizationStepResult,
    ),
    OptimizationStepError,
>
where
    B: AutodiffBackend,
    TPool: Copy + PartialEq + Eq + Hash,
    TToken: Clone + Copy + PartialEq + Eq + Hash,
{
    let reserves = crate::routing_filter::routable_reserves(
        reserves,
        session.session_config.init_asset,
        &session.session_config.bridges,
    );
    let incoming_keys = validate_reserve_snapshot(&reserves, &session.session_config, step_config)?;
    let OptimizationSession {
        model,
        optimizer,
        session_config,
        reserve_keys,
        ..
    } = session;

    if incoming_keys != reserve_keys {
        return initialize_optimization_session_with_status(
            reserves,
            session_config,
            step_config,
            OptimizationStepStatus::Reinitialized,
            disabled,
        );
    }

    match model.update(reserves.clone(), disabled) {
        Ok(model) => run_optimization_chunk(
            OptimizationSession {
                model,
                optimizer,
                session_config,
                reserve_keys: incoming_keys,
                disabled_count: effective_disabled_count(&reserves, disabled),
            },
            step_config,
            OptimizationStepStatus::Updated,
        ),
        Err(_) => initialize_optimization_session_with_status(
            reserves,
            session_config,
            step_config,
            OptimizationStepStatus::Reinitialized,
            disabled,
        ),
    }
}

/// Number of `disabled` ids that actually appear as a pool in `reserves` — the pools the mask
/// truly affects. Ids the routing filter already pruned (or that were never present) are excluded
/// so the diagnostic reflects reality rather than the raw request.
fn effective_disabled_count<TPool, TToken>(
    reserves: &[PoolReserves<TPool, TToken>],
    disabled: &HashSet<TPool>,
) -> usize
where
    TPool: Copy + Eq + Hash,
    TToken: Copy,
{
    reserves
        .iter()
        .map(|reserve| reserve.pool_id)
        .collect::<HashSet<_>>()
        .intersection(disabled)
        .count()
}

fn initialize_optimization_session_with_status<B, TPool, TToken, const LAYERS: usize>(
    reserves: Vec<PoolReserves<TPool, TToken>>,
    session_config: OptimizationSessionConfig<TToken>,
    step_config: &OptimizationStepConfig,
    status: OptimizationStepStatus,
    disabled: &HashSet<TPool>,
) -> Result<
    (
        OptimizationSession<B, TPool, TToken, LAYERS>,
        OptimizationStepResult,
    ),
    OptimizationStepError,
>
where
    B: AutodiffBackend,
    TPool: Copy + PartialEq + Eq + Hash,
    TToken: Clone + Copy + PartialEq + Eq + Hash,
{
    let reserve_keys = validate_reserve_snapshot(&reserves, &session_config, step_config)?;
    let disabled_count = effective_disabled_count(&reserves, disabled);
    let model = Model::<B, TPool, TToken, LAYERS>::init(
        session_config.init_asset,
        reserves,
        &session_config.bridges,
        disabled,
    )
    .map_err(|source| OptimizationStepError::ModelInit { source })?;
    let optimizer = Model::<B, TPool, TToken, LAYERS>::init_optimizer();

    run_optimization_chunk(
        OptimizationSession {
            model,
            optimizer,
            session_config,
            reserve_keys,
            disabled_count,
        },
        step_config,
        status,
    )
}

fn run_optimization_chunk<B, TPool, TToken, const LAYERS: usize>(
    session: OptimizationSession<B, TPool, TToken, LAYERS>,
    step_config: &OptimizationStepConfig,
    status: OptimizationStepStatus,
) -> Result<
    (
        OptimizationSession<B, TPool, TToken, LAYERS>,
        OptimizationStepResult,
    ),
    OptimizationStepError,
>
where
    B: AutodiffBackend,
    TPool: Copy + PartialEq + Eq + Hash,
    TToken: Clone + Copy + PartialEq + Eq + Hash,
{
    validate_step_config(step_config)?;

    let OptimizationSession {
        model,
        optimizer,
        session_config,
        reserve_keys,
        disabled_count,
    } = session;
    let reserves_count = reserve_keys.len();
    let (model, optimizer) = model.optimize_with(
        optimizer,
        B::FloatElem::from_elem(step_config.input_amount),
        step_config.iterations,
    );
    let output_amount = model
        .evaluate(B::FloatElem::from_elem(step_config.input_amount))
        .to_f32();
    let result = OptimizationStepResult {
        status,
        input_amount: step_config.input_amount,
        output_amount,
        profit_amount: output_amount - step_config.input_amount,
        reserves_count,
        disabled_count,
        iterations_completed: step_config.iterations,
    };

    Ok((
        OptimizationSession {
            model,
            optimizer,
            session_config,
            reserve_keys,
            disabled_count,
        },
        result,
    ))
}

/// Whether `reserves` can reach the session's init asset — i.e. whether feeding them to a step would
/// avoid the transient [`OptimizationStepError::InitAssetOutputNotFound`].
///
/// With cross-chain merging a merged snapshot can momentarily fail to reach the init asset (the chain
/// that owns the quote token reported late, a brief bootstrap/refresh gap, etc.). Initialization
/// already treats that as "not ready yet" and skips the snapshot; a running session must do the same,
/// because applying such a snapshot aborts the whole optimization worker. This mirrors the routing +
/// init-asset reachability check performed inside [`validate_reserve_snapshot`].
pub fn reserves_reach_init_asset<TPool, TToken>(
    reserves: &[PoolReserves<TPool, TToken>],
    session_config: &OptimizationSessionConfig<TToken>,
) -> bool
where
    TPool: Copy + Eq + Hash,
    TToken: Copy + Eq + Hash,
{
    crate::routing_filter::routable_reserves(
        reserves.to_vec(),
        session_config.init_asset,
        &session_config.bridges,
    )
    .iter()
    .any(|reserve| reserve.token1 == session_config.init_asset)
}

fn validate_reserve_snapshot<TPool, TToken>(
    reserves: &[PoolReserves<TPool, TToken>],
    session_config: &OptimizationSessionConfig<TToken>,
    step_config: &OptimizationStepConfig,
) -> Result<HashSet<ReserveKey<TPool, TToken>>, OptimizationStepError>
where
    TPool: Copy + Eq + Hash,
    TToken: Copy + Eq + Hash,
{
    validate_step_config(step_config)?;

    if reserves.is_empty() {
        return Err(OptimizationStepError::EmptyReserves);
    }

    let reserve_keys = reserve_keys(reserves)?;

    if reserves
        .iter()
        .all(|reserve| reserve.token1 != session_config.init_asset)
    {
        return Err(OptimizationStepError::InitAssetOutputNotFound);
    }

    Ok(reserve_keys)
}

fn validate_step_config(step_config: &OptimizationStepConfig) -> Result<(), OptimizationStepError> {
    if !step_config.input_amount.is_finite() || step_config.input_amount <= 0.0 {
        return Err(OptimizationStepError::InvalidInputAmount {
            input_amount: step_config.input_amount,
        });
    }

    Ok(())
}

fn reserve_keys<TPool, TToken>(
    reserves: &[PoolReserves<TPool, TToken>],
) -> Result<HashSet<ReserveKey<TPool, TToken>>, OptimizationStepError>
where
    TPool: Copy + Eq + Hash,
    TToken: Copy + Eq + Hash,
{
    reserves.iter().try_fold(
        HashSet::with_capacity(reserves.len()),
        |mut keys, reserve| {
            if keys.insert(reserve_key(reserve)) {
                Ok(keys)
            } else {
                Err(OptimizationStepError::DuplicateReserveKey)
            }
        },
    )
}

fn reserve_key<TPool: Copy, TToken: Copy>(
    reserve: &PoolReserves<TPool, TToken>,
) -> ReserveKey<TPool, TToken> {
    ReserveKey {
        pool_id: reserve.pool_id,
        token0: reserve.token0,
        token1: reserve.token1,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::{
        Invertible, PoolReserves, VirtualReserveValues,
        tokens::test::{self as tokens, TokenAddress},
    };

    use super::*;

    type CpuBackend = burn::backend::Autodiff<burn::backend::NdArray<f32>>;

    #[test]
    fn optimization_runner_cpu_init_returns_initialized_result() {
        let (runner, result) = OptimizationRunner::<i32, TokenAddress, 1>::init(
            OptimizationBackendSelection::Cpu,
            base_reserves(),
            session_config(),
            step_config(0),
        )
        .unwrap();

        assert_eq!(runner.backend_used(), OptimizationBackendUsed::Cpu);
        assert_eq!(result.status, OptimizationStepStatus::Initialized);
        assert_eq!(result.reserves_count, 2);
    }

    #[test]
    fn optimization_runner_cpu_continue_returns_continued_result() {
        let (runner, _) = initialized_cpu_runner();

        let (_runner, result) = runner.run(OptimizationStepUpdate::Continue).unwrap();

        assert_eq!(result.status, OptimizationStepStatus::Continued);
        assert_eq!(result.reserves_count, 2);
    }

    #[test]
    fn optimization_runner_cpu_same_keys_update_existing_runner() {
        let (runner, _) = initialized_cpu_runner();

        let (_runner, result) = runner
            .run(OptimizationStepUpdate::NewReserves {
                reserves: scaled_base_reserves(1.01),
                disabled: HashSet::new(),
            })
            .unwrap();

        assert_eq!(result.status, OptimizationStepStatus::Updated);
        assert_eq!(result.reserves_count, 2);
    }

    #[test]
    fn optimization_runner_cpu_changed_keys_reinitialize_runner() {
        let (runner, _) = initialized_cpu_runner();

        let (_runner, result) = runner
            .run(OptimizationStepUpdate::NewReserves {
                reserves: expanded_reserves(),
                disabled: HashSet::new(),
            })
            .unwrap();

        assert_eq!(result.status, OptimizationStepStatus::Reinitialized);
        assert_eq!(result.reserves_count, 3);
    }

    #[test]
    fn optimization_runner_empty_reserves_return_init_step_error() {
        let error = expect_init_error(OptimizationRunner::<i32, TokenAddress, 1>::init(
            OptimizationBackendSelection::Cpu,
            Vec::new(),
            session_config(),
            step_config(0),
        ));

        assert_eq!(
            error,
            OptimizationInitError::Step(OptimizationStepError::EmptyReserves)
        );
    }

    #[test]
    fn auto_backend_choice_prefers_wgpu_when_available() {
        assert_eq!(
            select_backend(OptimizationBackendSelection::Auto, true),
            Ok(OptimizationBackendUsed::Wgpu)
        );
    }

    #[test]
    fn auto_backend_choice_falls_back_to_cpu_when_wgpu_is_unavailable() {
        assert_eq!(
            select_backend(OptimizationBackendSelection::Auto, false),
            Ok(OptimizationBackendUsed::Cpu)
        );
    }

    #[test]
    fn forced_wgpu_backend_choice_requires_wgpu() {
        assert_eq!(
            select_backend(OptimizationBackendSelection::Wgpu, false),
            Err(OptimizationInitError::WgpuUnavailable)
        );
    }

    #[test]
    fn initialize_session_returns_initialized_result() {
        let (_session, result) =
            initialize_optimization_session::<CpuBackend, i32, TokenAddress, 1>(
                base_reserves(),
                session_config(),
                &step_config(0),
            )
            .unwrap();

        assert_eq!(result.status, OptimizationStepStatus::Initialized);
        assert_eq!(result.reserves_count, 2);
        assert_eq!(result.iterations_completed, 0);
        assert!(result.output_amount.is_finite());
        assert!(result.profit_amount.is_finite());
    }

    #[test]
    fn new_reserves_with_same_key_set_updates_existing_session() {
        let (session, _) = initialized_session();

        let (_session, result) = run_optimization_step(
            session,
            OptimizationStepUpdate::NewReserves {
                reserves: scaled_base_reserves(1.01),
                disabled: HashSet::new(),
            },
            &step_config(0),
        )
        .unwrap();

        assert_eq!(result.status, OptimizationStepStatus::Updated);
        assert_eq!(result.reserves_count, 2);
    }

    #[test]
    fn new_reserves_with_same_keys_in_different_order_updates_existing_session() {
        let (session, _) = initialized_session();
        let mut reserves = scaled_base_reserves(1.01);
        reserves.reverse();

        let (_session, result) = run_optimization_step(
            session,
            OptimizationStepUpdate::NewReserves {
                reserves,
                disabled: HashSet::new(),
            },
            &step_config(0),
        )
        .unwrap();

        assert_eq!(result.status, OptimizationStepStatus::Updated);
        assert_eq!(result.reserves_count, 2);
    }

    #[test]
    fn added_reserve_key_reinitializes_session() {
        let (session, _) = initialized_session();

        let (_session, result) = run_optimization_step(
            session,
            OptimizationStepUpdate::NewReserves {
                reserves: expanded_reserves(),
                disabled: HashSet::new(),
            },
            &step_config(0),
        )
        .unwrap();

        assert_eq!(result.status, OptimizationStepStatus::Reinitialized);
        assert_eq!(result.reserves_count, 3);
    }

    #[test]
    fn removed_reserve_key_reinitializes_session() {
        let (session, _) = initialize_optimization_session::<CpuBackend, i32, TokenAddress, 1>(
            expanded_reserves(),
            session_config(),
            &step_config(0),
        )
        .unwrap();

        let (_session, result) = run_optimization_step(
            session,
            OptimizationStepUpdate::NewReserves {
                reserves: base_reserves(),
                disabled: HashSet::new(),
            },
            &step_config(0),
        )
        .unwrap();

        assert_eq!(result.status, OptimizationStepStatus::Reinitialized);
        assert_eq!(result.reserves_count, 2);
    }

    #[test]
    fn reserve_not_found_during_update_reinitializes_session() {
        let (mut session, _) = initialized_session();
        let replacement_reserves = vec![reserve(
            99,
            tokens::WETH.address,
            tokens::USDC.address,
            2_000.0,
        )];
        session.reserve_keys = HashSet::from([ReserveKey {
            pool_id: 99,
            token0: tokens::WETH.address,
            token1: tokens::USDC.address,
        }]);

        let (_session, result) = run_optimization_step(
            session,
            OptimizationStepUpdate::NewReserves {
                reserves: replacement_reserves,
                disabled: HashSet::new(),
            },
            &step_config(0),
        )
        .unwrap();

        assert_eq!(result.status, OptimizationStepStatus::Reinitialized);
        assert_eq!(result.reserves_count, 1);
    }

    #[test]
    fn disabling_a_pool_keeps_key_set_and_updates_without_reinitializing() {
        let (session, _) = initialized_session();

        let (_session, result) = run_optimization_step(
            session,
            OptimizationStepUpdate::NewReserves {
                reserves: base_reserves(),
                disabled: HashSet::from([1]),
            },
            &step_config(0),
        )
        .unwrap();

        // Disabling a pool does not change the reserve key set, so the session updates in place
        // (weights preserved) rather than reinitializing.
        assert_eq!(result.status, OptimizationStepStatus::Updated);
        assert_eq!(result.reserves_count, 2);
        assert_eq!(result.disabled_count, 1);
    }

    #[test]
    fn disabled_id_absent_from_snapshot_is_ignored() {
        let (session, _) = initialized_session();

        let (_session, result) = run_optimization_step(
            session,
            OptimizationStepUpdate::NewReserves {
                reserves: base_reserves(),
                disabled: HashSet::from([999]),
            },
            &step_config(0),
        )
        .unwrap();

        // A disabled id no reserve carries (e.g. routing-filter-pruned upstream) is a normal
        // state, not an error: the step still succeeds and nothing is counted as disabled.
        assert_eq!(result.status, OptimizationStepStatus::Updated);
        assert_eq!(result.disabled_count, 0);
    }

    #[test]
    fn reinitialization_carries_the_disabled_set() {
        let (session, _) = initialized_session();

        // A changed key set forces reinitialization; the disable request must still be applied to
        // the freshly built model, and reflected in the diagnostic count.
        let (_session, result) = run_optimization_step(
            session,
            OptimizationStepUpdate::NewReserves {
                reserves: expanded_reserves(),
                disabled: HashSet::from([2]),
            },
            &step_config(0),
        )
        .unwrap();

        assert_eq!(result.status, OptimizationStepStatus::Reinitialized);
        assert_eq!(result.reserves_count, 3);
        assert_eq!(result.disabled_count, 1);
    }

    #[test]
    fn continue_runs_existing_session_without_reserve_update() {
        let (session, _) = initialized_session();

        let (_session, result) =
            run_optimization_step(session, OptimizationStepUpdate::Continue, &step_config(0))
                .unwrap();

        assert_eq!(result.status, OptimizationStepStatus::Continued);
        assert_eq!(result.reserves_count, 2);
        assert!(result.output_amount.is_finite());
        assert!(result.profit_amount.is_finite());
    }

    #[test]
    fn empty_reserves_return_typed_error() {
        let error = expect_step_error(initialize_optimization_session::<
            CpuBackend,
            i32,
            TokenAddress,
            1,
        >(Vec::new(), session_config(), &step_config(0)));

        assert_eq!(error, OptimizationStepError::EmptyReserves);
    }

    #[test]
    fn duplicate_reserve_keys_return_typed_error() {
        let reserve = reserve(1, tokens::USDC.address, tokens::WETH.address, 1_000.0);

        let error = expect_step_error(initialize_optimization_session::<
            CpuBackend,
            i32,
            TokenAddress,
            1,
        >(
            vec![reserve, reserve], session_config(), &step_config(0)
        ));

        assert_eq!(error, OptimizationStepError::DuplicateReserveKey);
    }

    #[test]
    fn invalid_input_amounts_return_typed_errors() {
        for input_amount in [0.0, -1.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
            let error = expect_step_error(initialize_optimization_session::<
                CpuBackend,
                i32,
                TokenAddress,
                1,
            >(
                base_reserves(),
                session_config(),
                &OptimizationStepConfig {
                    input_amount,
                    iterations: 0,
                },
            ));

            assert!(matches!(
                error,
                OptimizationStepError::InvalidInputAmount { .. }
            ));
        }
    }

    #[test]
    fn missing_init_asset_output_returns_typed_error() {
        let error = expect_step_error(initialize_optimization_session::<
            CpuBackend,
            i32,
            TokenAddress,
            1,
        >(
            vec![reserve(
                1,
                tokens::USDC.address,
                tokens::WETH.address,
                1_000.0,
            )],
            session_config(),
            &step_config(0),
        ));

        assert_eq!(error, OptimizationStepError::InitAssetOutputNotFound);
    }

    #[test]
    fn reserves_reach_init_asset_is_true_when_a_pool_outputs_the_init_asset() {
        assert!(reserves_reach_init_asset(&base_reserves(), &session_config()));
    }

    #[test]
    fn reserves_reach_init_asset_is_false_when_no_pool_outputs_the_init_asset() {
        // Same snapshot that drives `missing_init_asset_output_returns_typed_error`: a step fed this
        // would abort with `InitAssetOutputNotFound`, so the predicate must flag it as not-yet-ready.
        let reserves = vec![reserve(1, tokens::USDC.address, tokens::WETH.address, 1_000.0)];

        assert!(!reserves_reach_init_asset(&reserves, &session_config()));
    }

    #[test]
    fn reserves_reach_init_asset_is_false_for_an_empty_snapshot() {
        assert!(!reserves_reach_init_asset(
            &Vec::<PoolReserves<i32, TokenAddress>>::new(),
            &session_config(),
        ));
    }

    fn initialized_session() -> (
        OptimizationSession<CpuBackend, i32, TokenAddress, 1>,
        OptimizationStepResult,
    ) {
        initialize_optimization_session(base_reserves(), session_config(), &step_config(0)).unwrap()
    }

    fn initialized_cpu_runner() -> (
        OptimizationRunner<i32, TokenAddress, 1>,
        OptimizationStepResult,
    ) {
        OptimizationRunner::init(
            OptimizationBackendSelection::Cpu,
            base_reserves(),
            session_config(),
            step_config(0),
        )
        .unwrap()
    }

    fn expect_init_error(
        result: Result<
            (
                OptimizationRunner<i32, TokenAddress, 1>,
                OptimizationStepResult,
            ),
            OptimizationInitError,
        >,
    ) -> OptimizationInitError {
        match result {
            Ok(_) => panic!("optimization runner init unexpectedly succeeded"),
            Err(error) => error,
        }
    }

    fn expect_step_error(
        result: Result<
            (
                OptimizationSession<CpuBackend, i32, TokenAddress, 1>,
                OptimizationStepResult,
            ),
            OptimizationStepError,
        >,
    ) -> OptimizationStepError {
        match result {
            Ok(_) => panic!("optimization step unexpectedly succeeded"),
            Err(error) => error,
        }
    }

    fn session_config() -> OptimizationSessionConfig<TokenAddress> {
        OptimizationSessionConfig {
            init_asset: tokens::USDC.address,
            bridges: HashSet::new(),
        }
    }

    fn step_config(iterations: usize) -> OptimizationStepConfig {
        OptimizationStepConfig {
            input_amount: 100.0,
            iterations,
        }
    }

    fn base_reserves() -> Vec<PoolReserves<i32, TokenAddress>> {
        let reserve = reserve(1, tokens::USDC.address, tokens::WETH.address, 1_000.0);

        vec![reserve, reserve.inverse()]
    }

    fn scaled_base_reserves(scale: f32) -> Vec<PoolReserves<i32, TokenAddress>> {
        base_reserves()
            .into_iter()
            .map(|mut reserve| {
                reserve.value.token_0 *= scale;
                reserve.value.token_1 *= scale;
                reserve.value.max_swap_0 *= scale;
                reserve.value.max_swap_1 *= scale;
                reserve
            })
            .collect()
    }

    fn expanded_reserves() -> Vec<PoolReserves<i32, TokenAddress>> {
        let mut reserves = base_reserves();
        reserves.push(reserve(
            2,
            tokens::WETH.address,
            tokens::WBTC.address,
            2_000.0,
        ));
        reserves
    }

    fn reserve(
        pool_id: i32,
        token0: TokenAddress,
        token1: TokenAddress,
        amount: f32,
    ) -> PoolReserves<i32, TokenAddress> {
        PoolReserves {
            token0,
            token1,
            pool_id,
            value: VirtualReserveValues {
                token_0: amount,
                token_1: amount * 1.5,
                fee_multiplier: 0.997,
                max_swap_0: amount * 0.5,
                max_swap_1: amount * 0.75,
            },
        }
    }

    /// A USDC/WETH/WBTC triangle (all tokens degree 2): a genuine 2-core that
    /// survives routing-filter pruning.
    fn triangle_reserves() -> Vec<PoolReserves<i32, TokenAddress>> {
        let usdc = tokens::USDC.address;
        let weth = tokens::WETH.address;
        let wbtc = tokens::WBTC.address;
        let edges = [
            reserve(1, usdc, weth, 1_000.0),
            reserve(2, weth, wbtc, 1_000.0),
            reserve(3, wbtc, usdc, 1_000.0),
        ];
        edges
            .into_iter()
            .flat_map(|edge| [edge, edge.inverse()])
            .collect()
    }

    /// The triangle plus an unroutable leaf pool (USDC/UNI, UNI is degree 1).
    fn triangle_with_leaf_reserves() -> Vec<PoolReserves<i32, TokenAddress>> {
        let leaf = reserve(4, tokens::USDC.address, tokens::UNI.address, 500.0);
        let mut reserves = triangle_reserves();
        reserves.extend([leaf, leaf.inverse()]);
        reserves
    }

    fn scaled(
        reserves: Vec<PoolReserves<i32, TokenAddress>>,
        scale: f32,
    ) -> Vec<PoolReserves<i32, TokenAddress>> {
        reserves
            .into_iter()
            .map(|mut reserve| {
                reserve.value.token_0 *= scale;
                reserve.value.token_1 *= scale;
                reserve.value.max_swap_0 *= scale;
                reserve.value.max_swap_1 *= scale;
                reserve
            })
            .collect()
    }

    #[test]
    fn unroutable_leaf_pools_are_pruned_before_optimization() {
        let (_session, result) =
            initialize_optimization_session::<CpuBackend, i32, TokenAddress, 1>(
                triangle_with_leaf_reserves(),
                session_config(),
                &step_config(0),
            )
            .unwrap();

        // 8 directional reserves in, the 2 leaf reserves pruned, 6 kept.
        assert_eq!(result.reserves_count, 6);
    }

    #[test]
    fn stable_topology_with_unroutable_pools_keeps_updating() {
        let (session, init_result) =
            initialize_optimization_session::<CpuBackend, i32, TokenAddress, 1>(
                triangle_with_leaf_reserves(),
                session_config(),
                &step_config(0),
            )
            .unwrap();
        assert_eq!(init_result.status, OptimizationStepStatus::Initialized);
        assert_eq!(init_result.reserves_count, 6);

        // The next snapshot carries the same (leaf-padded) topology with only value
        // changes. Pruning is deterministic, so the surviving keys match and the
        // session updates in place instead of reinitializing.
        let (_session, result) = run_optimization_step(
            session,
            OptimizationStepUpdate::NewReserves {
                reserves: scaled(triangle_with_leaf_reserves(), 1.01),
                disabled: HashSet::new(),
            },
            &step_config(0),
        )
        .unwrap();

        assert_eq!(result.status, OptimizationStepStatus::Updated);
        assert_eq!(result.reserves_count, 6);
    }
}
