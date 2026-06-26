use std::{
    sync::{
        Arc,
        mpsc::{Receiver, Sender},
    },
    thread::JoinHandle,
};

use rayon::prelude::*;

/// Result of applying one input to one state.
///
/// The framework treats effects as declarative values. Runtime code outside the
/// pure core decides how to execute them.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Transition<State, Effect> {
    pub state: State,
    pub effects: Vec<Effect>,
}

#[derive(Debug)]
pub enum ApplicationError<T> {
    SendError(std::sync::mpsc::SendError<T>),
}

/// Domain-agnostic contract for a pure deterministic application core.
///
/// Implementations should be total and deterministic. They should not perform
/// I/O, access clocks, generate randomness, mutate external state, or execute
/// effects.
pub trait Application {
    type State;
    type Input: Send + 'static;
    type Effect: Send + 'static;
    type Subscription: Send + 'static;

    fn init() -> Transition<Self::State, Self::Effect>;

    fn transition(state: Self::State, input: Self::Input) -> Transition<Self::State, Self::Effect>;

    fn subscriptions() -> Vec<Self::Subscription>;
}

pub trait Runtime<App: Application>: Sized + Send + Sync + 'static {
    fn execute_effect(&self, effect: App::Effect) -> Vec<App::Input>;
    fn spawn_subscription(&self, sender: &Sender<App::Input>, subscription: App::Subscription);

    /// Worker count for the dedicated blocking-I/O pool that runs effects. The default suits
    /// CPU-bound effects; runtimes whose effects block on network I/O override this with a count
    /// sized to their concurrency budget (i.e. their aggregate provider rate limit), not the CPU
    /// count. This number is effectively the runtime's global cap on concurrent in-flight effects.
    fn effect_pool_size(&self) -> usize {
        8
    }

    fn log_input(&self, _input: &App::Input) {}
    fn log_error(&self, _error: ApplicationError<App::Input>) {}
    fn observe_state(&self, _state: &App::State) {}

    fn run(self) -> (Sender<App::Input>, JoinHandle<()>)
    where
        App: 'static,
    {
        let runtime = Arc::new(self);
        let (input_sender, input_receiver) = std::sync::mpsc::channel();
        let runtime_sender = input_sender.clone();
        let handle = std::thread::spawn(move || {
            run::<App, Self>(
                runtime,
                runtime_sender,
                input_receiver,
                <App as Application>::subscriptions(),
            )
        });
        (input_sender, handle)
    }
}

fn run<App: Application, R: Runtime<App>>(
    runtime: Arc<R>,
    input_sender: Sender<App::Input>,
    input_receiver: Receiver<App::Input>,
    subscriptions: Vec<App::Subscription>,
) -> () {
    let Transition { state, effects } = App::init();

    // One dedicated pool for the blocking effect work, sized for I/O rather than the CPU count of
    // rayon's global pool. Its width is the runtime's global cap on concurrent in-flight effects, so
    // a chain that falls behind can no longer pin every worker and starve the others. If the pool
    // ever fails to build, fall back to the global pool (`None`) rather than failing to start.
    let effect_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(runtime.effect_pool_size())
        .thread_name(|index| format!("effect-io-{index}"))
        .build()
        .map(Arc::new)
        .ok();

    drop(spawn_effects::<App, R>(
        runtime.clone(),
        effect_pool.clone(),
        &input_sender,
        effects,
    ));

    drop(spawn_subscriptions(
        runtime.clone(),
        &input_sender,
        subscriptions,
    ));

    let _final_state = input_receiver.into_iter().fold(state, |s, i| {
        runtime.log_input(&i);
        let Transition {
            state: next_state,
            effects,
        } = App::transition(s, i);

        runtime.observe_state(&next_state);

        // spawn a thread to execute effects in parallel
        drop(spawn_effects::<App, R>(
            runtime.clone(),
            effect_pool.clone(),
            &input_sender,
            effects,
        ));

        next_state
    });

    ()
}

fn spawn_effects<App: Application, R: Runtime<App>>(
    runtime: Arc<R>,
    effect_pool: Option<Arc<rayon::ThreadPool>>,
    sender: &Sender<App::Input>,
    effects: Vec<App::Effect>,
) -> JoinHandle<()> {
    let sender_clone = sender.clone();
    std::thread::spawn(move || {
        let run_effects = move || {
            effects.into_par_iter().for_each(|effect| {
                runtime
                    .execute_effect(effect)
                    .into_iter()
                    .for_each(|i| match sender_clone.send(i) {
                        Ok(_) => (),
                        Err(e) => runtime.log_error(ApplicationError::SendError(e)),
                    })
            });
        };

        // Run the blocking effect work on the dedicated I/O pool rather than rayon's global,
        // CPU-sized pool, so effects that block on network I/O get a worker each instead of
        // queueing behind the CPU count. Fall back to the global pool if no dedicated pool exists.
        match effect_pool {
            Some(pool) => pool.install(run_effects),
            None => run_effects(),
        }
    })
}

fn spawn_subscriptions<App: Application, R: Runtime<App>>(
    runtime: Arc<R>,
    sender: &Sender<App::Input>,
    subscriptions: Vec<App::Subscription>,
) -> Vec<JoinHandle<()>> {
    subscriptions
        .into_iter()
        .map(|subscription| {
            let runtime = runtime.clone();
            let sender = sender.clone();

            std::thread::spawn(move || {
                runtime.spawn_subscription(&sender, subscription);
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct AccumulatorState {
        value: i32,
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum AccumulatorInput {
        Add(i16),
        Subtract(i16),
        Reset,
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum AccumulatorEffect {
        Changed { from: i32, to: i32 },
        ResetObserved { from: i32 },
    }

    struct AccumulatorApp;

    #[derive(Clone, Debug, Eq, PartialEq)]
    struct DeliveryEffect(Vec<u16>);

    struct EffectDeliveryApp;

    struct EffectDeliveryRuntime;

    struct SendErrorLoggingRuntime {
        failed_inputs: std::sync::Arc<std::sync::Mutex<Vec<u16>>>,
    }

    struct StateRecordingRuntime {
        observed: std::sync::Arc<std::sync::Mutex<Vec<i32>>>,
    }

    impl Application for AccumulatorApp {
        type State = AccumulatorState;
        type Input = AccumulatorInput;
        type Effect = AccumulatorEffect;
        type Subscription = ();

        fn init() -> Transition<Self::State, Self::Effect> {
            Transition {
                state: AccumulatorState { value: 0 },
                effects: Vec::new(),
            }
        }

        fn transition(
            state: Self::State,
            input: Self::Input,
        ) -> Transition<Self::State, Self::Effect> {
            match input {
                AccumulatorInput::Add(amount) => {
                    let next = state.value.saturating_add(i32::from(amount));

                    Transition {
                        state: AccumulatorState { value: next },
                        effects: vec![AccumulatorEffect::Changed {
                            from: state.value,
                            to: next,
                        }],
                    }
                }
                AccumulatorInput::Subtract(amount) => {
                    let next = state.value.saturating_sub(i32::from(amount));

                    Transition {
                        state: AccumulatorState { value: next },
                        effects: vec![AccumulatorEffect::Changed {
                            from: state.value,
                            to: next,
                        }],
                    }
                }
                AccumulatorInput::Reset => Transition {
                    state: AccumulatorState { value: 0 },
                    effects: vec![AccumulatorEffect::ResetObserved { from: state.value }],
                },
            }
        }

        fn subscriptions() -> Vec<Self::Subscription> {
            Vec::new()
        }
    }

    impl Application for EffectDeliveryApp {
        type State = ();
        type Input = u16;
        type Effect = DeliveryEffect;
        type Subscription = ();

        fn init() -> Transition<Self::State, Self::Effect> {
            Transition {
                state: (),
                effects: Vec::new(),
            }
        }

        fn transition(
            state: Self::State,
            _input: Self::Input,
        ) -> Transition<Self::State, Self::Effect> {
            Transition {
                state,
                effects: Vec::new(),
            }
        }

        fn subscriptions() -> Vec<Self::Subscription> {
            Vec::new()
        }
    }

    impl Runtime<EffectDeliveryApp> for EffectDeliveryRuntime {
        fn execute_effect(&self, effect: DeliveryEffect) -> Vec<u16> {
            effect.0
        }

        fn spawn_subscription(&self, _sender: &Sender<u16>, _subscription: ()) {}
    }

    impl Runtime<AccumulatorApp> for StateRecordingRuntime {
        fn execute_effect(&self, _effect: AccumulatorEffect) -> Vec<AccumulatorInput> {
            Vec::new()
        }

        fn spawn_subscription(&self, _sender: &Sender<AccumulatorInput>, _subscription: ()) {}

        fn observe_state(&self, state: &AccumulatorState) {
            self.observed.lock().unwrap().push(state.value);
        }
    }

    impl Runtime<EffectDeliveryApp> for SendErrorLoggingRuntime {
        fn execute_effect(&self, effect: DeliveryEffect) -> Vec<u16> {
            effect.0
        }

        fn spawn_subscription(&self, _sender: &Sender<u16>, _subscription: ()) {}

        fn log_error(&self, error: ApplicationError<u16>) {
            match error {
                ApplicationError::SendError(error) => {
                    self.failed_inputs.lock().unwrap().push(error.0);
                }
            }
        }
    }

    proptest! {
        #[test]
        fn transition_fold_matches_reference_model(
            inputs in prop::collection::vec(accumulator_input(), 0..100),
        ) {
            let actual = fold_application(&inputs);
            let expected = fold_reference(&inputs);

            prop_assert_eq!(actual, expected);
        }

        #[test]
        fn spawn_effects_delivers_each_runtime_input_once(
            effects in prop::collection::vec(delivery_effect(), 0..50),
        ) {
            let expected = delivered_inputs(&effects);
            let (sender, receiver) = std::sync::mpsc::channel();
            let runtime = std::sync::Arc::new(EffectDeliveryRuntime);
            let handle = spawn_effects::<EffectDeliveryApp, EffectDeliveryRuntime>(
                runtime,
                test_effect_pool(4),
                &sender,
                effects,
            );

            prop_assert!(handle.join().is_ok());

            let mut actual = receiver.try_iter().collect::<Vec<_>>();
            let mut expected = expected;
            actual.sort_unstable();
            expected.sort_unstable();

            prop_assert_eq!(actual, expected);
        }
    }

    #[test]
    fn observe_state_is_called_with_each_post_transition_state_in_order() {
        let observed = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let runtime = StateRecordingRuntime {
            observed: observed.clone(),
        };
        // `run` loops forever (it holds its own sender clone for spawning effects),
        // so we never join it — we send inputs and wait for the observations.
        let (sender, _handle) = <StateRecordingRuntime as Runtime<AccumulatorApp>>::run(runtime);

        sender.send(AccumulatorInput::Add(5)).unwrap();
        sender.send(AccumulatorInput::Add(3)).unwrap();
        sender.send(AccumulatorInput::Reset).unwrap();

        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while observed.lock().unwrap().len() < 3 {
            assert!(
                std::time::Instant::now() < deadline,
                "timed out waiting for observations"
            );
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        assert_eq!(observed.lock().unwrap().as_slice(), &[5, 8, 0]);
    }

    #[test]
    fn spawn_effects_logs_send_error_when_receiver_is_dropped() {
        let (sender, receiver) = std::sync::mpsc::channel();
        drop(receiver);
        let failed_inputs = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let runtime = std::sync::Arc::new(SendErrorLoggingRuntime {
            failed_inputs: failed_inputs.clone(),
        });

        let handle = spawn_effects::<EffectDeliveryApp, SendErrorLoggingRuntime>(
            runtime,
            test_effect_pool(4),
            &sender,
            vec![DeliveryEffect(vec![7])],
        );

        assert!(handle.join().is_ok());
        assert_eq!(failed_inputs.lock().unwrap().as_slice(), &[7]);
    }

    #[test]
    fn effect_pool_size_defaults_to_eight() {
        assert_eq!(
            <EffectDeliveryRuntime as Runtime<EffectDeliveryApp>>::effect_pool_size(
                &EffectDeliveryRuntime
            ),
            8
        );
    }

    #[test]
    fn spawn_effects_delivers_every_input_on_a_single_worker_pool() {
        // A pool narrower than the effect count still delivers every input exactly once — effects
        // queue onto the lone worker instead of being dropped.
        let effects = vec![
            DeliveryEffect(vec![1, 2]),
            DeliveryEffect(vec![3]),
            DeliveryEffect(vec![4, 5, 6]),
        ];
        let (sender, receiver) = std::sync::mpsc::channel();
        let runtime = std::sync::Arc::new(EffectDeliveryRuntime);

        let handle = spawn_effects::<EffectDeliveryApp, EffectDeliveryRuntime>(
            runtime,
            test_effect_pool(1),
            &sender,
            effects,
        );
        assert!(handle.join().is_ok());

        let mut actual = receiver.try_iter().collect::<Vec<_>>();
        actual.sort_unstable();
        assert_eq!(actual, vec![1, 2, 3, 4, 5, 6]);
    }

    fn test_effect_pool(threads: usize) -> Option<std::sync::Arc<rayon::ThreadPool>> {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map(std::sync::Arc::new)
            .ok()
    }

    fn accumulator_input() -> impl Strategy<Value = AccumulatorInput> {
        prop_oneof![
            (0i16..=10_000).prop_map(AccumulatorInput::Add),
            (0i16..=10_000).prop_map(AccumulatorInput::Subtract),
            Just(AccumulatorInput::Reset),
        ]
    }

    fn delivery_effect() -> impl Strategy<Value = DeliveryEffect> {
        prop::collection::vec(any::<u16>(), 0..20).prop_map(DeliveryEffect)
    }

    fn delivered_inputs(effects: &[DeliveryEffect]) -> Vec<u16> {
        effects
            .iter()
            .flat_map(|effect| effect.0.iter().copied())
            .collect()
    }

    fn fold_application(inputs: &[AccumulatorInput]) -> (AccumulatorState, Vec<AccumulatorEffect>) {
        let Transition { state, effects } = AccumulatorApp::init();

        inputs
            .iter()
            .copied()
            .fold((state, effects), |(state, effects), input| {
                let Transition {
                    state: next_state,
                    effects: next_effects,
                } = AccumulatorApp::transition(state, input);

                (
                    next_state,
                    effects.into_iter().chain(next_effects).collect(),
                )
            })
    }

    fn fold_reference(inputs: &[AccumulatorInput]) -> (AccumulatorState, Vec<AccumulatorEffect>) {
        inputs.iter().copied().fold(
            (AccumulatorState { value: 0 }, Vec::new()),
            |(state, effects), input| {
                let (next_state, effect) = reference_transition(state, input);

                (next_state, effects.into_iter().chain([effect]).collect())
            },
        )
    }

    fn reference_transition(
        state: AccumulatorState,
        input: AccumulatorInput,
    ) -> (AccumulatorState, AccumulatorEffect) {
        match input {
            AccumulatorInput::Add(amount) => {
                let next = state.value.saturating_add(i32::from(amount));

                (
                    AccumulatorState { value: next },
                    AccumulatorEffect::Changed {
                        from: state.value,
                        to: next,
                    },
                )
            }
            AccumulatorInput::Subtract(amount) => {
                let next = state.value.saturating_sub(i32::from(amount));

                (
                    AccumulatorState { value: next },
                    AccumulatorEffect::Changed {
                        from: state.value,
                        to: next,
                    },
                )
            }
            AccumulatorInput::Reset => (
                AccumulatorState { value: 0 },
                AccumulatorEffect::ResetObserved { from: state.value },
            ),
        }
    }
}
