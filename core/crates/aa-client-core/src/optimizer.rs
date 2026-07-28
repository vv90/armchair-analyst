//! The optimizer worker: a **self-clocked** grind loop that owns the heavy mutable state the pure
//! reducer deliberately does not — `optimization::OptimizationRunner`, the move-based state machine
//! holding the `Model`/optimizer/session tensors — and turns a stream of fresh reserve snapshots into
//! [`Event::OptimizerStepped`] outcomes.
//!
//! This mirrors aa-cli's `run_optimization`: reserves arrive through a **coalescing** [`LatestReceiver`]
//! ([`crate::latest_slot`]), and the worker drives its own cadence. After each step it *pulls* — a
//! fresh snapshot ([`LatestReceiver::try_take`]) becomes a `NewReserves` step, an empty slot becomes a
//! `Continue`. `Continue` therefore never crosses a channel: the loop self-continues at full compute
//! speed with no round-trip through the reducer, and reserve snapshots coalesce (latest-wins) so the
//! optimizer always grinds the freshest reserves instead of a backlog. Contrast the old design, where
//! the reducer re-emitted every `Continue` and each slice queued a `NewReserves`, so the worker fell
//! linearly behind on ever-staler reserves.
//!
//! The reducer keeps the productivity gate (it only pushes reserves that cover the route), so init
//! succeeds on the first snapshot and the grind only ever `Continue`s on an empty slot. The worker still
//! carries aa-cli's internal skip guards (a transiently unreachable snapshot is skipped, not fatal) for
//! robustness. Every optimizer error degrades to an [`Event::EffectFailed`]; the loop never panics.

use std::collections::HashSet;
use std::sync::mpsc::Sender;

use client_evm::{PoolRef, TokenAddress};
use optimization::{
    OptimizationBackendSelection, OptimizationInitError, OptimizationRunner,
    OptimizationSessionConfig, OptimizationStepConfig, OptimizationStepError,
    OptimizationStepUpdate, PoolReserves, reserves_reach_route,
};

use crate::latest_slot::LatestReceiver;
use crate::state::{EffectError, Event, OptimizeStage};

/// One optimizer session's worth of layers. Matches aa-cli's `OPTIMIZATION_LAYERS`.
const LAYERS: usize = 1;

/// What crosses the slot: the projected reserves **and** the route they were admitted for.
/// Freshness/provenance is the reducer's concern and is tracked in `AppState`, so the worker needs
/// nothing else. Carrying the route makes each snapshot self-describing and leaves `AppState` its
/// single owner — the worker holds no config it could disagree with (see
/// [`crate::Effect::PushReserves`]).
pub(crate) struct ReserveSnapshot {
    /// The reserves to grind.
    pub(crate) reserves: Vec<PoolReserves<PoolRef, TokenAddress>>,
    /// The route and routing constraints they were gated against.
    pub(crate) session: OptimizationSessionConfig<TokenAddress>,
}

/// Whether a step error means "this snapshot does not cover the configured route (yet)" rather than
/// "something is wrong". Both ends of the route can be transiently missing — the chain owning the
/// sink token reported late, a bootstrap gap, or (for an open route) a source asset that this
/// server's catalog simply never mentions. None of those may kill the worker: nothing restarts it,
/// and a later snapshot or a later route can satisfy them.
fn is_route_gap(error: &OptimizationStepError) -> bool {
    matches!(
        error,
        OptimizationStepError::OutputAssetNotFound | OptimizationStepError::SourceAssetNotFound
    )
}

/// Own an `OptimizationRunner` and grind it, self-clocked, until the slot or the event sink closes.
///
/// Blocks for the first snapshot that initializes (skipping any that does not yet cover its route),
/// then loops forever: pull the freshest reserves if any (`NewReserves`) else `Continue`, step, and
/// forward the result as an [`Event`]. Returns — ending the subscription thread cleanly — when the
/// reserve slot closes (the runtime dropped its sender) or the event receiver is gone (the engine is
/// shutting down). A fatal init/step error is reported as [`Event::EffectFailed`] before returning.
///
/// The route is never configured in: it arrives on every snapshot, and a snapshot carrying a route
/// other than the one the live session was built for re-enters the init path (the model's tensors are
/// shaped by the route, so retargeting means a new session). The worker therefore has no route of its
/// own to fall out of step with the reducer's.
pub(crate) fn run(
    receiver: LatestReceiver<ReserveSnapshot>,
    backend: OptimizationBackendSelection,
    step_config: OptimizationStepConfig,
    events: Sender<Event>,
) {
    // Carries a retargeted snapshot from the grind loop back into the init path, so a route change
    // re-initializes on the very snapshot that announced it instead of waiting for the next push.
    let mut retargeted: Option<ReserveSnapshot> = None;

    loop {
        // Init loop: wait for the first snapshot that actually initializes. A snapshot that does not
        // cover its route (`SourceAssetNotFound`/`OutputAssetNotFound`) is a transient "not ready
        // yet", not a fatal error — skip it and wait for the next. Every other init error is fatal.
        let (mut runner, session, result, plan) = loop {
            let snapshot = match retargeted.take() {
                Some(snapshot) => snapshot,
                None => match receiver.wait_take() {
                    Ok(snapshot) => snapshot,
                    // Slot closed before we ever initialized: nothing to do.
                    Err(_) => return,
                },
            };
            let ReserveSnapshot { reserves, session } = snapshot;
            match OptimizationRunner::<PoolRef, TokenAddress, LAYERS>::init(
                backend,
                reserves,
                session.clone(),
                step_config,
            ) {
                Ok((runner, result, plan)) => break (runner, session, result, plan),
                Err(OptimizationInitError::Step(error)) if is_route_gap(&error) => {
                    continue;
                }
                Err(error) => {
                    let _ = events.send(Event::EffectFailed(EffectError::Optimize {
                        stage: OptimizeStage::Init,
                        message: error.to_string(),
                    }));
                    return;
                }
            }
        };

        if events
            .send(Event::OptimizerStepped { result, plan })
            .is_err()
        {
            return;
        }

        loop {
            let update = match receiver.try_take() {
                // A snapshot for a different route: this session cannot serve it, so hand it to the
                // init path and start a new one.
                Ok(Some(snapshot)) if snapshot.session != session => {
                    retargeted = Some(snapshot);
                    break;
                }
                // Fresh reserves that cover the route: step them.
                Ok(Some(snapshot)) if reserves_reach_route(&snapshot.reserves, &session) => {
                    OptimizationStepUpdate::NewReserves {
                        reserves: snapshot.reserves,
                        disabled: HashSet::new(),
                    }
                }
                // A snapshot that transiently does not cover the route would abort the session; skip
                // it and keep grinding the reserves already loaded.
                Ok(Some(_)) => OptimizationStepUpdate::Continue,
                // Nothing new: self-continue on the current session (the self-clock).
                Ok(None) => OptimizationStepUpdate::Continue,
                // Slot closed: the runtime is shutting down.
                Err(_) => return,
            };
            let (next_runner, result, plan) = match runner.run(update) {
                Ok(stepped) => stepped,
                Err(error) => {
                    let _ = events.send(Event::EffectFailed(EffectError::Optimize {
                        stage: OptimizeStage::Run,
                        message: error.to_string(),
                    }));
                    return;
                }
            };
            if events
                .send(Event::OptimizerStepped { result, plan })
                .is_err()
            {
                return;
            }
            runner = next_runner;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    use client_evm::{ChainKey, PoolRef, TokenAddress};
    use optimization::{
        Invertible, OptimizationBackendSelection, OptimizationSessionConfig,
        OptimizationStepConfig, OptimizationStepStatus, PoolReserves, VirtualReserveValues,
    };

    use super::*;
    use crate::latest_slot::latest_slot;

    const CHAIN: ChainKey = ChainKey::Ethereum;

    fn token() -> TokenAddress {
        TokenAddress(Default::default(), CHAIN)
    }

    fn pool() -> PoolRef {
        PoolRef::uniswap_v3(Default::default(), CHAIN)
    }

    fn session_config() -> OptimizationSessionConfig<TokenAddress> {
        OptimizationSessionConfig {
            source_asset: token(),
            output_asset: token(),
            bridges: HashSet::new(),
            whitelist: None,
        }
    }

    /// An open route whose source asset appears in no reserve of `base_reserves` (whose only token is
    /// `token()`): the output is reachable, but there is nothing to spend from the source.
    fn absent_source_config() -> OptimizationSessionConfig<TokenAddress> {
        OptimizationSessionConfig {
            source_asset: TokenAddress(Default::default(), ChainKey::Arbitrum),
            output_asset: token(),
            bridges: HashSet::new(),
            whitelist: None,
        }
    }

    /// A second token, so a snapshot can serve both a closed cycle on `token()` and an open route to
    /// this one — the two routes `a_snapshot_for_a_different_route_reinitializes_the_session` swaps.
    fn token_b() -> TokenAddress {
        TokenAddress(client_evm::Address::from([2; 20]), CHAIN)
    }

    /// The open route `token() → token_b()` over the same reserves `session_config`'s cycle uses.
    fn open_route_config() -> OptimizationSessionConfig<TokenAddress> {
        OptimizationSessionConfig {
            source_asset: token(),
            output_asset: token_b(),
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

    /// A snapshot: reserves plus the route they were admitted for, exactly as the reducer pushes it.
    fn snapshot(
        reserves: Vec<PoolReserves<PoolRef, TokenAddress>>,
        session: OptimizationSessionConfig<TokenAddress>,
    ) -> ReserveSnapshot {
        ReserveSnapshot { reserves, session }
    }

    /// A single self-token pool whose output is the route asset — the minimal reserve set that
    /// initializes (mirrors aa-cli's optimizer-test fixture).
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

    /// A `token()`/`token_b()` pool in both directions: initializes both for the closed cycle on
    /// `token()` and for the open route to `token_b()`, so swapping routes changes nothing but the
    /// route itself.
    fn two_token_reserves() -> Vec<PoolReserves<PoolRef, TokenAddress>> {
        let forward = PoolReserves {
            pool_id: pool(),
            token0: token(),
            token1: token_b(),
            value: VirtualReserveValues {
                token_0: 2.0,
                token_1: 3.0,
                fee_multiplier: 0.997,
                max_swap_0: 1.0,
                max_swap_1: 1.0,
            },
        };

        vec![forward, forward.inverse()]
    }

    /// Reserves that never output the session's output asset (a distinct, Arbitrum-stamped token), so
    /// `reserves_reach_route` is false and init would fail with `OutputAssetNotFound`.
    fn unreachable_reserves() -> Vec<PoolReserves<PoolRef, TokenAddress>> {
        let other = TokenAddress(Default::default(), ChainKey::Arbitrum);
        vec![PoolReserves {
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
        }]
    }

    fn step_status(event: Event) -> OptimizationStepStatus {
        match event {
            Event::OptimizerStepped { result, .. } => result.status,
            other => panic!("expected OptimizerStepped, got {other:?}"),
        }
    }

    fn run_worker(
        receiver: LatestReceiver<ReserveSnapshot>,
        events: Sender<Event>,
    ) -> thread::JoinHandle<()> {
        thread::spawn(move || {
            run(
                receiver,
                OptimizationBackendSelection::Cpu,
                step_config(),
                events,
            )
        })
    }

    #[test]
    fn slot_closed_before_first_snapshot_exits_without_stepping() {
        let (sender, receiver) = latest_slot();
        drop(sender);
        let (events_tx, events_rx) = mpsc::channel();
        // No thread needed: `wait_take` returns Closed immediately, so `run` returns at once.
        run(
            receiver,
            OptimizationBackendSelection::Cpu,
            step_config(),
            events_tx,
        );
        // No step was ever produced (and the sender end is now dropped).
        assert!(events_rx.recv().is_err());
    }

    #[test]
    fn one_snapshot_self_clocks_a_stream_of_continues() {
        // THE core parity behaviour: a single snapshot yields Initialized then a run of Continued steps
        // with no further sends — proving `Continue` needs no channel round-trip.
        let (sender, receiver) = latest_slot();
        let (events_tx, events_rx) = mpsc::channel();
        let handle = run_worker(receiver, events_tx);

        sender
            .send(snapshot(base_reserves(), session_config()))
            .expect("send initial snapshot");

        assert_eq!(
            step_status(events_rx.recv().expect("init event")),
            OptimizationStepStatus::Initialized
        );
        // With no additional sends, the next several steps are all self-driven Continues.
        for _ in 0..3 {
            assert_eq!(
                step_status(events_rx.recv().expect("continue event")),
                OptimizationStepStatus::Continued
            );
        }

        // Dropping the sender closes the slot; the worker's next `try_take` returns Closed and it exits.
        drop(sender);
        handle.join().expect("worker joins");
    }

    #[test]
    fn fresh_reserves_produce_an_updated_step() {
        let (sender, receiver) = latest_slot();
        let (events_tx, events_rx) = mpsc::channel();
        let handle = run_worker(receiver, events_tx);

        sender
            .send(snapshot(base_reserves(), session_config()))
            .expect("send initial snapshot");
        assert_eq!(
            step_status(events_rx.recv().expect("init event")),
            OptimizationStepStatus::Initialized
        );

        // A fresh snapshot must be applied as a NewReserves step; the worker may emit a Continue or two
        // before its `try_take` observes the send, so we look for the Updated within a bounded window.
        sender
            .send(snapshot(base_reserves(), session_config()))
            .expect("send fresh snapshot");
        let mut saw_updated = false;
        for _ in 0..16 {
            if step_status(
                events_rx
                    .recv_timeout(Duration::from_secs(5))
                    .expect("step event"),
            ) == OptimizationStepStatus::Updated
            {
                saw_updated = true;
                break;
            }
        }
        assert!(saw_updated, "a fresh snapshot must produce an Updated step");

        drop(sender);
        handle.join().expect("worker joins");
    }

    #[test]
    fn first_unreachable_snapshot_is_skipped_not_fatal() {
        // An unready snapshot at init must be skipped (no fatal error, no step), and the worker keeps
        // waiting. Closing the slot then ends it cleanly.
        let (sender, receiver) = latest_slot();
        let (events_tx, events_rx) = mpsc::channel();
        let handle = run_worker(receiver, events_tx);

        sender
            .send(snapshot(unreachable_reserves(), session_config()))
            .expect("send unready snapshot");
        // Give the worker a moment to consume-and-skip it, then close.
        thread::sleep(Duration::from_millis(20));
        drop(sender);

        // No step and no fatal error was ever emitted; the channel just closes.
        assert!(events_rx.recv().is_err());
        handle.join().expect("worker joins");
    }

    #[test]
    fn unreachable_snapshot_after_init_keeps_grinding() {
        let (sender, receiver) = latest_slot();
        let (events_tx, events_rx) = mpsc::channel();
        let handle = run_worker(receiver, events_tx);

        sender
            .send(snapshot(base_reserves(), session_config()))
            .expect("send initial snapshot");
        assert_eq!(
            step_status(events_rx.recv().expect("init event")),
            OptimizationStepStatus::Initialized
        );

        // A transiently unreachable snapshot must not abort the live session; the worker skips it and
        // keeps producing Continued steps (never an EffectFailed).
        sender
            .send(snapshot(unreachable_reserves(), session_config()))
            .expect("send unready snapshot");
        for _ in 0..4 {
            assert_eq!(
                step_status(events_rx.recv().expect("continue event")),
                OptimizationStepStatus::Continued
            );
        }

        drop(sender);
        handle.join().expect("worker joins");
    }

    #[test]
    fn first_snapshot_missing_the_source_asset_is_skipped_not_fatal() {
        // The open-route counterpart of `first_unreachable_snapshot_is_skipped_not_fatal`: the
        // snapshot *reaches* the output asset but carries nothing to spend from the source asset, so
        // the model has no input column. That is "not ready for this route yet" — a later slice (or a
        // later route) can fix it — and must never kill the worker thread, which nothing restarts.
        let (sender, receiver) = latest_slot();
        let (events_tx, events_rx) = mpsc::channel();
        let handle = run_worker(receiver, events_tx);

        sender
            .send(snapshot(base_reserves(), absent_source_config()))
            .expect("send snapshot without the source asset");
        thread::sleep(Duration::from_millis(20));
        drop(sender);

        // No step and no fatal error: the worker skipped the snapshot and kept waiting.
        assert!(events_rx.recv().is_err());
        handle.join().expect("worker joins");
    }

    #[test]
    fn a_snapshot_for_a_different_route_reinitializes_the_session() {
        // The retarget path. A running session's tensors are shaped by the route it was built for, so
        // a snapshot carrying a *different* route cannot be applied to it — grinding on would keep
        // optimizing the pair the user just navigated away from. The worker must start a new session
        // instead, which it announces with a second `Initialized`.
        let (sender, receiver) = latest_slot();
        let (events_tx, events_rx) = mpsc::channel();
        let handle = run_worker(receiver, events_tx);

        sender
            .send(snapshot(two_token_reserves(), session_config()))
            .expect("send arbitrage-route snapshot");
        assert_eq!(
            step_status(events_rx.recv().expect("init event")),
            OptimizationStepStatus::Initialized
        );

        // Same reserves, open route `token() → token_b()`: a different session entirely.
        sender
            .send(snapshot(two_token_reserves(), open_route_config()))
            .expect("send retargeted snapshot");

        let mut reinitialized = false;
        for _ in 0..16 {
            if step_status(
                events_rx
                    .recv_timeout(Duration::from_secs(5))
                    .expect("step event"),
            ) == OptimizationStepStatus::Initialized
            {
                reinitialized = true;
                break;
            }
        }
        assert!(
            reinitialized,
            "a snapshot carrying a new route must re-initialize the session"
        );

        drop(sender);
        handle.join().expect("worker joins");
    }

    #[test]
    fn fatal_init_error_is_reported_then_the_worker_exits() {
        // Empty reserves abort init with a fatal (non-transient) error; the reducer's productivity gate
        // keeps this from happening in practice, but the worker must report it and stay total.
        let (sender, receiver) = latest_slot();
        sender
            .send(snapshot(Vec::new(), session_config()))
            .expect("send empty snapshot");
        let (events_tx, events_rx) = mpsc::channel();
        // Pre-sent snapshot, so `run` returns after the init failure without needing another thread.
        run(
            receiver,
            OptimizationBackendSelection::Cpu,
            step_config(),
            events_tx,
        );
        assert!(matches!(
            events_rx.recv().expect("failure event"),
            Event::EffectFailed(EffectError::Optimize {
                stage: OptimizeStage::Init,
                ..
            })
        ));
    }

    #[test]
    fn exits_cleanly_when_event_receiver_is_dropped() {
        let (sender, receiver) = latest_slot();
        sender
            .send(snapshot(base_reserves(), session_config()))
            .expect("send snapshot");
        let (events_tx, events_rx) = mpsc::channel::<Event>();
        drop(events_rx);
        // The worker inits, fails to send the first result (receiver gone), and returns.
        run(
            receiver,
            OptimizationBackendSelection::Cpu,
            step_config(),
            events_tx,
        );
    }
}
