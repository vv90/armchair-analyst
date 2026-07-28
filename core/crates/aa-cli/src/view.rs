use std::{
    io::{self, Stdout, Write},
    sync::{Arc, Mutex},
};

use client_evm::{
    ChainKey,
    multi_chain_kernel::{ChainObservation, ChainProgress, PlanVerification, State},
};
use optimization::OptimizationStepResult;
use ratatui::{
    Terminal, TerminalOptions, Viewport, backend::CrosstermBackend, layout::Position, text::Line,
    widgets::Paragraph,
};

type LiveTerminal = Terminal<CrosstermBackend<Stdout>>;

/// Thin terminal adapter for the live status block.
///
/// Mirrors [`Logger`](crate::logger::Logger): a cheap, cloneable handle whose side effects
/// are isolated here. [`View::sink`] is the no-op used by tests; [`View::for_run`] drives a
/// real `ratatui` inline viewport. The pure [`format_lines`] call decides *what* to show; the
/// adapter only decides *how* to put it on screen, and `ratatui` owns the cursor arithmetic.
#[derive(Clone)]
pub(crate) struct View {
    inner: Arc<Mutex<ViewInner>>,
}

enum ViewInner {
    Sink,
    /// The terminal is created lazily on the first render, once the chain count (and thus the
    /// viewport height) is known.
    Live(Option<LiveTerminal>),
}

impl View {
    /// A view that drives a real inline viewport on stdout.
    pub(crate) fn for_run() -> View {
        View {
            inner: Arc::new(Mutex::new(ViewInner::Live(None))),
        }
    }

    /// A view that discards everything it is given. Used by tests.
    pub(crate) fn sink() -> View {
        View {
            inner: Arc::new(Mutex::new(ViewInner::Sink)),
        }
    }

    /// Redraws the live block in place from the current state. Never blocks on a poisoned lock
    /// and never panics; a terminal that cannot be initialized degrades to no display.
    pub(crate) fn render(&self, state: &State) {
        let Ok(mut guard) = self.inner.lock() else {
            return;
        };
        let ViewInner::Live(slot) = &mut *guard else {
            return;
        };

        let observations = state.observe();
        let lines = format_lines(
            &observations,
            state.latest_optimization_result(),
            state.latest_plan_verification(),
        );

        let terminal = match slot {
            Some(terminal) => terminal,
            None => match new_inline_terminal(reserved_rows(observations.len())) {
                Ok(terminal) => slot.insert(terminal),
                Err(_) => return,
            },
        };

        let _ = terminal.draw(|frame| {
            frame.render_widget(paragraph(lines), frame.area());
        });
    }

    /// Terminates the live block so the shell prompt lands on a fresh line below it.
    pub(crate) fn finish(&self) {
        let Ok(mut guard) = self.inner.lock() else {
            return;
        };
        if let ViewInner::Live(slot) = &mut *guard {
            if let Some(terminal) = slot.as_mut() {
                let bottom = terminal.get_frame().area().bottom().saturating_sub(1);
                let _ = terminal.set_cursor_position(Position::new(0, bottom));
                let _ = terminal.show_cursor();
                let _ = terminal.backend_mut().flush();
            }
            *slot = None;
            println!();
        }
    }
}

fn new_inline_terminal(height: u16) -> io::Result<LiveTerminal> {
    let backend = CrosstermBackend::new(io::stdout());
    Terminal::with_options(
        backend,
        TerminalOptions {
            viewport: Viewport::Inline(height),
        },
    )
}

/// Rows the inline viewport reserves: one per chain plus a fixed row for the optimization
/// summary, so the summary appearing or disappearing never shifts the block.
fn reserved_rows(chain_count: usize) -> u16 {
    u16::try_from(chain_count.saturating_add(1)).unwrap_or(u16::MAX)
}

fn paragraph(lines: Vec<String>) -> Paragraph<'static> {
    Paragraph::new(lines.into_iter().map(Line::from).collect::<Vec<_>>())
}

fn format_lines(
    observations: &[(ChainKey, ChainObservation)],
    optimization: Option<OptimizationStepResult>,
    verification: Option<PlanVerification>,
) -> Vec<String> {
    let mut lines: Vec<String> = observations
        .iter()
        .map(|(chain, observation)| {
            format!(
                "{}: {}",
                format_chain(*chain),
                format_observation(*observation)
            )
        })
        .collect();

    if let Some(result) = optimization {
        lines.push(format_optimization(result, verification));
    }

    lines
}

fn format_optimization(
    result: OptimizationStepResult,
    verification: Option<PlanVerification>,
) -> String {
    format!(
        "Optimization  status={:?}  profit={}  reserves={}  routed={}  eff_pools={:.2}{}",
        result.status,
        result.profit_amount,
        result.reserves_count,
        result.routed_pool_count,
        result.effective_pools,
        format_verification(verification),
    )
}

/// The lossless-replay verdict on the latest result's plan, appended to the optimization line so
/// the verified profit sits next to the claimed one. Empty when the step carried no plan.
fn format_verification(verification: Option<PlanVerification>) -> String {
    match verification {
        None => String::new(),
        Some(PlanVerification::Verified {
            profit,
            hit_tick_limit,
        }) => format!(
            "  verified={profit}{}",
            if hit_tick_limit {
                " (tick limited)"
            } else {
                ""
            }
        ),
        Some(PlanVerification::Unverifiable(failure)) => {
            format!("  verified=unverifiable({failure:?})")
        }
    }
}

fn format_chain(chain: ChainKey) -> &'static str {
    match chain {
        ChainKey::Ethereum => "Ethereum",
        ChainKey::Arbitrum => "Arbitrum",
        ChainKey::Base => "Base",
        ChainKey::Optimism => "Optimism",
        ChainKey::Polygon => "Polygon",
        ChainKey::Bnb => "BNB Chain",
        ChainKey::Avalanche => "Avalanche",
    }
}

fn format_observation(observation: ChainObservation) -> String {
    match observation {
        ChainObservation::Initializing { buffered_events } => {
            format!("Initializing  buffered={buffered_events}")
        }
        ChainObservation::Active(ChainProgress {
            verified_pools,
            blocks_behind_tip,
            canonical_window,
            in_flight_requests,
            ws_misses,
        }) => format!(
            "Active  pools={verified_pools}  behind={}  window={}  inflight={in_flight_requests}  ws_miss={ws_misses}",
            format_distance(blocks_behind_tip),
            format_distance(canonical_window)
        ),
    }
}

fn format_distance(blocks_behind_tip: Option<usize>) -> String {
    match blocks_behind_tip {
        Some(distance) => distance.to_string(),
        None => "?".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use optimization::OptimizationStepStatus;

    use super::*;

    #[test]
    fn format_lines_renders_initializing_observation_with_replay_buffer_depth() {
        let observations = [(
            ChainKey::Ethereum,
            ChainObservation::Initializing {
                buffered_events: 42,
            },
        )];

        assert_eq!(
            format_lines(&observations, None, None),
            vec!["Ethereum: Initializing  buffered=42"]
        );
    }

    #[test]
    fn format_lines_renders_arbitrum_chain_label() {
        let observations = [(
            ChainKey::Arbitrum,
            ChainObservation::Initializing { buffered_events: 0 },
        )];

        assert_eq!(
            format_lines(&observations, None, None),
            vec!["Arbitrum: Initializing  buffered=0"]
        );
    }

    #[test]
    fn format_lines_renders_active_progress() {
        let observations = [(
            ChainKey::Ethereum,
            ChainObservation::Active(ChainProgress {
                verified_pools: 37,
                blocks_behind_tip: Some(2),
                canonical_window: Some(64),
                in_flight_requests: 5,
                ws_misses: 1,
            }),
        )];

        assert_eq!(
            format_lines(&observations, None, None),
            vec!["Ethereum: Active  pools=37  behind=2  window=64  inflight=5  ws_miss=1"]
        );
    }

    #[test]
    fn format_lines_renders_unknown_distance_as_question_mark() {
        let observations = [(
            ChainKey::Ethereum,
            ChainObservation::Active(ChainProgress {
                verified_pools: 0,
                blocks_behind_tip: None,
                canonical_window: None,
                in_flight_requests: 0,
                ws_misses: 0,
            }),
        )];

        assert_eq!(
            format_lines(&observations, None, None),
            vec!["Ethereum: Active  pools=0  behind=?  window=?  inflight=0  ws_miss=0"]
        );
    }

    #[test]
    fn format_lines_appends_optimization_result_on_its_own_line() {
        let observations = [(
            ChainKey::Ethereum,
            ChainObservation::Initializing { buffered_events: 0 },
        )];

        assert_eq!(
            format_lines(&observations, Some(result()), None),
            vec![
                "Ethereum: Initializing  buffered=0".to_owned(),
                "Optimization  status=Initialized  profit=1.5  reserves=3  routed=3  eff_pools=2.46".to_owned(),
            ]
        );
    }

    #[test]
    fn reserved_rows_is_one_per_chain_plus_the_optimization_summary() {
        assert_eq!(reserved_rows(1), 2);
        assert_eq!(reserved_rows(3), 4);
    }

    #[test]
    fn reserved_rows_saturates_instead_of_overflowing() {
        assert_eq!(reserved_rows(usize::MAX), u16::MAX);
    }

    #[test]
    fn format_optimization_renders_status_profit_and_reserves() {
        assert_eq!(
            format_optimization(result(), None),
            "Optimization  status=Initialized  profit=1.5  reserves=3  routed=3  eff_pools=2.46"
        );
    }

    #[test]
    fn format_optimization_puts_the_verified_profit_next_to_the_claimed_one() {
        assert_eq!(
            format_optimization(
                result(),
                Some(PlanVerification::Verified {
                    profit: 1.25,
                    hit_tick_limit: false,
                }),
            ),
            "Optimization  status=Initialized  profit=1.5  reserves=3  routed=3  eff_pools=2.46  verified=1.25"
        );
        assert_eq!(
            format_optimization(
                result(),
                Some(PlanVerification::Verified {
                    profit: -0.5,
                    hit_tick_limit: true,
                }),
            ),
            "Optimization  status=Initialized  profit=1.5  reserves=3  routed=3  eff_pools=2.46  verified=-0.5 (tick limited)"
        );
    }

    fn result() -> OptimizationStepResult {
        OptimizationStepResult {
            status: OptimizationStepStatus::Initialized,
            input_amount: 1.0,
            output_amount: 2.5,
            profit_amount: 1.5,
            reserves_count: 3,
            disabled_count: 0,
            pool_slots: 3,
            route_entropy: 0.9,
            effective_pools: 2.46,
            routed_pool_count: 3,
            iterations_completed: 7,
        }
    }
}
