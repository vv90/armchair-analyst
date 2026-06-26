use std::{
    io::{self, Stdout, Write},
    sync::{Arc, Mutex},
};

use client_evm::{
    ChainKey,
    multi_chain_kernel::{ChainObservation, ChainProgress, State},
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
        let lines = format_lines(&observations, state.latest_optimization_result());

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
        lines.push(format_optimization(result));
    }

    lines
}

fn format_optimization(result: OptimizationStepResult) -> String {
    format!(
        "Optimization  status={:?}  profit={}  reserves={}",
        result.status, result.profit_amount, result.reserves_count
    )
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
        ChainObservation::Initializing => "Initializing".to_owned(),
        ChainObservation::Active(ChainProgress {
            verified_pools,
            blocks_behind_tip,
        }) => format!(
            "Active  pools={verified_pools}  behind={}",
            format_distance(blocks_behind_tip)
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
    fn format_lines_renders_initializing_observation() {
        let observations = [(ChainKey::Ethereum, ChainObservation::Initializing)];

        assert_eq!(
            format_lines(&observations, None),
            vec!["Ethereum: Initializing"]
        );
    }

    #[test]
    fn format_lines_renders_arbitrum_chain_label() {
        let observations = [(ChainKey::Arbitrum, ChainObservation::Initializing)];

        assert_eq!(
            format_lines(&observations, None),
            vec!["Arbitrum: Initializing"]
        );
    }

    #[test]
    fn format_lines_renders_active_progress() {
        let observations = [(
            ChainKey::Ethereum,
            ChainObservation::Active(ChainProgress {
                verified_pools: 37,
                blocks_behind_tip: Some(2),
            }),
        )];

        assert_eq!(
            format_lines(&observations, None),
            vec!["Ethereum: Active  pools=37  behind=2"]
        );
    }

    #[test]
    fn format_lines_renders_unknown_distance_as_question_mark() {
        let observations = [(
            ChainKey::Ethereum,
            ChainObservation::Active(ChainProgress {
                verified_pools: 0,
                blocks_behind_tip: None,
            }),
        )];

        assert_eq!(
            format_lines(&observations, None),
            vec!["Ethereum: Active  pools=0  behind=?"]
        );
    }

    #[test]
    fn format_lines_appends_optimization_result_on_its_own_line() {
        let observations = [(ChainKey::Ethereum, ChainObservation::Initializing)];

        assert_eq!(
            format_lines(&observations, Some(result())),
            vec![
                "Ethereum: Initializing".to_owned(),
                "Optimization  status=Initialized  profit=1.5  reserves=3".to_owned(),
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
            format_optimization(result()),
            "Optimization  status=Initialized  profit=1.5  reserves=3"
        );
    }

    fn result() -> OptimizationStepResult {
        OptimizationStepResult {
            status: OptimizationStepStatus::Initialized,
            input_amount: 1.0,
            output_amount: 2.5,
            profit_amount: 1.5,
            reserves_count: 3,
            iterations_completed: 7,
        }
    }
}
