use std::io::{self, Write};

use client_evm::{
    ChainKey,
    multi_chain_kernel::{ChainObservation, ChainProgress, State},
};
use optimization::OptimizationStepResult;

/// Renders the current state as an in-place multi-line block on stdout.
///
/// `previously_drawn` is the line count of the prior frame; the block is erased and
/// redrawn in place. Returns the number of lines drawn so the caller can pass it back on
/// the next frame. `State::observe`/`latest_optimization_result` are the pure calls the
/// view makes; the rest is a thin write.
pub(crate) fn render(state: &State, previously_drawn: usize) -> usize {
    let lines = format_lines(&state.observe(), state.latest_optimization_result());

    let mut stdout = io::stdout();
    if previously_drawn > 0 {
        let _ = write!(stdout, "\x1b[{previously_drawn}A");
    }
    let _ = write!(stdout, "\r\x1b[J{}\n", lines.join("\n"));
    let _ = stdout.flush();

    lines.len()
}

/// Terminates the live block so the shell prompt lands on a fresh line.
pub(crate) fn finish() {
    println!();
}

fn format_lines(
    observations: &[(ChainKey, ChainObservation)],
    optimization: Option<OptimizationStepResult>,
) -> Vec<String> {
    let mut lines: Vec<String> = observations
        .iter()
        .map(|(chain, observation)| {
            format!("{}: {}", format_chain(*chain), format_observation(*observation))
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

        assert_eq!(format_lines(&observations, None), vec!["Ethereum: Initializing"]);
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
