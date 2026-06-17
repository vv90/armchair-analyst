use std::io::{self, Write};

use client_evm::{
    ChainKey,
    multi_chain_kernel::{ChainObservation, ChainProgress, State},
};

/// Renders the current state as a single in-place status line on stdout.
///
/// `State::observe` is the one pure call the view makes; the rest is a thin write.
pub(crate) fn render(state: &State) {
    let observations = state.observe();

    let line = format_view(&observations);
    let mut stdout = io::stdout();
    let _ = write!(stdout, "\r\x1b[K{line}");
    let _ = stdout.flush();
}

/// Terminates the live status line so the shell prompt lands on a fresh line.
pub(crate) fn finish() {
    println!();
}

fn format_view(observations: &[(ChainKey, ChainObservation)]) -> String {
    observations
        .iter()
        .map(|(chain, observation)| {
            format!("{}: {}", format_chain(*chain), format_observation(*observation))
        })
        .collect::<Vec<_>>()
        .join("   ")
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
    use super::*;

    #[test]
    fn format_view_renders_initializing_observation() {
        let observations = [(ChainKey::Ethereum, ChainObservation::Initializing)];

        assert_eq!(format_view(&observations), "Ethereum: Initializing");
    }

    #[test]
    fn format_view_renders_active_progress() {
        let observations = [(
            ChainKey::Ethereum,
            ChainObservation::Active(ChainProgress {
                verified_pools: 37,
                blocks_behind_tip: Some(2),
            }),
        )];

        assert_eq!(format_view(&observations), "Ethereum: Active  pools=37  behind=2");
    }

    #[test]
    fn format_view_renders_unknown_distance_as_question_mark() {
        let observations = [(
            ChainKey::Ethereum,
            ChainObservation::Active(ChainProgress {
                verified_pools: 0,
                blocks_behind_tip: None,
            }),
        )];

        assert_eq!(format_view(&observations), "Ethereum: Active  pools=0  behind=?");
    }
}
