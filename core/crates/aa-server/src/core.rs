//! The pure server core: a single-chain lifecycle over the reorg-aware per-chain [`kernel`].
//!
//! The server owns no cross-chain logic — the only thing that couples chains in this codebase is the
//! optimization overlay, which the server does not use. It empty-inits the kernel at the finalized
//! anchor and warms up from live heads plus the kernel's own backfill/discovery scheduling (the
//! validated slow-start path). Empty init is safe because the orphaned-anchor probe re-anchors a bad
//! `finalized` instead of wedging, so a long-running server amortizes the one-time warmup.
//!
//! This module is pure: no I/O, no clocks. The [`crate::runtime`] adapter executes the effects and
//! feeds the events.

use client_evm::{BlockHash, ChainKey, kernel};

/// The single chain this server serves (increment 1: Ethereum only).
pub const CHAIN: ChainKey = ChainKey::Ethereum;

/// A block header reduced to what the finalized anchor needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AnchorHeader {
    pub hash: BlockHash,
    pub number: u64,
}

/// Server lifecycle. Until the finalized anchor arrives there is no kernel state, so a live head or a
/// query before then has nowhere to go — the two states are disjoint by construction (invalid states
/// unrepresentable: you cannot hold a `kernel::State` without an anchor).
// `Running` (holding a `kernel::State`) is the large, permanent state; `AwaitingAnchor` is the brief
// startup one. Boxing the kernel state would add indirection to every transition for the whole
// process lifetime to shrink a startup-only variant — not worth it.
#[allow(clippy::large_enum_variant)]
pub enum ServerState {
    /// Waiting for the initial finalized-header probe that anchors the kernel.
    AwaitingAnchor,
    /// Anchored: warming up and serving.
    Running(kernel::State),
}

/// Inputs to the server core.
// The `Kernel` variant (a full `kernel::Event`) is large and is also the hot, common case; the
// finalized-poll and tick variants are tiny and rare. Boxing the large variant to shrink the rare
// ones would heap-allocate on every event, so accept the size spread instead.
#[allow(clippy::large_enum_variant)]
pub enum ServerInput {
    /// The finalized-header poll result — the initial anchor and every periodic advance; `None` on a
    /// failed or absent fetch.
    FinalizedHeader(Option<AnchorHeader>),
    /// A per-chain kernel event: a live WS head or an RPC (effect) response.
    Kernel(kernel::Event),
    /// The driver tick: advances the kernel retry clock and schedules the next finalized poll.
    Tick,
}

/// Effects the server core requests.
// `Kernel` (a full `kernel::Effect`) is the large, hot variant; `FetchFinalizedHeader` is a tiny,
// rare one. Same rationale as [`ServerInput`]: don't box the hot path to shrink the rare variant.
#[allow(clippy::large_enum_variant)]
pub enum ServerEffect {
    /// A kernel-issued RPC request (the kernel's only effect variant).
    Kernel(kernel::Effect),
    /// The finalized-header poll. Not a kernel effect — the per-chain kernel emits none; driving
    /// finality is the server's job (its response feeds `FinalizedBlockObserved`).
    FetchFinalizedHeader,
}

/// The server's initial transition: no state yet, probe for the anchor.
pub fn server_init() -> (ServerState, Vec<ServerEffect>) {
    (
        ServerState::AwaitingAnchor,
        vec![ServerEffect::FetchFinalizedHeader],
    )
}

/// The pure single-chain lifecycle transition.
pub fn server_transition(
    state: ServerState,
    input: ServerInput,
) -> (ServerState, Vec<ServerEffect>) {
    match (state, input) {
        // ---- AwaitingAnchor: the only thing that matters is landing the anchor ----
        (ServerState::AwaitingAnchor, ServerInput::FinalizedHeader(Some(anchor))) => {
            // Empty kernel init at the true finalized block; warmup proceeds from the next live head.
            (
                ServerState::Running(kernel::State::init(anchor.hash, anchor.number)),
                Vec::new(),
            )
        }
        (ServerState::AwaitingAnchor, ServerInput::FinalizedHeader(None))
        | (ServerState::AwaitingAnchor, ServerInput::Tick) => {
            // No anchor yet: keep probing (the failed fetch, or the tick, reschedules).
            (
                ServerState::AwaitingAnchor,
                vec![ServerEffect::FetchFinalizedHeader],
            )
        }
        (ServerState::AwaitingAnchor, ServerInput::Kernel(_)) => {
            // A head/response before the anchor exists is dropped — warmup starts from the next head.
            (ServerState::AwaitingAnchor, Vec::new())
        }

        // ---- Running: delegate to the kernel, drive finality on tick ----
        (ServerState::Running(kernel_state), ServerInput::Kernel(event)) => {
            let (next, effects) = kernel::transition(CHAIN, kernel_state, event);
            (ServerState::Running(next), kernel_effects(effects))
        }
        (ServerState::Running(kernel_state), ServerInput::FinalizedHeader(Some(anchor))) => {
            let (next, effects) = kernel::transition(
                CHAIN,
                kernel_state,
                kernel::Event::FinalizedBlockObserved {
                    block_hash: anchor.hash,
                    number: anchor.number,
                },
            );
            (ServerState::Running(next), kernel_effects(effects))
        }
        (ServerState::Running(kernel_state), ServerInput::FinalizedHeader(None)) => {
            // Transient fetch failure; the next tick reschedules the poll.
            (ServerState::Running(kernel_state), Vec::new())
        }
        (ServerState::Running(kernel_state), ServerInput::Tick) => {
            // Advance the kernel retry clock, then schedule the next finalized poll so the anchor keeps
            // moving forward and the connected/pending window stays bounded on a long-running process.
            let (next, effects) = kernel::transition(CHAIN, kernel_state, kernel::Event::Tick);
            let mut server_effects = kernel_effects(effects);
            server_effects.push(ServerEffect::FetchFinalizedHeader);
            (ServerState::Running(next), server_effects)
        }
    }
}

fn kernel_effects(effects: Vec<kernel::Effect>) -> Vec<ServerEffect> {
    effects.into_iter().map(ServerEffect::Kernel).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use client_evm::Bloom;

    fn hash(byte: u8) -> BlockHash {
        BlockHash::with_last_byte(byte)
    }

    fn head_input(number: u64, hash: BlockHash, parent: BlockHash) -> ServerInput {
        ServerInput::Kernel(kernel::Event::HeadObserved {
            hash,
            parent_hash: parent,
            // Empty bloom: with no verified pools the kernel watches nothing, so a zero bloom keeps
            // every block a non-blocker and finalization can advance across the connected window.
            logs_bloom: Bloom::ZERO,
            number,
        })
    }

    fn only_finalized_probe(effects: &[ServerEffect]) -> bool {
        matches!(effects, [ServerEffect::FetchFinalizedHeader])
    }

    #[test]
    fn init_awaits_the_anchor_and_probes_finalized_once() {
        let (state, effects) = server_init();

        assert!(matches!(state, ServerState::AwaitingAnchor));
        assert!(only_finalized_probe(&effects));
    }

    #[test]
    fn finalized_header_while_awaiting_inits_running_at_that_anchor() {
        let (state, effects) = server_transition(
            ServerState::AwaitingAnchor,
            ServerInput::FinalizedHeader(Some(AnchorHeader {
                hash: hash(100),
                number: 100,
            })),
        );

        assert!(effects.is_empty());
        match state {
            ServerState::Running(kernel_state) => {
                assert_eq!(kernel_state.finalized_head(), (hash(100), 100));
                // Empty init: the observed head starts at the anchor, nothing tracked yet.
                assert_eq!(kernel_state.canonical_head(), hash(100));
                assert_eq!(kernel_state.verified_pool_count(), 0);
            }
            ServerState::AwaitingAnchor => panic!("expected Running after the anchor landed"),
        }
    }

    #[test]
    fn failed_probe_while_awaiting_reschedules_the_probe() {
        let (state, effects) =
            server_transition(ServerState::AwaitingAnchor, ServerInput::FinalizedHeader(None));

        assert!(matches!(state, ServerState::AwaitingAnchor));
        assert!(only_finalized_probe(&effects));
    }

    #[test]
    fn tick_while_awaiting_reschedules_the_probe() {
        let (state, effects) = server_transition(ServerState::AwaitingAnchor, ServerInput::Tick);

        assert!(matches!(state, ServerState::AwaitingAnchor));
        assert!(only_finalized_probe(&effects));
    }

    #[test]
    fn kernel_event_while_awaiting_is_dropped() {
        let (state, effects) = server_transition(
            ServerState::AwaitingAnchor,
            head_input(101, hash(101), hash(100)),
        );

        assert!(matches!(state, ServerState::AwaitingAnchor));
        assert!(effects.is_empty());
    }

    #[test]
    fn tick_while_running_advances_the_retry_clock_and_schedules_a_finalized_poll() {
        let (running, _) = server_transition(
            ServerState::AwaitingAnchor,
            ServerInput::FinalizedHeader(Some(AnchorHeader {
                hash: hash(100),
                number: 100,
            })),
        );

        let (state, effects) = server_transition(running, ServerInput::Tick);

        assert!(matches!(state, ServerState::Running(_)));
        // The finality driver is wired: a tick always ends by scheduling the next finalized poll.
        assert!(
            effects
                .iter()
                .any(|effect| matches!(effect, ServerEffect::FetchFinalizedHeader)),
            "tick must schedule a finalized poll"
        );
    }

    #[test]
    fn finalized_observation_while_running_advances_the_anchor_over_a_connected_window() {
        // Empty init at anchor@100.
        let (mut state, _) = server_transition(
            ServerState::AwaitingAnchor,
            ServerInput::FinalizedHeader(Some(AnchorHeader {
                hash: hash(100),
                number: 100,
            })),
        );

        // Feed a connected chain 101..104, each linking to its parent down to the anchor.
        let mut parent = hash(100);
        for number in 101..=104u64 {
            let child = hash(number as u8);
            let (next, _) = server_transition(state, head_input(number, child, parent));
            state = next;
            parent = child;
        }

        // Observe finality at 102 (inside the connected window): the anchor must advance to it.
        let (state, _) = server_transition(
            state,
            ServerInput::FinalizedHeader(Some(AnchorHeader {
                hash: hash(102),
                number: 102,
            })),
        );

        match state {
            ServerState::Running(kernel_state) => {
                assert_eq!(kernel_state.finalized_head(), (hash(102), 102));
            }
            ServerState::AwaitingAnchor => panic!("expected Running"),
        }
    }
}
