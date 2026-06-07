use std::collections::BTreeMap;

use alloy::primitives::BlockHash;

use crate::{chain::ChainKey, kernel};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChainStatus {
    Initializing,
    Active,
}

pub struct State {
    chains: BTreeMap<ChainKey, ChainLifecycle>,
}

enum ChainLifecycle {
    Initializing,
    Active(kernel::State),
}

impl State {
    pub fn init(chain: ChainKey) -> (State, Vec<Effect>) {
        let mut chains = BTreeMap::new();

        if chains.contains_key(&chain) {
            return (State { chains }, Vec::new());
        }

        chains.insert(chain, ChainLifecycle::Initializing);

        (
            State { chains },
            vec![Effect::FetchFinalizedHeader { chain }],
        )
    }

    pub fn status(&self, chain: ChainKey) -> Option<ChainStatus> {
        self.chains
            .get(&chain)
            .map(|chain_state| match chain_state {
                ChainLifecycle::Initializing => ChainStatus::Initializing,
                ChainLifecycle::Active(_) => ChainStatus::Active,
            })
    }
}

pub enum Event {
    FinalizedHeaderReceived {
        chain: ChainKey,
        block_hash: BlockHash,
    },
    FinalizedHeaderUnavailable {
        chain: ChainKey,
    },
    ChainEvent {
        chain: ChainKey,
        event: kernel::Event,
    },
}

pub enum Effect {
    FetchFinalizedHeader {
        chain: ChainKey,
    },
    ChainEffect {
        chain: ChainKey,
        effect: kernel::Effect,
    },
}

pub fn transition(state: State, event: Event) -> (State, Vec<Effect>) {
    match event {
        Event::FinalizedHeaderReceived { chain, block_hash } => {
            finalized_header_received(state, chain, block_hash)
        }
        Event::FinalizedHeaderUnavailable { chain } => finalized_header_unavailable(state, chain),
        Event::ChainEvent { chain, event } => chain_event(state, chain, event),
    }
}

fn finalized_header_received(
    state: State,
    chain: ChainKey,
    block_hash: BlockHash,
) -> (State, Vec<Effect>) {
    let mut chains = state.chains;

    match chains.remove(&chain) {
        Some(ChainLifecycle::Initializing) => {
            let finalized_state = kernel::FinalizedState::empty_at(block_hash);
            chains.insert(
                chain,
                ChainLifecycle::Active(kernel::State::init(finalized_state)),
            );
        }
        Some(existing_chain) => {
            chains.insert(chain, existing_chain);
        }
        None => {}
    }

    (State { chains }, Vec::new())
}

fn finalized_header_unavailable(state: State, chain: ChainKey) -> (State, Vec<Effect>) {
    let mut chains = state.chains;

    if matches!(chains.get(&chain), Some(ChainLifecycle::Initializing)) {
        chains.remove(&chain);
    }

    (State { chains }, Vec::new())
}

fn chain_event(state: State, chain: ChainKey, event: kernel::Event) -> (State, Vec<Effect>) {
    let mut chains = state.chains;

    match chains.remove(&chain) {
        Some(ChainLifecycle::Active(chain_state)) => {
            let (chain_state, effects) = kernel::transition(chain_state, event);
            chains.insert(chain, ChainLifecycle::Active(chain_state));

            (
                State { chains },
                effects
                    .into_iter()
                    .map(|effect| Effect::ChainEffect { chain, effect })
                    .collect(),
            )
        }
        Some(existing_chain) => {
            chains.insert(chain, existing_chain);
            (State { chains }, Vec::new())
        }
        None => (State { chains }, Vec::new()),
    }
}

#[cfg(test)]
mod tests {
    use alloy::primitives::BlockHash;

    use super::*;
    use crate::kernel;

    #[test]
    fn init_requests_finalized_header_and_marks_chain_initializing() {
        let chain = ChainKey::Ethereum;
        let (state, effects) = State::init(chain);

        assert_eq!(state.status(chain), Some(ChainStatus::Initializing));
        assert_single_fetch_finalized_header_effect(&effects, chain);
    }

    #[test]
    fn finalized_header_received_for_initializing_chain_activates_chain() {
        let chain = ChainKey::Ethereum;
        let finalized_hash = hash(1);
        let child_hash = hash(2);
        let (state, _) = State::init(chain);

        let (state, effects) = transition(
            state,
            Event::FinalizedHeaderReceived {
                chain,
                block_hash: finalized_hash,
            },
        );

        assert_eq!(state.status(chain), Some(ChainStatus::Active));
        assert!(effects.is_empty());

        let (_state, effects) = transition(
            state,
            Event::ChainEvent {
                chain,
                event: kernel::Event::HeadObserved {
                    hash: child_hash,
                    parent_hash: finalized_hash,
                },
            },
        );
        assert!(effects.is_empty());
    }

    #[test]
    fn finalized_header_unavailable_for_initializing_chain_removes_chain() {
        let chain = ChainKey::Ethereum;
        let (state, _) = State::init(chain);

        let (state, effects) = transition(state, Event::FinalizedHeaderUnavailable { chain });

        assert_eq!(state.status(chain), None);
        assert!(effects.is_empty());
    }

    #[test]
    fn chain_event_for_inactive_chain_is_ignored() {
        let chain = ChainKey::Ethereum;
        let (state, _) = State::init(chain);
        let (state, _) = transition(state, Event::FinalizedHeaderUnavailable { chain });

        let (state, effects) = transition(
            state,
            Event::ChainEvent {
                chain,
                event: kernel::Event::Tick,
            },
        );

        assert_eq!(state.status(chain), None);
        assert!(effects.is_empty());
    }

    #[test]
    fn active_chain_event_routes_inner_effects_with_chain_key() {
        let chain = ChainKey::Ethereum;
        let finalized_hash = hash(1);
        let missing_parent_hash = hash(2);
        let observed_hash = hash(3);
        let (state, _) = State::init(chain);
        let (state, _) = transition(
            state,
            Event::FinalizedHeaderReceived {
                chain,
                block_hash: finalized_hash,
            },
        );

        let (_state, effects) = transition(
            state,
            Event::ChainEvent {
                chain,
                event: kernel::Event::HeadObserved {
                    hash: observed_hash,
                    parent_hash: missing_parent_hash,
                },
            },
        );

        assert_single_header_request_chain_effect(&effects, chain, missing_parent_hash);
    }

    fn hash(value: u8) -> BlockHash {
        BlockHash::with_last_byte(value)
    }

    fn assert_single_fetch_finalized_header_effect(effects: &[Effect], chain: ChainKey) {
        assert_eq!(effects.len(), 1);
        assert!(matches!(
            &effects[0],
            Effect::FetchFinalizedHeader { chain: effect_chain }
                if *effect_chain == chain
        ));
    }

    fn assert_single_header_request_chain_effect(
        effects: &[Effect],
        chain: ChainKey,
        block_hash: BlockHash,
    ) {
        assert_eq!(effects.len(), 1);
        match &effects[0] {
            Effect::ChainEffect {
                chain: effect_chain,
                effect:
                    kernel::Effect::Request(crate::pending_requests::AnyIssuedRequest::BlockHeader(
                        request,
                    )),
            } => {
                assert_eq!(*effect_chain, chain);
                assert_eq!(request.request_payload.block_hash, block_hash);
            }
            _ => panic!("expected single chain-tagged block header request"),
        }
    }
}
