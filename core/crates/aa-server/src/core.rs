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

use std::collections::HashMap;

use client_evm::{BlockHash, ChainKey, TokenRegistry, TrustedPoolRegistry, kernel};

/// The single chain this server serves (increment 1: Ethereum only).
pub const CHAIN: ChainKey = ChainKey::Ethereum;

/// A block header reduced to what the finalized anchor needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AnchorHeader {
    pub hash: BlockHash,
    pub number: u64,
}

/// The disk-loaded warm-start seed: the pool and token registries re-hydrated from the persistent
/// metadata cache. Consumed exactly once, when the anchor lands, to activate the kernel already
/// knowing every previously-validated pool/token — so a restart does not re-discover them from
/// scratch. Built by the runtime (disk I/O); the pure core only folds it into the kernel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegistrySeed {
    pub pool_registry: TrustedPoolRegistry,
    pub token_registry: TokenRegistry,
}

/// Server lifecycle. The warm-start seed is loaded from disk first, then the anchor is probed, then
/// the kernel is activated with the seed folded in — a strictly linear progression. Each state makes
/// the next-invalid inputs unrepresentable: there is no `kernel::State` before the anchor, and the
/// seed exists only in `AwaitingAnchor`, consumed into `Running` (so it can never ride a recurring
/// input).
// `Running` (holding a `kernel::State`) is the large, permanent state; the awaiting variants are the
// brief startup ones. Boxing the kernel state would add indirection to every transition for the whole
// process lifetime to shrink startup-only variants — not worth it. The seed is boxed so the awaiting
// variant stays a pointer.
#[allow(clippy::large_enum_variant)]
pub enum ServerState {
    /// Initial: the disk re-hydration of the metadata registry is in flight.
    AwaitingSeed,
    /// Seed loaded; waiting for the initial finalized-header probe that anchors the kernel.
    AwaitingAnchor { seed: Box<RegistrySeed> },
    /// Anchored: warming up and serving.
    Running(kernel::State),
}

/// Inputs to the server core.
// The `Kernel` variant (a full `kernel::Event`) is large and is also the hot, common case; the
// finalized-poll, seed, and tick variants are tiny/rare. Boxing the large variant to shrink the rare
// ones would heap-allocate on every event, so accept the size spread instead (the seed is already
// boxed).
#[allow(clippy::large_enum_variant)]
pub enum ServerInput {
    /// The disk-loaded warm-start seed — produced exactly once, by executing `LoadRegistrySeed`.
    RegistrySeed(Box<RegistrySeed>),
    /// The finalized-header poll result — the initial anchor and every periodic advance; `None` on a
    /// failed or absent fetch.
    FinalizedHeader(Option<AnchorHeader>),
    /// A per-chain kernel event: a live WS head or an RPC (effect) response.
    Kernel(kernel::Event),
    /// The driver tick: advances the kernel retry clock and schedules the next finalized poll.
    Tick,
}

/// Effects the server core requests.
// `Kernel` (a full `kernel::Effect`) is the large, hot variant; the others are tiny, rare ones. Same
// rationale as [`ServerInput`]: don't box the hot path to shrink the rare variants.
#[allow(clippy::large_enum_variant)]
pub enum ServerEffect {
    /// A kernel-issued RPC request (the kernel's only effect variant).
    Kernel(kernel::Effect),
    /// The finalized-header poll. Not a kernel effect — the per-chain kernel emits none; driving
    /// finality is the server's job (its response feeds `FinalizedBlockObserved`).
    FetchFinalizedHeader,
    /// The one-shot warm-start disk read: re-hydrate the metadata registry from the persistent cache.
    /// Emitted only by [`server_init`] so it fires exactly once — never on a recurring path.
    LoadRegistrySeed,
}

/// The server's initial transition: no state yet, re-hydrate the registry from disk first. The anchor
/// probe is chained after the seed lands so the two never race.
pub fn server_init() -> (ServerState, Vec<ServerEffect>) {
    (
        ServerState::AwaitingSeed,
        vec![ServerEffect::LoadRegistrySeed],
    )
}

/// The pure single-chain lifecycle transition.
pub fn server_transition(
    state: ServerState,
    input: ServerInput,
) -> (ServerState, Vec<ServerEffect>) {
    match (state, input) {
        // ---- AwaitingSeed: hold until the one-shot disk re-hydration lands, then probe ----
        (ServerState::AwaitingSeed, ServerInput::RegistrySeed(seed)) => {
            // Seed loaded: advance to awaiting the anchor and start the finalized probe now (chained
            // after the seed so the two never race).
            (
                ServerState::AwaitingAnchor { seed },
                vec![ServerEffect::FetchFinalizedHeader],
            )
        }
        (ServerState::AwaitingSeed, _) => {
            // The single `LoadRegistrySeed` input is imminent; a tick / stray anchor / head before it
            // is an inert wait. Never re-emit the load (it must fire exactly once).
            (ServerState::AwaitingSeed, Vec::new())
        }

        // ---- AwaitingAnchor: seed in hand, the only thing that matters is landing the anchor ----
        (ServerState::AwaitingAnchor { seed }, ServerInput::FinalizedHeader(Some(anchor))) => {
            // Activate the kernel at the true finalized block with the re-hydrated registries folded
            // in — warm from the first anchor. Pool *state* is not seeded here (per-block, never
            // cached); the kernel's anchor-height `GetPoolData` seeding covers those pools from the
            // kernel loop. `activate_from_seed` with empty snapshots/blocks is `init` plus registries.
            let RegistrySeed {
                pool_registry,
                token_registry,
            } = *seed;
            let (kernel_state, effects) = kernel::State::activate_from_seed(
                anchor.hash,
                anchor.number,
                HashMap::new(),
                pool_registry,
                token_registry,
                Vec::new(),
            );
            (ServerState::Running(kernel_state), kernel_effects(effects))
        }
        (ServerState::AwaitingAnchor { seed }, ServerInput::FinalizedHeader(None))
        | (ServerState::AwaitingAnchor { seed }, ServerInput::Tick) => {
            // No anchor yet: keep probing (the failed fetch, or the tick, reschedules). Seed retained.
            (
                ServerState::AwaitingAnchor { seed },
                vec![ServerEffect::FetchFinalizedHeader],
            )
        }
        (ServerState::AwaitingAnchor { seed }, ServerInput::Kernel(_)) => {
            // A head/response before the anchor exists is dropped — warmup starts from the next head.
            // Seed retained.
            (ServerState::AwaitingAnchor { seed }, Vec::new())
        }
        (ServerState::AwaitingAnchor { seed }, ServerInput::RegistrySeed(_)) => {
            // Structurally impossible: the one-shot seed was consumed in `AwaitingSeed`. Drop it
            // rather than replace the held seed (mirrors the pre-anchor `Kernel(_)` drop above).
            (ServerState::AwaitingAnchor { seed }, Vec::new())
        }

        // ---- Running: delegate to the kernel, drive finality on tick ----
        (ServerState::Running(kernel_state), ServerInput::RegistrySeed(_)) => {
            // Structurally impossible (the one-shot seed fired once, before the kernel existed).
            // Inert no-op — never re-seed a live kernel.
            (ServerState::Running(kernel_state), Vec::new())
        }
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
    use client_evm::{
        Address, Bloom, PoolFee, PoolMetadata, PoolRef, ProtocolPoolKey, TokenAddress,
        TokenDecimals, TokenMetadata, U256, UniswapV3Fee,
    };

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

    fn empty_seed() -> Box<RegistrySeed> {
        Box::new(RegistrySeed {
            pool_registry: TrustedPoolRegistry::new(),
            token_registry: TokenRegistry::new(),
        })
    }

    /// A seed carrying one verified v3 pool (tokens `1`/`2`) and both token decimals — the warm-start
    /// re-hydration content.
    fn seed_with_pool() -> Box<RegistrySeed> {
        let candidate = ProtocolPoolKey::UniswapV3(Address::with_last_byte(7));
        let metadata = PoolMetadata {
            token0: Address::with_last_byte(1),
            token1: Address::with_last_byte(2),
            fee: PoolFee::Tiered(UniswapV3Fee::Fee3000),
        };
        let pool_registry = TrustedPoolRegistry::new()
            .with_metadata_results(CHAIN, HashMap::from([(candidate, Ok(metadata))]));
        let token_registry = TokenRegistry::new().with_metadata_results(HashMap::from([
            (
                TokenAddress(Address::with_last_byte(1), CHAIN),
                Ok(decimals(18)),
            ),
            (
                TokenAddress(Address::with_last_byte(2), CHAIN),
                Ok(decimals(6)),
            ),
        ]));
        Box::new(RegistrySeed {
            pool_registry,
            token_registry,
        })
    }

    fn decimals(value: u8) -> TokenMetadata {
        TokenMetadata {
            decimals: TokenDecimals::try_from_u256(U256::from(value)).expect("decimals in range"),
        }
    }

    /// Drives `AwaitingSeed → AwaitingAnchor{seed} → Running` at `number` with the given seed.
    fn running_with_seed(number: u64, seed: Box<RegistrySeed>) -> ServerState {
        let (state, _) = server_transition(
            ServerState::AwaitingAnchor { seed },
            ServerInput::FinalizedHeader(Some(AnchorHeader {
                hash: hash(number as u8),
                number,
            })),
        );
        state
    }

    #[test]
    fn init_awaits_the_seed_and_loads_it_once() {
        let (state, effects) = server_init();

        assert!(matches!(state, ServerState::AwaitingSeed));
        // The one-shot disk re-hydration is the only startup effect; the anchor probe waits for it.
        assert!(matches!(
            effects.as_slice(),
            [ServerEffect::LoadRegistrySeed]
        ));
    }

    #[test]
    fn seed_while_awaiting_seed_advances_to_awaiting_anchor_and_probes() {
        let (state, effects) = server_transition(
            ServerState::AwaitingSeed,
            ServerInput::RegistrySeed(empty_seed()),
        );

        assert!(matches!(state, ServerState::AwaitingAnchor { .. }));
        assert!(only_finalized_probe(&effects));
    }

    #[test]
    fn tick_while_awaiting_seed_is_an_inert_wait() {
        let (state, effects) = server_transition(ServerState::AwaitingSeed, ServerInput::Tick);

        assert!(matches!(state, ServerState::AwaitingSeed));
        assert!(effects.is_empty());
    }

    #[test]
    fn finalized_header_while_awaiting_anchor_activates_running_with_the_seed() {
        let (state, effects) = server_transition(
            ServerState::AwaitingAnchor {
                seed: seed_with_pool(),
            },
            ServerInput::FinalizedHeader(Some(AnchorHeader {
                hash: hash(100),
                number: 100,
            })),
        );

        // Empty seed_blocks/snapshots ⇒ `activate_from_seed` issues no effects; the re-hydrated pool's
        // state is seeded later from the kernel loop.
        assert!(effects.is_empty());
        match state {
            ServerState::Running(kernel_state) => {
                assert_eq!(kernel_state.finalized_head(), (hash(100), 100));
                assert_eq!(kernel_state.canonical_head(), hash(100));
                // The re-hydrated registry is warm from the first anchor.
                assert_eq!(kernel_state.verified_pool_count(), 1);
                assert_eq!(
                    kernel_state.verified_pool_metadata(PoolRef {
                        key: ProtocolPoolKey::UniswapV3(Address::with_last_byte(7)),
                        chain: CHAIN,
                    }),
                    Some(&PoolMetadata {
                        token0: Address::with_last_byte(1),
                        token1: Address::with_last_byte(2),
                        fee: PoolFee::Tiered(UniswapV3Fee::Fee3000),
                    })
                );
                assert_eq!(
                    kernel_state
                        .verified_token_metadata(TokenAddress(Address::with_last_byte(1), CHAIN)),
                    Some(&decimals(18))
                );
            }
            _ => panic!("expected Running after the anchor landed"),
        }
    }

    #[test]
    fn empty_seed_activates_running_with_an_empty_registry() {
        match running_with_seed(100, empty_seed()) {
            ServerState::Running(kernel_state) => {
                assert_eq!(kernel_state.finalized_head(), (hash(100), 100));
                assert_eq!(kernel_state.verified_pool_count(), 0);
            }
            _ => panic!("expected Running"),
        }
    }

    #[test]
    fn failed_probe_while_awaiting_anchor_retains_seed_and_reprobes() {
        let (state, effects) = server_transition(
            ServerState::AwaitingAnchor { seed: empty_seed() },
            ServerInput::FinalizedHeader(None),
        );

        assert!(matches!(state, ServerState::AwaitingAnchor { .. }));
        assert!(only_finalized_probe(&effects));
    }

    #[test]
    fn tick_while_awaiting_anchor_retains_seed_and_reprobes() {
        let (state, effects) = server_transition(
            ServerState::AwaitingAnchor { seed: empty_seed() },
            ServerInput::Tick,
        );

        assert!(matches!(state, ServerState::AwaitingAnchor { .. }));
        assert!(only_finalized_probe(&effects));
    }

    #[test]
    fn kernel_event_while_awaiting_anchor_is_dropped() {
        let (state, effects) = server_transition(
            ServerState::AwaitingAnchor { seed: empty_seed() },
            head_input(101, hash(101), hash(100)),
        );

        assert!(matches!(state, ServerState::AwaitingAnchor { .. }));
        assert!(effects.is_empty());
    }

    #[test]
    fn stray_seed_while_running_is_inert() {
        // The one-shot seed already fired before the kernel existed; a second one is a dead no-op.
        let (state, effects) = server_transition(
            running_with_seed(100, empty_seed()),
            ServerInput::RegistrySeed(seed_with_pool()),
        );

        match state {
            ServerState::Running(kernel_state) => {
                // Not re-seeded: the live kernel's (empty) registry is untouched.
                assert_eq!(kernel_state.verified_pool_count(), 0);
            }
            _ => panic!("expected Running"),
        }
        assert!(effects.is_empty());
    }

    #[test]
    fn tick_while_running_advances_the_retry_clock_and_schedules_a_finalized_poll() {
        let running = running_with_seed(100, empty_seed());

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
        let mut state = running_with_seed(100, empty_seed());

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
            _ => panic!("expected Running"),
        }
    }
}
