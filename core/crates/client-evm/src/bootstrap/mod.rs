//! Per-chain bootstrap phase: fetches a recent canonical window ahead of activation so the
//! kernel starts with seeded finalized pool snapshots and block graph instead of warming up
//! from empty state. Mirrors the `kernel` module's structure; orchestration is pure.

pub(crate) mod pending_requests;

use std::collections::{BTreeMap, HashMap, HashSet};

use alloy::primitives::{BlockHash, keccak256};

// Pool/token/data request payloads are shared with the kernel — reused here to keep a single
// definition rather than duplicating the request models.
use crate::{
    ChainKey, PoolRef, PoolCandidateAddress, PoolMetadataResult, PoolState, RangeLogBlock,
    TokenAddress, TokenMetadataResult, TokenRegistry, TrustedPoolRegistry, tick::Tick,
};
pub use crate::{GetPoolMetadata, GetTokenMetadata};
use pending_requests::PendingRequests;
// Re-exported so runtimes can dispatch bootstrap requests and build their response events, mirroring
// the kernel's crate-level request re-exports.
pub use pending_requests::{
    AnyIssuedRequest, AnyRequestId, GetFinalizedHeader, GetPoolCandidatesInRange, IssuedRequest,
    RequestId,
};

/// Per-chain tuning for the bootstrap phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BootstrapPolicy {
    /// Reorg-prone blocks nearest the tip to leave out of the seeded graph.
    pub tip_trim: usize,
    /// Ticks after which bootstrap gives up and activates best-effort with what it has.
    pub deadline_ticks: u64,
}

/// The finalized block the bootstrap anchors on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FinalizedAnchor {
    pub hash: BlockHash,
    pub number: u64,
}

/// One recent canonical block to pre-insert into the kernel block graph, with its pool-event
/// candidates. `parent_hash` is inferred from block-number adjacency in the range-logs snapshot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SeedBlock {
    pub hash: BlockHash,
    pub parent_hash: BlockHash,
    pub candidates: HashSet<PoolCandidateAddress>,
}

/// Inferred block-graph seed for the `finalized..tip` window (parents by number adjacency).
#[derive(Debug, PartialEq, Eq)]
pub struct GraphSeed {
    blocks: Vec<SeedBlock>,
}

/// Domain tag distinguishing a filler-hash preimage from any other keccak input.
const FILLER_DOMAIN: &[u8] = b"aa-gap-filler";

/// Deterministic synthetic hash for the filler block bridging a no-log gap between two real blocks.
///
/// Derived as a domain-separated keccak of the two real block hashes it connects, so it is
/// reproducible (bootstrap is pure), unique per gap, and — by keccak preimage resistance — collides
/// with neither a real block hash nor its own `lower`/`upper` bounds. A filler carries no logs and no
/// pool state; its only role is to thread one `parent_hash` link across the gap.
fn filler_hash(lower: BlockHash, upper: BlockHash) -> BlockHash {
    let mut preimage = Vec::with_capacity(FILLER_DOMAIN.len() + 64);
    preimage.extend_from_slice(FILLER_DOMAIN);
    preimage.extend_from_slice(lower.as_slice());
    preimage.extend_from_slice(upper.as_slice());
    keccak256(preimage)
}

impl GraphSeed {
    /// Infers the `finalized..tip` block graph from a ranged-logs snapshot.
    ///
    /// A single `eth_getLogs` response is one canonical view, so consecutive retained blocks are
    /// parent-linked by number adjacency. No-log blocks are absent from the response, leaving gaps;
    /// rather than dropping the blocks above a gap (which would force a per-block header walk to
    /// reconnect them), each gap is bridged by a single synthetic [`filler_hash`] block whose only
    /// role is connectivity. Every retained block — and the run anchored above the finalized block —
    /// therefore links down to the anchor. The top `tip_trim` blocks near the tip are reorg-prone and
    /// left out.
    fn from_window(
        window: &[RangeLogBlock],
        anchor: FinalizedAnchor,
        tip_trim: usize,
    ) -> GraphSeed {
        // Recent canonical blocks above the anchor, keyed by number (deduped, ascending).
        let mut recent: BTreeMap<u64, &RangeLogBlock> = BTreeMap::new();
        for block in window {
            if block.number > anchor.number {
                recent.entry(block.number).or_insert(block);
            }
        }

        let Some(max_number) = recent.keys().next_back().copied() else {
            return GraphSeed { blocks: Vec::new() };
        };

        // Drop the reorg-prone blocks within `tip_trim` of the observed tip. The highest retained
        // block stays a real log-bearing block, which the live head-walk relies on as a join point.
        let highest_retained = max_number.saturating_sub(tip_trim as u64);
        let retained: BTreeMap<u64, &RangeLogBlock> = recent
            .into_iter()
            .filter(|(number, _)| *number <= highest_retained)
            .collect();

        // Walk retained blocks oldest→newest, tracking the previous connected block (starting at the
        // anchor). Adjacent blocks link directly; a gap inserts one filler bridging the previous
        // connected hash to this block, so nothing is ever dropped.
        let mut blocks = Vec::new();
        let mut previous_number = anchor.number;
        let mut previous_hash = anchor.hash;

        for (number, block) in &retained {
            let parent_hash = if *number == previous_number + 1 {
                previous_hash
            } else {
                let filler = filler_hash(previous_hash, block.hash);
                blocks.push(SeedBlock {
                    hash: filler,
                    parent_hash: previous_hash,
                    candidates: HashSet::new(),
                });
                filler
            };

            blocks.push(SeedBlock {
                hash: block.hash,
                parent_hash,
                candidates: block.candidates.clone(),
            });

            previous_number = *number;
            previous_hash = block.hash;
        }

        GraphSeed { blocks }
    }

    fn into_blocks(self) -> Vec<SeedBlock> {
        self.blocks
    }
}

/// Validated pool metadata results, the single source for the verified pool set, their tokens,
/// and the trusted-pool registry handed to the kernel — none of which are stored redundantly.
#[derive(Clone, Debug, PartialEq, Eq)]
struct VerifiedPools {
    metadata: HashMap<PoolCandidateAddress, PoolMetadataResult>,
}

impl VerifiedPools {
    fn referenced_tokens(&self, chain: ChainKey) -> HashSet<TokenAddress> {
        self.metadata
            .values()
            .filter_map(|result| result.as_ref().ok())
            .flat_map(|metadata| {
                [
                    TokenAddress(metadata.token0, chain),
                    TokenAddress(metadata.token1, chain),
                ]
            })
            .collect()
    }

    fn into_registry(self, chain: ChainKey) -> TrustedPoolRegistry {
        TrustedPoolRegistry::new().with_metadata_results(chain, self.metadata)
    }
}

/// Everything the bootstrap accumulated, ready to construct an active kernel state (Stage 2).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BootstrapOutcome {
    pub anchor: FinalizedAnchor,
    pub pool_snapshots: HashMap<PoolRef, PoolState>,
    pub pool_registry: TrustedPoolRegistry,
    pub token_registry: TokenRegistry,
    pub seed_blocks: Vec<SeedBlock>,
}

/// Terminal result of the bootstrap phase.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Completion {
    /// Activate the chain with this seeded state (full or best-effort degraded).
    Ready(BootstrapOutcome),
    /// The finalized anchor could never be obtained; the chain should be dropped.
    Abandoned,
}

/// Sequential bootstrap phase. Each variant carries exactly the facts proven by reaching it.
enum Phase {
    AnchoringChain,
    Discovering {
        anchor: FinalizedAnchor,
    },
    ValidatingPools {
        anchor: FinalizedAnchor,
        seed: GraphSeed,
    },
    ResolvingTokens {
        anchor: FinalizedAnchor,
        seed: GraphSeed,
        verified: VerifiedPools,
    },
    Ready(BootstrapOutcome),
    Abandoned,
}

pub struct State {
    pending: PendingRequests,
    phase: Phase,
    policy: BootstrapPolicy,
    started_at: Tick,
    tick: Tick,
}

pub enum Event {
    FinalizedHeaderReceived {
        request_id: RequestId<GetFinalizedHeader>,
        anchor: FinalizedAnchor,
    },
    PoolCandidatesReceived {
        request_id: RequestId<GetPoolCandidatesInRange>,
        blocks: Vec<RangeLogBlock>,
    },
    PoolMetadataReceived {
        request_id: RequestId<GetPoolMetadata>,
        metadata: HashMap<PoolCandidateAddress, PoolMetadataResult>,
    },
    TokenMetadataReceived {
        request_id: RequestId<GetTokenMetadata>,
        metadata: HashMap<TokenAddress, TokenMetadataResult>,
    },
    RequestFailed {
        request_id: AnyRequestId,
    },
    Tick,
}

pub enum Effect {
    Request(AnyIssuedRequest),
}

/// Starts a chain bootstrap by requesting the finalized header that will anchor it.
pub fn init(policy: BootstrapPolicy) -> (State, Vec<Effect>) {
    let tick = Tick::initial();
    let (pending, request_id) = PendingRequests::new().with_new_request(GetFinalizedHeader, tick);

    let state = State {
        pending,
        phase: Phase::AnchoringChain,
        policy,
        started_at: tick,
        tick,
    };

    (
        state,
        vec![Effect::Request(AnyIssuedRequest::FinalizedHeader(
            IssuedRequest {
                request_id,
                request_payload: GetFinalizedHeader,
            },
        ))],
    )
}

/// Reports the terminal result once the bootstrap has finished or given up.
pub fn completion(state: &State) -> Option<Completion> {
    match &state.phase {
        Phase::Ready(outcome) => Some(Completion::Ready(outcome.clone())),
        Phase::Abandoned => Some(Completion::Abandoned),
        _ => None,
    }
}

/// Applies one bootstrap event, advancing the phase and issuing the next request.
pub fn transition(chain: ChainKey, state: State, event: Event) -> (State, Vec<Effect>) {
    let State {
        pending,
        phase,
        policy,
        started_at,
        tick,
    } = state;

    match event {
        Event::FinalizedHeaderReceived { request_id, anchor } => {
            let (pending, taken) = pending.take(&request_id);
            match phase {
                Phase::AnchoringChain if taken.is_some() => {
                    // Scan only `finalized..tip`: the seed graph already discards every block at or
                    // below the anchor, and the historical pool set below finalized is owned by the
                    // persistent metadata cache (unioned into the candidate set by the runtime).
                    let payload = GetPoolCandidatesInRange {
                        from_block: anchor.number,
                    };
                    let (pending, request_id) = pending.with_new_request(payload.clone(), tick);

                    (
                        rebuild(
                            pending,
                            Phase::Discovering { anchor },
                            policy,
                            started_at,
                            tick,
                        ),
                        vec![issued(
                            AnyIssuedRequest::PoolCandidates,
                            request_id,
                            payload,
                        )],
                    )
                }
                phase => (
                    rebuild(pending, phase, policy, started_at, tick),
                    Vec::new(),
                ),
            }
        }
        Event::PoolCandidatesReceived { request_id, blocks } => {
            let (pending, taken) = pending.take(&request_id);
            match phase {
                Phase::Discovering { anchor } if taken.is_some() => {
                    let seed = GraphSeed::from_window(&blocks, anchor, policy.tip_trim);
                    let candidates = blocks
                        .into_iter()
                        .flat_map(|block| block.candidates)
                        .collect();
                    let payload = GetPoolMetadata {
                        at: anchor.hash,
                        candidates,
                    };
                    let (pending, request_id) = pending.with_new_request(payload.clone(), tick);

                    (
                        rebuild(
                            pending,
                            Phase::ValidatingPools { anchor, seed },
                            policy,
                            started_at,
                            tick,
                        ),
                        vec![issued(AnyIssuedRequest::PoolMetadata, request_id, payload)],
                    )
                }
                phase => (
                    rebuild(pending, phase, policy, started_at, tick),
                    Vec::new(),
                ),
            }
        }
        Event::PoolMetadataReceived {
            request_id,
            metadata,
        } => {
            let (pending, taken) = pending.take(&request_id);
            match phase {
                Phase::ValidatingPools { anchor, seed } if taken.is_some() => {
                    let verified = VerifiedPools { metadata };
                    let payload = GetTokenMetadata {
                        at: anchor.hash,
                        tokens: verified.referenced_tokens(chain),
                    };
                    let (pending, request_id) = pending.with_new_request(payload.clone(), tick);

                    (
                        rebuild(
                            pending,
                            Phase::ResolvingTokens {
                                anchor,
                                seed,
                                verified,
                            },
                            policy,
                            started_at,
                            tick,
                        ),
                        vec![issued(AnyIssuedRequest::TokenMetadata, request_id, payload)],
                    )
                }
                phase => (
                    rebuild(pending, phase, policy, started_at, tick),
                    Vec::new(),
                ),
            }
        }
        Event::TokenMetadataReceived {
            request_id,
            metadata,
        } => {
            let (pending, taken) = pending.take(&request_id);
            match phase {
                Phase::ResolvingTokens {
                    anchor,
                    seed,
                    verified,
                } if taken.is_some() => {
                    // Activation no longer blocks on a pool-state snapshot: hand the kernel an empty
                    // finalized snapshot and let its live dirty-pool / backfill paths fetch state at
                    // the tip after activation, prioritizing frontier freshness over a complete set.
                    let outcome = BootstrapOutcome {
                        anchor,
                        pool_snapshots: HashMap::new(),
                        pool_registry: verified.into_registry(chain),
                        token_registry: TokenRegistry::new().with_metadata_results(metadata),
                        seed_blocks: seed.into_blocks(),
                    };

                    (
                        rebuild(pending, Phase::Ready(outcome), policy, started_at, tick),
                        Vec::new(),
                    )
                }
                phase => (
                    rebuild(pending, phase, policy, started_at, tick),
                    Vec::new(),
                ),
            }
        }
        Event::RequestFailed { request_id } => {
            let (pending, issued_request) = pending.retry(request_id, tick);

            (
                rebuild(pending, phase, policy, started_at, tick),
                issued_request.into_iter().map(Effect::Request).collect(),
            )
        }
        Event::Tick => {
            let tick = tick.next();

            if matches!(phase, Phase::Ready(_) | Phase::Abandoned) {
                return (
                    rebuild(pending, phase, policy, started_at, tick),
                    Vec::new(),
                );
            }

            if tick.elapsed_since(started_at) >= policy.deadline_ticks {
                return (
                    rebuild(pending, degraded(chain, phase), policy, started_at, tick),
                    Vec::new(),
                );
            }

            let (pending, issued_requests) = pending.retry_expired(tick);

            (
                rebuild(pending, phase, policy, started_at, tick),
                issued_requests.into_iter().map(Effect::Request).collect(),
            )
        }
    }
}

fn rebuild(
    pending: PendingRequests,
    phase: Phase,
    policy: BootstrapPolicy,
    started_at: Tick,
    tick: Tick,
) -> State {
    State {
        pending,
        phase,
        policy,
        started_at,
        tick,
    }
}

fn issued<R>(
    wrap: fn(IssuedRequest<R>) -> AnyIssuedRequest,
    request_id: RequestId<R>,
    request_payload: R,
) -> Effect {
    Effect::Request(wrap(IssuedRequest {
        request_id,
        request_payload,
    }))
}

/// Builds the best-effort terminal phase when the deadline is reached, carrying exactly the
/// facts the current phase has accumulated. The pre-anchor case cannot activate, so it abandons.
fn degraded(chain: ChainKey, phase: Phase) -> Phase {
    match phase {
        Phase::AnchoringChain => Phase::Abandoned,
        Phase::Discovering { anchor } => Phase::Ready(BootstrapOutcome {
            anchor,
            pool_snapshots: HashMap::new(),
            pool_registry: TrustedPoolRegistry::new(),
            token_registry: TokenRegistry::new(),
            seed_blocks: Vec::new(),
        }),
        Phase::ValidatingPools { anchor, seed } => Phase::Ready(BootstrapOutcome {
            anchor,
            pool_snapshots: HashMap::new(),
            pool_registry: TrustedPoolRegistry::new(),
            token_registry: TokenRegistry::new(),
            seed_blocks: seed.into_blocks(),
        }),
        Phase::ResolvingTokens {
            anchor,
            seed,
            verified,
        } => Phase::Ready(BootstrapOutcome {
            anchor,
            pool_snapshots: HashMap::new(),
            pool_registry: verified.into_registry(chain),
            token_registry: TokenRegistry::new(),
            seed_blocks: seed.into_blocks(),
        }),
        terminal @ (Phase::Ready(_) | Phase::Abandoned) => terminal,
    }
}

#[cfg(test)]
mod tests {
    use alloy::primitives::{Address, U256};
    use proptest::prelude::*;

    use super::*;
    use crate::{
        PoolFee, PoolMetadata, PoolMetadataCall, PoolMetadataFailure, TokenDecimals, TokenMetadata,
        TokenMetadataCall, TokenMetadataFailure, TokenMetadataResult, UniswapV3Fee,
    };

    fn hash(byte: u8) -> BlockHash {
        BlockHash::with_last_byte(byte)
    }

    fn candidate(byte: u8) -> PoolCandidateAddress {
        PoolCandidateAddress(Address::with_last_byte(byte))
    }

    fn anchor(number: u64, hash_byte: u8) -> FinalizedAnchor {
        FinalizedAnchor {
            hash: hash(hash_byte),
            number,
        }
    }

    fn window_block(number: u64, hash_byte: u8) -> RangeLogBlock {
        RangeLogBlock {
            number,
            hash: hash(hash_byte),
            candidates: HashSet::from([candidate(hash_byte)]),
        }
    }

    fn seed_block(hash_byte: u8, parent_byte: u8) -> SeedBlock {
        SeedBlock {
            hash: hash(hash_byte),
            parent_hash: hash(parent_byte),
            candidates: HashSet::from([candidate(hash_byte)]),
        }
    }

    /// A real seed block whose parent is an arbitrary hash (e.g. a filler), not a window block.
    fn seed_block_with_parent(hash_byte: u8, parent_hash: BlockHash) -> SeedBlock {
        SeedBlock {
            hash: hash(hash_byte),
            parent_hash,
            candidates: HashSet::from([candidate(hash_byte)]),
        }
    }

    /// The filler bridging the gap between the real blocks `lower_byte` and `upper_byte`.
    fn filler(lower_byte: u8, upper_byte: u8) -> SeedBlock {
        SeedBlock {
            hash: filler_hash(hash(lower_byte), hash(upper_byte)),
            parent_hash: hash(lower_byte),
            candidates: HashSet::new(),
        }
    }

    #[test]
    fn contiguous_window_links_first_block_to_anchor_then_chains() {
        let anchor = anchor(10, 100);
        let window = [
            window_block(11, 11),
            window_block(12, 12),
            window_block(13, 13),
        ];

        let seed = GraphSeed::from_window(&window, anchor, 0);

        assert_eq!(
            seed.into_blocks(),
            vec![seed_block(11, 100), seed_block(12, 11), seed_block(13, 12),]
        );
    }

    #[test]
    fn gap_is_bridged_by_a_filler_connecting_both_segments() {
        let anchor = anchor(10, 100);
        // 13 is a no-log block: absent from the window, leaving a gap between 12 and 14.
        let window = [
            window_block(11, 11),
            window_block(12, 12),
            window_block(14, 14),
            window_block(15, 15),
        ];

        let seed = GraphSeed::from_window(&window, anchor, 0);

        assert_eq!(
            seed.into_blocks(),
            vec![
                seed_block(11, 100),
                seed_block(12, 11),
                // The 13-gap is bridged: a filler parented to 12, then 14 parented to the filler.
                filler(12, 14),
                seed_block_with_parent(14, filler_hash(hash(12), hash(14))),
                seed_block(15, 14),
            ]
        );
    }

    #[test]
    fn gap_between_anchor_and_first_block_is_bridged() {
        let anchor = anchor(10, 100);
        // 11..=13 are no-log blocks: the first observed block is 14.
        let window = [window_block(14, 14), window_block(15, 15)];

        let seed = GraphSeed::from_window(&window, anchor, 0);

        assert_eq!(
            seed.into_blocks(),
            vec![
                filler(100, 14),
                seed_block_with_parent(14, filler_hash(hash(100), hash(14))),
                seed_block(15, 14),
            ]
        );
    }

    #[test]
    fn multiple_gaps_each_get_a_distinct_filler() {
        let anchor = anchor(10, 100);
        let window = [
            window_block(12, 12),
            window_block(15, 15),
            window_block(18, 18),
        ];

        let blocks = GraphSeed::from_window(&window, anchor, 0).into_blocks();

        assert_eq!(
            blocks,
            vec![
                filler(100, 12),
                seed_block_with_parent(12, filler_hash(hash(100), hash(12))),
                filler(12, 15),
                seed_block_with_parent(15, filler_hash(hash(12), hash(15))),
                filler(15, 18),
                seed_block_with_parent(18, filler_hash(hash(15), hash(18))),
            ]
        );

        // The three fillers are distinct nodes.
        let filler_hashes = HashSet::from([
            filler_hash(hash(100), hash(12)),
            filler_hash(hash(12), hash(15)),
            filler_hash(hash(15), hash(18)),
        ]);
        assert_eq!(filler_hashes.len(), 3);
    }

    #[test]
    fn filler_hash_is_deterministic_and_distinct_from_its_bounds() {
        let lower = hash(100);
        let upper = hash(14);

        // Reproducible.
        assert_eq!(filler_hash(lower, upper), filler_hash(lower, upper));
        // Distinct from both bounds and order-sensitive, so no collisions with real or sibling nodes.
        assert_ne!(filler_hash(lower, upper), lower);
        assert_ne!(filler_hash(lower, upper), upper);
        assert_ne!(filler_hash(lower, upper), filler_hash(upper, lower));
    }

    #[test]
    fn tip_trim_drops_the_highest_blocks() {
        let anchor = anchor(10, 100);
        let window = [
            window_block(11, 11),
            window_block(12, 12),
            window_block(13, 13),
            window_block(14, 14),
            window_block(15, 15),
        ];

        let seed = GraphSeed::from_window(&window, anchor, 2);

        assert_eq!(
            seed.into_blocks(),
            vec![seed_block(11, 100), seed_block(12, 11), seed_block(13, 12),]
        );
    }

    #[test]
    fn blocks_at_or_below_the_anchor_are_ignored() {
        let anchor = anchor(10, 100);
        let window = [
            window_block(8, 8),
            window_block(9, 9),
            window_block(10, 10),
            window_block(11, 11),
        ];

        let seed = GraphSeed::from_window(&window, anchor, 0);

        assert_eq!(seed.into_blocks(), vec![seed_block(11, 100)]);
    }

    #[test]
    fn empty_window_seeds_nothing() {
        let seed = GraphSeed::from_window(&[], anchor(10, 100), 0);

        assert!(seed.into_blocks().is_empty());
    }

    proptest! {
        #[test]
        fn every_seed_block_connects_to_the_anchor(
            anchor_number in 0u64..50,
            offsets in prop::collection::hash_set(1u64..60, 0..25),
            tip_trim in 0usize..12,
        ) {
            let anchor = anchor(anchor_number, 200);
            let window = offsets
                .iter()
                .map(|offset| {
                    let number = anchor_number + offset;
                    window_block(number, u8::try_from(number).unwrap_or(199))
                })
                .collect::<Vec<_>>();

            // Map every real block hash (anchor + window) back to its number; fillers are absent here.
            let mut number_by_hash = std::collections::HashMap::new();
            number_by_hash.insert(anchor.hash, anchor.number);
            for block in &window {
                number_by_hash.insert(block.hash, block.number);
            }

            let max_number = window.iter().map(|block| block.number).max();
            let blocks = GraphSeed::from_window(&window, anchor, tip_trim).into_blocks();
            let by_hash: std::collections::HashMap<_, _> =
                blocks.iter().map(|seed| (seed.hash, seed)).collect();

            // No node is emitted twice (distinct real blocks, distinct synthetic fillers).
            prop_assert_eq!(by_hash.len(), blocks.len());

            // Completeness: every retained window block (above the anchor, within the untrimmed
            // range) is present — nothing is dropped at a gap.
            if let Some(max_number) = max_number {
                let highest_retained = max_number.saturating_sub(tip_trim as u64);
                for block in &window {
                    if block.number > anchor.number && block.number <= highest_retained {
                        prop_assert!(by_hash.contains_key(&block.hash));
                    }
                }
            }

            let mut previous_real_number = None;
            for seed in &blocks {
                match number_by_hash.get(&seed.hash).copied() {
                    // A real block: above the anchor, within range, emitted in increasing order.
                    Some(number) => {
                        prop_assert!(number > anchor.number);
                        if let Some(max_number) = max_number {
                            prop_assert!(number <= max_number.saturating_sub(tip_trim as u64));
                        }
                        if let Some(previous) = previous_real_number {
                            prop_assert!(number > previous);
                        }
                        previous_real_number = Some(number);
                    }
                    // A filler: a synthetic node, never a window block, carrying no candidates.
                    None => prop_assert!(seed.candidates.is_empty()),
                }

                // Connectivity: following parent_hash from this block reaches the anchor through
                // nodes that are all present in the seed.
                let mut current = seed.parent_hash;
                let mut steps = 0;
                while current != anchor.hash {
                    let node = by_hash.get(&current);
                    prop_assert!(node.is_some());
                    current = node.expect("present per assert above").parent_hash;
                    steps += 1;
                    prop_assert!(steps <= blocks.len() + 1);
                }
            }
        }

        #[test]
        fn seed_inference_is_deterministic(
            anchor_number in 0u64..50,
            offsets in prop::collection::hash_set(1u64..60, 0..25),
            tip_trim in 0usize..12,
        ) {
            let anchor = anchor(anchor_number, 200);
            let window = offsets
                .iter()
                .map(|offset| {
                    let number = anchor_number + offset;
                    window_block(number, u8::try_from(number).unwrap_or(199))
                })
                .collect::<Vec<_>>();

            let first = GraphSeed::from_window(&window, anchor, tip_trim).into_blocks();
            let second = GraphSeed::from_window(&window, anchor, tip_trim).into_blocks();

            prop_assert_eq!(first, second);
        }
    }

    fn test_policy() -> BootstrapPolicy {
        BootstrapPolicy {
            tip_trim: 0,
            deadline_ticks: 100,
        }
    }

    fn test_anchor() -> FinalizedAnchor {
        FinalizedAnchor {
            hash: hash(200),
            number: 10,
        }
    }

    fn pool(byte: u8) -> PoolRef {
        PoolRef::uniswap_v3(Address::with_last_byte(byte), ChainKey::Ethereum)
    }

    fn token(byte: u8) -> TokenAddress {
        TokenAddress(Address::with_last_byte(byte), ChainKey::Ethereum)
    }

    fn uniswap_pool_metadata() -> PoolMetadata {
        PoolMetadata {
            token0: Address::with_last_byte(1),
            token1: Address::with_last_byte(2),
            fee: PoolFee::Tiered(UniswapV3Fee::Fee3000),
        }
    }

    fn pool_metadata_with_tokens(token0: u8, token1: u8) -> PoolMetadata {
        PoolMetadata {
            token0: Address::with_last_byte(token0),
            token1: Address::with_last_byte(token1),
            fee: PoolFee::Tiered(UniswapV3Fee::Fee3000),
        }
    }

    /// Last byte of an address, used to recover the seed `u8` a test fixture was built from.
    fn last_byte(address: &Address) -> u8 {
        address.into_array().into_iter().next_back().unwrap_or(0)
    }

    fn token_metadata(decimals: u8) -> TokenMetadata {
        TokenMetadata {
            decimals: TokenDecimals::try_from_u256(U256::from(decimals)).expect("valid decimals"),
        }
    }

    fn single_request(effects: &[Effect]) -> &AnyIssuedRequest {
        match effects {
            [Effect::Request(issued)] => issued,
            _ => panic!("expected exactly one request effect, got {}", effects.len()),
        }
    }

    fn finalized_request_id(effects: &[Effect]) -> RequestId<GetFinalizedHeader> {
        match single_request(effects) {
            AnyIssuedRequest::FinalizedHeader(issued) => issued.request_id,
            _ => panic!("expected finalized header request"),
        }
    }

    fn candidates_request(effects: &[Effect]) -> &IssuedRequest<GetPoolCandidatesInRange> {
        match single_request(effects) {
            AnyIssuedRequest::PoolCandidates(issued) => issued,
            _ => panic!("expected pool candidates request"),
        }
    }

    fn pool_metadata_request(effects: &[Effect]) -> &IssuedRequest<GetPoolMetadata> {
        match single_request(effects) {
            AnyIssuedRequest::PoolMetadata(issued) => issued,
            _ => panic!("expected pool metadata request"),
        }
    }

    fn token_metadata_request_id(effects: &[Effect]) -> RequestId<GetTokenMetadata> {
        match single_request(effects) {
            AnyIssuedRequest::TokenMetadata(issued) => issued.request_id,
            _ => panic!("expected token metadata request"),
        }
    }

    fn token_metadata_request(effects: &[Effect]) -> &IssuedRequest<GetTokenMetadata> {
        match single_request(effects) {
            AnyIssuedRequest::TokenMetadata(issued) => issued,
            _ => panic!("expected token metadata request"),
        }
    }

    /// Drives a freshly initialized bootstrap up to (and including) the pool-candidates response,
    /// returning the state in `ValidatingPools` with a two-block window seeded from `test_anchor`.
    fn drive_to_validating_pools() -> (State, Vec<Effect>) {
        let (state, effects) = init(test_policy());
        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::FinalizedHeaderReceived {
                request_id: finalized_request_id(&effects),
                anchor: test_anchor(),
            },
        );
        let candidates_id = candidates_request(&effects).request_id;

        transition(
            ChainKey::Ethereum,
            state,
            Event::PoolCandidatesReceived {
                request_id: candidates_id,
                blocks: vec![window_block(11, 11), window_block(12, 12)],
            },
        )
    }

    #[test]
    fn discovery_request_scans_from_the_finalized_anchor_to_the_tip() {
        let (state, effects) = init(test_policy());
        let (_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::FinalizedHeaderReceived {
                request_id: finalized_request_id(&effects),
                anchor: test_anchor(),
            },
        );

        // The scan starts at the finalized anchor itself; the historical region below finalized is
        // owned by the persistent metadata cache.
        assert_eq!(
            candidates_request(&effects).request_payload.from_block,
            test_anchor().number
        );
    }

    #[test]
    fn happy_path_reaches_ready_with_seeded_outcome() {
        let anchor = test_anchor();
        let (state, effects) = drive_to_validating_pools();
        let metadata_request = pool_metadata_request(&effects);
        assert_eq!(metadata_request.request_payload.at, anchor.hash);
        let metadata_id = metadata_request.request_id;

        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::PoolMetadataReceived {
                request_id: metadata_id,
                metadata: HashMap::from([(candidate(11), Ok(uniswap_pool_metadata()))]),
            },
        );
        let token_metadata_id = token_metadata_request_id(&effects);

        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::TokenMetadataReceived {
                request_id: token_metadata_id,
                metadata: HashMap::from([
                    (token(1), Ok(token_metadata(18))),
                    (token(2), Ok(token_metadata(6))),
                ]),
            },
        );
        // Token metadata is the final bootstrap step: activation no longer blocks on a pool-state
        // snapshot, so the chain is Ready immediately with an empty finalized snapshot.
        assert!(effects.is_empty());

        let outcome = match completion(&state) {
            Some(Completion::Ready(outcome)) => outcome,
            other => panic!("expected ready, got {other:?}"),
        };
        assert_eq!(outcome.anchor, anchor);
        // Graph seed covers every candidate block; pool state is fetched live after activation.
        assert_eq!(
            outcome.seed_blocks,
            vec![seed_block(11, 200), seed_block(12, 11)]
        );
        assert!(outcome.pool_snapshots.is_empty());
        assert!(outcome.pool_registry.verified_metadata(pool(11)).is_some());
        assert!(outcome.token_registry.verified_metadata(token(1)).is_some());
    }

    #[test]
    fn stale_response_is_ignored_and_real_request_still_advances() {
        let (state, effects) = init(test_policy());
        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::FinalizedHeaderReceived {
                request_id: finalized_request_id(&effects),
                anchor: test_anchor(),
            },
        );
        let real_candidates_id = candidates_request(&effects).request_id;

        let bogus_id = RequestId::<GetPoolCandidatesInRange>::from_raw_for_test(99_999);
        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::PoolCandidatesReceived {
                request_id: bogus_id,
                blocks: vec![window_block(11, 11)],
            },
        );
        assert!(effects.is_empty());
        assert!(completion(&state).is_none());

        let (_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::PoolCandidatesReceived {
                request_id: real_candidates_id,
                blocks: vec![window_block(11, 11)],
            },
        );
        assert!(matches!(
            single_request(&effects),
            AnyIssuedRequest::PoolMetadata(_)
        ));
    }

    #[test]
    fn request_failed_reissues_the_same_request_with_a_new_id() {
        let (state, effects) = init(test_policy());
        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::FinalizedHeaderReceived {
                request_id: finalized_request_id(&effects),
                anchor: test_anchor(),
            },
        );
        let candidates = candidates_request(&effects);
        let from_block = candidates.request_payload.from_block;
        let candidates_id = candidates.request_id;

        let (_state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::RequestFailed {
                request_id: AnyRequestId::PoolCandidates(candidates_id),
            },
        );

        let reissued = candidates_request(&effects);
        assert_eq!(reissued.request_payload.from_block, from_block);
        assert_ne!(
            reissued.request_id.raw_for_test(),
            candidates_id.raw_for_test()
        );
    }

    #[test]
    fn deadline_after_anchor_degrades_to_ready_with_seed_and_empty_snapshot() {
        let policy = BootstrapPolicy {
            tip_trim: 0,
            deadline_ticks: 3,
        };
        let (state, effects) = init(policy);
        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::FinalizedHeaderReceived {
                request_id: finalized_request_id(&effects),
                anchor: test_anchor(),
            },
        );
        let candidates_id = candidates_request(&effects).request_id;
        let (mut state, _effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::PoolCandidatesReceived {
                request_id: candidates_id,
                blocks: vec![window_block(11, 11), window_block(12, 12)],
            },
        );

        for _ in 0..policy.deadline_ticks {
            let (next, _effects) = transition(ChainKey::Ethereum, state, Event::Tick);
            state = next;
        }

        let outcome = match completion(&state) {
            Some(Completion::Ready(outcome)) => outcome,
            other => panic!("expected ready, got {other:?}"),
        };
        assert_eq!(outcome.anchor, test_anchor());
        assert_eq!(
            outcome.seed_blocks,
            vec![seed_block(11, 200), seed_block(12, 11)]
        );
        assert!(outcome.pool_snapshots.is_empty());
        assert_eq!(outcome.pool_registry, TrustedPoolRegistry::new());
    }

    #[test]
    fn deadline_before_anchor_abandons_the_chain() {
        let policy = BootstrapPolicy {
            tip_trim: 0,
            deadline_ticks: 2,
        };
        let (mut state, _effects) = init(policy);

        for _ in 0..policy.deadline_ticks {
            let (next, _effects) = transition(ChainKey::Ethereum, state, Event::Tick);
            state = next;
        }

        assert_eq!(completion(&state), Some(Completion::Abandoned));
    }

    /// Drives a bootstrap from `init` to a `Ready` outcome, feeding `pool_metadata` as the
    /// metadata response and synthesizing the token/pool-data responses from the request payloads
    /// the bootstrap actually emits (so they answer exactly what was asked). Mirrors the way the
    /// kernel property tests build a populated state, but ends at the `BootstrapOutcome` so the
    /// same invariants can be asserted against the bootstrap's *output*.
    fn drive_to_ready(
        pool_metadata: HashMap<PoolCandidateAddress, PoolMetadataResult>,
        token_response: impl Fn(&TokenAddress) -> TokenMetadataResult,
    ) -> BootstrapOutcome {
        let (state, effects) = init(test_policy());
        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::FinalizedHeaderReceived {
                request_id: finalized_request_id(&effects),
                anchor: test_anchor(),
            },
        );
        let candidates_id = candidates_request(&effects).request_id;
        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::PoolCandidatesReceived {
                request_id: candidates_id,
                blocks: vec![window_block(11, 11), window_block(12, 12)],
            },
        );
        let metadata_id = pool_metadata_request(&effects).request_id;
        let (state, effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::PoolMetadataReceived {
                request_id: metadata_id,
                metadata: pool_metadata,
            },
        );

        let token_request = token_metadata_request(&effects);
        let token_id = token_request.request_id;
        let token_metadata = token_request
            .request_payload
            .tokens
            .iter()
            .map(|token| (*token, token_response(token)))
            .collect();
        let (state, _effects) = transition(
            ChainKey::Ethereum,
            state,
            Event::TokenMetadataReceived {
                request_id: token_id,
                metadata: token_metadata,
            },
        );

        match completion(&state) {
            Some(Completion::Ready(outcome)) => outcome,
            other => panic!("expected ready outcome, got {other:?}"),
        }
    }

    /// The single in-flight request carried out of a transition, if any. The bootstrap is
    /// sequential, so this is the live request the runtime is expected to answer next.
    fn first_issued(effects: Vec<Effect>) -> Option<AnyIssuedRequest> {
        effects
            .into_iter()
            .map(|Effect::Request(issued)| issued)
            .next()
    }

    /// A well-formed success response for whatever request is currently outstanding.
    fn deliver_event(issued: &AnyIssuedRequest) -> Event {
        match issued {
            AnyIssuedRequest::FinalizedHeader(request) => Event::FinalizedHeaderReceived {
                request_id: request.request_id,
                anchor: test_anchor(),
            },
            AnyIssuedRequest::PoolCandidates(request) => Event::PoolCandidatesReceived {
                request_id: request.request_id,
                blocks: vec![window_block(11, 11), window_block(12, 12)],
            },
            AnyIssuedRequest::PoolMetadata(request) => Event::PoolMetadataReceived {
                request_id: request.request_id,
                metadata: request
                    .request_payload
                    .candidates
                    .iter()
                    .map(|candidate| (*candidate, Ok(uniswap_pool_metadata())))
                    .collect(),
            },
            AnyIssuedRequest::TokenMetadata(request) => Event::TokenMetadataReceived {
                request_id: request.request_id,
                metadata: request
                    .request_payload
                    .tokens
                    .iter()
                    .map(|token| (*token, Ok(token_metadata(18))))
                    .collect(),
            },
        }
    }

    /// A failure event targeting whatever request is currently outstanding.
    fn fail_event(issued: &AnyIssuedRequest) -> Event {
        let request_id = match issued {
            AnyIssuedRequest::FinalizedHeader(request) => {
                AnyRequestId::FinalizedHeader(request.request_id)
            }
            AnyIssuedRequest::PoolCandidates(request) => {
                AnyRequestId::PoolCandidates(request.request_id)
            }
            AnyIssuedRequest::PoolMetadata(request) => {
                AnyRequestId::PoolMetadata(request.request_id)
            }
            AnyIssuedRequest::TokenMetadata(request) => {
                AnyRequestId::TokenMetadata(request.request_id)
            }
        };

        Event::RequestFailed { request_id }
    }

    #[derive(Clone, Copy, Debug)]
    enum Action {
        Deliver,
        Fail,
        Tick,
    }

    fn action_strategy() -> impl Strategy<Value = Action> {
        prop_oneof![
            Just(Action::Deliver),
            Just(Action::Fail),
            Just(Action::Tick)
        ]
    }

    proptest! {
        // Liveness: for any interleaving of successful responses, failures, and ticks, a
        // non-terminal bootstrap always holds at least one in-flight request, so it is never
        // stuck — a later response or tick-driven retry can always advance it. Every transition
        // that consumes the outstanding request immediately issues its successor (or retries),
        // so the sequential phase machine maintains exactly one request until it terminates.
        #[test]
        fn bootstrap_is_never_stuck_under_arbitrary_results_and_order(
            actions in prop::collection::vec(action_strategy(), 0..48),
        ) {
            let (mut state, effects) = init(test_policy());
            prop_assert_eq!(state.pending.len_for_test(), 1);
            let mut current = first_issued(effects);

            for action in actions {
                let Some(in_flight) = current.as_ref() else {
                    // Only reachable once terminal: no request was ever dropped while live.
                    prop_assert!(completion(&state).is_some());
                    break;
                };

                let event = match action {
                    Action::Deliver => deliver_event(in_flight),
                    Action::Fail => fail_event(in_flight),
                    Action::Tick => Event::Tick,
                };

                let (next_state, effects) = transition(ChainKey::Ethereum, state, event);
                state = next_state;
                if let Some(issued) = first_issued(effects) {
                    current = Some(issued);
                }

                // The invariant under test: not terminal ⇒ a request is pending to advance on.
                if completion(&state).is_none() {
                    prop_assert!(state.pending.len_for_test() >= 1);
                }
            }
        }

        // Mirrors `kernel::pool_registry::verified_and_rejected_sets_never_overlap_after_applying_results`:
        // the trusted pool registry the bootstrap hands off must never classify a pool as both
        // verified and rejected, whatever metadata responses it folded in.
        #[test]
        fn outcome_pool_registry_verified_and_rejected_never_overlap(
            result_bytes in proptest::collection::vec((any::<u8>(), any::<bool>()), 0..48),
        ) {
            let metadata = result_bytes
                .into_iter()
                .map(|(byte, verify)| {
                    let result = if verify {
                        Ok(pool_metadata_with_tokens(byte, byte.wrapping_add(64)))
                    } else {
                        Err(PoolMetadataFailure::CallFailed(PoolMetadataCall::Token0))
                    };
                    (candidate(byte), result)
                })
                .collect();

            let outcome = drive_to_ready(
                metadata,
                |_| Ok(token_metadata(18)),
            );

            for pool in outcome.pool_registry.verified_pools_for_test() {
                let address = pool.uniswap_v3_address().expect("v3 pool");
                prop_assert!(!outcome.pool_registry.is_rejected(PoolCandidateAddress(address)));
            }
        }

        // Mirrors `kernel::token_registry::verified_and_unsupported_sets_never_overlap_after_applying_results`:
        // the seeded token registry must never classify a token as both verified and unsupported.
        #[test]
        fn outcome_token_registry_verified_and_unsupported_never_overlap(
            pool_bytes in proptest::collection::hash_set(any::<u8>(), 0..16),
            unsupported_bytes in proptest::collection::hash_set(any::<u8>(), 0..32),
        ) {
            let metadata = pool_bytes
                .iter()
                .map(|byte| {
                    (
                        candidate(*byte),
                        Ok(pool_metadata_with_tokens(*byte, byte.wrapping_add(64))),
                    )
                })
                .collect();

            let outcome = drive_to_ready(
                metadata,
                |token| {
                    if unsupported_bytes.contains(&last_byte(&token.0)) {
                        Err(TokenMetadataFailure::CallFailed(TokenMetadataCall::Decimals))
                    } else {
                        Ok(token_metadata(18))
                    }
                },
            );

            for token in outcome.token_registry.verified_tokens_for_test() {
                prop_assert!(!outcome.token_registry.is_unsupported(token));
            }
        }

        // Bootstrap no longer snapshots pool state (activation hands the kernel an empty finalized
        // snapshot and the live dirty-pool / backfill paths fetch it at the tip), so the outcome's
        // `pool_snapshots` is always empty. The "snapshots cover only verified pools" invariant now
        // lives in the kernel's live pool-data scheduling tests.
        #[test]
        fn outcome_carries_no_pool_snapshots(
            verified_bytes in proptest::collection::hash_set(any::<u8>(), 0..16),
        ) {
            let metadata = verified_bytes
                .iter()
                .map(|byte| {
                    (
                        candidate(*byte),
                        Ok(pool_metadata_with_tokens(*byte, byte.wrapping_add(64))),
                    )
                })
                .collect();

            let outcome = drive_to_ready(metadata, |_| Ok(token_metadata(18)));

            prop_assert!(outcome.pool_snapshots.is_empty());
        }

        // Mirrors `kernel::canonical_verified_pool_tokens_are_known_or_pending`: every token of a
        // verified pool is scheduled for resolution. The bootstrap requests exactly the referenced
        // tokens, so once they all answer, every verified pool's tokens are known in the output.
        #[test]
        fn outcome_verified_pool_tokens_are_all_resolved(
            verified_bytes in proptest::collection::hash_set(any::<u8>(), 0..16),
        ) {
            let metadata = verified_bytes
                .iter()
                .map(|byte| {
                    (
                        candidate(*byte),
                        Ok(pool_metadata_with_tokens(*byte, byte.wrapping_add(64))),
                    )
                })
                .collect();

            let outcome = drive_to_ready(
                metadata,
                |_| Ok(token_metadata(18)),
            );

            for byte in &verified_bytes {
                let metadata = outcome
                    .pool_registry
                    .verified_metadata(pool(*byte))
                    .expect("verified pool stays verified in the outcome");
                for token in [TokenAddress(metadata.token0, ChainKey::Ethereum), TokenAddress(metadata.token1, ChainKey::Ethereum)] {
                    prop_assert!(outcome.token_registry.is_known(token));
                }
            }
        }
    }
}
