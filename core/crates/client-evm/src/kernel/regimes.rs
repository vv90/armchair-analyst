//! Regime exploration for the chain kernel: how the pure event-driven kernel behaves — and fails —
//! under degraded runtime conditions observed in real runs (silently dying RPC requests, WS log
//! stalls and churn, archive-`getLogs` refusal, duplicate head fan-out, tick starvation, stalled
//! finalization). Each scenario drives `kernel::transition_outcome` against a deterministic
//! simulated chain and a policy-driven provider simulator, mirroring the multi-chain wrapper's
//! post-transition work (the `optimization_update_if_changed` dispatch gate that produces the
//! production `behind` gauge reference), and asserts the characteristic failure signature so the
//! findings stay pinned.
//!
//! This lives as an in-crate `#[cfg(test)]` module (not a separate binary) because the
//! observability it needs — fold frontier, in-flight counts, canonical window length, the streamed
//! staging buffer — is `pub(crate)`/private by design; a binary would force production API
//! changes. The tests are heavy (long simulated runs with O(window) walks), so every one is
//! `#[ignore]`d and the suite is run explicitly, preferably in release mode:
//!
//! ```text
//! cargo test -p client-evm --lib --release kernel::regimes -- --ignored --nocapture --test-threads=1
//! ```

use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use alloy::primitives::{Address, BloomInput, U160, U256, aliases::I24};

use super::pending_requests::{AnyIssuedRequest, AnyRequestId};
use super::pool_registry::{
    PoolFee, PoolMetadata, PoolMetadataResult, TrustedPoolRegistry, UniswapV3Fee,
};
use super::token_registry::{
    TokenAddress, TokenDecimals, TokenMetadata, TokenMetadataResult, TokenRegistry,
};
use super::{
    Effect, Event, MAX_STREAMED_LOG_BLOCKS, State, TransitionOutcome, transition_outcome,
};
use crate::pool_state::{PoolState, ProtocolPoolKey};
use crate::tick::REQUEST_TTL_FOR_TEST as REQUEST_TTL;
use crate::{BlockHash, Bloom, ChainKey, PoolLog, PoolLogEvent};

/// One simulated tick corresponds to one production `Event::Tick` (1s in aa-cli) and, at the
/// default one block per tick, one new head — a Polygon-ish cadence.
const CHAIN: ChainKey = ChainKey::Polygon;
const ANCHOR_NUMBER: u64 = 1_000;
const POOL_COUNT: u64 = 16;

// ---- Deterministic world -----------------------------------------------------------------------

/// The canonical chain is fully determined by block number: the hash encodes the number, every
/// block extends its predecessor, and every block carries two swap logs whose emitters are baked
/// into the header bloom (worst case for the hole detector: every block bloom-touches the trusted
/// set, so an unresolved block always blocks the fold frontier).
fn block_hash(number: u64) -> BlockHash {
    let mut bytes = [0u8; 32];
    bytes[0] = 0xB1;
    bytes[24..].copy_from_slice(&number.to_be_bytes());
    BlockHash::from(bytes)
}

fn number_of(hash: BlockHash) -> Option<u64> {
    let bytes = hash.as_slice();
    if bytes.first() != Some(&0xB1) {
        return None;
    }
    bytes
        .get(24..32)
        .and_then(|tail| tail.try_into().ok())
        .map(u64::from_be_bytes)
}

fn pool_address(index: u64) -> Address {
    Address::with_last_byte(10 + (index % POOL_COUNT) as u8)
}

fn pool_key(index: u64) -> ProtocolPoolKey {
    ProtocolPoolKey::UniswapV3(pool_address(index))
}

fn token_address(index: u64) -> Address {
    Address::with_last_byte(100 + (index % (POOL_COUNT + 1)) as u8)
}

/// The two pools a block touches; distinct by construction (7 is coprime-enough to 16).
fn block_pools(number: u64) -> [u64; 2] {
    [number % POOL_COUNT, (number + 7) % POOL_COUNT]
}

fn block_logs(number: u64) -> Vec<PoolLog> {
    block_pools(number)
        .iter()
        .enumerate()
        .map(|(position, pool)| PoolLog {
            pool: pool_key(*pool),
            log_index: position as u64,
            event: PoolLogEvent::Swap {
                sqrt_price_x96: U160::from(number),
                tick: I24::try_from(0).expect("zero tick fits int24"),
                liquidity: 1 + u128::from(number),
            },
        })
        .collect()
}

fn block_bloom(number: u64) -> Bloom {
    let mut bloom = Bloom::default();
    for pool in block_pools(number) {
        bloom.accrue(BloomInput::Raw(pool_address(pool).as_slice()));
    }
    bloom
}

fn world_pool_metadata(index: u64) -> PoolMetadata {
    PoolMetadata {
        token0: token_address(index),
        token1: token_address(index + 1),
        fee: PoolFee::Tiered(UniswapV3Fee::Fee3000),
    }
}

fn world_pool_state() -> PoolState {
    PoolState {
        sqrt_price_x96: U160::from(1u64),
        tick: I24::try_from(0).expect("zero tick fits int24"),
        liquidity: 1,
    }
}

/// Every simulated pool pre-verified, so scenarios exercise the log path rather than discovery.
fn verified_pool_registry() -> TrustedPoolRegistry {
    let results: HashMap<ProtocolPoolKey, PoolMetadataResult> = (0..POOL_COUNT)
        .map(|index| (pool_key(index), Ok(world_pool_metadata(index))))
        .collect();
    TrustedPoolRegistry::new().with_metadata_results(CHAIN, results)
}

fn verified_token_registry() -> TokenRegistry {
    let results: HashMap<TokenAddress, TokenMetadataResult> = (0..=POOL_COUNT)
        .map(|index| {
            (
                TokenAddress(token_address(index), CHAIN),
                Ok(TokenMetadata {
                    decimals: TokenDecimals::try_from_u256(U256::from(18))
                        .expect("18 decimals supported"),
                }),
            )
        })
        .collect();
    TokenRegistry::new().with_metadata_results(results)
}

// ---- Provider simulator --------------------------------------------------------------------------

/// What the simulated provider does with one accepted request.
#[derive(Clone, Copy)]
enum ReplyPolicy {
    /// Correct answer from the world, `delay` ticks later.
    Respond { delay: u64 },
    /// `Event::RequestFailed` `delay` ticks later — the runtime's failover-exhaustion signal.
    Fail { delay: u64 },
    /// The request is eaten: no response, no failure — a hung connection / unlogged per-endpoint
    /// death, the signature observed on Polygon in the 2026-07-15 run.
    SilentlyDrop,
    /// Ranged `getLogs` only: refuse (fail) requests reaching deeper than `archive_depth` below
    /// the current tip — the free-tier archive refusal observed on BNB — and answer the rest.
    ArchiveGated { delay: u64, archive_depth: u64 },
}

#[derive(Clone, Copy)]
struct Policies {
    header: ReplyPolicy,
    block_logs: ReplyPolicy,
    logs_range: ReplyPolicy,
    pool_metadata: ReplyPolicy,
    token_metadata: ReplyPolicy,
    pool_data: ReplyPolicy,
}

impl Policies {
    fn healthy(delay: u64) -> Policies {
        Policies {
            header: ReplyPolicy::Respond { delay },
            block_logs: ReplyPolicy::Respond { delay },
            logs_range: ReplyPolicy::Respond { delay },
            pool_metadata: ReplyPolicy::Respond { delay },
            token_metadata: ReplyPolicy::Respond { delay },
            pool_data: ReplyPolicy::Respond { delay },
        }
    }
}

#[derive(Clone, Copy, Default)]
struct KindStats {
    issued: u64,
    answered: u64,
    failed: u64,
    dropped: u64,
}

enum Decision {
    Answer(u64),
    Fail(u64),
    Drop,
}

fn decide(policy: ReplyPolicy, now: u64, tip: u64, range_from: Option<u64>) -> Decision {
    match policy {
        ReplyPolicy::Respond { delay } => Decision::Answer(now + delay),
        ReplyPolicy::Fail { delay } => Decision::Fail(now + delay),
        ReplyPolicy::SilentlyDrop => Decision::Drop,
        ReplyPolicy::ArchiveGated {
            delay,
            archive_depth,
        } => match range_from {
            Some(from) if tip.saturating_sub(from) > archive_depth => Decision::Fail(now + delay),
            _ => Decision::Answer(now + delay),
        },
    }
}

struct Provider {
    /// `(from_tick, policies)`; the entry with the largest `from_tick <= now` applies, so a
    /// scenario can degrade or heal the provider mid-run.
    schedule: Vec<(u64, Policies)>,
    due: BTreeMap<u64, Vec<Event>>,
    stats: BTreeMap<&'static str, KindStats>,
}

impl Provider {
    fn new(policies: Policies) -> Provider {
        Provider {
            schedule: vec![(0, policies)],
            due: BTreeMap::new(),
            stats: BTreeMap::new(),
        }
    }

    fn switching_to(mut self, from_tick: u64, policies: Policies) -> Provider {
        self.schedule.push((from_tick, policies));
        self.schedule.sort_by_key(|(from, _)| *from);
        self
    }

    fn policies_at(&self, now: u64) -> Policies {
        self.schedule
            .iter()
            .rev()
            .find(|(from, _)| *from <= now)
            .map(|(_, policies)| *policies)
            .unwrap_or_else(|| Policies::healthy(1))
    }

    fn take_due(&mut self, now: u64) -> Vec<Event> {
        self.due.remove(&now).unwrap_or_default()
    }

    fn stat(&self, kind: &'static str) -> KindStats {
        self.stats.get(kind).copied().unwrap_or_default()
    }

    fn issued_total(&self) -> u64 {
        self.stats.values().map(|stat| stat.issued).sum()
    }

    fn settle(
        &mut self,
        kind: &'static str,
        decision: Decision,
        failure_id: AnyRequestId,
        answer: impl FnOnce() -> Event,
    ) {
        let stat = self.stats.entry(kind).or_default();
        stat.issued += 1;
        match decision {
            Decision::Answer(at) => {
                stat.answered += 1;
                self.due.entry(at).or_default().push(answer());
            }
            Decision::Fail(at) => {
                stat.failed += 1;
                self.due
                    .entry(at)
                    .or_default()
                    .push(Event::RequestFailed {
                        request_id: failure_id,
                    });
            }
            Decision::Drop => {
                stat.dropped += 1;
            }
        }
    }

    fn accept(&mut self, now: u64, tip: u64, request: AnyIssuedRequest) {
        let policies = self.policies_at(now);
        match request {
            AnyIssuedRequest::BlockHeader(issued) => {
                let decision = decide(policies.header, now, tip, None);
                let hash = issued.request_payload.block_hash;
                let request_id = issued.request_id;
                self.settle(
                    "header",
                    decision,
                    AnyRequestId::BlockHeader(request_id),
                    || match number_of(hash) {
                        Some(number) => Event::BlockHeaderReceived {
                            request_id,
                            hash,
                            parent_hash: block_hash(number - 1),
                            logs_bloom: block_bloom(number),
                            number,
                        },
                        None => Event::BlockHeaderNotFound { request_id },
                    },
                );
            }
            AnyIssuedRequest::BlockLogs(issued) => {
                let decision = decide(policies.block_logs, now, tip, None);
                let hash = issued.request_payload.block_hash;
                let request_id = issued.request_id;
                self.settle(
                    "block_logs",
                    decision,
                    AnyRequestId::BlockLogs(request_id),
                    || Event::BlockLogsReceived {
                        request_id,
                        logs: number_of(hash).map(block_logs).unwrap_or_default(),
                    },
                );
            }
            AnyIssuedRequest::LogsRange(issued) => {
                let from = issued.request_payload.from_block();
                let to = issued.request_payload.to_block();
                let decision = decide(policies.logs_range, now, tip, Some(from));
                let request_id = issued.request_id;
                self.settle(
                    "logs_range",
                    decision,
                    AnyRequestId::LogsRange(request_id),
                    || Event::BlockLogsRangeReceived {
                        request_id,
                        blocks: (from..=to)
                            .map(|number| (block_hash(number), block_logs(number)))
                            .collect(),
                    },
                );
            }
            AnyIssuedRequest::PoolMetadata(issued) => {
                let decision = decide(policies.pool_metadata, now, tip, None);
                let request_id = issued.request_id;
                let candidates = issued.request_payload.candidates;
                self.settle(
                    "pool_metadata",
                    decision,
                    AnyRequestId::PoolMetadata(request_id),
                    || Event::PoolMetadataReceived {
                        request_id,
                        metadata: candidates
                            .into_iter()
                            .map(|candidate| (candidate, Ok(world_pool_metadata(0))))
                            .collect(),
                    },
                );
            }
            AnyIssuedRequest::TokenMetadata(issued) => {
                let decision = decide(policies.token_metadata, now, tip, None);
                let request_id = issued.request_id;
                let tokens = issued.request_payload.tokens;
                self.settle(
                    "token_metadata",
                    decision,
                    AnyRequestId::TokenMetadata(request_id),
                    || Event::TokenMetadataReceived {
                        request_id,
                        metadata: tokens
                            .into_iter()
                            .map(|token| {
                                (
                                    token,
                                    Ok(TokenMetadata {
                                        decimals: TokenDecimals::try_from_u256(U256::from(18))
                                            .expect("18 decimals supported"),
                                    }),
                                )
                            })
                            .collect(),
                    },
                );
            }
            AnyIssuedRequest::PoolData(issued) => {
                let decision = decide(policies.pool_data, now, tip, None);
                let request_id = issued.request_id;
                let at = issued.request_payload.at;
                let pools = issued.request_payload.pools;
                self.settle(
                    "pool_data",
                    decision,
                    AnyRequestId::PoolData(request_id),
                    move || Event::PoolDataReceived {
                        request_id,
                        pools: pools
                            .into_iter()
                            .map(|pool| (pool, Ok(world_pool_state())))
                            .collect(),
                    },
                );
                let _ = at;
            }
            AnyIssuedRequest::CanonicalHeader(issued) => {
                let decision = decide(policies.header, now, tip, None);
                let request_id = issued.request_id;
                let number = issued.request_payload.number;
                self.settle(
                    "canonical_header",
                    decision,
                    AnyRequestId::CanonicalHeader(request_id),
                    // The deterministic world's canonical block at a height is `block_hash(number)`,
                    // so the probe answers with the true chain — an orphaned anchor's height then
                    // resolves to a differing hash and drives re-init.
                    || Event::CanonicalHeaderAtHeightReceived {
                        request_id,
                        hash: block_hash(number),
                        number,
                    },
                );
            }
        }
    }
}

// ---- Scenario driver ------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Ws {
    /// The subscription delivers the block's full log set right after its head.
    Full,
    /// The subscription delivers only part of the block's logs (provider hiccup): the kernel
    /// trusts the partial set at the tip and only finds out at finalization (`ws_miss`).
    Partial,
    /// The subscription delivers nothing for this block.
    Missing,
}

struct Scenario {
    name: &'static str,
    ticks: u64,
    blocks_per_tick: u64,
    /// Multi-provider fan-out: how many times each head is delivered (Arbitrum showed 8x).
    head_dup: u64,
    ws: fn(u64) -> Ws,
    provider: Provider,
    /// `(every_ticks, lag_blocks)`: an observed finality signal for `tip - lag`, mirroring the
    /// wrapper's finalized-header refresh loop; `None` = the finalized-header source is dead.
    finality: Option<(u64, u64)>,
    /// `false` simulates a starved transition thread that never delivers `Event::Tick` — the
    /// kernel's only retry clock.
    deliver_ticks: bool,
    sample_every: u64,
}

#[derive(Clone, Copy)]
struct EventCost {
    window_before: usize,
    micros: u128,
    inert: bool,
}

#[derive(Clone, Copy)]
struct Sample {
    tick: u64,
    head: u64,
    frontier: Option<u64>,
    behind: Option<usize>,
    window: Option<usize>,
    inflight: usize,
    ws_miss: u64,
    issued: u64,
    staged: usize,
}

struct Report {
    samples: Vec<Sample>,
    head_costs: Vec<EventCost>,
    dup_costs: Vec<EventCost>,
    dispatches: u64,
    provider: Provider,
    state: State,
    last_dispatched: Option<BlockHash>,
    head: u64,
}

impl Report {
    fn frontier_number(&self) -> Option<u64> {
        self.last_dispatched.and_then(number_of)
    }

    fn behind(&self) -> Option<usize> {
        self.last_dispatched
            .and_then(|reference| self.state.blocks_behind(reference))
    }

    fn stat(&self, kind: &'static str) -> KindStats {
        self.provider.stat(kind)
    }

    fn print_summary(&self) {
        println!("  ---- summary ----");
        println!(
            "  head={} frontier={:?} behind={:?} window={:?} inflight={} ws_miss={} staged={} dispatches={}",
            self.head,
            self.frontier_number(),
            self.behind(),
            self.state.canonical_path_len_from_finalized(),
            self.state.in_flight_request_count(),
            self.state.ws_miss_count(),
            self.state.streamed_logs.len(),
            self.dispatches,
        );
        for (kind, stat) in &self.provider.stats {
            println!(
                "  {kind}: issued={} answered={} failed={} dropped={}",
                stat.issued, stat.answered, stat.failed, stat.dropped
            );
        }
        let firsts: Vec<u128> = self.head_costs.iter().map(|cost| cost.micros).collect();
        let dups: Vec<u128> = self.dup_costs.iter().map(|cost| cost.micros).collect();
        println!(
            "  head-event cost: first mean={}us max={}us (n={}); duplicate mean={}us max={}us (n={})",
            mean(&firsts),
            firsts.iter().max().copied().unwrap_or(0),
            firsts.len(),
            mean(&dups),
            dups.iter().max().copied().unwrap_or(0),
            dups.len(),
        );
    }
}

fn mean(values: &[u128]) -> u128 {
    if values.is_empty() {
        0
    } else {
        values.iter().sum::<u128>() / values.len() as u128
    }
}

enum CostKind {
    HeadFirst,
    HeadDup,
    Other,
}

struct Harness {
    provider: Provider,
    state: Option<State>,
    last_dispatched: Option<BlockHash>,
    dispatches: u64,
    head: u64,
    now: u64,
    head_costs: Vec<EventCost>,
    dup_costs: Vec<EventCost>,
}

impl Harness {
    /// One event through the kernel plus the multi-chain wrapper's post-transition mirror: route
    /// effects to the provider, then run the production dispatch gate (`Inert` skips it exactly as
    /// `chain_event` does) and advance `last_dispatched` — the reference the `behind` gauge uses.
    fn apply(&mut self, event: Event, cost: CostKind) {
        let state = self.state.take().expect("state present between events");
        let window_before = state.canonical_path_len_from_finalized().unwrap_or(0);
        let started = Instant::now();
        let (state, effects, inert) = match transition_outcome(CHAIN, state, event) {
            TransitionOutcome::Inert(state) => (state, Vec::new(), true),
            TransitionOutcome::Progressed(state, effects) => (state, effects, false),
        };
        if !inert
            && let Some(update) = state.optimization_update_if_changed(CHAIN, self.last_dispatched)
        {
            self.last_dispatched = Some(update.block_hash);
            self.dispatches += 1;
        }
        let micros = started.elapsed().as_micros();
        for effect in effects {
            let Effect::Request(request) = effect;
            self.provider.accept(self.now, self.head, request);
        }
        let recorded = EventCost {
            window_before,
            micros,
            inert,
        };
        match cost {
            CostKind::HeadFirst => self.head_costs.push(recorded),
            CostKind::HeadDup => self.dup_costs.push(recorded),
            CostKind::Other => {}
        }
        self.state = Some(state);
    }

    fn sample(&self, tick: u64, issued: u64) -> Sample {
        let state = self.state.as_ref().expect("state present between events");
        Sample {
            tick,
            head: self.head,
            frontier: self.last_dispatched.and_then(number_of),
            behind: self
                .last_dispatched
                .and_then(|reference| state.blocks_behind(reference)),
            window: state.canonical_path_len_from_finalized(),
            inflight: state.in_flight_request_count(),
            ws_miss: state.ws_miss_count(),
            issued,
            staged: state.streamed_logs.len(),
        }
    }
}

fn head_event(number: u64) -> Event {
    Event::HeadObserved {
        hash: block_hash(number),
        parent_hash: block_hash(number - 1),
        logs_bloom: block_bloom(number),
        number,
    }
}

fn run(scenario: Scenario) -> Report {
    println!("== regime: {} ==", scenario.name);
    println!(
        "  {:>6} {:>8} {:>9} {:>7} {:>7} {:>9} {:>8} {:>7} {:>7}",
        "tick", "head", "frontier", "behind", "window", "inflight", "ws_miss", "issued", "staged"
    );

    let (state, seed_effects) = State::activate_from_seed(
        block_hash(ANCHOR_NUMBER),
        ANCHOR_NUMBER,
        HashMap::new(),
        verified_pool_registry(),
        verified_token_registry(),
        Vec::new(),
    );
    let mut harness = Harness {
        provider: scenario.provider,
        state: Some(state),
        last_dispatched: None,
        dispatches: 0,
        head: ANCHOR_NUMBER,
        now: 0,
        head_costs: Vec::new(),
        dup_costs: Vec::new(),
    };
    for effect in seed_effects {
        let Effect::Request(request) = effect;
        harness.provider.accept(0, ANCHOR_NUMBER, request);
    }

    let mut samples = Vec::new();
    for now in 1..=scenario.ticks {
        harness.now = now;
        for event in harness.provider.take_due(now) {
            harness.apply(event, CostKind::Other);
        }
        for _ in 0..scenario.blocks_per_tick {
            harness.head += 1;
            let number = harness.head;
            harness.apply(head_event(number), CostKind::HeadFirst);
            for _ in 1..scenario.head_dup {
                harness.apply(head_event(number), CostKind::HeadDup);
            }
            match (scenario.ws)(number) {
                Ws::Full => harness.apply(
                    Event::LogObserved {
                        block_hash: block_hash(number),
                        logs: block_logs(number),
                    },
                    CostKind::Other,
                ),
                Ws::Partial => harness.apply(
                    Event::LogObserved {
                        block_hash: block_hash(number),
                        logs: block_logs(number).into_iter().take(1).collect(),
                    },
                    CostKind::Other,
                ),
                Ws::Missing => {}
            }
        }
        if let Some((every, lag)) = scenario.finality
            && now % every == 0
        {
            let target = harness.head.saturating_sub(lag);
            if target > ANCHOR_NUMBER {
                harness.apply(
                    Event::FinalizedBlockObserved {
                        block_hash: block_hash(target),
                        number: target,
                    },
                    CostKind::Other,
                );
            }
        }
        if scenario.deliver_ticks {
            harness.apply(Event::Tick, CostKind::Other);
        }

        if now % scenario.sample_every == 0 {
            let sample = harness.sample(now, harness.provider.issued_total());
            println!(
                "  {:>6} {:>8} {:>9} {:>7} {:>7} {:>9} {:>8} {:>7} {:>7}",
                sample.tick,
                sample.head,
                sample
                    .frontier
                    .map(|number| number.to_string())
                    .unwrap_or_else(|| "?".to_owned()),
                sample
                    .behind
                    .map(|behind| behind.to_string())
                    .unwrap_or_else(|| "?".to_owned()),
                sample
                    .window
                    .map(|window| window.to_string())
                    .unwrap_or_else(|| "?".to_owned()),
                sample.inflight,
                sample.ws_miss,
                sample.issued,
                sample.staged,
            );
            samples.push(sample);
        }
    }

    let report = Report {
        samples,
        head_costs: harness.head_costs,
        dup_costs: harness.dup_costs,
        dispatches: harness.dispatches,
        provider: harness.provider,
        state: harness.state.take().expect("state present at end"),
        last_dispatched: harness.last_dispatched,
        head: harness.head,
    };
    report.print_summary();
    report
}

/// `behind` never decreases across the samples taken at or after `from_tick`.
fn behind_is_monotonic_from(samples: &[Sample], from_tick: u64) -> bool {
    let mut previous = 0usize;
    for sample in samples.iter().filter(|sample| sample.tick >= from_tick) {
        let behind = sample.behind.unwrap_or(0);
        if behind < previous {
            return false;
        }
        previous = behind;
    }
    true
}

fn ws_always_full(_number: u64) -> Ws {
    Ws::Full
}

fn ws_always_partial(_number: u64) -> Ws {
    Ws::Partial
}

/// WS dies permanently after block 1200 (tick 200 at one block per tick).
fn ws_dies_after_1200(number: u64) -> Ws {
    if number > 1200 { Ws::Missing } else { Ws::Full }
}

/// Alternating 40-block WS outage waves — the BNB churn shape (44 disconnect/reconnect cycles).
fn ws_waves_of_40(number: u64) -> Ws {
    if (number / 40) % 2 == 1 {
        Ws::Missing
    } else {
        Ws::Full
    }
}

// ---- Scenarios -----------------------------------------------------------------------------------

/// Baseline: healthy WS, healthy provider, regular finality. Pins what "good" looks like so the
/// degraded scenarios' signatures are attributable to the degradation alone.
#[test]
#[ignore = "heavy regime simulation; run explicitly with --ignored"]
fn regime_baseline_healthy_provider_keeps_pace() {
    let report = run(Scenario {
        name: "baseline: healthy provider, healthy ws",
        ticks: 400,
        blocks_per_tick: 1,
        head_dup: 1,
        ws: ws_always_full,
        provider: Provider::new(Policies::healthy(1)),
        finality: Some((16, 64)),
        deliver_ticks: true,
        sample_every: 40,
    });

    assert!(
        report.behind().unwrap_or(usize::MAX) <= 2,
        "healthy chain must track the tip, behind={:?}",
        report.behind()
    );
    assert!(
        report.state.canonical_path_len_from_finalized().unwrap_or(usize::MAX) <= 100,
        "finalization must bound the window, window={:?}",
        report.state.canonical_path_len_from_finalized()
    );
    assert!(
        report.state.in_flight_request_count() <= 8,
        "no request leak on a healthy chain, inflight={}",
        report.state.in_flight_request_count()
    );
    assert_eq!(report.state.ws_miss_count(), 0, "full WS delivery is never caught wrong");
    assert_eq!(report.stat("block_logs").issued, 0, "no tip holes when WS delivers everything");
    assert!(
        report.dispatches >= 350,
        "nearly every block should dispatch an optimization, dispatches={}",
        report.dispatches
    );
}

/// WS stalls but the per-block backstop is healthy: the kernel recovers, at the cost of one
/// `GetBlockLogs` per block — i.e. WS death silently reintroduces the per-block fetch rate the
/// WS-primary flip was meant to eliminate (exactly what free-tier providers then throttle).
#[test]
#[ignore = "heavy regime simulation; run explicitly with --ignored"]
fn regime_ws_stall_recovers_via_backstop_at_per_block_fetch_cost() {
    let report = run(Scenario {
        name: "ws stall at block 1200, healthy backstop",
        ticks: 500,
        blocks_per_tick: 1,
        head_dup: 1,
        ws: ws_dies_after_1200,
        provider: Provider::new(Policies::healthy(1)),
        finality: Some((16, 64)),
        deliver_ticks: true,
        sample_every: 50,
    });

    assert!(
        report.behind().unwrap_or(usize::MAX) <= 6,
        "backstop keeps the chain near the tip, behind={:?}",
        report.behind()
    );
    let backstop = report.stat("block_logs");
    // ~300 post-stall blocks, each needing the per-block backstop once it passes settle depth.
    assert!(
        backstop.issued >= 280,
        "per-block fetch rate returns when WS dies, block_logs issued={}",
        backstop.issued
    );
    assert_eq!(backstop.issued, backstop.answered, "healthy backstop answers everything");
}

/// The Polygon 2026-07-15 freeze: WS stalls AND the repair requests (backstop + ranged
/// verification) die silently. The kernel's TTL retries fire every REQUEST_TTL ticks but the
/// provider eats every reissue: the fold frontier freezes forever, `behind` grows monotonically
/// while heads keep streaming, and in-flight count climbs as every new block adds another
/// permanently-pending hole request.
#[test]
#[ignore = "heavy regime simulation; run explicitly with --ignored"]
fn regime_silent_request_death_freezes_frontier_despite_ttl_retries() {
    let degraded = Policies {
        block_logs: ReplyPolicy::SilentlyDrop,
        logs_range: ReplyPolicy::SilentlyDrop,
        ..Policies::healthy(1)
    };
    let report = run(Scenario {
        name: "ws stall + silent repair-request death (Polygon freeze)",
        ticks: 500,
        blocks_per_tick: 1,
        head_dup: 1,
        ws: ws_dies_after_1200,
        provider: Provider::new(Policies::healthy(1)).switching_to(200, degraded),
        finality: Some((16, 64)),
        deliver_ticks: true,
        sample_every: 50,
    });

    let frontier = report.frontier_number().expect("dispatched before the stall");
    assert!(
        frontier <= 1202,
        "frontier must freeze at the first unrepaired hole, frontier={frontier}"
    );
    assert!(
        report.behind().unwrap_or(0) >= 250,
        "behind grows monotonically to ~head-frontier, behind={:?}",
        report.behind()
    );
    assert!(
        behind_is_monotonic_from(&report.samples, 200),
        "behind never recovers while requests die silently"
    );
    assert!(
        report.state.in_flight_request_count() >= 200,
        "every new block leaks another permanently-pending request, inflight={}",
        report.state.in_flight_request_count()
    );
    let backstop = report.stat("block_logs");
    // ~300 holes, each retried every REQUEST_TTL ticks for its lifetime: issuance far exceeds
    // the hole count, proving the TTL retry clock fires (and is uselessly eaten).
    assert!(
        backstop.issued > 1_000,
        "TTL retries keep reissuing into the void, block_logs issued={}",
        backstop.issued
    );
    assert_eq!(backstop.answered, 0, "nothing ever comes back");
    let _ = REQUEST_TTL;
}

/// Same silent death, but the transition thread is starved and never delivers `Event::Tick`: the
/// kernel's ONLY retry clock stops, each hole is requested exactly once, and the freeze is
/// permanent with no reissue traffic at all. Tick starvation (observed under transition-thread
/// saturation in runs 2-3) therefore disables the kernel's entire self-healing path.
#[test]
#[ignore = "heavy regime simulation; run explicitly with --ignored"]
fn regime_tick_starvation_disables_all_retries() {
    let degraded = Policies {
        block_logs: ReplyPolicy::SilentlyDrop,
        logs_range: ReplyPolicy::SilentlyDrop,
        ..Policies::healthy(1)
    };
    let report = run(Scenario {
        name: "silent repair death + starved tick clock",
        ticks: 500,
        blocks_per_tick: 1,
        head_dup: 1,
        ws: ws_dies_after_1200,
        provider: Provider::new(Policies::healthy(1)).switching_to(200, degraded),
        finality: Some((16, 64)),
        deliver_ticks: false,
        sample_every: 50,
    });

    let backstop = report.stat("block_logs");
    // ~300 post-stall holes; without ticks each is issued exactly once (no TTL reissues). The
    // small slack covers scheduling-order variation around the settle window.
    assert!(
        backstop.issued <= 320,
        "no tick clock means no retries: one request per hole, issued={}",
        backstop.issued
    );
    assert!(
        report.behind().unwrap_or(0) >= 250,
        "frontier frozen, behind={:?}",
        report.behind()
    );
}

/// Silent death that later heals: once the provider answers again, the TTL retry clock re-issues
/// every pending request within one REQUEST_TTL window and the chain fully catches up — the
/// kernel-level design self-heals. A production freeze that never recovered therefore means the
/// runtime kept eating the retries too (hung failover with no per-endpoint logging), not that the
/// kernel lacks a retry path.
#[test]
#[ignore = "heavy regime simulation; run explicitly with --ignored"]
fn regime_kernel_self_heals_once_provider_recovers() {
    let degraded = Policies {
        block_logs: ReplyPolicy::SilentlyDrop,
        logs_range: ReplyPolicy::SilentlyDrop,
        ..Policies::healthy(1)
    };
    let report = run(Scenario {
        name: "silent repair death healing at tick 350",
        ticks: 500,
        blocks_per_tick: 1,
        head_dup: 1,
        ws: ws_dies_after_1200,
        provider: Provider::new(Policies::healthy(1))
            .switching_to(200, degraded)
            .switching_to(350, Policies::healthy(1)),
        finality: Some((16, 64)),
        deliver_ticks: true,
        sample_every: 50,
    });

    assert!(
        report.behind().unwrap_or(usize::MAX) <= 6,
        "chain fully catches up after the provider heals, behind={:?}",
        report.behind()
    );
    assert!(
        report.frontier_number().unwrap_or(0) >= report.head - 6,
        "frontier reaches the tip again, frontier={:?} head={}",
        report.frontier_number(),
        report.head
    );
}

/// The BNB 2026-07-15 signature: WS churns in outage waves, the per-block backstop is dead
/// (rate-limited), and ranged `getLogs` deeper than ~128 blocks is refused by every provider
/// (free-tier archive gating). `RequestFailed` triggers an *immediate* reissue, so refused ranges
/// ping-pong — issuance far exceeds answers (the 350-issued/174-answered shape), amplifying the
/// very rate-limiting that caused the refusals, while the frontier stays frozen and the window
/// grows without bound.
#[test]
#[ignore = "heavy regime simulation; run explicitly with --ignored"]
fn regime_archive_refused_ranges_ping_pong_and_freeze_finalization() {
    let bnb_like = Policies {
        // Rate-limited fast failure (429 across the failover set): each failure triggers an
        // immediate kernel reissue, like the refused ranges below.
        block_logs: ReplyPolicy::Fail { delay: 1 },
        logs_range: ReplyPolicy::ArchiveGated {
            delay: 1,
            archive_depth: 128,
        },
        ..Policies::healthy(1)
    };
    let report = run(Scenario {
        name: "ws waves + dead backstop + archive-gated ranges (BNB)",
        ticks: 600,
        blocks_per_tick: 1,
        head_dup: 1,
        ws: ws_waves_of_40,
        provider: Provider::new(bnb_like),
        finality: Some((10, 150)),
        deliver_ticks: true,
        sample_every: 60,
    });

    // The first WS outage wave starts at block 1040; the frontier can never pass its first hole.
    let frontier = report.frontier_number().expect("dispatched before the first wave");
    assert!(
        frontier <= 1042,
        "frontier frozen at the first unrepaired outage wave, frontier={frontier}"
    );
    assert!(
        report.behind().unwrap_or(0) >= 500,
        "behind grows unbounded, behind={:?}",
        report.behind()
    );
    let ranges = report.stat("logs_range");
    assert!(
        ranges.failed >= ranges.answered,
        "archive gating refuses the deep repairs, failed={} answered={}",
        ranges.failed,
        ranges.answered
    );
    assert!(
        ranges.issued >= 2 * ranges.answered + 50,
        "immediate retry-on-failure amplifies issuance well past answers, issued={} answered={}",
        ranges.issued,
        ranges.answered
    );
    let window = report.state.canonical_path_len_from_finalized().unwrap_or(0);
    assert!(
        window >= 500,
        "finalization cannot advance over the frozen holes: window grows unbounded, window={window}"
    );
}

/// Multi-provider head fan-out (Arbitrum delivered each head up to 8x): duplicate deliveries of
/// the current tip must be `Inert` — no scheduling walk, no fold — and orders of magnitude cheaper
/// than first deliveries, or the duplicate traffic alone saturates the transition thread (run 2's
/// failure before the duplicate-of-tip early-out).
#[test]
#[ignore = "heavy regime simulation; run explicitly with --ignored"]
fn regime_duplicate_head_fanout_is_inert_and_cheap() {
    let report = run(Scenario {
        name: "8x duplicate heads on a long window (Arbitrum fan-out)",
        ticks: 600,
        blocks_per_tick: 2,
        head_dup: 8,
        ws: ws_always_full,
        provider: Provider::new(Policies::healthy(1)),
        finality: Some((32, 800)),
        deliver_ticks: true,
        sample_every: 60,
    });

    assert!(
        report.dup_costs.iter().all(|cost| cost.inert),
        "every duplicate-of-tip delivery must be provably inert"
    );
    let firsts: Vec<u128> = report.head_costs.iter().map(|cost| cost.micros).collect();
    let dups: Vec<u128> = report.dup_costs.iter().map(|cost| cost.micros).collect();
    assert!(
        mean(&firsts) >= 10 * mean(&dups).max(1),
        "duplicates must be far cheaper than first deliveries: first mean={}us dup mean={}us",
        mean(&firsts),
        mean(&dups)
    );
    assert!(
        report.behind().unwrap_or(usize::MAX) <= 4,
        "fan-out must not stall the chain, behind={:?}",
        report.behind()
    );
}

/// Dead finality source: the window (anchor -> tip) grows without bound and the per-event cost of
/// the O(window) walks (scheduling, candidate collection, fold-gate frontier) grows with it — the
/// transition-thread saturation regime of runs 2-4. Quantifies how event cost scales as the
/// window stretches.
#[test]
#[ignore = "heavy regime simulation; run explicitly with --ignored"]
fn regime_finalization_stall_grows_window_and_event_cost() {
    let report = run(Scenario {
        name: "finality source dead: unbounded window growth",
        ticks: 2_500,
        blocks_per_tick: 1,
        head_dup: 1,
        ws: ws_always_full,
        provider: Provider::new(Policies::healthy(1)),
        finality: None,
        deliver_ticks: true,
        sample_every: 250,
    });

    let window = report.state.canonical_path_len_from_finalized().unwrap_or(0);
    assert!(window >= 2_400, "window grows with every block, window={window}");
    assert!(
        report.behind().unwrap_or(usize::MAX) <= 2,
        "the chain still tracks the tip; the cost is CPU, not lag, behind={:?}",
        report.behind()
    );

    let early: Vec<u128> = report
        .head_costs
        .iter()
        .filter(|cost| cost.window_before < 500)
        .map(|cost| cost.micros)
        .collect();
    let late: Vec<u128> = report
        .head_costs
        .iter()
        .filter(|cost| cost.window_before >= 2_000)
        .map(|cost| cost.micros)
        .collect();
    println!(
        "  head-event cost by window: <500 -> mean {}us (n={}); >=2000 -> mean {}us (n={})",
        mean(&early),
        early.len(),
        mean(&late),
        late.len()
    );
    assert!(
        mean(&late) >= 2 * mean(&early).max(1),
        "per-event cost grows with the window: early mean={}us late mean={}us",
        mean(&early),
        mean(&late)
    );
}

/// Partial WS delivery is trusted at the tip: the optimizer runs on the incomplete log set until
/// finalization's ranged verification replaces it — each correction increments `ws_miss`, the
/// permanent gauge of "the optimizer ran on wrong data". Confirms the miss is detected and
/// corrected, and measures the exposure window.
#[test]
#[ignore = "heavy regime simulation; run explicitly with --ignored"]
fn regime_partial_ws_delivery_is_trusted_then_counted_as_ws_miss() {
    let report = run(Scenario {
        name: "partial ws delivery corrected at finalization",
        ticks: 300,
        blocks_per_tick: 1,
        head_dup: 1,
        ws: ws_always_partial,
        provider: Provider::new(Policies::healthy(1)),
        finality: Some((16, 64)),
        deliver_ticks: true,
        sample_every: 30,
    });

    assert!(
        report.behind().unwrap_or(usize::MAX) <= 2,
        "partial delivery does not stall the chain, behind={:?}",
        report.behind()
    );
    // Blocks finalized-and-verified so far: roughly head - anchor-lag; every one of them was
    // trusted with a wrong (partial) log set first.
    assert!(
        report.state.ws_miss_count() >= 100,
        "every corrected block counts one ws_miss, ws_miss={}",
        report.state.ws_miss_count()
    );
}

/// Orphaned streamed logs (WS logs for blocks whose heads never arrive) are staged in a bounded
/// buffer: the cap holds under a flood, and logs staged for a block that later arrives still drain
/// into the graph.
#[test]
#[ignore = "heavy regime simulation; run explicitly with --ignored"]
fn regime_orphaned_streamed_log_flood_stays_bounded() {
    let (mut state, _) = State::activate_from_seed(
        block_hash(ANCHOR_NUMBER),
        ANCHOR_NUMBER,
        HashMap::new(),
        verified_pool_registry(),
        verified_token_registry(),
        Vec::new(),
    );

    for index in 0..3_000u64 {
        let (next, effects) = super::transition(
            CHAIN,
            state,
            Event::LogObserved {
                block_hash: block_hash(9_000_000 + index),
                logs: block_logs(ANCHOR_NUMBER + 1),
            },
        );
        assert!(effects.is_empty(), "staging produces no effects");
        state = next;
    }
    assert_eq!(
        state.streamed_logs.len(),
        MAX_STREAMED_LOG_BLOCKS,
        "the staging buffer must cap at MAX_STREAMED_LOG_BLOCKS under a flood"
    );

    // A staged block whose head finally arrives still drains into the graph.
    let staged_hash = block_hash(9_000_000);
    assert!(state.streamed_logs.contains_key(&staged_hash), "first flood entry is staged");
    let (state, _) = super::transition(
        CHAIN,
        state,
        Event::HeadObserved {
            hash: staged_hash,
            parent_hash: block_hash(ANCHOR_NUMBER),
            logs_bloom: block_bloom(ANCHOR_NUMBER + 1),
            number: ANCHOR_NUMBER + 1,
        },
    );
    assert!(
        !state.streamed_logs.contains_key(&staged_hash),
        "staged logs drain when the block enters the graph"
    );
    println!(
        "== regime: orphaned streamed-log flood == cap held at {} staged blocks",
        MAX_STREAMED_LOG_BLOCKS
    );
}
