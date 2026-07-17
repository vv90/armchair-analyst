//! Scaling microbenchmarks for the chain kernel's two O(window) read paths — the costs that drove
//! the transition-thread starvation spiral in mainnet runs 2-4.
//!
//! The failure was a *scaling* bug: per-event work grows with the canonical window
//! (`finalized → tip`), so a single blended number hides it (fine at small window, detonates at
//! large). These benches therefore parametrize over the input that blows up and read the growth
//! *shape* of the curve, not one absolute µs figure.
//!
//! Two hot paths, both exercised here as non-consuming `&self` reads (the `bench_*` shims exposed
//! under the `test-support` feature), so one window is built once and measured repeatedly without a
//! `Clone` on `State`:
//!
//! - **fold read** (`State::bench_optimization_fold`): the post-transition optimization fold the
//!   multi-chain wrapper runs — O(window) over foldable (`Complete`) blocks.
//! - **hole scan** (`State::bench_unknown_hole_scan`): the tip-hole scheduling scan — the
//!   `path × trusted-addresses` bloom-touch keccaks that were the run-4 root cause, over `Unknown`
//!   blocks.
//!
//! Axes: **window size** (primary — both reads), and **trusted-set size** (the `× addresses`
//! multiplier, hole scan only). Chain count is deliberately absent: a single read is per-chain and
//! cannot see other chains — chain-count effects are shared-thread throughput, a separate macro
//! benchmark. The duplicate-of-tip `Inert` early-out is a *consuming* `transition` property and
//! stays pinned by `regime_duplicate_head_fanout_is_inert` rather than here.
//!
//! Run: `cargo bench -p client-evm --features test-support --bench kernel_scaling`.

#![allow(clippy::expect_used)]

use std::collections::HashMap;

use alloy::primitives::{Address, BloomInput, U160, U256, aliases::I24};
use client_evm::kernel::{Event, State, transition};
use client_evm::{
    BlockHash, Bloom, ChainKey, PoolFee, PoolLog, PoolLogEvent, PoolMetadata, PoolMetadataResult,
    ProtocolPoolKey, TokenAddress, TokenDecimals, TokenMetadata, TokenMetadataResult, TokenRegistry,
    TrustedPoolRegistry, UniswapV3Fee,
};
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

const CHAIN: ChainKey = ChainKey::Ethereum;
const ANCHOR: u64 = 1_000;

// ---- Deterministic world (mirrors kernel::regimes so the benched cost matches the pinned one) ----

/// Block hash encodes its number, so every block deterministically extends its predecessor.
fn block_hash(number: u64) -> BlockHash {
    let mut bytes = [0u8; 32];
    bytes[0] = 0xB1;
    bytes[24..].copy_from_slice(&number.to_be_bytes());
    BlockHash::from(bytes)
}

/// Unique pool address per index (full index encoded), so the trusted set can grow past 256 for the
/// `× addresses` axis.
fn pool_address(index: u64) -> Address {
    let mut bytes = [0u8; 20];
    bytes[0] = 0xA0;
    bytes[12..20].copy_from_slice(&index.to_be_bytes());
    Address::from(bytes)
}

fn token_address(index: u64) -> Address {
    let mut bytes = [0u8; 20];
    bytes[0] = 0xC0;
    bytes[12..20].copy_from_slice(&index.to_be_bytes());
    Address::from(bytes)
}

fn pool_key(index: u64) -> ProtocolPoolKey {
    ProtocolPoolKey::UniswapV3(pool_address(index))
}

/// The two pools a block touches; distinct by construction (7 coprime-enough to the pool count).
fn block_pools(number: u64, pools: u64) -> [u64; 2] {
    [number % pools, (number + 7) % pools]
}

fn block_logs(number: u64, pools: u64) -> Vec<PoolLog> {
    block_pools(number, pools)
        .iter()
        .enumerate()
        .map(|(position, &pool)| PoolLog {
            pool: pool_key(pool),
            log_index: position as u64,
            event: PoolLogEvent::Swap {
                sqrt_price_x96: U160::from(number),
                tick: I24::try_from(0).expect("zero tick fits int24"),
                liquidity: 1 + u128::from(number),
            },
        })
        .collect()
}

/// Every block bloom-touches its two (verified) pools, so an `Unknown` block always registers as a
/// hole — the worst case the scheduling scan pays for.
fn block_bloom(number: u64, pools: u64) -> Bloom {
    let mut bloom = Bloom::default();
    for pool in block_pools(number, pools) {
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

/// `pools` pre-verified pools, so the scan's trusted set has exactly `pools` addresses.
fn verified_pool_registry(pools: u64) -> TrustedPoolRegistry {
    let results: HashMap<ProtocolPoolKey, PoolMetadataResult> = (0..pools)
        .map(|index| (pool_key(index), Ok(world_pool_metadata(index))))
        .collect();
    TrustedPoolRegistry::new().with_metadata_results(CHAIN, results)
}

fn verified_token_registry(pools: u64) -> TokenRegistry {
    let results: HashMap<TokenAddress, TokenMetadataResult> = (0..=pools)
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

fn head_event(number: u64, pools: u64) -> Event {
    Event::HeadObserved {
        hash: block_hash(number),
        parent_hash: block_hash(number - 1),
        logs_bloom: block_bloom(number, pools),
        number,
    }
}

fn log_event(number: u64, pools: u64) -> Event {
    Event::LogObserved {
        block_hash: block_hash(number),
        logs: block_logs(number, pools),
    }
}

// ---- Window construction (live head path — the one proven to reach window=2500 in regimes) -------

fn base_state(pools: u64) -> State {
    let (state, _) = State::activate_from_seed(
        block_hash(ANCHOR),
        HashMap::new(),
        verified_pool_registry(pools),
        verified_token_registry(pools),
        Vec::new(),
    );
    state
}

/// A window of `n` foldable (`Complete`) blocks: each block's head then its full log set, so the
/// fold walks the whole window. No finalization, so the window stays at `n`. O(n²) to build (each
/// head is itself an O(window) walk) — done once per size, off the measured path.
fn complete_window(n: u64, pools: u64) -> State {
    let mut state = base_state(pools);
    for k in 1..=n {
        let number = ANCHOR + k;
        let (next, _) = transition(CHAIN, state, head_event(number, pools));
        let (next, _) = transition(CHAIN, next, log_event(number, pools));
        state = next;
    }
    state
}

/// A window of `n` `Unknown` blocks: heads only, logs never delivered, so every block is a
/// bloom-touching hole the scheduling scan must check against the trusted set. O(n²) to build.
fn unknown_window(n: u64, pools: u64) -> State {
    let mut state = base_state(pools);
    for k in 1..=n {
        let (next, _) = transition(CHAIN, state, head_event(ANCHOR + k, pools));
        state = next;
    }
    state
}

// ---- Benchmarks ----------------------------------------------------------------------------------

const WINDOWS: [u64; 4] = [100, 500, 1000, 2500];
const FOLD_POOLS: u64 = 16;
const HOLE_POOLS: u64 = 16;
const TRUSTED_SET_WINDOW: u64 = 500;
const TRUSTED_SETS: [u64; 4] = [16, 64, 256, 1024];

/// PRIMARY window axis: the post-transition optimization fold over a foldable window. Expect the
/// curve to grow with window (the O(window) fold); a regression that re-inflates the constant bends
/// it up further.
fn fold_read_by_window(c: &mut Criterion) {
    let mut group = c.benchmark_group("fold_read_by_window");
    for &window in &WINDOWS {
        let state = complete_window(window, FOLD_POOLS);
        group.bench_with_input(BenchmarkId::from_parameter(window), &window, |b, _| {
            b.iter(|| black_box(state.bench_optimization_fold(CHAIN)));
        });
    }
    group.finish();
}

/// Window axis for the scheduling scan: the `path × addresses` bloom hot path over an all-`Unknown`
/// window (trusted set fixed). Isolates how the run-4 keccak cost scales with window length.
fn hole_scan_by_window(c: &mut Criterion) {
    let mut group = c.benchmark_group("hole_scan_by_window");
    for &window in &WINDOWS {
        let state = unknown_window(window, HOLE_POOLS);
        group.bench_with_input(BenchmarkId::from_parameter(window), &window, |b, _| {
            b.iter(|| black_box(state.bench_unknown_hole_scan(CHAIN)));
        });
    }
    group.finish();
}

/// Trusted-set axis: window fixed, trusted-address count swept. Exposes the `× addresses` multiplier
/// directly, so the measured cost reflects the real `path × addresses` product, not just path length.
fn hole_scan_by_trusted_set(c: &mut Criterion) {
    let mut group = c.benchmark_group("hole_scan_by_trusted_set");
    for &pools in &TRUSTED_SETS {
        let state = unknown_window(TRUSTED_SET_WINDOW, pools);
        group.bench_with_input(BenchmarkId::from_parameter(pools), &pools, |b, _| {
            b.iter(|| black_box(state.bench_unknown_hole_scan(CHAIN)));
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    fold_read_by_window,
    hole_scan_by_window,
    hole_scan_by_trusted_set
);
criterion_main!(benches);
