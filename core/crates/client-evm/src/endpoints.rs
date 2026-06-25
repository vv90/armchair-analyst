//! Multi-provider HTTP endpoint pools with weighted selection and per-request failover.
//!
//! A single upstream (e.g. the dRPC free tier) is a single point of failure: when it returns a
//! gateway error storm there is nowhere else to send a request. This module spreads each chain's HTTP
//! load across several endpoints via weighted round-robin and, when one endpoint errors, transparently
//! fails the request over to an alternative. Selection and health live entirely in this IO-side layer;
//! request semantics (ids, parsing) are unchanged, so the pure kernel/bootstrap state machines and the
//! existing TTL retry are untouched.

use std::{
    collections::BTreeMap,
    sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use crate::{
    ClientEvmError,
    chain::{ACTIVE_CHAINS, ChainKey},
    config::{RpcConfig, TheGraphConfig, compose_graph_endpoint, compose_http_endpoint},
    error::ConfigScope,
    uniswap_v4::state_view_address,
};

/// Base cooldown applied after one failure; doubles per consecutive failure up to [`COOLDOWN_CAP`].
const COOLDOWN_BASE: Duration = Duration::from_secs(2);
/// Upper bound on a failing endpoint's cooldown, so it always returns to rotation eventually.
const COOLDOWN_CAP: Duration = Duration::from_secs(60);
/// Largest backoff doubling exponent (`COOLDOWN_BASE << 5` = 64s, clamped to [`COOLDOWN_CAP`]).
const MAX_BACKOFF_SHIFT: u32 = 5;
/// Weight given to the configured dRPC primary relative to keyless public fallbacks (weight 1).
const DRPC_PRIMARY_WEIGHT: u32 = 3;

/// A description of one endpoint before it is installed into a pool. `url` is the fully-formed POST
/// target (network path and any key already baked in); `label` identifies the provider in logs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EndpointSpec {
    pub url: String,
    pub label: String,
    pub weight: u32,
}

impl EndpointSpec {
    pub fn new(label: impl Into<String>, url: impl Into<String>, weight: u32) -> EndpointSpec {
        EndpointSpec {
            url: url.into(),
            label: label.into(),
            weight,
        }
    }
}

#[derive(Debug)]
struct Endpoint {
    url: String,
    #[allow(dead_code)]
    label: String,
}

#[derive(Clone, Copy, Debug)]
struct Health {
    consecutive_failures: u32,
    cooldown_until: Option<Instant>,
}

impl Health {
    const fn healthy() -> Health {
        Health {
            consecutive_failures: 0,
            cooldown_until: None,
        }
    }
}

/// A chain's HTTP endpoint pool: a weighted, health-aware rotation shared (by `&ref`) across the
/// concurrent multicall batch workers. `cursor`/`health` are interior-mutable so the pool stays
/// `Sync`.
#[derive(Debug)]
pub struct EndpointPool {
    endpoints: Vec<Endpoint>,
    /// Smooth-weighted-round-robin expansion of endpoint indices (length = sum of weights); the
    /// cursor walks this so picks interleave by weight instead of bursting one provider.
    order: Vec<usize>,
    cursor: AtomicUsize,
    health: Mutex<Vec<Health>>,
}

impl EndpointPool {
    pub fn new(specs: Vec<EndpointSpec>) -> Result<EndpointPool, ClientEvmError> {
        if specs.is_empty() {
            return Err(ClientEvmError::InvalidConfig {
                scope: ConfigScope::Http,
                reason: "at least one rpc endpoint is required".to_owned(),
            });
        }

        let weights: Vec<u32> = specs.iter().map(|spec| spec.weight.max(1)).collect();
        let order = smooth_weighted_order(&weights);
        let health = vec![Health::healthy(); specs.len()];
        let endpoints = specs
            .into_iter()
            .map(|spec| Endpoint {
                url: spec.url,
                label: spec.label,
            })
            .collect();

        Ok(EndpointPool {
            endpoints,
            order,
            cursor: AtomicUsize::new(0),
            health: Mutex::new(health),
        })
    }

    /// Builds a single-endpoint pool. Convenience for callers (and tests) that have exactly one URL.
    pub fn single(label: impl Into<String>, url: impl Into<String>) -> EndpointPool {
        EndpointPool::new(vec![EndpointSpec::new(label, url, 1)])
            .expect("single endpoint spec is always non-empty")
    }

    /// Runs `attempt` against endpoints from the pool, failing over to an untried alternative on any
    /// retryable error. `attempt` covers the whole build+send+parse unit, so JSON-RPC gateway errors
    /// (detected during parsing) fail over just like transport errors. Returns the last error once the
    /// bounded attempts are exhausted.
    pub fn with_failover<T>(
        &self,
        attempt: impl Fn(&str) -> Result<T, ClientEvmError>,
    ) -> Result<T, ClientEvmError> {
        let attempts = self.endpoints.len();
        let mut tried: Vec<usize> = Vec::with_capacity(attempts);
        let mut last_error: Option<ClientEvmError> = None;

        for _ in 0..attempts {
            let now = Instant::now();
            let idx = self.pick_excluding(&tried, now);

            match attempt(&self.endpoints[idx].url) {
                Ok(value) => {
                    self.report_success(idx);
                    return Ok(value);
                }
                Err(error) if is_retryable(&error) => {
                    self.report_failure(idx, now);
                    tried.push(idx);
                    last_error = Some(error);
                }
                // A non-retryable error (config fault, or a request-shaped HTTP 400/404/405) is the
                // same on every endpoint; surface it without burning the rest of the pool.
                Err(error) => return Err(error),
            }
        }

        Err(last_error.expect("with_failover runs at least one attempt"))
    }

    /// Selects the next endpoint, preferring one that is both untried (this request) and not cooling
    /// down. Falls back to the untried endpoint whose cooldown expires soonest, so a request never
    /// stalls even when every endpoint is penalized.
    fn pick_excluding(&self, tried: &[usize], now: Instant) -> usize {
        let health = self.health.lock().expect("endpoint health mutex poisoned");
        let len = self.order.len();
        let start = self.cursor.fetch_add(1, Ordering::Relaxed);

        let mut fallback: Option<(usize, Instant)> = None;
        for offset in 0..len {
            let idx = self.order[(start + offset) % len];
            if tried.contains(&idx) {
                continue;
            }
            match health[idx].cooldown_until {
                None => return idx,
                Some(until) if until <= now => return idx,
                Some(until) => {
                    if fallback.is_none_or(|(_, earliest)| until < earliest) {
                        fallback = Some((idx, until));
                    }
                }
            }
        }

        fallback
            .map(|(idx, _)| idx)
            .unwrap_or_else(|| self.order[start % len])
    }

    /// Test/inspection helper: the next pick assuming nothing has been tried in this request.
    #[cfg(test)]
    fn pick(&self, now: Instant) -> usize {
        self.pick_excluding(&[], now)
    }

    fn report_success(&self, idx: usize) {
        let mut health = self.health.lock().expect("endpoint health mutex poisoned");
        health[idx] = Health::healthy();
    }

    fn report_failure(&self, idx: usize, now: Instant) {
        let mut health = self.health.lock().expect("endpoint health mutex poisoned");
        let entry = &mut health[idx];
        entry.consecutive_failures = entry.consecutive_failures.saturating_add(1);
        entry.cooldown_until = Some(now + backoff_duration(entry.consecutive_failures));
    }
}

/// All chains' HTTP endpoint pools. Built once at startup and shared by `&ref`, so per-endpoint health
/// persists across requests.
#[derive(Debug)]
pub struct ChainEndpoints {
    pools: BTreeMap<ChainKey, EndpointPool>,
}

impl ChainEndpoints {
    pub fn pool(&self, chain: ChainKey) -> Result<&EndpointPool, ClientEvmError> {
        self.pools
            .get(&chain)
            .ok_or_else(|| ClientEvmError::InvalidConfig {
                scope: ConfigScope::Http,
                reason: format!("no rpc endpoints configured for chain {chain:?}"),
            })
    }

    /// Builds a single-endpoint, single-chain `ChainEndpoints`. Convenience for tests and minimal setups.
    pub fn single(
        chain: ChainKey,
        label: impl Into<String>,
        url: impl Into<String>,
    ) -> ChainEndpoints {
        let mut pools = BTreeMap::new();
        pools.insert(chain, EndpointPool::single(label, url));
        ChainEndpoints { pools }
    }
}

/// Assembles each active chain's pool from, in priority order: the configured dRPC primary (weight
/// [`DRPC_PRIMARY_WEIGHT`]), any custom providers from the endpoints file, and — unless disabled — the
/// built-in keyless public fallbacks (weight 1). A chain that ends up with no endpoint is a hard error.
pub fn assemble_chain_endpoints(
    config: &RpcConfig,
    custom: &BTreeMap<ChainKey, Vec<EndpointSpec>>,
    include_public_fallbacks: bool,
) -> Result<ChainEndpoints, ClientEvmError> {
    let mut pools = BTreeMap::new();

    for &chain in ACTIVE_CHAINS {
        let mut specs = Vec::new();
        specs.push(EndpointSpec::new(
            "drpc",
            compose_http_endpoint(config, chain)?,
            DRPC_PRIMARY_WEIGHT,
        ));

        if let Some(extra) = custom.get(&chain) {
            specs.extend(extra.iter().cloned());
        }

        if include_public_fallbacks {
            specs.extend(default_public_endpoints(chain));
        }

        pools.insert(chain, EndpointPool::new(specs)?);
    }

    Ok(ChainEndpoints { pools })
}

/// Per-chain Uniswap v4 subgraph endpoint pools. Unlike [`ChainEndpoints`], coverage is **partial and
/// optional**: only chains with a v4 deployment carry a pool, and an unconfigured runtime holds none at
/// all (`empty()`), in which case v4 metadata resolution is simply skipped. Reuses [`EndpointPool`]
/// verbatim — the gateway primary plus any same-schema mirrors fail over through [`EndpointPool::with_failover`].
#[derive(Debug)]
pub struct GraphEndpoints {
    pools: BTreeMap<ChainKey, EndpointPool>,
}

impl GraphEndpoints {
    /// The subgraph pool for a chain, or `None` when no v4 subgraph is configured for it. `None` is a
    /// normal "skip v4 here" signal, not a fault — so this returns `Option`, not a `Result` like
    /// [`ChainEndpoints::pool`].
    pub fn pool(&self, chain: ChainKey) -> Option<&EndpointPool> {
        self.pools.get(&chain)
    }

    /// An unconfigured set with no subgraph pools — the runtime default when The Graph is not set up.
    pub fn empty() -> GraphEndpoints {
        GraphEndpoints {
            pools: BTreeMap::new(),
        }
    }

    /// Builds a single-endpoint, single-chain `GraphEndpoints`. Convenience for tests and minimal setups.
    pub fn single(
        chain: ChainKey,
        label: impl Into<String>,
        url: impl Into<String>,
    ) -> GraphEndpoints {
        let mut pools = BTreeMap::new();
        pools.insert(chain, EndpointPool::single(label, url));
        GraphEndpoints { pools }
    }
}

/// Assembles each v4-enabled chain's subgraph pool from the configured gateway primary plus any
/// same-schema `custom` mirrors. A chain is v4-enabled iff [`state_view_address`] knows it — the single
/// source of truth for "v4 is deployed here" — so non-v4 chains get no pool (and v4 metadata is skipped
/// for them). Cross-provider failover only works for mirrors serving the canonical Uniswap v4 subgraph
/// schema, which the query/parser assume.
pub fn assemble_graph_endpoints(
    config: &TheGraphConfig,
    custom: &BTreeMap<ChainKey, Vec<EndpointSpec>>,
) -> Result<GraphEndpoints, ClientEvmError> {
    let mut pools = BTreeMap::new();

    for &chain in ACTIVE_CHAINS {
        if state_view_address(chain).is_none() {
            continue;
        }

        let mut specs = vec![EndpointSpec::new(
            "thegraph",
            compose_graph_endpoint(config)?,
            DRPC_PRIMARY_WEIGHT,
        )];

        if let Some(extra) = custom.get(&chain) {
            specs.extend(extra.iter().cloned());
        }

        pools.insert(chain, EndpointPool::new(specs)?);
    }

    Ok(GraphEndpoints { pools })
}

/// Keyless public RPC endpoints bundled per chain (no signup required), used as low-weight fallbacks so
/// multi-provider resilience works out of the box.
pub fn default_public_endpoints(chain: ChainKey) -> Vec<EndpointSpec> {
    let entries: &[(&str, &str)] = match chain {
        ChainKey::Ethereum => &[
            ("publicnode", "https://ethereum-rpc.publicnode.com"),
            ("ankr", "https://rpc.ankr.com/eth"),
            ("1rpc", "https://1rpc.io/eth"),
        ],
        ChainKey::Arbitrum => &[
            ("publicnode", "https://arbitrum-one-rpc.publicnode.com"),
            ("ankr", "https://rpc.ankr.com/arbitrum"),
            ("1rpc", "https://1rpc.io/arb"),
        ],
    };

    entries
        .iter()
        .map(|(label, url)| EndpointSpec::new(*label, *url, 1))
        .collect()
}

/// Whether an error warrants failing the request over to another endpoint. Transport faults, retryable
/// HTTP statuses, malformed responses, and any JSON-RPC error (here always gateway/node-level — pruned
/// state, "temporary internal error", rate limits, unsupported method) are all worth trying elsewhere;
/// only a configuration fault is identical on every endpoint and so is not retried.
pub(crate) fn is_retryable(error: &ClientEvmError) -> bool {
    match error {
        ClientEvmError::HttpTransport(_) => true,
        ClientEvmError::HttpStatus { status, .. } => {
            // 401/403 are per-endpoint conditions (missing/insufficient key, archive-token-required,
            // some providers' rate-limit), not request-shaped: another endpoint may serve the same
            // request, so fail over (and bench this one). 400/404/405 are the same on every endpoint.
            *status == 401
                || *status == 403
                || *status == 408
                || *status == 425
                || *status == 429
                || *status >= 500
        }
        ClientEvmError::JsonRpcError { .. } => true,
        ClientEvmError::MalformedResponse { .. } => true,
        ClientEvmError::InvalidConfig { .. }
        | ClientEvmError::WebSocketError(_)
        | ClientEvmError::EventReceiverDropped => false,
    }
}

fn backoff_duration(consecutive_failures: u32) -> Duration {
    let shift = consecutive_failures
        .saturating_sub(1)
        .min(MAX_BACKOFF_SHIFT);
    (COOLDOWN_BASE * (1u32 << shift)).min(COOLDOWN_CAP)
}

/// Smooth weighted round-robin (nginx SWRR): produces one balanced cycle of endpoint indices of length
/// `sum(weights)`, interleaving higher-weight endpoints rather than emitting them in a block.
fn smooth_weighted_order(weights: &[u32]) -> Vec<usize> {
    let total: i64 = weights.iter().map(|&weight| i64::from(weight)).sum();
    let mut current = vec![0i64; weights.len()];
    let mut order = Vec::with_capacity(total as usize);

    for _ in 0..total {
        for (value, &weight) in current.iter_mut().zip(weights) {
            *value += i64::from(weight);
        }
        let selected = current
            .iter()
            .enumerate()
            .max_by_key(|&(_, &value)| value)
            .map(|(index, _)| index)
            .expect("weights is non-empty when total > 0");
        current[selected] -= total;
        order.push(selected);
    }

    order
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pool(weights: &[u32]) -> EndpointPool {
        let specs = weights
            .iter()
            .enumerate()
            .map(|(index, &weight)| {
                EndpointSpec::new(
                    format!("ep{index}"),
                    format!("http://ep{index}.invalid"),
                    weight,
                )
            })
            .collect();
        EndpointPool::new(specs).expect("non-empty specs")
    }

    #[test]
    fn smooth_weighted_order_distributes_by_weight() {
        let order = smooth_weighted_order(&[3, 1, 1]);

        assert_eq!(order.len(), 5);
        assert_eq!(order.iter().filter(|&&idx| idx == 0).count(), 3);
        assert_eq!(order.iter().filter(|&&idx| idx == 1).count(), 1);
        assert_eq!(order.iter().filter(|&&idx| idx == 2).count(), 1);
    }

    #[test]
    fn smooth_weighted_order_interleaves_rather_than_bursts() {
        // The highest-weight endpoint must not occupy a contiguous run the length of its weight.
        let order = smooth_weighted_order(&[3, 1, 1]);
        let max_run = order
            .iter()
            .fold((0usize, 0usize, usize::MAX), |(max, run, prev), &idx| {
                let run = if idx == prev { run + 1 } else { 1 };
                (max.max(run), run, idx)
            })
            .0;
        assert!(
            max_run < 3,
            "weighted picks should interleave, got {order:?}"
        );
    }

    #[test]
    fn empty_pool_is_rejected() {
        assert!(matches!(
            EndpointPool::new(Vec::new()),
            Err(ClientEvmError::InvalidConfig {
                scope: ConfigScope::Http,
                ..
            })
        ));
    }

    #[test]
    fn weighted_picks_track_weights_over_a_cycle() {
        let pool = pool(&[3, 1, 1]);
        let now = Instant::now();

        let mut counts = [0usize; 3];
        for _ in 0..5 {
            counts[pool.pick(now)] += 1;
        }

        assert_eq!(counts, [3, 1, 1]);
    }

    #[test]
    fn failover_advances_to_next_endpoint_and_penalizes_the_failed_one() {
        let pool = pool(&[1, 1]);
        let calls = std::cell::Cell::new(0);
        // Fail whichever endpoint is picked first, then succeed on the (different) second pick.
        let first_url = std::cell::RefCell::new(None::<String>);

        let result = pool.with_failover(|url| {
            calls.set(calls.get() + 1);
            let mut first = first_url.borrow_mut();
            match first.as_deref() {
                None => {
                    *first = Some(url.to_owned());
                    Err(ClientEvmError::HttpStatus {
                        status: 503,
                        body: "down".to_owned(),
                    })
                }
                Some(previous) => {
                    assert_ne!(previous, url, "failover must use a different endpoint");
                    Ok(url.to_owned())
                }
            }
        });

        assert!(result.is_ok());
        assert_eq!(calls.get(), 2);
        // Exactly one endpoint (the one tried first) was penalized.
        let health = pool.health.lock().unwrap();
        let penalized = health
            .iter()
            .filter(|entry| entry.cooldown_until.is_some())
            .count();
        assert_eq!(penalized, 1);
        assert!(health.iter().any(|entry| entry.consecutive_failures == 1));
    }

    #[test]
    fn failover_returns_last_error_when_all_attempts_fail() {
        let pool = pool(&[1, 1]);

        let result: Result<(), _> = pool.with_failover(|_| {
            Err(ClientEvmError::HttpStatus {
                status: 503,
                body: "down".to_owned(),
            })
        });

        assert!(matches!(
            result,
            Err(ClientEvmError::HttpStatus { status: 503, .. })
        ));
    }

    #[test]
    fn non_retryable_error_surfaces_without_trying_other_endpoints() {
        let pool = pool(&[1, 1]);
        let calls = std::cell::Cell::new(0);

        let result: Result<(), _> = pool.with_failover(|_| {
            calls.set(calls.get() + 1);
            Err(ClientEvmError::InvalidConfig {
                scope: ConfigScope::Http,
                reason: "bad".to_owned(),
            })
        });

        assert!(matches!(result, Err(ClientEvmError::InvalidConfig { .. })));
        assert_eq!(calls.get(), 1);
    }

    #[test]
    fn http_403_fails_over_and_benches_the_endpoint() {
        // Regression: a 403 (archive-token-required / auth) from one endpoint must not abort the
        // request — it should fail over to the next endpoint and bench the 403 one.
        let pool = pool(&[1, 1]);
        let first_url = std::cell::RefCell::new(None::<String>);

        let result = pool.with_failover(|url| {
            let mut first = first_url.borrow_mut();
            match first.as_deref() {
                None => {
                    *first = Some(url.to_owned());
                    Err(ClientEvmError::HttpStatus {
                        status: 403,
                        body: "archive token required".to_owned(),
                    })
                }
                Some(previous) => {
                    assert_ne!(previous, url, "403 must fail over to a different endpoint");
                    Ok(url.to_owned())
                }
            }
        });

        assert!(result.is_ok());
        let health = pool.health.lock().unwrap();
        assert_eq!(
            health
                .iter()
                .filter(|entry| entry.cooldown_until.is_some())
                .count(),
            1,
            "the 403 endpoint should be benched"
        );
    }

    #[test]
    fn failover_tries_every_endpoint_in_a_larger_pool() {
        // With the per-request attempt cap removed, a request reaches all endpoints before giving up.
        let pool = pool(&[1, 1, 1, 1]);
        let calls = std::cell::Cell::new(0);

        let result: Result<(), _> = pool.with_failover(|_| {
            calls.set(calls.get() + 1);
            Err(ClientEvmError::HttpStatus {
                status: 503,
                body: "down".to_owned(),
            })
        });

        assert!(result.is_err());
        assert_eq!(calls.get(), 4, "every endpoint should be tried once");
    }

    #[test]
    fn penalized_endpoint_is_skipped_until_cooldown_then_returns() {
        let pool = pool(&[1, 1]);
        let now = Instant::now();

        // Penalize ep0; while cooling down, picks avoid it.
        pool.report_failure(0, now);
        let during_cooldown = now + Duration::from_millis(1);
        for _ in 0..4 {
            assert_eq!(pool.pick(during_cooldown), 1);
        }

        // After the cooldown elapses, ep0 re-enters rotation.
        let after_cooldown = now + COOLDOWN_BASE + Duration::from_secs(1);
        let mut saw_zero = false;
        for _ in 0..4 {
            if pool.pick(after_cooldown) == 0 {
                saw_zero = true;
            }
        }
        assert!(saw_zero, "ep0 should return after cooldown");
    }

    #[test]
    fn all_endpoints_cooling_down_still_yields_a_pick() {
        let pool = pool(&[1, 1]);
        let now = Instant::now();
        pool.report_failure(0, now);
        pool.report_failure(1, now);

        // No panic, returns a valid index even though both are penalized.
        let idx = pool.pick(now + Duration::from_millis(1));
        assert!(idx < 2);
    }

    #[test]
    fn report_success_clears_penalty() {
        let pool = pool(&[1, 1]);
        let now = Instant::now();
        pool.report_failure(0, now);
        pool.report_success(0);

        let health = pool.health.lock().unwrap();
        assert_eq!(health[0].consecutive_failures, 0);
        assert!(health[0].cooldown_until.is_none());
    }

    #[test]
    fn backoff_grows_then_caps() {
        assert_eq!(backoff_duration(1), COOLDOWN_BASE);
        assert_eq!(backoff_duration(2), COOLDOWN_BASE * 2);
        assert_eq!(backoff_duration(3), COOLDOWN_BASE * 4);
        assert_eq!(backoff_duration(100), COOLDOWN_CAP);
    }

    #[test]
    fn is_retryable_classifies_errors() {
        assert!(is_retryable(&ClientEvmError::HttpStatus {
            status: 500,
            body: String::new()
        }));
        assert!(is_retryable(&ClientEvmError::HttpStatus {
            status: 503,
            body: String::new()
        }));
        assert!(is_retryable(&ClientEvmError::HttpStatus {
            status: 429,
            body: String::new()
        }));
        // 401/403 are per-endpoint (auth / archive-token / provider rate-limit) → fail over.
        assert!(is_retryable(&ClientEvmError::HttpStatus {
            status: 401,
            body: String::new()
        }));
        assert!(is_retryable(&ClientEvmError::HttpStatus {
            status: 403,
            body: String::new()
        }));
        assert!(is_retryable(&ClientEvmError::JsonRpcError {
            code: "19".to_owned(),
            message: "temporary internal error".to_owned()
        }));
        assert!(is_retryable(&ClientEvmError::MalformedResponse {
            context: String::new(),
            detail: String::new()
        }));
        assert!(!is_retryable(&ClientEvmError::HttpStatus {
            status: 400,
            body: String::new()
        }));
        assert!(!is_retryable(&ClientEvmError::HttpStatus {
            status: 404,
            body: String::new()
        }));
        assert!(!is_retryable(&ClientEvmError::InvalidConfig {
            scope: ConfigScope::Http,
            reason: String::new()
        }));
    }

    #[test]
    fn default_public_endpoints_cover_active_chains() {
        assert!(!default_public_endpoints(ChainKey::Ethereum).is_empty());
        assert!(!default_public_endpoints(ChainKey::Arbitrum).is_empty());
    }

    #[test]
    fn assemble_includes_primary_and_publics_and_omits_publics_when_disabled() {
        let config = RpcConfig {
            http_url: "https://lb.drpc.org".to_owned(),
            ws_url: "wss://lb.drpc.org".to_owned(),
            api_key: "key".to_owned(),
        };

        let with_publics =
            assemble_chain_endpoints(&config, &BTreeMap::new(), true).expect("assembly succeeds");
        let eth = with_publics
            .pool(ChainKey::Ethereum)
            .expect("ethereum pool");
        let expected = 1 + default_public_endpoints(ChainKey::Ethereum).len();
        assert_eq!(eth.endpoints.len(), expected);

        let without_publics =
            assemble_chain_endpoints(&config, &BTreeMap::new(), false).expect("assembly succeeds");
        let eth = without_publics
            .pool(ChainKey::Ethereum)
            .expect("ethereum pool");
        assert_eq!(eth.endpoints.len(), 1);
    }

    #[test]
    fn assemble_appends_custom_providers() {
        let config = RpcConfig {
            http_url: "https://lb.drpc.org".to_owned(),
            ws_url: "wss://lb.drpc.org".to_owned(),
            api_key: "key".to_owned(),
        };
        let mut custom = BTreeMap::new();
        custom.insert(
            ChainKey::Arbitrum,
            vec![EndpointSpec::new(
                "alchemy",
                "https://arb.example/v2/key",
                3,
            )],
        );

        let endpoints =
            assemble_chain_endpoints(&config, &custom, false).expect("assembly succeeds");
        let arb = endpoints.pool(ChainKey::Arbitrum).expect("arbitrum pool");
        // primary + one custom.
        assert_eq!(arb.endpoints.len(), 2);
    }

    fn graph_config() -> TheGraphConfig {
        TheGraphConfig {
            url: "https://gateway.thegraph.com/api/{key}/subgraphs/id/v4".to_owned(),
            api_key: "graph-key".to_owned(),
        }
    }

    #[test]
    fn assemble_graph_builds_a_pool_only_for_v4_enabled_chains() {
        let endpoints =
            assemble_graph_endpoints(&graph_config(), &BTreeMap::new()).expect("assembly succeeds");

        // Ethereum has a v4 deployment (StateView known); Arbitrum does not, so it gets no pool.
        let eth = endpoints.pool(ChainKey::Ethereum).expect("ethereum pool");
        assert_eq!(eth.endpoints.len(), 1);
        assert!(endpoints.pool(ChainKey::Arbitrum).is_none());
    }

    #[test]
    fn assemble_graph_appends_same_schema_mirrors() {
        let mut custom = BTreeMap::new();
        custom.insert(
            ChainKey::Ethereum,
            vec![EndpointSpec::new(
                "goldsky",
                "https://api.goldsky.com/subgraphs/v4",
                1,
            )],
        );

        let endpoints =
            assemble_graph_endpoints(&graph_config(), &custom).expect("assembly succeeds");
        let eth = endpoints.pool(ChainKey::Ethereum).expect("ethereum pool");
        // gateway primary + one mirror.
        assert_eq!(eth.endpoints.len(), 2);
    }

    #[test]
    fn empty_graph_endpoints_expose_no_pools() {
        let endpoints = GraphEndpoints::empty();

        assert!(endpoints.pool(ChainKey::Ethereum).is_none());
        assert!(endpoints.pool(ChainKey::Arbitrum).is_none());
    }
}
