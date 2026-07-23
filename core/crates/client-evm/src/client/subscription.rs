//! Pure logic for the multi-provider WebSocket adapter.
//!
//! Two decisions live here as pure, property-tested functions, so the impure IO shell
//! (sockets/threads/timer in [`client_effects`](super::client_effects) and the runtime wiring)
//! stays minimal and delegates immediately:
//!
//! - [`plan_ws_subscriptions`]: config → the flat list of `(chain, provider)` connections to open.
//! - [`LogBatchBuffer`]: a clockless fixed-interval batcher that consolidates and dedups a burst of
//!   streamed logs into one `log_index`-ordered batch per block. The window lives in the caller
//!   (the impure loop drives [`LogBatchBuffer::flush`] on a timer), so the buffer is deterministic.

use std::collections::{BTreeMap, HashMap};

use alloy::primitives::BlockHash;

use crate::{ACTIVE_CHAINS, ChainKey, PoolLog, endpoints::ChainSubscriptions};

/// One WebSocket connection to open: a single provider's stream for a single chain. The impure
/// wiring spawns one reconnect loop per descriptor, so every configured provider is a live feed and
/// a single dead one no longer blinds the chain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WsSubscriptionEndpoint {
    pub chain: ChainKey,
    pub label: String,
    pub url: String,
}

/// The full fan-out plan: one descriptor per (active chain × configured WS endpoint), in chain then
/// configuration order. Pure — the runtime iterates this instead of computing the fan-out itself.
/// [`ChainSubscriptions::new`] already guarantees every active chain has at least one endpoint, so
/// the per-chain lookup never fails here.
pub fn plan_ws_subscriptions(subscriptions: &ChainSubscriptions) -> Vec<WsSubscriptionEndpoint> {
    ACTIVE_CHAINS
        .iter()
        .flat_map(|&chain| {
            subscriptions
                .ws_endpoints(chain)
                .into_iter()
                .flatten()
                .map(move |spec| WsSubscriptionEndpoint {
                    chain,
                    label: spec.label.clone(),
                    url: spec.url.clone(),
                })
        })
        .collect()
}

/// A clockless accumulator for streamed pool logs, consolidating a fixed-interval burst.
///
/// [`observe`](Self::observe) keys each log by its intra-block `log_index` (the same identity the
/// kernel's ingestion boundary uses), so a log delivered by two providers within the window
/// collapses to one entry. [`flush`](Self::flush) drains the buffer into one `log_index`-ordered
/// batch per block. Holding no clock keeps the type deterministic and fully property-testable — the
/// impure loop owns the interval and calls `flush` when it elapses.
#[derive(Default)]
pub(crate) struct LogBatchBuffer {
    blocks: HashMap<BlockHash, BTreeMap<u64, PoolLog>>,
}

impl LogBatchBuffer {
    pub(crate) fn new() -> LogBatchBuffer {
        LogBatchBuffer::default()
    }

    /// Records one streamed log, deduping by `(block_hash, log_index)` — a repeat from another
    /// provider overwrites with the identical log, so the set is unchanged.
    pub(crate) fn observe(&mut self, block_hash: BlockHash, log: PoolLog) {
        self.blocks
            .entry(block_hash)
            .or_default()
            .insert(log.log_index, log);
    }

    /// Whether nothing has been observed since the last flush (the impure loop skips emitting then).
    pub(crate) fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Drains every accumulated block into one `log_index`-ordered batch and empties the buffer.
    pub(crate) fn flush(&mut self) -> Vec<(BlockHash, Vec<PoolLog>)> {
        std::mem::take(&mut self.blocks)
            .into_iter()
            .map(|(block_hash, logs)| (block_hash, logs.into_values().collect()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use alloy::primitives::{Address, U160, aliases::I24};
    use proptest::prelude::*;

    use super::*;
    use crate::{EndpointSpec, PoolLogEvent, ProtocolPoolKey};

    fn pool_key(seed: u8) -> ProtocolPoolKey {
        ProtocolPoolKey::UniswapV3(Address::with_last_byte(seed))
    }

    /// A distinct log identified by its `(block, log_index)`; `seed` varies the payload so a genuine
    /// duplicate (same identity, same payload) is distinguishable from a collision in tests.
    fn pool_log(seed: u8, log_index: u64) -> PoolLog {
        PoolLog {
            pool: pool_key(seed),
            log_index,
            event: PoolLogEvent::Swap {
                sqrt_price_x96: U160::from(seed),
                tick: I24::try_from(0).unwrap(),
                liquidity: u128::from(log_index),
            },
        }
    }

    #[test]
    fn plan_fans_out_one_descriptor_per_chain_and_provider() {
        let mut ws = BTreeMap::new();
        for &chain in ACTIVE_CHAINS {
            ws.insert(
                chain,
                vec![
                    EndpointSpec::new("drpc", "wss://a/key", 3),
                    EndpointSpec::new("publicnode", "wss://b", 1),
                ],
            );
        }
        let subscriptions = ChainSubscriptions::new(ws).expect("complete set");

        let plan = plan_ws_subscriptions(&subscriptions);

        assert_eq!(plan.len(), ACTIVE_CHAINS.len() * 2);
        let ethereum: Vec<_> = plan
            .iter()
            .filter(|endpoint| endpoint.chain == ChainKey::Ethereum)
            .map(|endpoint| (endpoint.label.as_str(), endpoint.url.as_str()))
            .collect();
        assert_eq!(
            ethereum,
            vec![("drpc", "wss://a/key"), ("publicnode", "wss://b")]
        );
    }

    #[test]
    fn flush_after_flush_is_empty() {
        let mut buffer = LogBatchBuffer::new();
        buffer.observe(BlockHash::with_last_byte(1), pool_log(1, 0));

        assert!(!buffer.is_empty());
        assert_eq!(buffer.flush().len(), 1);
        assert!(buffer.is_empty());
        assert!(buffer.flush().is_empty());
    }

    /// The flat map of `(block_hash, log_index) -> PoolLog` a batch set represents, for order-free
    /// comparison across insertion orders.
    fn batched_index(
        batches: Vec<(BlockHash, Vec<PoolLog>)>,
    ) -> BTreeMap<(BlockHash, u64), PoolLog> {
        batches
            .into_iter()
            .flat_map(|(hash, logs)| {
                logs.into_iter()
                    .map(move |log| ((hash, log.log_index), log))
            })
            .collect()
    }

    proptest! {
        /// Distinct logs, each keyed by a unique `(block, log_index)`, so a duplicate is an exact
        /// clone. Bytes keep the block/pool space small enough to force collisions and repeats.
        #[test]
        fn flush_is_permutation_invariant_and_deduped(
            raw in proptest::collection::vec((0u8..4, 0u8..3, 0u64..5), 0..40),
        ) {
            // Canonicalize inputs to distinct logs keyed by (block, log_index): a repeated key is a
            // genuine duplicate (identical clone), mirroring two providers delivering the same log.
            let mut canonical: BTreeMap<(u8, u64), (u8, PoolLog)> = BTreeMap::new();
            for (block_seed, pool_seed, log_index) in &raw {
                canonical
                    .entry((*block_seed, *log_index))
                    .or_insert_with(|| (*pool_seed, pool_log(*pool_seed, *log_index)));
            }

            let ordered: Vec<(BlockHash, PoolLog)> = raw
                .iter()
                .map(|(block_seed, _, log_index)| {
                    let (_, log) = &canonical[&(*block_seed, *log_index)];
                    (BlockHash::with_last_byte(*block_seed), log.clone())
                })
                .collect();

            let mut forward = LogBatchBuffer::new();
            for (hash, log) in &ordered {
                forward.observe(*hash, log.clone());
            }
            let mut reverse = LogBatchBuffer::new();
            for (hash, log) in ordered.iter().rev() {
                reverse.observe(*hash, log.clone());
            }

            let forward_batches = forward.flush();
            let reverse_batches = reverse.flush();

            // (i) permutation invariance and (ii)/(iii) dedup: both orders collapse to the same set,
            // exactly the distinct observed keys, each appearing once.
            prop_assert_eq!(
                batched_index(forward_batches.clone()),
                batched_index(reverse_batches)
            );
            let observed_keys: BTreeMap<(BlockHash, u64), PoolLog> = canonical
                .into_iter()
                .map(|((block_seed, log_index), (_, log))| {
                    ((BlockHash::with_last_byte(block_seed), log_index), log)
                })
                .collect();
            prop_assert_eq!(batched_index(forward_batches.clone()), observed_keys);

            // (iv) within each block, logs are log_index-ordered.
            for (_, logs) in &forward_batches {
                let indices: Vec<u64> = logs.iter().map(|log| log.log_index).collect();
                let mut sorted = indices.clone();
                sorted.sort_unstable();
                prop_assert_eq!(indices, sorted);
            }
        }
    }
}
