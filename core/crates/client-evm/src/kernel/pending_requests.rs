use std::{
    collections::{HashMap, HashSet},
    fmt,
};

use alloy::primitives::BlockHash;

use super::{pool_registry::PoolCandidateAddress, token_registry::TokenAddress};
pub use crate::request_tracking::{IssuedRequest, PendingPayload, RequestId};
use crate::{
    pool_state::PoolRef,
    request_tracking::{
        RequestCollection, RequestIdSequence, RequestIssuer, RequestStore, expired_request_ids,
        issue_request, retry_request, take_request,
    },
    tick::Tick,
};

#[derive(Clone)]
pub struct GetBlockLogs {
    pub block_hash: BlockHash,
}

#[derive(Clone)]
pub struct GetBlockHeader {
    pub block_hash: BlockHash,
}

#[derive(Clone)]
pub struct GetPoolData {
    pub at: BlockHash,
    pub pools: HashSet<PoolRef>,
}

#[derive(Clone)]
pub struct GetPoolMetadata {
    pub at: BlockHash,
    pub candidates: HashSet<PoolCandidateAddress>,
}

#[derive(Clone)]
pub struct GetTokenMetadata {
    pub at: BlockHash,
    pub tokens: HashSet<TokenAddress>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum AnyRequestId {
    BlockHeader(RequestId<GetBlockHeader>),
    BlockLogs(RequestId<GetBlockLogs>),
    PoolData(RequestId<GetPoolData>),
    PoolMetadata(RequestId<GetPoolMetadata>),
    TokenMetadata(RequestId<GetTokenMetadata>),
}

impl fmt::Debug for AnyRequestId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AnyRequestId::BlockHeader(request_id) => {
                write!(formatter, "block_header#{request_id:?}")
            }
            AnyRequestId::BlockLogs(request_id) => write!(formatter, "block_logs#{request_id:?}"),
            AnyRequestId::PoolData(request_id) => write!(formatter, "pool_data#{request_id:?}"),
            AnyRequestId::PoolMetadata(request_id) => {
                write!(formatter, "pool_metadata#{request_id:?}")
            }
            AnyRequestId::TokenMetadata(request_id) => {
                write!(formatter, "token_metadata#{request_id:?}")
            }
        }
    }
}

pub enum AnyIssuedRequest {
    BlockHeader(IssuedRequest<GetBlockHeader>),
    BlockLogs(IssuedRequest<GetBlockLogs>),
    PoolData(IssuedRequest<GetPoolData>),
    PoolMetadata(IssuedRequest<GetPoolMetadata>),
    TokenMetadata(IssuedRequest<GetTokenMetadata>),
}

// pub enum AnyPendingPayload {
//     BlockHeader(PendingPayload<GetBlockHeader>),
//     BlockLogs(PendingPayload<GetBlockLogs>),
//     PoolData(PendingPayload<GetPoolData>),
//     PoolMetadata(PendingPayload<GetPoolMetadata>),
//     TokenMetadata(PendingPayload<GetTokenMetadata>),
// }

pub struct PendingRequests {
    request_ids: RequestIdSequence,
    block_logs: RequestCollection<GetBlockLogs>,
    block_headers: RequestCollection<GetBlockHeader>,
    pool_data: RequestCollection<GetPoolData>,
    pool_metadata: RequestCollection<GetPoolMetadata>,
    token_metadata: RequestCollection<GetTokenMetadata>,
}

impl PendingRequests {
    pub fn new() -> Self {
        PendingRequests {
            request_ids: RequestIdSequence::new(),
            block_logs: RequestCollection::new(),
            block_headers: RequestCollection::new(),
            pool_data: RequestCollection::new(),
            pool_metadata: RequestCollection::new(),
            token_metadata: RequestCollection::new(),
        }
    }
    pub fn take<R>(self, request_id: &RequestId<R>) -> (Self, Option<PendingPayload<R>>)
    where
        Self: RequestStore<R>,
    {
        take_request(self, request_id)
    }

    fn expired_ids(&self, tick: Tick) -> Vec<AnyRequestId> {
        expired_request_ids::<Self, GetBlockHeader>(self, tick)
            .into_iter()
            .map(AnyRequestId::BlockHeader)
            .chain(
                expired_request_ids::<Self, GetBlockLogs>(self, tick)
                    .into_iter()
                    .map(AnyRequestId::BlockLogs),
            )
            .chain(
                expired_request_ids::<Self, GetPoolData>(self, tick)
                    .into_iter()
                    .map(AnyRequestId::PoolData),
            )
            .chain(
                expired_request_ids::<Self, GetPoolMetadata>(self, tick)
                    .into_iter()
                    .map(AnyRequestId::PoolMetadata),
            )
            .chain(
                expired_request_ids::<Self, GetTokenMetadata>(self, tick)
                    .into_iter()
                    .map(AnyRequestId::TokenMetadata),
            )
            .collect()
    }

    pub fn with_new_request<R>(self, payload: R, tick: Tick) -> (Self, RequestId<R>)
    where
        Self: RequestStore<R>,
    {
        issue_request(self, payload, tick)
    }

    /// Reports whether no request is in flight across any collection. Used to gate the background
    /// pool-state backfill behind an idle priority tier.
    pub(crate) fn is_empty(&self) -> bool {
        self.block_headers.values().next().is_none()
            && self.block_logs.values().next().is_none()
            && self.pool_data.values().next().is_none()
            && self.pool_metadata.values().next().is_none()
            && self.token_metadata.values().next().is_none()
    }

    pub(crate) fn pending_block_log_hashes(&self) -> HashSet<BlockHash> {
        self.block_logs
            .values()
            .map(|request| request.payload.block_hash)
            .collect()
    }

    /// Reports whether a block-header request for `block_hash` is already in flight.
    /// Added so ancestry-reconnection sites skip re-issuing a header already being fetched, since the
    /// in-flight request will deliver it and continue the walk (the TTL retry path still covers a loss).
    pub(crate) fn has_pending_header_request(&self, block_hash: BlockHash) -> bool {
        self.block_headers
            .values()
            .any(|request| request.payload.block_hash == block_hash)
    }

    pub(crate) fn pending_pool_metadata_candidates(&self) -> HashSet<PoolCandidateAddress> {
        self.pool_metadata
            .values()
            .flat_map(|request| request.payload.candidates.iter().copied())
            .collect()
    }

    pub(crate) fn pending_token_metadata_tokens(&self) -> HashSet<TokenAddress> {
        self.token_metadata
            .values()
            .flat_map(|request| request.payload.tokens.iter().copied())
            .collect()
    }

    pub(crate) fn pending_pool_data_pools_by_block(
        &self,
    ) -> HashMap<BlockHash, HashSet<PoolRef>> {
        self.pool_data
            .values()
            .fold(HashMap::new(), |mut pools_by_block, request| {
                pools_by_block
                    .entry(request.payload.at)
                    .or_default()
                    .extend(request.payload.pools.iter().copied());

                pools_by_block
            })
    }

    pub(crate) fn retaining_block_targets(self, retained_blocks: &HashSet<BlockHash>) -> Self {
        let mut requests = self;

        requests
            .block_headers
            .retain(|_, request| retained_blocks.contains(&request.payload.block_hash));
        requests
            .block_logs
            .retain(|_, request| retained_blocks.contains(&request.payload.block_hash));
        requests
            .pool_data
            .retain(|_, request| retained_blocks.contains(&request.payload.at));
        requests
            .pool_metadata
            .retain(|_, request| retained_blocks.contains(&request.payload.at));
        requests
            .token_metadata
            .retain(|_, request| retained_blocks.contains(&request.payload.at));

        requests
    }

    pub fn retry_expired(self, tick: Tick) -> (Self, Vec<AnyIssuedRequest>) {
        let expired_ids = self.expired_ids(tick);

        expired_ids
            .into_iter()
            .fold((self, Vec::new()), |(requests, mut effects), request_id| {
                let (requests, issued) = requests.retry(request_id, tick);
                effects.extend(issued);
                (requests, effects)
            })
    }

    pub fn retry(self, request_id: AnyRequestId, tick: Tick) -> (Self, Option<AnyIssuedRequest>) {
        match request_id {
            AnyRequestId::BlockHeader(request_id) => {
                self.retry_typed(request_id, tick, AnyIssuedRequest::BlockHeader)
            }
            AnyRequestId::BlockLogs(request_id) => {
                self.retry_typed(request_id, tick, AnyIssuedRequest::BlockLogs)
            }
            AnyRequestId::PoolData(request_id) => {
                self.retry_typed(request_id, tick, AnyIssuedRequest::PoolData)
            }
            AnyRequestId::PoolMetadata(request_id) => {
                self.retry_typed(request_id, tick, AnyIssuedRequest::PoolMetadata)
            }
            AnyRequestId::TokenMetadata(request_id) => {
                self.retry_typed(request_id, tick, AnyIssuedRequest::TokenMetadata)
            }
        }
    }

    fn retry_typed<R: Clone>(
        self,
        request_id: RequestId<R>,
        tick: Tick,
        wrap: fn(IssuedRequest<R>) -> AnyIssuedRequest,
    ) -> (Self, Option<AnyIssuedRequest>)
    where
        Self: RequestStore<R>,
    {
        let (pending_requests, issued_request) = retry_request(self, request_id, tick);

        (pending_requests, issued_request.map(wrap))
    }
}

#[cfg(test)]
impl PendingRequests {
    pub(crate) fn len_for_test(&self) -> usize {
        self.block_headers.len()
            + self.block_logs.len()
            + self.pool_data.len()
            + self.pool_metadata.len()
            + self.token_metadata.len()
    }

    pub(crate) fn is_empty_for_test(&self) -> bool {
        self.len_for_test() == 0
    }

    pub(crate) fn last_request_id_for_test(&self) -> u64 {
        self.request_ids.raw_for_test()
    }

    pub(crate) fn contains<R>(&self, request_id: &RequestId<R>) -> bool
    where
        Self: RequestStore<R>,
    {
        self.get(request_id).is_some()
    }

    pub(crate) fn get<R>(&self, request_id: &RequestId<R>) -> Option<&PendingPayload<R>>
    where
        Self: RequestStore<R>,
    {
        crate::request_tracking::test_support::get_request(self, request_id)
    }

    pub(crate) fn contains_any_for_test(&self, request_id: AnyRequestId) -> bool {
        match request_id {
            AnyRequestId::BlockHeader(request_id) => self.contains(&request_id),
            AnyRequestId::BlockLogs(request_id) => self.contains(&request_id),
            AnyRequestId::PoolData(request_id) => self.contains(&request_id),
            AnyRequestId::PoolMetadata(request_id) => self.contains(&request_id),
            AnyRequestId::TokenMetadata(request_id) => self.contains(&request_id),
        }
    }

    pub(crate) fn pending_header_hashes_for_test(&self) -> HashSet<BlockHash> {
        self.block_headers
            .values()
            .map(|request| request.payload.block_hash)
            .collect()
    }

    /// Counts in-flight header requests including any duplicates for the same block hash.
    /// Compared against the distinct-hash count to assert no duplicate header request is ever live.
    pub(crate) fn pending_header_request_count_for_test(&self) -> usize {
        self.block_headers.len()
    }

    pub(crate) fn header_dispatch_tick_for_test(
        &self,
        request_id: &RequestId<GetBlockHeader>,
    ) -> Option<Tick> {
        self.block_headers
            .get(request_id)
            .map(|request| request.dispatched_at)
    }

    pub(crate) fn dispatch_ticks_for_test(&self) -> Vec<Tick> {
        self.block_headers
            .values()
            .map(|request| request.dispatched_at)
            .chain(
                self.block_logs
                    .values()
                    .map(|request| request.dispatched_at),
            )
            .chain(self.pool_data.values().map(|request| request.dispatched_at))
            .chain(
                self.pool_metadata
                    .values()
                    .map(|request| request.dispatched_at),
            )
            .chain(
                self.token_metadata
                    .values()
                    .map(|request| request.dispatched_at),
            )
            .collect()
    }
}

impl RequestIssuer for PendingRequests {
    fn request_ids_mut(&mut self) -> &mut RequestIdSequence {
        &mut self.request_ids
    }
}

impl RequestStore<GetBlockLogs> for PendingRequests {
    fn request_collection(&self) -> &RequestCollection<GetBlockLogs> {
        &self.block_logs
    }

    fn request_collection_mut(&mut self) -> &mut RequestCollection<GetBlockLogs> {
        &mut self.block_logs
    }
}

impl RequestStore<GetBlockHeader> for PendingRequests {
    fn request_collection(&self) -> &RequestCollection<GetBlockHeader> {
        &self.block_headers
    }

    fn request_collection_mut(&mut self) -> &mut RequestCollection<GetBlockHeader> {
        &mut self.block_headers
    }
}

impl RequestStore<GetPoolData> for PendingRequests {
    fn request_collection(&self) -> &RequestCollection<GetPoolData> {
        &self.pool_data
    }

    fn request_collection_mut(&mut self) -> &mut RequestCollection<GetPoolData> {
        &mut self.pool_data
    }
}

impl RequestStore<GetPoolMetadata> for PendingRequests {
    fn request_collection(&self) -> &RequestCollection<GetPoolMetadata> {
        &self.pool_metadata
    }

    fn request_collection_mut(&mut self) -> &mut RequestCollection<GetPoolMetadata> {
        &mut self.pool_metadata
    }
}

impl RequestStore<GetTokenMetadata> for PendingRequests {
    fn request_collection(&self) -> &RequestCollection<GetTokenMetadata> {
        &self.token_metadata
    }

    fn request_collection_mut(&mut self) -> &mut RequestCollection<GetTokenMetadata> {
        &mut self.token_metadata
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn has_pending_header_request_tracks_issued_and_taken_headers() {
        let hash = BlockHash::with_last_byte(7);
        let other = BlockHash::with_last_byte(8);

        let (pending, request_id) = PendingRequests::new()
            .with_new_request(GetBlockHeader { block_hash: hash }, Tick::initial());

        assert!(pending.has_pending_header_request(hash));
        assert!(!pending.has_pending_header_request(other));

        let (pending, _) = pending.take(&request_id);
        assert!(!pending.has_pending_header_request(hash));
    }
}
