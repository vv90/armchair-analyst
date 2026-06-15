use std::{
    collections::{HashMap, HashSet},
    fmt,
};

use alloy::primitives::BlockHash;

use super::{pool_registry::PoolCandidateAddress, token_registry::TokenAddress};
pub use crate::request_tracking::{IssuedRequest, PendingPayload, RequestId};
use crate::{
    pool_state::PoolAddress,
    request_tracking::{RequestCollection, RequestIdSequence},
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
    pub pools: HashSet<PoolAddress>,
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

pub enum AnyPendingPayload {
    BlockHeader(PendingPayload<GetBlockHeader>),
    BlockLogs(PendingPayload<GetBlockLogs>),
    PoolData(PendingPayload<GetPoolData>),
    PoolMetadata(PendingPayload<GetPoolMetadata>),
    TokenMetadata(PendingPayload<GetTokenMetadata>),
}

pub struct PendingRequests {
    request_ids: RequestIdSequence,
    block_logs: RequestCollection<GetBlockLogs>,
    block_headers: RequestCollection<GetBlockHeader>,
    pool_data: RequestCollection<GetPoolData>,
    pool_metadata: RequestCollection<GetPoolMetadata>,
    token_metadata: RequestCollection<GetTokenMetadata>,
}

pub trait RequestKind: Sized {
    fn get_collection(requests: &PendingRequests) -> &RequestCollection<Self>;
    fn get_collection_mut(requests: &mut PendingRequests) -> &mut RequestCollection<Self>;
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
    pub fn get<R: RequestKind>(&self, request_id: &RequestId<R>) -> Option<&PendingPayload<R>> {
        <R as RequestKind>::get_collection(self).get(request_id)
    }

    pub fn take<R: RequestKind>(
        self,
        request_id: &RequestId<R>,
    ) -> (Self, Option<PendingPayload<R>>) {
        let mut self_mut = self;
        let pending_request =
            <R as RequestKind>::get_collection_mut(&mut self_mut).remove(request_id);

        (self_mut, pending_request)
    }

    fn expired_ids(&self, tick: Tick) -> Vec<AnyRequestId> {
        self.block_headers
            .iter()
            .filter(|(_, req)| tick.is_expired_since(req.dispatched_at))
            .map(|(id, _)| AnyRequestId::BlockHeader(*id))
            .chain(
                self.block_logs
                    .iter()
                    .filter(|(_, req)| tick.is_expired_since(req.dispatched_at))
                    .map(|(id, _)| AnyRequestId::BlockLogs(*id)),
            )
            .chain(
                self.pool_data
                    .iter()
                    .filter(|(_, req)| tick.is_expired_since(req.dispatched_at))
                    .map(|(id, _)| AnyRequestId::PoolData(*id)),
            )
            .chain(
                self.pool_metadata
                    .iter()
                    .filter(|(_, req)| tick.is_expired_since(req.dispatched_at))
                    .map(|(id, _)| AnyRequestId::PoolMetadata(*id)),
            )
            .chain(
                self.token_metadata
                    .iter()
                    .filter(|(_, req)| tick.is_expired_since(req.dispatched_at))
                    .map(|(id, _)| AnyRequestId::TokenMetadata(*id)),
            )
            .collect()
    }

    pub fn with_new_request<R: RequestKind>(self, payload: R, tick: Tick) -> (Self, RequestId<R>) {
        let mut self_mut = self;
        let new_request_id = self_mut.request_ids.issue();

        <R as RequestKind>::get_collection_mut(&mut self_mut).insert(new_request_id, payload, tick);

        (self_mut, new_request_id)
    }

    pub(crate) fn pending_block_log_hashes(&self) -> HashSet<BlockHash> {
        self.block_logs
            .values()
            .map(|request| request.payload.block_hash)
            .collect()
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
    ) -> HashMap<BlockHash, HashSet<PoolAddress>> {
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

    fn retry_typed<R: RequestKind + Clone>(
        self,
        request_id: RequestId<R>,
        tick: Tick,
        wrap: fn(IssuedRequest<R>) -> AnyIssuedRequest,
    ) -> (Self, Option<AnyIssuedRequest>) {
        let (pending_requests, pending_payload) = self.take(&request_id);

        match pending_payload {
            Some(PendingPayload { payload, .. }) => {
                let (pending_requests, new_request_id) =
                    pending_requests.with_new_request(payload.clone(), tick);

                (
                    pending_requests,
                    Some(wrap(IssuedRequest {
                        request_id: new_request_id,
                        request_payload: payload,
                    })),
                )
            }
            None => (pending_requests, None),
        }
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

    pub(crate) fn contains<R: RequestKind>(&self, request_id: &RequestId<R>) -> bool {
        self.get(request_id).is_some()
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

impl RequestKind for GetBlockLogs {
    fn get_collection(requests: &PendingRequests) -> &RequestCollection<Self> {
        &requests.block_logs
    }

    fn get_collection_mut(requests: &mut PendingRequests) -> &mut RequestCollection<Self> {
        &mut requests.block_logs
    }
}

impl RequestKind for GetBlockHeader {
    fn get_collection(requests: &PendingRequests) -> &RequestCollection<Self> {
        &requests.block_headers
    }

    fn get_collection_mut(requests: &mut PendingRequests) -> &mut RequestCollection<Self> {
        &mut requests.block_headers
    }
}

impl RequestKind for GetPoolData {
    fn get_collection(requests: &PendingRequests) -> &RequestCollection<Self> {
        &requests.pool_data
    }

    fn get_collection_mut(requests: &mut PendingRequests) -> &mut RequestCollection<Self> {
        &mut requests.pool_data
    }
}

impl RequestKind for GetPoolMetadata {
    fn get_collection(requests: &PendingRequests) -> &RequestCollection<Self> {
        &requests.pool_metadata
    }

    fn get_collection_mut(requests: &mut PendingRequests) -> &mut RequestCollection<Self> {
        &mut requests.pool_metadata
    }
}

impl RequestKind for GetTokenMetadata {
    fn get_collection(requests: &PendingRequests) -> &RequestCollection<Self> {
        &requests.token_metadata
    }

    fn get_collection_mut(requests: &mut PendingRequests) -> &mut RequestCollection<Self> {
        &mut requests.token_metadata
    }
}
