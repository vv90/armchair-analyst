use std::{
    collections::{HashMap, HashSet},
    fmt,
    marker::PhantomData,
};

use alloy::primitives::BlockHash;

use crate::{pool_registry::PoolCandidateAddress, pool_state::PoolAddress, tick::Tick};

// #[derive(Hash, PartialEq, Eq, Clone, PartialOrd, Ord, Copy)]
// struct RequestId<T>(u64, PhantomData<T>);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct RawRequestId(u64);

pub struct RequestId<R> {
    raw: RawRequestId,
    marker: PhantomData<fn() -> R>,
}

impl<R> Copy for RequestId<R> {}

impl<R> Clone for RequestId<R> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<R> PartialEq for RequestId<R> {
    fn eq(&self, other: &Self) -> bool {
        self.raw == other.raw
    }
}

impl<R> Eq for RequestId<R> {}

impl<R> std::hash::Hash for RequestId<R> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.raw.hash(state);
    }
}

impl<R> fmt::Debug for RequestId<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.raw.0)
    }
}

#[cfg(any(test, feature = "test-support"))]
impl<R> RequestId<R> {
    pub fn from_raw_for_test(value: u64) -> Self {
        RawRequestId(value).typed()
    }

    pub fn raw_for_test(self) -> u64 {
        self.raw.0
    }
}

impl RawRequestId {
    fn next(self) -> RawRequestId {
        RawRequestId(self.0.wrapping_add(1))
    }

    fn typed<R>(self) -> RequestId<R> {
        RequestId {
            raw: self,
            marker: PhantomData,
        }
    }
}

pub struct PendingPayload<R> {
    pub payload: R,
    pub dispatched_at: Tick,
}

pub struct IssuedRequest<R> {
    pub request_id: RequestId<R>,
    pub request_payload: R,
}

pub struct PendingRequestsCollection<R> {
    requests: HashMap<RequestId<R>, PendingPayload<R>>,
}

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

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum AnyRequestId {
    BlockHeader(RequestId<GetBlockHeader>),
    BlockLogs(RequestId<GetBlockLogs>),
    PoolData(RequestId<GetPoolData>),
    PoolMetadata(RequestId<GetPoolMetadata>),
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
        }
    }
}

pub enum AnyIssuedRequest {
    BlockHeader(IssuedRequest<GetBlockHeader>),
    BlockLogs(IssuedRequest<GetBlockLogs>),
    PoolData(IssuedRequest<GetPoolData>),
    PoolMetadata(IssuedRequest<GetPoolMetadata>),
}

pub enum AnyPendingPayload {
    BlockHeader(PendingPayload<GetBlockHeader>),
    BlockLogs(PendingPayload<GetBlockLogs>),
    PoolData(PendingPayload<GetPoolData>),
    PoolMetadata(PendingPayload<GetPoolMetadata>),
}

pub struct PendingRequests {
    last_request_id: RawRequestId,
    block_logs: PendingRequestsCollection<GetBlockLogs>,
    block_headers: PendingRequestsCollection<GetBlockHeader>,
    pool_data: PendingRequestsCollection<GetPoolData>,
    pool_metadata: PendingRequestsCollection<GetPoolMetadata>,
}

pub trait RequestKind: Sized {
    fn get_collection(requests: &PendingRequests) -> &PendingRequestsCollection<Self>;
    fn get_collection_mut(requests: &mut PendingRequests) -> &mut PendingRequestsCollection<Self>;
}

impl PendingRequests {
    pub fn new() -> Self {
        PendingRequests {
            last_request_id: RawRequestId(0),
            block_logs: PendingRequestsCollection {
                requests: HashMap::new(),
            },
            block_headers: PendingRequestsCollection {
                requests: HashMap::new(),
            },
            pool_data: PendingRequestsCollection {
                requests: HashMap::new(),
            },
            pool_metadata: PendingRequestsCollection {
                requests: HashMap::new(),
            },
        }
    }
    pub fn get<R: RequestKind>(&self, request_id: &RequestId<R>) -> Option<&PendingPayload<R>> {
        <R as RequestKind>::get_collection(self)
            .requests
            .get(request_id)
    }

    pub fn take<R: RequestKind>(
        self,
        request_id: &RequestId<R>,
    ) -> (Self, Option<PendingPayload<R>>) {
        let mut self_mut = self;
        let pending_request = <R as RequestKind>::get_collection_mut(&mut self_mut)
            .requests
            .remove(request_id);

        (self_mut, pending_request)
    }

    // pub fn take_expired(self, tick: Tick) -> (Self, HashMap<AnyRequestId, AnyPendingPayload>) {
    //     let mut self_mut = self;

    //     let exp_header_reqs = self_mut
    //         .block_headers
    //         .requests
    //         .extract_if(|_, req| tick.is_expired_since(req.dispatched_at))
    //         .map(|(id, req)| {
    //             (
    //                 AnyRequestId::BlockHeader(id),
    //                 AnyPendingPayload::BlockHeader(req),
    //             )
    //         });

    //     let exp_logs_reqs = self_mut
    //         .block_logs
    //         .requests
    //         .extract_if(|_, req| tick.is_expired_since(req.dispatched_at))
    //         .map(|(id, req)| {
    //             (
    //                 AnyRequestId::BlockLogs(id),
    //                 AnyPendingPayload::BlockLogs(req),
    //             )
    //         });

    //     let exp_pool_reqs = self_mut
    //         .pool_data
    //         .requests
    //         .extract_if(|_, req| tick.is_expired_since(req.dispatched_at))
    //         .map(|(id, req)| (AnyRequestId::PoolData(id), AnyPendingPayload::PoolData(req)));

    //     let expired = exp_header_reqs
    //         .chain(exp_logs_reqs)
    //         .chain(exp_pool_reqs)
    //         .collect();
    //     (self_mut, expired)
    // }

    fn expired_ids(&self, tick: Tick) -> Vec<AnyRequestId> {
        self.block_headers
            .requests
            .iter()
            .filter(|(_, req)| tick.is_expired_since(req.dispatched_at))
            .map(|(id, _)| AnyRequestId::BlockHeader(*id))
            .chain(
                self.block_logs
                    .requests
                    .iter()
                    .filter(|(_, req)| tick.is_expired_since(req.dispatched_at))
                    .map(|(id, _)| AnyRequestId::BlockLogs(*id)),
            )
            .chain(
                self.pool_data
                    .requests
                    .iter()
                    .filter(|(_, req)| tick.is_expired_since(req.dispatched_at))
                    .map(|(id, _)| AnyRequestId::PoolData(*id)),
            )
            .chain(
                self.pool_metadata
                    .requests
                    .iter()
                    .filter(|(_, req)| tick.is_expired_since(req.dispatched_at))
                    .map(|(id, _)| AnyRequestId::PoolMetadata(*id)),
            )
            .collect()
    }

    pub fn with_new_request<R: RequestKind>(self, payload: R, tick: Tick) -> (Self, RequestId<R>) {
        let new_request_id = self.last_request_id.next();
        let mut self_mut = self;

        let requests = &mut <R as RequestKind>::get_collection_mut(&mut self_mut).requests;

        requests.insert(
            new_request_id.typed(),
            PendingPayload {
                payload,
                dispatched_at: tick,
            },
        );

        self_mut.last_request_id = new_request_id;
        (self_mut, new_request_id.typed())
    }

    pub(crate) fn pending_block_log_hashes(&self) -> HashSet<BlockHash> {
        self.block_logs
            .requests
            .values()
            .map(|request| request.payload.block_hash)
            .collect()
    }

    pub(crate) fn pending_pool_metadata_candidates(&self) -> HashSet<PoolCandidateAddress> {
        self.pool_metadata
            .requests
            .values()
            .flat_map(|request| request.payload.candidates.iter().copied())
            .collect()
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
        self.block_headers.requests.len()
            + self.block_logs.requests.len()
            + self.pool_data.requests.len()
            + self.pool_metadata.requests.len()
    }

    pub(crate) fn is_empty_for_test(&self) -> bool {
        self.len_for_test() == 0
    }

    pub(crate) fn last_request_id_for_test(&self) -> u64 {
        self.last_request_id.0
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
        }
    }

    pub(crate) fn pending_header_hashes_for_test(&self) -> HashSet<BlockHash> {
        self.block_headers
            .requests
            .values()
            .map(|request| request.payload.block_hash)
            .collect()
    }

    pub(crate) fn header_dispatch_tick_for_test(
        &self,
        request_id: &RequestId<GetBlockHeader>,
    ) -> Option<Tick> {
        self.block_headers
            .requests
            .get(request_id)
            .map(|request| request.dispatched_at)
    }

    pub(crate) fn dispatch_ticks_for_test(&self) -> Vec<Tick> {
        self.block_headers
            .requests
            .values()
            .map(|request| request.dispatched_at)
            .chain(
                self.block_logs
                    .requests
                    .values()
                    .map(|request| request.dispatched_at),
            )
            .chain(
                self.pool_data
                    .requests
                    .values()
                    .map(|request| request.dispatched_at),
            )
            .chain(
                self.pool_metadata
                    .requests
                    .values()
                    .map(|request| request.dispatched_at),
            )
            .collect()
    }
}

impl RequestKind for GetBlockLogs {
    fn get_collection(requests: &PendingRequests) -> &PendingRequestsCollection<Self> {
        &requests.block_logs
    }

    fn get_collection_mut(requests: &mut PendingRequests) -> &mut PendingRequestsCollection<Self> {
        &mut requests.block_logs
    }
}

impl RequestKind for GetBlockHeader {
    fn get_collection(requests: &PendingRequests) -> &PendingRequestsCollection<Self> {
        &requests.block_headers
    }

    fn get_collection_mut(requests: &mut PendingRequests) -> &mut PendingRequestsCollection<Self> {
        &mut requests.block_headers
    }
}

impl RequestKind for GetPoolData {
    fn get_collection(requests: &PendingRequests) -> &PendingRequestsCollection<Self> {
        &requests.pool_data
    }

    fn get_collection_mut(requests: &mut PendingRequests) -> &mut PendingRequestsCollection<Self> {
        &mut requests.pool_data
    }
}

impl RequestKind for GetPoolMetadata {
    fn get_collection(requests: &PendingRequests) -> &PendingRequestsCollection<Self> {
        &requests.pool_metadata
    }

    fn get_collection_mut(requests: &mut PendingRequests) -> &mut PendingRequestsCollection<Self> {
        &mut requests.pool_metadata
    }
}
