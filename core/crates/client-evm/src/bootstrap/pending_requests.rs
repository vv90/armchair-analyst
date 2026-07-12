use std::fmt;

use super::{GetPoolMetadata, GetTokenMetadata};
pub use crate::request_tracking::{IssuedRequest, PendingPayload, RequestId};
use crate::{
    request_tracking::{
        RequestCollection, RequestIdSequence, RequestIssuer, RequestStore, expired_request_ids,
        issue_request, retry_request, take_request,
    },
    tick::Tick,
};

/// Request for the chain's `finalized`-tagged header that anchors the bootstrap.
#[derive(Clone)]
pub struct GetFinalizedHeader;

/// Request for pool-event candidates over one window of the canonical look-back range. The scan is
/// paged one window per request by the bootstrap state machine; `scan_tip` freezes the ceiling:
/// `None` on the first window (the executor resolves and reports the tip), `Some(tip)` on every
/// continuation so the paged scan targets a fixed `finalized..tip` range instead of chasing a
/// moving tip.
#[derive(Clone)]
pub struct GetPoolCandidatesInRange {
    pub from_block: u64,
    pub scan_tip: Option<u64>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum AnyRequestId {
    FinalizedHeader(RequestId<GetFinalizedHeader>),
    PoolCandidates(RequestId<GetPoolCandidatesInRange>),
    PoolMetadata(RequestId<GetPoolMetadata>),
    TokenMetadata(RequestId<GetTokenMetadata>),
}

impl fmt::Debug for AnyRequestId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AnyRequestId::FinalizedHeader(request_id) => {
                write!(formatter, "finalized_header#{request_id:?}")
            }
            AnyRequestId::PoolCandidates(request_id) => {
                write!(formatter, "pool_candidates#{request_id:?}")
            }
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
    FinalizedHeader(IssuedRequest<GetFinalizedHeader>),
    PoolCandidates(IssuedRequest<GetPoolCandidatesInRange>),
    PoolMetadata(IssuedRequest<GetPoolMetadata>),
    TokenMetadata(IssuedRequest<GetTokenMetadata>),
}

/// Single-host request ledger for the bootstrap phase, mirroring `kernel::pending_requests`.
/// Bootstrap is sequential, so at most one collection holds an entry, but the uniform machinery
/// reuses `request_tracking` for typed-id correlation and Tick-driven retry.
pub struct PendingRequests {
    request_ids: RequestIdSequence,
    finalized_header: RequestCollection<GetFinalizedHeader>,
    pool_candidates: RequestCollection<GetPoolCandidatesInRange>,
    pool_metadata: RequestCollection<GetPoolMetadata>,
    token_metadata: RequestCollection<GetTokenMetadata>,
}

impl PendingRequests {
    pub fn new() -> Self {
        PendingRequests {
            request_ids: RequestIdSequence::new(),
            finalized_header: RequestCollection::new(),
            pool_candidates: RequestCollection::new(),
            pool_metadata: RequestCollection::new(),
            token_metadata: RequestCollection::new(),
        }
    }

    pub fn with_new_request<R>(self, payload: R, tick: Tick) -> (Self, RequestId<R>)
    where
        Self: RequestStore<R>,
    {
        issue_request(self, payload, tick)
    }

    pub fn take<R>(self, request_id: &RequestId<R>) -> (Self, Option<PendingPayload<R>>)
    where
        Self: RequestStore<R>,
    {
        take_request(self, request_id)
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

    fn expired_ids(&self, tick: Tick) -> Vec<AnyRequestId> {
        expired_request_ids::<Self, GetFinalizedHeader>(self, tick)
            .into_iter()
            .map(AnyRequestId::FinalizedHeader)
            .chain(
                expired_request_ids::<Self, GetPoolCandidatesInRange>(self, tick)
                    .into_iter()
                    .map(AnyRequestId::PoolCandidates),
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

    pub(crate) fn retry(
        self,
        request_id: AnyRequestId,
        tick: Tick,
    ) -> (Self, Option<AnyIssuedRequest>) {
        match request_id {
            AnyRequestId::FinalizedHeader(request_id) => {
                self.retry_typed(request_id, tick, AnyIssuedRequest::FinalizedHeader)
            }
            AnyRequestId::PoolCandidates(request_id) => {
                self.retry_typed(request_id, tick, AnyIssuedRequest::PoolCandidates)
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
    /// Total number of requests currently in flight across every collection.
    pub(crate) fn len_for_test(&self) -> usize {
        self.finalized_header.len()
            + self.pool_candidates.len()
            + self.pool_metadata.len()
            + self.token_metadata.len()
    }
}

impl RequestIssuer for PendingRequests {
    fn request_ids_mut(&mut self) -> &mut RequestIdSequence {
        &mut self.request_ids
    }
}

impl RequestStore<GetFinalizedHeader> for PendingRequests {
    fn request_collection(&self) -> &RequestCollection<GetFinalizedHeader> {
        &self.finalized_header
    }

    fn request_collection_mut(&mut self) -> &mut RequestCollection<GetFinalizedHeader> {
        &mut self.finalized_header
    }
}

impl RequestStore<GetPoolCandidatesInRange> for PendingRequests {
    fn request_collection(&self) -> &RequestCollection<GetPoolCandidatesInRange> {
        &self.pool_candidates
    }

    fn request_collection_mut(&mut self) -> &mut RequestCollection<GetPoolCandidatesInRange> {
        &mut self.pool_candidates
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
    use proptest::prelude::*;

    use super::*;
    use crate::tick::{REQUEST_TTL_FOR_TEST, Tick};

    fn tick(value: u64) -> Tick {
        Tick::from_raw_for_test(value)
    }

    #[test]
    fn issued_request_is_retrievable_then_consumed() {
        let pending = PendingRequests::new();
        let (pending, request_id) =
            pending.with_new_request(GetPoolCandidatesInRange { from_block: 7, scan_tip: None }, tick(1));

        let (pending, taken) = pending.take(&request_id);
        assert!(matches!(
            taken,
            Some(PendingPayload {
                payload: GetPoolCandidatesInRange { from_block: 7, scan_tip: None },
                ..
            })
        ));

        let (_pending, taken_again) = pending.take(&request_id);
        assert!(taken_again.is_none());
    }

    #[test]
    fn distinct_requests_get_distinct_ids() {
        let pending = PendingRequests::new();
        let (pending, finalized_id) = pending.with_new_request(GetFinalizedHeader, tick(1));
        let (_pending, candidates_id) =
            pending.with_new_request(GetPoolCandidatesInRange { from_block: 1, scan_tip: None }, tick(1));

        assert_ne!(finalized_id.raw_for_test(), candidates_id.raw_for_test());
    }

    #[test]
    fn retry_expired_before_ttl_keeps_request_pending() {
        let pending = PendingRequests::new();
        let (pending, request_id) =
            pending.with_new_request(GetPoolCandidatesInRange { from_block: 5, scan_tip: None }, tick(0));

        let (pending, reissued) = pending.retry_expired(tick(REQUEST_TTL_FOR_TEST - 1));
        assert!(reissued.is_empty());

        let (_pending, taken) = pending.take(&request_id);
        assert!(matches!(
            taken,
            Some(PendingPayload {
                payload: GetPoolCandidatesInRange { from_block: 5, scan_tip: None },
                ..
            })
        ));
    }

    #[test]
    fn retry_expired_after_ttl_reissues_same_payload_with_new_id() {
        let pending = PendingRequests::new();
        let (pending, request_id) =
            pending.with_new_request(GetPoolCandidatesInRange { from_block: 42, scan_tip: None }, tick(0));

        let (pending, reissued) = pending.retry_expired(tick(REQUEST_TTL_FOR_TEST));
        assert!(matches!(
            reissued.as_slice(),
            [AnyIssuedRequest::PoolCandidates(IssuedRequest {
                request_payload: GetPoolCandidatesInRange { from_block: 42, scan_tip: None },
                request_id: reissued_id,
            })] if reissued_id.raw_for_test() != request_id.raw_for_test()
        ));

        let (_pending, taken) = pending.take(&request_id);
        assert!(taken.is_none());
    }

    proptest! {
        #[test]
        fn retry_expired_reissues_exactly_when_elapsed_reaches_ttl(
            dispatched in 0u64..1_000,
            elapsed in 0u64..(2 * REQUEST_TTL_FOR_TEST),
            from_block in any::<u64>(),
        ) {
            let pending = PendingRequests::new();
            let (pending, _request_id) =
                pending.with_new_request(
                    GetPoolCandidatesInRange {
                        from_block,
                        scan_tip: None,
                    },
                    tick(dispatched),
                );

            let (_pending, reissued) = pending.retry_expired(tick(dispatched + elapsed));

            let expected = usize::from(elapsed >= REQUEST_TTL_FOR_TEST);
            prop_assert_eq!(reissued.len(), expected);
        }
    }
}
