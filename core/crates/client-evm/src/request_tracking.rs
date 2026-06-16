use std::{collections::HashMap, fmt, marker::PhantomData};

use crate::tick::Tick;

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

pub(crate) struct RequestIdSequence {
    last_request_id: RawRequestId,
}

impl RequestIdSequence {
    pub(crate) fn new() -> Self {
        RequestIdSequence {
            last_request_id: RawRequestId(0),
        }
    }

    pub(crate) fn issue<R>(&mut self) -> RequestId<R> {
        let next_request_id = self.last_request_id.next();

        self.last_request_id = next_request_id;
        next_request_id.typed()
    }

    #[cfg(test)]
    pub(crate) fn raw_for_test(&self) -> u64 {
        self.last_request_id.0
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

pub(crate) struct RequestCollection<R> {
    requests: HashMap<RequestId<R>, PendingPayload<R>>,
}

pub(crate) trait RequestIssuer {
    fn request_ids_mut(&mut self) -> &mut RequestIdSequence;
}

pub(crate) trait RequestStore<R> {
    fn request_collection(&self) -> &RequestCollection<R>;
    fn request_collection_mut(&mut self) -> &mut RequestCollection<R>;
}

impl<R> RequestCollection<R> {
    pub(crate) fn new() -> Self {
        RequestCollection {
            requests: HashMap::new(),
        }
    }

    #[cfg(test)]
    pub(crate) fn get(&self, request_id: &RequestId<R>) -> Option<&PendingPayload<R>> {
        self.requests.get(request_id)
    }

    pub(crate) fn insert(&mut self, request_id: RequestId<R>, payload: R, dispatched_at: Tick) {
        let _ = self.requests.insert(
            request_id,
            PendingPayload {
                payload,
                dispatched_at,
            },
        );
    }

    pub(crate) fn remove(&mut self, request_id: &RequestId<R>) -> Option<PendingPayload<R>> {
        self.requests.remove(request_id)
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (&RequestId<R>, &PendingPayload<R>)> {
        self.requests.iter()
    }

    pub(crate) fn values(&self) -> impl Iterator<Item = &PendingPayload<R>> {
        self.requests.values()
    }

    pub(crate) fn retain<F>(&mut self, retain: F)
    where
        F: FnMut(&RequestId<R>, &mut PendingPayload<R>) -> bool,
    {
        self.requests.retain(retain);
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.requests.len()
    }
}

pub(crate) fn issue_request<S, R>(store: S, payload: R, tick: Tick) -> (S, RequestId<R>)
where
    S: RequestIssuer + RequestStore<R>,
{
    let mut store = store;
    let request_id = store.request_ids_mut().issue();

    store
        .request_collection_mut()
        .insert(request_id, payload, tick);

    (store, request_id)
}

pub(crate) fn take_request<S, R>(
    store: S,
    request_id: &RequestId<R>,
) -> (S, Option<PendingPayload<R>>)
where
    S: RequestStore<R>,
{
    let mut store = store;
    let pending_payload = store.request_collection_mut().remove(request_id);

    (store, pending_payload)
}

pub(crate) fn expired_request_ids<S, R>(store: &S, tick: Tick) -> Vec<RequestId<R>>
where
    S: RequestStore<R>,
{
    store
        .request_collection()
        .iter()
        .filter(|(_, request)| tick.is_expired_since(request.dispatched_at))
        .map(|(request_id, _)| *request_id)
        .collect()
}

pub(crate) fn retry_request<S, R>(
    store: S,
    request_id: RequestId<R>,
    tick: Tick,
) -> (S, Option<IssuedRequest<R>>)
where
    S: RequestIssuer + RequestStore<R>,
    R: Clone,
{
    let (store, pending_payload) = take_request(store, &request_id);

    match pending_payload {
        Some(PendingPayload { payload, .. }) => {
            let request_payload = payload.clone();
            let (store, request_id) = issue_request(store, request_payload, tick);

            (
                store,
                Some(IssuedRequest {
                    request_id,
                    request_payload: payload,
                }),
            )
        }
        None => (store, None),
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;

    pub(crate) fn get_request<'a, S, R>(
        store: &'a S,
        request_id: &RequestId<R>,
    ) -> Option<&'a PendingPayload<R>>
    where
        S: RequestStore<R>,
    {
        store.request_collection().get(request_id)
    }
}
