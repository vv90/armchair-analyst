//! The fetch-request ledger: a pure, singleton-per-kind record of the data-plane requests currently
//! in flight, correlated by an issued [`FetchId`]. It is the client-side, right-sized analogue of the
//! kernel's `pending_requests` (whose generic many-per-kind machinery is `pub(crate)` to `client-evm`
//! and heavier than this crate needs): the client issues **at most one** `/pools/meta`, `/slice`, and
//! `/health` request at a time, so each kind is a single [`Option`] slot rather than a keyed collection.
//!
//! It gives the reducer three things the poll loop was missing:
//! - **in-flight gating** — [`PendingFetches::ensure`] issues a request only when that kind's slot is
//!   free, so a slow server can no longer pile up concurrent fetches;
//! - **bounded retry** — the same `ensure` re-issues a slot whose request has been outstanding longer
//!   than [`FETCH_TTL_TICKS`], reclaiming a lost/hung fetch (there is no separate retry pass: every
//!   desired fetch is re-`ensure`d each tick, so an expired one is re-issued naturally);
//! - **mismatched-response rejection** — [`PendingFetches::accept`] takes a slot only when the response
//!   carries the exact `FetchId` still recorded there, so a superseded (re-issued-past) response is
//!   dropped instead of applied. This is what makes retry safe: a re-issue bumps the id, so the old
//!   response no longer matches.
//!
//! Everything here is pure and synchronous; the runtime advances the clock by folding [`Event::Tick`],
//! which calls [`PendingFetches::advance`].
//!
//! [`Event::Tick`]: crate::state::Event::Tick

use crate::state::FetchKind;

/// Ticks a request may be outstanding before [`PendingFetches::ensure`] re-issues it. Mirrors the
/// kernel's `REQUEST_TTL`; at the runtime's 500ms poll interval this is ≈5s before a hung fetch is
/// retried.
pub(crate) const FETCH_TTL_TICKS: u64 = 10;

/// An opaque, monotonic fetch-request id — the correlation token echoed back on the response so a
/// superseded reply can be told apart from the current one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FetchId(u64);

/// One in-flight request: its id and the tick it was dispatched at (for the TTL comparison).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Slot {
    id: FetchId,
    dispatched_at: u64,
}

/// The singleton-per-kind in-flight ledger: at most one meta/slice/health request outstanding, plus a
/// monotonic tick clock (a local `u64`; `client_evm::Tick` is not exported from that crate) and the
/// next id to hand out. Pure — the reducer owns it in [`crate::AppState`] and threads it through.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PendingFetches {
    next_id: u64,
    now: u64,
    meta: Option<Slot>,
    slice: Option<Slot>,
    health: Option<Slot>,
}

impl PendingFetches {
    /// An empty ledger at tick 0 with nothing in flight.
    pub fn new() -> PendingFetches {
        PendingFetches::default()
    }

    /// Advance the tick clock one step; the runtime calls this once per `Event::Tick`.
    pub fn advance(&mut self) {
        self.now = self.now.wrapping_add(1);
    }

    /// Ensure a request of `kind` is in flight, issuing a fresh [`FetchId`] and occupying the slot when
    /// it is free or its current request has expired (outstanding ≥ [`FETCH_TTL_TICKS`]). Returns the
    /// new id to dispatch, or `None` when a still-fresh request is already outstanding (skip — the gate).
    pub fn ensure(&mut self, kind: FetchKind) -> Option<FetchId> {
        if self.is_fresh(kind) {
            return None;
        }
        let id = FetchId(self.next_id);
        self.next_id = self.next_id.wrapping_add(1);
        self.set_slot(
            kind,
            Some(Slot {
                id,
                dispatched_at: self.now,
            }),
        );
        Some(id)
    }

    /// Take the slot for `kind` iff it still holds exactly `id`, returning whether the response is
    /// accepted. A mismatch (a superseded, re-issued-past response) leaves the ledger untouched and
    /// returns `false`, so the caller drops it.
    pub fn accept(&mut self, kind: FetchKind, id: FetchId) -> bool {
        let matched = matches!(self.slot(kind), Some(slot) if slot.id == id);
        if matched {
            self.set_slot(kind, None);
        }
        matched
    }

    /// Whether `kind`'s slot holds a request still within its TTL (so `ensure` must not re-issue).
    fn is_fresh(&self, kind: FetchKind) -> bool {
        match self.slot(kind) {
            Some(slot) => self.now.wrapping_sub(slot.dispatched_at) < FETCH_TTL_TICKS,
            None => false,
        }
    }

    fn slot(&self, kind: FetchKind) -> Option<&Slot> {
        match kind {
            FetchKind::Meta => self.meta.as_ref(),
            FetchKind::Health => self.health.as_ref(),
            FetchKind::Slice => self.slice.as_ref(),
        }
    }

    fn set_slot(&mut self, kind: FetchKind, slot: Option<Slot>) {
        match kind {
            FetchKind::Meta => self.meta = slot,
            FetchKind::Health => self.health = slot,
            FetchKind::Slice => self.slice = slot,
        }
    }
}

#[cfg(test)]
impl FetchId {
    /// A fixed id for tests that need a value the ledger did not issue (e.g. a deliberately stale one).
    pub(crate) fn from_raw_for_test(value: u64) -> FetchId {
        FetchId(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensure_issues_distinct_ids_and_gates_a_fresh_slot() {
        let mut pending = PendingFetches::new();
        let first = pending.ensure(FetchKind::Slice).expect("free slot issues");
        // A second ensure while the request is fresh gates: nothing issued.
        assert!(pending.ensure(FetchKind::Slice).is_none());
        // A different kind is independent.
        let meta = pending
            .ensure(FetchKind::Meta)
            .expect("distinct kind issues");
        assert_ne!(first, meta);
    }

    #[test]
    fn ensure_reissues_after_the_ttl_with_a_new_id() {
        let mut pending = PendingFetches::new();
        let first = pending.ensure(FetchKind::Slice).expect("issue");
        // Just under the TTL: still gated.
        for _ in 0..(FETCH_TTL_TICKS - 1) {
            pending.advance();
            assert!(pending.ensure(FetchKind::Slice).is_none());
        }
        // The tick that reaches the TTL re-issues a fresh id.
        pending.advance();
        let reissued = pending
            .ensure(FetchKind::Slice)
            .expect("expired slot re-issues");
        assert_ne!(first, reissued);
    }

    #[test]
    fn accept_takes_a_matching_id_and_frees_the_slot() {
        let mut pending = PendingFetches::new();
        let id = pending.ensure(FetchKind::Health).expect("issue");
        assert!(pending.accept(FetchKind::Health, id));
        // Slot is free again, so the next ensure issues immediately.
        assert!(pending.ensure(FetchKind::Health).is_some());
    }

    #[test]
    fn accept_rejects_a_mismatched_id_and_preserves_the_slot() {
        let mut pending = PendingFetches::new();
        let id = pending.ensure(FetchKind::Slice).expect("issue");
        let stale = FetchId::from_raw_for_test(9999);
        assert_ne!(id, stale);
        // A wrong id is rejected and changes nothing.
        assert!(!pending.accept(FetchKind::Slice, stale));
        // The real id is still accepted afterwards.
        assert!(pending.accept(FetchKind::Slice, id));
    }
}
