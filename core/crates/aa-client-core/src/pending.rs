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
    use proptest::prelude::*;

    use super::*;

    /// Every [`FetchKind`], so the ledger properties are parametric over the whole enum rather than
    /// over one hand-picked kind — the next data-plane request added lands *inside* these pins
    /// instead of beside them. (Mirrors the kernel's `retaining_block_targets` property, which is
    /// likewise parametric over every request type.)
    fn fetch_kind() -> impl Strategy<Value = FetchKind> {
        prop_oneof![
            Just(FetchKind::Meta),
            Just(FetchKind::Health),
            Just(FetchKind::Slice),
        ]
    }

    /// One symbolic ledger operation. `AcceptCurrent` answers with whatever id the slot currently
    /// holds (so it models a *timely* response); `AcceptStale` answers with an arbitrary raw id (so
    /// it models a superseded or forged one).
    #[derive(Clone, Copy, Debug)]
    enum LedgerOp {
        Ensure(FetchKind),
        AcceptCurrent(FetchKind),
        AcceptStale(FetchKind, u64),
        Advance,
    }

    fn ledger_op() -> impl Strategy<Value = LedgerOp> {
        prop_oneof![
            fetch_kind().prop_map(LedgerOp::Ensure),
            fetch_kind().prop_map(LedgerOp::AcceptCurrent),
            (fetch_kind(), 0u64..32).prop_map(|(kind, id)| LedgerOp::AcceptStale(kind, id)),
            Just(LedgerOp::Advance),
        ]
    }

    fn ledger_ops() -> impl Strategy<Value = Vec<LedgerOp>> {
        prop::collection::vec(ledger_op(), 0..64)
    }

    /// Ledger histories without [`LedgerOp::AcceptStale`], for the independence property.
    ///
    /// A forged id is drawn from the same small range `ensure` mints from, and the id counter is
    /// **shared across kinds** — so *which* id a given kind's slot holds depends on how many ids the
    /// other kinds consumed first. A forged `2` can therefore miss in an interleaved history and hit
    /// in the isolated one. That is a true fact about the shared id space, not slot-routing leakage,
    /// so it must not be conflated with the independence being tested here; stale-id rejection is
    /// pinned separately by `an_unissued_id_is_never_accepted_and_changes_nothing`.
    fn id_independent_ledger_ops() -> impl Strategy<Value = Vec<LedgerOp>> {
        prop::collection::vec(
            prop_oneof![
                fetch_kind().prop_map(LedgerOp::Ensure),
                fetch_kind().prop_map(LedgerOp::AcceptCurrent),
                Just(LedgerOp::Advance),
            ],
            0..64,
        )
    }

    /// What the ledger did for one kind, observably: `Some(true/false)` for an accept's verdict and
    /// `None`/`Some` shape for an issuance. Ids are deliberately *not* recorded — the id counter is
    /// shared across kinds, so comparing ids would defeat the independence property below.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Observation {
        Issued(bool),
        Accepted(bool),
    }

    /// Fold `ops` through a fresh ledger, returning the final ledger, every id `ensure` ever issued,
    /// and the observations for `focus` only.
    fn drive(
        ops: &[LedgerOp],
        focus: FetchKind,
    ) -> (PendingFetches, Vec<FetchId>, Vec<Observation>) {
        let mut pending = PendingFetches::new();
        let mut issued = Vec::new();
        let mut observed = Vec::new();
        // The id each kind's slot currently holds, so `AcceptCurrent` can answer with the right one.
        let mut live: Vec<(FetchKind, FetchId)> = Vec::new();

        for op in ops {
            match *op {
                LedgerOp::Ensure(kind) => {
                    let id = pending.ensure(kind);
                    if let Some(id) = id {
                        issued.push(id);
                        live.retain(|(held, _)| *held != kind);
                        live.push((kind, id));
                    }
                    if kind == focus {
                        observed.push(Observation::Issued(id.is_some()));
                    }
                }
                LedgerOp::AcceptCurrent(kind) => {
                    let id = live
                        .iter()
                        .find(|(held, _)| *held == kind)
                        .map(|(_, id)| *id)
                        .unwrap_or_else(|| FetchId::from_raw_for_test(u64::MAX));
                    let accepted = pending.accept(kind, id);
                    if accepted {
                        live.retain(|(held, _)| *held != kind);
                    }
                    if kind == focus {
                        observed.push(Observation::Accepted(accepted));
                    }
                }
                LedgerOp::AcceptStale(kind, raw) => {
                    let accepted = pending.accept(kind, FetchId::from_raw_for_test(raw));
                    if accepted {
                        live.retain(|(held, _)| *held != kind);
                    }
                    if kind == focus {
                        observed.push(Observation::Accepted(accepted));
                    }
                }
                LedgerOp::Advance => pending.advance(),
            }
        }

        (pending, issued, observed)
    }

    proptest! {
        /// Every id `ensure` ever hands out is distinct, across all kinds and any interleaving. This
        /// is what makes `accept`'s id-gate meaningful: a re-issue must not be able to mint an id
        /// that a still-in-flight (superseded) response could match, or a stale reply would be
        /// applied as if it were current.
        #[test]
        fn every_issued_id_is_distinct(ops in ledger_ops()) {
            let (_, issued, _) = drive(&ops, FetchKind::Slice);

            let unique = issued.iter().collect::<std::collections::HashSet<_>>();
            prop_assert_eq!(unique.len(), issued.len(), "ensure minted a duplicate id");
        }

        /// A response carrying an id the ledger never issued is *always* rejected, and leaves the
        /// ledger bit-for-bit unchanged. The forged id is drawn from the same small range the real
        /// counter uses, so this genuinely collides with issued values rather than testing an
        /// obviously-out-of-band number.
        #[test]
        fn an_unissued_id_is_never_accepted_and_changes_nothing(
            ops in ledger_ops(),
            kind in fetch_kind(),
            raw in 0u64..32,
        ) {
            let (pending, issued, _) = drive(&ops, kind);
            let forged = FetchId::from_raw_for_test(raw);
            prop_assume!(!issued.contains(&forged));

            let mut probed = pending.clone();
            prop_assert!(!probed.accept(kind, forged));
            prop_assert_eq!(probed, pending, "a rejected accept must not mutate the ledger");
        }

        /// Accepting frees the slot: whatever the history, a successful `accept` is immediately
        /// followed by an `ensure` that issues. This is the liveness half of the gate — a delivered
        /// response must never leave its kind wedged until the TTL.
        #[test]
        fn a_successful_accept_frees_the_slot_for_immediate_reissue(
            ops in ledger_ops(),
            kind in fetch_kind(),
        ) {
            let (mut pending, _, _) = drive(&ops, kind);
            // Put a known request in flight (either it issues, or one is already live).
            let live = match pending.ensure(kind) {
                Some(id) => id,
                None => {
                    // Gated ⇒ a fresh request is in flight; drive the TTL out to learn its successor.
                    for _ in 0..FETCH_TTL_TICKS {
                        pending.advance();
                    }
                    match pending.ensure(kind) {
                        Some(id) => id,
                        None => return Ok(()),
                    }
                }
            };

            prop_assert!(pending.accept(kind, live));
            prop_assert!(
                pending.ensure(kind).is_some(),
                "a freed slot must issue immediately, not wait for the TTL"
            );
        }

        /// The kinds are independent: the observable behaviour of one kind is unchanged by
        /// arbitrary interleaved operations on the *other* kinds. Only the shared tick clock may
        /// couple them, so `Advance` is kept in both runs. Without this, a slot-indexing bug (say
        /// `set_slot` writing the wrong field) could let a `/health` reply free the `/slice` slot.
        #[test]
        fn operations_on_other_kinds_do_not_affect_a_kind(
            ops in id_independent_ledger_ops(),
            focus in fetch_kind(),
        ) {
            let (_, _, interleaved) = drive(&ops, focus);

            // The same history with every other kind's operations removed (the clock survives).
            let isolated_ops = ops
                .iter()
                .copied()
                .filter(|op| match op {
                    LedgerOp::Ensure(kind)
                    | LedgerOp::AcceptCurrent(kind)
                    | LedgerOp::AcceptStale(kind, _) => *kind == focus,
                    LedgerOp::Advance => true,
                })
                .collect::<Vec<_>>();
            let (_, _, isolated) = drive(&isolated_ops, focus);

            prop_assert_eq!(interleaved, isolated, "another kind's traffic changed this kind");
        }

        /// The TTL is measured from dispatch, not from an absolute epoch: a slot issued at *any*
        /// clock value is gated for exactly `FETCH_TTL_TICKS - 1` advances and re-issues on the
        /// `FETCH_TTL_TICKS`-th, with a fresh id. Parametric over kind and over the clock the
        /// request was dispatched at.
        #[test]
        fn a_slot_is_gated_for_exactly_the_ttl_then_reissues(
            kind in fetch_kind(),
            lead in 0u64..16,
        ) {
            let mut pending = PendingFetches::new();
            for _ in 0..lead {
                pending.advance();
            }
            let first = pending.ensure(kind).expect("a free slot always issues");

            for elapsed in 1..FETCH_TTL_TICKS {
                pending.advance();
                prop_assert!(
                    pending.ensure(kind).is_none(),
                    "re-issued after only {elapsed} of {FETCH_TTL_TICKS} ticks"
                );
            }

            pending.advance();
            let reissued = pending.ensure(kind).expect("an expired slot re-issues");
            prop_assert_ne!(reissued, first, "a re-issue must mint a new id");
        }

        /// A re-issue supersedes: once the TTL has elapsed and the slot has been re-issued, the
        /// *original* id can no longer be accepted. This is the property that makes retry safe —
        /// the late reply to the abandoned request is dropped instead of applied.
        #[test]
        fn a_reissue_supersedes_the_previous_id(kind in fetch_kind()) {
            let mut pending = PendingFetches::new();
            let first = pending.ensure(kind).expect("a free slot always issues");
            for _ in 0..FETCH_TTL_TICKS {
                pending.advance();
            }
            let second = pending.ensure(kind).expect("an expired slot re-issues");

            prop_assert!(!pending.accept(kind, first), "the superseded id must be rejected");
            prop_assert!(pending.accept(kind, second), "the current id must be accepted");
        }
    }

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
