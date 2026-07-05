# Kernel BlocksGraph — Reorg & Finalization Safety Analysis

Branch: `kernel-blocks-graph-split-refactor`
Companion to: `KERNEL_BLOCKS_GRAPH_REFACTOR.md` (goal & staged plan),
`KERNEL_BLOCKS_GRAPH_INVARIANTS.md` (invariant contract)
Subject: the swap from the legacy `kernel::BlocksGraph` to the log-sourced `blocks_graph::BlocksGraph`

This is the `[[validate-reorg-edge-cases-first]]` sign-off artifact for embedding the log-sourced graph
into `State`. It answers one question: **the legacy path resets the whole `State` on two admission
errors — what adversarial inputs was that reset protecting against, and is the new graph safe without
it?**

---

## Background: where legacy resets

`kernel::transition` maps two `with_new_block` errors, in *both* the `HeadObserved` and
`BlockHeaderReceived` arms (`mod.rs:1665, 1767`), to a full `State::reset` — it discards all recent
blocks, the streamed-log buffer, and the canonical tip, keeping only the finalized snapshot, registries,
tick, and in-flight requests:

- `NewBlockError::ConflictingBlockParent` — a known block hash is re-observed with a **different**
  parent hash (`mod.rs:96-101`).
- `NewBlockError::CycleDetected` — the parent walk from a newly inserted block loops without reaching
  the finalized boundary (`find_missing_block_hash` → `BlocksGraphCycleError`, `mod.rs:117-120`).

Everything else (self-parent, existing block, missing parent) is handled locally without a reset.

The new graph's `with_block` (`blocks_graph.rs:317`) has a different error set — `SelfParent`,
`AnchorReadmit`, `DuplicateBlock`, `ConflictingParent`, `PendingBufferFull` — and **every variant hands
`self` back unchanged**. It never resets. So the swap must justify dropping each reset.

---

## Case (a): `CycleDetected` — reset is NOT needed

**Adversarial input.** A provider (WS subscription or `eth_getBlockByHash`) returns headers that form a
parent cycle — e.g. header A claims parent B, header B claims parent A. A block hash cryptographically
commits to its parent, so no honest provider on a real chain can produce this; it requires
inconsistent or malicious data.

**What the reset prevented.** Legacy stores raw `parent_hash` pointers and reconstructs ancestry by
walking them (`find_missing_block_hash`, `block_descends_from`). A cycle would make that walk
non-terminating or mis-rooted; the `CycleDetected` guard catches it at insert and the reset throws away
the corrupt adjacency.

**Why the new graph is safe by construction.**

- **The connected forest cannot contain a cycle.** Connectivity is decided *at insert time*
  (`with_block_capped`, `blocks_graph.rs:344`): a block is `Connected` only if its parent is the anchor
  or an already-`Connected` node. A `Connected` node's parent is therefore always a node that already
  reaches the anchor, so the connected set is a forest rooted at the anchor. `ConnectedHash` — the
  token that authorizes treating a hash as connected — is minted only on this insert path and in
  `promote_reachable` (invariant I1). The canonical walk and every fold (`canonical_oldest_to_newest`,
  `folded_pool_states`, `reanchored_to`) traverse **only** `Connected` nodes, so they walk an acyclic
  forest and always terminate at the anchor.
- **A pending↔pending cycle is representable but inert.** Two pending nodes can reference each other's
  hashes (each parent is absent-or-pending at insert, so both land `Pending`). But pending nodes are
  never traversed as a chain: `promote_reachable` only ever walks *downward* from a newly-*connected*
  node to its pending children (`blocks_graph.rs:384`), and no fold reads pending nodes. A
  pending-pending cycle disconnected from the anchor is thus never followed. It is also **bounded**: the
  pending staging area is capped at `MAX_PENDING_BLOCKS = 1024` (B1, `blocks_graph.rs:350`), and
  finalization drops all pending nodes (`reanchored_to` retains only connected descendants of the new
  anchor — none descend, T6).

**Decision (a): drop the cycle reset.** No guard is needed; the hazard is structurally absent for the
connected forest and inert+bounded for pending nodes. There is no `CycleDetected` analog to wire.

---

## Case (b): `ConflictingBlockParent` — the one real safety delta

**Adversarial input.** A block hash `H` already present in the graph is re-observed with a **different**
parent than the stored one. Because `H` commits to exactly one parent, a conflicting report is
*provably* wrong data — a buggy/desynced provider or a malicious peer injecting a fabricated `(H,
parent)` pair (the kernel trusts the provider; it does not re-verify that `H` hashes to its claimed
parent).

**What legacy does.** Resets the whole `State`. This is a *recovery* behavior: it assumes the stored
view might be the stale/wrong one and rebuilds the recent region from the finalized anchor on the next
observations. It recovers immediately but with a sledgehammer — every unfinalized block and every
buffered streamed log is discarded on a single bad report, which an adversary can therefore use to
**force repeated full resets** (a cheap liveness/DoS lever: one poisoned header per finalized window
keeps the recent graph empty).

**What the new graph does.** `with_block` returns `ConflictingParent(self)` — it **refuses the one
block and keeps the graph unchanged** (first-seen-wins, `blocks_graph.rs:331-340`). It never *persists*
the inconsistency, so unlike legacy it has nothing to recover *from*: the graph was never corrupted.

**The residual gap — bounded, self-healing.** First-seen-wins means: if the *first* report of `H` was
the poisoned one, the later correct `H` (same hash) hits the same conflict and is refused again, so `H`
stays poisoned. Legacy would have recovered via reset; the new graph heals only when the next
finalization prunes `H` — a poisoned block on a fork (or mis-parented) is not a connected descendant of
the advancing anchor and is dropped by `reanchored_to`. The poisoning window is therefore **bounded by
finalization depth**, and self-healing, rather than instantaneous.

Trade, stated plainly:

| | Legacy reset | New graph refuse-and-keep |
|---|---|---|
| Persists bad data | No (resets) | No (refuses) |
| Heals a poisoned-first-report | Immediately | At next finalization (bounded) |
| Blast radius per bad report | Whole recent graph + streamed logs | One block |
| Adversary can force repeated full wipes | **Yes** | No |

**Decision (b): re-root naturally — refuse-and-keep, accept the bounded self-heal.** This is the new
graph's designed behavior (`[[observed-head-derived-canonical-chain]]`). It is strictly better than
legacy on blast radius and on the repeated-wipe DoS lever, and its only regression — delayed healing of
a poisoned first report — is bounded and, being fork/mis-parented, invisible to the canonical read
until it heals. The `ConflictingParent` error is handled locally (log/ignore), **not** mapped to any
reset.

**Documented fallback (not chosen).** If strict legacy parity is later required, map `ConflictingParent`
— and *only* that variant — to rebuilding the new graph as `BlocksGraph::new(finalized_hash)`, mirroring
`State::reset`. Rejected here because it re-imports legacy's whole-graph blast radius and the
repeated-wipe lever purely to shorten an already-bounded, canonically-invisible healing window.

---

## Case (c): incomplete-path finalization — known, benign divergence (wiring requirement)

Not a reset case, but the other by-design divergence the embed must handle so the graphs stay in step.

`reanchored_to` **refuses** an incomplete path (`ReanchorError::Incomplete(self, ranges)`,
`blocks_graph.rs:639`) — if any bloom-hit block on `anchor → target` lacks `Complete` logs, it folds
nothing and returns the hole ranges. Legacy `with_finalized_block_observed` instead **silently compacts
to the latest complete block ≤ target** (`latest_complete_pool_state_update_from`, `mod.rs:1035`). So on
the same `FinalizedBlockObserved`, feeding the new graph `reanchored_to(target)` blindly would refuse to
advance while legacy advances partially — a spurious lockstep divergence.

**Wiring requirement.** Finalization must reanchor the new graph to the **latest complete connected
block ≤ the finalized target** (fold-on-demand, `[[fold-on-demand-design]]`), not to `target` directly.

**Helper gap (confirmed).** No such helper exists on the new graph today — `blocks_graph.rs:604` only
*references* the legacy `State::latest_complete_pool_state_update`. Increment 2 must add a "latest
complete connected block ≤ target" query, composed from `canonical_oldest_to_newest` (the ordered
canonical suffix) + `missing_complete_ranges` (the completeness gate `reanchored_to` already uses), and
reanchor to its result. The Stage-2 differential proptest already pins that the resulting fold equals
legacy's on base-resident pools.

---

## Case (d): off-canonical finalization target — no-op and wait (gating re-derivation)

Added with the finalization-gating re-derivation (shadow `finalized_to` hoisted above the legacy
early-return gates in `with_finalized_block_observed`, so the shadow gates on its own graph).

**The input.** A `FinalizedBlockObserved` whose hash is *connected* in the graph (descends from the
anchor) but is **not on the canonical chain** `anchor → observed_head` — a side fork from the head's
perspective. This happens when the finality feed and the head feed transiently disagree (head lagging
on another branch, or head on a doomed fork the finality signal has already left behind).

**The hazard.** `finalized_to` reanchors via `reanchored_to`, which prunes every non-descendant of the
new anchor (A3) and resets a pruned `observed_head` to the anchor. Blindly honoring an off-canonical
target would prune the entire current head branch — destructive if the disagreement is a transient
feed glitch rather than a real reorg.

**Decision (d): mirror legacy — no-op and wait.** `finalized_to` only advances when the target lies on
the canonical chain (`canonical_oldest_to_newest` membership); otherwise it returns the graph and base
unchanged. This is exactly legacy's `connected_path_contains(tip, target, finalized)` gate re-derived
from the graph. If the head is merely lagging, a later `HeadObserved` on the finalized branch flips the
canonical chain there and the next (recurring) finality observation lands. An empty canonical chain
(head pending/absent, e.g. right after seed activation) refuses too — matching legacy, whose canonical
tip also sits at the anchor until the first post-activation head connects.

**Documented alternative (not chosen): trust finality immediately.** Reanchor onto the fork and prune
the head branch — finality is authoritative, so this is eventually right on a real reorg. Rejected for
the shadow period: it is destructive on a transiently disagreeing feed pair, diverges from legacy, and
buys nothing (the wait resolves within one head + finality observation cycle).

**Accepted divergence (Blockers 1b/1c, signed off 2026-07-05).** The hoist deliberately does NOT
re-derive legacy's completeness gates: legacy holds finalization while a verified pool awaits
`GetPoolData` (1b) or a candidate is pending validation (1c); the shadow folds past any `Complete`
block, so its anchor can advance ahead of legacy's `finalized_state`. The affected pool stays absent
from the shadow base (absent-never-stale; re-seeds on the next absolute event). Pinned one-sided by
`shadow_finalizes_past_legacy_pending_validation_halt` (kernel) and
`delta_only_new_pool_new_graph_advances_but_legacy_waits` (differential). The anchor-height
`GetPoolData` seeding that closes the gap is a **deferred coverage/liveness follow-up** (decided
2026-07-05), not an Increment-4 blocker: absent-never-stale is structural, production reads are
already shadow-only, active pools heal on their next absolute event, and dormant pools heal at the
next restart's bootstrap seeding. Increment 4 must still explicitly decide `GetPoolData`'s fate
(dormant vs delete-and-readd) — this seeding is its only remaining post-swap role.

---

## Summary of decisions (for sign-off)

1. **(a) Cycle reset:** dropped — acyclic-by-construction connected forest; pending cycles inert +
   bounded (`MAX_PENDING_BLOCKS`). No guard wired.
2. **(b) Conflicting-parent reset:** dropped in favor of the new graph's refuse-and-keep; `ConflictingParent`
   handled locally, never a reset. Bounded, self-healing gap accepted; mirror-reset fallback documented
   but not chosen.
3. **(c) Finalization:** reanchor to the latest complete connected block ≤ target (needs a new helper),
   not blindly to target — matching legacy's partial-compaction and the fold-on-demand design.
4. **(d) Off-canonical finalization target:** no-op and wait (canonical-membership gate in
   `finalized_to`), mirroring legacy's connected-path gate; trust-finality-immediately documented but
   not chosen. 1b/1c wait-vs-advance divergence accepted until the anchor-height `GetPoolData` fix.

Sign-off on these three gates Increment 2 (embedding + feeding the new graph from `transition`);
decision (d) signed off with the finalization-gating re-derivation (2026-07-05).
