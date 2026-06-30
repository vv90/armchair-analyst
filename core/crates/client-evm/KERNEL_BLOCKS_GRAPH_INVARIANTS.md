# Kernel BlocksGraph — Invariant Contract

Branch: `kernel-blocks-graph-split-refactor`
Companion to: `KERNEL_BLOCKS_GRAPH_REFACTOR.md` (goal & staged plan)
Subject type: `kernel/blocks_graph.rs`

This pins the invariants of the **new, log-sourced** blocks graph. The extract and the payload
swap are done together deliberately: the payload model *changes which invariants exist* (per-block
absolute snapshots and snapshot/failure disjointness disappear; log-completeness and a finalization
foldability gate appear), so encoding the old semantics first would be wasted work.

---

## Enforcement taxonomy (strict)

- **Structural** — the data type *cannot represent* a violation. There is no code path, correct or
  buggy, that produces an invalid value, because the value has no encoding. Needs **no test**.
- **Runtime** — enforced by a function: a constructor precondition, a transition guard, or a debug
  assertion. A smart constructor is **runtime**, not structural — it is ordinary code that can be
  wrong, and it must be **covered by tests** (invariant tests + differential/property tests) exactly
  like any other function.

The design goal is to move as many invariants as possible into *structural*, and to make the
remaining *runtime* set small, explicit, and individually tested.

---

## I. Topology — connected/pending tree

| # | Invariant | Enforcement | Test obligation | Origin |
|---|-----------|-------------|-----------------|--------|
| T1 | `anchor` is the root and has no parent within the graph | **Structural** — `anchor: BlockHash` is a field, not a node; there is no parent slot for it | none | carried |
| T2 | Every `connected` node's parent chain reaches `Anchor` (referential integrity: each `ConnectedHash` in a parent resolves to a present `connected` node) | **Runtime** — upheld by the single admission/promotion path | invariant test + proptest across insertion orders | carried (parent-walk) |
| T3 | `connected` is acyclic | **Runtime** — corollary of T2's referential integrity (not free, because T2 is runtime) | covered by the T2 proptest (walk terminates) | carried (cycle defense) |
| T4 | `canonical_tip` resolves to a present connected node (or `Anchor`) | **Runtime** — referential integrity of the stored `AnchoredRef` | invariant test | carried ("tip known-or-finalized") |
| T5 | `{anchor} / connected / pending` key-sets are pairwise disjoint | **Runtime** — nothing stops the same hash being inserted into two maps | debug assert + proptest | carried (self-parent/overlap) |
| T6 | A `pending` node is genuinely *not yet reachable* to the anchor (not merely unchecked) | **Runtime** — requires promotion to be *exhaustive* on every admission | proptest: after any admission, no `pending` node is anchor-reachable | new framing |

### Structural sub-property of `AnchoredRef` (the part that *is* free)

`AnchoredRef = Anchor | Block(ConnectedHash)` **structurally** eliminates the *"missing / unknown
parent"* state for a connected node and for `canonical_tip`. The old
`while current != finalized { … else break }` walk had to handle a dangling case at every step;
that case is now **unrepresentable**. What remains *runtime* is only **referential integrity** —
that the wrapped hash is actually present in `connected` (see T2/T4, I1). Keep the two apart:
*shape totality* is structural; *the referent exists* is runtime.

---

## II. Identity / proof

| # | Invariant | Enforcement | Test obligation |
|---|-----------|-------------|-----------------|
| I1 | A `ConnectedHash` names a hash present in `connected` | **Runtime** — `ConnectedHash` is a plain newtype; `ConnectedHash(any_hash)` is constructible. Upheld by minting only on the insert/promote path and consuming in-transition | proptest: every minted `ConnectedHash` resolves |
| I2 | At most one node per block hash | **Structural** — `HashMap`/`BTreeMap` key uniqueness | none |

> Note: `ConnectedHash` is a *runtime* proof-of-existence convention, not a structural guarantee.
> Branding it with a generative lifetime would make it structural but does not survive removal
> (finalization prune), so it stays a runtime convention — and therefore gets tested (I1).

---

## III. Log payload — source of truth

Governing principle: **no derived state**. A non-anchor block stores *logs*, never folded
snapshots. Absolute pool state is reconstructed by fold-on-demand.

| # | Invariant | Enforcement | Test obligation |
|---|-----------|-------------|-----------------|
| L1 | Per-block logs are deduped by `log_index` | **Structural** — `BTreeMap<u64, _>` key uniqueness | none |
| L2 | Per-block logs iterate ascending by `log_index` | **Structural** — `BTreeMap` ordering | none |
| L3 | Each stored log's `log_index` equals its map key | **Runtime** — value is whole `PoolLog` (Decision 1 → (a)); debug-assert key == `PoolLog::log_index`. The field is `#[deprecated]`: the key is the ordering index, new code never reads the field | assert |
| L4 | Every stored log is bloom-positive against the block's `logs_bloom` (when present) | **Runtime** — bloom is consensus-derived from the real logs | invariant test + shadow check |
| L5 | `BlockLogs` authority is monotone: `Unknown → Streamed → Complete`, never backward; `Streamed` only grows; `Complete ⊇ Streamed` | **Runtime** — enforced by the log-merge transition. (Partial structural help: the 3-variant enum forbids states *outside* the ladder, but not a backward step.) | proptest on the merge op |
| L6 | No absolute pool snapshot is stored on any non-anchor block | **Structural** — `BlockData` has no snapshot field; a per-block snapshot is unrepresentable | none |

**Why L5 is sound:** for a fixed block hash the canonical log set is *immutable* (same hash ⇒ same
block ⇒ same logs). So streamed logs accumulate toward a fixed truth, never conflict, and `Complete`
is a terminal authoritative superset.

---

## IV. Anchor & finalization

| # | Invariant | Enforcement | Test obligation |
|---|-----------|-------------|-----------------|
| A1 | The finalized hash is held in exactly one place (`anchor`); `FinalizedState.block_hash` is removed | **Structural** *after* the swap — once no other field can hold it, duplication is unrepresentable. Pre-swap: anchor is the sole home (Decision 2 → own now); `FinalizedState.block_hash` is `#[deprecated]` and unread by new code until removed at swap | none (post-swap) |
| A2 | `reanchored_to(new)` targets a connected descendant of the current anchor | **Runtime** — `ConnectedHash` gives shape (runtime referent, I1); descendant relation is a runtime check | reorg test |
| A3 | After reanchor, every surviving `connected` node descends from the new anchor; non-descendant forks and now-unreachable `pending` are pruned | **Runtime** — the prune step | reorg test + post-state invariant check |
| A4 | **Finalization safety gate:** the anchor advances to `new` only if the path old-anchor→`new` is fully *foldable* for every tracked pool (each block `Complete`, or bloom-clear for that pool). Otherwise backfill (getLogs) first | **Runtime** — precondition before reanchor | proptest: reanchor refuses an incompletely-resolved path |

A4 is the critical correctness gate of the whole refactor: the anchor snapshot is authoritative and
permanent, so a fold may never cross a merely-`Streamed` (best-effort WS) block. This is exactly
where `eth_getLogs` is retained — as the authoritative verification/backfill path, not the per-block
driver.

---

## V. Boundedness

| # | Invariant | Enforcement | Test obligation |
|---|-----------|-------------|-----------------|
| B1 | `|pending|` is bounded (staging area; today's analog `MAX_STREAMED_LOG_BLOCKS = 1024`) | **Runtime** — eviction at admission | invariant test |
| B2 | `connected` depth is bounded by finalization distance | **Runtime** — consequence of A3/A4 pruning | covered by reorg tests |

B1 also bounds the "parent permanently below the anchor" case: an unpromotable fork ages out rather
than leaking.

---

## Derived queries (NOT invariants — computed, never stored)

- **`foldability(B, P)`** — walking anchor→B, every block is `Complete` **or** bloom-clear for `P`,
  short-circuiting if an absolute event (Swap/Initialize) for `P` appears mid-path (self-seeding).
  Reconstructed on demand per [[no-derived-state-reconstruct-pure]].
- **pool candidates at B** — `logs.values().map(|l| l.pool)`, derived from the stored logs rather
  than a separate stored set (replaces the old `PoolLogsStatus` candidate `HashSet`).

---

## Structural vs runtime — the scorecard

**Structural (no test needed):** I2, L1, L2, L6; A1 (post-swap); the *shape totality* of
`AnchoredRef` (no dangling-parent state).

**Runtime (must be tested):** T2, T3, T4, T5, T6, I1, L3 (if field retained), L4, L5, A2, A3, A4,
B1, B2.

The two runtime properties to pin hardest in the differential proptest:
1. **A4** — new fold-from-logs == old stored-derivation, *and* finalization refuses an incomplete path.
2. **L5** — log merge is monotone and `Complete` is an authoritative superset.

Everything else runtime is a localized invariant test or a corollary of these.

---

## Resolved decisions

1. **L3 — `log_index` duplication. RESOLVED → (a).** Per-block logs are `BTreeMap<u64, PoolLog>` with
   a debug-assert that the map key equals `PoolLog::log_index`; we keep `PoolLog` whole rather than
   fragmenting a domain type. **`PoolLog::log_index` is now `#[deprecated]`**: intra-block ordering is
   the `BTreeMap` key, and **no new code may read or write `log_index`** — it survives only so
   pre-swap code paths keep compiling, and is removed at the swap.
2. **A1 timing — anchor ownership. RESOLVED → own the anchor here now.** `BlocksGraph::anchor` is the
   sole home of the finalized hash from the start (transient duplication while the new graph is
   parallel/untrusted is invisible pre-swap and keeps the new type clean of a temporary borrow).
   **`FinalizedState.block_hash` is now `#[deprecated]`**: **no new code may read it** — read the
   anchor from the graph instead. It survives only so pre-swap code paths keep compiling, and is
   removed at the swap.
