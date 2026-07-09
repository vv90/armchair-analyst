# Kernel BlocksGraph Refactor — Goal & Target Approach

Branch: `kernel-blocks-graph-split-refactor`

## TL;DR

Make the **multi-provider WebSocket logs stream the source of truth** for pool reserves, eliminating
the per-block `eth_getLogs` storm. Do it as **small, precise, individually-reviewable increments** —
and use the same effort to **carve the blocks graph out of the 11k-line `kernel/mod.rs`** into its own
module. The chain kernel is trusted code; a regression here is critical, so correctness guardrails take
priority over speed.

---

## Why

### Problem with the current data flow
- On **every new block** the kernel schedules an `eth_getLogs` request to discover which pools changed.
- This is ~the dominant share of RPC volume and, worse, it **scales with how far behind a chain is**:
  the further behind, the more blocks to backfill, the more getLogs calls, the faster we hit provider
  rate limits — a runaway feedback loop.
- The data we actually want (the freshest possible pool reserve snapshots to optimize over) is already
  arriving on the wire via logs subscriptions. Re-fetching it per block is wasteful and slow.

### Goal
1. **Freshest reserves.** Track latest pool reserves directly from incoming WS log events so the
   optimizer runs on the most recent state possible.
2. **Robust ingestion.** Combine **multiple independent provider WS connections**, then **debounce and
   deduplicate** events, minimizing the probability of skipped logs or connection stutter/drops
   silently losing data.
3. **Drop per-block getLogs** as the steady-state discovery mechanism (kept only as an authoritative
   verification/backfill path, see Non-goals).

---

## Lesson from the previous attempt

A prior "shotgun" refactor on a **separate branch** reproduced these benefits but grew too large to
review or reason about safely — exactly the wrong outcome for trusted kernel code. It is abandoned in
place as a reference, **not** merged.

**This branch deliberately does the opposite:**
- Each change is as small as possible and understandable at a glance.
- No change is made that isn't immediately necessary for the current increment.
- Structural cleanup (splitting the blocks graph out) happens **alongside** the behavior change, so the
  end state is both better-behaved *and* better-organized — but each step is still atomic.

---

## Target approach

### 1. Separate the blocks graph into its own module
`kernel/mod.rs` is ~11k lines and mixes the block DAG, pool/token registries, request scheduling, and
state transitions. The blocks graph (`BlocksGraph` / `BlockNode` and their pure operations) is a
cohesive unit with a clear surface and is the locus of this refactor. Extracting it:
- shrinks the trusted monolith into reviewable pieces,
- gives the new log-driven implementation a clean place to live, and
- makes the old/new boundary explicit.

### 2. Parallel implementation, then swap
Rather than mutating the trusted graph in place, build the **new log-driven blocks graph as a separate
type/module** that satisfies the same boundary contract. Verify it against the existing one
(differential property tests — new path == old path across generated chains and event orderings), and
only **switch the kernel over once verified**. The old implementation stays untouched and runnable
until the swap.

### 3. Source-of-truth model (target)
- Per block we store a **deduped decoded pool-log set + header bloom**, not a derived snapshot.
- Absolute pool states live in **exactly one place** (the finalized anchor). Unfinalized state for a
  pool at block B is `fold(finalized_state[P], canonical P-logs from finalized→B)`, self-seeding when a
  log carries an absolute event (e.g. Swap/Initialize).
- Reorgs are owned by the graph (parent-hash linked DAG); `removed` log flags are ignored.
- Dedup key: `(block_hash, log_index)`. Bloom is used **negatively only** (bloom-clear ⇒ fast-path
  skip; a bloom hit is never on its own a fetch trigger, given high false-positive rates on busy
  blocks).

### 4. The clean old/new split boundary
The split is drawn at the **`BlocksGraph` type's pure interface** — the set of methods the surrounding
kernel calls to ingest headers/logs, query canonical/pool state, schedule work, and compact on
finalization. The new module must honor that boundary so the swap is a single, localized change rather
than a diffuse rewrite. (Defining and pinning this exact method-level contract is the first concrete
deliverable — see Stage 0.)

---

## Guardrails (non-negotiable for trusted kernel code)

- **No kernel test regression.** The existing proptest sequence tests and `assert_state_invariants`
  must stay **green and unchanged**. A *forced* change to an invariant/property test is a red flag that
  semantics drifted and needs explicit justification — never relax a property to make a refactor pass.
- **Differential proptest before deletion.** Before removing any old derivation path, land a property
  test asserting new-path == old-path across generated chains/event orderings, get it green, *then*
  delete. *Extend* generators to cover new events; never weaken them.
- **Test-first for new logic.** Write the failing test, confirm it fails for the intended reason, then
  implement. Stubs/placeholders first to establish compile-time contracts.
- **Reorg/finalization edge cases signed off first.** Walk through fork-above / fork-into-region,
  finalization no-op vs advance (including advancing into self-seeded regions), and orphan pruning
  *before* implementing, and pin each with a dedicated reorg test.
- **Purity.** Graph logic stays pure and deterministic; all I/O (multi-WS merge, debounce, dedup) lives
  in thin adapters outside the kernel.
- **No edits without an explicit command** from the human; smallest possible increments throughout.

---

## Staged plan (incremental)

> Each stage is independently buildable, testable, and reviewable. Order may adjust, but each step
> stays small.

- **Stage 0 — Define the boundary.** Document and (where useful) encode the exact `BlocksGraph`
  method-level contract the kernel depends on. No behavior change. This is the seam old/new are split
  on.
- **Stage 1 — Extract the graph.** Move the existing `BlocksGraph`/`BlockNode` and their pure ops into
  a dedicated module behind that boundary. Pure mechanical move; all existing tests stay green
  unchanged.
- **Stage 2 — Parallel log-driven graph.** Implement the new log-source-of-truth graph (deduped log
  set + bloom; fold-on-demand pool state) as a separate type satisfying the boundary. Cover it with its
  own tests plus a **differential proptest** vs the existing graph.
- **Stage 3 — Multi-provider ingestion adapter.** Outside the kernel: merge multiple provider WS
  feeds, debounce, and dedupe into the single log stream the new graph consumes. Run in **shadow**
  (compare settled WS-derived logs vs authoritative getLogs) without trusting it yet.
- **Stage 4 — Swap + drop per-block getLogs.** Once the differential tests and shadow comparison hold,
  switch the kernel to the new graph and remove per-block `eth_getLogs` as the discovery mechanism.

> **STATUS (2026-07-06): the swap half of Stage 4 is COMPLETE.** The legacy graph, `FinalizedState`,
> `canonical_tip`, the reset paths, and the tip-targeted `GetPoolData` plumbing are deleted; the
> log-sourced graph is the kernel's sole chain-state authority (schedulers, finalization, lag
> metrics, optimization reads). Ordering deviation from the plan above: Stage 3 (multi-provider WS)
> was deliberately deferred — it hardens the streamed-log input but adds no correctness invariant.
> The Stage-2 differential proptest and the shadow-parity suite were deleted with the legacy graph,
> as planned.
>
> **STATUS (2026-07-09): Stage 4 is COMPLETE — per-block getLogs is dropped as the every-block
> driver (the WS-primary trust flip).** Prerequisites landed first: Stage 3's multi-provider WS
> fan-out + debounce, and the anchor-height `GetPoolData` seeding (Blockers 1b/1c). Post-flip:
> - **Tip:** the WS stream is primary. `Streamed` blocks are trusted and never re-fetched; the
>   per-block `GetBlockLogs` survives only as the rare-hole **backstop** — `Unknown` bloom-touching
>   canonical blocks deeper than `STREAM_SETTLE_DEPTH` below the head (hash-keyed, hence fork-proof
>   in the unfinalized region). The gate-inactive "fetch every block during warmup" discovery
>   channel is retired (discovery = topic-filtered WS stream + bootstrap range scan).
> - **Finalization:** authoritative verification moved here. A `FinalizedBlockObserved` whose fold
>   stalls on holes schedules number-ranged `GetLogsRange` requests
>   (`missing_complete_ranges_to`); the payload's `covered` hash set bounds which absent blocks may
>   be proven empty. The anchor advances on the next finality re-poll (stride-bounded).
> - **Trust metric:** every authoritative replace of a divergent `Streamed` set increments the
>   permanent per-chain `ws_miss` counter (gauge + view) — the shadow comparison of the original
>   Stage 3, made permanent instead of temporary.

---

## Non-goals / explicitly retained

- `eth_getLogs` is **not** deleted outright — it is retained as an **authoritative
  verification/backfill** path (e.g. finalization-range verification before committing finalized
  snapshots, and seeding newly-discovered un-baseable pools). The change is that it stops being the
  per-block steady-state driver.
- No change to the optimizer, the pool/token registries' semantics, or the multi-chain kernel surface
  beyond what the borrowed→owned reserve projection requires.
- This is not a rewrite of `kernel/mod.rs` wholesale — only the blocks graph is extracted and replaced.
