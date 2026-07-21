# Continuous Swap Cost Monitoring — Feature Specification

Date: 2026-07-20 (architecture and increments added 2026-07-21)  
Status: Validation target  
Working name: Pool Exitability Watch

## Summary

Continuously reassess the expected execution cost of a user-defined token swap and notify the user when that cost materially deteriorates or the assessment becomes unreliable.

The feature is intended for a DEX user who holds a liquidity-sensitive token and wants to monitor a possible exit without repeatedly requesting quotes manually.

The distinguishing behavior is continuous, amount-specific monitoring with block-specific evidence. It is not another generic liquidity dashboard, price alert, token safety score, or one-time swap quote.

## Problem

DEX users who hold positions in liquidity-sensitive tokens cannot continuously know when changing pool liquidity makes executing a specific swap materially more expensive.

Existing tools provide generic liquidity metrics, composite risk scores, price alerts, LP range alerts, or one-time quotes. They generally do not persistently monitor a user-defined swap amount and explain, with comparable block-specific evidence, when and why its execution conditions deteriorate.

## Evidence Status

This is a product hypothesis derived from broader market research, not an exact user problem directly proven by that research.

The research supports the broader need for trustworthy DEX risk information, reliable monitoring, executable-liquidity evidence, data freshness, and explainable alerts. It does not yet establish the prevalence of continuous position-sized exit-cost monitoring or willingness to pay for it.

The purpose of the validation build is to test those missing claims before investing in a complete routing, simulation, or trading product.

## Target User

The initial target is an active EVM DEX trader who:

- Holds a low- or medium-liquidity token.
- Has a real amount they may need to swap into a more liquid asset.
- Uses tools such as a DEX scanner, Uniswap, or an aggregator.
- Currently checks liquidity or requests quotes manually.
- Has an explicit tolerance for execution loss.

LP-specific monitoring is adjacent but not part of the initial user hypothesis. Out-of-range alerts, LP P&L, fee analysis, and automated rebalancing address a different decision and already have established competitors.

## User Outcome

The user should be able to answer:

> Can I still execute this swap within my acceptable cost, what changed since the previous assessment, and can I trust the current result?

The feature may help the user decide to:

- Investigate a liquidity change.
- Reduce exposure.
- Split an intended trade.
- Select a different route or venue.
- Wait for conditions to improve.
- Avoid increasing a position that is becoming difficult to exit.

The feature does not make the decision for the user and must not represent a pool, token, or route as safe.

## Core Concepts

### Monitored Swap

A hypothetical exact-input swap defined by:

- Chain.
- Input token.
- Output token.
- Exact input amount.
- Route mode and, for a fixed route, its ordered pools.
- Maximum acceptable execution loss.

The user does not sign or execute a transaction.

### Assessment

An immutable evaluation of a monitored swap against one identifiable chain state.

An assessment includes:

- Block number, block hash, and timestamp.
- Confirmation or finality status.
- Route and pool identities.
- Input and expected output amounts.
- Pre-swap reference price.
- Effective execution price.
- Execution loss.
- Pool fees when they can be separated reliably.
- Estimated gas, reported separately from execution loss.
- Post-swap price.
- Initialized ticks crossed.
- Per-pool state evidence.
- Data freshness and completeness.
- Calculation version.

All pool reads and quote calculations in one assessment must refer to the same block. If that guarantee cannot be met, the assessment is incomplete.

### Execution Loss

The primary monitored metric is the normalized loss between:

1. The output implied by the route's pre-swap marginal prices at the assessed block.
2. The output expected for the user's complete input amount.

This metric includes pool fees and price impact. Gas is initially displayed separately because reliable conversion into the output asset adds price and chain-specific assumptions.

Raw output changes alone must not be treated as liquidity deterioration because the market price of either token may have changed.

### Assessment Status

Every assessment has one explicit status:

- `Current`: complete and based on sufficiently recent canonical state.
- `Provisional`: complete but not yet confirmed to the configured confidence level.
- `Stale`: complete at its original block but too old for the current decision.
- `Incomplete`: one or more required pool reads or calculations failed.
- `Unsupported`: the route contains behavior the evaluator cannot model honestly.
- `Invalidated`: its block was removed from the canonical chain.

Unknown, stale, or unsupported evidence must never be interpreted as a safe result.

## Route Modes

### Single-Pool Route

The first validation mode monitors one swap through one specified pool.

It provides the simplest causal explanation and aligns with the current implementation's strongest pool-state capabilities.

The result is pool-local. It is not a claim about the best execution available from aggregators or other venues.

### Fixed Multi-Hop Route

A subsequent validation increment may monitor a user-specified path such as:

`TOKEN -> WETH -> USDC`

Requirements:

- Every hop uses a specified pool.
- Every hop is assessed at the same block.
- The output of each hop becomes the input of the next.
- The report shows aggregate and per-hop execution loss.
- A deterioration event identifies the hop or hops that contributed materially.
- Failure or unsupported behavior in any hop makes the route assessment incomplete or unsupported.

Fixed multi-hop monitoring remains explainable because the route identity does not change between assessments.

### Dynamic Best Route

Dynamic routing would recalculate the best route, potentially including split routes, on every assessment.

This is outside the validation scope because it adds:

- A pool and route discovery universe.
- Route churn between assessments.
- Gas-aware optimization.
- Split-route attribution.
- Cross-protocol semantics.
- Token transfer-tax behavior.
- Uniswap v4 hook behavior.
- MEV and execution-timing assumptions.

Dynamic routing should be considered only after users validate the core monitoring behavior.

## Functional Requirements

### FR1: Create a Monitor

The user can configure:

- One supported chain.
- One supported single-pool or fixed-route swap.
- Exact input amount in token units.
- Maximum acceptable execution loss.
- Notification destination.

The boundary validates token addresses, pool identities, route continuity, token decimals, amount, and threshold before creating the monitor.

No wallet connection, token approval, API key, or trading permission is required.

### FR2: Produce Comparable Assessments

The system periodically:

1. Selects an identifiable chain block.
2. Reads all required pool state at that block.
3. Evaluates the complete swap at that block.
4. Assigns an explicit assessment status.
5. Persists the immutable result.
6. Compares it only with a compatible previous assessment.

Assessments are compatible only when the chain, input, output, amount, fixed route, calculation version, and relevant configuration are equal.

### FR3: Explain the Current Result

The current report shows:

- Configured input and output.
- Expected output.
- Execution loss and user threshold.
- Estimated gas separately.
- Current status and assessment age.
- Assessed block and confirmation state.
- Route summary.
- Per-hop input, output, fee, price impact, ticks crossed, and post-swap price when available.
- Explicit unknown, stale, incomplete, and unsupported evidence.

### FR4: Detect a Meaningful Transition

The system derives a state transition rather than alerting on every metric change.

The primary transition is:

`Acceptable -> Degraded`

It occurs when execution loss rises above the configured threshold for the configured persistence period.

The implementation should support hysteresis or an equivalent recovery buffer so normal movement near the threshold does not repeatedly alternate between degraded and recovered states.

Underlying changes such as active-liquidity reduction or additional tick crossings are causal evidence for the transition, not independent alerts by default.

### FR5: Detect an Unreliable Assessment

The system derives:

`Assessable -> Unreliable`

when it can no longer produce a sufficiently fresh and complete assessment.

The event identifies the reason, including:

- Chain observation lag.
- Missing pool state.
- Quote or simulation failure.
- Unsupported pool behavior.
- Reorganization invalidation.

The last valid result remains visible with its original block and age, but must not be presented as current.

### FR6: Deliver One Causal Alert

An alert is emitted once per meaningful state transition, not once per observation.

An exitability-degradation alert contains:

- Monitored swap and amount.
- Previous and current execution loss.
- Configured threshold.
- Previous and current blocks.
- The principal observable causes.
- Current assessment confidence.
- A link or identifier for the complete evidence trace.

An unreliable-assessment alert contains:

- The reason assessment failed or became stale.
- The last valid result and its age.
- Whether monitoring has recovered.

Alert delivery status must be recorded separately from the market event so a delivery failure does not alter the underlying assessment history.

### FR7: Provide an Evidence Trace

Every transition can be inspected as a deterministic before-and-after comparison containing:

- Configuration identity.
- Previous and current assessment status.
- Block numbers and hashes.
- Quote inputs and outputs.
- Aggregate and per-hop changes.
- Pool-state changes used in the explanation.
- Missing or unsupported evidence.
- Calculation version.
- Explorer links where available.

The trace must not reduce incomplete evidence to a numeric trust or safety score.

### FR8: Show Minimal Token Evidence

Existing token examination may be shown as secondary context:

- Contract code presence.
- Decimals validation.
- Proxy detection.
- Verified-source evidence.
- Existing code-level warning reasons.

Token evidence is not part of the execution-loss calculation and must not be combined into a composite safety score.

Token, holder, creator, liquidity-lock, and honeypot monitoring are outside the initial alert set.

## Validation Scope

### Required

- Uniswap v3.
- One EVM chain selected based on participant concentration.
- Single-pool exact-input monitoring.
- User-defined input amount and execution-loss threshold.
- Same-block pool state and quote.
- Current, stale, incomplete, unsupported, and invalidated outcomes.
- Immutable assessment history.
- `Acceptable -> Degraded` detection.
- `Assessable -> Unreliable` detection.
- Before-and-after evidence trace.
- One notification channel.
- Read-only operation.

### Optional Second Increment

- Fixed routes of up to three Uniswap v3 pools.
- Per-hop degradation attribution.
- Recovery notifications.
- Minimal token evidence from the existing examiner.

### Excluded

- Dynamic best-route or split-route optimization.
- Swap execution.
- Wallet connection or portfolio discovery.
- Generic charts and technical indicators.
- Pool or token discovery.
- Composite risk scores.
- Holder-cluster or creator-wallet analysis.
- Liquidity-lock analysis.
- Honeypot or transfer-tax simulation.
- LP P&L, fee accounting, or range management.
- Automated rebalancing or exit.
- Predictions or trade recommendations.
- Cross-chain routes.
- Uniswap v4 hooks.

## Validation Hypotheses

### H1: Problem

Qualified users currently spend meaningful effort manually checking whether a real DEX position can be exited within their tolerance.

### H2: Amount-Specific Value

A quote based on the user's real amount is more useful than generic TVL or liquidity metrics.

### H3: Continuous Value

Users value being notified about deterioration before they decide to execute, rather than relying only on a quote requested at execution time.

### H4: Explainability

Before-and-after block evidence increases trust and helps users distinguish liquidity deterioration from ordinary market-price movement.

### H5: Retention

Users keep a real monitor active and inspect its transitions over a multi-week period.

### H6: Willingness to Pay

After experiencing a real or replayed deterioration event, some users commit to a paid pilot.

## Validation Method

### Concierge Phase

With qualified participants:

1. Observe their current exit-check workflow.
2. Collect one real pool, amount, and acceptable execution loss.
3. Produce a current assessment.
4. Present a recorded or historical before-and-after deterioration event.
5. Ask what decision the evidence would change.

Do not count general interest or positive design feedback as validation.

### Live Phase

Run real monitors for approximately two weeks and record:

- Real monitors created.
- Monitors retained through the trial.
- Alerts delivered and opened.
- Alerts cross-checked against another tool.
- Alerts judged useful or noisy.
- Investigations or decisions caused by an alert.
- Paid-pilot commitments.

Suggested evidence of promise from approximately twelve qualified participants:

- At least eight can provide a real pool and amount without inventing a hypothetical use case.
- At least six activate a monitor.
- At least four keep it active through the trial.
- At least three verify or act on an event.
- At least two commit to a paid pilot.

These are decision gates for an early qualitative test, not statistically representative market estimates.

## Kill or Pivot Conditions

The hypothesis should be rejected or narrowed if:

- A one-time aggregator or Uniswap quote satisfies the target users.
- Users care only about token price changes.
- Users do not have an explicit position-sized exit tolerance.
- Threshold crossings are too rare or too noisy to support monitoring.
- Users ignore freshness, block provenance, and causal evidence.
- The single-pool result is not useful because users always require dynamic aggregate routing.
- The target users primarily want LP accounting or out-of-range automation.
- Users will not keep real monitors active.
- No user commits to a paid pilot after experiencing the feature.

## Implementation Fit and Constraints

The existing implementation provides useful foundations:

- Canonical and reorganization-aware chain state.
- Chain progress and freshness signals.
- Uniswap v3 and v4 pool metadata and active state.
- Exact current-tick U256 replay.
- Token contract examination.
- Multi-pool route optimization structures.

Current limitations include:

- No complete initialized-tick and liquidity-net state for exact local multi-tick replay.
- No production gas-cost model.
- No complete MEV or execution-timing model.
- Synthetic cross-chain edges are not production execution semantics.
- Uniswap v4 hooks can add behavior not captured by standard concentrated-liquidity math.

For validation, Uniswap QuoterV2 can provide multi-tick single-pool and fixed-path quotes, including output, estimated gas, ticks crossed, and post-swap price. The current exact replay can cross-check quotes that remain within the current tick.

The long-term implementation should internalize additional execution logic only after the monitoring hypothesis is validated.

## Deployment and Architecture

The feature ships as a **desktop application**, not the current terminal (ratatui) interface. The target shape reuses the existing pure-core/effect/runtime architecture (see [Architecture](architecture.md)) rather than replacing it.

### Tiers

Three platform-agnostic pure crates are shared by every tier, with two thin runtimes over them:

- **`multi_chain_kernel` (existing, extended).** Owns the canonical, reorganization-aware, finality-anchored global pool state. It gains a public, read-only slice-query surface (see increment 1). The state is user-independent: it is canonical chain state and does not depend on which user is watching which swap.
- **`monitor-core` (new, pure).** The `Monitor`, `Assessment`, `AssessmentStatus`, and route types; the execution-loss and status assessment; the transition/hysteresis logic; and the evidence-trace diff. Location-agnostic so the same code can run on the server (always-on alerting) or in a client (interactive what-if).
- **`view-model` (new, pure).** A `project(state, monitors) -> ViewModels` function returning serializable, testable view data (monitor view, assessment view, evidence-trace view, chain-health view). Generalizes the existing `observe()` + `format_lines()` split.

- **Server runtime (the always-on agent).** The current single-process runtime minus the terminal view, plus an outbound query/stream API. It owns all live I/O — WebSocket fan-out across chains, RPC endpoint pools, pool discovery, the metadata cache, reorg-aware fold, finalized snapshots — and a **QuoterV2 proxy** endpoint. The pure core is unchanged; a serving adapter is added. This is the "background agent that keeps monitoring when the UI window is closed" from the architecture document.
- **Client runtime (the thin desktop shell).** A second, small `Application`/`Runtime` pair whose core is `monitor-core`. Its `Subscription` is "stream these pools from the server," its `Effect`s are query-slice / request-quote / notify-user / user commands, and the server's pushes are its `Event` inputs. It maintains no WebSocket or RPC connections and rebuilds no pool snapshots.

Each desktop platform (for example WPF over IPC, or an embedded UI) provides only a thin `ViewModel -> widgets` renderer and forwards user actions as `Command` inputs. All view derivation and formatting lives in `view-model` and is unit- and property-tested; the current ratatui `View` becomes one such thin renderer.

### Server-side monitoring rationale

Live chain monitoring moves server-side so that clients do not each run multi-chain WebSocket/RPC fan-out and rebuild pool snapshots. Because the pool state is global, running one shared monitor instead of one per client removes the multiplied provider rate-limit pressure the current single-process runtime already encounters, and lets the server answer cheap, same-block-consistent slice queries. Multi-tick quotes require a node call (QuoterV2), so the server also exposes a quote endpoint; raw slice queries alone are not sufficient for a faithful assessment.

### Consistency and open decision

- **Same-block consistency over the wire.** A slice response is coherent at one block hash and carries its confirmation/finality state. On reorganization the server pushes an invalidation so the client marks affected assessments `Invalidated` (FR5). The block graph already detects this internally; the work is surfacing it in the API contract.
- **Where assessment and alerting run (decision required before build).** Because `monitor-core` is pure, it can run on the client or the server. Continuous alerting while the desktop app is closed requires the always-on server to own each user's monitors and run assessment plus alerting, with the client as a near-pure view. Interactive, per-keystroke what-if assessment argues for running the same logic client-side against fetched slices. The recommended default is server-side live monitoring and alerting, with slice and quote endpoints still available for local what-if. This is a product fork (offline alerting versus server simplicity) to settle before implementation.

## Domain and Test Guidance

If implementation is authorized:

- Model assessment status and unsupported states with explicit algebraic data types.
- Keep quote normalization, compatibility checks, snapshot comparison, hysteresis, and event derivation pure and deterministic.
- Keep RPC, time, persistence, and notification delivery behind thin adapters.
- Write tests before production logic.
- Prefer property-based tests for comparison and transition invariants.
- Ensure an unknown, incomplete, stale, or invalidated result can never produce an acceptable or safe conclusion.
- Ensure identical compatible assessments produce no transition.
- Ensure one market transition produces at most one user-facing alert regardless of delivery retries.
- Avoid panics on all runtime paths.

## Implementation Increments

If implementation is authorized, build in the following order. Each increment is small and independently landable, tests are written before production logic, and stubs establish the compile-time contracts first. Increments 1–7 deliver the single-pool validation scope; multi-hop (the optional second increment above) and the client/server split follow.

1. **Public slice-query surface on `kernel::State`.** Read-only accessors — pool state at a block, canonical and finalized heads, verified metadata, and reorganization/invalidation status. This is the single enabling change: it is both what the assessment function reads and what the server's slice API serves. Cover with tests before exposing.
2. **`monitor-core` skeleton.** `Monitor`, `Assessment`, `AssessmentStatus`, and route types as ADT stubs that make invalid states unrepresentable; contracts compile before behavior exists.
3. **Execution-loss and status assessment.** A pure function over a same-block pool slice, initially using the existing within-current-tick exact replay and flagging `Unsupported` or lower fidelity when a swap leaves the current tick (`hit_tick_limit`). Property tests: identical compatible slices produce identical assessments; incomplete, stale, or invalidated evidence never yields an acceptable result.
4. **Transition and hysteresis.** `Acceptable -> Degraded` and `Assessable -> Unreliable` over assessment history, with a recovery buffer. Property test: one market transition yields at most one alert regardless of delivery retries.
5. **Evidence-trace diff.** A pure before-and-after comparison of two assessments; no reduction to a numeric trust or safety score.
6. **Quote effect and QuoterV2 adapter.** `Effect::FetchQuote` plus `Event::QuoteReceived` and a server-side QuoterV2 adapter for multi-tick output, gas, ticks crossed, and post-swap price. The within-tick exact replay is retained as a cross-check.
7. **Notification effect and delivery log.** `Effect::Notify` plus delivery events, recorded in a delivery log kept separate from assessment history so a delivery failure never mutates the market record (FR6).
8. **`view-model` crate and `View` refactor.** Extract the pure projection layer and refactor the existing terminal `View` to consume it; existing `format_lines` tests are the behavior-preserving regression net.
9. **Client/server boundary.** Serializable slice, stream, and quote contracts; a server serving adapter over the unchanged core; and a client remote-subscription adapter that replaces direct RPC/WebSocket with server pushes.

## Related Documents

- [Crypto Trading Software User Needs Validation](market-research-user-needs-validation.md)
- [Initial Market Research](market-research-initial.md)
- [Architecture](architecture.md)
- [Uniswap V3 Pool Events](uniswap-v3-pool-events.md)
