# Optimizer evolution roadmap: from closed-path arbitrage to risk-aware portfolio optimization

> **Status: loose exploration / strategic roadmap — NOT an authoritative spec.**
> This captures a 15-step direction for growing the optimizer from atomic constant-product
> arbitrage into an inventory- and derivative-aware portfolio optimizer. Each step is a
> candidate; the phase gates below (especially the **Step 4 closed→open pivot**) are strategic
> decisions that need explicit sign-off before implementation. Steps are cleaned up and
> annotated with compatibility notes against the current codebase. Where a step conflicts with
> how the optimizer actually works today, the conflict is called out in a **⚠ note** rather than
> silently accepted.

## Where the code is today (baseline)

- `optimization` crate: a **differentiable, vectorized** model (`burn` autodiff tensors,
  column-wise `softmax` routing weights, `Adam`) folded over a single **pool-layout matrix**
  (`ModelLayout`: rows = outputs, columns = input tokens; `access_reserve!` for safe indexing).
- **Constant-product** pool math, **closed-path** objective: init asset in → maximize init asset
  out, with a bridge/bypass mask for multi-hop cycles.
- Grow-only `Model::reconcile` (append-only layout; new pools cold-started at `COLD_WEIGHT`,
  removed pools masked) already landed — see `[[grow-only-reconcile-landed]]`.
- Execution: `run_optimization_step_with_reserves` runs Adam chunks per reserve snapshot and emits
  `OptimizationStepResult`.

### Guiding principle the roadmap must respect

The optimizer's power comes from being **differentiable and batched over the whole pool set at
once**. Any abstraction added below must stay expressible as **structured tensor channels**, not
runtime-dispatched scalar objects. Two steps as originally written (Step 1's `trait PoolModel`,
Step 7's `enum Action`) violate this if taken literally — they are reframed in-place with a ⚠ note.
Method stays as in `AGENTS.md`: test-first, property tests by default, panic-free prod paths,
minimal surgical changes.

### Callout legend

- **⚠ Compatibility / issue** — where the idea fights the current design, adds a new external
  dependency, or is mis-ordered.
- **Adopt-now** — valuable independent of the strategic pivot; can land against today's closed-path
  system.

---

## Phase 0 — Lock the baseline & get observability (adopt-now, low risk)

### Step 0 — Freeze the constant-product router as a golden baseline

**Goal:** every later change is comparable against today's system.

Keep the current model exactly as-is (token-swap tensor layers, constant-product quote, weight
optimization, closed-path objective, final-output maximization) and pin it with a small
deterministic golden universe:

- Tokens: `A, B, C`
- Pools: `A/B`, `B/C`, `C/A`, `A/C`

Save expected outputs for: one-hop route, two-hop route, closed cycle, split route,
no-arbitrage market, obvious-arbitrage market. From here on, every refactor must reproduce these
in **legacy closed-route mode**.

> **Adopt-now.** We already have `layer_forward_matches_scalar_reference` (scalar oracle) and
> `test_model_v4_arbitrage`. This adds a *named, human-readable* golden fixture on top — cheap and
> worth it as the regression anchor for the whole roadmap.

### Step 2 — Explicit action logging on the current optimizer

**Goal:** make every optimized solution inspectable.

Per optimized run, record per allocated flow: `layer, token_in, token_out, pool_id, amount_in,
amount_out, weight`. Aggregate: `pool_inputs[pool_id]`, `token_flows[token]`, `route_entropy`,
`effective_number_of_pools`.

**Verify:**
- Per layer: `Σ outgoing allocated amount ≤ available amount`.
- Per pool: `logged amount_out = f(logged amount_in)` (matches the pool quote).
- Closed routes: `profit = final_start_token − initial_start_token − gas`.

> **Adopt-now.** This is a pure read-off from the softmax weights and reserves — no objective
> change, high debugging value, and it feeds Step 12's candidate extraction. Recommend doing this
> early. (Renumbered before Step 3 because Step 3 depends on it.)

### Step 3 — Surface diagnostics around the current optimizer

**Goal:** "surface explorer" insight without changing the objective.

For each input size `aᵢ`, run the existing optimizer to get `Wᵢ* = argmax_W P(W; aᵢ, θ)` and
record: optimized profit, pool-input vector, route entropy, route concentration, marginal profit
(finite difference), route-change distance between neighboring sizes, reserve sensitivity,
robustness under perturbation.

**Verify** with synthetic cases (one shallow profitable pool + one deep less-profitable pool):
small input uses the shallow pool; larger input splits; route-change score spikes at the
transition; slippage contribution identifies the shallow pool.

> **⚠ Cost.** Re-optimizing per input size is an outer sweep over the (already dominant) chunk
> loop — see the per-event fold cost in `[[fourth-run-merged-refold-dominates]]`. Keep this as an
> offline/diagnostic tool, not something on the live hot path.

---

## Phase 1 — The strategic pivot: closed arbitrage → open-path portfolio value

> **Gate.** Step 4 changes the system's risk posture from **atomic/riskless closed cycles** to
> **inventory-bearing open paths**. That is a product decision, not a refactor. It also introduces
> a hard new dependency (a reference-price oracle). Do not start Phase 1 without explicit sign-off,
> and land Steps 5–6 in the same phase — an open-path optimizer without exit-cost modelling
> (Step 6) will systematically overvalue illiquid leftovers.

### Step 1 — A pool-model abstraction (reframed)

**Goal:** separate routing logic from pool math so non-constant-product pools (Step 13) slot in
without touching the optimizer.

Original sketch:

```rust
trait PoolModel {
    fn quote(&self, token_in: Token, token_out: Token, amount_in: Amount) -> Amount;
    fn apply_swap(&self, token_in: Token, token_out: Token, amount_in: Amount) -> PoolState;
}
```

First implementation stays constant-product:

```
Δy = (y · γ · Δx) / (x + γ · Δx)
```

**Verify (unit):** zero in → zero out; output positive; output < output reserve; monotone in size;
marginal output decreases with size; fee reduces output; `apply_swap` updates reserves correctly.
Compare `quote` against the current implementation within tolerance.

> **⚠ Compatibility — this is the biggest design clash.** A scalar, `dyn`-dispatched
> `PoolModel::quote(...) -> Amount` **cannot be backpropagated through** and cannot be batched, so
> it breaks the differentiable/vectorized optimizer. The abstraction we actually want is a
> **per-protocol vectorized quote curve** that contributes to the tensor forward pass — i.e. each
> protocol (v2 / weighted / stable / v3-sampled) supplies the tensor ops for its slice of the
> layout, matching today's `access_reserve!` + Uniswap-v4 seam (`[[uniswap-v4-staged-rollout]]`).
> Keep the scalar `quote`/`apply_swap` **only** for the exact-replay path (Step 12) and tests, not
> for the differentiable core. Treat "extract a PoolModel" as "define the protocol-curve trait over
> tensors," and Step 13 depends on getting this right.

### Step 4 — Replace "final token amount" with a portfolio value function

**Goal:** allow open paths (end in any token) without yet adding derivatives.

Portfolio state and valuation:

```rust
struct PortfolioState { spot: HashMap<Token, Amount> }
```

```
V(s) = Σ_i π_i · h_i          // h_i = amount of token i, π_i = reference price in numéraire (e.g. USDC)
```

Objective becomes:

```
max_W [ V(s_T) − V(s_0) − gas − fees ]
```

Closed arbitrage is the special case where only the start token has nonzero terminal value.

**Verify:**
1. **Closed-mode equivalence** — with `π_A = 1` and all other terminal prices 0, the new optimizer
   must match the old final-token optimizer (ties back to the Step 0 golden fixture).
2. **No fake profit** — if all pools match reference prices and charge fees, no positive value.
3. **Open-path opportunity** — if `A/B` sells `B` cheaply vs `π_B`, the optimizer stops in `B`
   instead of forcing `B→A`.

> **⚠ New dependency + circularity.** This needs a **reference-price oracle** (`π_i`) we do not
> have. If `π_i` is derived from the same AMMs being traded, "no fake profit" collapses into a
> tautology and open-path edges become measurement noise. Decide the numéraire and an *independent*
> price source (external oracle / venue mid) before building this. This is a subsystem, not a
> function.

### Step 5 — Inventory constraints and penalties

**Goal:** stop the optimizer "profiting" by accumulating risky tokens.

Soft penalty (differentiable, fits the optimizer):

```
inventory_penalty = Σ_i λ_i · (h_i − h_i^target)²      // larger λ_i for risky/illiquid tokens
```

Hard limits: `max_position_per_token`, `max_notional_per_token`, `allowed_inventory_tokens`,
`min_exit_liquidity`.

**Verify:** a cheap illiquid token with huge apparent edge is rejected under low max exposure; a
blue-chip within target inventory is accepted; raising `λ_i` reduces allocation into token `i`;
setting max exposure to zero recovers closed-route behavior.

> **Note.** The quadratic penalty is differentiable and clean; the hard limits are a
> constraint/clamping layer applied post-optimization (or via projected gradient) — keep them out
> of the autodiff graph.

### Step 6 — Exact exit-cost / liquidation-value estimation

**Goal:** don't value tokens at reference price if exiting them is expensive.

For each non-numéraire token estimate liquidation value `L_i(q)` from available pools, then value
conservatively:

```
V_i(q) = min(π_i · q, L_i(q))            // or  V_i(q) = π_i · q − estimated_exit_cost(q)
```

**Verify:** high reference price but no exit liquidity → heavily discounted; deep USDC liquidity →
value near reference; larger inventory → worse liquidation discount (slippage).

> **Note.** This reuses the AMM quote engine to price exits — a good correction to Step 4's naive
> valuation, and it **must** ship with Step 4, not after. Cost: a differentiable liquidation quote
> per held token per evaluation.

### Step 7 — Generalize layers to portfolio-state transitions (reframed)

**Goal:** make a swap just one kind of action, so perps/options can be added later.

Each layer holds a full state; each action is a transition `s_{t+1} = T_a(s_t)`:

```rust
struct State { spot: Vec<Amount>, derivative_positions: Vec<Position>, collateral: Vec<Amount> }
enum Action { SpotSwap { pool_id, token_in, token_out, amount_fraction } }
```

**Verify:** all old tests pass in the new state-transition engine — same closed-mode outputs as the
old router, same open-path behavior as Step 4, no negative balances, transitions compose across
layers.

> **⚠ Compatibility — keep it tensorized.** A runtime `enum Action` dispatched per step reintroduces
> the differentiability problem. Model the generalized state as **fixed tensor channels** (spot
> channel now; derivative/collateral channels appended later as zero-width until Phases 2–3), and
> each "action type" as a differentiable sub-forward selected by mask/weight — not a `match` in the
> hot loop. This is the load-bearing architectural step for everything after it; do it only once
> Phase 1 (Steps 4–6) is stable so two hard problems aren't mixed.

---

## Phase 2 — Robustness, execution safety, and multi-protocol (much of it adopt-now)

### Step 8 — Robustness analysis as a first-class score

**Goal:** rank opportunities by reliability, not just point expected value.

For an optimized `W*`, sample perturbed states `θ₁..θₙ` (reserve perturbation, competing pre-trade,
gas increase, reference-price shift, exit-liquidity worsening), evaluate the fixed action sequence
under each: `pᵢ = V(T_{W*}(s₀; θᵢ)) − V(s₀)`, and record `mean/median/p5/prob_positive/worst_case`.
Score conservatively:

```
score = P5[p]            // or  score = E[p] − λ · std(p)
```

**Verify:** deep pools score higher than shallow; single-shallow-pool opportunities have bad p5;
larger perturbations monotonically reduce robustness; closed atomic cycles are usually less
inventory-risky than open paths.

> **Adopt-now (partially).** The perturbation-ranking outer loop works against **today's**
> closed-path system already and is one of the highest-value additions for live safety.
> **⚠ Cost:** it is an `N×` re-evaluation on top of the chunk loop — budget it (small `N`, cache the
> forward, or run it only on the top-k candidates).

### Step 12 — Candidate extraction and exact replay

**Goal:** turn soft optimized weights into trustworthy executable actions.

From the continuous fractions: drop tiny flows, merge same-pool flows, dependency-sort actions,
simulate exact execution, estimate gas, check constraints. Then exact replay
`s_T^exact = T_n(…T_2(T_1(s_0)))` with exact pool math (and exact derivative accounting later).
Compare `predicted_value`, `exact_replay_value`, `approximation_error`, `risk_score`.

**Verify:** exact replay matches the differentiable prediction within tolerance for
constant-product pools; tiny-flow pruning barely changes value; if exact replay is below threshold
the candidate is rejected; no candidate spends more than available balances/margin.

> **Adopt-now — pull this forward.** Bridging soft softmax weights to a discrete, exactly-simulated,
> gas-checked action list is required before **any** live use of even the current closed-path
> optimizer. Recommend implementing this right after Step 2, well before the Phase 1 pivot. It is
> mis-numbered here at 12.

### Step 13 — Non-constant-product pools behind the same interface

**Goal:** expand beyond constant-product without changing the optimizer.

Add weighted pools, StableSwap-like, Uniswap-V3-like piecewise liquidity, and sampled quote-curve
pools. For awkward math: sample the quote curve → fit a monotone spline surrogate → optimize on the
surrogate → validate with exact quote / `eth_call` / simulation (Step 12).

**Verify per pool type:** quote monotonicity; zero-in → zero-out; marginal output behaves as
expected; surrogate error bounded on the sampled grid; exact replay rejects bad surrogate
candidates.

> **Note.** Fits our existing multi-protocol direction (`[[uniswap-v4-staged-rollout]]`,
> `[[uniswap-v4-subgraph-ids]]`) but **depends on Step 1 done right** (vectorized protocol curves,
> not a scalar trait). The monotone-spline surrogate is a sound way to keep v3 concentrated
> liquidity inside a differentiable framework.

---

## Phase 3 — Derivatives (large new domains; gated, sequenced last)

> **Gate.** Each of these is a new data-integration and modelling project (venue feeds, funding,
> mark prices, vol surface). Only proceed after Phases 1–2 are stable and the state engine (Step 7)
> is tensorized. Sequencing is deliberate: perps before options.

### Step 9 — Perp positions as synthetic exposure actions

**Goal:** let the optimizer hedge open spot inventory.

```rust
struct PerpPosition { underlying: Token, quantity: f64, entry_price: f64,
                      collateral_token: Token, collateral_amount: Amount }
// Actions: OpenPerpLong { market, notional }, OpenPerpShort { market, notional }, ClosePerp { market, quantity }
```

Linear perp: `PnL = q · (S − S₀)`; portfolio value gains
`V_perp = collateral + q·(S − S₀) − fees − funding`.

**Verify:** long delta (+$1 price → +q value); short delta (+$1 price → −q value); `q` spot + `q`
short ≈ zero delta; open-then-close immediately loses fees. Enables `USDC→ETH` + short-ETH-perp
instead of forcing `ETH→USDC`.

### Step 10 — Delta, margin, and liquidation constraints

**Goal:** stop "free money" via hidden leverage.

Net delta per underlying: `Δ_u = h_u − Σ_k ∂V_k/∂S_u` (linear perp: `∂V/∂S_u = q`). Constrain
`|Δ_u| < ε_u` or penalize `λ_Δ · Σ_u Δ_u² · σ_u²`. Add `available_margin ≥ maintenance_margin` and
`liquidation_distance ≥ minimum_distance`.

**Verify:** spot ETH + equal short ≈ zero ETH delta; unhedged ETH has positive delta; more leverage
→ higher liquidation-risk penalty; margin-violating trades rejected; high delta penalty → optimizer
prefers the hedged version.

### Step 11 — Funding and basis as edge components

**Goal:** model spot/perp carry.

Funding over holding period `τ`: `funding_PnL = −q · S · r_f · τ` (sign convention is venue-specific
— be explicit). Objective:
`ΔV = spot_edge + basis_edge + expected_funding − fees − risk_penalties`.

**Verify:** if shorts receive funding, long-spot + short-perp becomes more attractive; if shorts pay
funding, less; zero funding reduces to basis only; longer holding period magnifies funding.

### Step 14 — Options (only after perps are stable)

**Goal:** nonlinear payoffs.

```rust
struct OptionPosition { underlying: Token, strike: f64, expiry: Timestamp,
                        kind: CallOrPut, quantity: f64, premium_paid: f64 }
```

Valuation via venue mark / Black-Scholes-like model / implied-vol surface / scenario payoff. Track
Greeks `Δ, Γ, ν, Θ`; penalize `λ_Δ·Δ² + λ_Γ·Γ² + λ_ν·ν²`.

**Verify:** call payoff `max(S−K,0)`; put `max(K−S,0)`; long call positive delta; long put negative
delta; long option positive gamma; time decay under the model. Do **not** start before spot + perps
work.

---

## Phase 4 — Validation

### Step 15 — Backtesting harness

**Goal:** prove the full system improves *decisions*, not just marked profit.

Modes: **A** closed arbitrage (baseline) · **B** open-path portfolio value · **C** B + inventory
penalty · **D** C + perp hedging · **E** full robustness-ranked optimizer. Compare total return, max
drawdown, inventory exposure, turnover, gas, failed trades, profit variance, time in risky
inventory, liquidation proximity.

**Verify:** the new system improves **risk-adjusted** outcomes, not raw profit. Healthy signature:
closed-only lower but stable; open-path higher but volatile; +inventory lower volatility; +perp
hedge better risk-adjusted. If open-path profit disappears under conservative exit-cost modelling,
that is itself a useful result.

> **Adopt-now (Mode A).** Build the harness early with Mode A against the current system; it becomes
> the scoreboard that justifies (or kills) each later phase.

---

## Cross-cutting findings, risks & open decisions

1. **This is a product pivot, not a refactor.** The end state (open-path, inventory, perps, options)
   is a different, inventory-bearing risk product from today's atomic arbitrage. Commit to it as
   phased go/no-go, not one plan. The pivot point is **Step 4** and needs explicit sign-off.
2. **Two steps fight the differentiable/vectorized core** and are reframed in place: Step 1's scalar
   `trait PoolModel::quote` and Step 7's runtime `enum Action`. Keep the differentiable core as
   structured tensor channels; keep scalar/dispatch forms only for exact replay and tests.
3. **Undeclared external dependencies:** a reference-price oracle (Step 4, with a circularity trap
   if sourced from the same AMMs), perp venue + funding feeds (Steps 9–11), options vol surface
   (Step 14). Each is its own integration project.
4. **Ordering fixes applied:** Step 2 (logging) → before Step 3; **Step 12 (exact replay) pulled
   forward** as a pre-live-use requirement; Step 6 (exit cost) must ship with Step 4; Step 15 Mode A
   can start now.
5. **Performance budget ignored by the source.** Steps 4, 6, and 8 each multiply per-evaluation cost
   (portfolio valuation, per-token liquidation quotes, `N`-sample robustness) on top of a chunk loop
   that already dominates CPU (`[[fourth-run-merged-refold-dominates]]`,
   `[[third-run-fold-dominates]]`). Every new per-evaluation term needs a cost bound.
6. **Highest-value, lowest-risk, adopt now (independent of the pivot):** Step 0 (golden fixture),
   Step 2 (flow logging), Step 12 (candidate extraction + exact replay), Step 8 (robustness ranking
   on the current system), Step 15 Mode A (baseline backtest).
7. **Still respect `AGENTS.md`:** test-first, property tests by default, panic-free prod paths,
   minimal surgical changes, no derived state, make invalid states unrepresentable.
