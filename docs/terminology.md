# Terminology

This document defines the shared vocabulary for the core architecture.

## Core

The core is the pure deterministic Rust domain engine. It owns state transitions, validation, aggregation, analysis, effect declarations, subscription derivation, and projections.

The core performs no side effects.

## State

`State` is the complete deterministic domain state owned by the core.

Given the same initial `State` and the same ordered `Input`s, the core must always produce the same resulting states and effects.

## Input

`Input` is anything entering the core.

```rust
enum Input {
    Command(Command),
    Event(Event),
}
```

## Command

A `Command` is a request or intent for the system to do something.

Examples include:

- Start monitoring a market.
- Stop monitoring a market.
- Change analysis settings.
- Request historical backfill.
- Acknowledge an alert.

Commands can come from the UI, host application, startup flow, tests, or runtime control paths. Commands are not guaranteed to be valid.

## Event

An `Event` is a reported fact that supposedly happened.

Examples include:

- A trade was received.
- An order book delta was received.
- An HTTP response was received.
- A persistence write completed.
- A subscription connected.
- A subscription disconnected.
- A timer tick was received.

Events can come from runtime effect execution, subscriptions, external services, UI callbacks, tests, or replay. Events are not trusted blindly.

## Transition

`Transition` is the pure result of applying one `Input` to one `State`.

```rust
struct Transition {
    state: State,
    effects: Vec<Effect>,
}
```

## Transition Function

The transition function is the primary state machine function.

```rust
fn transition(state: State, input: Input) -> Transition;
```

The transition function should be total:

- No panics.
- No side effects.
- Prefer no `Result` return.
- Handle invalid commands, malformed events, stale data, duplicates, and impossible external claims as ordinary inputs.

## Effect

An `Effect` is a declarative one-shot request for impure work outside the core.

Examples include:

- Send an HTTP request.
- Persist data.
- Load a snapshot.
- Request the current time.
- Emit telemetry.
- Show an OS notification.

Effects are data, not executable logic.

## Runtime

The runtime is the impure execution environment outside the core.

It executes effects, manages subscriptions, handles I/O, talks to exchanges, persists data, receives UI and system callbacks, and feeds `Input`s back into the core.

## Background Agent

The background agent is the long-running per-user Windows process that hosts the production runtime. It is the real production application host for the initial Windows architecture.

It owns the production `State` while the user is logged in, calls the transition function, executes effects, reconciles subscriptions, receives external data, feeds events back into the core, persists data needed for recovery or replay, and coordinates notifications.

The background agent continues running when the main WPF window is closed. It should be single-instance per user.

Because the initial background agent is not a Windows Service, monitoring should stop when the user logs out and resume when the user logs back in.

## WPF UI

The WPF UI is the initial Windows user interaction layer.

It is a thin client of the background agent. It sends commands, reads projection outputs, and presents user-facing controls and views. It should not own the production `State`, run exchange monitoring, or contain domain logic.

The WPF UI is not required for background monitoring to continue while the user remains logged in.

## Installer Or Package

The installer or package is the Windows deployment host.

It installs app files, registers the per-user background agent startup behavior, registers the update mechanism, registers notification identity and activation, creates shortcuts, and provides uninstall behavior.

Automatic update implementation belongs to the installer, package, updater, and host layer, not to the pure core.

## Subscription

A `Subscription` is a declarative description of long-running external data the core currently wants.

Examples include:

- Binance trades for BTC/USDT.
- Coinbase order book updates for ETH/USD.
- Ethereum new block headers.
- One-second timer ticks.

Subscriptions are derived from `State`, not returned by `transition`.

```rust
fn subscriptions(state: &State) -> SubscriptionSet;
```

The runtime reconciles the desired `SubscriptionSet` with whatever subscriptions it is currently running.

## Projection

A projection is not a single core type. It is a concept and naming convention for pure read functions over `State`.

A projection has this general shape:

```rust
fn some_projection(state: &State) -> SomeReadModel;
```

Examples include:

```rust
fn user_interface_projection(state: &State) -> UserInterfaceModel;

fn monitored_markets_projection(state: &State) -> Vec<MonitoredMarket>;

fn active_alerts_projection(state: &State) -> ActiveAlertsView;

fn runtime_status_projection(state: &State) -> RuntimeStatusView;
```

Projection functions:

- Are pure.
- Read from `State`.
- May return any useful type or slice.
- Do not perform side effects.
- Do not mutate state.
- May reshape, filter, summarize, or denormalize state for a specific consumer.
- Can have many different output shapes over the same state.

## Boundary API

The core boundary is centered on one transition function and multiple pure derivation functions.

```rust
fn transition(state: State, input: Input) -> Transition;

fn subscriptions(state: &State) -> SubscriptionSet;

fn user_interface_projection(state: &State) -> UserInterfaceModel;
```

In summary:

```text
State + Input -> State + Effects
State -> SubscriptionSet
State -> arbitrary projection outputs
```

## Windows Service

A Windows Service is not part of the initial architecture.

It is a possible future host for machine-level background work that must continue before user login, after user logoff, or across multiple user sessions.

If added later, a Windows Service should remain outside the pure core and should not directly own interactive UI behavior. User-facing UI and notifications would still need a per-user companion process.
