# Armchair Analyst Architecture

Armchair Analyst should be organized around a pure deterministic logic core surrounded by one or more impure runtimes.

The core idea is to keep domain behavior testable, replayable, and portable by making the core a state machine with no side effects. Everything outside that core should be treated as unreliable and potentially unpredictable.

## Architecture Principles

- The core is pure, deterministic logic.
- The core is modeled as a state machine.
- The core should be extensively testable, including with property tests.
- The core does not perform I/O, make network requests, read clocks, generate randomness, spawn tasks, or run any other non-deterministic computation.
- The core does not return executable impure logic, callbacks, closures, or runtime-specific instructions.
- When the core needs an external action to happen, it returns a declarative effect value describing that action.
- Every effect returned by the core is executed outside the core by a runtime.
- All external inputs are considered unreliable until validated or handled by the core.

## Pure Core

The Rust core owns the deterministic state transition logic. Given the same starting state and the same ordered inputs, it should always produce the same resulting state and the same declared effects.

The core should own:

- Domain state for monitored markets, venues, subscriptions, aggregates, and analysis results.
- Normalized data models for market, trade, order book, liquidity, and chain-derived data.
- Aggregation logic that reconciles data from many venues.
- Analysis routines that operate over live and historical data.
- Validation and interpretation of incoming events or commands.
- Declarative effect definitions.
- Subscription derivation.
- A boundary API suitable for consumption by runtimes and thin UI shells.

The core should not own:

- Exchange connector implementations.
- WebSocket clients.
- HTTP clients.
- Filesystem access.
- Database drivers.
- Timers, clocks, randomness, or thread scheduling.
- UI framework logic.
- OS-specific behavior.

## Effects

Effects are plain data structures returned by the core to describe work that must happen outside the deterministic boundary.

Examples include:

- Sending an HTTP or WebSocket request.
- Reading or writing persisted data.
- Requesting the current time.
- Performing non-deterministic or environment-dependent computation.
- Emitting telemetry or logs.
- Notifying a UI shell about work that should be presented to the user.

Effect values should be declarative and serializable where practical. They should describe what needs to happen, not how a specific runtime must implement it.

## Runtime

A runtime is an impure effect execution engine. It receives declarative effects from the core, executes them, and feeds the resulting data back into the core as new inputs.

There can be many runtimes for the same set of supported effects, including:

- A production runtime hosted by the Windows background agent.
- A mock runtime for tests.
- A deterministic replay runtime for debugging.
- A tracing runtime for inspecting effect flow.
- A portable runtime for another operating system or host environment.

Each effect implementation should be as simple as possible. The goal is to minimize the amount of code that performs impure work, because that code cannot be tested as comprehensively as the pure core.

## Inputs

New data enters the core as commands or events.

Inputs can be produced by:

- Long-running subscription processes, such as exchange WebSocket monitors.
- Polling workers.
- Effect execution results reported as events.
- User actions from a thin UI shell.
- Startup or shutdown lifecycle events from a runtime.

The core should handle inputs defensively. External systems can send malformed data, return stale data, produce duplicate messages, reorder messages, fail intermittently, or stop responding.

## Subscriptions

Exchange monitoring is a continuous background workload. Long-running subscription processes live outside the core because they require I/O, scheduling, network clients, reconnection logic, and other impure behavior.

The core describes desired long-running external data with `Subscription` values derived from state. A runtime reconciles the desired subscription set with active impure subscriptions and later reports subscription updates, failures, reconnections, or completions back to the core as events.

```text
State -> SubscriptionSet
```

## Initial System Shape

The first implementation should use:

- A Rust core for deterministic domain logic.
- A Windows background agent that is the production application host and owns the production runtime and core state while the user is logged in.
- A Windows desktop shell implemented with WPF as an optional thin client.
- An IPC boundary between the WPF shell and the background agent.
- A Windows installer or package that installs the app, registers the background agent startup behavior, registers notifications, and provides uninstall entries.
- Automatic updates handled outside the core by the Windows host, installer, package, and updater layer.
- Windows notifications emitted by the impure host layer in response to explicit effects or derived state.
- One or more additional runtimes for tests, replay, tracing, and future portability.
- A thin UI boundary that presents monitored markets, venues, and analysis output while delegating business rules and heavy computation to the Rust core through the background agent.

The architecture should preserve room for additional OS-specific shells without forcing the Rust core to depend on a specific UI framework.

## Windows Host

The initial Windows host should be organized as:

```text
WPF UI process
  -> sends Command values over IPC
  -> reads projection outputs over IPC
  -> may be opened or closed without owning production State

Background agent process
  -> owns State
  -> calls transition
  -> executes Effect values
  -> reconciles subscriptions(State)
  -> receives exchange, chain, timer, persistence, and system data
  -> feeds Event values back into the core
  -> emits Windows notifications when requested or appropriate

Installer or package
  -> installs app files
  -> registers the background agent startup behavior
  -> registers automatic update behavior
  -> registers notification identity and activation
  -> creates shortcuts and uninstall entries
```

The WPF process should not be the owner of the production core state. Closing the main window should not stop monitoring. The background agent should continue running in the user's logged-in session, maintain stream subscriptions, execute effects, persist data needed for recovery or replay, and provide notifications.

Automatic update behavior belongs outside the deterministic core. The Windows host, installer, package, and updater layer should install updates, coordinate shutdown and restart of the UI and background agent, and report update-related facts to the core only when those facts matter to product behavior. The background agent may cooperate by quiescing subscriptions, flushing durable state, and exiting on updater request.

Windows Services are out of scope for the initial implementation. Because the initial background agent is per-user, monitoring stops when the user logs out and resumes when the user logs back in. A Windows Service may be considered later if the product needs continuous monitoring before user login, after user logoff, or across multiple user sessions. That future design would add a machine-level service plus a per-user UI or notification companion, because service-hosted work should remain separate from interactive UI concerns.
