# Distribution

Production Windows distribution should provide automatic updates and register the background agent that hosts the application while the user is logged in.

The initial distribution path should be a minimal proof of concept with no WPF UI. That POC should still preserve the final process boundary so a WPF UI can be added later as a clean client.

## Preferred Direction

Use Velopack for the first install and update implementation.

Velopack is a good fit for the initial Windows host because it provides:

- A Windows installer.
- Automatic updates.
- Application lifecycle hooks for install, update, and uninstall.
- A straightforward static-file release feed.
- Enough flexibility to coordinate a continuously running per-user background agent.

MSIX remains a possible future packaging option if native Windows package identity, Store compatibility, or stricter Windows integration becomes more important than installer flexibility.

## Process Layout

The Velopack package should contain:

```text
ArmchairLauncher.exe
ArmchairAgent.exe
Rust core/runtime libraries or sidecar binaries
```

Later, the package can add:

```text
ArmchairUi.exe
```

The launcher should be the Velopack main executable:

```text
ArmchairLauncher.exe <- Velopack mainExe
```

The agent should be the production application host:

```text
ArmchairAgent.exe
  -> owns production State
  -> runs transition(State, Input)
  -> executes Effect values
  -> reconciles subscriptions(State)
  -> persists recovery and replay data
  -> exposes IPC endpoints
```

## Minimal POC

The first distribution POC should not include WPF.

It should include:

- A launcher executable that integrates Velopack.
- A long-running per-user background agent.
- Single-instance agent behavior.
- A durable app-data directory.
- A clean shutdown command.
- A simple IPC boundary.
- CLI commands that act as temporary clients.
- A fake subscription source, such as timer ticks or generated market events.
- A pure Rust core stub with `State + Input -> State + Effects`.
- At least one projection endpoint, such as status.
- Velopack packaging around the launcher and agent.
- A local update test path.

The temporary CLI should not become the application architecture. It should only exercise the same IPC boundary that the future WPF UI will use.

```text
POC:
CLI or launcher -> IPC -> Agent -> Core

Future:
WPF UI          -> IPC -> Agent -> Core
CLI or launcher -> IPC -> Agent -> Core
```

## Launcher Responsibilities

The launcher should handle Velopack integration and user-facing process startup.

Example commands:

```text
ArmchairLauncher.exe
  -> ensure the agent is running
  -> later, open the WPF UI

ArmchairLauncher.exe --background
  -> ensure the agent is running
  -> exit

ArmchairLauncher.exe start
  -> start the agent if needed

ArmchairLauncher.exe stop
  -> ask the agent to flush state and exit

ArmchairLauncher.exe status
  -> read status projection from the agent

ArmchairLauncher.exe monitor BTC-USDT
  -> send a monitoring command to the agent
```

Startup registration should point to:

```text
ArmchairLauncher.exe --background
```

Do not register the WPF UI as the startup process.

## Velopack Hooks

Velopack should run from the launcher entry point as early as possible.

Install, update, and uninstall hooks should be trivial. They may:

- Register or remove startup behavior.
- Register or remove notification activation behavior.
- Ask the agent to prepare for update.
- Clean up install-owned resources during uninstall.

Hooks should not contain domain logic. They should not run long computations or show UI.

## Update Behavior

The background agent should check for updates because it is the process expected to run while the UI is closed.

The update flow should be:

```text
agent checks update feed
agent downloads available update
agent quiesces subscriptions
agent flushes durable state
agent asks UI clients to close or disconnect
agent applies update through Velopack
updated launcher or agent restarts
```

The pure core must not download, verify, install, or restart application binaries. Update behavior is an impure host concern.

The agent may feed update-related events into the core only when they matter to product behavior.

## Release Feed

Velopack releases should be uploaded to a static update location such as HTTPS object storage, a web server, or a release hosting service.

The update location should contain the generated release index and packages, for example:

```text
https://updates.armchairanalyst.app/windows/
  releases.win.json
  ArmchairAnalyst-0.1.0-full.nupkg
  ArmchairAnalyst-0.1.0-delta.nupkg
  ArmchairAnalyst-Setup.exe
```

The agent should use this release feed when checking for updates.

## Build Shape

The release build should:

- Compile the Rust core and runtime artifacts.
- Publish the launcher and agent into one release directory.
- Include required Rust libraries or sidecar binaries.
- Package that directory with Velopack.
- Sign release binaries and installers for non-local distribution.

Example package shape:

```text
publish/
  ArmchairLauncher.exe
  ArmchairAgent.exe
  armchair_core.dll
```

Example Velopack command shape:

```text
vpk pack \
  --packId ArmchairAnalyst \
  --packTitle "Armchair Analyst" \
  --packVersion 0.1.0 \
  --packDir publish \
  --mainExe ArmchairLauncher.exe \
  --runtime win-x64 \
  --shortcuts Desktop,StartMenuRoot \
  --icon assets/app.ico
```

## Mutable Data

Do not store mutable data beside the installed binaries.

The update system may replace the installed application directory during update. Store durable data in explicit app-data locations instead.

Examples:

- User settings.
- Agent state snapshots.
- Replay logs.
- Local caches.
- Diagnostics.
- Crash reports.

## Adding WPF Later

Adding WPF later should not change the ownership model.

The WPF UI should:

- Start or connect to the existing background agent.
- Send commands over IPC.
- Read projection outputs over IPC.
- Present user-facing views and controls.
- Close without stopping the agent.

The WPF UI should not:

- Own production `State`.
- Run exchange monitoring.
- Execute domain effects directly.
- Contain domain logic.
- Become required for automatic updates or background monitoring.

## Future Windows Service

A Windows Service is not part of the initial distribution plan.

It may be considered later if monitoring must continue before user login, after user logoff, or across multiple user sessions. That would require a separate service process and likely a per-user companion process for UI and notifications.
