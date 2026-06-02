# Windows Host

The initial Windows implementation should use a thin WPF UI process plus a long-running per-user background agent.

The background agent is the real production application host. It owns the production core state while the user is logged in. The WPF UI is an optional thin client for user interaction.

The Windows host is outside the pure core. It owns impure responsibilities such as process lifetime, IPC, networking, storage, notifications, startup registration, installation, and automatic updates.

## Initial Process Model

```text
WPF UI process
  -> sends Command values to the background agent
  -> reads projection outputs from the background agent
  -> can be opened or closed without owning production State

Background agent process
  -> owns the production State
  -> calls transition(State, Input)
  -> executes Effect values
  -> reconciles subscriptions(State)
  -> converts external data into Event values
  -> persists data needed for recovery or replay
  -> cooperates with update shutdown and restart
  -> emits Windows notifications

Installer or package
  -> installs app files
  -> registers the background agent startup behavior
  -> registers the update mechanism
  -> registers notification identity and activation
  -> creates shortcuts and uninstall entries
```

The WPF UI should not own the production core state. Closing the main window should close only the UI process. Monitoring should continue in the background agent while the user remains logged in.

The background agent should be single-instance per user. UI instances, notification activations, and host callbacks should connect to that agent instead of creating competing production core instances.

## Automatic Updates

Automatic updates are required for production distribution.

Update installation is an impure Windows host concern, not a core concern. The core should not download, verify, install, or restart the application.

The installer, package, and update layer should be responsible for:

- Checking for available updates.
- Downloading update packages.
- Verifying update integrity and publisher identity.
- Coordinating shutdown of the WPF UI and background agent.
- Installing the update.
- Restarting the background agent and, when appropriate, the UI.
- Reporting update-related facts back to the core only if product behavior needs them.

The background agent may participate in update coordination by quiescing subscriptions, flushing durable state, and exiting when requested by the updater. It should not contain the pure update policy, package verification, or installer implementation.

Early internal builds may still use portable binaries or zip packages, but the production Windows path should use a signed installer or package format with automatic update support.

## Background Monitoring

The background agent should allow stream monitoring to continue when the main application window is closed.

The agent should:

- Start when the user logs in, subject to user configuration and OS policy.
- Own the production runtime and `State` while the user is logged in.
- Reconcile the desired subscription set derived from state.
- Maintain exchange, chain, timer, and other long-running subscriptions.
- Execute one-shot effects returned by the core.
- Feed subscription data and effect results back into the core as events.
- Persist enough data to recover, debug, or replay important behavior.
- Emit Windows notifications when requested by effects or when host policy maps derived state to notifications.

The agent should keep impure code as small and direct as possible. Any non-trivial decision logic should be moved into pure core logic or pure helper functions that can be tested.

Because the initial agent is not a Windows Service, monitoring should be expected to stop when the user logs out and resume when the user logs back in.

## Notifications

Notifications are a Windows host responsibility.

The core may return declarative effects that request notifications, or it may expose state that the host maps to notification behavior. In both cases, the actual OS notification call is impure runtime work and must remain outside the core.

Notification activation should feed commands or events back into the core. For example, clicking an alert notification might produce a command to focus a market, acknowledge an alert, or open a relevant view.

## Windows Services

Do not use a Windows Service for the initial implementation.

The initial requirement is background monitoring while the main app is closed but the user is still logged in. A per-user background agent is a better fit for that because it can coordinate naturally with user settings, UI, and notifications.

A Windows Service may be considered later if the product needs monitoring that persists across user login boundaries, such as:

- Starting before any user logs in.
- Continuing after a user logs off.
- Running machine-level monitoring shared by multiple users.
- Integrating with enterprise deployment or administration requirements.

If introduced later, the service should be a separate impure host process. It would likely need a per-user companion process for UI and notifications, while the pure core contract remains unchanged.
