# UI shells

Native UI front-ends for the armchair-analyst GUI client. These live **outside** the Rust workspace
(`core/`) because C#/Swift projects don't belong in it.

Each shell is deliberately thin: it does exactly two things — render an `aa-client-api::ViewModel`
and emit an `aa-client-api::AppCommand`. All application logic lives in the Rust `aa-client-core`
engine; a shell holds no polling, session, reserve, or optimization logic.

## Binding

Shells load the `aa-client-ffi` `cdylib` artifact (`.dll`/`.dylib`/`.so`) **in-process** over a small
stable C ABI — no sidecar process, no local socket. On Windows that is C# P/Invoke; other platforms
use the equivalent FFI mechanism.

## Planned layout (no code yet)

- `ui/windows/` — WPF (C#/.NET), the primary target.
- `ui/macos/` — SwiftUI/AppKit (later; soft target).
- `ui/linux/` — GTK (later; soft target).
