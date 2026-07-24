//! `aa-client-ffi` — the in-process FFI boundary for the GUI client. Compiles `aa-client-core` into
//! a `cdylib` behind a small, stable C ABI that a native UI process loads directly (WPF/C# via
//! P/Invoke on Windows first; SwiftUI/GTK later).
//!
//! This is the only crate permitted `unsafe` (see the lint override in `Cargo.toml`), confined to
//! the `extern "C"` boundary. Stub: no exported functions yet — those and the ABI land in a later
//! increment.
