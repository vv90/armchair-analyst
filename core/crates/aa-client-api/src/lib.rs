//! `aa-client-api` — the UI contract between `aa-client-core` and every native UI shell.
//!
//! Defines the two DTOs a UI speaks: `ViewModel` (what to render) and `AppCommand` (what the user
//! did). serde-only, no logic — the single source of truth a binding generator can target so no
//! business logic ever crosses into the UI language. Stub: the types land in a later increment.
