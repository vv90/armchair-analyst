//! `aa-client-core` — the headless engine for the GUI client. Owns *all* application logic so the
//! UI stays thin: polling the `aa-server` data plane, the `aa-wire` → `optimization::PoolReserves`
//! adapter, client-side session config (init asset + bridges), the reconcile/optimize loop,
//! candidate tracking, and the `AppState` it projects into an `aa-client-api::ViewModel`.
//!
//! Transport-agnostic: it consumes `AppCommand`s and produces `ViewModel`s and knows nothing about
//! the binding (FFI/`cdylib`) or the UI framework. Stub: the engine lands in later increments.
