//! `aa-wire` — the data-plane HTTP wire contract shared by `aa-server` (serialize) and the GUI
//! client (deserialize), giving the request/response shapes a single owner so the two sides can't
//! drift.
//!
//! It will hold the DTOs for `POST /slice`, `GET /pools/meta`, and `GET /health`. Stub: the types
//! are extracted here out of `aa-server`'s `serve` module in a later increment.
