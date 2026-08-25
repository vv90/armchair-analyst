# Armchair Analyst

Armchair Analyst is a work-in-progress, local market-analysis system. It streams Uniswap v3/v4 pool
state from EVM networks and runs differentiable routing and optimization over the observed reserves.
Results are analytical and read-only: the project does not sign or submit transactions.

The functional implementation is currently a Rust workspace with a terminal client, an Ethereum data
service, and reusable client/core libraries. Native desktop shells are planned but not yet implemented;
the FFI crate is also still a stub.

## Quick start

The supported development environment is the pinned Nix shell. A recent Rust toolchain supporting
edition 2024 (Rust 1.85+) can also be used directly.

```bash
nix develop
cd core
cp aa.example.toml aa.toml
export AA_CONFIG_FILE=aa.toml
cargo run -p aa-cli --release
```

[`aa.example.toml`](core/aa.example.toml) documents RPC, WebSocket, subgraph, and provider-key
configuration. API keys stay in environment variables and missing keys are prompted for at startup.
Optional runtime settings include `AA_METADATA_CACHE_PATH` and `AA_TOKEN_WHITELIST_FILE`.

Run the workspace checks from `core/`:

```bash
cargo test --workspace
cargo fmt --all --check
cargo clippy --workspace --all-targets
```

## Repository layout

- `core/crates/aa-cli` — live terminal monitor and optimizer.
- `core/crates/client-evm` — EVM ingestion, pool tracking, replay, and metadata.
- `core/crates/optimization` — differentiable routing and reserve optimization.
- `core/crates/aa-server` — Ethereum pool-state HTTP data service.
- `core/crates/aa-client-*` and `aa-wire` — headless GUI engine, presentation contract, wire types,
  and planned native FFI boundary.
- `core/crates/aa-token-vetting` — offline token discovery and whitelist generation.
- `ui` — planned thin native desktop shells.
- `docs` — product direction, architecture, research, and feature specifications.

Start with [the vision](docs/vision.md), [architecture](docs/architecture.md), and
[planned analysis features](docs/analysis-feature-set.md). The workspace is currently marked
`UNLICENSED` and is not published as a collection of crates.
