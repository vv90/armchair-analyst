# armchair-analyst (core)

Monitors Uniswap-V3-style liquidity pools across **Ethereum** and **Arbitrum** and continuously runs a
cross-chain optimization over their live reserves. It activates each chain subscription-first (streaming
`newHeads` + pool `logs` over WebSocket) and seeds/keeps pool state fresh over HTTP RPC.

The `aa-cli` binary in this workspace is the entry point.

## Prerequisites

- A Rust toolchain supporting **edition 2024**. The supported path is the repo's Nix dev shell:
  ```bash
  nix develop            # from the repo root; provides the pinned toolchain
  ```
  Otherwise install a recent stable Rust via `rustup` (1.85+).
- Network access to your RPC providers.

All commands below are run from this `core/` directory.

## Configuration model

Configuration is split by what each value *is*:

- **Immutable chain data** (contract & token addresses, network slugs) — hardcoded in the binary.
- **Mutable endpoint settings** (provider URLs, weights, WebSocket URLs, subgraph URLs) — a single
  **required** TOML config file.
- **Secrets** (API keys) — loaded **only** from the environment, and prompted for at startup if missing.
  Optional keys may be left blank.

### The config file (`AA_CONFIG_FILE`)

Point the runtime at one TOML file holding every endpoint setting. A ready-to-edit
[`aa.example.toml`](./aa.example.toml) ships in this directory — copy it and edit:

```bash
cp aa.example.toml aa.toml
export AA_CONFIG_FILE=aa.toml
```

The file has two kinds of entry:

- `[[rpc]]` — one weighted **HTTP** endpoint per chain (these form the failover pool) plus an optional
  **WebSocket** URL. Per chain, the highest-weight `[[rpc]]` entry that declares a `ws` serves that
  chain's subscription channel (a single connection — HTTP failover is multi-provider, WS is not).
- `[[subgraph]]` — one Uniswap v4 subgraph query URL per chain, used to resolve v4 pool metadata by id.
  Optional: omit it (or skip its key) to disable v4 metadata resolution.

Each chain's HTTP requests are distributed by weight (smooth weighted round-robin) and, on a retryable
error, retried on the next healthy endpoint; a failing endpoint is benched on an exponential cooldown.
Every active chain (`ethereum`, `arbitrum`) must end up with at least one HTTP endpoint and one WS URL,
or startup fails — there are no built-in endpoint defaults.

### Secrets (environment)

API keys use a `{key}` placeholder in the config URLs, resolved per provider from:

1. the environment variable named by the provider's `key_env` (or the derived `AA_RPC_KEY_<NAME>` /
   `AA_GRAPH_KEY_<NAME>`), then
2. an interactive prompt at startup if that variable is unset.

A **blank** answer skips that whole provider — so you can keep providers you haven't configured and just
press Enter past them. For RPC this may leave a chain short an endpoint (a hard error); for the optional
subgraph it simply disables v4 resolution.

```bash
export AA_RPC_KEY_DRPC=...
export AA_RPC_KEY_ALCHEMY=...
export AA_GRAPH_API_KEY=...      # optional
```

See the comments in `aa.example.toml` for the recommended free-tier setup (dRPC for HTTP+WS, Alchemy +
Infura as the two-chain backbone, Chainstack on Arbitrum) and how its weights are derived.

### Other environment variables

| Variable | Default | Meaning |
|---|---|---|
| `AA_CONFIG_FILE` | — (required) | Path to the unified TOML config file above. |
| `AA_METADATA_CACHE_PATH` | `metadata-cache.redb` | On-disk cache of immutable pool/token metadata. |
| `AA_RPC_KEY_<NAME>` / `AA_GRAPH_KEY_<NAME>` | — | API key for a provider named in the config file. |

### Checking your environment

To see which of the variables the binary uses are set (reports set/unset only — never prints values),
run from this directory. The list is derived from the `key_env`s declared in `aa.toml`, so it stays
correct as you add or remove providers:

```bash
for v in AA_CONFIG_FILE AA_METADATA_CACHE_PATH \
  $(grep -oP 'key_env\s*=\s*"\K[^"]+' aa.toml | sort -u); do
  if [ -n "${!v}" ]; then echo "✓ set    $v"; else echo "✗ unset  $v"; fi
done
```

An unset key is not fatal: you'll be prompted at startup, and a blank answer skips that provider
(`publicnode` is keyless, so it never appears here).

## Running

```bash
cargo run -p aa-cli            # debug
cargo run -p aa-cli --release  # recommended for real runs
```

On startup the binary:
1. Reads `AA_CONFIG_FILE` and resolves each provider's key (env → prompt → skip).
2. Assembles the per-chain HTTP pools, WebSocket subscriptions, and v4 subgraph pools.
3. Opens the metadata cache.
4. Starts the runtime: subscribes to both chains, bootstraps pools, and begins streaming + optimizing.

Because missing keys are prompted for, a first run can be interactive. For unattended/CI runs, export
every key you need (and press nothing) — a missing optional key read as EOF is treated as "skip".

## Output

- **Live view:** an inline `ratatui` viewport renders on stdout while running.
- **Logs:** each run writes a timestamped file `logs/aa-cli-<millis>-<pid>.log` under this directory.
- **Metadata cache:** persisted to `AA_METADATA_CACHE_PATH` (default `metadata-cache.redb`).

## Notes

- Your `aa.toml` contains no secrets when you use `{key}` + env vars (keys stay in the environment), but
  it may hold account-specific hosts (e.g. Chainstack) — keep your customized copy untracked if that
  matters to you. Never commit raw API keys.
- The set of tracked chains (Ethereum, Arbitrum) is fixed in code (`ACTIVE_CHAINS`). The WebSocket
  subscription channel uses a single endpoint per chain (the highest-weight `[[rpc]]` entry with a `ws`);
  HTTP failover is multi-provider, WS is not.
