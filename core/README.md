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

## Required configuration

The runtime needs one **dRPC** account for its primary HTTP endpoint and its WebSocket subscriptions.
Provide these via environment variables; any that are missing are **prompted for interactively** at
startup:

| Variable | Meaning | Example |
|---|---|---|
| `AA_RPC_HTTP_URL` | dRPC HTTP base URL | `https://lb.drpc.org` |
| `AA_RPC_WS_URL` | dRPC WebSocket base URL | `wss://lb.drpc.org` |
| `AA_RPC_API_KEY` | dRPC API key | `<your-dkey>` |

The per-chain endpoint is composed as `{base}/{network}/{key}` (e.g.
`https://lb.drpc.org/ethereum/<key>`), with `network` being `ethereum` or `arbitrum`.

```bash
export AA_RPC_HTTP_URL="https://lb.drpc.org"
export AA_RPC_WS_URL="wss://lb.drpc.org"
export AA_RPC_API_KEY="your-dkey"
```

## Optional configuration

| Variable | Default | Meaning |
|---|---|---|
| `AA_RPC_ENDPOINTS_FILE` | unset | Path to a multi-provider endpoints file (see below). Absent → only dRPC + built-in public nodes. |
| `AA_RPC_PUBLIC_FALLBACKS` | on | Set to `0`/`false`/`no`/`off` to drop the bundled keyless public endpoints from every pool. |
| `AA_METADATA_CACHE_PATH` | `metadata-cache.redb` | On-disk cache of immutable pool/token metadata. |
| `AA_RPC_KEY_<NAME>` | — | API key for a provider named in the endpoints file (see below). |

### Multi-provider RPC (`endpoints.toml`)

Beyond the dRPC primary, you can spread HTTP load across additional providers and fail requests over to
an alternative when one errors. Point `AA_RPC_ENDPOINTS_FILE` at a TOML file (a ready-to-edit
[`endpoints.toml`](./endpoints.toml) ships in this directory):

```bash
export AA_RPC_ENDPOINTS_FILE=endpoints.toml
```

Each chain's pool combines: the **dRPC primary** (weight 3) + your **file providers** + the **built-in
public nodes** (weight 1, unless disabled). Requests are distributed by weight (smooth weighted
round-robin) and, on a retryable error, retried on the next healthy endpoint; a failing endpoint is
benched on an exponential cooldown.

Provider API keys use a `{key}` placeholder in the URL, resolved per provider from:
1. the environment variable named by the provider's `key_env` (or the derived `AA_RPC_KEY_<NAME>`), then
2. an interactive prompt at startup if that variable is unset.

Leaving a key prompt **blank skips that whole provider** — so you can keep providers you haven't
configured in the file and just press Enter past them. See the comments in `endpoints.toml` for the
recommended free-tier setup (Alchemy + Infura as the two-chain backbone, Chainstack on Arbitrum) and
how its weights are derived.

```bash
export AA_RPC_KEY_ALCHEMY=...
export AA_RPC_KEY_INFURA=...
export AA_RPC_KEY_CHAINSTACK=...
```

## Running

```bash
cargo run -p aa-cli            # debug
cargo run -p aa-cli --release  # recommended for real runs
```

On startup the binary:
1. Loads the dRPC config (prompting for any missing required value).
2. Loads `AA_RPC_ENDPOINTS_FILE` if set and resolves each provider's key (env → prompt → skip).
3. Opens the metadata cache.
4. Starts the runtime: subscribes to both chains, bootstraps pools, and begins streaming + optimizing.

Because missing keys are prompted for, a first run with no env vars is fully interactive. For
unattended/CI runs, export every variable you need (and press nothing) — a missing optional provider key
read as EOF is treated as "skip".

## Output

- **Live view:** an inline `ratatui` viewport renders on stdout while running.
- **Logs:** each run writes a timestamped file `logs/aa-cli-<millis>-<pid>.log` under this directory.
- **Metadata cache:** persisted to `AA_METADATA_CACHE_PATH` (default `metadata-cache.redb`).

## Notes

- Only `/target/` is git-ignored. `endpoints.toml` contains no secrets when you use `{key}` + env vars
  (keys stay in the environment), but it does hold your account-specific Chainstack/QuickNode host —
  consider keeping a customized copy untracked if that matters to you. Never commit raw API keys.
- The set of tracked chains (Ethereum, Arbitrum) is fixed in code (`ACTIVE_CHAINS`); the WebSocket
  subscription channel uses the dRPC config only (HTTP failover is multi-provider, WS is not).
