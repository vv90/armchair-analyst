use std::{collections::BTreeMap, error, fmt, io, path::PathBuf};

use client_evm::{
    ACTIVE_CHAINS, ChainKey, EndpointSpec, TokenAddress, TokenWhitelist, TokenWhitelistFile,
    chain_key_for_network_path, drpc_network_path,
};
use serde::Deserialize;

/// Path to the unified TOML config file holding every mutable endpoint setting (RPC providers, their
/// WebSocket URLs, and Uniswap v4 subgraphs). Required — the runtime has no built-in endpoint defaults.
pub(crate) const CONFIG_FILE_ENV: &str = "AA_CONFIG_FILE";
pub(crate) const METADATA_CACHE_PATH_ENV: &str = "AA_METADATA_CACHE_PATH";
/// Path to the per-chain token whitelist artifact (written offline by the vetting tool). Optional:
/// unset/blank disables whitelisting entirely (every token allowed). A file that is set but
/// unreadable or invalid is a startup error — a malformed whitelist must never silently mean
/// allow-all.
pub(crate) const TOKEN_WHITELIST_FILE_ENV: &str = "AA_TOKEN_WHITELIST_FILE";

/// Prefix for a provider's derived key env var on `[[rpc]]` entries (`AA_RPC_KEY_<NAME>`).
const RPC_KEY_PREFIX: &str = "AA_RPC_KEY_";
/// Prefix for a provider's derived key env var on `[[subgraph]]` entries (`AA_GRAPH_KEY_<NAME>`).
const GRAPH_KEY_PREFIX: &str = "AA_GRAPH_KEY_";

const DEFAULT_METADATA_CACHE_PATH: &str = "metadata-cache.redb";

/// Token substituted with a provider's resolved API key wherever it appears in that provider's URLs.
/// The only secret-bearing piece of any URL: keys live in the environment, never in the config file.
const KEY_PLACEHOLDER: &str = "{key}";

#[derive(Debug, Eq, PartialEq)]
pub(crate) enum CliError {
    MissingRequiredConfig {
        env_name: &'static str,
    },
    PromptFailed {
        prompt: String,
        message: String,
    },
    RuntimeFailed {
        message: String,
    },
    LogInitFailed {
        message: String,
    },
    CacheInitFailed {
        message: String,
    },
    EndpointConfigFailed {
        message: String,
    },
    WhitelistConfigFailed {
        message: String,
    },
    InitAssetNotWhitelisted {
        init_asset: String,
    },
}

impl fmt::Display for CliError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingRequiredConfig { env_name } => {
                write!(formatter, "missing required configuration value {env_name}")
            }
            Self::PromptFailed { prompt, message } => {
                write!(formatter, "failed to read {prompt} {message}")
            }
            Self::RuntimeFailed { message } => {
                write!(formatter, "runtime failed: {message}")
            }
            Self::LogInitFailed { message } => {
                write!(formatter, "failed to initialize log file: {message}")
            }
            Self::CacheInitFailed { message } => {
                write!(formatter, "failed to open metadata cache: {message}")
            }
            Self::EndpointConfigFailed { message } => {
                write!(formatter, "failed to load endpoints config: {message}")
            }
            Self::WhitelistConfigFailed { message } => {
                write!(formatter, "failed to load token whitelist: {message}")
            }
            Self::InitAssetNotWhitelisted { init_asset } => {
                write!(
                    formatter,
                    "token whitelist does not allow the optimization init asset {init_asset}"
                )
            }
        }
    }
}

impl error::Error for CliError {}

/// Resolves the metadata-cache file path from the environment, defaulting to a file in the working
/// directory when unset.
pub(crate) fn metadata_cache_path_with<Env>(mut read_env: Env) -> PathBuf
where
    Env: FnMut(&'static str) -> Option<String>,
{
    read_env(METADATA_CACHE_PATH_ENV)
        .and_then(normalize_config_value)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_METADATA_CACHE_PATH))
}

/// Loads the token whitelist artifact named by [`TOKEN_WHITELIST_FILE_ENV`]. `Ok(None)` when the
/// env var is unset or blank (whitelisting disabled); any failure to read, parse, or validate a
/// *configured* file is an error rather than a silent allow-all.
pub(crate) fn load_token_whitelist_with<Env, ReadFile>(
    mut read_env: Env,
    read_file: ReadFile,
) -> Result<Option<TokenWhitelist>, CliError>
where
    Env: FnMut(&str) -> Option<String>,
    ReadFile: FnOnce(&str) -> io::Result<String>,
{
    let Some(path) = read_env(TOKEN_WHITELIST_FILE_ENV).and_then(normalize_config_value) else {
        return Ok(None);
    };

    let content = read_file(&path).map_err(|error| CliError::WhitelistConfigFailed {
        message: format!("failed to read {path}: {error}"),
    })?;

    let file: TokenWhitelistFile =
        toml::from_str(&content).map_err(|error| CliError::WhitelistConfigFailed {
            message: error.to_string(),
        })?;

    file.into_whitelist()
        .map(Some)
        .map_err(|error| CliError::WhitelistConfigFailed {
            message: error.to_string(),
        })
}

/// Startup log lines describing the token-whitelist configuration: one status line (per-chain
/// allowed-token counts, or `disabled`), then a warning per active chain with no whitelisted
/// tokens (nothing on it can be optimized), per present chain missing the native zero-address
/// token (its v4 native-currency pools can't route), and per bridge pair dropped because an
/// endpoint isn't whitelisted.
pub(crate) fn summarize_token_whitelist(
    whitelist: Option<&TokenWhitelist>,
    dropped_bridges: &[(TokenAddress, TokenAddress)],
) -> Vec<String> {
    let Some(whitelist) = whitelist else {
        return vec!["token_whitelist disabled".to_string()];
    };

    let counts = whitelist.chain_counts();
    let per_chain = ACTIVE_CHAINS
        .iter()
        .map(|&chain| {
            format!(
                "{}={}",
                drpc_network_path(chain),
                counts.get(&chain).copied().unwrap_or(0)
            )
        })
        .collect::<Vec<_>>()
        .join(" ");

    let mut lines = vec![format!("token_whitelist enabled {per_chain}")];

    for &chain in ACTIVE_CHAINS {
        if !counts.contains_key(&chain) {
            lines.push(format!(
                "warn token_whitelist_empty_chain chain={} no tokens allowed on this chain",
                drpc_network_path(chain)
            ));
        } else if !whitelist.allows_native(chain) {
            lines.push(format!(
                "warn token_whitelist_missing_native_token chain={} v4 native-currency pools cannot route",
                drpc_network_path(chain)
            ));
        }
    }

    for &(from, to) in dropped_bridges {
        lines.push(format!(
            "warn token_whitelist_dropped_bridge from={}:{} to={}:{}",
            drpc_network_path(from.1),
            from.0,
            drpc_network_path(to.1),
            to.0
        ));
    }

    lines
}

/// The per-chain HTTP/WS URLs for one `[[rpc]]` provider on one chain. `http` feeds the failover pool;
/// the optional `ws` feeds the (single-connection) subscription channel.
#[derive(Debug, Deserialize)]
struct RpcChainUrls {
    http: String,
    ws: Option<String>,
}

/// One `[[rpc]]` provider: a friendly `name`, an optional `weight` (defaulting to 1), an optional
/// `key_env` naming the env var holding this provider's API key, and one `{ http, ws }` table per chain
/// keyed by its network slug. URLs may embed [`KEY_PLACEHOLDER`], substituted from the resolved key.
#[derive(Debug, Deserialize)]
struct RpcProviderEntry {
    name: String,
    weight: Option<u32>,
    key_env: Option<String>,
    #[serde(flatten)]
    chains: BTreeMap<String, RpcChainUrls>,
}

/// One `[[subgraph]]` provider: same shape as an RPC provider but with a single query URL per chain
/// (subgraphs have no WebSocket side). Used to resolve Uniswap v4 pool metadata by id.
#[derive(Debug, Deserialize)]
struct SubgraphProviderEntry {
    name: String,
    weight: Option<u32>,
    key_env: Option<String>,
    #[serde(flatten)]
    chains: BTreeMap<String, String>,
}

#[derive(Debug, Default, Deserialize)]
struct ConfigFile {
    #[serde(default)]
    rpc: Vec<RpcProviderEntry>,
    #[serde(default)]
    subgraph: Vec<SubgraphProviderEntry>,
}

/// Outcome of resolving a provider's API key: the provider needs no key, a key was resolved, or the key
/// was skipped (no env value and a blank prompt) — in which case the whole provider is dropped.
enum KeyResolution {
    NotNeeded,
    Resolved(String),
    Skipped,
}

/// Where a provider's key came from, for a non-secret startup log line. Carries the env var *name* but
/// never the key value, so a key that the gateway rejects can be traced to the exact origin to fix
/// (which env var, or the interactive prompt) from the run log alone.
#[derive(Debug, Eq, PartialEq)]
pub(crate) enum KeySource {
    NotNeeded,
    Env(String),
    Prompt,
    Skipped,
}

/// Every endpoint setting resolved from the config file, with keys substituted: per-chain HTTP specs for
/// the failover pools, every provider's WS spec per chain for the fanned-out subscription channels, and
/// per-chain subgraph specs for v4 metadata resolution. The caller assembles these into the client-evm
/// endpoint types.
#[derive(Debug, Default, Eq, PartialEq)]
pub(crate) struct ResolvedEndpoints {
    pub rpc_http: BTreeMap<ChainKey, Vec<EndpointSpec>>,
    pub rpc_ws: BTreeMap<ChainKey, Vec<EndpointSpec>>,
    pub subgraph: BTreeMap<ChainKey, Vec<EndpointSpec>>,
    /// Names of `[[rpc]]` providers dropped because their key was skipped (blank prompt / unset env).
    /// A dropped provider is absent from every pool above; recorded so startup can surface it, since a
    /// silent drop otherwise looks identical to a provider that was never configured.
    pub skipped_rpc_providers: Vec<String>,
    /// Each provider's key origin (never the value), in config order across `[[rpc]]` then
    /// `[[subgraph]]`. Surfaced at startup so a gateway auth rejection points at the exact key to fix.
    pub key_sources: Vec<(String, KeySource)>,
}

/// Loads and resolves the unified config file named by [`CONFIG_FILE_ENV`]. The file is **required** (an
/// absent/blank env var is an error); each provider's API key is resolved from the environment (under
/// `key_env` or a derived `<prefix><NAME>`), then prompted for if missing — a blank answer drops that
/// provider. RPC keys are effectively required (dropping a provider may leave a chain with no endpoint,
/// caught downstream); subgraph keys are optional (dropping the subgraph just skips v4 resolution).
pub(crate) fn load_config_with<Env, ReadFile, Prompt>(
    mut read_env: Env,
    read_file: ReadFile,
    mut prompt: Prompt,
) -> Result<ResolvedEndpoints, CliError>
where
    Env: FnMut(&str) -> Option<String>,
    ReadFile: FnOnce(&str) -> io::Result<String>,
    Prompt: FnMut(&str) -> Result<String, CliError>,
{
    let path = read_env(CONFIG_FILE_ENV)
        .and_then(normalize_config_value)
        .ok_or(CliError::MissingRequiredConfig {
            env_name: CONFIG_FILE_ENV,
        })?;

    let content = read_file(&path).map_err(|error| CliError::EndpointConfigFailed {
        message: format!("failed to read {path}: {error}"),
    })?;

    let file: ConfigFile = toml::from_str(&content).map_err(|error| CliError::EndpointConfigFailed {
        message: error.to_string(),
    })?;

    resolve_config(file, &mut read_env, &mut prompt)
}

fn resolve_config<Env, Prompt>(
    file: ConfigFile,
    read_env: &mut Env,
    prompt: &mut Prompt,
) -> Result<ResolvedEndpoints, CliError>
where
    Env: FnMut(&str) -> Option<String>,
    Prompt: FnMut(&str) -> Result<String, CliError>,
{
    let mut resolved = ResolvedEndpoints::default();

    for provider in file.rpc {
        let needs_key = provider.chains.values().any(|urls| {
            urls.http.contains(KEY_PLACEHOLDER)
                || urls
                    .ws
                    .as_deref()
                    .is_some_and(|ws| ws.contains(KEY_PLACEHOLDER))
        });
        let (resolution, source) = resolve_provider_key(
            &provider.name,
            provider.key_env.as_deref(),
            needs_key,
            RPC_KEY_PREFIX,
            read_env,
            prompt,
        )?;
        resolved.key_sources.push((provider.name.clone(), source));
        let key = match resolution {
            KeyResolution::Skipped => {
                resolved.skipped_rpc_providers.push(provider.name.clone());
                continue;
            }
            KeyResolution::NotNeeded => None,
            KeyResolution::Resolved(key) => Some(key),
        };

        let weight = provider.weight.unwrap_or(1);
        for (network, urls) in &provider.chains {
            let chain = chain_for_network(network, &provider.name)?;
            let http = substitute_key(&urls.http, key.as_deref());
            resolved
                .rpc_http
                .entry(chain)
                .or_default()
                .push(EndpointSpec::new(provider.name.clone(), http, weight));

            if let Some(ws) = &urls.ws {
                let ws = substitute_key(ws, key.as_deref());
                // Keep every provider's WS URL: the runtime fans subscriptions out across all of
                // them per chain (redundancy), rather than picking a single best endpoint.
                resolved
                    .rpc_ws
                    .entry(chain)
                    .or_default()
                    .push(EndpointSpec::new(provider.name.clone(), ws, weight));
            }
        }
    }

    for provider in file.subgraph {
        let needs_key = provider
            .chains
            .values()
            .any(|url| url.contains(KEY_PLACEHOLDER));
        let (resolution, source) = resolve_provider_key(
            &provider.name,
            provider.key_env.as_deref(),
            needs_key,
            GRAPH_KEY_PREFIX,
            read_env,
            prompt,
        )?;
        resolved.key_sources.push((provider.name.clone(), source));
        let key = match resolution {
            KeyResolution::Skipped => continue,
            KeyResolution::NotNeeded => None,
            KeyResolution::Resolved(key) => Some(key),
        };

        let weight = provider.weight.unwrap_or(1);
        for (network, url) in &provider.chains {
            let chain = chain_for_network(network, &provider.name)?;
            let url = substitute_key(url, key.as_deref());
            resolved
                .subgraph
                .entry(chain)
                .or_default()
                .push(EndpointSpec::new(provider.name.clone(), url, weight));
        }
    }

    Ok(resolved)
}

/// Human-readable startup lines describing the resolved RPC pools, so the live provider composition is
/// visible in the run log. One `endpoints chain=…` line per chain lists each pool entry's provider and
/// weight plus whether the chain has a WS subscription; a trailing line names any providers dropped for
/// a skipped key. This makes a silently-dropped provider, or a chain down to a single endpoint, obvious
/// at a glance — the per-request failover only ever logs its last endpoint's error, so the pool's true
/// membership is otherwise invisible.
pub(crate) fn summarize_endpoints(resolved: &ResolvedEndpoints) -> Vec<String> {
    let mut lines: Vec<String> = resolved
        .rpc_http
        .iter()
        .map(|(chain, specs)| {
            let providers = specs
                .iter()
                .map(|spec| format!("{}:{}", spec.label, spec.weight))
                .collect::<Vec<_>>()
                .join(",");
            let ws = resolved
                .rpc_ws
                .get(chain)
                .map(|specs| {
                    specs
                        .iter()
                        .map(|spec| format!("{}:{}", spec.label, spec.weight))
                        .collect::<Vec<_>>()
                        .join(",")
                })
                .unwrap_or_default();
            format!("endpoints chain={chain:?} http={providers} ws={ws}")
        })
        .collect();

    // Subgraph pools are partial-coverage and single-provider today, so a `graph pools` auth failure
    // otherwise names no provider. Mirror the RPC line per v4-enabled chain so the failing subgraph is
    // identifiable from the log.
    for (chain, specs) in &resolved.subgraph {
        let providers = specs
            .iter()
            .map(|spec| format!("{}:{}", spec.label, spec.weight))
            .collect::<Vec<_>>()
            .join(",");
        lines.push(format!("endpoints chain={chain:?} graph={providers}"));
    }

    // Where each provider's key came from — env var name, prompt, or skipped — never the value, so a
    // gateway auth rejection points straight at the origin to fix.
    for (provider, source) in &resolved.key_sources {
        lines.push(format_key_source(provider, source));
    }

    lines.push(format!(
        "endpoints skipped_rpc_providers={}",
        resolved.skipped_rpc_providers.join(",")
    ));

    lines
}

fn chain_for_network(network: &str, provider: &str) -> Result<ChainKey, CliError> {
    chain_key_for_network_path(network).ok_or_else(|| CliError::EndpointConfigFailed {
        message: format!("provider '{provider}' references unknown chain '{network}'"),
    })
}

fn substitute_key(url: &str, key: Option<&str>) -> String {
    match key {
        Some(key) => url.replace(KEY_PLACEHOLDER, key),
        None => url.to_owned(),
    }
}

/// Resolves a provider's API key when `needs_key` (any of its URLs embed [`KEY_PLACEHOLDER`]): the
/// environment (`key_env`, or the derived `<key_prefix><NAME>`) is consulted first, then the user is
/// prompted; a blank prompt (or EOF on a non-interactive run) skips the provider.
fn resolve_provider_key<Env, Prompt>(
    name: &str,
    key_env: Option<&str>,
    needs_key: bool,
    key_prefix: &str,
    read_env: &mut Env,
    prompt: &mut Prompt,
) -> Result<(KeyResolution, KeySource), CliError>
where
    Env: FnMut(&str) -> Option<String>,
    Prompt: FnMut(&str) -> Result<String, CliError>,
{
    if !needs_key {
        return Ok((KeyResolution::NotNeeded, KeySource::NotNeeded));
    }

    let env_name = key_env
        .map(str::to_owned)
        .unwrap_or_else(|| derived_key_env(key_prefix, name));
    if let Some(key) = read_env(&env_name).and_then(normalize_config_value) {
        return Ok((KeyResolution::Resolved(key), KeySource::Env(env_name)));
    }

    let answer = prompt(&format!("{name} API key (leave blank to skip):"))?;
    Ok(match normalize_config_value(answer) {
        Some(key) => (KeyResolution::Resolved(key), KeySource::Prompt),
        None => (KeyResolution::Skipped, KeySource::Skipped),
    })
}

/// Renders one non-secret key-origin line: `key provider=<name> source=env:<VAR>|prompt|skipped|none`.
/// Never emits the key value.
fn format_key_source(provider: &str, source: &KeySource) -> String {
    let origin = match source {
        KeySource::NotNeeded => "none".to_owned(),
        KeySource::Env(name) => format!("env:{name}"),
        KeySource::Prompt => "prompt".to_owned(),
        KeySource::Skipped => "skipped".to_owned(),
    };
    format!("key provider={provider} source={origin}")
}

/// Default environment variable name for a provider's key: `<key_prefix><NAME>`, with the name upcased
/// and any non-alphanumeric character replaced by `_` (e.g. prefix `AA_RPC_KEY_` + `my-node` →
/// `AA_RPC_KEY_MY_NODE`).
fn derived_key_env(key_prefix: &str, name: &str) -> String {
    let sanitized: String = name
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect();
    format!("{key_prefix}{sanitized}")
}

fn normalize_config_value(value: String) -> Option<String> {
    let trimmed = value.trim();

    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: &str = r#"
        [[rpc]]
        name = "drpc"
        weight = 3
        ethereum = { http = "https://lb.drpc.org/ethereum/{key}", ws = "wss://lb.drpc.org/ethereum/{key}" }
        arbitrum = { http = "https://lb.drpc.org/arbitrum/{key}", ws = "wss://lb.drpc.org/arbitrum/{key}" }

        [[rpc]]
        name = "publicnode"
        ethereum = { http = "https://ethereum-rpc.publicnode.com" }
        arbitrum = { http = "https://arbitrum-one-rpc.publicnode.com" }

        [[subgraph]]
        name = "thegraph"
        key_env = "AA_GRAPH_API_KEY"
        ethereum = "https://gateway.thegraph.com/api/{key}/subgraphs/id/v4"
    "#;

    #[test]
    fn missing_config_file_env_is_an_error() {
        let result = load_config_with(
            env_from([]),
            |_| panic!("file must not be read"),
            never_prompt(),
        );

        assert_eq!(
            result,
            Err(CliError::MissingRequiredConfig {
                env_name: CONFIG_FILE_ENV
            })
        );
    }

    #[test]
    fn unreadable_config_file_is_an_error() {
        let result = load_config_with(
            env_from([(CONFIG_FILE_ENV, "missing.toml")]),
            |_| Err(io::Error::new(io::ErrorKind::NotFound, "no such file")),
            never_prompt(),
        );

        assert!(matches!(result, Err(CliError::EndpointConfigFailed { .. })));
    }

    #[test]
    fn unknown_chain_is_rejected() {
        let toml = r#"
            [[rpc]]
            name = "mystery"
            solana = { http = "https://nope.example" }
        "#;
        let result = load_config_with(
            env_from([(CONFIG_FILE_ENV, "config.toml")]),
            |_| Ok(toml.to_owned()),
            never_prompt(),
        );

        assert!(matches!(result, Err(CliError::EndpointConfigFailed { .. })));
    }

    #[test]
    fn resolves_rpc_http_ws_and_subgraph_with_keys_substituted() {
        let resolved = load_config_with(
            env_from([
                (CONFIG_FILE_ENV, "config.toml"),
                ("AA_RPC_KEY_DRPC", "rpc-secret"),
                ("AA_GRAPH_API_KEY", "graph-secret"),
            ]),
            |path| {
                assert_eq!(path, "config.toml");
                Ok(CONFIG.to_owned())
            },
            never_prompt(),
        )
        .expect("resolution succeeds");

        // Two HTTP providers on Ethereum, dRPC first with key substituted and weight 3.
        let eth_http = resolved.rpc_http.get(&ChainKey::Ethereum).expect("eth http");
        assert_eq!(eth_http.len(), 2);
        assert_eq!(
            eth_http[0],
            EndpointSpec::new("drpc", "https://lb.drpc.org/ethereum/rpc-secret", 3)
        );
        assert_eq!(
            eth_http[1],
            EndpointSpec::new("publicnode", "https://ethereum-rpc.publicnode.com", 1)
        );

        // Every provider that declares a WS URL is kept per chain (here only dRPC does; publicnode
        // declares no ws), with the key substituted.
        assert_eq!(
            resolved.rpc_ws.get(&ChainKey::Ethereum),
            Some(&vec![EndpointSpec::new(
                "drpc",
                "wss://lb.drpc.org/ethereum/rpc-secret",
                3
            )])
        );
        assert_eq!(
            resolved.rpc_ws.get(&ChainKey::Arbitrum),
            Some(&vec![EndpointSpec::new(
                "drpc",
                "wss://lb.drpc.org/arbitrum/rpc-secret",
                3
            )])
        );

        // Subgraph on Ethereum with the graph key substituted.
        let eth_graph = resolved.subgraph.get(&ChainKey::Ethereum).expect("eth graph");
        assert_eq!(
            eth_graph[0],
            EndpointSpec::new(
                "thegraph",
                "https://gateway.thegraph.com/api/graph-secret/subgraphs/id/v4",
                1
            )
        );
    }

    #[test]
    fn every_provider_ws_url_is_kept_for_fan_out() {
        let toml = r#"
            [[rpc]]
            name = "low"
            weight = 1
            ethereum = { http = "https://low/http", ws = "wss://low/ws" }

            [[rpc]]
            name = "high"
            weight = 5
            ethereum = { http = "https://high/http", ws = "wss://high/ws" }
        "#;
        let resolved = load_config_with(
            env_from([(CONFIG_FILE_ENV, "config.toml")]),
            |_| Ok(toml.to_owned()),
            never_prompt(),
        )
        .expect("resolution succeeds");

        // Both providers' WS URLs survive, in configuration order — no single-slot collapse.
        assert_eq!(
            resolved.rpc_ws.get(&ChainKey::Ethereum),
            Some(&vec![
                EndpointSpec::new("low", "wss://low/ws", 1),
                EndpointSpec::new("high", "wss://high/ws", 5),
            ])
        );
    }

    #[test]
    fn provider_key_is_prompted_when_env_missing() {
        let toml = r#"
            [[rpc]]
            name = "alchemy"
            ethereum = { http = "https://eth.example/{key}" }
            arbitrum = { http = "https://arb.example/{key}" }
        "#;
        let mut prompt_calls = 0;
        let resolved = load_config_with(
            env_from([(CONFIG_FILE_ENV, "config.toml")]),
            |_| Ok(toml.to_owned()),
            |prompt: &str| {
                prompt_calls += 1;
                assert!(prompt.contains("alchemy"));
                Ok("prompted-key".to_owned())
            },
        )
        .expect("resolution succeeds");

        // Prompted once for the provider, reused across both of its chains.
        assert_eq!(prompt_calls, 1);
        assert_eq!(
            resolved.rpc_http.get(&ChainKey::Ethereum).expect("eth")[0].url,
            "https://eth.example/prompted-key"
        );
        assert_eq!(
            resolved.rpc_http.get(&ChainKey::Arbitrum).expect("arb")[0].url,
            "https://arb.example/prompted-key"
        );
    }

    #[test]
    fn skipped_subgraph_key_drops_the_subgraph() {
        let toml = r#"
            [[rpc]]
            name = "drpc"
            ethereum = { http = "https://eth/http", ws = "wss://eth/ws" }

            [[subgraph]]
            name = "thegraph"
            ethereum = "https://gateway/{key}/v4"
        "#;
        let resolved = load_config_with(
            env_from([(CONFIG_FILE_ENV, "config.toml")]),
            |_| Ok(toml.to_owned()),
            // Blank answer skips the keyed subgraph; EOF on a non-interactive run behaves the same.
            |_: &str| Ok("   ".to_owned()),
        )
        .expect("resolution succeeds");

        assert!(resolved.subgraph.is_empty());
        // The keyless RPC provider is unaffected.
        assert!(resolved.rpc_http.contains_key(&ChainKey::Ethereum));
    }

    #[test]
    fn skipped_rpc_key_is_recorded_and_drops_the_provider() {
        let toml = r#"
            [[rpc]]
            name = "infura"
            ethereum = { http = "https://eth.infura/{key}" }

            [[rpc]]
            name = "publicnode"
            ethereum = { http = "https://eth.public.example", ws = "wss://eth/ws" }
        "#;
        let resolved = load_config_with(
            env_from([(CONFIG_FILE_ENV, "config.toml")]),
            |_| Ok(toml.to_owned()),
            // Blank answer skips the keyed infura provider; the keyless publicnode is never prompted.
            |_: &str| Ok(String::new()),
        )
        .expect("resolution succeeds");

        assert_eq!(resolved.skipped_rpc_providers, vec!["infura".to_owned()]);
        // The dropped provider is absent from the pool; only the surviving one remains.
        let eth = resolved.rpc_http.get(&ChainKey::Ethereum).expect("eth");
        assert_eq!(eth.len(), 1);
        assert_eq!(eth[0].label, "publicnode");
    }

    #[test]
    fn summarize_endpoints_lists_providers_ws_composition_and_skips() {
        let mut resolved = ResolvedEndpoints::default();
        resolved.rpc_http.insert(
            ChainKey::Ethereum,
            vec![
                EndpointSpec::new("drpc", "https://d", 3),
                EndpointSpec::new("alchemy", "https://a", 4),
            ],
        );
        resolved
            .rpc_http
            .insert(ChainKey::Arbitrum, vec![EndpointSpec::new("publicnode", "https://p", 1)]);
        // Ethereum has two WS providers; Arbitrum has none, so its `ws=` list must render empty.
        resolved.rpc_ws.insert(
            ChainKey::Ethereum,
            vec![
                EndpointSpec::new("drpc", "wss://d", 3),
                EndpointSpec::new("alchemy", "wss://a", 4),
            ],
        );
        resolved.skipped_rpc_providers.push("infura".to_owned());

        assert_eq!(
            summarize_endpoints(&resolved),
            vec![
                "endpoints chain=Ethereum http=drpc:3,alchemy:4 ws=drpc:3,alchemy:4".to_owned(),
                "endpoints chain=Arbitrum http=publicnode:1 ws=".to_owned(),
                "endpoints skipped_rpc_providers=infura".to_owned(),
            ]
        );
    }

    #[test]
    fn summarize_endpoints_lists_subgraph_pools_and_key_sources() {
        let mut resolved = ResolvedEndpoints::default();
        resolved
            .rpc_http
            .insert(ChainKey::Ethereum, vec![EndpointSpec::new("drpc", "https://d", 3)]);
        resolved
            .subgraph
            .insert(ChainKey::Ethereum, vec![EndpointSpec::new("thegraph", "https://g", 3)]);
        resolved.key_sources.push((
            "drpc".to_owned(),
            KeySource::Env("AA_RPC_KEY_DRPC".to_owned()),
        ));
        resolved.key_sources.push((
            "thegraph".to_owned(),
            KeySource::Env("AA_GRAPH_API_KEY".to_owned()),
        ));

        assert_eq!(
            summarize_endpoints(&resolved),
            vec![
                "endpoints chain=Ethereum http=drpc:3 ws=".to_owned(),
                "endpoints chain=Ethereum graph=thegraph:3".to_owned(),
                "key provider=drpc source=env:AA_RPC_KEY_DRPC".to_owned(),
                "key provider=thegraph source=env:AA_GRAPH_API_KEY".to_owned(),
                "endpoints skipped_rpc_providers=".to_owned(),
            ]
        );
    }

    #[test]
    fn format_key_source_renders_each_origin_without_the_value() {
        assert_eq!(
            format_key_source("thegraph", &KeySource::Env("AA_GRAPH_API_KEY".to_owned())),
            "key provider=thegraph source=env:AA_GRAPH_API_KEY"
        );
        assert_eq!(
            format_key_source("thegraph", &KeySource::Prompt),
            "key provider=thegraph source=prompt"
        );
        assert_eq!(
            format_key_source("infura", &KeySource::Skipped),
            "key provider=infura source=skipped"
        );
        assert_eq!(
            format_key_source("publicnode", &KeySource::NotNeeded),
            "key provider=publicnode source=none"
        );
    }

    #[test]
    fn subgraph_key_source_records_the_resolving_env_var_name() {
        let toml = r#"
            [[rpc]]
            name = "publicnode"
            ethereum = { http = "https://eth.public.example" }

            [[subgraph]]
            name = "thegraph"
            key_env = "AA_GRAPH_API_KEY"
            ethereum = "https://gateway.example/api/{key}/subgraphs/id/abc"
        "#;
        let resolved = load_config_with(
            env_from([
                (CONFIG_FILE_ENV, "config.toml"),
                ("AA_GRAPH_API_KEY", "deadbeef"),
            ]),
            |_| Ok(toml.to_owned()),
            never_prompt(),
        )
        .expect("resolution succeeds");

        // The failing-key scenario: the subgraph key's origin is recoverable as the exact env var.
        assert!(resolved.key_sources.contains(&(
            "thegraph".to_owned(),
            KeySource::Env("AA_GRAPH_API_KEY".to_owned()),
        )));
        // A keyless RPC provider needs no key and is recorded as such.
        assert!(
            resolved
                .key_sources
                .contains(&("publicnode".to_owned(), KeySource::NotNeeded))
        );
    }

    #[test]
    fn provider_without_placeholder_is_not_prompted() {
        let toml = r#"
            [[rpc]]
            name = "publicnode"
            ethereum = { http = "https://eth.public.example" }
        "#;
        let resolved = load_config_with(
            env_from([(CONFIG_FILE_ENV, "config.toml")]),
            |_| Ok(toml.to_owned()),
            never_prompt(),
        )
        .expect("resolution succeeds");

        assert_eq!(
            resolved.rpc_http.get(&ChainKey::Ethereum).expect("eth")[0].url,
            "https://eth.public.example"
        );
    }

    #[test]
    fn provider_key_uses_derived_env_name() {
        let toml = r#"
            [[rpc]]
            name = "my-node"
            ethereum = { http = "https://eth.example/{key}" }
        "#;
        let resolved = load_config_with(
            env_from([
                (CONFIG_FILE_ENV, "config.toml"),
                ("AA_RPC_KEY_MY_NODE", "derived-key"),
            ]),
            |_| Ok(toml.to_owned()),
            never_prompt(),
        )
        .expect("resolution succeeds");

        assert_eq!(
            resolved.rpc_http.get(&ChainKey::Ethereum).expect("eth")[0].url,
            "https://eth.example/derived-key"
        );
    }

    #[test]
    fn derived_key_env_sanitizes_provider_name() {
        assert_eq!(
            derived_key_env(RPC_KEY_PREFIX, "alchemy"),
            "AA_RPC_KEY_ALCHEMY"
        );
        assert_eq!(
            derived_key_env(RPC_KEY_PREFIX, "my-node.io"),
            "AA_RPC_KEY_MY_NODE_IO"
        );
        assert_eq!(
            derived_key_env(GRAPH_KEY_PREFIX, "goldsky"),
            "AA_GRAPH_KEY_GOLDSKY"
        );
    }

    #[test]
    fn metadata_cache_path_defaults_when_unset_and_reads_env() {
        assert_eq!(
            metadata_cache_path_with(static_env_from([])),
            PathBuf::from(DEFAULT_METADATA_CACHE_PATH)
        );
        assert_eq!(
            metadata_cache_path_with(static_env_from([(METADATA_CACHE_PATH_ENV, " /tmp/cache.redb ")])),
            PathBuf::from("/tmp/cache.redb")
        );
    }

    #[test]
    fn normalize_config_value_trims_non_empty_values() {
        assert_eq!(
            normalize_config_value("  value with space  ".to_owned()),
            Some("value with space".to_owned())
        );
    }

    #[test]
    fn normalize_config_value_rejects_whitespace_only_values() {
        assert_eq!(normalize_config_value(" \t\n ".to_owned()), None);
    }

    #[test]
    fn missing_required_config_display_is_stable() {
        assert_eq!(
            CliError::MissingRequiredConfig {
                env_name: CONFIG_FILE_ENV
            }
            .to_string(),
            "missing required configuration value AA_CONFIG_FILE"
        );
    }

    #[test]
    fn prompt_failed_display_is_stable() {
        assert_eq!(
            CliError::PromptFailed {
                prompt: "drpc API key (leave blank to skip):".to_owned(),
                message: "permission denied".to_owned(),
            }
            .to_string(),
            "failed to read drpc API key (leave blank to skip): permission denied"
        );
    }

    #[test]
    fn runtime_failed_display_is_stable() {
        assert_eq!(
            CliError::RuntimeFailed {
                message: "runtime thread panicked".to_owned(),
            }
            .to_string(),
            "runtime failed: runtime thread panicked"
        );
    }

    #[test]
    fn log_init_failed_display_is_stable() {
        assert_eq!(
            CliError::LogInitFailed {
                message: "permission denied".to_owned(),
            }
            .to_string(),
            "failed to initialize log file: permission denied"
        );
    }

    fn env_from<const N: usize>(
        values: [(&'static str, &'static str); N],
    ) -> impl FnMut(&str) -> Option<String> {
        move |name| {
            values
                .iter()
                .find(|(env_name, _)| *env_name == name)
                .map(|(_, value)| (*value).to_owned())
        }
    }

    fn static_env_from<const N: usize>(
        values: [(&'static str, &'static str); N],
    ) -> impl FnMut(&'static str) -> Option<String> {
        move |name| {
            values
                .iter()
                .find(|(env_name, _)| *env_name == name)
                .map(|(_, value)| (*value).to_owned())
        }
    }

    fn never_prompt() -> impl FnMut(&str) -> Result<String, CliError> {
        |_prompt: &str| panic!("prompt must not be called")
    }

    const WHITELIST: &str = r#"
        generated_at = "2026-07-15T12:00:00Z"
        examiner = "approve-all/0.1.0"

        [chains.ethereum]
        tokens = [
            { address = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48", symbol = "USDC" },
            { address = "0x0000000000000000000000000000000000000000", symbol = "ETH" },
        ]
    "#;

    #[test]
    fn absent_whitelist_env_disables_whitelisting() {
        let result =
            load_token_whitelist_with(env_from([]), |_| panic!("file must not be read"));

        assert_eq!(result, Ok(None));
    }

    #[test]
    fn blank_whitelist_env_disables_whitelisting() {
        let result = load_token_whitelist_with(
            env_from([(TOKEN_WHITELIST_FILE_ENV, "  ")]),
            |_| panic!("file must not be read"),
        );

        assert_eq!(result, Ok(None));
    }

    #[test]
    fn valid_whitelist_file_loads() {
        let whitelist = load_token_whitelist_with(
            env_from([(TOKEN_WHITELIST_FILE_ENV, "token-whitelist.toml")]),
            |path| {
                assert_eq!(path, "token-whitelist.toml");
                Ok(WHITELIST.to_owned())
            },
        )
        .expect("whitelist loads")
        .expect("whitelist enabled");

        assert!(whitelist.allows(client_evm::ETHEREUM_USDC_TOKEN_ADDRESS));
        assert!(whitelist.allows_native(ChainKey::Ethereum));
        assert!(!whitelist.allows(client_evm::ARBITRUM_USDC_TOKEN_ADDRESS));
    }

    #[test]
    fn unreadable_whitelist_file_is_an_error() {
        let result = load_token_whitelist_with(
            env_from([(TOKEN_WHITELIST_FILE_ENV, "missing.toml")]),
            |_| Err(io::Error::new(io::ErrorKind::NotFound, "no such file")),
        );

        assert!(matches!(result, Err(CliError::WhitelistConfigFailed { .. })));
    }

    #[test]
    fn malformed_whitelist_toml_is_an_error() {
        let result = load_token_whitelist_with(
            env_from([(TOKEN_WHITELIST_FILE_ENV, "token-whitelist.toml")]),
            |_| Ok("chains = 42".to_owned()),
        );

        assert!(matches!(result, Err(CliError::WhitelistConfigFailed { .. })));
    }

    #[test]
    fn unknown_whitelist_chain_slug_is_an_error() {
        let toml = r#"
            [chains.etherium]
            tokens = [{ address = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48" }]
        "#;
        let result = load_token_whitelist_with(
            env_from([(TOKEN_WHITELIST_FILE_ENV, "token-whitelist.toml")]),
            |_| Ok(toml.to_owned()),
        );

        assert!(matches!(result, Err(CliError::WhitelistConfigFailed { .. })));
    }

    fn loaded_whitelist(toml: &str) -> TokenWhitelist {
        load_token_whitelist_with(
            env_from([(TOKEN_WHITELIST_FILE_ENV, "token-whitelist.toml")]),
            |_| Ok(toml.to_owned()),
        )
        .expect("whitelist loads")
        .expect("whitelist enabled")
    }

    #[test]
    fn disabled_whitelist_summary_is_a_single_line() {
        assert_eq!(
            summarize_token_whitelist(None, &[]),
            vec!["token_whitelist disabled".to_string()]
        );
    }

    #[test]
    fn whitelist_summary_counts_every_active_chain_and_warns_on_empty_ones() {
        let whitelist = loaded_whitelist(WHITELIST);
        let lines = summarize_token_whitelist(Some(&whitelist), &[]);

        assert_eq!(
            lines.first().map(String::as_str),
            Some(
                "token_whitelist enabled ethereum=2 base=0 optimism=0 avalanche=0"
            )
        );
        // Every active chain but Ethereum is absent from the file → one warning each.
        assert_eq!(lines.len(), 1 + (ACTIVE_CHAINS.len() - 1));
        assert!(
            lines[1..]
                .iter()
                .all(|line| line.starts_with("warn token_whitelist_empty_chain"))
        );
    }

    #[test]
    fn whitelist_summary_warns_when_a_present_chain_lacks_the_native_token() {
        let toml = r#"
            [chains.ethereum]
            tokens = [{ address = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48" }]
        "#;
        let whitelist = loaded_whitelist(toml);
        let lines = summarize_token_whitelist(Some(&whitelist), &[]);

        assert!(lines.iter().any(|line| {
            line.starts_with("warn token_whitelist_missing_native_token chain=ethereum")
        }));
    }

    #[test]
    fn whitelist_summary_reports_dropped_bridges() {
        let whitelist = loaded_whitelist(WHITELIST);
        let dropped = [(
            client_evm::ETHEREUM_USDC_TOKEN_ADDRESS,
            client_evm::ARBITRUM_USDC_TOKEN_ADDRESS,
        )];
        let lines = summarize_token_whitelist(Some(&whitelist), &dropped);

        assert!(lines.iter().any(|line| {
            line.starts_with("warn token_whitelist_dropped_bridge from=ethereum:0x")
                && line.contains("to=arbitrum:0x")
        }));
    }
}
