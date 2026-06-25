use std::{collections::BTreeMap, error, fmt, io, path::PathBuf};

use client_evm::{ChainKey, EndpointSpec, chain_key_for_network_path};
use serde::Deserialize;

/// Path to the unified TOML config file holding every mutable endpoint setting (RPC providers, their
/// WebSocket URLs, and Uniswap v4 subgraphs). Required — the runtime has no built-in endpoint defaults.
pub(crate) const CONFIG_FILE_ENV: &str = "AA_CONFIG_FILE";
pub(crate) const METADATA_CACHE_PATH_ENV: &str = "AA_METADATA_CACHE_PATH";

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

/// Every endpoint setting resolved from the config file, with keys substituted: per-chain HTTP specs for
/// the failover pools, a single WS URL per chain for the subscription channel, and per-chain subgraph
/// specs for v4 metadata resolution. The caller assembles these into the client-evm endpoint types.
#[derive(Debug, Default, Eq, PartialEq)]
pub(crate) struct ResolvedEndpoints {
    pub rpc_http: BTreeMap<ChainKey, Vec<EndpointSpec>>,
    pub rpc_ws: BTreeMap<ChainKey, String>,
    pub subgraph: BTreeMap<ChainKey, Vec<EndpointSpec>>,
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
    // The highest-weight provider that declares a WS URL wins each chain's single subscription slot.
    let mut best_ws: BTreeMap<ChainKey, (u32, String)> = BTreeMap::new();

    for provider in file.rpc {
        let needs_key = provider.chains.values().any(|urls| {
            urls.http.contains(KEY_PLACEHOLDER)
                || urls
                    .ws
                    .as_deref()
                    .is_some_and(|ws| ws.contains(KEY_PLACEHOLDER))
        });
        let key = match resolve_provider_key(
            &provider.name,
            provider.key_env.as_deref(),
            needs_key,
            RPC_KEY_PREFIX,
            read_env,
            prompt,
        )? {
            KeyResolution::Skipped => continue,
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
                let replace = best_ws
                    .get(&chain)
                    .is_none_or(|(best_weight, _)| weight > *best_weight);
                if replace {
                    best_ws.insert(chain, (weight, ws));
                }
            }
        }
    }

    resolved.rpc_ws = best_ws
        .into_iter()
        .map(|(chain, (_, url))| (chain, url))
        .collect();

    for provider in file.subgraph {
        let needs_key = provider
            .chains
            .values()
            .any(|url| url.contains(KEY_PLACEHOLDER));
        let key = match resolve_provider_key(
            &provider.name,
            provider.key_env.as_deref(),
            needs_key,
            GRAPH_KEY_PREFIX,
            read_env,
            prompt,
        )? {
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
) -> Result<KeyResolution, CliError>
where
    Env: FnMut(&str) -> Option<String>,
    Prompt: FnMut(&str) -> Result<String, CliError>,
{
    if !needs_key {
        return Ok(KeyResolution::NotNeeded);
    }

    let env_name = key_env
        .map(str::to_owned)
        .unwrap_or_else(|| derived_key_env(key_prefix, name));
    if let Some(key) = read_env(&env_name).and_then(normalize_config_value) {
        return Ok(KeyResolution::Resolved(key));
    }

    let answer = prompt(&format!("{name} API key (leave blank to skip):"))?;
    Ok(match normalize_config_value(answer) {
        Some(key) => KeyResolution::Resolved(key),
        None => KeyResolution::Skipped,
    })
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

        // Single WS per chain (the weight-3 dRPC wins; publicnode declares no ws).
        assert_eq!(
            resolved.rpc_ws.get(&ChainKey::Ethereum).map(String::as_str),
            Some("wss://lb.drpc.org/ethereum/rpc-secret")
        );
        assert_eq!(
            resolved.rpc_ws.get(&ChainKey::Arbitrum).map(String::as_str),
            Some("wss://lb.drpc.org/arbitrum/rpc-secret")
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
    fn highest_weight_provider_wins_the_ws_slot() {
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

        assert_eq!(
            resolved.rpc_ws.get(&ChainKey::Ethereum).map(String::as_str),
            Some("wss://high/ws")
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
}
