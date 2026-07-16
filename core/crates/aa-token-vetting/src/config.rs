//! Minimal loader for the `[[subgraph]]` section of the same endpoints config file the main binary
//! uses (`AA_CONFIG_FILE`), so a vetting run points at exactly the subgraphs the runtime queries.
//!
//! This deliberately duplicates a slice of `aa-cli`'s loader rather than sharing it: that loader is
//! entangled with interactive key prompting and RPC resolution this offline tool doesn't want. Keys
//! are resolved from the environment only (`key_env`, or the derived `AA_GRAPH_KEY_<NAME>`); a
//! provider whose key is unset is skipped with a warning instead of prompting.

use std::collections::BTreeMap;
use std::io;

use client_evm::{
    ChainEndpoints, ChainKey, EndpointSpec, GraphEndpoints, assemble_chain_endpoints,
    assemble_graph_endpoints, chain_key_for_network_path,
};
use serde::Deserialize;

use crate::VettingError;

pub const CONFIG_FILE_ENV: &str = "AA_CONFIG_FILE";
const RPC_KEY_PREFIX: &str = "AA_RPC_KEY_";
const GRAPH_KEY_PREFIX: &str = "AA_GRAPH_KEY_";
const KEY_PLACEHOLDER: &str = "{key}";

/// One `[[subgraph]]` provider, mirroring the shape in `aa-cli` (the other sections of the file are
/// ignored here).
#[derive(Debug, Deserialize)]
struct SubgraphProviderEntry {
    name: String,
    weight: Option<u32>,
    key_env: Option<String>,
    #[serde(flatten)]
    chains: BTreeMap<String, String>,
}

/// One `[[rpc]]` provider, mirroring the runtime config shape enough for offline vetting.
#[derive(Debug, Deserialize)]
struct RpcProviderEntry {
    name: String,
    weight: Option<u32>,
    key_env: Option<String>,
    #[serde(flatten)]
    chains: BTreeMap<String, RpcChainUrls>,
}

#[derive(Debug, Deserialize)]
struct RpcChainUrls {
    http: String,
}

#[derive(Debug, Default, Deserialize)]
struct ConfigFile {
    #[serde(default)]
    rpc: Vec<RpcProviderEntry>,
    #[serde(default)]
    subgraph: Vec<SubgraphProviderEntry>,
}

/// The resolved subgraph endpoints plus the names of providers skipped for a missing key, so the
/// caller can surface them (a skipped provider can silently cost a whole chain's coverage).
pub struct LoadedGraphEndpoints {
    pub endpoints: GraphEndpoints,
    pub rpc_endpoints: ChainEndpoints,
    pub skipped_providers: Vec<String>,
}

/// Loads and resolves the subgraph endpoints from the file named by [`CONFIG_FILE_ENV`], with every
/// effectful dependency injected for testability.
pub fn load_graph_endpoints_with<Env, ReadFile>(
    mut read_env: Env,
    read_file: ReadFile,
) -> Result<LoadedGraphEndpoints, VettingError>
where
    Env: FnMut(&str) -> Option<String>,
    ReadFile: FnOnce(&str) -> io::Result<String>,
{
    let path = read_env(CONFIG_FILE_ENV).and_then(normalize).ok_or(
        VettingError::MissingRequiredConfig {
            env_name: CONFIG_FILE_ENV,
        },
    )?;

    let content = read_file(&path).map_err(|error| VettingError::ConfigFailed {
        message: format!("failed to read {path}: {error}"),
    })?;

    let file: ConfigFile =
        toml::from_str(&content).map_err(|error| VettingError::ConfigFailed {
            message: error.to_string(),
        })?;

    let mut rpc_specs: BTreeMap<ChainKey, Vec<EndpointSpec>> = BTreeMap::new();
    let mut graph_specs: BTreeMap<ChainKey, Vec<EndpointSpec>> = BTreeMap::new();
    let mut skipped_providers = Vec::new();

    for provider in file.rpc {
        let Some(key) = provider_key(
            &provider.name,
            provider.key_env.as_ref(),
            &provider.chains,
            RPC_KEY_PREFIX,
            &mut read_env,
        ) else {
            skipped_providers.push(provider.name);
            continue;
        };

        let weight = provider.weight.unwrap_or(1);
        for (network, urls) in &provider.chains {
            let chain =
                chain_key_for_network_path(network).ok_or_else(|| VettingError::ConfigFailed {
                    message: format!(
                        "provider '{}' references unknown chain '{network}'",
                        provider.name
                    ),
                })?;
            let url = substitute_key(&urls.http, key.as_deref());
            rpc_specs.entry(chain).or_default().push(EndpointSpec::new(
                provider.name.clone(),
                url,
                weight,
            ));
        }

    }

    for provider in file.subgraph {
        let Some(key) = provider_key(
            &provider.name,
            provider.key_env.as_ref(),
            &provider.chains,
            GRAPH_KEY_PREFIX,
            &mut read_env,
        ) else {
            skipped_providers.push(provider.name);
            continue;
        };

        let weight = provider.weight.unwrap_or(1);
        for (network, url) in &provider.chains {
            let chain =
                chain_key_for_network_path(network).ok_or_else(|| VettingError::ConfigFailed {
                    message: format!(
                        "provider '{}' references unknown chain '{network}'",
                        provider.name
                    ),
                })?;
            let url = substitute_key(url, key.as_deref());
            graph_specs.entry(chain).or_default().push(EndpointSpec::new(
                provider.name.clone(),
                url,
                weight,
            ));
        }

    }

    let rpc_endpoints =
        assemble_chain_endpoints(&rpc_specs).map_err(|error| VettingError::ConfigFailed {
            message: error.to_string(),
        })?;
    let endpoints =
        assemble_graph_endpoints(&graph_specs).map_err(|error| VettingError::ConfigFailed {
            message: error.to_string(),
        })?;

    Ok(LoadedGraphEndpoints {
        endpoints,
        rpc_endpoints,
        skipped_providers,
    })
}

fn provider_key<Env, T>(
    name: &str,
    key_env: Option<&String>,
    chains: &BTreeMap<String, T>,
    default_prefix: &str,
    read_env: &mut Env,
) -> Option<Option<String>>
where
    Env: FnMut(&str) -> Option<String>,
    T: UrlTemplate,
{
    let needs_key = chains.values().any(UrlTemplate::contains_key_placeholder);
    if !needs_key {
        return Some(None);
    }

    let env_name = key_env
        .cloned()
        .unwrap_or_else(|| derived_key_env_with_prefix(default_prefix, name));
    read_env(&env_name)
        .and_then(normalize)
        .map(Some)
}

trait UrlTemplate {
    fn contains_key_placeholder(&self) -> bool;
}

impl UrlTemplate for String {
    fn contains_key_placeholder(&self) -> bool {
        self.contains(KEY_PLACEHOLDER)
    }
}

impl UrlTemplate for RpcChainUrls {
    fn contains_key_placeholder(&self) -> bool {
        self.http.contains(KEY_PLACEHOLDER)
    }
}

fn substitute_key(url: &str, key: Option<&str>) -> String {
    match key {
        Some(key) => url.replace(KEY_PLACEHOLDER, key),
        None => url.to_owned(),
    }
}

/// Default key env var for a provider: `AA_GRAPH_KEY_<NAME>` with the name upcased and
/// non-alphanumerics replaced by `_` — the same derivation the main binary uses.
fn derived_key_env_with_prefix(prefix: &str, name: &str) -> String {
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
    format!("{prefix}{sanitized}")
}

fn normalize(value: String) -> Option<String> {
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
        key_env = "AA_RPC_KEY_DRPC"
        ethereum = { http = "https://lb.drpc.org/ethereum/{key}" }
        arbitrum = { http = "https://lb.drpc.org/arbitrum/{key}" }
        base = { http = "https://lb.drpc.org/base/{key}" }
        optimism = { http = "https://lb.drpc.org/optimism/{key}" }
        polygon = { http = "https://lb.drpc.org/polygon/{key}" }
        bnb = { http = "https://lb.drpc.org/bsc/{key}" }
        avalanche = { http = "https://lb.drpc.org/avalanche/{key}" }

        [[subgraph]]
        name = "thegraph"
        key_env = "AA_GRAPH_API_KEY"
        ethereum = "https://gateway.thegraph.com/api/{key}/subgraphs/id/v4"
    "#;

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

    #[test]
    fn missing_config_env_is_an_error() {
        let result = load_graph_endpoints_with(env_from([]), |_| panic!("file must not be read"));

        assert!(matches!(
            result,
            Err(VettingError::MissingRequiredConfig {
                env_name: CONFIG_FILE_ENV
            })
        ));
    }

    #[test]
    fn resolves_subgraph_endpoints_with_key_substituted_and_ignores_rpc() {
        let loaded = load_graph_endpoints_with(
            env_from([
                (CONFIG_FILE_ENV, "aa.toml"),
                ("AA_GRAPH_API_KEY", "graph-secret"),
                ("AA_RPC_KEY_DRPC", "rpc-secret"),
            ]),
            |path| {
                assert_eq!(path, "aa.toml");
                Ok(CONFIG.to_owned())
            },
        )
        .expect("config loads");

        assert!(loaded.endpoints.pool(ChainKey::Ethereum).is_some());
        assert!(loaded.rpc_endpoints.pool(ChainKey::Ethereum).is_ok());
        assert!(loaded.endpoints.pool(ChainKey::Arbitrum).is_none());
        assert!(loaded.skipped_providers.is_empty());
    }

    #[test]
    fn provider_without_a_key_is_skipped_and_reported() {
        let loaded = load_graph_endpoints_with(
            env_from([(CONFIG_FILE_ENV, "aa.toml"), ("AA_RPC_KEY_DRPC", "rpc-secret")]),
            |_| Ok(CONFIG.to_owned()),
        )
        .expect("config loads");

        assert!(loaded.endpoints.pool(ChainKey::Ethereum).is_none());
        assert_eq!(loaded.skipped_providers, vec!["thegraph".to_owned()]);
    }

    #[test]
    fn rpc_provider_without_key_is_a_config_error() {
        let config = r#"
            [[rpc]]
            name = "drpc"
            key_env = "AA_RPC_KEY_DRPC"
            ethereum = { http = "https://lb.drpc.org/ethereum/{key}" }
            arbitrum = { http = "https://lb.drpc.org/arbitrum/{key}" }
            base = { http = "https://lb.drpc.org/base/{key}" }
            optimism = { http = "https://lb.drpc.org/optimism/{key}" }
            polygon = { http = "https://lb.drpc.org/polygon/{key}" }
            bnb = { http = "https://lb.drpc.org/bsc/{key}" }
            avalanche = { http = "https://lb.drpc.org/avalanche/{key}" }
        "#;

        let result =
            load_graph_endpoints_with(env_from([(CONFIG_FILE_ENV, "aa.toml")]), |_| {
                Ok(config.to_owned())
            });

        assert!(matches!(result, Err(VettingError::ConfigFailed { .. })));
    }

    #[test]
    fn derived_key_env_matches_the_main_binary_convention() {
        assert_eq!(
            derived_key_env_with_prefix(GRAPH_KEY_PREFIX, "the-graph"),
            "AA_GRAPH_KEY_THE_GRAPH"
        );
    }
}
