use std::{collections::BTreeMap, error, fmt, io, path::PathBuf};

use client_evm::{ChainKey, EndpointSpec, RpcConfig, TheGraphConfig, chain_key_for_network_path};
use serde::Deserialize;

pub(crate) const RPC_HTTP_URL_ENV: &str = "AA_RPC_HTTP_URL";
pub(crate) const RPC_WS_URL_ENV: &str = "AA_RPC_WS_URL";
pub(crate) const RPC_API_KEY_ENV: &str = "AA_RPC_API_KEY";
pub(crate) const RPC_ENDPOINTS_FILE_ENV: &str = "AA_RPC_ENDPOINTS_FILE";
pub(crate) const RPC_PUBLIC_FALLBACKS_ENV: &str = "AA_RPC_PUBLIC_FALLBACKS";
pub(crate) const METADATA_CACHE_PATH_ENV: &str = "AA_METADATA_CACHE_PATH";
pub(crate) const GRAPH_URL_ENV: &str = "AA_GRAPH_URL";
pub(crate) const GRAPH_API_KEY_ENV: &str = "AA_GRAPH_API_KEY";
pub(crate) const GRAPH_ENDPOINTS_FILE_ENV: &str = "AA_GRAPH_ENDPOINTS_FILE";

/// Prefix for the per-provider derived key env var on the RPC endpoints file (`AA_RPC_KEY_<NAME>`).
const RPC_KEY_PREFIX: &str = "AA_RPC_KEY_";
/// Prefix for the per-provider derived key env var on the graph mirrors file (`AA_GRAPH_KEY_<NAME>`).
const GRAPH_KEY_PREFIX: &str = "AA_GRAPH_KEY_";

const DEFAULT_METADATA_CACHE_PATH: &str = "metadata-cache.redb";

/// Token substituted with a provider's resolved API key wherever it appears in that provider's URLs.
const KEY_PLACEHOLDER: &str = "{key}";

const RPC_HTTP_URL_PROMPT: &str = "RPC HTTP URL:";
pub(crate) const RPC_WS_URL_PROMPT: &str = "RPC WebSocket URL:";
pub(crate) const RPC_API_KEY_PROMPT: &str = "RPC API key:";

#[derive(Debug, Eq, PartialEq)]
pub(crate) enum CliError {
    MissingRequiredConfig {
        env_name: &'static str,
    },
    PromptFailed {
        prompt: &'static str,
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
                write!(formatter, "failed to load rpc endpoints config: {message}")
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

/// One `[[provider]]` entry in the endpoints file: a friendly `name`, an optional `weight` (defaulting
/// to 1), an optional `key_env` naming the environment variable holding this provider's API key, and
/// one URL per chain keyed by its network slug (e.g. `ethereum = "https://…/{key}"`). URLs may embed
/// the [`KEY_PLACEHOLDER`]; the key is resolved from `key_env` (or a derived default) and substituted in.
#[derive(Debug, Deserialize)]
struct ProviderEntry {
    name: String,
    weight: Option<u32>,
    key_env: Option<String>,
    #[serde(flatten)]
    chains: BTreeMap<String, String>,
}

/// Outcome of resolving a provider's API key: the provider needs no key, a key was resolved, or the key
/// was skipped (no env value and a blank prompt) — in which case the whole provider is dropped.
enum KeyResolution {
    NotNeeded,
    Resolved(String),
    Skipped,
}

#[derive(Debug, Default, Deserialize)]
struct EndpointsFile {
    #[serde(default)]
    provider: Vec<ProviderEntry>,
}

/// Whether the built-in keyless public endpoint fallbacks are enabled. Defaults to on; any of
/// `0/false/no/off` (case-insensitive) turns them off, keeping all traffic on configured providers.
pub(crate) fn public_fallbacks_enabled_with<Env>(mut read_env: Env) -> bool
where
    Env: FnMut(&'static str) -> Option<String>,
{
    match read_env(RPC_PUBLIC_FALLBACKS_ENV).and_then(normalize_config_value) {
        Some(value) => !matches!(
            value.to_ascii_lowercase().as_str(),
            "0" | "false" | "no" | "off"
        ),
        None => true,
    }
}

/// Loads the optional RPC endpoints file named by [`RPC_ENDPOINTS_FILE_ENV`] into per-chain endpoint
/// specs. Thin wrapper over [`load_endpoints_file_with`] with the RPC file env and key prefix.
pub(crate) fn load_custom_endpoints_with<Env, ReadFile, Prompt>(
    read_env: Env,
    read_file: ReadFile,
    prompt: Prompt,
) -> Result<BTreeMap<ChainKey, Vec<EndpointSpec>>, CliError>
where
    Env: FnMut(&str) -> Option<String>,
    ReadFile: FnOnce(&str) -> io::Result<String>,
    Prompt: FnMut(&str) -> Result<String, CliError>,
{
    load_endpoints_file_with(
        RPC_ENDPOINTS_FILE_ENV,
        RPC_KEY_PREFIX,
        read_env,
        read_file,
        prompt,
    )
}

/// Loads the optional Uniswap v4 subgraph mirrors file named by [`GRAPH_ENDPOINTS_FILE_ENV`] into
/// per-chain endpoint specs — same-schema mirrors of the canonical subgraph that fail over alongside the
/// gateway primary. Thin wrapper over [`load_endpoints_file_with`] with the graph file env and key prefix.
pub(crate) fn load_graph_endpoints_with<Env, ReadFile, Prompt>(
    read_env: Env,
    read_file: ReadFile,
    prompt: Prompt,
) -> Result<BTreeMap<ChainKey, Vec<EndpointSpec>>, CliError>
where
    Env: FnMut(&str) -> Option<String>,
    ReadFile: FnOnce(&str) -> io::Result<String>,
    Prompt: FnMut(&str) -> Result<String, CliError>,
{
    load_endpoints_file_with(
        GRAPH_ENDPOINTS_FILE_ENV,
        GRAPH_KEY_PREFIX,
        read_env,
        read_file,
        prompt,
    )
}

/// Loads an optional endpoints file named by `file_env` into per-chain endpoint specs, resolving each
/// provider's API key (env first under `key_prefix`, then prompt) and substituting it into the URLs.
/// Absent env var → no custom providers (empty map). A read or parse failure is a hard error; a skipped
/// key drops that provider. Shared by the RPC and graph endpoint loaders.
fn load_endpoints_file_with<Env, ReadFile, Prompt>(
    file_env: &str,
    key_prefix: &str,
    mut read_env: Env,
    read_file: ReadFile,
    mut prompt: Prompt,
) -> Result<BTreeMap<ChainKey, Vec<EndpointSpec>>, CliError>
where
    Env: FnMut(&str) -> Option<String>,
    ReadFile: FnOnce(&str) -> io::Result<String>,
    Prompt: FnMut(&str) -> Result<String, CliError>,
{
    let Some(path) = read_env(file_env).and_then(normalize_config_value) else {
        return Ok(BTreeMap::new());
    };

    let content = read_file(&path).map_err(|error| CliError::EndpointConfigFailed {
        message: format!("failed to read {path}: {error}"),
    })?;

    let file = parse_endpoints_toml(&content)?;
    resolve_endpoints(file, key_prefix, &mut read_env, &mut prompt)
}

/// Loads the optional Uniswap v4 subgraph config (gateway URL + API key). Env-only — unlike the required
/// RPC config it never prompts, so an unconfigured aggregator simply yields `None` (v4 metadata
/// resolution is then skipped) rather than blocking startup. Both values must be present and non-empty.
pub(crate) fn load_graph_config_with<Env>(mut read_env: Env) -> Option<TheGraphConfig>
where
    Env: FnMut(&'static str) -> Option<String>,
{
    let url = read_env(GRAPH_URL_ENV).and_then(normalize_config_value)?;
    let api_key = read_env(GRAPH_API_KEY_ENV).and_then(normalize_config_value)?;

    Some(TheGraphConfig { url, api_key })
}

fn parse_endpoints_toml(content: &str) -> Result<EndpointsFile, CliError> {
    toml::from_str(content).map_err(|error| CliError::EndpointConfigFailed {
        message: error.to_string(),
    })
}

fn resolve_endpoints<Env, Prompt>(
    file: EndpointsFile,
    key_prefix: &str,
    read_env: &mut Env,
    prompt: &mut Prompt,
) -> Result<BTreeMap<ChainKey, Vec<EndpointSpec>>, CliError>
where
    Env: FnMut(&str) -> Option<String>,
    Prompt: FnMut(&str) -> Result<String, CliError>,
{
    let mut endpoints: BTreeMap<ChainKey, Vec<EndpointSpec>> = BTreeMap::new();

    for provider in file.provider {
        let key = match resolve_provider_key(&provider, key_prefix, read_env, prompt)? {
            // A skipped key drops the whole provider: none of its endpoints enter the pool.
            KeyResolution::Skipped => continue,
            KeyResolution::NotNeeded => None,
            KeyResolution::Resolved(key) => Some(key),
        };

        let weight = provider.weight.unwrap_or(1);
        for (network, url) in &provider.chains {
            let chain = chain_key_for_network_path(network).ok_or_else(|| {
                CliError::EndpointConfigFailed {
                    message: format!(
                        "provider '{}' references unknown chain '{network}'",
                        provider.name
                    ),
                }
            })?;
            let url = match &key {
                Some(key) => url.replace(KEY_PLACEHOLDER, key),
                None => url.clone(),
            };
            endpoints.entry(chain).or_default().push(EndpointSpec::new(
                provider.name.clone(),
                url,
                weight,
            ));
        }
    }

    Ok(endpoints)
}

/// Resolves a provider's API key when any of its URLs embed [`KEY_PLACEHOLDER`]: the environment
/// (`key_env`, or the derived `<key_prefix><NAME>`) is consulted first, then the user is prompted; a
/// blank prompt (or EOF on a non-interactive run) skips the provider.
fn resolve_provider_key<Env, Prompt>(
    provider: &ProviderEntry,
    key_prefix: &str,
    read_env: &mut Env,
    prompt: &mut Prompt,
) -> Result<KeyResolution, CliError>
where
    Env: FnMut(&str) -> Option<String>,
    Prompt: FnMut(&str) -> Result<String, CliError>,
{
    let needs_key = provider
        .chains
        .values()
        .any(|url| url.contains(KEY_PLACEHOLDER));
    if !needs_key {
        return Ok(KeyResolution::NotNeeded);
    }

    let env_name = provider
        .key_env
        .clone()
        .unwrap_or_else(|| derived_key_env(key_prefix, &provider.name));
    if let Some(key) = read_env(&env_name).and_then(normalize_config_value) {
        return Ok(KeyResolution::Resolved(key));
    }

    let answer = prompt(&format!("{} API key (leave blank to skip):", provider.name))?;
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

pub(crate) fn load_rpc_config_with<Env, Prompt>(
    mut read_env: Env,
    mut prompt: Prompt,
) -> Result<RpcConfig, CliError>
where
    Env: FnMut(&'static str) -> Option<String>,
    Prompt: FnMut(&'static str) -> Result<String, CliError>,
{
    Ok(RpcConfig {
        http_url: required_value(
            RPC_HTTP_URL_ENV,
            RPC_HTTP_URL_PROMPT,
            &mut read_env,
            &mut prompt,
        )?,
        ws_url: required_value(
            RPC_WS_URL_ENV,
            RPC_WS_URL_PROMPT,
            &mut read_env,
            &mut prompt,
        )?,
        api_key: required_value(
            RPC_API_KEY_ENV,
            RPC_API_KEY_PROMPT,
            &mut read_env,
            &mut prompt,
        )?,
    })
}

fn required_value<Env, Prompt>(
    env_name: &'static str,
    prompt_text: &'static str,
    read_env: &mut Env,
    prompt: &mut Prompt,
) -> Result<String, CliError>
where
    Env: FnMut(&'static str) -> Option<String>,
    Prompt: FnMut(&'static str) -> Result<String, CliError>,
{
    if let Some(value) = read_env(env_name).and_then(normalize_config_value) {
        return Ok(value);
    }

    let value = prompt(prompt_text)?;
    normalize_config_value(value).ok_or(CliError::MissingRequiredConfig { env_name })
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

    #[test]
    fn all_env_values_present_returns_config_without_prompting() {
        let mut prompt_calls = 0;
        let result = load_rpc_config_with(
            env_from([
                (RPC_HTTP_URL_ENV, " https://example.invalid/http "),
                (RPC_WS_URL_ENV, " wss://example.invalid/ws "),
                (RPC_API_KEY_ENV, " key-from-env "),
            ]),
            |_| {
                prompt_calls += 1;
                Ok("unused".to_owned())
            },
        );

        assert_eq!(
            result,
            Ok(RpcConfig {
                http_url: "https://example.invalid/http".to_owned(),
                ws_url: "wss://example.invalid/ws".to_owned(),
                api_key: "key-from-env".to_owned(),
            })
        );
        assert_eq!(prompt_calls, 0);
    }

    #[test]
    fn missing_values_are_prompted_for() {
        let mut answers = vec![
            (RPC_WS_URL_PROMPT, "wss://prompted.example/ws"),
            (RPC_API_KEY_PROMPT, "prompted-key"),
        ]
        .into_iter();
        let mut prompts = Vec::new();

        let result = load_rpc_config_with(
            env_from([(RPC_HTTP_URL_ENV, "https://env.example/http")]),
            |prompt| match answers.next() {
                Some((expected_prompt, answer)) => {
                    prompts.push(prompt);
                    assert_eq!(prompt, expected_prompt);
                    Ok(answer.to_owned())
                }
                None => Err(CliError::PromptFailed {
                    prompt,
                    message: "no test answer available".to_owned(),
                }),
            },
        );

        assert_eq!(
            result,
            Ok(RpcConfig {
                http_url: "https://env.example/http".to_owned(),
                ws_url: "wss://prompted.example/ws".to_owned(),
                api_key: "prompted-key".to_owned(),
            })
        );
        assert_eq!(prompts, vec![RPC_WS_URL_PROMPT, RPC_API_KEY_PROMPT]);
    }

    #[test]
    fn empty_env_values_are_prompted_for() {
        let result = load_rpc_config_with(
            env_from([
                (RPC_HTTP_URL_ENV, " "),
                (RPC_WS_URL_ENV, "\t"),
                (RPC_API_KEY_ENV, "\n"),
            ]),
            prompt_from([
                (RPC_HTTP_URL_PROMPT, "https://prompted.example/http"),
                (RPC_WS_URL_PROMPT, "wss://prompted.example/ws"),
                (RPC_API_KEY_PROMPT, "prompted-key"),
            ]),
        );

        assert_eq!(
            result,
            Ok(RpcConfig {
                http_url: "https://prompted.example/http".to_owned(),
                ws_url: "wss://prompted.example/ws".to_owned(),
                api_key: "prompted-key".to_owned(),
            })
        );
    }

    #[test]
    fn empty_prompt_value_is_rejected() {
        let result = load_rpc_config_with(
            env_from([
                (RPC_HTTP_URL_ENV, "https://env.example/http"),
                (RPC_WS_URL_ENV, "wss://env.example/ws"),
            ]),
            prompt_from([(RPC_API_KEY_PROMPT, " ")]),
        );

        assert_eq!(
            result,
            Err(CliError::MissingRequiredConfig {
                env_name: RPC_API_KEY_ENV
            })
        );
    }

    #[test]
    fn public_fallbacks_enabled_defaults_on_and_respects_disable_values() {
        assert!(public_fallbacks_enabled_with(env_from([])));
        for disable in ["0", "false", "NO", "Off"] {
            assert!(
                !public_fallbacks_enabled_with(env_from([(RPC_PUBLIC_FALLBACKS_ENV, disable)])),
                "{disable} should disable public fallbacks"
            );
        }
        assert!(public_fallbacks_enabled_with(env_from([(
            RPC_PUBLIC_FALLBACKS_ENV,
            "1"
        )])));
    }

    #[test]
    fn no_endpoints_file_yields_no_custom_providers() {
        let result = load_custom_endpoints_with(
            env_from([]),
            |_| panic!("file must not be read"),
            never_prompt(),
        );

        assert_eq!(result, Ok(BTreeMap::new()));
    }

    #[test]
    fn endpoints_file_parses_into_per_chain_specs() {
        let toml = r#"
            [[provider]]
            name = "alchemy"
            weight = 3
            ethereum = "https://eth.example/v2/key"
            arbitrum = "https://arb.example/v2/key"

            [[provider]]
            name = "publicnode"
            ethereum = "https://eth.public.example"
        "#;

        let result = load_custom_endpoints_with(
            env_from([(RPC_ENDPOINTS_FILE_ENV, "endpoints.toml")]),
            |path| {
                assert_eq!(path, "endpoints.toml");
                Ok(toml.to_owned())
            },
            never_prompt(),
        )
        .expect("parse succeeds");

        let ethereum = result.get(&ChainKey::Ethereum).expect("ethereum entries");
        assert_eq!(ethereum.len(), 2);
        assert_eq!(
            ethereum[0],
            EndpointSpec::new("alchemy", "https://eth.example/v2/key", 3)
        );
        // Missing weight defaults to 1.
        assert_eq!(
            ethereum[1],
            EndpointSpec::new("publicnode", "https://eth.public.example", 1)
        );

        let arbitrum = result.get(&ChainKey::Arbitrum).expect("arbitrum entries");
        assert_eq!(arbitrum.len(), 1);
        assert_eq!(arbitrum[0].weight, 3);
    }

    #[test]
    fn endpoints_file_with_unknown_chain_is_rejected() {
        let toml = r#"
            [[provider]]
            name = "mystery"
            solana = "https://nope.example"
        "#;

        let result = load_custom_endpoints_with(
            env_from([(RPC_ENDPOINTS_FILE_ENV, "endpoints.toml")]),
            |_| Ok(toml.to_owned()),
            never_prompt(),
        );

        assert!(matches!(result, Err(CliError::EndpointConfigFailed { .. })));
    }

    #[test]
    fn unreadable_endpoints_file_is_an_error() {
        let result = load_custom_endpoints_with(
            env_from([(RPC_ENDPOINTS_FILE_ENV, "missing.toml")]),
            |_| Err(io::Error::new(io::ErrorKind::NotFound, "no such file")),
            never_prompt(),
        );

        assert!(matches!(result, Err(CliError::EndpointConfigFailed { .. })));
    }

    #[test]
    fn provider_key_is_substituted_from_explicit_env() {
        let toml = r#"
            [[provider]]
            name = "alchemy"
            key_env = "MY_ALCHEMY_KEY"
            ethereum = "https://eth.example/v2/{key}"
        "#;

        let result = load_custom_endpoints_with(
            env_from([
                (RPC_ENDPOINTS_FILE_ENV, "endpoints.toml"),
                ("MY_ALCHEMY_KEY", "secret-key"),
            ]),
            |_| Ok(toml.to_owned()),
            never_prompt(),
        )
        .expect("resolution succeeds");

        let ethereum = result.get(&ChainKey::Ethereum).expect("ethereum entries");
        assert_eq!(
            ethereum[0],
            EndpointSpec::new("alchemy", "https://eth.example/v2/secret-key", 1)
        );
    }

    #[test]
    fn provider_key_uses_derived_env_name() {
        let toml = r#"
            [[provider]]
            name = "my-node"
            ethereum = "https://eth.example/{key}"
        "#;

        let result = load_custom_endpoints_with(
            env_from([
                (RPC_ENDPOINTS_FILE_ENV, "endpoints.toml"),
                ("AA_RPC_KEY_MY_NODE", "derived-key"),
            ]),
            |_| Ok(toml.to_owned()),
            never_prompt(),
        )
        .expect("resolution succeeds");

        let ethereum = result.get(&ChainKey::Ethereum).expect("ethereum entries");
        assert_eq!(ethereum[0].url, "https://eth.example/derived-key");
    }

    #[test]
    fn provider_key_is_prompted_when_env_missing() {
        let toml = r#"
            [[provider]]
            name = "alchemy"
            ethereum = "https://eth.example/{key}"
            arbitrum = "https://arb.example/{key}"
        "#;

        let mut prompt_calls = 0;
        let result = load_custom_endpoints_with(
            env_from([(RPC_ENDPOINTS_FILE_ENV, "endpoints.toml")]),
            |_| Ok(toml.to_owned()),
            |prompt: &str| {
                prompt_calls += 1;
                assert!(prompt.contains("alchemy"));
                Ok("prompted-key".to_owned())
            },
        )
        .expect("resolution succeeds");

        // Prompted once for the provider, then reused across both of its chains.
        assert_eq!(prompt_calls, 1);
        assert_eq!(
            result.get(&ChainKey::Ethereum).expect("ethereum")[0].url,
            "https://eth.example/prompted-key"
        );
        assert_eq!(
            result.get(&ChainKey::Arbitrum).expect("arbitrum")[0].url,
            "https://arb.example/prompted-key"
        );
    }

    #[test]
    fn skipped_provider_key_drops_the_provider() {
        let toml = r#"
            [[provider]]
            name = "alchemy"
            ethereum = "https://eth.example/{key}"

            [[provider]]
            name = "publicnode"
            ethereum = "https://eth.public.example"
        "#;

        let result = load_custom_endpoints_with(
            env_from([(RPC_ENDPOINTS_FILE_ENV, "endpoints.toml")]),
            |_| Ok(toml.to_owned()),
            // Blank answer skips the keyed provider; EOF on a non-interactive run behaves the same.
            |_: &str| Ok("   ".to_owned()),
        )
        .expect("resolution succeeds");

        let ethereum = result.get(&ChainKey::Ethereum).expect("ethereum entries");
        assert_eq!(ethereum.len(), 1);
        assert_eq!(ethereum[0].label, "publicnode");
    }

    #[test]
    fn provider_without_placeholder_is_not_prompted() {
        let toml = r#"
            [[provider]]
            name = "alchemy"
            key_env = "MY_ALCHEMY_KEY"
            ethereum = "https://eth.example/no-placeholder"
        "#;

        let result = load_custom_endpoints_with(
            env_from([(RPC_ENDPOINTS_FILE_ENV, "endpoints.toml")]),
            |_| Ok(toml.to_owned()),
            never_prompt(),
        )
        .expect("resolution succeeds");

        assert_eq!(
            result.get(&ChainKey::Ethereum).expect("ethereum")[0].url,
            "https://eth.example/no-placeholder"
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
    fn graph_config_present_when_both_url_and_key_set() {
        let config = load_graph_config_with(env_from([
            (GRAPH_URL_ENV, " https://gateway.thegraph.com/api/{key}/subgraphs/id/v4 "),
            (GRAPH_API_KEY_ENV, " graph-key "),
        ]));

        assert_eq!(
            config,
            Some(TheGraphConfig {
                url: "https://gateway.thegraph.com/api/{key}/subgraphs/id/v4".to_owned(),
                api_key: "graph-key".to_owned(),
            })
        );
    }

    #[test]
    fn graph_config_absent_when_url_or_key_missing() {
        assert_eq!(load_graph_config_with(env_from([])), None);
        assert_eq!(
            load_graph_config_with(env_from([(GRAPH_URL_ENV, "https://gateway.example")])),
            None
        );
        assert_eq!(
            load_graph_config_with(env_from([(GRAPH_API_KEY_ENV, "graph-key")])),
            None
        );
    }

    #[test]
    fn graph_mirrors_file_parses_with_graph_key_prefix() {
        let toml = r#"
            [[provider]]
            name = "goldsky"
            ethereum = "https://api.goldsky.com/subgraphs/v4/{key}"
        "#;

        let result = load_graph_endpoints_with(
            env_from([
                (GRAPH_ENDPOINTS_FILE_ENV, "graph-endpoints.toml"),
                ("AA_GRAPH_KEY_GOLDSKY", "mirror-key"),
            ]),
            |path| {
                assert_eq!(path, "graph-endpoints.toml");
                Ok(toml.to_owned())
            },
            never_prompt(),
        )
        .expect("resolution succeeds");

        let ethereum = result.get(&ChainKey::Ethereum).expect("ethereum entries");
        assert_eq!(
            ethereum[0],
            EndpointSpec::new("goldsky", "https://api.goldsky.com/subgraphs/v4/mirror-key", 1)
        );
    }

    #[test]
    fn no_graph_endpoints_file_yields_no_mirrors() {
        let result = load_graph_endpoints_with(
            env_from([]),
            |_| panic!("file must not be read"),
            never_prompt(),
        );

        assert_eq!(result, Ok(BTreeMap::new()));
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
                env_name: RPC_API_KEY_ENV
            }
            .to_string(),
            "missing required configuration value AA_RPC_API_KEY"
        );
    }

    #[test]
    fn prompt_failed_display_is_stable() {
        assert_eq!(
            CliError::PromptFailed {
                prompt: RPC_API_KEY_PROMPT,
                message: "permission denied".to_owned(),
            }
            .to_string(),
            "failed to read RPC API key: permission denied"
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

    fn never_prompt() -> impl FnMut(&str) -> Result<String, CliError> {
        |_prompt: &str| panic!("prompt must not be called")
    }

    fn prompt_from<const N: usize>(
        values: [(&'static str, &'static str); N],
    ) -> impl FnMut(&'static str) -> Result<String, CliError> {
        let mut answers = values.into_iter();

        move |prompt| match answers.next() {
            Some((expected_prompt, value)) => {
                assert_eq!(prompt, expected_prompt);
                Ok(value.to_owned())
            }
            None => Err(CliError::PromptFailed {
                prompt,
                message: "no test answer available".to_owned(),
            }),
        }
    }
}
