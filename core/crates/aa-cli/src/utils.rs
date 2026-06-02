use std::{error, fmt};

use client_evm::{ClientEvent, EvmNetwork, RpcConfig};

pub(crate) const RPC_HTTP_URL_ENV: &str = "AA_RPC_HTTP_URL";
pub(crate) const RPC_WS_URL_ENV: &str = "AA_RPC_WS_URL";
pub(crate) const RPC_API_KEY_ENV: &str = "AA_RPC_API_KEY";

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
    SubscriptionFailed {
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
            Self::SubscriptionFailed { message } => {
                write!(formatter, "subscription failed: {message}")
            }
        }
    }
}

impl error::Error for CliError {}

#[derive(Debug, Eq, PartialEq)]
pub(crate) struct EventFeedback {
    pub(crate) event_count: u64,
    pub(crate) line: String,
}

pub(crate) fn format_event_feedback(event: &ClientEvent, event_count: u64) -> EventFeedback {
    match event {
        ClientEvent::Subscribed { subscription_id } => EventFeedback {
            event_count,
            line: format!("connected subscription={subscription_id}"),
        },
        ClientEvent::Notification { result, .. } => {
            let event_count = event_count + 1;
            let address = result.address();
            EventFeedback {
                event_count,
                line: address.to_string(),
            }
        }
        ClientEvent::NewHead { header, .. } => {
            let event_count = event_count + 1;

            EventFeedback {
                event_count,
                line: format!(
                    "head block={} hash={}",
                    header.inner.inner.number, header.inner.hash
                ),
            }
        }
        ClientEvent::Closed { subscription_id } => EventFeedback {
            event_count,
            line: format!("closed subscription={subscription_id}"),
        },
    }
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
        network: EvmNetwork::Ethereum,
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
    use client_evm::{ClientEvent, ClientHead};
    use serde_json::json;

    use super::*;

    #[test]
    fn subscribed_event_feedback_includes_subscription_id() {
        let feedback = format_event_feedback(
            &ClientEvent::Subscribed {
                subscription_id: "0xsubscription".to_owned(),
            },
            3,
        );

        assert_eq!(feedback.event_count, 3);
        assert_eq!(feedback.line, "connected subscription=0xsubscription");
    }

    #[test]
    fn closed_event_feedback_includes_subscription_id() {
        let feedback = format_event_feedback(
            &ClientEvent::Closed {
                subscription_id: "0xsubscription".to_owned(),
            },
            3,
        );

        assert_eq!(feedback.event_count, 3);
        assert_eq!(feedback.line, "closed subscription=0xsubscription");
    }

    #[test]
    fn new_head_event_feedback_includes_block_number_and_hash() -> Result<(), serde_json::Error> {
        let feedback = format_event_feedback(
            &ClientEvent::NewHead {
                subscription_id: "0xsubscription".to_owned(),
                header: new_head()?,
            },
            3,
        );

        assert_eq!(feedback.event_count, 4);
        assert_eq!(
            feedback.line,
            "head block=9 hash=0x0000000000000000000000000000000000000000000000000000000000000001"
        );

        Ok(())
    }

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
                network: EvmNetwork::Ethereum,
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
                network: EvmNetwork::Ethereum,
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
                network: EvmNetwork::Ethereum,
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
    fn subscription_failed_display_is_stable() {
        assert_eq!(
            CliError::SubscriptionFailed {
                message: "websocket error".to_owned(),
            }
            .to_string(),
            "subscription failed: websocket error"
        );
    }

    fn new_head() -> Result<ClientHead, serde_json::Error> {
        serde_json::from_value(json!({
            "hash": "0x0000000000000000000000000000000000000000000000000000000000000001",
            "parentHash": "0x0000000000000000000000000000000000000000000000000000000000000002",
            "sha3Uncles": "0x0000000000000000000000000000000000000000000000000000000000000003",
            "miner": "0x0000000000000000000000000000000000000004",
            "stateRoot": "0x0000000000000000000000000000000000000000000000000000000000000005",
            "transactionsRoot": "0x0000000000000000000000000000000000000000000000000000000000000006",
            "receiptsRoot": "0x0000000000000000000000000000000000000000000000000000000000000007",
            "logsBloom": zero_logs_bloom(),
            "difficulty": "0xd",
            "number": "0x9",
            "gasLimit": "0xb",
            "gasUsed": "0xa",
            "timestamp": "0xc",
            "extraData": "0x010203",
            "mixHash": "0x000000000000000000000000000000000000000000000000000000000000000e",
            "nonce": "0x000000000000000f"
        }))
    }

    fn zero_logs_bloom() -> String {
        format!("0x{}", "00".repeat(256))
    }

    fn env_from<const N: usize>(
        values: [(&'static str, &'static str); N],
    ) -> impl FnMut(&'static str) -> Option<String> {
        move |name| {
            values
                .iter()
                .find(|(env_name, _)| *env_name == name)
                .map(|(_, value)| (*value).to_owned())
        }
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
