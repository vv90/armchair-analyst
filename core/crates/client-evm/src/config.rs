use crate::{
    chain::{ChainKey, drpc_network_path},
    error::{ClientEvmError, ConfigScope},
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RpcConfig {
    pub http_url: String,
    pub ws_url: String,
    pub api_key: String,
}

/// Token substituted with the resolved API key wherever it appears in a graph endpoint URL. The Graph
/// gateway carries its key in the path (`…/api/{key}/subgraphs/id/<id>`), so the URL template embeds
/// this placeholder rather than appending the key like the dRPC RPC endpoints do.
const GRAPH_KEY_PLACEHOLDER: &str = "{key}";

/// Connection details for a Uniswap v4 subgraph (The Graph gateway or a same-schema mirror). `url` is
/// the full query URL and may embed [`GRAPH_KEY_PLACEHOLDER`]; `api_key` is substituted into it. v4
/// metadata is resolved by id from this indexer rather than from chain (a `PoolId` is a one-way hash of
/// its `PoolKey`, so the chain reveals the key only once, in the `Initialize` event).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TheGraphConfig {
    pub url: String,
    pub api_key: String,
}

/// Composes a subgraph's query URL, substituting the API key into the [`GRAPH_KEY_PLACEHOLDER`]. A URL
/// without the placeholder is used verbatim (keyless self-hosted indexers); the key must still be
/// present so a misconfiguration is caught at startup rather than on the first query. Mirrors
/// [`compose_http_endpoint`] but for the GraphQL side channel.
pub(crate) fn compose_graph_endpoint(config: &TheGraphConfig) -> Result<String, ClientEvmError> {
    let url = config.url.trim();
    if url.is_empty() {
        return Err(ClientEvmError::InvalidConfig {
            scope: ConfigScope::Graph,
            reason: "graph url is required".to_owned(),
        });
    }

    let api_key = config.api_key.trim();
    if api_key.is_empty() {
        return Err(ClientEvmError::InvalidConfig {
            scope: ConfigScope::Graph,
            reason: "graph api key is required".to_owned(),
        });
    }

    Ok(url.replace(GRAPH_KEY_PLACEHOLDER, api_key))
}

pub(crate) fn compose_ws_endpoint(
    config: &RpcConfig,
    chain: ChainKey,
) -> Result<String, ClientEvmError> {
    let ws_url = config.ws_url.trim();
    if ws_url.is_empty() {
        return Err(ClientEvmError::InvalidConfig {
            scope: ConfigScope::Subscription,
            reason: "websocket url is required".to_owned(),
        });
    }

    let api_key = config.api_key.trim();
    if api_key.is_empty() {
        return Err(ClientEvmError::InvalidConfig {
            scope: ConfigScope::Subscription,
            reason: "rpc api key is required".to_owned(),
        });
    }

    Ok(format!(
        "{}/{}/{}",
        ws_url.trim_end_matches('/'),
        drpc_network_path(chain),
        api_key
    ))
}

pub(crate) fn compose_http_endpoint(
    config: &RpcConfig,
    chain: ChainKey,
) -> Result<String, ClientEvmError> {
    let http_url = config.http_url.trim();
    if http_url.is_empty() {
        return Err(ClientEvmError::InvalidConfig {
            scope: ConfigScope::Http,
            reason: "http url is required".to_owned(),
        });
    }

    let api_key = config.api_key.trim();
    if api_key.is_empty() {
        return Err(ClientEvmError::InvalidConfig {
            scope: ConfigScope::Http,
            reason: "rpc api key is required".to_owned(),
        });
    }

    Ok(format!(
        "{}/{}/{}",
        http_url.trim_end_matches('/'),
        drpc_network_path(chain),
        api_key
    ))
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use crate::{ChainKey, ClientEvmError, ConfigScope};

    use super::*;

    #[test]
    fn compose_ws_endpoint_appends_network_and_key() {
        let config = rpc_config(" wss://lb.drpc.org/ ", " api-key ");

        let result = compose_ws_endpoint(&config, ChainKey::Ethereum);

        assert!(matches!(
            result.as_deref(),
            Ok("wss://lb.drpc.org/ethereum/api-key")
        ));
    }

    #[test]
    fn compose_ws_endpoint_rejects_empty_ws_url() {
        let config = rpc_config(" ", "api-key");

        assert!(matches!(
            compose_ws_endpoint(&config, ChainKey::Ethereum),
            Err(ClientEvmError::InvalidConfig {
                scope: ConfigScope::Subscription,
                ..
            })
        ));
    }

    #[test]
    fn compose_ws_endpoint_rejects_empty_api_key() {
        let config = rpc_config("wss://lb.drpc.org", "\t");

        assert!(matches!(
            compose_ws_endpoint(&config, ChainKey::Ethereum),
            Err(ClientEvmError::InvalidConfig {
                scope: ConfigScope::Subscription,
                ..
            })
        ));
    }

    #[test]
    fn compose_http_endpoint_appends_network_and_key() {
        let config = rpc_config_with_http(" https://lb.drpc.org/ ", " api-key ");

        let result = compose_http_endpoint(&config, ChainKey::Ethereum);

        assert!(matches!(
            result.as_deref(),
            Ok("https://lb.drpc.org/ethereum/api-key")
        ));
    }

    #[test]
    fn compose_http_endpoint_rejects_empty_http_url() {
        let config = rpc_config_with_http(" ", "api-key");

        assert!(matches!(
            compose_http_endpoint(&config, ChainKey::Ethereum),
            Err(ClientEvmError::InvalidConfig {
                scope: ConfigScope::Http,
                ..
            })
        ));
    }

    #[test]
    fn compose_http_endpoint_rejects_empty_api_key() {
        let config = rpc_config_with_http("https://lb.drpc.org", "\t");

        assert!(matches!(
            compose_http_endpoint(&config, ChainKey::Ethereum),
            Err(ClientEvmError::InvalidConfig {
                scope: ConfigScope::Http,
                ..
            })
        ));
    }

    proptest! {
        #[test]
        fn compose_ws_endpoint_preserves_non_empty_parts(
            ws_url in "[a-z]{3,10}://[a-z0-9.-]{1,30}/*",
            api_key in "[A-Za-z0-9_-]{1,40}",
        ) {
            let config = rpc_config(
                &format!(" {ws_url} "),
                &format!(" {api_key} "),
            );
            let expected = format!(
                "{}/ethereum/{}",
                ws_url.trim_end_matches('/'),
                api_key
            );

            prop_assert_eq!(compose_ws_endpoint(&config, ChainKey::Ethereum)?, expected);
        }

        #[test]
        fn compose_ws_endpoint_rejects_whitespace_ws_url(
            whitespace in "[ \\t\\n\\r]{1,12}",
            api_key in "[A-Za-z0-9_-]{1,40}",
        ) {
            let config = rpc_config(&whitespace, &api_key);

            let rejected = matches!(
                compose_ws_endpoint(&config, ChainKey::Ethereum),
                Err(ClientEvmError::InvalidConfig {
                    scope: ConfigScope::Subscription,
                    ..
                })
            );
            prop_assert!(rejected);
        }

        #[test]
        fn compose_ws_endpoint_rejects_whitespace_api_key(
            ws_url in "[a-z]{3,10}://[a-z0-9.-]{1,30}",
            whitespace in "[ \\t\\n\\r]{1,12}",
        ) {
            let config = rpc_config(&ws_url, &whitespace);

            let rejected = matches!(
                compose_ws_endpoint(&config, ChainKey::Ethereum),
                Err(ClientEvmError::InvalidConfig {
                    scope: ConfigScope::Subscription,
                    ..
                })
            );
            prop_assert!(rejected);
        }

        #[test]
        fn compose_ws_endpoint_ignores_http_url(
            http_url in "\\PC*",
            ws_url in "[a-z]{3,10}://[a-z0-9.-]{1,30}/*",
            api_key in "[A-Za-z0-9_-]{1,40}",
        ) {
            let mut config = rpc_config(&ws_url, &api_key);
            let expected = compose_ws_endpoint(&config, ChainKey::Ethereum)?;
            config.http_url = http_url;

            prop_assert_eq!(compose_ws_endpoint(&config, ChainKey::Ethereum)?, expected);
        }

        #[test]
        fn compose_http_endpoint_preserves_non_empty_parts(
            http_url in "[a-z]{3,10}://[a-z0-9.-]{1,30}/*",
            api_key in "[A-Za-z0-9_-]{1,40}",
        ) {
            let config = rpc_config_with_http(
                &format!(" {http_url} "),
                &format!(" {api_key} "),
            );
            let expected = format!(
                "{}/ethereum/{}",
                http_url.trim_end_matches('/'),
                api_key
            );

            prop_assert_eq!(compose_http_endpoint(&config, ChainKey::Ethereum)?, expected);
        }

        #[test]
        fn compose_http_endpoint_ignores_ws_url(
            ws_url in "\\PC*",
            http_url in "[a-z]{3,10}://[a-z0-9.-]{1,30}/*",
            api_key in "[A-Za-z0-9_-]{1,40}",
        ) {
            let mut config = rpc_config_with_http(&http_url, &api_key);
            let expected = compose_http_endpoint(&config, ChainKey::Ethereum)?;
            config.ws_url = ws_url;

            prop_assert_eq!(compose_http_endpoint(&config, ChainKey::Ethereum)?, expected);
        }
    }

    #[test]
    fn compose_graph_endpoint_substitutes_the_api_key() {
        let config = TheGraphConfig {
            url: " https://gateway.thegraph.com/api/{key}/subgraphs/id/v4 ".to_owned(),
            api_key: " graph-key ".to_owned(),
        };

        assert!(matches!(
            compose_graph_endpoint(&config).as_deref(),
            Ok("https://gateway.thegraph.com/api/graph-key/subgraphs/id/v4")
        ));
    }

    #[test]
    fn compose_graph_endpoint_uses_a_placeholderless_url_verbatim() {
        let config = TheGraphConfig {
            url: "https://self-hosted.example/subgraphs/name/v4".to_owned(),
            api_key: "unused".to_owned(),
        };

        assert!(matches!(
            compose_graph_endpoint(&config).as_deref(),
            Ok("https://self-hosted.example/subgraphs/name/v4")
        ));
    }

    #[test]
    fn compose_graph_endpoint_rejects_empty_url() {
        let config = TheGraphConfig {
            url: " ".to_owned(),
            api_key: "graph-key".to_owned(),
        };

        assert!(matches!(
            compose_graph_endpoint(&config),
            Err(ClientEvmError::InvalidConfig {
                scope: ConfigScope::Graph,
                ..
            })
        ));
    }

    #[test]
    fn compose_graph_endpoint_rejects_empty_api_key() {
        let config = TheGraphConfig {
            url: "https://gateway.thegraph.com/api/{key}/subgraphs/id/v4".to_owned(),
            api_key: "\t".to_owned(),
        };

        assert!(matches!(
            compose_graph_endpoint(&config),
            Err(ClientEvmError::InvalidConfig {
                scope: ConfigScope::Graph,
                ..
            })
        ));
    }

    fn rpc_config(ws_url: &str, api_key: &str) -> RpcConfig {
        RpcConfig {
            http_url: "https://lb.drpc.live/".to_owned(),
            ws_url: ws_url.to_owned(),
            api_key: api_key.to_owned(),
        }
    }

    fn rpc_config_with_http(http_url: &str, api_key: &str) -> RpcConfig {
        RpcConfig {
            http_url: http_url.to_owned(),
            ws_url: "wss://lb.drpc.live/".to_owned(),
            api_key: api_key.to_owned(),
        }
    }
}
