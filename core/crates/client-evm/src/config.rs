use crate::{
    chain::{ChainKey, drpc_network_path},
    error::ClientEvmError,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RpcConfig {
    pub chain: ChainKey,
    pub http_url: String,
    pub ws_url: String,
    pub api_key: String,
}

pub(crate) fn compose_ws_endpoint(config: &RpcConfig) -> Result<String, ClientEvmError> {
    let ws_url = config.ws_url.trim();
    if ws_url.is_empty() {
        return Err(ClientEvmError::InvalidSubscriptionConfig(
            "websocket url is required".to_owned(),
        ));
    }

    let api_key = config.api_key.trim();
    if api_key.is_empty() {
        return Err(ClientEvmError::InvalidSubscriptionConfig(
            "rpc api key is required".to_owned(),
        ));
    }

    Ok(format!(
        "{}/{}/{}",
        ws_url.trim_end_matches('/'),
        drpc_network_path(config.chain),
        api_key
    ))
}

pub(crate) fn compose_http_endpoint(config: &RpcConfig) -> Result<String, ClientEvmError> {
    let http_url = config.http_url.trim();
    if http_url.is_empty() {
        return Err(ClientEvmError::InvalidHttpConfig(
            "http url is required".to_owned(),
        ));
    }

    let api_key = config.api_key.trim();
    if api_key.is_empty() {
        return Err(ClientEvmError::InvalidHttpConfig(
            "rpc api key is required".to_owned(),
        ));
    }

    Ok(format!(
        "{}/{}/{}",
        http_url.trim_end_matches('/'),
        drpc_network_path(config.chain),
        api_key
    ))
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use crate::{ChainKey, ClientEvmError};

    use super::*;

    #[test]
    fn compose_ws_endpoint_appends_network_and_key() {
        let config = rpc_config(" wss://lb.drpc.org/ ", " api-key ");

        let result = compose_ws_endpoint(&config);

        assert!(matches!(
            result.as_deref(),
            Ok("wss://lb.drpc.org/ethereum/api-key")
        ));
    }

    #[test]
    fn compose_ws_endpoint_rejects_empty_ws_url() {
        let config = rpc_config(" ", "api-key");

        assert!(matches!(
            compose_ws_endpoint(&config),
            Err(ClientEvmError::InvalidSubscriptionConfig(_))
        ));
    }

    #[test]
    fn compose_ws_endpoint_rejects_empty_api_key() {
        let config = rpc_config("wss://lb.drpc.org", "\t");

        assert!(matches!(
            compose_ws_endpoint(&config),
            Err(ClientEvmError::InvalidSubscriptionConfig(_))
        ));
    }

    #[test]
    fn compose_http_endpoint_appends_network_and_key() {
        let config = rpc_config_with_http(" https://lb.drpc.org/ ", " api-key ");

        let result = compose_http_endpoint(&config);

        assert!(matches!(
            result.as_deref(),
            Ok("https://lb.drpc.org/ethereum/api-key")
        ));
    }

    #[test]
    fn compose_http_endpoint_rejects_empty_http_url() {
        let config = rpc_config_with_http(" ", "api-key");

        assert!(matches!(
            compose_http_endpoint(&config),
            Err(ClientEvmError::InvalidHttpConfig(_))
        ));
    }

    #[test]
    fn compose_http_endpoint_rejects_empty_api_key() {
        let config = rpc_config_with_http("https://lb.drpc.org", "\t");

        assert!(matches!(
            compose_http_endpoint(&config),
            Err(ClientEvmError::InvalidHttpConfig(_))
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

            prop_assert_eq!(compose_ws_endpoint(&config)?, expected);
        }

        #[test]
        fn compose_ws_endpoint_rejects_whitespace_ws_url(
            whitespace in "[ \\t\\n\\r]{1,12}",
            api_key in "[A-Za-z0-9_-]{1,40}",
        ) {
            let config = rpc_config(&whitespace, &api_key);

            prop_assert!(matches!(
                compose_ws_endpoint(&config),
                Err(ClientEvmError::InvalidSubscriptionConfig(_))
            ));
        }

        #[test]
        fn compose_ws_endpoint_rejects_whitespace_api_key(
            ws_url in "[a-z]{3,10}://[a-z0-9.-]{1,30}",
            whitespace in "[ \\t\\n\\r]{1,12}",
        ) {
            let config = rpc_config(&ws_url, &whitespace);

            prop_assert!(matches!(
                compose_ws_endpoint(&config),
                Err(ClientEvmError::InvalidSubscriptionConfig(_))
            ));
        }

        #[test]
        fn compose_ws_endpoint_ignores_http_url(
            http_url in "\\PC*",
            ws_url in "[a-z]{3,10}://[a-z0-9.-]{1,30}/*",
            api_key in "[A-Za-z0-9_-]{1,40}",
        ) {
            let mut config = rpc_config(&ws_url, &api_key);
            let expected = compose_ws_endpoint(&config)?;
            config.http_url = http_url;

            prop_assert_eq!(compose_ws_endpoint(&config)?, expected);
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

            prop_assert_eq!(compose_http_endpoint(&config)?, expected);
        }

        #[test]
        fn compose_http_endpoint_ignores_ws_url(
            ws_url in "\\PC*",
            http_url in "[a-z]{3,10}://[a-z0-9.-]{1,30}/*",
            api_key in "[A-Za-z0-9_-]{1,40}",
        ) {
            let mut config = rpc_config_with_http(&http_url, &api_key);
            let expected = compose_http_endpoint(&config)?;
            config.ws_url = ws_url;

            prop_assert_eq!(compose_http_endpoint(&config)?, expected);
        }
    }

    fn rpc_config(ws_url: &str, api_key: &str) -> RpcConfig {
        RpcConfig {
            chain: ChainKey::Ethereum,
            http_url: "https://lb.drpc.live/".to_owned(),
            ws_url: ws_url.to_owned(),
            api_key: api_key.to_owned(),
        }
    }

    fn rpc_config_with_http(http_url: &str, api_key: &str) -> RpcConfig {
        RpcConfig {
            chain: ChainKey::Ethereum,
            http_url: http_url.to_owned(),
            ws_url: "wss://lb.drpc.live/".to_owned(),
            api_key: api_key.to_owned(),
        }
    }
}
