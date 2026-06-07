use client_evm::{
    ClientEvmError, ClientHead, RpcConfig, fetch_finalized_block_header,
    kernel::{FinalizedState, State},
};

use crate::utils::CliError;

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn initialize_client_evm_state(
    agent: &ureq::Agent,
    config: &RpcConfig,
) -> Result<State, CliError> {
    initialize_client_evm_state_with(agent, config, fetch_finalized_block_header)
}

fn initialize_client_evm_state_with<Fetch>(
    agent: &ureq::Agent,
    config: &RpcConfig,
    fetch: Fetch,
) -> Result<State, CliError>
where
    Fetch: FnOnce(&ureq::Agent, &RpcConfig) -> Result<Option<ClientHead>, ClientEvmError>,
{
    let header = fetch(agent, config).map_err(|error| CliError::ClientInitializationFailed {
        message: format!("failed to fetch finalized block header: {error}"),
    })?;

    header
        .map(finalized_state_from_header)
        .map(State::init)
        .ok_or_else(|| CliError::ClientInitializationFailed {
            message: "finalized block header was not returned".to_owned(),
        })
}

fn finalized_state_from_header(header: ClientHead) -> FinalizedState {
    FinalizedState::empty_at(header.inner.hash)
}

#[cfg(test)]
mod tests {
    use client_evm::{ChainKey, ClientEvmError, ClientHead, RpcConfig};
    use serde_json::json;

    use super::*;

    #[test]
    fn production_initializer_has_expected_signature() {
        let _initializer: fn(&ureq::Agent, &RpcConfig) -> Result<State, CliError> =
            initialize_client_evm_state;
    }

    #[test]
    fn finalized_state_from_header_uses_header_hash() -> Result<(), serde_json::Error> {
        let header = finalized_header()?;

        let finalized_state = finalized_state_from_header(header);

        assert_eq!(
            finalized_state.block_hash.to_string(),
            "0x0000000000000000000000000000000000000000000000000000000000000001"
        );

        Ok(())
    }

    #[test]
    fn initialize_client_evm_state_with_successful_fetch_returns_state()
    -> Result<(), serde_json::Error> {
        let header = finalized_header()?;
        let agent = ureq::Agent::new_with_defaults();
        let config = rpc_config();

        let result =
            initialize_client_evm_state_with(&agent, &config, |_agent, _config| Ok(Some(header)));

        assert!(result.is_ok());

        Ok(())
    }

    #[test]
    fn initialize_client_evm_state_with_missing_header_returns_initialization_error() {
        let agent = ureq::Agent::new_with_defaults();
        let config = rpc_config();

        let result = initialize_client_evm_state_with(&agent, &config, |_agent, _config| Ok(None));

        assert!(matches!(
            result,
            Err(CliError::ClientInitializationFailed {
                ref message,
            }) if message == "finalized block header was not returned"
        ));
    }

    #[test]
    fn initialize_client_evm_state_with_fetch_error_returns_initialization_error() {
        let agent = ureq::Agent::new_with_defaults();
        let config = rpc_config();

        let result = initialize_client_evm_state_with(&agent, &config, |_agent, _config| {
            Err(ClientEvmError::InvalidHttpConfig("bad config".to_owned()))
        });

        assert!(matches!(
            result,
            Err(CliError::ClientInitializationFailed {
                ref message,
            }) if message
                == "failed to fetch finalized block header: invalid http config: bad config"
        ));
    }

    fn rpc_config() -> RpcConfig {
        RpcConfig {
            chain: ChainKey::Ethereum,
            http_url: "https://example.invalid/http".to_owned(),
            ws_url: "wss://example.invalid/ws".to_owned(),
            api_key: "api-key".to_owned(),
        }
    }

    fn finalized_header() -> Result<ClientHead, serde_json::Error> {
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
}
