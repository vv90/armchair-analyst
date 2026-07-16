//! Candidate discovery: the top tokens per chain by total value locked, from the same Uniswap v4
//! subgraphs the runtime uses for pool metadata, unioned with the chain's hardcoded baseline
//! (USDC / native / WETH) so the optimizer's init asset and bridge endpoints can never be missing
//! from a generated artifact.

use std::collections::HashSet;

use client_evm::{
    Address, ChainKey, ClientEvmError, GraphEndpoints, TokenAddress, chain_key_for_network_path,
    send_graphql_request,
};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::examiner::TokenCandidate;

/// GraphQL query for the top-TVL `Token` entities. `decimals` is a BigInt and the USD figures are
/// BigDecimals — all serialized as decimal strings by the canonical schema.
const TOP_TOKENS_QUERY: &str = "query TopTokens($first: Int!) { \
tokens(first: $first, orderBy: totalValueLockedUSD, orderDirection: desc) \
{ id symbol decimals volumeUSD totalValueLockedUSD } }";
const GECKOTERMINAL_BASE_URL: &str = "https://api.geckoterminal.com/api/v2";
const UNISWAP_DEFAULT_TOKEN_LIST_URL: &str = "https://tokens.uniswap.org";

#[derive(Debug, Deserialize)]
struct TokensResponse {
    data: Option<TokensData>,
    errors: Option<Vec<TokensError>>,
}

#[derive(Debug, Deserialize)]
struct TokensData {
    tokens: Vec<TokenRow>,
}

#[derive(Debug, Deserialize)]
struct TokensError {
    message: String,
}

#[derive(Debug, Deserialize)]
struct TokenRow {
    id: Address,
    symbol: Option<String>,
    decimals: Option<String>,
    #[serde(rename = "volumeUSD")]
    volume_usd: Option<String>,
    #[serde(rename = "totalValueLockedUSD")]
    total_value_locked_usd: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GeckoTerminalResponse {
    #[serde(default)]
    included: Vec<GeckoTerminalIncluded>,
}

#[derive(Debug, Deserialize)]
struct GeckoTerminalIncluded {
    #[serde(rename = "type")]
    kind: String,
    attributes: GeckoTerminalTokenAttributes,
}

#[derive(Debug, Deserialize)]
struct GeckoTerminalTokenAttributes {
    address: Address,
    symbol: Option<String>,
    decimals: Option<u8>,
    coingecko_coin_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct UniswapDefaultTokenList {
    tokens: Vec<UniswapDefaultToken>,
}

#[derive(Debug, Deserialize)]
struct UniswapDefaultToken {
    #[serde(rename = "chainId")]
    chain_id: u64,
    address: String,
}

/// Fetches the `first` top-TVL tokens on `chain`. Returns `Ok(None)` when no subgraph is
/// configured for the chain (the caller decides whether that fails the run). Transport, HTTP, and
/// GraphQL-level failures fail the call after exhausting same-schema mirrors.
pub fn fetch_top_tokens(
    agent: &ureq::Agent,
    endpoints: &GraphEndpoints,
    chain: ChainKey,
    first: usize,
) -> Result<Option<Vec<TokenCandidate>>, ClientEvmError> {
    let Some(pool) = endpoints.pool(chain) else {
        return Ok(None);
    };

    let request = json!({ "query": TOP_TOKENS_QUERY, "variables": { "first": first } });

    pool.with_failover(|endpoint| {
        let value = send_graphql_request(agent, endpoint, &request)?;
        decode_tokens_response(value, chain)
    })
    .map(Some)
}

/// Fetches tokens appearing in GeckoTerminal's top pools for `chain`, page by page. This is a
/// secondary popularity signal for v3-heavy liquidity; approval still depends on contract vetting.
pub fn fetch_geckoterminal_top_pool_tokens(
    agent: &ureq::Agent,
    chain: ChainKey,
    pages: usize,
) -> Vec<TokenCandidate> {
    let Some(network) = geckoterminal_network_id(chain) else {
        return Vec::new();
    };

    let mut tokens = Vec::new();
    for page in 1..=pages {
        let url = format!(
            "{GECKOTERMINAL_BASE_URL}/networks/{network}/pools?include=base_token,quote_token&page={page}"
        );
        let Ok(mut response) = agent.get(&url).call() else {
            continue;
        };
        let Ok(value) = response.body_mut().read_json::<Value>() else {
            continue;
        };
        if let Ok(mut decoded) = decode_geckoterminal_tokens(value, chain) {
            tokens.append(&mut decoded);
        }
    }

    tokens
}

/// Fetches Uniswap's default-token-list as a trust set. This function deliberately returns only a
/// set for corroborating already-discovered candidates; it should not by itself expand the whitelist
/// candidate universe.
pub fn fetch_uniswap_default_token_set(
    agent: &ureq::Agent,
) -> Result<HashSet<TokenAddress>, ClientEvmError> {
    let mut response = agent
        .get(UNISWAP_DEFAULT_TOKEN_LIST_URL)
        .call()
        .map_err(ClientEvmError::HttpTransport)?;
    let value = response
        .body_mut()
        .read_json::<Value>()
        .map_err(|error| malformed(error.to_string()))?;
    decode_uniswap_default_token_set(value)
}

/// Decodes a top-tokens response. A GraphQL `errors` payload or a response with neither `data` nor
/// `errors` is a response-level error; advisory fields that fail to parse are dropped per token
/// (the address is the only load-bearing field), never a failure.
fn decode_tokens_response(
    value: Value,
    chain: ChainKey,
) -> Result<Vec<TokenCandidate>, ClientEvmError> {
    let response: TokensResponse =
        serde_json::from_value(value).map_err(|error| malformed(error.to_string()))?;

    if let Some(errors) = response.errors
        && !errors.is_empty()
    {
        let detail = errors
            .into_iter()
            .map(|error| error.message)
            .collect::<Vec<_>>()
            .join("; ");
        return Err(malformed(detail));
    }

    let data = response
        .data
        .ok_or_else(|| malformed("response has neither data nor errors".to_owned()))?;

    Ok(data
        .tokens
        .into_iter()
        .filter_map(|row| {
            let tvl_usd = row
                .total_value_locked_usd
                .and_then(|value| value.parse::<f64>().ok())
                .filter(|value| value.is_finite() && *value > 0.0)?;
            Some(TokenCandidate {
                token: TokenAddress(row.id, chain),
                symbol: row.symbol,
                decimals: row.decimals.and_then(|value| value.parse().ok()),
                volume_usd: row.volume_usd.and_then(|value| value.parse().ok()),
                tvl_usd: Some(tvl_usd),
                trusted_listing: false,
            })
        })
        .collect())
}

fn decode_geckoterminal_tokens(
    value: Value,
    chain: ChainKey,
) -> Result<Vec<TokenCandidate>, ClientEvmError> {
    let response: GeckoTerminalResponse =
        serde_json::from_value(value).map_err(|error| malformed(error.to_string()))?;

    Ok(response
        .included
        .into_iter()
        .filter(|included| included.kind == "token")
        .map(|included| TokenCandidate {
            token: TokenAddress(included.attributes.address, chain),
            symbol: included.attributes.symbol,
            decimals: included.attributes.decimals,
            volume_usd: None,
            tvl_usd: None,
            trusted_listing: included.attributes.coingecko_coin_id.is_some(),
        })
        .collect())
}

fn decode_uniswap_default_token_set(value: Value) -> Result<HashSet<TokenAddress>, ClientEvmError> {
    let list: UniswapDefaultTokenList =
        serde_json::from_value(value).map_err(|error| malformed(error.to_string()))?;
    Ok(list
        .tokens
        .into_iter()
        .filter_map(|token| {
            let chain = chain_for_chain_id(token.chain_id)?;
            let address = token.address.parse().ok()?;
            Some(TokenAddress(address, chain))
        })
        .collect())
}

fn malformed(detail: String) -> ClientEvmError {
    ClientEvmError::MalformedResponse {
        context: "graph tokens".to_owned(),
        detail,
    }
}

/// The chain's always-proposed baseline: its USDC (the init-asset/bridge hub currency), the native
/// pseudo-token (zero address, as v4 native pools key it), and WETH — so a generated artifact can
/// never omit the tokens the optimizer's session config depends on.
pub fn baseline_candidates(chain: ChainKey) -> Vec<TokenCandidate> {
    use client_evm::*;

    let (usdc, native, weth) = match chain {
        ChainKey::Ethereum => (
            ETHEREUM_USDC_TOKEN_ADDRESS,
            ETHEREUM_NATIVE_TOKEN_ADDRESS,
            ETHEREUM_WETH_TOKEN_ADDRESS,
        ),
        ChainKey::Arbitrum => (
            ARBITRUM_USDC_TOKEN_ADDRESS,
            ARBITRUM_NATIVE_TOKEN_ADDRESS,
            ARBITRUM_WETH_TOKEN_ADDRESS,
        ),
        ChainKey::Base => (
            BASE_USDC_TOKEN_ADDRESS,
            BASE_NATIVE_TOKEN_ADDRESS,
            BASE_WETH_TOKEN_ADDRESS,
        ),
        ChainKey::Optimism => (
            OPTIMISM_USDC_TOKEN_ADDRESS,
            OPTIMISM_NATIVE_TOKEN_ADDRESS,
            OPTIMISM_WETH_TOKEN_ADDRESS,
        ),
        ChainKey::Polygon => (
            POLYGON_USDC_TOKEN_ADDRESS,
            POLYGON_NATIVE_TOKEN_ADDRESS,
            POLYGON_WETH_TOKEN_ADDRESS,
        ),
        ChainKey::Bnb => (
            BNB_USDC_TOKEN_ADDRESS,
            BNB_NATIVE_TOKEN_ADDRESS,
            BNB_WETH_TOKEN_ADDRESS,
        ),
        ChainKey::Avalanche => (
            AVALANCHE_USDC_TOKEN_ADDRESS,
            AVALANCHE_NATIVE_TOKEN_ADDRESS,
            AVALANCHE_WETH_TOKEN_ADDRESS,
        ),
    };

    [usdc, native, weth]
        .into_iter()
        .map(|token| TokenCandidate {
            token,
            symbol: None,
            decimals: None,
            tvl_usd: None,
            volume_usd: None,
            trusted_listing: true,
        })
        .collect()
}

fn geckoterminal_network_id(chain: ChainKey) -> Option<&'static str> {
    match chain {
        ChainKey::Ethereum => Some("eth"),
        ChainKey::Arbitrum => Some("arbitrum"),
        ChainKey::Base => Some("base"),
        ChainKey::Optimism => Some("optimism"),
        ChainKey::Polygon => Some("polygon_pos"),
        ChainKey::Bnb => Some("bsc"),
        ChainKey::Avalanche => Some("avax"),
    }
}

fn chain_for_chain_id(chain_id: u64) -> Option<ChainKey> {
    match chain_id {
        1 => chain_key_for_network_path("ethereum"),
        10 => chain_key_for_network_path("optimism"),
        56 => chain_key_for_network_path("bnb"),
        137 => chain_key_for_network_path("polygon"),
        8453 => chain_key_for_network_path("base"),
        42161 => chain_key_for_network_path("arbitrum"),
        43114 => chain_key_for_network_path("avalanche"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use client_evm::ACTIVE_CHAINS;

    fn usdc_row() -> Value {
        json!({
            "id": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
            "symbol": "USDC",
            "decimals": "6",
            "volumeUSD": "123456789.5",
            "totalValueLockedUSD": "98765432.25",
        })
    }

    #[test]
    fn decodes_a_token_row_with_parsed_advisory_fields() {
        let value = json!({ "data": { "tokens": [usdc_row()] } });

        let candidates = decode_tokens_response(value, ChainKey::Ethereum).expect("decodes");

        assert_eq!(candidates.len(), 1);
        let candidate = &candidates[0];
        assert_eq!(
            candidate.token,
            client_evm::ETHEREUM_USDC_TOKEN_ADDRESS,
            "id + chain must form the TokenAddress"
        );
        assert_eq!(candidate.symbol.as_deref(), Some("USDC"));
        assert_eq!(candidate.decimals, Some(6));
        assert_eq!(candidate.volume_usd, Some(123456789.5));
        assert_eq!(candidate.tvl_usd, Some(98765432.25));
    }

    #[test]
    fn unparseable_advisory_fields_are_dropped_not_fatal() {
        let value = json!({ "data": { "tokens": [{
            "id": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
            "symbol": "USDC",
            "decimals": "not-a-number",
            "volumeUSD": "-",
            "totalValueLockedUSD": "98765432.25",
        }] } });

        let candidates = decode_tokens_response(value, ChainKey::Ethereum).expect("decodes");

        assert_eq!(candidates[0].decimals, None);
        assert_eq!(candidates[0].volume_usd, None);
        assert_eq!(candidates[0].tvl_usd, Some(98765432.25));
    }

    #[test]
    fn subgraph_candidates_require_positive_parsed_tvl() {
        let value = json!({ "data": { "tokens": [
            usdc_row(),
            {
                "id": "0x0000000000000000000000000000000000000001",
                "symbol": "ZERO",
                "decimals": "18",
                "volumeUSD": "1",
                "totalValueLockedUSD": "0",
            },
            {
                "id": "0x0000000000000000000000000000000000000002",
                "symbol": "BAD",
                "decimals": "18",
                "volumeUSD": "1",
                "totalValueLockedUSD": "-",
            },
            {
                "id": "0x0000000000000000000000000000000000000003",
                "symbol": "MISSING",
                "decimals": "18",
                "volumeUSD": "1",
                "totalValueLockedUSD": null,
            }
        ] } });

        let candidates = decode_tokens_response(value, ChainKey::Ethereum).expect("decodes");

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].token, client_evm::ETHEREUM_USDC_TOKEN_ADDRESS);
    }

    #[test]
    fn graphql_errors_fail_the_decode() {
        let value = json!({ "errors": [{ "message": "bad indexers" }] });

        let error = decode_tokens_response(value, ChainKey::Ethereum)
            .expect_err("errors payload must fail");

        assert!(matches!(error, ClientEvmError::MalformedResponse { .. }));
    }

    #[test]
    fn response_without_data_or_errors_fails_the_decode() {
        let error = decode_tokens_response(json!({}), ChainKey::Ethereum)
            .expect_err("empty response must fail");

        assert!(matches!(error, ClientEvmError::MalformedResponse { .. }));
    }

    #[test]
    fn no_subgraph_for_chain_returns_none_without_a_request() {
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_top_tokens(&agent, &GraphEndpoints::empty(), ChainKey::Ethereum, 5)
            .expect("fetch succeeds");

        assert!(result.is_none());
    }

    #[test]
    fn every_active_chain_has_a_three_token_baseline_on_its_own_chain() {
        for &chain in ACTIVE_CHAINS {
            let baseline = baseline_candidates(chain);
            assert_eq!(baseline.len(), 3, "chain {chain:?}");
            assert!(
                baseline.iter().all(|candidate| candidate.token.1 == chain),
                "chain {chain:?} baseline tokens must live on that chain"
            );
        }
    }

    #[test]
    fn geckoterminal_network_ids_cover_every_active_chain() {
        for &chain in ACTIVE_CHAINS {
            assert!(geckoterminal_network_id(chain).is_some(), "chain {chain:?}");
        }
    }

    #[test]
    fn decodes_geckoterminal_included_tokens() {
        let value = json!({
            "included": [{
                "id": "eth_0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
                "type": "token",
                "attributes": {
                    "address": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
                    "symbol": "USDC",
                    "decimals": 6
                }
            }]
        });

        let candidates = decode_geckoterminal_tokens(value, ChainKey::Ethereum).expect("decodes");

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].token, client_evm::ETHEREUM_USDC_TOKEN_ADDRESS);
        assert_eq!(candidates[0].symbol.as_deref(), Some("USDC"));
        assert_eq!(candidates[0].decimals, Some(6));
    }

    #[test]
    fn decodes_uniswap_default_token_set_for_active_chains_only() {
        let value = json!({
            "tokens": [
                {
                    "chainId": 1,
                    "address": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
                },
                {
                    "chainId": 999999,
                    "address": "0x0000000000000000000000000000000000000001"
                }
            ]
        });

        let trusted = decode_uniswap_default_token_set(value).expect("decodes");

        assert!(trusted.contains(&client_evm::ETHEREUM_USDC_TOKEN_ADDRESS));
        assert_eq!(trusted.len(), 1);
    }

    #[test]
    fn uniswap_default_token_set_skips_malformed_addresses() {
        let value = json!({
            "tokens": [
                {
                    "chainId": 1,
                    "address": "not-an-address"
                },
                {
                    "chainId": 1,
                    "address": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
                }
            ]
        });

        let trusted = decode_uniswap_default_token_set(value).expect("decodes");

        assert!(trusted.contains(&client_evm::ETHEREUM_USDC_TOKEN_ADDRESS));
        assert_eq!(trusted.len(), 1);
    }
}
