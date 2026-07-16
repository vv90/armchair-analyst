//! The token-examination seam. Contract-aware examiners plug in behind [`TokenExaminer`] without
//! touching discovery or the artifact writer, so the whitelist can get stricter independently of the
//! main binary.

use client_evm::{
    Address, ChainEndpoints, ChainKey, ClientEvmError, EndpointPool, TokenAddress,
};
use serde_json::{Value, json};

const HTTP_REQUEST_ID: u64 = 1;
const DECIMALS_SELECTOR: &str = "0x313ce567";
const MIN_VERIFIED_TOKEN_TVL_USD: f64 = 10_000.0;
const EIP1967_IMPLEMENTATION_SLOT: &str =
    "0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc";
const EIP1967_BEACON_SLOT: &str =
    "0xa3f0ad74e5423aebfd80d3ef4346578335a9a72aeaee59ff6cb3582b35133d50";

/// A token proposed for whitelisting, with the subgraph-sourced context an examiner (or a human
/// reviewing the artifact) can weigh. Only `token` is identity; the rest is advisory.
#[derive(Clone, Debug, PartialEq)]
pub struct TokenCandidate {
    pub token: TokenAddress,
    pub symbol: Option<String>,
    pub decimals: Option<u8>,
    pub tvl_usd: Option<f64>,
    pub volume_usd: Option<f64>,
    pub trusted_listing: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExaminationVerdict {
    Approved {
        reason: String,
        decimals: Option<u8>,
    },
    Rejected { reason: String },
}

pub trait TokenExaminer {
    fn examine(&self, candidate: &TokenCandidate) -> ExaminationVerdict;

    /// Name/version recorded in the artifact's `examiner` provenance field.
    fn label(&self) -> String;
}

/// Placeholder examiner that approves every candidate. Exists so the whitelisting pipeline is
/// complete end to end before any real examination logic lands.
pub struct ApproveAll;

impl TokenExaminer for ApproveAll {
    fn examine(&self, _candidate: &TokenCandidate) -> ExaminationVerdict {
        ExaminationVerdict::Approved {
            reason: "approve-all".to_owned(),
            decimals: None,
        }
    }

    fn label(&self) -> String {
        "approve-all/0.1.0".to_owned()
    }
}

pub struct ContractExaminer<'a> {
    agent: &'a ureq::Agent,
    endpoints: &'a ChainEndpoints,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ContractEvidence {
    code_present: bool,
    decimals: Option<u8>,
    source_verified: bool,
    source_flags: Vec<String>,
    proxy_target: Option<Address>,
}

impl<'a> ContractExaminer<'a> {
    pub fn new(agent: &'a ureq::Agent, endpoints: &'a ChainEndpoints) -> ContractExaminer<'a> {
        ContractExaminer { agent, endpoints }
    }

    fn inspect(&self, token: TokenAddress) -> Result<ContractEvidence, String> {
        let pool = self
            .endpoints
            .pool(token.1)
            .map_err(|error| error.to_string())?;

        let code = eth_get_code(self.agent, pool, token.0).map_err(|error| error.to_string())?;
        let code_present = has_runtime_code(&code);
        let decimals = eth_call_decimals(self.agent, pool, token.0).ok().flatten();
        let proxy_target = proxy_target(self.agent, pool, token.0, &code);
        let source = fetch_sourcify_source(self.agent, token);
        let source_flags = source
            .as_deref()
            .map(source_risk_flags)
            .unwrap_or_default();

        Ok(ContractEvidence {
            code_present,
            decimals,
            source_verified: source.is_some(),
            source_flags,
            proxy_target,
        })
    }
}

impl TokenExaminer for ContractExaminer<'_> {
    fn examine(&self, candidate: &TokenCandidate) -> ExaminationVerdict {
        if candidate.token.0 == Address::ZERO {
            return ExaminationVerdict::Approved {
                reason: "native currency pseudo-token".to_owned(),
                decimals: Some(18),
            };
        }

        let evidence = match self.inspect(candidate.token) {
            Ok(evidence) => evidence,
            Err(error) => {
                return ExaminationVerdict::Rejected {
                    reason: format!("contract inspection failed: {error}"),
                };
            }
        };

        if let Some(reason) = managed_exception_reason(candidate.token) {
            return ExaminationVerdict::Approved {
                reason: format!("managed blue-chip exception: {reason}"),
                decimals: evidence.decimals,
            };
        }

        if !evidence.code_present {
            return ExaminationVerdict::Rejected {
                reason: "no runtime bytecode".to_owned(),
            };
        }

        let Some(decimals) = evidence.decimals else {
            return ExaminationVerdict::Rejected {
                reason: "decimals() missing or undecodable".to_owned(),
            };
        };

        if let Some(candidate_decimals) = candidate.decimals
            && candidate_decimals != decimals
        {
            return ExaminationVerdict::Rejected {
                reason: format!(
                    "candidate decimals {candidate_decimals} disagree with on-chain decimals {decimals}"
                ),
            };
        }

        if let Some(target) = evidence.proxy_target {
            return ExaminationVerdict::Rejected {
                reason: format!("proxy or beacon target detected at {target}"),
            };
        }

        if !evidence.source_verified {
            return ExaminationVerdict::Rejected {
                reason: "source not verified on Sourcify".to_owned(),
            };
        }

        if !candidate.trusted_listing {
            return ExaminationVerdict::Rejected {
                reason: "not corroborated by trusted token source".to_owned(),
            };
        }

        if !has_minimum_popularity(candidate) {
            return ExaminationVerdict::Rejected {
                reason: format!(
                    "insufficient popularity signal: requires at least {MIN_VERIFIED_TOKEN_TVL_USD:.0} USD TVL or known-popular exception"
                ),
            };
        }

        if !evidence.source_flags.is_empty() {
            return ExaminationVerdict::Rejected {
                reason: format!("source risk flags: {}", evidence.source_flags.join(",")),
            };
        }

        let reason = known_popular_plain_token_reason(candidate.token)
            .map(|known| format!("verified plain ERC20-compatible token: {known}"))
            .unwrap_or_else(|| "verified plain ERC20-compatible token".to_owned());

        ExaminationVerdict::Approved {
            reason,
            decimals: Some(decimals),
        }
    }

    fn label(&self) -> String {
        "contract-examiner/0.1.0".to_owned()
    }
}

fn eth_get_code(
    agent: &ureq::Agent,
    pool: &EndpointPool,
    address: Address,
) -> Result<String, ClientEvmError> {
    rpc_string_result(
        agent,
        pool,
        json!({
            "jsonrpc": "2.0",
            "id": HTTP_REQUEST_ID,
            "method": "eth_getCode",
            "params": [address, "latest"],
        }),
        "eth_getCode",
    )
}

fn eth_call_decimals(
    agent: &ureq::Agent,
    pool: &EndpointPool,
    address: Address,
) -> Result<Option<u8>, ClientEvmError> {
    let result = rpc_string_result(
        agent,
        pool,
        json!({
            "jsonrpc": "2.0",
            "id": HTTP_REQUEST_ID,
            "method": "eth_call",
            "params": [{
                "to": address,
                "data": DECIMALS_SELECTOR,
            }, "latest"],
        }),
        "eth_call decimals",
    )?;

    Ok(decode_decimals_result(&result))
}

fn eth_get_storage_at(
    agent: &ureq::Agent,
    pool: &EndpointPool,
    address: Address,
    slot: &str,
) -> Result<String, ClientEvmError> {
    rpc_string_result(
        agent,
        pool,
        json!({
            "jsonrpc": "2.0",
            "id": HTTP_REQUEST_ID,
            "method": "eth_getStorageAt",
            "params": [address, slot, "latest"],
        }),
        "eth_getStorageAt",
    )
}

fn rpc_string_result(
    agent: &ureq::Agent,
    pool: &EndpointPool,
    request: Value,
    context: &str,
) -> Result<String, ClientEvmError> {
    let response = pool.with_failover(|endpoint| send_rpc_request(agent, endpoint, &request))?;
    if let Some(error) = response.get("error") {
        return Err(ClientEvmError::MalformedResponse {
            context: context.to_owned(),
            detail: error.to_string(),
        });
    }

    response
        .get("result")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| ClientEvmError::MalformedResponse {
            context: context.to_owned(),
            detail: "missing string result".to_owned(),
        })
}

fn send_rpc_request(
    agent: &ureq::Agent,
    endpoint: &str,
    request: &impl serde::Serialize,
) -> Result<Value, ClientEvmError> {
    let mut response = agent
        .post(endpoint)
        .config()
        .http_status_as_error(false)
        .build()
        .send_json(request)
        .map_err(ClientEvmError::HttpTransport)?;

    let status = response.status();
    if !status.is_success() {
        let body = response
            .body_mut()
            .read_to_string()
            .unwrap_or_else(|error| format!("<unreadable body: {error}>"));
        return Err(ClientEvmError::HttpStatus {
            status: status.as_u16(),
            body,
        });
    }

    response
        .body_mut()
        .read_json::<Value>()
        .map_err(|error| ClientEvmError::MalformedResponse {
            context: "vetting rpc json".to_owned(),
            detail: error.to_string(),
        })
}

fn proxy_target(
    agent: &ureq::Agent,
    pool: &EndpointPool,
    address: Address,
    code: &str,
) -> Option<Address> {
    minimal_proxy_target(code)
        .or_else(|| {
            eth_get_storage_at(agent, pool, address, EIP1967_IMPLEMENTATION_SLOT)
                .ok()
                .and_then(|word| address_from_storage_word(&word))
        })
        .or_else(|| {
            eth_get_storage_at(agent, pool, address, EIP1967_BEACON_SLOT)
                .ok()
                .and_then(|word| address_from_storage_word(&word))
        })
        .filter(|target| *target != Address::ZERO)
}

fn minimal_proxy_target(code: &str) -> Option<Address> {
    let hex = strip_0x(code).to_ascii_lowercase();
    let prefix = "363d3d373d3d3d363d73";
    let suffix = "5af43d82803e903d91602b57fd5bf3";
    if hex.len() == 90 && hex.starts_with(prefix) && hex.ends_with(suffix) {
        let start = prefix.len();
        let end = start + 40;
        hex.get(start..end).and_then(parse_address_hex)
    } else {
        None
    }
}

fn has_runtime_code(code: &str) -> bool {
    let stripped = strip_0x(code);
    !stripped.is_empty() && stripped != "0"
}

fn address_from_storage_word(word: &str) -> Option<Address> {
    let hex = strip_0x(word);
    if hex.len() < 40 {
        return None;
    }
    hex.get(hex.len().saturating_sub(40)..)
        .and_then(parse_address_hex)
}

fn parse_address_hex(hex: &str) -> Option<Address> {
    format!("0x{hex}").parse().ok()
}

fn decode_decimals_result(word: &str) -> Option<u8> {
    let significant = strip_0x(word).trim_start_matches('0');
    if significant.is_empty() {
        return Some(0);
    }
    if significant.len() > 4 {
        return None;
    }
    u16::from_str_radix(significant, 16)
        .ok()
        .and_then(|value| u8::try_from(value).ok())
        .filter(|value| *value <= 36)
}

fn source_risk_flags(source: &str) -> Vec<String> {
    let source = source.to_ascii_lowercase();
    let mut flags = Vec::new();
    push_flag(
        &mut flags,
        &source,
        "transfer-tax",
        &[
            "buytax",
            "selltax",
            "transfertax",
            "transferfee",
            "taxfee",
            "feeontransfer",
            "takefee",
            "_tax",
            "liquidityfee",
            "marketingfee",
        ],
    );
    push_flag(
        &mut flags,
        &source,
        "blacklist",
        &["blacklist", "isblacklisted", "botlist"],
    );
    push_flag(
        &mut flags,
        &source,
        "trading-gate",
        &[
            "tradingenabled",
            "enabletrading",
            "tradingopen",
            "cooldown",
            "maxwallet",
            "maxtransaction",
            "maxtx",
            "antibot",
        ],
    );
    push_flag(
        &mut flags,
        &source,
        "rebase-or-reflection",
        &["rebase", "reflection", "_rowned", "_towned", "deliver("],
    );
    push_flag(
        &mut flags,
        &source,
        "pausable",
        &["pausable", "paused", "pause()"],
    );
    push_flag(
        &mut flags,
        &source,
        "upgradeable",
        &["delegatecall", "upgradeability", "upgradeable", "implementation"],
    );
    flags
}

fn push_flag(flags: &mut Vec<String>, source: &str, flag: &str, needles: &[&str]) {
    if needles.iter().any(|needle| source.contains(needle)) {
        flags.push(flag.to_owned());
    }
}

fn fetch_sourcify_source(agent: &ureq::Agent, token: TokenAddress) -> Option<String> {
    let chain_id = sourcify_chain_id(token.1);
    let url = format!(
        "https://sourcify.dev/server/v2/contract/{chain_id}/{}?fields=all",
        token.0
    );
    let mut response = agent
        .get(&url)
        .config()
        .http_status_as_error(false)
        .build()
        .call()
        .ok()?;

    if !response.status().is_success() {
        return None;
    }

    let value = response.body_mut().read_json::<Value>().ok()?;
    let sources = value.get("sources")?.as_object()?;
    let mut combined = String::new();
    for source in sources.values() {
        if let Some(content) = source.get("content").and_then(Value::as_str) {
            combined.push_str(content);
            combined.push('\n');
        }
    }

    (!combined.trim().is_empty()).then_some(combined)
}

fn sourcify_chain_id(chain: ChainKey) -> u64 {
    match chain {
        ChainKey::Ethereum => 1,
        ChainKey::Arbitrum => 42161,
        ChainKey::Base => 8453,
        ChainKey::Optimism => 10,
        ChainKey::Polygon => 137,
        ChainKey::Bnb => 56,
        ChainKey::Avalanche => 43114,
    }
}

fn managed_exception_reason(token: TokenAddress) -> Option<&'static str> {
    use client_evm::{
        ARBITRUM_USDC_TOKEN_ADDRESS, ARBITRUM_WETH_TOKEN_ADDRESS,
        AVALANCHE_USDC_TOKEN_ADDRESS, AVALANCHE_WETH_TOKEN_ADDRESS, BASE_USDC_TOKEN_ADDRESS,
        BASE_WETH_TOKEN_ADDRESS, BNB_USDC_TOKEN_ADDRESS, BNB_WETH_TOKEN_ADDRESS,
        ETHEREUM_USDC_TOKEN_ADDRESS, ETHEREUM_WETH_TOKEN_ADDRESS, OPTIMISM_USDC_TOKEN_ADDRESS,
        OPTIMISM_WETH_TOKEN_ADDRESS, POLYGON_USDC_TOKEN_ADDRESS, POLYGON_WETH_TOKEN_ADDRESS,
    };

    let canonical = [
        (ETHEREUM_USDC_TOKEN_ADDRESS, "canonical USDC init asset"),
        (ARBITRUM_USDC_TOKEN_ADDRESS, "canonical USDC bridge asset"),
        (BASE_USDC_TOKEN_ADDRESS, "canonical USDC bridge asset"),
        (OPTIMISM_USDC_TOKEN_ADDRESS, "canonical USDC bridge asset"),
        (POLYGON_USDC_TOKEN_ADDRESS, "canonical USDC bridge asset"),
        (BNB_USDC_TOKEN_ADDRESS, "canonical USDC bridge asset"),
        (AVALANCHE_USDC_TOKEN_ADDRESS, "canonical USDC bridge asset"),
        (ETHEREUM_WETH_TOKEN_ADDRESS, "canonical wrapped native asset"),
        (ARBITRUM_WETH_TOKEN_ADDRESS, "canonical wrapped native asset"),
        (BASE_WETH_TOKEN_ADDRESS, "canonical wrapped native asset"),
        (OPTIMISM_WETH_TOKEN_ADDRESS, "canonical wrapped native asset"),
        (POLYGON_WETH_TOKEN_ADDRESS, "canonical wrapped ETH asset"),
        (BNB_WETH_TOKEN_ADDRESS, "canonical bridged ETH asset"),
        (AVALANCHE_WETH_TOKEN_ADDRESS, "canonical bridged ETH asset"),
    ];

    canonical
        .iter()
        .find_map(|(known, reason)| (*known == token).then_some(*reason))
        .or_else(|| known_managed_token_reason(token))
}

fn known_managed_token_reason(token: TokenAddress) -> Option<&'static str> {
    let entries = [
        (ChainKey::Ethereum, "0xdac17f958d2ee523a2206206994597c13d831ec7", "canonical USDT"),
        (ChainKey::Arbitrum, "0xfd086bc7cd5c481dcc9c85ebe478a1c0b69fcbb9", "canonical USDT"),
        (ChainKey::Optimism, "0x94b008aa00579c1307b0ef2c499ad98a8ce58e58", "canonical USDT"),
        (ChainKey::Polygon, "0xc2132d05d31c914a87c6611c10748aeb04b58e8f", "canonical USDT"),
        (ChainKey::Bnb, "0x55d398326f99059ff775485246999027b3197955", "canonical USDT"),
        (ChainKey::Avalanche, "0x9702230a8ea53601f5cd2dc00fdbc13d4df4a8c7", "canonical USDT"),
        (ChainKey::Ethereum, "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599", "canonical WBTC"),
        (ChainKey::Arbitrum, "0x2f2a2543b76a4166549f7aab2e75bef0aefc5b0f", "canonical WBTC"),
        (ChainKey::Base, "0xcbb7c0000ab88b473b1f5afd9ef808440eed33bf", "canonical cbBTC"),
        (ChainKey::Optimism, "0x68f180fcce6836688e9084f035309e29bf0a2095", "canonical WBTC"),
        (ChainKey::Polygon, "0x1bfd67037b42cf73acf2047067bd4f2c47d9bfd6", "canonical WBTC"),
        (ChainKey::Bnb, "0x7130d2a12b9bcbfae4f2634d864a1ee1ce3ead9c", "canonical BTCB"),
        (ChainKey::Avalanche, "0x50b7545627a5162f82a992c33b87adc75187b218", "canonical WBTC.e"),
    ];

    let rendered = token.0.to_string().to_ascii_lowercase();
    entries.iter().find_map(|(chain, address, reason)| {
        (*chain == token.1 && rendered == *address).then_some(*reason)
    })
}

fn has_minimum_popularity(candidate: &TokenCandidate) -> bool {
    candidate
        .tvl_usd
        .is_some_and(|value| value >= MIN_VERIFIED_TOKEN_TVL_USD)
        || known_popular_plain_token_reason(candidate.token).is_some()
}

fn known_popular_plain_token_reason(token: TokenAddress) -> Option<&'static str> {
    let entries = [
        (
            ChainKey::Ethereum,
            "0x6b175474e89094c44da98b954eedeac495271d0f",
            "canonical DAI",
        ),
        (
            ChainKey::Optimism,
            "0xda10009cbd5d07dd0cecc66161fc93d7c9000da1",
            "canonical DAI",
        ),
        (
            ChainKey::Ethereum,
            "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984",
            "Uniswap governance token",
        ),
        (
            ChainKey::Ethereum,
            "0x5a98fcbea516cf06857215779fd812ca3bef1b32",
            "Lido governance token",
        ),
        (
            ChainKey::Ethereum,
            "0xf939e0a03fb07f59a73314e73794be0e57ac1b4e",
            "Curve crvUSD",
        ),
    ];

    let rendered = token.0.to_string().to_ascii_lowercase();
    entries.iter().find_map(|(chain, address, reason)| {
        (*chain == token.1 && rendered == *address).then_some(*reason)
    })
}

fn strip_0x(value: &str) -> &str {
    value.strip_prefix("0x").unwrap_or(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use client_evm::{ChainKey, ETHEREUM_USDC_TOKEN_ADDRESS};

    pub(crate) fn candidate(token: TokenAddress) -> TokenCandidate {
        TokenCandidate {
            token,
            symbol: None,
            decimals: None,
            tvl_usd: None,
            volume_usd: None,
            trusted_listing: false,
        }
    }

    /// A custom examiner slots in behind the trait: the seam the future contract-examination
    /// logic will use, exercised here with a symbol-based rejection.
    #[test]
    fn a_rejecting_examiner_excludes_candidates() {
        struct RejectUnnamed;
        impl TokenExaminer for RejectUnnamed {
            fn examine(&self, candidate: &TokenCandidate) -> ExaminationVerdict {
                if candidate.symbol.is_some() {
                    ExaminationVerdict::Approved {
                        reason: "named".to_owned(),
                        decimals: None,
                    }
                } else {
                    ExaminationVerdict::Rejected {
                        reason: "no symbol".to_owned(),
                    }
                }
            }

            fn label(&self) -> String {
                "reject-unnamed/test".to_owned()
            }
        }

        let unnamed = candidate(ETHEREUM_USDC_TOKEN_ADDRESS);
        let named = TokenCandidate {
            symbol: Some("USDC".to_owned()),
            ..candidate(ETHEREUM_USDC_TOKEN_ADDRESS)
        };

        assert_eq!(
            RejectUnnamed.examine(&unnamed),
            ExaminationVerdict::Rejected {
                reason: "no symbol".to_owned()
            }
        );
        assert_eq!(
            RejectUnnamed.examine(&named),
            ExaminationVerdict::Approved {
                reason: "named".to_owned(),
                decimals: None
            }
        );
    }

    #[test]
    fn approve_all_approves_everything() {
        let examiner = ApproveAll;

        assert_eq!(
            examiner.examine(&candidate(ETHEREUM_USDC_TOKEN_ADDRESS)),
            ExaminationVerdict::Approved {
                reason: "approve-all".to_owned(),
                decimals: None
            }
        );
        assert_eq!(
            examiner.examine(&candidate(TokenAddress(
                client_evm::Address::ZERO,
                ChainKey::Avalanche
            ))),
            ExaminationVerdict::Approved {
                reason: "approve-all".to_owned(),
                decimals: None
            }
        );
    }

    #[test]
    fn source_scan_rejects_transfer_tax_and_trapdoor_words() {
        let source = "contract T { uint256 public buyTax; mapping(address => bool) blacklist; }";
        let flags = source_risk_flags(source);

        assert!(flags.iter().any(|flag| flag == "transfer-tax"));
        assert!(flags.iter().any(|flag| flag == "blacklist"));
    }

    #[test]
    fn source_scan_allows_plain_balance_conserving_erc20_words() {
        let source = "contract T { function transfer(address to, uint256 amount) public returns (bool); }";

        assert!(source_risk_flags(source).is_empty());
    }

    #[test]
    fn storage_word_decodes_last_20_bytes_as_address() {
        let word = "0x000000000000000000000000a0b86991c6218b36c1d19d4a2e9eb0ce3606eb48";

        assert_eq!(
            address_from_storage_word(word).map(|address| address.to_string().to_ascii_lowercase()),
            Some("0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48".to_owned())
        );
    }

    #[test]
    fn decimals_result_decodes_low_integer() {
        let word = "0x0000000000000000000000000000000000000000000000000000000000000006";

        assert_eq!(decode_decimals_result(word), Some(6));
    }

    #[test]
    fn minimum_popularity_requires_tvl_or_known_plain_exception() {
        let mut low_tvl = candidate(ETHEREUM_USDC_TOKEN_ADDRESS);
        low_tvl.tvl_usd = Some(9_999.99);
        assert!(!has_minimum_popularity(&low_tvl));

        let mut enough_tvl = candidate(ETHEREUM_USDC_TOKEN_ADDRESS);
        enough_tvl.tvl_usd = Some(10_000.0);
        assert!(has_minimum_popularity(&enough_tvl));

        let dai = candidate(TokenAddress(
            "0x6b175474e89094c44da98b954eedeac495271d0f"
                .parse()
                .expect("DAI address parses"),
            ChainKey::Ethereum,
        ));
        assert!(has_minimum_popularity(&dai));
    }
}
