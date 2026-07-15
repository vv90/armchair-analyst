//! Renders approved candidates into the shared whitelist artifact
//! ([`client_evm::TokenWhitelistFile`]) as pretty TOML, with provenance (generation time and
//! examiner label) so an artifact is auditable on its own.

use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use client_evm::{ChainTokens, TokenEntry, TokenWhitelistFile, drpc_network_path};

use crate::VettingError;
use crate::examiner::TokenCandidate;

/// Builds the artifact from the approved candidates, keyed by the same chain slugs the main
/// binary's loader resolves.
pub fn build_whitelist_file(
    approved: &[TokenCandidate],
    examiner: String,
    generated_at: String,
) -> TokenWhitelistFile {
    let mut chains: BTreeMap<String, ChainTokens> = BTreeMap::new();

    for candidate in approved {
        chains
            .entry(drpc_network_path(candidate.token.1).to_owned())
            .or_insert_with(|| ChainTokens { tokens: Vec::new() })
            .tokens
            .push(TokenEntry {
                address: candidate.token.0,
                symbol: candidate.symbol.clone(),
                decimals: candidate.decimals,
                examined_at: Some(generated_at.clone()),
                tvl_usd: candidate.tvl_usd,
            });
    }

    TokenWhitelistFile {
        generated_at: Some(generated_at),
        examiner: Some(examiner),
        chains,
    }
}

pub fn render_toml(file: &TokenWhitelistFile) -> Result<String, VettingError> {
    toml::to_string_pretty(file).map_err(|error| VettingError::WriteFailed {
        message: error.to_string(),
    })
}

/// Current UTC time as RFC 3339 (`YYYY-MM-DDTHH:MM:SSZ`), from the system clock alone — provenance
/// precision doesn't warrant a date-time dependency. Uses the standard civil-from-days algorithm.
pub fn rfc3339_utc_now() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    rfc3339_from_unix(seconds)
}

fn rfc3339_from_unix(seconds: u64) -> String {
    let days = seconds / 86_400;
    let seconds_of_day = seconds % 86_400;
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        seconds_of_day / 3600,
        (seconds_of_day % 3600) / 60,
        seconds_of_day % 60,
    )
}

/// Days-since-epoch → (year, month, day), Howard Hinnant's `civil_from_days` restricted to the
/// post-1970 range (the input is unsigned).
fn civil_from_days(days: u64) -> (u64, u64, u64) {
    let z = days + 719_468;
    let era = z / 146_097;
    let day_of_era = z % 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let mp = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    (if month <= 2 { year + 1 } else { year }, month, day)
}

#[cfg(test)]
mod tests {
    use super::*;
    use client_evm::{
        ARBITRUM_USDC_TOKEN_ADDRESS, ChainKey, ETHEREUM_USDC_TOKEN_ADDRESS, TokenAddress,
    };

    fn candidate(token: TokenAddress, symbol: &str) -> TokenCandidate {
        TokenCandidate {
            token,
            symbol: Some(symbol.to_owned()),
            decimals: Some(6),
            tvl_usd: Some(1000.0),
            volume_usd: None,
        }
    }

    #[test]
    fn rendered_artifact_round_trips_through_the_shared_schema() {
        let approved = [
            candidate(ETHEREUM_USDC_TOKEN_ADDRESS, "USDC"),
            candidate(ARBITRUM_USDC_TOKEN_ADDRESS, "USDC"),
        ];
        let file = build_whitelist_file(
            &approved,
            "approve-all/0.1.0".to_owned(),
            "2026-07-15T12:00:00Z".to_owned(),
        );

        let rendered = render_toml(&file).expect("renders");
        let parsed: TokenWhitelistFile = toml::from_str(&rendered).expect("parses back");
        let whitelist = parsed.into_whitelist().expect("validates");

        assert!(whitelist.allows(ETHEREUM_USDC_TOKEN_ADDRESS));
        assert!(whitelist.allows(ARBITRUM_USDC_TOKEN_ADDRESS));
        assert!(!whitelist.allows(TokenAddress(ETHEREUM_USDC_TOKEN_ADDRESS.0, ChainKey::Base)));
    }

    #[test]
    fn artifact_carries_provenance_and_chain_sections() {
        let file = build_whitelist_file(
            &[candidate(ETHEREUM_USDC_TOKEN_ADDRESS, "USDC")],
            "approve-all/0.1.0".to_owned(),
            "2026-07-15T12:00:00Z".to_owned(),
        );

        assert_eq!(file.examiner.as_deref(), Some("approve-all/0.1.0"));
        assert_eq!(file.generated_at.as_deref(), Some("2026-07-15T12:00:00Z"));
        let ethereum = file.chains.get("ethereum").expect("ethereum section");
        assert_eq!(ethereum.tokens.len(), 1);
        assert_eq!(
            ethereum
                .tokens
                .first()
                .and_then(|t| t.examined_at.as_deref()),
            Some("2026-07-15T12:00:00Z")
        );
    }

    #[test]
    fn rfc3339_formats_known_instants() {
        assert_eq!(rfc3339_from_unix(0), "1970-01-01T00:00:00Z");
        // 2026-07-15 12:34:56 UTC.
        assert_eq!(rfc3339_from_unix(1_784_118_896), "2026-07-15T12:34:56Z");
        // Leap-year day: 2024-02-29 00:00:00 UTC.
        assert_eq!(rfc3339_from_unix(1_709_164_800), "2024-02-29T00:00:00Z");
    }
}
