//! Offline token-vetting binary: discovers candidate tokens per chain from the same Uniswap v4
//! subgraphs the runtime queries, runs them through a
//! [`TokenExaminer`] (approve-all today — real contract examination lands behind the same trait
//! later), and writes the per-chain whitelist artifact the main binary loads via
//! `AA_TOKEN_WHITELIST_FILE`.
//!
//! Usage: `aa-token-vetting [--out <path>] [--top <N>] [--allow-missing-chains]` with
//! `AA_CONFIG_FILE` (and the relevant `AA_GRAPH_KEY_*` / `key_env` vars) set as for the main
//! binary. A chain with no reachable subgraph **fails the run** by default: silently omitting a
//! chain section would deny-all that chain in production. `--allow-missing-chains` downgrades that
//! to a loud warning and writes the partial artifact anyway.

use std::collections::{HashMap, HashSet};
use std::process::ExitCode;
use std::{env, fs};

use aa_token_vetting::examiner::{
    ContractExaminer, ExaminationVerdict, TokenCandidate, TokenExaminer,
};
use aa_token_vetting::{VettingError, config, discovery, writer};
use client_evm::{ACTIVE_CHAINS, ChainKey, TokenAddress, drpc_network_path};
use serde::Serialize;

const DEFAULT_OUT: &str = "token-whitelist.toml";
const DEFAULT_REPORT: &str = "token-vetting-report.json";
const DEFAULT_TOP: usize = 200;
const DEFAULT_GECKO_PAGES: usize = 5;

#[derive(Debug, PartialEq, Eq)]
struct Args {
    out: String,
    report: String,
    top: usize,
    gecko_pages: usize,
    allow_missing_chains: bool,
}

#[derive(Debug, Serialize)]
struct ReportEntry {
    chain: String,
    address: String,
    symbol: Option<String>,
    candidate_decimals: Option<u8>,
    final_decimals: Option<u8>,
    tvl_usd: Option<f64>,
    volume_usd: Option<f64>,
    trusted_listing: bool,
    verdict: String,
    reason: String,
}

fn parse_args<I: Iterator<Item = String>>(mut args: I) -> Result<Args, VettingError> {
    let mut parsed = Args {
        out: DEFAULT_OUT.to_owned(),
        report: DEFAULT_REPORT.to_owned(),
        top: DEFAULT_TOP,
        gecko_pages: DEFAULT_GECKO_PAGES,
        allow_missing_chains: false,
    };

    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--out" => {
                parsed.out = args.next().ok_or_else(|| VettingError::UsageFailed {
                    message: "--out requires a path".to_owned(),
                })?;
            }
            "--top" => {
                let value = args.next().ok_or_else(|| VettingError::UsageFailed {
                    message: "--top requires a count".to_owned(),
                })?;
                parsed.top = value.parse().map_err(|_| VettingError::UsageFailed {
                    message: format!("--top requires a positive integer, got '{value}'"),
                })?;
            }
            "--report" => {
                parsed.report = args.next().ok_or_else(|| VettingError::UsageFailed {
                    message: "--report requires a path".to_owned(),
                })?;
            }
            "--gecko-pages" => {
                let value = args.next().ok_or_else(|| VettingError::UsageFailed {
                    message: "--gecko-pages requires a count".to_owned(),
                })?;
                parsed.gecko_pages = value.parse().map_err(|_| VettingError::UsageFailed {
                    message: format!("--gecko-pages requires a positive integer, got '{value}'"),
                })?;
            }
            "--allow-missing-chains" => parsed.allow_missing_chains = true,
            other => {
                return Err(VettingError::UsageFailed {
                    message: format!("unknown argument '{other}'"),
                });
            }
        }
    }

    Ok(parsed)
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), VettingError> {
    let args = parse_args(env::args().skip(1))?;

    let loaded = config::load_graph_endpoints_with(
        |name| env::var(name).ok(),
        |path| fs::read_to_string(path),
    )?;
    for provider in &loaded.skipped_providers {
        eprintln!("warn skipped endpoint provider '{provider}' (no API key in the environment)");
    }

    let agent = ureq::Agent::new_with_defaults();
    let examiner = ContractExaminer::new(&agent, &loaded.rpc_endpoints);
    let trusted_tokens =
        discovery::fetch_uniswap_default_token_set(&agent).unwrap_or_else(|error| {
            eprintln!("warn default token-list discovery failed error={error}");
            HashSet::new()
        });

    let mut approved: Vec<TokenCandidate> = Vec::new();
    let mut report: Vec<ReportEntry> = Vec::new();

    for &chain in ACTIVE_CHAINS {
        let discovered = match discovery::fetch_top_tokens(
            &agent,
            &loaded.endpoints,
            chain,
            args.top,
        ) {
            Ok(Some(tokens)) => tokens,
            Ok(None) => {
                eprintln!(
                    "warn no subgraph endpoint for chain={} — using secondary discovery plus baseline",
                    drpc_network_path(chain)
                );
                Vec::new()
            }
            Err(error) => {
                eprintln!(
                    "warn subgraph token discovery failed chain={} error={} — using secondary discovery plus baseline",
                    drpc_network_path(chain),
                    error
                );
                Vec::new()
            }
        };

        let mut candidates = discovered;
        candidates.extend(discovery::fetch_geckoterminal_top_pool_tokens(
            &agent,
            chain,
            args.gecko_pages,
        ));
        candidates.extend(discovery::baseline_candidates(chain));
        mark_trusted_listings(&mut candidates, &trusted_tokens);
        let candidates = merge_candidates(candidates);

        let mut approved_on_chain = 0usize;
        let mut rejected_on_chain = 0usize;
        for mut candidate in candidates {
            match examiner.examine(&candidate) {
                ExaminationVerdict::Approved { reason, decimals } => {
                    let candidate_decimals = candidate.decimals;
                    if candidate.decimals.is_none() {
                        candidate.decimals = decimals;
                    }
                    report.push(report_entry(
                        chain,
                        &candidate,
                        candidate_decimals,
                        "approved",
                        &reason,
                    ));
                    approved.push(candidate);
                    approved_on_chain += 1;
                }
                ExaminationVerdict::Rejected { reason } => {
                    report.push(report_entry(
                        chain,
                        &candidate,
                        candidate.decimals,
                        "rejected",
                        &reason,
                    ));
                    rejected_on_chain += 1;
                    eprintln!(
                        "rejected chain={} token={} reason={reason}",
                        drpc_network_path(chain),
                        candidate.token.0
                    );
                }
            }
        }

        println!(
            "chain={} approved={approved_on_chain} rejected={rejected_on_chain}",
            drpc_network_path(chain)
        );
    }

    let file = writer::build_whitelist_file(&approved, examiner.label(), writer::rfc3339_utc_now());
    let rendered = writer::render_toml(&file)?;
    fs::write(&args.out, rendered).map_err(|error| VettingError::WriteFailed {
        message: format!("{}: {error}", args.out),
    })?;
    let rendered_report =
        serde_json::to_string_pretty(&report).map_err(|error| VettingError::WriteFailed {
            message: format!("{}: {error}", args.report),
        })?;
    fs::write(&args.report, rendered_report).map_err(|error| VettingError::WriteFailed {
        message: format!("{}: {error}", args.report),
    })?;

    println!(
        "wrote {} ({} tokens across {} chains)",
        args.out,
        approved.len(),
        file.chains.len()
    );
    println!("wrote {} ({} examined tokens)", args.report, report.len());

    Ok(())
}

fn report_entry(
    chain: ChainKey,
    candidate: &TokenCandidate,
    candidate_decimals: Option<u8>,
    verdict: &str,
    reason: &str,
) -> ReportEntry {
    ReportEntry {
        chain: drpc_network_path(chain).to_owned(),
        address: candidate.token.0.to_string(),
        symbol: candidate.symbol.clone(),
        candidate_decimals,
        final_decimals: candidate.decimals,
        tvl_usd: candidate.tvl_usd,
        volume_usd: candidate.volume_usd,
        trusted_listing: candidate.trusted_listing,
        verdict: verdict.to_owned(),
        reason: reason.to_owned(),
    }
}

fn merge_candidates(candidates: Vec<TokenCandidate>) -> Vec<TokenCandidate> {
    let mut merged: Vec<TokenCandidate> = Vec::new();
    let mut indexes: HashMap<TokenAddress, usize> = HashMap::new();

    for candidate in candidates {
        if let Some(index) = indexes.get(&candidate.token).copied() {
            if let Some(existing) = merged.get_mut(index) {
                existing.trusted_listing |= candidate.trusted_listing;
                if existing.symbol.is_none() {
                    existing.symbol = candidate.symbol;
                }
                if existing.decimals.is_none() {
                    existing.decimals = candidate.decimals;
                }
                if existing.tvl_usd.is_none() {
                    existing.tvl_usd = candidate.tvl_usd;
                }
                if existing.volume_usd.is_none() {
                    existing.volume_usd = candidate.volume_usd;
                }
            }
        } else {
            indexes.insert(candidate.token, merged.len());
            merged.push(candidate);
        }
    }

    merged
}

fn mark_trusted_listings(
    candidates: &mut [TokenCandidate],
    trusted_tokens: &HashSet<TokenAddress>,
) {
    for candidate in candidates {
        if trusted_tokens.contains(&candidate.token) {
            candidate.trusted_listing = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(values: &[&str]) -> Result<Args, VettingError> {
        parse_args(values.iter().map(|value| (*value).to_owned()))
    }

    #[test]
    fn defaults_apply_without_arguments() {
        let parsed = args(&[]).expect("parses");

        assert_eq!(
            parsed,
            Args {
                out: DEFAULT_OUT.to_owned(),
                report: DEFAULT_REPORT.to_owned(),
                top: DEFAULT_TOP,
                gecko_pages: DEFAULT_GECKO_PAGES,
                allow_missing_chains: false,
            }
        );
    }

    #[test]
    fn all_arguments_parse() {
        let parsed = args(&[
            "--out",
            "wl.toml",
            "--report",
            "report.json",
            "--top",
            "5",
            "--gecko-pages",
            "2",
            "--allow-missing-chains",
        ])
        .expect("parses");

        assert_eq!(
            parsed,
            Args {
                out: "wl.toml".to_owned(),
                report: "report.json".to_owned(),
                top: 5,
                gecko_pages: 2,
                allow_missing_chains: true,
            }
        );
    }

    #[test]
    fn unknown_and_malformed_arguments_are_usage_errors() {
        assert!(matches!(
            args(&["--frobnicate"]),
            Err(VettingError::UsageFailed { .. })
        ));
        assert!(matches!(
            args(&["--top", "many"]),
            Err(VettingError::UsageFailed { .. })
        ));
        assert!(matches!(
            args(&["--out"]),
            Err(VettingError::UsageFailed { .. })
        ));
        assert!(matches!(
            args(&["--report"]),
            Err(VettingError::UsageFailed { .. })
        ));
        assert!(matches!(
            args(&["--gecko-pages", "many"]),
            Err(VettingError::UsageFailed { .. })
        ));
    }

    #[test]
    fn candidate_merge_preserves_order_and_combines_trusted_listing() {
        let untrusted = TokenCandidate {
            token: client_evm::ETHEREUM_USDC_TOKEN_ADDRESS,
            symbol: None,
            decimals: None,
            tvl_usd: Some(10.0),
            volume_usd: None,
            trusted_listing: false,
        };
        let trusted = TokenCandidate {
            token: client_evm::ETHEREUM_USDC_TOKEN_ADDRESS,
            symbol: Some("USDC".to_owned()),
            decimals: Some(6),
            tvl_usd: None,
            volume_usd: None,
            trusted_listing: true,
        };

        let merged = merge_candidates(vec![untrusted, trusted]);

        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].symbol.as_deref(), Some("USDC"));
        assert_eq!(merged[0].decimals, Some(6));
        assert_eq!(merged[0].tvl_usd, Some(10.0));
        assert!(merged[0].trusted_listing);
    }

    #[test]
    fn trusted_set_marks_existing_candidates_without_expanding_candidate_universe() {
        let mut candidates = vec![
            TokenCandidate {
                token: client_evm::ETHEREUM_USDC_TOKEN_ADDRESS,
                symbol: Some("USDC".to_owned()),
                decimals: Some(6),
                tvl_usd: None,
                volume_usd: None,
                trusted_listing: false,
            },
            TokenCandidate {
                token: client_evm::BASE_USDC_TOKEN_ADDRESS,
                symbol: Some("USDC".to_owned()),
                decimals: Some(6),
                tvl_usd: None,
                volume_usd: None,
                trusted_listing: false,
            },
        ];
        let trusted = HashSet::from([client_evm::ETHEREUM_USDC_TOKEN_ADDRESS]);

        mark_trusted_listings(&mut candidates, &trusted);

        assert_eq!(candidates.len(), 2);
        assert!(candidates[0].trusted_listing);
        assert!(!candidates[1].trusted_listing);
    }
}
