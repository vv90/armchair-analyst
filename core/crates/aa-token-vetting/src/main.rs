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

use std::collections::HashSet;
use std::process::ExitCode;
use std::{env, fs};

use aa_token_vetting::examiner::{ApproveAll, ExaminationVerdict, TokenCandidate, TokenExaminer};
use aa_token_vetting::{VettingError, config, discovery, writer};
use client_evm::{ACTIVE_CHAINS, ChainKey, TokenAddress, drpc_network_path};

const DEFAULT_OUT: &str = "token-whitelist.toml";
const DEFAULT_TOP: usize = 50;

#[derive(Debug, PartialEq, Eq)]
struct Args {
    out: String,
    top: usize,
    allow_missing_chains: bool,
}

fn parse_args<I: Iterator<Item = String>>(mut args: I) -> Result<Args, VettingError> {
    let mut parsed = Args {
        out: DEFAULT_OUT.to_owned(),
        top: DEFAULT_TOP,
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
        eprintln!("warn skipped subgraph provider '{provider}' (no API key in the environment)");
    }

    let agent = ureq::Agent::new_with_defaults();
    let examiner = ApproveAll;

    let mut approved: Vec<TokenCandidate> = Vec::new();
    let mut missing_chains: Vec<ChainKey> = Vec::new();

    for &chain in ACTIVE_CHAINS {
        let discovered =
            match discovery::fetch_top_tokens(&agent, &loaded.endpoints, chain, args.top) {
                Ok(Some(tokens)) => tokens,
                Ok(None) => {
                    missing_chains.push(chain);
                    continue;
                }
                Err(error) => {
                    return Err(VettingError::DiscoveryFailed {
                        chain,
                        message: error.to_string(),
                    });
                }
            };

        let mut candidates = discovered;
        candidates.extend(discovery::baseline_candidates(chain));

        let mut seen: HashSet<TokenAddress> = HashSet::new();
        let mut approved_on_chain = 0usize;
        for candidate in candidates {
            if !seen.insert(candidate.token) {
                continue;
            }
            match examiner.examine(&candidate) {
                ExaminationVerdict::Approved => {
                    approved.push(candidate);
                    approved_on_chain += 1;
                }
                ExaminationVerdict::Rejected { reason } => {
                    eprintln!(
                        "rejected chain={} token={} reason={reason}",
                        drpc_network_path(chain),
                        candidate.token.0
                    );
                }
            }
        }

        println!(
            "chain={} approved={approved_on_chain}",
            drpc_network_path(chain)
        );
    }

    if !missing_chains.is_empty() {
        if args.allow_missing_chains {
            for &chain in &missing_chains {
                eprintln!(
                    "warn no subgraph endpoint for chain={} — its section is OMITTED and the \
                     runtime will deny-all that chain under this artifact",
                    drpc_network_path(chain)
                );
            }
        } else {
            return Err(VettingError::MissingChains {
                chains: missing_chains,
            });
        }
    }

    let file = writer::build_whitelist_file(&approved, examiner.label(), writer::rfc3339_utc_now());
    let rendered = writer::render_toml(&file)?;
    fs::write(&args.out, rendered).map_err(|error| VettingError::WriteFailed {
        message: format!("{}: {error}", args.out),
    })?;

    println!(
        "wrote {} ({} tokens across {} chains)",
        args.out,
        approved.len(),
        file.chains.len()
    );

    Ok(())
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
                top: DEFAULT_TOP,
                allow_missing_chains: false,
            }
        );
    }

    #[test]
    fn all_arguments_parse() {
        let parsed =
            args(&["--out", "wl.toml", "--top", "5", "--allow-missing-chains"]).expect("parses");

        assert_eq!(
            parsed,
            Args {
                out: "wl.toml".to_owned(),
                top: 5,
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
    }
}
