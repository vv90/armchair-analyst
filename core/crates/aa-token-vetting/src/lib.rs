//! Library surface of the offline token-vetting tool: candidate discovery from the Uniswap v4
//! subgraphs, the [`examiner::TokenExaminer`] seam (where real contract-examination logic will
//! plug in and evolve separately from the main binary), and the whitelist-artifact writer. The
//! `aa-token-vetting` binary is a thin orchestration over these modules.

pub mod config;
pub mod discovery;
pub mod examiner;
pub mod writer;

use std::fmt;

use client_evm::{ChainKey, drpc_network_path};

#[derive(Debug)]
pub enum VettingError {
    MissingRequiredConfig { env_name: &'static str },
    ConfigFailed { message: String },
    UsageFailed { message: String },
    DiscoveryFailed { chain: ChainKey, message: String },
    MissingChains { chains: Vec<ChainKey> },
    WriteFailed { message: String },
}

impl fmt::Display for VettingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingRequiredConfig { env_name } => {
                write!(formatter, "missing required configuration value {env_name}")
            }
            Self::ConfigFailed { message } => {
                write!(formatter, "failed to load endpoints config: {message}")
            }
            Self::UsageFailed { message } => {
                write!(
                    formatter,
                    "{message}\nusage: aa-token-vetting [--out <path>] [--report <path>] [--top <N>] [--gecko-pages <N>] [--allow-missing-chains]"
                )
            }
            Self::DiscoveryFailed { chain, message } => {
                write!(
                    formatter,
                    "token discovery failed chain={chain:?}: {message}"
                )
            }
            Self::MissingChains { chains } => {
                let slugs = chains
                    .iter()
                    .map(|&chain| drpc_network_path(chain))
                    .collect::<Vec<_>>()
                    .join(",");
                write!(
                    formatter,
                    "no subgraph endpoint for active chains [{slugs}] — the artifact would \
                     deny-all those chains; configure them or pass --allow-missing-chains"
                )
            }
            Self::WriteFailed { message } => {
                write!(formatter, "failed to write whitelist artifact: {message}")
            }
        }
    }
}

impl std::error::Error for VettingError {}
