//! The token-examination seam. Today's only examiner approves everything; the eventual production
//! examiners (contract-source inspection, honeypot simulation, transfer-tax detection, …) plug in
//! behind [`TokenExaminer`] without touching discovery or the artifact writer, and are expected to
//! evolve independently of the main binary.

use client_evm::TokenAddress;

/// A token proposed for whitelisting, with the subgraph-sourced context an examiner (or a human
/// reviewing the artifact) can weigh. Only `token` is identity; the rest is advisory.
#[derive(Clone, Debug, PartialEq)]
pub struct TokenCandidate {
    pub token: TokenAddress,
    pub symbol: Option<String>,
    pub decimals: Option<u8>,
    pub tvl_usd: Option<f64>,
    pub volume_usd: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExaminationVerdict {
    Approved,
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
        ExaminationVerdict::Approved
    }

    fn label(&self) -> String {
        "approve-all/0.1.0".to_owned()
    }
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
                    ExaminationVerdict::Approved
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
        assert_eq!(RejectUnnamed.examine(&named), ExaminationVerdict::Approved);
    }

    #[test]
    fn approve_all_approves_everything() {
        let examiner = ApproveAll;

        assert_eq!(
            examiner.examine(&candidate(ETHEREUM_USDC_TOKEN_ADDRESS)),
            ExaminationVerdict::Approved
        );
        assert_eq!(
            examiner.examine(&candidate(TokenAddress(
                client_evm::Address::ZERO,
                ChainKey::Avalanche
            ))),
            ExaminationVerdict::Approved
        );
    }
}
