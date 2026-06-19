//! Integration logic for EVM-based chains.

pub use alloy::primitives::BlockHash;

pub mod bootstrap;
pub mod chain;
pub mod client;
mod config;
mod erc20;
mod error;
pub mod kernel;
pub mod multi_chain_kernel;
mod pool_state;
mod request_tracking;
mod tick;
mod tick_math;
mod tokens;
pub mod uniswap_v3;
mod utils;

pub use chain::{ACTIVE_CHAINS, ChainKey, drpc_network_path};
pub use client::{
    ClientEvent, ClientHead, RangeLogBlock, fetch_block_header, fetch_block_logs,
    fetch_finalized_block_header, fetch_pool_candidates_in_range, fetch_pool_data,
    fetch_pool_metadata, fetch_token_metadata, subscribe_new_heads,
};
pub use config::RpcConfig;
pub use error::ClientEvmError;
pub use kernel::pending_requests::{
    AnyIssuedRequest, AnyRequestId, GetBlockHeader, GetBlockLogs, GetPoolData, GetPoolMetadata,
    GetTokenMetadata, IssuedRequest, RequestId,
};
pub use kernel::pool_registry::{
    PoolCandidateAddress, PoolMetadata, PoolMetadataCall, PoolMetadataFailure, PoolMetadataResult,
    TrustedPoolLogs, TrustedPoolRegistry, UniswapV3Fee,
};
pub use kernel::token_registry::{
    TokenAddress, TokenDecimals, TokenMetadata, TokenMetadataCall, TokenMetadataFailure,
    TokenMetadataResult, TokenRegistry,
};
pub use pool_state::{
    PoolAddress, PoolDataCall, PoolDataFailure, PoolDataResult, PoolState, PoolStateError,
};
pub use tick_math::TickMathError;
pub use tokens::ETHEREUM_USDC_TOKEN_ADDRESS;
pub use utils::{TokenAmountConversionError, u256_token_amount_to_f32};
