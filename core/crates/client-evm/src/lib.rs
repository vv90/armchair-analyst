//! Integration logic for EVM-based chains.

pub use alloy::primitives::BlockHash;

pub mod chain;
pub mod client;
mod config;
mod erc20;
mod error;
pub mod kernel;
pub mod multi_chain_kernel;
mod pending_requests;
mod pool_registry;
mod pool_state;
mod tick;
mod tick_math;
mod token_registry;
pub mod uniswap_v3;

pub use chain::{ChainKey, drpc_network_path};
pub use client::{
    ClientEvent, ClientHead, fetch_block_header, fetch_block_logs, fetch_finalized_block_header,
    fetch_pool_data, fetch_pool_metadata, fetch_token_metadata, subscribe_new_heads,
};
pub use config::RpcConfig;
pub use error::ClientEvmError;
pub use pending_requests::{
    AnyIssuedRequest, AnyRequestId, GetBlockHeader, GetBlockLogs, GetPoolData, GetPoolMetadata,
    GetTokenMetadata, IssuedRequest, RequestId,
};
pub use pool_registry::{
    PoolCandidateAddress, PoolMetadata, PoolMetadataCall, PoolMetadataFailure, PoolMetadataResult,
    TrustedPoolLogs, TrustedPoolRegistry, UniswapV3Fee,
};
pub use pool_state::{
    PoolAddress, PoolDataCall, PoolDataFailure, PoolDataResult, PoolState, PoolStateError,
};
pub use tick_math::TickMathError;
pub use token_registry::{
    TokenAddress, TokenDecimals, TokenMetadata, TokenMetadataCall, TokenMetadataFailure,
    TokenMetadataResult, TokenRegistry,
};
