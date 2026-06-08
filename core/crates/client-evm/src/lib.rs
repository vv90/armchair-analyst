//! Integration logic for EVM-based chains.

pub use alloy::primitives::BlockHash;

pub mod chain;
pub mod client;
mod config;
mod error;
pub mod kernel;
pub mod multi_chain_kernel;
mod pending_requests;
mod pool_state;
mod tick;
pub mod uniswap_v3;

pub use chain::{ChainKey, drpc_network_path};
pub use client::{
    ClientEvent, ClientHead, fetch_block_header, fetch_block_logs, fetch_finalized_block_header,
    fetch_pool_data, subscribe_new_heads,
};
pub use config::RpcConfig;
pub use error::ClientEvmError;
pub use pending_requests::{
    AnyIssuedRequest, AnyRequestId, GetBlockHeader, GetBlockLogs, GetPoolData, IssuedRequest,
    RequestId,
};
pub use pool_state::{PoolAddress, PoolDataCall, PoolDataFailure, PoolDataResult, PoolState};
