//! Integration logic for EVM-based chains.

pub use alloy::primitives::{BlockHash, Bloom};

pub mod bootstrap;
pub mod chain;
pub mod client;
pub mod endpoints;
mod erc20;
mod error;
pub mod kernel;
pub mod multi_chain_kernel;
mod pool_log;
mod pool_state;
mod request_tracking;
mod tick;
mod tick_math;
mod tokens;
pub mod uniswap_v3;
pub mod uniswap_v4;
pub mod uniswap_v4_subgraph;
mod utils;

pub use chain::{ACTIVE_CHAINS, ChainKey, chain_key_for_network_path, drpc_network_path};
pub use client::{
    ClientEvent, ClientHead, MetadataCache, MetadataCacheError, POOL_LOG_BATCH_WINDOW,
    RangeLogBlock, consolidate_pool_logs, fetch_block_header, fetch_block_logs,
    fetch_finalized_block_header, fetch_pool_candidates_in_range, fetch_pool_data,
    fetch_pool_metadata, fetch_token_metadata, plan_ws_subscriptions, subscribe_new_heads,
    subscribe_pool_events, WsSubscriptionEndpoint,
};
pub use endpoints::{
    ChainEndpoints, ChainSubscriptions, EndpointPool, EndpointSpec, GraphEndpoints,
    assemble_chain_endpoints, assemble_graph_endpoints,
};
pub use error::{ClientEvmError, ConfigScope};
pub use kernel::pending_requests::{
    AnyIssuedRequest, AnyRequestId, GetBlockHeader, GetBlockLogs, GetPoolData, GetPoolMetadata,
    GetTokenMetadata, IssuedRequest, RequestId,
};
pub use kernel::pool_registry::{
    PoolFee, PoolMetadata, PoolMetadataCall, PoolMetadataFailure, PoolMetadataResult,
    TrustedPoolLogs, TrustedPoolRegistry, UniswapV3Fee,
};
pub use kernel::token_registry::{
    TokenAddress, TokenDecimals, TokenMetadata, TokenMetadataCall, TokenMetadataFailure,
    TokenMetadataResult, TokenRegistry,
};
pub use pool_log::{PoolLog, PoolLogEvent, decode_pool_log, derive_pool_state};
pub use pool_state::{
    PoolDataCall, PoolDataFailure, PoolDataResult, PoolRef, PoolState, PoolStateError,
    ProtocolPoolKey,
};
pub use tick_math::TickMathError;
pub use uniswap_v4_subgraph::fetch_v4_pool_metadata;
pub use tokens::{
    ARBITRUM_NATIVE_TOKEN_ADDRESS, ARBITRUM_USDC_TOKEN_ADDRESS, ARBITRUM_WETH_TOKEN_ADDRESS,
    AVALANCHE_NATIVE_TOKEN_ADDRESS, AVALANCHE_USDC_TOKEN_ADDRESS, AVALANCHE_WETH_TOKEN_ADDRESS,
    BASE_NATIVE_TOKEN_ADDRESS, BASE_USDC_TOKEN_ADDRESS, BASE_WETH_TOKEN_ADDRESS,
    BNB_NATIVE_TOKEN_ADDRESS, BNB_USDC_TOKEN_ADDRESS, BNB_WETH_TOKEN_ADDRESS,
    ETHEREUM_NATIVE_TOKEN_ADDRESS, ETHEREUM_USDC_TOKEN_ADDRESS, ETHEREUM_WETH_TOKEN_ADDRESS,
    OPTIMISM_NATIVE_TOKEN_ADDRESS, OPTIMISM_USDC_TOKEN_ADDRESS, OPTIMISM_WETH_TOKEN_ADDRESS,
    POLYGON_NATIVE_TOKEN_ADDRESS, POLYGON_USDC_TOKEN_ADDRESS, POLYGON_WETH_TOKEN_ADDRESS,
};
pub use utils::{TokenAmountConversionError, u256_token_amount_to_f32};
