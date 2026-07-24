//! Integration logic for EVM-based chains.

pub use alloy::primitives::{Address, BlockHash, Bloom, U160, U256, aliases::I24};

pub mod bootstrap;
pub mod chain;
pub mod client;
pub mod endpoints;
mod erc20;
mod error;
pub mod kernel;
mod lossless_replay;
pub mod multi_chain_kernel;
mod pool_log;
mod pool_state;
mod request_tracking;
mod tick;
mod tick_math;
pub mod token_whitelist;
mod tokens;
pub mod uniswap_v3;
pub mod uniswap_v4;
pub mod uniswap_v4_subgraph;
mod utils;

pub use chain::{ACTIVE_CHAINS, ChainKey, chain_key_for_network_path, drpc_network_path};
pub use client::{
    ClientEvent, ClientHead, MetadataCache, MetadataCacheError, POOL_LOG_BATCH_WINDOW,
    RangeLogBlock, WsSubscriptionEndpoint, consolidate_pool_logs, fetch_block_header,
    fetch_block_logs, fetch_canonical_block_header_at, fetch_finalized_block_header,
    fetch_pool_candidates_window, fetch_pool_data, fetch_pool_logs_in_range, fetch_pool_metadata,
    fetch_token_metadata, plan_ws_subscriptions, subscribe_new_heads, subscribe_pool_events,
};
pub use endpoints::{
    ChainEndpoints, ChainSubscriptions, EndpointPool, EndpointSpec, GraphEndpoints,
    assemble_chain_endpoints, assemble_graph_endpoints,
};
pub use error::{ClientEvmError, ConfigScope};
pub use kernel::metadata_catalog::MetadataCatalog;
pub use kernel::pending_requests::{
    AnyIssuedRequest, AnyRequestId, GetBlockHeader, GetBlockLogs, GetCanonicalHeaderAtHeight,
    GetLogsRange, GetPoolData, GetPoolMetadata, GetTokenMetadata, IssuedRequest, RequestId,
};
pub use kernel::pool_registry::{
    PoolFee, PoolMetadata, PoolMetadataCall, PoolMetadataFailure, PoolMetadataResult,
    TrustedPoolRegistry, UniswapV3Fee,
};
pub use kernel::token_registry::{
    TokenAddress, TokenDecimals, TokenMetadata, TokenMetadataCall, TokenMetadataFailure,
    TokenMetadataResult, TokenRegistry,
};
pub use lossless_replay::{
    LosslessOutcome, LosslessPool, LosslessReplayError, replay_plan_lossless,
};
pub use pool_log::{PoolLog, PoolLogEvent, decode_pool_log, derive_pool_state};
pub use pool_state::{
    PoolDataCall, PoolDataFailure, PoolDataResult, PoolRef, PoolState, PoolStateError,
    ProtocolPoolKey,
};
pub use tick_math::TickMathError;
pub use token_whitelist::{
    ChainTokens, TokenEntry, TokenWhitelist, TokenWhitelistError, TokenWhitelistFile,
};
pub use tokens::{
    ARBITRUM_NATIVE_TOKEN_ADDRESS, ARBITRUM_USDC_TOKEN_ADDRESS, ARBITRUM_USDT_TOKEN_ADDRESS,
    ARBITRUM_WBTC_TOKEN_ADDRESS, ARBITRUM_WETH_TOKEN_ADDRESS, AVALANCHE_NATIVE_TOKEN_ADDRESS,
    AVALANCHE_USDC_TOKEN_ADDRESS, AVALANCHE_USDT_TOKEN_ADDRESS, AVALANCHE_WBTC_TOKEN_ADDRESS,
    AVALANCHE_WETH_TOKEN_ADDRESS, BASE_CBBTC_TOKEN_ADDRESS, BASE_NATIVE_TOKEN_ADDRESS,
    BASE_USDC_TOKEN_ADDRESS, BASE_WETH_TOKEN_ADDRESS, BNB_BTCB_TOKEN_ADDRESS,
    BNB_NATIVE_TOKEN_ADDRESS, BNB_USDC_TOKEN_ADDRESS, BNB_USDT_TOKEN_ADDRESS,
    BNB_WETH_TOKEN_ADDRESS, ETHEREUM_DAI_TOKEN_ADDRESS, ETHEREUM_NATIVE_TOKEN_ADDRESS,
    ETHEREUM_USDC_TOKEN_ADDRESS, ETHEREUM_USDT_TOKEN_ADDRESS, ETHEREUM_WBTC_TOKEN_ADDRESS,
    ETHEREUM_WETH_TOKEN_ADDRESS, OPTIMISM_DAI_TOKEN_ADDRESS, OPTIMISM_NATIVE_TOKEN_ADDRESS,
    OPTIMISM_USDC_TOKEN_ADDRESS, OPTIMISM_USDT_TOKEN_ADDRESS, OPTIMISM_WBTC_TOKEN_ADDRESS,
    OPTIMISM_WETH_TOKEN_ADDRESS, POLYGON_NATIVE_TOKEN_ADDRESS, POLYGON_USDC_TOKEN_ADDRESS,
    POLYGON_USDT_TOKEN_ADDRESS, POLYGON_WBTC_TOKEN_ADDRESS, POLYGON_WETH_TOKEN_ADDRESS,
};
pub use uniswap_v4_subgraph::{fetch_v4_pool_metadata, send_graphql_request};
pub use utils::{TokenAmountConversionError, u256_token_amount_to_f32};
