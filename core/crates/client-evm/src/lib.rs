//! Integration logic for EVM-based chains.

pub mod client;
mod config;
mod error;
pub mod kernel;
mod pending_requests;
mod tick;
pub mod uniswap_v3;

pub use client::{
    ClientEvent, ClientHead, fetch_block_header, fetch_finalized_block_header, subscribe_new_heads,
    subscribe_pool_events,
};
pub use config::{EvmNetwork, RpcConfig};
pub use error::ClientEvmError;
