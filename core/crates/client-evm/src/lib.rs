//! Integration logic for EVM-based chains.

pub mod client;
mod config;
mod error;
pub mod kernel;
pub mod uniswap_v3;

pub use client::{ClientEvent, subscribe_pool_events};
pub use config::{EvmNetwork, RpcConfig};
pub use error::ClientEvmError;
