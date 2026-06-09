use thiserror::Error;

#[derive(Debug, Error)]
pub enum OptimizationError {
    #[error("init_asset not found")]
    InitAssetNotFound,

    #[error("Invalid init asset index")]
    InvalidInitAssetIndex,

    #[error("Invalid layout index")]
    InvalidLayoutIndex,

    #[error("Invalid output index")]
    InvalidOutputIndex,
}
