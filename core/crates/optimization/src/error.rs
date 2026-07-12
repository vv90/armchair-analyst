use thiserror::Error;

#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum OptimizationError {
    #[error("init_asset not found")]
    InitAssetNotFound,

    #[error("no reserve outputs the init_asset")]
    InitAssetOutputNotFound,

    #[error("Invalid init asset index")]
    InvalidInitAssetIndex,

    #[error("Invalid layout index")]
    InvalidLayoutIndex,

    #[error("Invalid output index")]
    InvalidOutputIndex,
}
