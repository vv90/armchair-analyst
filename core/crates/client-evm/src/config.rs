#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvmNetwork {
    Ethereum,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RpcConfig {
    pub network: EvmNetwork,
    pub http_url: String,
    pub ws_url: String,
    pub api_key: String,
}
