use std::{
    env,
    io::{self, Write},
    process::ExitCode,
};

use aa_framework::Runtime;
use client_evm::ChainEndpoints;

use crate::{core::CHAIN, runtime::ServerRuntime};

mod core;
mod runtime;
// Pure response-producing surface; the runtime's serve loop binds a blocking HTTP server and calls
// `serve::http_response`.
mod serve;

fn main() -> ExitCode {
    warn_if_rustls_provider_already_installed(install_rustls_provider(), &mut io::stderr());

    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}

/// Boots the single-chain server: one HTTP RPC endpoint and one WS URL from the environment, an
/// empty-init kernel (no bootstrap stage), and the framework event loop. Blocks for the life of the
/// process (the runtime loop and its subscriptions never return).
fn run() -> Result<(), String> {
    let rpc_url = env::var("AA_ETH_RPC_URL")
        .map_err(|_| "AA_ETH_RPC_URL is not set (Ethereum HTTP RPC endpoint)".to_owned())?;
    let ws_url = env::var("AA_ETH_WS_URL")
        .map_err(|_| "AA_ETH_WS_URL is not set (Ethereum WebSocket endpoint)".to_owned())?;
    // Loopback by default so the data plane is never exposed by accident; set an explicit address
    // (e.g. `0.0.0.0:8080`) to serve clients over the network.
    let bind_addr = env::var("AA_SERVER_BIND").unwrap_or_else(|_| "127.0.0.1:8080".to_owned());

    let endpoints = ChainEndpoints::single(CHAIN, "primary", rpc_url);
    let runtime = ServerRuntime::new(endpoints, ws_url, bind_addr);

    let (_input_sender, handle) = runtime.run();
    handle
        .join()
        .map_err(|_| "runtime thread panicked".to_owned())
}

fn install_rustls_provider() -> Result<(), ()> {
    rustls::crypto::ring::default_provider()
        .install_default()
        .map_err(|_| ())
}

fn warn_if_rustls_provider_already_installed<Output: Write>(
    install_result: Result<(), ()>,
    output: &mut Output,
) {
    if install_result.is_err() {
        let _ = writeln!(
            output,
            "warning: rustls crypto provider was already installed"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{install_rustls_provider, warn_if_rustls_provider_already_installed};

    #[test]
    fn rustls_provider_installation_selects_process_default() {
        let _ = install_rustls_provider();

        assert!(rustls::crypto::CryptoProvider::get_default().is_some());
    }

    #[test]
    fn rustls_provider_installation_warns_when_default_already_exists() {
        let mut output = Vec::new();

        warn_if_rustls_provider_already_installed(Err(()), &mut output);

        assert_eq!(
            String::from_utf8(output),
            Ok("warning: rustls crypto provider was already installed\n".to_owned())
        );
    }
}
