use std::{env, io, io::Write, process::ExitCode, thread};

use client_evm::{
    ChainSubscriptions, ClientEvmError, MetadataCache, assemble_chain_endpoints,
    assemble_graph_endpoints,
};

use crate::{
    app::{optimization_session_config, start_runtime},
    logger::Logger,
    utils::{
        CliError, load_config_with, load_token_whitelist_with, metadata_cache_path_with,
        summarize_endpoints, summarize_token_whitelist,
    },
    view::View,
};

pub(crate) fn main_exit_code() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), CliError> {
    let resolved = load_config_with(
        |name| env::var(name).ok(),
        |path| std::fs::read_to_string(path),
        |prompt| prompt_for_key(prompt),
    )?;

    // Captured before `resolved.rpc_ws` is moved into the subscription channel below, then logged once
    // the run logger exists so the live provider composition lands alongside the bootstrap events.
    let startup_summary = summarize_endpoints(&resolved);

    let whitelist =
        load_token_whitelist_with(|name| env::var(name).ok(), |path| std::fs::read_to_string(path))?;
    let (session_config, dropped_bridges) = optimization_session_config(whitelist.as_ref())?;
    let whitelist_summary = summarize_token_whitelist(whitelist.as_ref(), &dropped_bridges);

    let endpoints = assemble_chain_endpoints(&resolved.rpc_http).map_err(endpoint_config_error)?;
    let subscriptions = ChainSubscriptions::new(resolved.rpc_ws).map_err(endpoint_config_error)?;
    // Empty when no subgraph is configured (or its key was skipped), in which case v4 metadata
    // resolution is simply skipped; the RPC path is unaffected.
    let graph_endpoints =
        assemble_graph_endpoints(&resolved.subgraph).map_err(endpoint_config_error)?;
    let metadata_cache = open_metadata_cache()?;
    let logger = Logger::create_for_run().map_err(|error| CliError::LogInitFailed {
        message: error.to_string(),
    })?;
    for line in startup_summary.iter().chain(&whitelist_summary) {
        logger.log(line);
    }
    let view = View::for_run();
    let handle = start_runtime(
        subscriptions,
        endpoints,
        graph_endpoints,
        metadata_cache,
        session_config,
        whitelist,
        logger,
        view.clone(),
    );

    let result = finish_runtime(handle.join());
    view.finish();

    result
}

fn endpoint_config_error(error: ClientEvmError) -> CliError {
    CliError::EndpointConfigFailed {
        message: error.to_string(),
    }
}

fn open_metadata_cache() -> Result<MetadataCache, CliError> {
    let path = metadata_cache_path_with(|name| env::var(name).ok());
    MetadataCache::open(&path).map_err(|error| CliError::CacheInitFailed {
        message: error.to_string(),
    })
}

fn finish_runtime(result: thread::Result<()>) -> Result<(), CliError> {
    match result {
        Ok(()) => Ok(()),
        Err(_) => Err(CliError::RuntimeFailed {
            message: "runtime thread panicked".to_owned(),
        }),
    }
}

/// Console prompt for a provider API key. Mirrors [`prompt_for_value`] but takes dynamic prompt text
/// (the provider name) and maps I/O failures to [`CliError::EndpointConfigFailed`]. A blank line (or
/// EOF on a non-interactive run) is returned verbatim so the caller can treat it as "skip".
fn prompt_for_key(prompt: &str) -> Result<String, CliError> {
    print!("{prompt} ");
    io::stdout()
        .flush()
        .map_err(|error| key_prompt_error(prompt, &error))?;

    let mut value = String::new();
    io::stdin()
        .read_line(&mut value)
        .map_err(|error| key_prompt_error(prompt, &error))?;

    Ok(value)
}

fn key_prompt_error(prompt: &str, error: &io::Error) -> CliError {
    CliError::EndpointConfigFailed {
        message: format!("failed to read {prompt} {error}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_thread_success_returns_ok() {
        assert_eq!(finish_runtime(Ok(())), Ok(()));
    }

    #[test]
    fn runtime_thread_panic_maps_to_runtime_failed_error() {
        let result = finish_runtime(Err(Box::new("panic message")));

        assert_eq!(
            result,
            Err(CliError::RuntimeFailed {
                message: "runtime thread panicked".to_owned(),
            })
        );
    }
}
