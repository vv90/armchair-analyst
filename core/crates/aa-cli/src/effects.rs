use std::{env, io, io::Write, process::ExitCode, thread};

use client_evm::{MetadataCache, RpcConfig, assemble_chain_endpoints};

use crate::{
    app::start_runtime,
    logger::Logger,
    utils::{
        CliError, load_custom_endpoints_with, load_rpc_config_with, metadata_cache_path_with,
        public_fallbacks_enabled_with,
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
    let config = load_rpc_config()?;
    let custom_endpoints = load_custom_endpoints_with(
        |name| env::var(name).ok(),
        |path| std::fs::read_to_string(path),
        |prompt| prompt_for_key(prompt),
    )?;
    let include_public_fallbacks = public_fallbacks_enabled_with(|name| env::var(name).ok());
    let endpoints = assemble_chain_endpoints(&config, &custom_endpoints, include_public_fallbacks)
        .map_err(|error| CliError::EndpointConfigFailed {
            message: error.to_string(),
        })?;
    let metadata_cache = open_metadata_cache()?;
    let logger = Logger::create_for_run().map_err(|error| CliError::LogInitFailed {
        message: error.to_string(),
    })?;
    let view = View::for_run();
    let handle = start_runtime(config, endpoints, metadata_cache, logger, view.clone());

    let result = finish_runtime(handle.join());
    view.finish();

    result
}

fn load_rpc_config() -> Result<RpcConfig, CliError> {
    load_rpc_config_with(|name| env::var(name).ok(), prompt_for_value)
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

fn prompt_for_value(prompt: &'static str) -> Result<String, CliError> {
    print!("{prompt} ");
    io::stdout()
        .flush()
        .map_err(|error| prompt_error(prompt, error))?;

    let mut value = String::new();
    io::stdin()
        .read_line(&mut value)
        .map_err(|error| prompt_error(prompt, error))?;

    Ok(value)
}

fn prompt_error(prompt: &'static str, error: io::Error) -> CliError {
    CliError::PromptFailed {
        prompt,
        message: error.to_string(),
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
