use std::{env, io, io::Write, process::ExitCode, thread};

use client_evm::{MetadataCache, RpcConfig};

use crate::{
    app::start_runtime,
    logger::Logger,
    utils::{CliError, load_rpc_config_with, metadata_cache_path_with},
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
    let metadata_cache = open_metadata_cache()?;
    let logger = Logger::create_for_run().map_err(|error| CliError::LogInitFailed {
        message: error.to_string(),
    })?;
    let view = View::for_run();
    let handle = start_runtime(config, metadata_cache, logger, view.clone());

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
