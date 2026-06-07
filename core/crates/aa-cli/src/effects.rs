use std::{env, io, io::Write, process::ExitCode, thread};

use client_evm::RpcConfig;

use crate::{
    app::start_runtime,
    utils::{CliError, load_rpc_config_with},
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
    let handle = start_runtime(config);

    finish_runtime(handle.join())
}

fn load_rpc_config() -> Result<RpcConfig, CliError> {
    load_rpc_config_with(|name| env::var(name).ok(), prompt_for_value)
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
