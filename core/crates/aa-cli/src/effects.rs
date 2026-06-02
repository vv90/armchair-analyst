use std::{
    env, io,
    io::Write,
    process::ExitCode,
    sync::mpsc::{self, Receiver},
    thread::{self, JoinHandle},
};

use client_evm::{ClientEvent, ClientEvmError, RpcConfig, subscribe_pool_events};

use crate::utils::{CliError, load_rpc_config_with};

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
    run_subscription(config)
}

fn load_rpc_config() -> Result<RpcConfig, CliError> {
    load_rpc_config_with(|name| env::var(name).ok(), prompt_for_value)
}

fn run_subscription(config: RpcConfig) -> Result<(), CliError> {
    let (sender, receiver) = mpsc::channel();
    let worker = thread::spawn(move || subscribe_pool_events(config, sender));

    receive_subscription_events(receiver, worker)
}

fn receive_subscription_events(
    receiver: Receiver<ClientEvent>,
    worker: JoinHandle<Result<(), ClientEvmError>>,
) -> Result<(), CliError> {
    let mut event_count = 0;

    for event in receiver {
        let feedback = format_event_feedback(&event, event_count);
        event_count = feedback.event_count;
        println!("{}", feedback.line);

        if feedback.closed {
            return finish_subscription_worker(worker.join());
        }
    }

    finish_subscription_worker(worker.join())
}

fn finish_subscription_worker(
    result: thread::Result<Result<(), ClientEvmError>>,
) -> Result<(), CliError> {
    match result {
        Ok(Ok(())) => Ok(()),
        Ok(Err(error)) => Err(CliError::SubscriptionFailed {
            message: error.to_string(),
        }),
        Err(_) => Err(CliError::SubscriptionFailed {
            message: "subscription worker panicked".to_owned(),
        }),
    }
}

#[derive(Debug, Eq, PartialEq)]
struct EventFeedback {
    event_count: u64,
    line: String,
    closed: bool,
}

fn format_event_feedback(event: &ClientEvent, event_count: u64) -> EventFeedback {
    match event {
        ClientEvent::Subscribed { subscription_id } => EventFeedback {
            event_count,
            line: format!("connected subscription={subscription_id}"),
            closed: false,
        },
        ClientEvent::Notification { result, .. } => {
            let event_count = event_count + 1;

            EventFeedback {
                event_count,
                line: event_count.to_string(),
                closed: false,
            }
        }
        ClientEvent::Closed => EventFeedback {
            event_count,
            line: "subscription closed".to_owned(),
            closed: true,
        },
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
    use client_evm::{ClientEvent, ClientEvmError};
    use serde_json::json;

    use super::*;

    #[test]
    fn subscribed_event_feedback_includes_subscription_id() {
        let feedback = format_event_feedback(
            &ClientEvent::Subscribed {
                subscription_id: "0xsubscription".to_owned(),
            },
            3,
        );

        assert_eq!(feedback.event_count, 3);
        assert_eq!(feedback.line, "connected subscription=0xsubscription");
        assert!(!feedback.closed);
    }

    #[test]
    fn closed_event_feedback_marks_closed() {
        let feedback = format_event_feedback(&ClientEvent::Closed, 3);

        assert_eq!(feedback.event_count, 3);
        assert_eq!(feedback.line, "subscription closed");
        assert!(feedback.closed);
    }

    #[test]
    fn subscription_worker_error_maps_to_cli_error() {
        let result = finish_subscription_worker(Ok(Err(
            ClientEvmError::InvalidSubscriptionConfig("bad config".to_owned()),
        )));

        assert_eq!(
            result,
            Err(CliError::SubscriptionFailed {
                message: "invalid subscription config: bad config".to_owned(),
            })
        );
    }

    #[test]
    fn subscription_worker_panic_maps_to_cli_error() {
        let result = finish_subscription_worker(Err(Box::new("panic message")));

        assert_eq!(
            result,
            Err(CliError::SubscriptionFailed {
                message: "subscription worker panicked".to_owned(),
            })
        );
    }
}
