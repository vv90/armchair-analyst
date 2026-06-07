use std::{
    env, io,
    io::Write,
    process::ExitCode,
    sync::mpsc::{self, Receiver, RecvTimeoutError},
    thread::{self, JoinHandle},
    time::Duration,
};

use client_evm::{ClientEvent, ClientEvmError, RpcConfig, subscribe_new_heads};

use crate::utils::{CliError, EventFeedback, format_event_feedback, load_rpc_config_with};

const EVENT_POLL_INTERVAL: Duration = Duration::from_millis(100);

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
    run_subscriptions(config)
}

fn load_rpc_config() -> Result<RpcConfig, CliError> {
    load_rpc_config_with(|name| env::var(name).ok(), prompt_for_value)
}

fn run_subscriptions(config: RpcConfig) -> Result<(), CliError> {
    let (sender, receiver) = mpsc::channel();
    let heads_sender = sender;
    let workers = vec![SubscriptionWorker::spawn("new heads", move || {
        subscribe_new_heads(&config, &heads_sender, Some)
    })];
    let mut output = io::stdout();

    receive_subscription_events(receiver, workers, &mut output)
}

struct SubscriptionWorker {
    name: &'static str,
    handle: JoinHandle<Result<(), ClientEvmError>>,
}

impl SubscriptionWorker {
    fn spawn<Run>(name: &'static str, run: Run) -> Self
    where
        Run: FnOnce() -> Result<(), ClientEvmError> + Send + 'static,
    {
        Self {
            name,
            handle: thread::spawn(run),
        }
    }
}

fn receive_subscription_events<Output>(
    receiver: Receiver<ClientEvent>,
    workers: Vec<SubscriptionWorker>,
    output: &mut Output,
) -> Result<(), CliError>
where
    Output: Write,
{
    receive_subscription_events_with_timeout(receiver, workers, output, EVENT_POLL_INTERVAL)
}

fn receive_subscription_events_with_timeout<Output>(
    receiver: Receiver<ClientEvent>,
    mut workers: Vec<SubscriptionWorker>,
    output: &mut Output,
    poll_interval: Duration,
) -> Result<(), CliError>
where
    Output: Write,
{
    let mut event_count = 0;

    loop {
        workers = join_finished_workers(workers)?;

        match receiver.recv_timeout(poll_interval) {
            Ok(event) => {
                let feedback = format_event_feedback(&event, event_count);
                event_count = feedback.event_count;
                write_event_feedback(output, &feedback)?;
            }
            Err(RecvTimeoutError::Timeout) => {
                if workers.is_empty() {
                    return Ok(());
                }
            }
            Err(RecvTimeoutError::Disconnected) => return join_remaining_workers(workers),
        }
    }
}

fn write_event_feedback<Output>(
    output: &mut Output,
    feedback: &EventFeedback,
) -> Result<(), CliError>
where
    Output: Write,
{
    writeln!(output, "{}", feedback.line).map_err(|error| CliError::SubscriptionFailed {
        message: format!("failed to write subscription event: {error}"),
    })
}

fn join_finished_workers(
    workers: Vec<SubscriptionWorker>,
) -> Result<Vec<SubscriptionWorker>, CliError> {
    let mut remaining_workers = Vec::new();

    for worker in workers {
        if worker.handle.is_finished() {
            finish_subscription_worker(worker.name, worker.handle.join())?;
        } else {
            remaining_workers.push(worker);
        }
    }

    Ok(remaining_workers)
}

fn join_remaining_workers(workers: Vec<SubscriptionWorker>) -> Result<(), CliError> {
    for worker in workers {
        finish_subscription_worker(worker.name, worker.handle.join())?;
    }

    Ok(())
}

fn finish_subscription_worker(
    name: &'static str,
    result: thread::Result<Result<(), ClientEvmError>>,
) -> Result<(), CliError> {
    match result {
        Ok(Ok(())) => Ok(()),
        Ok(Err(error)) => Err(CliError::SubscriptionFailed {
            message: format!("{name} worker failed: {error}"),
        }),
        Err(_) => Err(CliError::SubscriptionFailed {
            message: format!("{name} worker panicked"),
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
    use client_evm::{ClientEvent, ClientEvmError, ClientHead};
    use serde_json::json;

    use super::*;

    #[test]
    fn subscription_worker_error_maps_to_cli_error() {
        let result = finish_subscription_worker(
            "pool events",
            Ok(Err(ClientEvmError::InvalidSubscriptionConfig(
                "bad config".to_owned(),
            ))),
        );

        assert_eq!(
            result,
            Err(CliError::SubscriptionFailed {
                message: "pool events worker failed: invalid subscription config: bad config"
                    .to_owned(),
            })
        );
    }

    #[test]
    fn subscription_worker_panic_maps_to_cli_error() {
        let result = finish_subscription_worker("new heads", Err(Box::new("panic message")));

        assert_eq!(
            result,
            Err(CliError::SubscriptionFailed {
                message: "new heads worker panicked".to_owned(),
            })
        );
    }

    #[test]
    fn all_successful_workers_join() {
        let workers = vec![
            test_worker("pool events", || Ok(())),
            test_worker("new heads", || Ok(())),
        ];

        assert_eq!(join_remaining_workers(workers), Ok(()));
    }

    #[test]
    fn receive_loop_returns_finished_worker_error_while_sender_remains_open() {
        let (sender, receiver) = mpsc::channel();
        let workers = vec![test_worker("pool events", || {
            Err(ClientEvmError::InvalidSubscriptionConfig(
                "bad config".to_owned(),
            ))
        })];
        let mut output = Vec::new();

        let result = receive_subscription_events_with_timeout(
            receiver,
            workers,
            &mut output,
            std::time::Duration::from_millis(1),
        );

        drop(sender);
        assert_eq!(
            result,
            Err(CliError::SubscriptionFailed {
                message: "pool events worker failed: invalid subscription config: bad config"
                    .to_owned(),
            })
        );
    }

    #[test]
    fn closed_event_does_not_stop_processing_other_subscription_events()
    -> Result<(), serde_json::Error> {
        let (sender, receiver) = mpsc::channel();
        let pool_sender = sender.clone();
        let head_sender = sender.clone();
        let head = new_head()?;
        drop(sender);

        let workers = vec![
            test_worker("pool events", move || {
                pool_sender
                    .send(ClientEvent::Closed {
                        subscription_id: "0xpool".to_owned(),
                    })
                    .map_err(|_| ClientEvmError::EventReceiverDropped)
            }),
            test_worker("new heads", move || {
                thread::sleep(std::time::Duration::from_millis(5));
                head_sender
                    .send(ClientEvent::NewHead {
                        subscription_id: "0xheads".to_owned(),
                        header: head,
                    })
                    .map_err(|_| ClientEvmError::EventReceiverDropped)
            }),
        ];
        let mut output = Vec::new();

        let result = receive_subscription_events_with_timeout(
            receiver,
            workers,
            &mut output,
            std::time::Duration::from_millis(1),
        );

        assert_eq!(result, Ok(()));
        assert_eq!(
            String::from_utf8(output),
            Ok(
                "closed subscription=0xpool\nhead block=9 hash=0x0000000000000000000000000000000000000000000000000000000000000001\n"
                    .to_owned()
            )
        );

        Ok(())
    }

    fn new_head() -> Result<ClientHead, serde_json::Error> {
        serde_json::from_value(json!({
            "hash": "0x0000000000000000000000000000000000000000000000000000000000000001",
            "parentHash": "0x0000000000000000000000000000000000000000000000000000000000000002",
            "sha3Uncles": "0x0000000000000000000000000000000000000000000000000000000000000003",
            "miner": "0x0000000000000000000000000000000000000004",
            "stateRoot": "0x0000000000000000000000000000000000000000000000000000000000000005",
            "transactionsRoot": "0x0000000000000000000000000000000000000000000000000000000000000006",
            "receiptsRoot": "0x0000000000000000000000000000000000000000000000000000000000000007",
            "logsBloom": zero_logs_bloom(),
            "difficulty": "0xd",
            "number": "0x9",
            "gasLimit": "0xb",
            "gasUsed": "0xa",
            "timestamp": "0xc",
            "extraData": "0x010203",
            "mixHash": "0x000000000000000000000000000000000000000000000000000000000000000e",
            "nonce": "0x000000000000000f"
        }))
    }

    fn zero_logs_bloom() -> String {
        format!("0x{}", "00".repeat(256))
    }

    fn test_worker<Run>(name: &'static str, run: Run) -> SubscriptionWorker
    where
        Run: FnOnce() -> Result<(), ClientEvmError> + Send + 'static,
    {
        SubscriptionWorker {
            name,
            handle: thread::spawn(run),
        }
    }
}
