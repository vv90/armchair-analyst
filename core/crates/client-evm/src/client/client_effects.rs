use std::{net::TcpStream, str, sync::mpsc::Sender};

use alloy::rpc::types::Log;
use serde::Deserialize;
use serde_json::Value;
use tungstenite::{Message, WebSocket, connect, stream::MaybeTlsStream};

use crate::{ClientEvmError, RpcConfig};

use super::{
    ClientEvent,
    client_utils::{
        build_pool_events_subscribe_request, compose_ws_endpoint, parse_subscription_response,
    },
};

const SUBSCRIBE_REQUEST_ID: u64 = 1;

type BlockingWebSocket = WebSocket<MaybeTlsStream<TcpStream>>;

#[derive(Debug, Deserialize)]
struct SubscriptionDataParams<T> {
    subscription: String,
    result: T,
}

#[derive(Debug, Deserialize)]
struct SubscriptionNotification<T> {
    params: SubscriptionDataParams<T>,
}

pub fn subscribe_pool_events(
    config: RpcConfig,
    sender: Sender<ClientEvent>,
) -> Result<(), ClientEvmError> {
    let endpoint = compose_ws_endpoint(&config)?;
    let (mut socket, _) =
        connect(endpoint.as_str()).map_err(|ws_error| ClientEvmError::WebSocketError(ws_error))?;

    let subscribe_request = build_pool_events_subscribe_request(SUBSCRIBE_REQUEST_ID);
    socket
        .send(Message::text(subscribe_request.to_string()))
        .map_err(|ws_error| ClientEvmError::WebSocketError(ws_error))?;

    let subscription_id = read_subscription_id(&mut socket, SUBSCRIBE_REQUEST_ID)?;
    send_event(
        &sender,
        ClientEvent::Subscribed {
            subscription_id: subscription_id.clone(),
        },
    )?;

    read_subscription_events(&mut socket, &subscription_id, &sender)
}

fn read_subscription_id(
    socket: &mut BlockingWebSocket,
    expected_request_id: u64,
) -> Result<String, ClientEvmError> {
    loop {
        let Some(message) = read_json_rpc_message::<Value>(socket)? else {
            return Err(ClientEvmError::MalformedJsonRpcResponse(
                "websocket closed before subscription response".to_owned(),
            ));
        };

        if let Some(subscription_id) = parse_subscription_response(&message, expected_request_id)? {
            return Ok(subscription_id);
        }
    }
}

fn read_subscription_events(
    socket: &mut BlockingWebSocket,
    subscription_id: &str,
    sender: &Sender<ClientEvent>,
) -> Result<(), ClientEvmError> {
    loop {
        let Some(sub_notification) =
            read_json_rpc_message::<SubscriptionNotification<Log>>(socket)?
        else {
            send_event(sender, ClientEvent::Closed)?;
            return Ok(());
        };

        if sub_notification.params.subscription != subscription_id {
            println!(
                "Received notification for unexpected subscription: {}. Expected: {}. Ignoring.",
                sub_notification.params.subscription, subscription_id
            );
            continue;
        }

        send_event(
            sender,
            ClientEvent::Notification {
                subscription_id: sub_notification.params.subscription,
                result: sub_notification.params.result,
            },
        )?;
    }
}

fn read_json_rpc_message<T>(socket: &mut BlockingWebSocket) -> Result<Option<T>, ClientEvmError>
where
    T: for<'de> Deserialize<'de>,
{
    loop {
        match socket.read() {
            Ok(Message::Text(text)) => {
                let message = serde_json::from_slice::<T>(text.as_bytes())
                    .map_err(|error| ClientEvmError::JsonError(error))?;
                return Ok(Some(message));
            }
            Ok(Message::Binary(bytes)) => {
                let text = str::from_utf8(bytes.as_ref())
                    .map_err(|error| ClientEvmError::MalformedJsonRpcResponse(error.to_string()))?;
                let message = serde_json::from_str::<T>(text)
                    .map_err(|error| ClientEvmError::JsonError(error))?;
                return Ok(Some(message));
            }
            Ok(Message::Ping(_) | Message::Pong(_) | Message::Frame(_)) => {}
            Ok(Message::Close(_)) => return Ok(None),
            Err(tungstenite::Error::ConnectionClosed) => return Ok(None),
            Err(error) => return Err(ClientEvmError::WebSocketError(error)),
        }
    }
}

fn send_event(sender: &Sender<ClientEvent>, event: ClientEvent) -> Result<(), ClientEvmError> {
    sender
        .send(event)
        .map_err(|_| ClientEvmError::EventReceiverDropped)
}

#[cfg(test)]
mod tests {
    use alloy::{primitives::B256, rpc::types::Log as RpcLog};
    use serde_json::json;

    use super::*;

    #[test]
    fn subscription_notification_preserves_rpc_log_metadata() {
        let notification = json!({
            "jsonrpc": "2.0",
            "method": "eth_subscription",
            "params": {
                "subscription": "0xsubscription",
                "result": {
                    "address": "0x0000000000000000000000000000000000000001",
                    "topics": [
                        "0x0000000000000000000000000000000000000000000000000000000000000002"
                    ],
                    "data": "0x",
                    "blockHash": "0x0000000000000000000000000000000000000000000000000000000000000003",
                    "blockNumber": "0x4",
                    "transactionHash": "0x0000000000000000000000000000000000000000000000000000000000000005",
                    "transactionIndex": "0x6",
                    "logIndex": "0x7",
                    "removed": true
                }
            }
        });

        let result = serde_json::from_value::<SubscriptionNotification<RpcLog>>(notification);

        assert!(matches!(
            result,
            Ok(SubscriptionNotification {
                params: SubscriptionDataParams {
                    subscription,
                    result,
                },
            }) if subscription == "0xsubscription"
                && result.address().to_string()
                    == "0x0000000000000000000000000000000000000001"
                && result.topic0() == Some(&B256::with_last_byte(2))
                && result.block_hash == Some(B256::with_last_byte(3))
                && result.block_number == Some(4)
                && result.transaction_hash == Some(B256::with_last_byte(5))
                && result.transaction_index == Some(6)
                && result.log_index == Some(7)
                && result.removed
        ));
    }

    #[test]
    fn client_notification_carries_rpc_log() {
        let notification = json!({
            "params": {
                "subscription": "0xsubscription",
                "result": {
                    "address": "0x0000000000000000000000000000000000000001",
                    "topics": [],
                    "data": "0x",
                    "blockHash": "0x0000000000000000000000000000000000000000000000000000000000000003",
                    "blockNumber": "0x4",
                    "transactionHash": null,
                    "transactionIndex": null,
                    "logIndex": null,
                    "removed": false
                }
            }
        });
        let parsed = serde_json::from_value::<SubscriptionNotification<RpcLog>>(notification);

        assert!(matches!(
            parsed.map(|notification| ClientEvent::Notification {
                subscription_id: notification.params.subscription,
                result: notification.params.result,
            }),
            Ok(ClientEvent::Notification { result, .. })
                if result.block_hash == Some(B256::with_last_byte(3))
                    && result.block_number == Some(4)
        ));
    }
}
