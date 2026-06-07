use std::{
    collections::{HashMap, HashSet},
    net::TcpStream,
    str,
    sync::mpsc::Sender,
};

use alloy::primitives::BlockHash;
use serde::Deserialize;
use serde_json::Value;
use tungstenite::{Message, WebSocket, connect, stream::MaybeTlsStream};

use crate::{
    ClientEvmError, PoolAddress, PoolState, RpcConfig,
    config::{compose_http_endpoint, compose_ws_endpoint},
};

use super::{
    ClientEvent, ClientHead,
    client_utils::{
        build_block_header_request, build_block_logs_request, build_finalized_block_header_request,
        build_new_heads_subscribe_request, parse_block_header_response,
        parse_block_header_response_by_id, parse_block_logs_response, parse_subscription_response,
    },
};

const HTTP_REQUEST_ID: u64 = 1;
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

pub fn fetch_block_header(
    agent: &ureq::Agent,
    config: &RpcConfig,
    block_hash: BlockHash,
) -> Result<Option<ClientHead>, ClientEvmError> {
    let endpoint = compose_http_endpoint(config)?;
    let request = build_block_header_request(HTTP_REQUEST_ID, block_hash);
    let mut response = agent
        .post(endpoint.as_str())
        .send_json(&request)
        .map_err(ClientEvmError::HttpError)?;
    let response_value = response
        .body_mut()
        .read_json::<Value>()
        .map_err(ClientEvmError::HttpError)?;

    parse_block_header_response(&response_value, HTTP_REQUEST_ID, block_hash)
}

pub fn fetch_block_logs(
    agent: &ureq::Agent,
    config: &RpcConfig,
    block_hash: BlockHash,
) -> Result<HashSet<PoolAddress>, ClientEvmError> {
    let endpoint = compose_http_endpoint(config)?;
    let request = build_block_logs_request(HTTP_REQUEST_ID, block_hash);
    let mut response = agent
        .post(endpoint.as_str())
        .send_json(&request)
        .map_err(ClientEvmError::HttpError)?;
    let response_value = response
        .body_mut()
        .read_json::<Value>()
        .map_err(ClientEvmError::HttpError)?;

    parse_block_logs_response(&response_value, HTTP_REQUEST_ID, block_hash)
}

pub fn fetch_pool_data(
    _agent: &ureq::Agent,
    _config: &RpcConfig,
    _at: BlockHash,
    _pools: HashSet<PoolAddress>,
) -> Result<HashMap<PoolAddress, PoolState>, ClientEvmError> {
    Ok(HashMap::new())
}

pub fn fetch_finalized_block_header(
    agent: &ureq::Agent,
    config: &RpcConfig,
) -> Result<Option<ClientHead>, ClientEvmError> {
    let endpoint = compose_http_endpoint(config)?;
    let request = build_finalized_block_header_request(HTTP_REQUEST_ID);
    let mut response = agent
        .post(endpoint.as_str())
        .send_json(&request)
        .map_err(ClientEvmError::HttpError)?;
    let response_value = response
        .body_mut()
        .read_json::<Value>()
        .map_err(ClientEvmError::HttpError)?;

    parse_block_header_response_by_id(&response_value, HTTP_REQUEST_ID)
}

pub fn subscribe_new_heads<T, F>(
    config: &RpcConfig,
    sender: &Sender<T>,
    map_event: F,
) -> Result<(), ClientEvmError>
where
    F: Fn(ClientEvent) -> Option<T>,
{
    let endpoint = compose_ws_endpoint(&config)?;
    let (mut socket, _) =
        connect(endpoint.as_str()).map_err(|ws_error| ClientEvmError::WebSocketError(ws_error))?;

    let subscribe_request = build_new_heads_subscribe_request(SUBSCRIBE_REQUEST_ID);
    socket
        .send(Message::text(subscribe_request.to_string()))
        .map_err(|ws_error| ClientEvmError::WebSocketError(ws_error))?;

    let subscription_id = read_subscription_id(&mut socket, SUBSCRIBE_REQUEST_ID)?;
    send_event(
        sender,
        ClientEvent::Subscribed {
            subscription_id: subscription_id.clone(),
        },
        &map_event,
    )?;

    read_new_head_events(&mut socket, &subscription_id, sender, &map_event)
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

fn read_new_head_events<T, F>(
    socket: &mut BlockingWebSocket,
    subscription_id: &str,
    sender: &Sender<T>,
    map_event: &F,
) -> Result<(), ClientEvmError>
where
    F: Fn(ClientEvent) -> Option<T>,
{
    loop {
        let Some(sub_notification) =
            read_json_rpc_message::<SubscriptionNotification<ClientHead>>(socket)?
        else {
            send_event(
                sender,
                ClientEvent::Closed {
                    subscription_id: subscription_id.to_owned(),
                },
                map_event,
            )?;
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
            ClientEvent::NewHead {
                subscription_id: sub_notification.params.subscription,
                header: sub_notification.params.result,
            },
            map_event,
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

fn send_event<T, F>(
    sender: &Sender<T>,
    event: ClientEvent,
    map_event: &F,
) -> Result<(), ClientEvmError>
where
    F: Fn(ClientEvent) -> Option<T>,
{
    match map_event(event) {
        Some(e) => sender
            .send(e)
            .map_err(|_| ClientEvmError::EventReceiverDropped),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashSet,
        io::{Read, Write},
        net::TcpListener,
        sync::mpsc::{Receiver, channel},
        thread::{self, JoinHandle},
    };

    use alloy::primitives::{Address, B256};
    use serde_json::{Value, json};

    use crate::PoolAddress;

    use super::*;

    #[test]
    fn fetch_block_header_posts_expected_request_and_decodes_response() {
        let block_hash = B256::with_last_byte(1);
        let response = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": block_header_result(block_hash)
        });
        let (http_url, received_request, server) = spawn_json_rpc_server(response);
        let config = rpc_config(&http_url);
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_block_header(&agent, &config, block_hash);

        assert!(matches!(
            result,
            Ok(Some(header))
                if header.inner.hash == block_hash
                    && header.inner.inner.parent_hash == B256::with_last_byte(2)
        ));

        let request = received_request
            .recv()
            .expect("server must report received request");
        assert_eq!(request.path, "/ethereum/api-key");
        assert_eq!(
            request.body,
            json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "eth_getBlockByHash",
                "params": [block_hash, false]
            })
        );
        server.join().expect("server thread must complete");
    }

    #[test]
    fn fetch_block_header_maps_transport_failure_to_http_error() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("test listener must bind");
        let address = listener
            .local_addr()
            .expect("test listener must have local address");
        drop(listener);
        let config = rpc_config(&format!("http://{address}"));
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_block_header(&agent, &config, B256::with_last_byte(1));

        assert!(matches!(result, Err(ClientEvmError::HttpError(_))));
    }

    #[test]
    fn fetch_block_logs_posts_expected_request_and_decodes_pool_addresses() {
        let block_hash = B256::with_last_byte(1);
        let first_pool = Address::with_last_byte(2);
        let second_pool = Address::with_last_byte(3);
        let response = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": [
                log_result(first_pool, block_hash),
                log_result(second_pool, block_hash),
                log_result(first_pool, block_hash)
            ]
        });
        let (http_url, received_request, server) = spawn_json_rpc_server(response);
        let config = rpc_config(&http_url);
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_block_logs(&agent, &config, block_hash);

        assert!(matches!(
            result,
            Ok(ref pools)
                if pools.len() == 2
                    && pools.contains(&PoolAddress(first_pool))
                    && pools.contains(&PoolAddress(second_pool))
        ));

        let request = received_request
            .recv()
            .expect("server must report received request");
        assert_eq!(request.path, "/ethereum/api-key");
        assert_eq!(
            request.body,
            json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "eth_getLogs",
                "params": [{
                    "blockHash": block_hash,
                    "topics": [crate::uniswap_v3::pool_event_signature_hashes()
                        .into_iter()
                        .map(|topic| topic.to_string())
                        .collect::<Vec<_>>()]
                }]
            })
        );
        server.join().expect("server thread must complete");
    }

    #[test]
    fn fetch_block_logs_maps_transport_failure_to_http_error() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("test listener must bind");
        let address = listener
            .local_addr()
            .expect("test listener must have local address");
        drop(listener);
        let config = rpc_config(&format!("http://{address}"));
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_block_logs(&agent, &config, B256::with_last_byte(1));

        assert!(matches!(result, Err(ClientEvmError::HttpError(_))));
    }

    #[test]
    fn fetch_pool_data_placeholder_returns_empty_results() {
        let config = rpc_config("http://127.0.0.1:9");
        let agent = ureq::Agent::new_with_defaults();
        let requested_pools = HashSet::from([PoolAddress(Address::with_last_byte(2))]);

        let result = fetch_pool_data(&agent, &config, B256::with_last_byte(1), requested_pools);

        assert!(matches!(result, Ok(ref pools) if pools.is_empty()));
    }

    #[test]
    fn fetch_finalized_block_header_posts_expected_request_and_decodes_response() {
        let block_hash = B256::with_last_byte(1);
        let response = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": block_header_result(block_hash)
        });
        let (http_url, received_request, server) = spawn_json_rpc_server(response);
        let config = rpc_config(&http_url);
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_finalized_block_header(&agent, &config);

        assert!(matches!(
            result,
            Ok(Some(header))
                if header.inner.hash == block_hash
                    && header.inner.inner.parent_hash == B256::with_last_byte(2)
        ));

        let request = received_request
            .recv()
            .expect("server must report received request");
        assert_eq!(request.path, "/ethereum/api-key");
        assert_eq!(
            request.body,
            json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "eth_getBlockByNumber",
                "params": ["finalized", false]
            })
        );
        server.join().expect("server thread must complete");
    }

    #[test]
    fn fetch_finalized_block_header_maps_transport_failure_to_http_error() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("test listener must bind");
        let address = listener
            .local_addr()
            .expect("test listener must have local address");
        drop(listener);
        let config = rpc_config(&format!("http://{address}"));
        let agent = ureq::Agent::new_with_defaults();

        let result = fetch_finalized_block_header(&agent, &config);

        assert!(matches!(result, Err(ClientEvmError::HttpError(_))));
    }

    #[test]
    fn new_head_notification_preserves_header_fields() {
        let result =
            serde_json::from_value::<SubscriptionNotification<ClientHead>>(new_head_notification());

        assert!(matches!(
            result,
            Ok(SubscriptionNotification {
                params: SubscriptionDataParams {
                    subscription,
                    result,
                },
            }) if subscription == "0xsubscription"
                && result.inner.hash == B256::with_last_byte(1)
                && result.inner.inner.parent_hash == B256::with_last_byte(2)
                && result.inner.inner.number == 9
                && result.inner.inner.timestamp == 12
        ));
    }

    #[test]
    fn new_head_notification_preserves_extra_header_fields() {
        let result =
            serde_json::from_value::<SubscriptionNotification<ClientHead>>(new_head_notification());

        assert!(matches!(
            result,
            Ok(SubscriptionNotification {
                params: SubscriptionDataParams { result, .. },
            }) if matches!(
                result.other.get_deserialized::<String>("providerTag"),
                Some(Ok(ref tag)) if tag == "observed"
            )
        ));
    }

    #[test]
    fn client_new_head_event_carries_header() {
        let parsed =
            serde_json::from_value::<SubscriptionNotification<ClientHead>>(new_head_notification());

        assert!(matches!(
            parsed.map(|notification| ClientEvent::NewHead {
                subscription_id: notification.params.subscription,
                header: notification.params.result,
            }),
            Ok(ClientEvent::NewHead { header, .. })
                if header.inner.hash == B256::with_last_byte(1)
                    && header.inner.inner.number == 9
        ));
    }

    fn new_head_notification() -> Value {
        json!({
            "jsonrpc": "2.0",
            "method": "eth_subscription",
            "params": {
                "subscription": "0xsubscription",
                "result": {
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
                    "nonce": "0x000000000000000f",
                    "providerTag": "observed"
                }
            }
        })
    }

    fn zero_logs_bloom() -> String {
        format!("0x{}", "00".repeat(256))
    }

    struct ReceivedHttpRequest {
        path: String,
        body: Value,
    }

    fn spawn_json_rpc_server(
        response: Value,
    ) -> (String, Receiver<ReceivedHttpRequest>, JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("test server must bind");
        let address = listener
            .local_addr()
            .expect("test server must have local address");
        let (sender, receiver) = channel();

        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("test server must accept request");
            let request = read_http_request(&mut stream);
            sender
                .send(request)
                .expect("test server must report received request");

            let response_body =
                serde_json::to_vec(&response).expect("test response must serialize");
            let response_headers = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                response_body.len()
            );

            stream
                .write_all(response_headers.as_bytes())
                .expect("test server must write response headers");
            stream
                .write_all(&response_body)
                .expect("test server must write response body");
        });

        (format!("http://{address}"), receiver, handle)
    }

    fn read_http_request(stream: &mut TcpStream) -> ReceivedHttpRequest {
        let mut request_bytes = Vec::new();
        let mut buffer = [0; 1024];
        let (body_start, content_length) = loop {
            let bytes_read = stream
                .read(&mut buffer)
                .expect("test server must read request");
            assert!(bytes_read > 0, "request must contain headers and body");
            request_bytes.extend_from_slice(&buffer[..bytes_read]);

            if let Some(header_end) = request_bytes
                .windows(4)
                .position(|window| window == b"\r\n\r\n")
            {
                let body_start = header_end + 4;
                let headers = str::from_utf8(&request_bytes[..header_end])
                    .expect("request headers must be utf-8");
                let content_length = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().ok())
                            .flatten()
                    })
                    .expect("request must contain content-length");

                if request_bytes.len() >= body_start + content_length {
                    break (body_start, content_length);
                }
            }
        };

        let headers =
            str::from_utf8(&request_bytes[..body_start]).expect("request headers must be utf-8");
        let path = headers
            .lines()
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
            .expect("request line must contain path")
            .to_owned();
        let body = serde_json::from_slice(&request_bytes[body_start..body_start + content_length])
            .expect("request body must be json");

        ReceivedHttpRequest { path, body }
    }

    fn rpc_config(http_url: &str) -> RpcConfig {
        RpcConfig {
            chain: crate::ChainKey::Ethereum,
            http_url: http_url.to_owned(),
            ws_url: "wss://example.invalid".to_owned(),
            api_key: "api-key".to_owned(),
        }
    }

    fn block_header_result(block_hash: B256) -> Value {
        json!({
            "hash": block_hash,
            "parentHash": B256::with_last_byte(2),
            "sha3Uncles": B256::with_last_byte(3),
            "miner": "0x0000000000000000000000000000000000000004",
            "stateRoot": B256::with_last_byte(5),
            "transactionsRoot": B256::with_last_byte(6),
            "receiptsRoot": B256::with_last_byte(7),
            "logsBloom": zero_logs_bloom(),
            "difficulty": "0xd",
            "number": "0x9",
            "gasLimit": "0xb",
            "gasUsed": "0xa",
            "timestamp": "0xc",
            "extraData": "0x010203",
            "mixHash": B256::with_last_byte(14),
            "nonce": "0x000000000000000f",
            "providerTag": "observed"
        })
    }

    fn log_result(address: Address, block_hash: B256) -> Value {
        json!({
            "address": address,
            "topics": [
                crate::uniswap_v3::pool_event_signature_hashes()[0]
            ],
            "data": "0x",
            "blockHash": block_hash,
            "blockNumber": "0x4",
            "transactionHash": B256::with_last_byte(5),
            "transactionIndex": "0x6",
            "logIndex": "0x7",
            "removed": false
        })
    }
}
