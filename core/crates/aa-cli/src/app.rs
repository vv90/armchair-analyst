use aa_framework::{Application, ApplicationError, Runtime, Transition};
use client_evm::{
    AnyIssuedRequest, AnyRequestId, BlockHash, ChainKey, ClientEvent, ClientEvmError, ClientHead,
    PoolAddress, PoolState, RpcConfig, fetch_block_header, fetch_block_logs,
    fetch_finalized_block_header, fetch_pool_data, kernel,
    multi_chain_kernel::{Effect, Event, State, transition},
    subscribe_new_heads,
};
use std::{
    collections::{HashMap, HashSet},
    sync::mpsc::Sender,
    time,
};

struct ClientEvmApp {}

struct ClientEvmRuntime {
    agent: ureq::Agent,
    ethereum_config: RpcConfig,
}

impl ClientEvmRuntime {
    fn get_config(&self, chain: ChainKey) -> &RpcConfig {
        match chain {
            ChainKey::Ethereum => &self.ethereum_config,
        }
    }
}

enum ClientEvmSubscription {
    NewHeadsSubscription(ChainKey),
    TickSubscription(time::Duration),
}

impl Application for ClientEvmApp {
    type State = State;
    type Input = Event;
    type Effect = Effect;
    type Subscription = ClientEvmSubscription;

    fn init() -> Transition<Self::State, Self::Effect> {
        let (state, effects) = State::init(client_evm::ChainKey::Ethereum);
        Transition { state, effects }
    }

    fn transition(state: Self::State, input: Self::Input) -> Transition<Self::State, Self::Effect> {
        let (new_state, effects) = transition(state, input);
        Transition {
            state: new_state,
            effects,
        }
    }

    fn subscriptions() -> Vec<Self::Subscription> {
        vec![
            ClientEvmSubscription::NewHeadsSubscription(ChainKey::Ethereum),
            ClientEvmSubscription::TickSubscription(time::Duration::from_millis(1000)),
        ]
    }
}

impl Runtime<ClientEvmApp> for ClientEvmRuntime {
    fn execute_effect(
        &self,
        effect: <ClientEvmApp as Application>::Effect,
    ) -> Vec<<ClientEvmApp as Application>::Input> {
        match effect {
            Effect::FetchFinalizedHeader { chain } => {
                let r = fetch_finalized_block_header(&self.agent, self.get_config(chain));
                match r {
                    Ok(Some(header)) => vec![Event::FinalizedHeaderReceived {
                        chain,
                        block_hash: header.inner.hash,
                    }],
                    Ok(None) => vec![Event::FinalizedHeaderUnavailable { chain }],
                    Err(_) => vec![Event::FinalizedHeaderUnavailable { chain }],
                }
            }
            Effect::ChainEffect { chain, effect } => self.execute_chain_effect(chain, effect),
        }
    }

    fn spawn_subscription(
        &self,
        sender: &Sender<<ClientEvmApp as Application>::Input>,
        subscription: <ClientEvmApp as Application>::Subscription,
    ) {
        match subscription {
            ClientEvmSubscription::NewHeadsSubscription(chain) => {
                let map_client_event = |client_event: ClientEvent| {
                    map_client_chain_event(client_event)
                        .map(|event| Event::ChainEvent { chain, event })
                };
                let _ = subscribe_new_heads(self.get_config(chain), sender, map_client_event);
            }
            ClientEvmSubscription::TickSubscription(_interval) => {}
        }
    }

    fn log(&self, _error: ApplicationError<<ClientEvmApp as Application>::Input>) {}
}

impl ClientEvmRuntime {
    fn execute_chain_effect(&self, chain: ChainKey, effect: kernel::Effect) -> Vec<Event> {
        execute_chain_effect_with(
            chain,
            effect,
            |block_hash| fetch_block_header(&self.agent, self.get_config(chain), block_hash),
            |block_hash| fetch_block_logs(&self.agent, self.get_config(chain), block_hash),
            |at, pools| fetch_pool_data(&self.agent, self.get_config(chain), at, pools),
        )
    }
}

fn map_client_chain_event(client_chain_event: ClientEvent) -> Option<client_evm::kernel::Event> {
    match client_chain_event {
        ClientEvent::NewHead { header, .. } => Some(client_evm::kernel::Event::HeadObserved {
            hash: header.inner.hash,
            parent_hash: header.inner.inner.parent_hash,
        }),
        ClientEvent::Subscribed { .. } => None,
        ClientEvent::Closed { .. } => None,
    }
}

fn execute_chain_effect_with<FetchBlockHeader, FetchBlockLogs, FetchPoolData>(
    chain: ChainKey,
    effect: kernel::Effect,
    fetch_block_header: FetchBlockHeader,
    fetch_block_logs: FetchBlockLogs,
    fetch_pool_data: FetchPoolData,
) -> Vec<Event>
where
    FetchBlockHeader: FnOnce(BlockHash) -> Result<Option<ClientHead>, ClientEvmError>,
    FetchBlockLogs: FnOnce(BlockHash) -> Result<HashSet<PoolAddress>, ClientEvmError>,
    FetchPoolData: FnOnce(
        BlockHash,
        HashSet<PoolAddress>,
    ) -> Result<HashMap<PoolAddress, PoolState>, ClientEvmError>,
{
    match effect {
        kernel::Effect::Request(request) => match request {
            AnyIssuedRequest::BlockHeader(request) => {
                let request_id = request.request_id;
                let block_hash = request.request_payload.block_hash;

                match fetch_block_header(block_hash) {
                    Ok(Some(header)) => vec![Event::ChainEvent {
                        chain,
                        event: kernel::Event::BlockHeaderReceived {
                            request_id,
                            hash: header.inner.hash,
                            parent_hash: header.inner.inner.parent_hash,
                        },
                    }],
                    Ok(None) => vec![Event::ChainEvent {
                        chain,
                        event: kernel::Event::BlockHeaderNotFound { request_id },
                    }],
                    Err(_) => vec![Event::ChainEvent {
                        chain,
                        event: kernel::Event::RequestFailed {
                            request_id: AnyRequestId::BlockHeader(request_id),
                        },
                    }],
                }
            }
            AnyIssuedRequest::BlockLogs(request) => {
                let request_id = request.request_id;
                let block_hash = request.request_payload.block_hash;

                match fetch_block_logs(block_hash) {
                    Ok(logs) => vec![Event::ChainEvent {
                        chain,
                        event: kernel::Event::BlockLogsReceived { request_id, logs },
                    }],
                    Err(_) => vec![Event::ChainEvent {
                        chain,
                        event: kernel::Event::RequestFailed {
                            request_id: AnyRequestId::BlockLogs(request_id),
                        },
                    }],
                }
            }
            AnyIssuedRequest::PoolData(request) => {
                let request_id = request.request_id;
                let at = request.request_payload.at;
                let pools = request.request_payload.pools;

                match fetch_pool_data(at, pools) {
                    Ok(pools) => vec![Event::ChainEvent {
                        chain,
                        event: kernel::Event::PoolDataReceived { request_id, pools },
                    }],
                    Err(_) => vec![Event::ChainEvent {
                        chain,
                        event: kernel::Event::RequestFailed {
                            request_id: AnyRequestId::PoolData(request_id),
                        },
                    }],
                }
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use client_evm::{
        GetBlockHeader, GetBlockLogs, GetPoolData, IssuedRequest, PoolAddress, PoolState, RequestId,
    };
    use serde_json::json;

    use super::*;

    #[test]
    fn block_header_request_success_maps_to_chain_event() -> Result<(), serde_json::Error> {
        let chain = ChainKey::Ethereum;
        let requested_hash = hash(2);
        let parent_hash = hash(4);
        let (effect, expected_request_id) = block_header_request_effect(requested_hash);
        let header = block_header(requested_hash, parent_hash)?;

        let events = execute_chain_effect_with(
            chain,
            effect,
            |block_hash| {
                assert_eq!(block_hash, requested_hash);
                Ok(Some(header))
            },
            unexpected_block_logs_fetch,
            unexpected_pool_data_fetch,
        );

        assert!(matches!(
            events.as_slice(),
            [Event::ChainEvent {
                chain: event_chain,
                event:
                    kernel::Event::BlockHeaderReceived {
                        request_id,
                        hash: event_hash,
                        parent_hash: event_parent_hash,
                    },
            }] if *event_chain == chain
                && *request_id == expected_request_id
                && *event_hash == requested_hash
                && *event_parent_hash == parent_hash
        ));

        Ok(())
    }

    #[test]
    fn block_header_request_not_found_maps_to_chain_event() {
        let chain = ChainKey::Ethereum;
        let requested_hash = hash(2);
        let (effect, expected_request_id) = block_header_request_effect(requested_hash);

        let events = execute_chain_effect_with(
            chain,
            effect,
            |block_hash| {
                assert_eq!(block_hash, requested_hash);
                Ok(None)
            },
            unexpected_block_logs_fetch,
            unexpected_pool_data_fetch,
        );

        assert!(matches!(
            events.as_slice(),
            [Event::ChainEvent {
                chain: event_chain,
                event: kernel::Event::BlockHeaderNotFound { request_id },
            }] if *event_chain == chain && *request_id == expected_request_id
        ));
    }

    #[test]
    fn block_header_request_failure_maps_to_request_failed_event() {
        let chain = ChainKey::Ethereum;
        let requested_hash = hash(2);
        let (effect, expected_request_id) = block_header_request_effect(requested_hash);

        let events = execute_chain_effect_with(
            chain,
            effect,
            |block_hash| {
                assert_eq!(block_hash, requested_hash);
                Err(ClientEvmError::InvalidHttpConfig("bad config".to_owned()))
            },
            unexpected_block_logs_fetch,
            unexpected_pool_data_fetch,
        );

        assert!(matches!(
            events.as_slice(),
            [Event::ChainEvent {
                chain: event_chain,
                event:
                    kernel::Event::RequestFailed {
                        request_id: client_evm::AnyRequestId::BlockHeader(request_id),
                    },
            }] if *event_chain == chain && *request_id == expected_request_id
        ));
    }

    #[test]
    fn block_logs_request_success_maps_to_chain_event() {
        let chain = ChainKey::Ethereum;
        let block_hash = hash(2);
        let (effect, expected_request_id) = block_logs_request_effect(block_hash);

        let events = execute_chain_effect_with(
            chain,
            effect,
            unexpected_block_header_fetch,
            |requested_hash| {
                assert_eq!(requested_hash, block_hash);
                Ok(HashSet::new())
            },
            unexpected_pool_data_fetch,
        );

        assert!(matches!(
            events.as_slice(),
            [Event::ChainEvent {
                chain: event_chain,
                event:
                    kernel::Event::BlockLogsReceived {
                        request_id,
                        logs,
                    },
            }] if *event_chain == chain
                && *request_id == expected_request_id
                && logs.is_empty()
        ));
    }

    #[test]
    fn block_logs_request_failure_maps_to_request_failed_event() {
        let chain = ChainKey::Ethereum;
        let block_hash = hash(2);
        let (effect, expected_request_id) = block_logs_request_effect(block_hash);

        let events = execute_chain_effect_with(
            chain,
            effect,
            unexpected_block_header_fetch,
            |requested_hash| {
                assert_eq!(requested_hash, block_hash);
                Err(ClientEvmError::InvalidHttpConfig("bad config".to_owned()))
            },
            unexpected_pool_data_fetch,
        );

        assert!(matches!(
            events.as_slice(),
            [Event::ChainEvent {
                chain: event_chain,
                event:
                    kernel::Event::RequestFailed {
                        request_id: client_evm::AnyRequestId::BlockLogs(request_id),
                    },
            }] if *event_chain == chain && *request_id == expected_request_id
        ));
    }

    #[test]
    fn pool_data_request_success_maps_to_chain_event() {
        let chain = ChainKey::Ethereum;
        let at = hash(2);
        let (effect, expected_request_id) = pool_data_request_effect(at);

        let events = execute_chain_effect_with(
            chain,
            effect,
            unexpected_block_header_fetch,
            unexpected_block_logs_fetch,
            |requested_at, requested_pools| {
                assert_eq!(requested_at, at);
                assert!(requested_pools.is_empty());
                Ok(HashMap::new())
            },
        );

        assert!(matches!(
            events.as_slice(),
            [Event::ChainEvent {
                chain: event_chain,
                event:
                    kernel::Event::PoolDataReceived {
                        request_id,
                        pools,
                    },
            }] if *event_chain == chain
                && *request_id == expected_request_id
                && pools.is_empty()
        ));
    }

    #[test]
    fn pool_data_request_failure_maps_to_request_failed_event() {
        let chain = ChainKey::Ethereum;
        let at = hash(2);
        let (effect, expected_request_id) = pool_data_request_effect(at);

        let events = execute_chain_effect_with(
            chain,
            effect,
            unexpected_block_header_fetch,
            unexpected_block_logs_fetch,
            |requested_at, requested_pools| {
                assert_eq!(requested_at, at);
                assert!(requested_pools.is_empty());
                Err(ClientEvmError::InvalidHttpConfig("bad config".to_owned()))
            },
        );

        assert!(matches!(
            events.as_slice(),
            [Event::ChainEvent {
                chain: event_chain,
                event:
                    kernel::Event::RequestFailed {
                        request_id: client_evm::AnyRequestId::PoolData(request_id),
                    },
            }] if *event_chain == chain && *request_id == expected_request_id
        ));
    }

    fn block_header_request_effect(
        block_hash: BlockHash,
    ) -> (kernel::Effect, RequestId<GetBlockHeader>) {
        let finalized_hash = hash(1);
        let observed_hash = hash(3);
        let state = kernel::State::init(kernel::FinalizedState::empty_at(finalized_hash));

        let (_state, effects) = kernel::transition(
            state,
            kernel::Event::HeadObserved {
                hash: observed_hash,
                parent_hash: block_hash,
            },
        );

        let effect = effects
            .into_iter()
            .next()
            .expect("missing parent should request block header");
        let request_id = match &effect {
            kernel::Effect::Request(AnyIssuedRequest::BlockHeader(request)) => request.request_id,
            _ => panic!("expected block header request"),
        };

        (effect, request_id)
    }

    fn block_logs_request_effect(
        block_hash: BlockHash,
    ) -> (kernel::Effect, RequestId<GetBlockLogs>) {
        let request_id = RequestId::from_raw_for_test(7);
        (
            kernel::Effect::Request(AnyIssuedRequest::BlockLogs(IssuedRequest {
                request_id,
                request_payload: GetBlockLogs { block_hash },
            })),
            request_id,
        )
    }

    fn pool_data_request_effect(at: BlockHash) -> (kernel::Effect, RequestId<GetPoolData>) {
        let request_id = RequestId::from_raw_for_test(8);
        (
            kernel::Effect::Request(AnyIssuedRequest::PoolData(IssuedRequest {
                request_id,
                request_payload: GetPoolData {
                    at,
                    pools: HashSet::new(),
                },
            })),
            request_id,
        )
    }

    fn unexpected_block_header_fetch(
        _block_hash: BlockHash,
    ) -> Result<Option<ClientHead>, ClientEvmError> {
        panic!("block header fetch must not be called")
    }

    fn unexpected_block_logs_fetch(
        _block_hash: BlockHash,
    ) -> Result<HashSet<PoolAddress>, ClientEvmError> {
        panic!("block logs fetch must not be called")
    }

    fn unexpected_pool_data_fetch(
        _at: BlockHash,
        _pools: HashSet<PoolAddress>,
    ) -> Result<HashMap<PoolAddress, PoolState>, ClientEvmError> {
        panic!("pool data fetch must not be called")
    }

    fn block_header(
        block_hash: BlockHash,
        parent_hash: BlockHash,
    ) -> Result<ClientHead, serde_json::Error> {
        serde_json::from_value(json!({
            "hash": block_hash,
            "parentHash": parent_hash,
            "sha3Uncles": hash(5),
            "miner": "0x0000000000000000000000000000000000000006",
            "stateRoot": hash(7),
            "transactionsRoot": hash(8),
            "receiptsRoot": hash(9),
            "logsBloom": zero_logs_bloom(),
            "difficulty": "0xd",
            "number": "0x9",
            "gasLimit": "0xb",
            "gasUsed": "0xa",
            "timestamp": "0xc",
            "extraData": "0x010203",
            "mixHash": hash(10),
            "nonce": "0x000000000000000f"
        }))
    }

    fn hash(value: u8) -> BlockHash {
        BlockHash::with_last_byte(value)
    }

    fn zero_logs_bloom() -> String {
        format!("0x{}", "00".repeat(256))
    }
}
