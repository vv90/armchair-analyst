# Incremental Plan: Blocking Uniswap V3 Pool Subscriptions

## Summary

No code changes first. Execution starts by saving this plan to `docs/client-evm-uniswap-v3-pool-subscriptions-plan.md`, then implements one narrow layer at a time: dependencies, ABI decoding, JSON-RPC shaping, blocking WebSocket transport, event reading, and close behavior.

## Implementation Steps

1. [x] Persist the plan file.
   - Add `docs/client-evm-uniswap-v3-pool-subscriptions-plan.md` containing this incremental plan.
   - Verify the diff only adds that docs file.

2. [x] Add dependencies only.
   - Update `client-evm/Cargo.toml` with `alloy = 2.0.5` using `default-features = false` and `features = ["std", "sol-types", "serde"]`.
   - Add `serde` with `derive`, `serde_json`, `thiserror`, and blocking `tungstenite` with Rustls TLS.
   - Run `cargo check -p client-evm`.

3. [x] Add public error and module skeleton.
   - Introduce `ClientEvmError`.
   - Add `uniswap_v3` module exports, but no transport logic.
   - Run `cargo check -p client-evm`.

4. [x] Add Uniswap v3 pool ABI types.
   - Use Alloy `sol!` to define all canonical pool events from `IUniswapV3PoolEvents.sol`.
   - Add a helper returning the 9 known topic0 hashes.
   - Unit-test event count and topic hash uniqueness.

5. [x] Add RPC config, event documentation, and pure subscription request builders.
   - Define `EvmNetwork { Ethereum }` and `RpcConfig { network, http_url, ws_url, api_key }` in `client-evm`.
   - Document all canonical Uniswap v3 pool events, their state impact, topic-only discovery, and future validation options.
   - Build pure `eth_subscribe` and `eth_unsubscribe` JSON-RPC messages.
   - Subscribe with topic0 filters only and discover candidate pools from `log.address`.
   - Unit-test subscribe/unsubscribe JSON shape.

6. [x] Add blocking WebSocket subscription function with raw channel events.
   - Add `client-evm/src/client.rs`.
   - Define `ClientEvent` with `Subscribed`, raw JSON `Notification`, and `Closed` variants.
   - Expose `subscribe_uniswap_v3_pool_events(config: RpcConfig, sender: std::sync::mpsc::Sender<ClientEvent>)`.
   - Compose the dRPC-style WebSocket endpoint from `RpcConfig`.
   - Connect with blocking `tungstenite`, send the topic-only subscribe request, parse the subscription ID, and forward matching raw notifications.
   - Unit-test endpoint composition and pure JSON-RPC response/notification parsing without live network calls.

7. Add typed pool event model.
   - Define `PoolEvent`, `PoolEventKind`, and log metadata fields.
   - Store the discovered candidate pool address from the EVM log address.
   - Wrap decoded Alloy event structs in `PoolEventKind`.
   - Run `cargo test -p client-evm`.

8. Add raw JSON-RPC log parsing.
   - Define internal serde structs for `eth_subscription` notifications and log payloads.
   - Parse address, topics, data, block/tx/log indexes, hashes, and `removed`.
   - Unit-test with fixture notification JSON.

9. Add ABI decoder.
   - Implement `decode_pool_log(raw) -> Result<PoolEvent, ClientEvmError>`.
   - Match `topic0`; decode with Alloy `SolEvent::decode_raw_log`.
   - Unit-test every event variant with Alloy-encoded sample events.

10. Send typed decoded events from the WebSocket subscription function.
   - Change or extend `ClientEvent::Notification` so callers can receive decoded `PoolEvent` values.
   - Use the raw log parser and ABI decoder inside the blocking read loop.
   - Continue ignoring unrelated subscription IDs and returning errors for malformed matching logs.
   - Unit-test message handling through pure helpers.

11. Add close, unsubscribe, and cancellation behavior.
   - Add a way for callers to stop the blocking subscription loop.
   - Send `eth_unsubscribe` for the stored subscription ID before closing the WebSocket.
   - Keep reconnect and retry behavior out of scope for this increment.

12. Add future validation, backfill, and reconnect follow-ups.
   - Validate discovered pool candidates against trusted factories or pool methods.
   - Optionally backfill history with `eth_getLogs`.
   - Add reconnect/resubscribe behavior after transport failures.

## Test Plan

- Run `cargo check -p client-evm` after dependency and skeleton steps.
- Run `cargo test -p client-evm` after ABI, request-builder, parsing, and decoding steps.
- Keep all tests local and deterministic; no live RPC endpoint is required.
- Final verification: `cargo test -p client-evm` and `cargo check --workspace`.

## Assumptions

- The public API is blocking and does not expose async or Tokio.
- Alloy is used only for primitives, `sol!`, topic hashes, and ABI decoding.
- Transport is manual Ethereum JSON-RPC over blocking `tungstenite`.
- The first live subscription discovers candidate pools from log addresses and does not validate them.
- No historical backfill, pool validation, reconnect loop, HTTP polling fallback, or `aa-framework` integration in this implementation pass.
