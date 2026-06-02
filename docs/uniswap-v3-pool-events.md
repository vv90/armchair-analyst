# Uniswap V3 Pool Events

This document describes the canonical Uniswap v3 pool events used by `client-evm` subscriptions.

The first subscription implementation listens chain-wide for pool event topic hashes and discovers candidate pool addresses from `log.address`. It does not fetch historical logs and does not validate that discovered addresses are canonical Uniswap v3 pools yet.

Canonical event source: https://github.com/Uniswap/v3-core/blob/main/contracts/interfaces/pool/IUniswapV3PoolEvents.sol

## Events

| Event | When emitted | Pool state impact |
| --- | --- | --- |
| `Initialize` | The pool is initialized with its first price and tick. | Sets initial `sqrtPriceX96` and `tick`; marks the pool initialized. |
| `Mint` | Liquidity is added to a tick range. | Updates position liquidity, tick liquidity, token balances, and active liquidity if the range includes the current tick. |
| `Collect` | Accrued fees are collected from a position. | Reduces position tokens owed and changes pool token balances. |
| `Burn` | Liquidity is removed from a tick range. | Updates position liquidity, tick liquidity, active liquidity if in range, token balances, and tokens owed. |
| `Swap` | A swap executes against the pool. | Updates price, tick, liquidity when crossing ticks, fee growth, protocol fee accounting, observations, and token balances. |
| `Flash` | Tokens are flash-borrowed and repaid. | Changes token balances and fee accounting when fees are paid back to the pool. |
| `IncreaseObservationCardinalityNext` | The pool oracle observation capacity target is increased. | Updates oracle observation configuration used for future observations. |
| `SetFeeProtocol` | Protocol fee settings are changed. | Updates protocol fee configuration for token0 and token1. |
| `CollectProtocol` | Accrued protocol fees are collected. | Reduces protocol fees owed and changes pool token balances. |

All nine events can affect pool state, pool accounting, or future pool behavior, so the initial subscription includes all of them.

## Discovery

The initial subscription uses a topic-only `eth_subscribe` logs filter:

```json
{
  "topics": [["<all canonical Uniswap v3 pool event topic0 hashes>"]]
}
```

No pool address filter is included. Each matching notification carries an EVM log address, and `client-evm` treats that address as a discovered pool candidate.

This means discovery is broad by design. Any contract can emit logs with the same event signatures, so discovered addresses are observations, not proof that the address is a canonical Uniswap v3 pool.

## Future Validation

Future validation can narrow discovered candidates with one or more checks:

- Compare discovered addresses against `PoolCreated` events from trusted Uniswap v3 factory addresses.
- Call pool methods such as `factory()`, `token0()`, `token1()`, `fee()`, and `tickSpacing()` and verify they are coherent.
- Recompute the expected CREATE2 pool address from factory, token pair, fee tier, and init code hash.
- Compare pool runtime bytecode or bytecode hash against known canonical deployments.
- Maintain a per-network allowlist of trusted factory addresses, starting with Ethereum.
- Backfill historical factory or pool logs with `eth_getLogs` before starting live subscriptions.

Validation should remain outside the first live subscription increment.
