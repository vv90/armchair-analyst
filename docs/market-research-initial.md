# Crypto Trading Software Market Research - Initial Synthesis

Date: 2026-05-28

This report summarizes the initial market-research pass for crypto trading software, based on public web sources and sub-agent research. It is product-oriented: it maps user pain points and competitor gaps back to Armchair Analyst's current concept as a local desktop market monitor.

## Bottom Line

Armchair Analyst should not start as "another trading bot" or generic charting terminal. The strongest wedge appears to be a local, read-only, always-on crypto market monitor that makes fragmented CEX/DEX/chain data trustworthy, auditable, and alertable.

## Main Trader Pain Points

1. Fragmented workflows: traders use many exchanges, wallets, DEX tools, portfolio trackers, and charting apps. Existing tools cover slices, not the whole operating picture.
2. Wrong balances/P&L: portfolio tools advertise broad sync, but users repeatedly hit missing transaction history, bad transfer matching, stale prices, and DeFi misclassification.
3. Alert unreliability: TradingView has powerful alerts, but users report webhook delay/missed-alert problems and want scanner-level alerts that reduce screen watching.
4. API-key trust: 3Commas' confirmed 2022 API-key disclosure remains the clearest market proof that cloud-stored trading keys are a serious objection.
5. DEX risk: DEX Screener, GeckoTerminal, and Birdeye are strong for discovery, but traders still need scam, liquidity, honeypot, wash-trade, holder, and creator-wallet context.
6. Backtest/live mismatch: bot tools are popular, but users complain that backtests ignore fees, slippage, liquidity, webhook delivery, partial fills, and exchange rejection.
7. UX/support/pricing friction: 3Commas, Cryptohopper, Coinrule, and Bitsgap reviews show that onboarding, plan limits, support, and hidden assumptions matter as much as features.

## Competitor Coverage

| Category | Covered Well | Main Gaps |
|---|---|---|
| TradingView | Best-in-class charts, Pine, alerts, screeners, desktop app | Weak crypto-specific passive monitoring; recurring screener-alert complaints |
| Altrady/Coinigy/Atani/GoodCrypto | Multi-exchange terminals, smart orders, portfolio views, scanners | Execution/bot complexity; less local-first/privacy differentiation |
| 3Commas/Bitsgap/Cryptohopper/Coinrule/Pionex | Grid/DCA bots, templates, TradingView webhooks, simple setup | Security trust, backtest realism, failed-trade observability |
| CoinStats/Delta/Koinly/CoinTracker | Portfolio aggregation, tax/P&L, many integrations | Sync accuracy, DeFi decoding, manual reconciliation burden |
| DEX Screener/DEXTools/Birdeye/GeckoTerminal | DEX discovery, pools, alerts, wallet/token views | Scam/risk provenance, fake liquidity, latency transparency |
| Nansen/Arkham/Dune/Glassnode/CryptoQuant | Wallet labels, on-chain intelligence, macro metrics | Expensive/complex; not a lightweight local trader monitor |
| Bookmap/Exocharts | Order-flow and liquidity depth | Specialist tools, not broad background monitoring |

## Recommended Feature Set

### V1

- Read-only CEX, wallet, DEX pool, and chain monitoring.
- A unified watchlist across CEX pairs, DEX pools, wallets, and tokens.
- Alert engine for price, volume, liquidity, wallet activity, exchange/API health, and stale data.
- Data confidence indicators: source, last sync, latency, missing history, calculated vs reported values.
- Local encrypted credential storage, read-only API defaults, permission checks, and key-rotation reminders.
- DEX risk panel: pool age, liquidity lock/burn, holder concentration, creator wallet history, sellability/tax flags, label provenance.
- Reconciliation-lite: unmatched transfers, duplicate transactions, missing prices, manual overrides, notes.

### V1.5

- Transaction-derived P&L with audit trail.
- DeFi position decoding for selected high-value protocols.
- Webhook/alert observability: fired, received, delayed, dropped, executed, rejected.
- Exchange status overlays and volatility/outage risk alerts.

### Later

- Full execution, automation, tax-grade reports, advanced backtesting, and copy/social features.

## Key Sources

Official/product sources:

- TradingView features: https://www.tradingview.com/features/
- Altrady features: https://www.altrady.com/features
- Bitsgap trading terminal: https://bitsgap.com/en-AU/trading-terminal
- 3Commas SmartTrade: https://3commas.io/smart-trade
- CoinStats overview: https://help.coinstats.app/en/articles/1537032-what-is-coinstats-app
- Koinly portfolio tracker: https://koinly.io/crypto-portfolio-tracker/
- CoinTracker portfolio tracker: https://www.cointracker.io/portfolio-tracker
- GeckoTerminal about: https://www.geckoterminal.com/about-us
- Birdeye alerts: https://learn.birdeye.so/docs/how-to-set-alerts
- DEXTools overview: https://info.dextools.io/
- Glassnode Studio: https://glassnode.com/products/studio

Sentiment, security, and community sources:

- 3Commas Trustpilot: https://www.trustpilot.com/review/3commas.io
- Cryptohopper Trustpilot: https://www.trustpilot.com/review/cryptohopper.com
- Bitsgap Trustpilot: https://www.trustpilot.com/review/bitsgap.com
- Coinrule Trustpilot: https://www.trustpilot.com/review/coinrule.com
- 3Commas API incident notice: https://3commas.io/blog/notice-on-api-data-disclosure-incident
- BleepingComputer on 3Commas API key leak: https://www.bleepingcomputer.com/news/security/crypto-platform-3commas-admits-hackers-stole-api-keys/
- TradingView webhook-delay discussion: https://www.reddit.com/r/TradingView/comments/1dap1qu/ongoing_issues_with_alert_webhook_delays_on/
- CoinTracker/Koinly cost-basis discussion: https://www.reddit.com/r/CryptoTax/comments/1rdz3dx/cointracker_cost_basis_incorrect_for_old_coinbase/
- Solana DEX rug/scam discussion: https://www.reddit.com/r/solana/comments/1blp455/solana_scammers_rugging_around_50_mil_a_day_by_creating_meme_coins/

