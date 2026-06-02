# Crypto Trading Software User Needs Validation

Date: 2026-05-28

This second pass intentionally ignores Armchair Analyst's current implementation, architecture, and product direction. The goal is to reduce confirmation bias, sunk-cost bias, and solution-first thinking by looking only at public user evidence and real-world incidents.

## Method

- Evidence types used: user reviews, Reddit/community discussions, public incident/security reports, official status/security disclosures, and public market-risk data.
- Vendor feature claims were treated only as context, not proof of user need.
- A need is marked strong only when it appears across multiple independent source types or is tied to direct money-at-risk incidents.
- Product-specific preferences such as "desktop", "local-first", or "background agent" were not counted as validated unless users explicitly asked for them.

## Refined Pain Points

| Rank | User Need | Evidence Strength | What People Are Really Asking For |
|---|---:|---|---|
| 1 | Correct portfolio, balances, and P&L across exchanges, wallets, and DeFi | Strong | Accurate imports, transfer matching, cost basis, current holdings, manual fixes, and clear explanations when numbers are wrong |
| 2 | Reliable alerts and event monitoring | Strong | Alerts that fire on time, scanner/watchlist alerts, webhook audit logs, missed-alert visibility, and fewer false positives |
| 3 | API-key and account security | Strong | Read-only defaults, permission clarity, key storage trust, IP allowlisting, revocation/rotation guidance, and minimal third-party blast radius |
| 4 | DEX/new-token risk detection | Strong | Scam/rug/honeypot warnings, liquidity quality, pool age, holder concentration, creator/dev-wallet behavior, and sellability checks |
| 5 | Backtest/live and automation observability | Medium-strong | Realistic assumptions, fees/slippage/liquidity modeling, paper/live parity, and explanations for missed or rejected trades |
| 6 | Exchange/API reliability during volatility | Medium-strong | Status awareness, stale-data warnings, API health, and notification when a venue may block trading or withdrawals |
| 7 | Usability, support, and pricing transparency | Medium | Simple setup, less jargon, predictable subscription limits, better support, and fewer hidden plan restrictions |
| 8 | Tool consolidation | Medium | Fewer disconnected dashboards, but users do not consistently ask for one monolithic replacement |
| 9 | AI/signal recommendations | Weak or negative | Users are skeptical of opaque signals; explainability and verifiable track records matter more than "AI" branding |

## Evidence Notes

### 1. Portfolio, Balance, And P&L Accuracy

This is the strongest validated need. Users repeatedly complain that portfolio/tax tools lose trust when imports miss transactions, balances diverge from exchange-reported values, or P&L/cost basis cannot be explained.

Evidence:

- CoinStats support and app/user discussions show recurring sync, missing-transaction, and inaccurate-balance complaints: https://help.coinstats.app/en/articles/1540984-what-can-i-do-with-the-coinstats-app and https://www.reddit.com/r/CoinStats/
- Delta's own support acknowledges that total portfolio balance may not match after adding an exchange connection and requires manual troubleshooting: https://support.delta.app/troubleshooting/exchange-account-connections/total-portfolio-balance-does-not-match-after-adding-exchange-account-connection
- CoinTracker support explicitly documents troubleshooting inaccurate balances, missing transactions, incorrect cost basis, and stale imports: https://support.cointracker.io/hc/en-us/articles/4413071375505-Troubleshoot-portfolio-tracking-issues
- Koinly and CoinTracker users in tax communities repeatedly report missing Coinbase/Coinbase Pro history, incorrect cost basis, and manual review burdens: https://www.reddit.com/r/CryptoTax/ and https://www.reddit.com/r/Cointracker/
- CoinTracking's own documentation says incorrect balances often come from missing or incorrectly imported transactions: https://cointracking.info/missing_transactions.php

Refinement:

- The real need is not "portfolio dashboard"; it is "numbers I can trust and repair."
- A feature is incomplete unless users can inspect sources, import history, sync time, missing data, and manual corrections.

### 2. Reliable Alerts And Event Monitoring

Users care less about having many alert types and more about whether alerts fire on time, can be audited, and can be applied to scanners/watchlists instead of one chart at a time.

Evidence:

- TradingView users complain about webhook delays and missed alerts, including financially meaningful delays: https://www.reddit.com/r/TradingView/comments/1dap1qu/ongoing_issues_with_alert_webhook_delays_on/
- TradingView users repeatedly ask for screener alerts and complain that scanner-based alerting is missing or removed: https://www.reddit.com/r/TradingView/comments/1r100q0/screener_alerts_we_were_told_they_would_be_coming/ and https://www.reddit.com/r/TradingView/comments/1rxxr7t/alerts_on_screener/
- TradingView webhook docs warn users about webhook payload shape and credential safety, which reinforces that alert delivery is part of an automation chain, not just a notification: https://www.tradingview.com/support/solutions/43000529348-about-webhooks/
- Cryptohopper and other bot communities show users struggling to identify whether a failed trade came from the alert, webhook bridge, bot, strategy, or exchange: https://www.reddit.com/r/CryptoHopper/

Refinement:

- The need is "observability of alerts", not simply "more notification channels."
- Desired features include event logs, latency timestamps, duplicate detection, fired-vs-delivered-vs-actioned states, retries, and failure explanations.

### 3. API-Key And Account Security

This is validated by both user anxiety and real incidents.

Evidence:

- 3Commas confirmed an API data disclosure incident in December 2022: https://3commas.io/blog/notice-on-api-data-disclosure-incident
- BleepingComputer reported that 3Commas admitted hackers stole API keys after earlier user reports: https://www.bleepingcomputer.com/news/security/crypto-platform-3commas-admits-hackers-stole-api-keys/
- Halborn's incident write-up explains that attackers could abuse trade permissions even without withdrawal permissions: https://www.halborn.com/blog/post/explained-the-3commas-breach-december-2022
- Bitsgap, Cryptohopper, and Pionex security docs all emphasize no-withdrawal permissions, encrypted storage, and IP allowlisting, implying these are baseline buyer concerns: https://bitsgap.com/security, https://support.cryptohopper.com/en/articles/9140271-is-cryptohopper-safe-to-use, and https://www.pionex.com/docs/api-docs

Refinement:

- "Read-only" matters, but it is not enough. Users also need to understand trade permissions, IP allowlists, key age, revocation, and what an attacker could do with each integration.
- Any feature requiring trading permissions has a much higher trust burden than monitoring-only features.

### 4. DEX/New-Token Risk Detection

This is strongly validated by user behavior and market data. Users want fast token discovery, but the risk environment is hostile.

Evidence:

- DEX Screener lists tokens automatically after liquidity and a transaction; its docs state the process is automatic and chain-indexed, which helps discovery but does not guarantee safety: https://docs.dexscreener.com/token-listing and https://docs.dexscreener.com/
- DEXTools markets risk indicators such as DEXTScore, holder distribution, liquidity status, contract checks, and honeypot guidance: https://info.dextools.io/crypto-glossary/dextscore/ and https://www.dextools.io/tutorials/how-to-check-if-token-is-honeypot-with-dextools-2026
- Chainalysis found 90,408 Ethereum tokens launched in 2023 matched suspicious pump-and-dump style criteria and estimated $241.6M in profit for actors behind those tokens: https://www.chainalysis.com/blog/crypto-crime-2024-pump-and-dump/
- Academic honeypot research confirms that DEXs allow unaudited tokens that can let users buy but block selling: https://arxiv.org/abs/2309.13501
- Solana and memecoin communities repeatedly discuss fake liquidity, scam tokens, and unsafe wallet/trading-terminal usage: https://www.reddit.com/r/solana/

Refinement:

- The need is not "DEX data"; it is "can I trust this pool/token enough to act?"
- Strong desired features include pool age, executable liquidity, holder concentration, LP lock/burn status, creator/dev-wallet behavior, sell simulation/tax flags, scam reports, and source confidence.

### 5. Automation, Backtesting, And Live Mismatch

Users like bot setup convenience, but user and vendor evidence show persistent realism gaps.

Evidence:

- Cryptohopper's own backtester documentation lists major limitations: one-month max period, no trigger support, no TradingView-alert backtesting, once-per-minute sell checks, and weak liquidity realism on illiquid pairs: https://support.cryptohopper.com/en/articles/10709691-bot-backtester-troubleshooting-limitations-common-issues
- 3Commas documents DCA bot backtesting limits and subscription-based quotas: https://help.3commas.io/en/articles/4829733-dca-bots-backtesting
- Freqtrade issue discussions and docs show that live/dry-run/backtest divergence can come from trailing stops, pricing, slippage, and order-fill dynamics: https://github.com/freqtrade/freqtrade/issues/10294 and https://docs.freqtrade.io/en/stable/backtesting/
- Pionex and Cryptohopper users discuss backtests not matching forward performance and paper/live differences: https://www.reddit.com/r/Pionex/ and https://www.reddit.com/r/CryptoHopper/

Refinement:

- "Build bots" is not the validated need. The validated need is "show me why strategy results differ from reality."
- Desired features include realistic fee/spread/slippage/liquidity models, partial-fill assumptions, exchange rejection logs, market-regime splits, and paper/live comparison.

### 6. Exchange/API Reliability During Volatility

This is validated, but the user need is operational visibility rather than another trading interface.

Evidence:

- Coinbase has had major outage reports during volatile crypto periods, including users seeing zero balances or being unable to access accounts: https://www.cnbc.com/2024/02/28/coinbase-users-report-accounts-show-zero-balance-amid-bitcoin-rally.html
- Exchange status pages and public incident histories show that API, login, order, and withdrawal disruptions are normal operating risks, not rare edge cases: https://status.coinbase.com/
- Traders in exchange communities complain that volatility-period downtime can prevent position management: https://www.reddit.com/r/CryptoExchange/

Refinement:

- Users need stale-data warnings, exchange/API health, status overlays, and position-risk alerts when venues degrade.
- This need is stronger for leveraged/futures traders than spot-only holders.

### 7. Usability, Support, And Pricing Transparency

The evidence is broad but less technically specific.

Evidence:

- 3Commas, Cryptohopper, Bitsgap, Coinrule, TradingView, and portfolio-tool Trustpilot pages contain recurring complaints about billing, plan limits, support, confusing setup, and feature gates: https://www.trustpilot.com/review/3commas.io, https://www.trustpilot.com/review/cryptohopper.com, https://www.trustpilot.com/review/bitsgap.com, https://www.trustpilot.com/review/coinrule.com, and https://www.trustpilot.com/review/tradingview.com
- Bot and trading-terminal reviews praise quick setup, templates, and support when they work, which implies that ease of setup is a purchase driver.

Refinement:

- Simpler UX is a real need, but "simple" means fewer unexplained assumptions, visible limits, clear setup state, and fast support paths.
- Overloaded pro dashboards are not always bad; advanced users tolerate density when it saves time.

## Desired Feature List, Evidence-First

| Feature Need | Evidence Strength | Notes |
|---|---:|---|
| Data provenance and confidence flags | Strong | Needed because users distrust balances, P&L, imports, and prices |
| Manual correction/reconciliation tools | Strong | Missing transactions and bad classifications are unavoidable in real use |
| Scanner/watchlist/event alerts | Strong | Users explicitly request scanner alerts and reliable notification behavior |
| Alert/webhook audit log | Strong | Needed to diagnose missed trades and alert-chain failures |
| API permission and key-risk dashboard | Strong | Validated by API-key incidents and repeated security positioning |
| DEX token/pool risk panel | Strong | Validated by scams, Chainalysis data, DEXTools behavior, and user communities |
| Exchange/API health and stale-data warnings | Medium-strong | Important during volatility; more important to leveraged traders |
| Realistic backtesting/paper-vs-live comparison | Medium-strong | Strong for bot users; less relevant for non-automation users |
| Tax-grade reporting | Medium | Strong for tax users, but not clearly the same segment as active traders |
| Full trade execution | Medium/contested | Desired by active terminal users but raises security/support burden |
| Mobile-first experience | Medium | Often valued in reviews; evidence is stronger than demand for local desktop |
| Local desktop app | Weak | Some traders use desktop/multi-monitor workflows, but user evidence rarely asks for local-only software directly |
| AI-generated trade signals | Weak/negative | Users show skepticism; explainability and verified results matter more |

## Bias-Control Findings

- The first report's "local desktop/background monitor" recommendation is plausible but not strongly proven by direct user demand. Users more often ask for reliability, accuracy, mobile access, scanner alerts, and tool consolidation than for local Windows software.
- "Privacy/local-first" is a defensible differentiator because of API-key incidents, but the user-stated need is safer credentials and less third-party risk, not necessarily a fully local architecture.
- "CEX+DEX+chain aggregation" is supported, but only if it improves accuracy and risk visibility. Broad integration count alone is not valued when balances and positions are wrong.
- "Signals" should be treated carefully. Users show distrust of opaque signal providers; explainable alerts and verifiable evidence are safer than predictive recommendations.
- "Automation" is not the safest v1 assumption. The real unmet need around automation is observability, realism, and debugging.
- "Tax-grade reporting" is a separate market. It is a real need, but it may pull the product into accounting workflows rather than active trading workflows.

## Refined User-Need Thesis

The strongest evidence-supported need is not a specific app shape. It is trustworthy crypto situational awareness:

- What do I own?
- Where is it?
- Is the data fresh?
- Why do these numbers not match?
- What changed that I should care about?
- Can I trust this token, pool, wallet, alert, exchange, or bot result enough to act?
- If something failed, where did it fail?

Any product direction should be evaluated against those questions before committing to platform, architecture, or advanced trading features.

