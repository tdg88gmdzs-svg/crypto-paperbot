# crypto-paperbot

A **paper-trading** cryptocurrency bot. Strictly a simulation: there is no
code path anywhere in this project that can place a real order, hold an API
key, or move funds. It reads public market data, trades imaginary money, and
produces deliberately pessimistic performance reports.

Trades BTC/USDT and ETH/USDT, long-only (simulated spot, no margin).

## Current honest verdict (as of July 2026 data)

Both bundled strategies **lose to buy-and-hold out-of-sample after costs**
over 2022–2026:

| strategy    | stitched OOS return | buy & hold same period | verdict |
|-------------|--------------------:|-----------------------:|---------|
| trendfollow | −53.9%              | −10.5%                 | loses   |
| meanrevert  | −93.5%              | −10.5%                 | loses badly |

At 0.26% taker fees per side, hourly-bar strategies with frequent trades pay
more in costs than their edge earns. The bot reports this plainly instead of
hiding it — that is the point of the project. See "Why paper results
overstate live results" below before trusting *any* positive number.

## Setup

Requires Python 3.11+. With [uv](https://docs.astral.sh/uv/):

```bash
cd crypto-paperbot
uv venv --python 3.12
uv pip install -e ".[dev]"
```

Or with plain pip: `python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]"`

## Commands

```bash
# Single full-period backtest with default params (in-sample, optimistic):
.venv/bin/bot backtest --strategy trendfollow --from 2021-01-01

# Walk-forward analysis (the number to believe):
.venv/bin/bot walkforward --strategy trendfollow --from 2021-01-01

# Live paper loop (runs unattended, survives restarts, Ctrl-C to stop):
.venv/bin/bot paper                     # all strategies
.venv/bin/bot paper --strategy trendfollow

# Current paper-account equity, drawdown, positions, recent trades:
.venv/bin/bot report

# Unit tests (fill logic and risk sizing — where bugs fake profits):
.venv/bin/python -m pytest
```

All tunables (symbols, fees, risk limits, strategy parameters, walk-forward
windows) live in [config.yaml](config.yaml).

## How it works

- **Data** ([paperbot/data.py](paperbot/data.py)) — public OHLCV via ccxt,
  cached in SQLite (`data/candles.sqlite`) so nothing is re-downloaded.
  Backfill comes from Binance because Kraken's public OHLC endpoint only
  returns the most recent ~720 candles per timeframe; the live paper loop
  polls Kraken. The two venues' prices differ by basis points; the trade
  log records which source each candle came from.
- **Strategies** ([paperbot/strategies/](paperbot/strategies/)) — pluggable
  `Strategy` subclasses. A strategy only ever says "enter long with this
  stop distance" or "exit"; it cannot size positions or touch cash.
- **Backtester** ([paperbot/backtest/engine.py](paperbot/backtest/engine.py))
  — event-driven, bar by bar, no vectorized lookahead. Signals fire on a
  bar's close and fill at the **next** bar's open. Every fill pays 0.26%
  taker fee plus 0.05% slippage against you. Stops are checked intrabar;
  a bar that gaps through a stop fills at the (worse) open; a bar that
  touches both stop and take-profit is assumed to hit the stop first.
- **Risk** ([paperbot/execution.py](paperbot/execution.py)) — one place
  enforces: ≤1% of equity risked per trade (sized off the stop distance),
  notional capped by cash (no leverage), max 2 concurrent positions, and a
  kill switch that flattens and halts at 15% drawdown from peak equity.
  The backtester and paper engine share this code, so paper fills obey
  exactly the rules that were backtested.
- **Walk-forward** ([paperbot/walkforward.py](paperbot/walkforward.py)) —
  optimizes on a rolling 12-month window, tests on the next 3 months,
  rolls forward 3 months. Out-of-sample windows chain their equity, so the
  stitched OOS curve compounds like a real account. In-sample numbers are
  reported only to show how much the optimizer flattered itself.
- **Paper engine** ([paperbot/paper/](paperbot/paper/)) — $10,000 virtual
  balance; positions, pending orders, trades, and the equity curve persist
  in SQLite (`data/paper_state.sqlite`). On restart it replays any candles
  it missed. Logs to console and `logs/paperbot.log`, with a plain-English
  daily summary. If the kill switch halts the account, it refuses to trade
  until you deliberately delete the state DB.

## Adding a strategy

1. Create `paperbot/strategies/mystrat.py` subclassing `Strategy`:
   implement `prepare()` (add indicator columns — causal only, row *i* may
   never read rows > *i*), `should_enter()`, `manage()`, and
   `default_params()`.
2. Register it in `STRATEGIES` in
   [paperbot/strategies/__init__.py](paperbot/strategies/__init__.py).
3. Add a section under `strategies:` in config.yaml, including a small
   `grid:` block if you want walk-forward to tune it. Keep the grid tiny —
   every extra dimension is another way to overfit.
4. Judge it **only** by `bot walkforward` output.

## Why paper results overstate live results

Treat every number this bot produces as an upper bound. Reasons, in rough
order of how much money they cost people:

- **Overfitting.** Any parameter you chose because it backtested well is
  fitted to the past. Walk-forward analysis reduces this but does not
  eliminate it: the strategy *structure* (EMA crossovers, Donchian windows,
  RSI thresholds) was itself selected from ideas that are popular because
  they used to work. The market you deploy into is not the market you
  fitted.
- **Slippage variance.** The simulator charges a flat 0.05% per fill.
  Real slippage is not flat: it explodes exactly when your orders are most
  urgent — stop-outs in fast markets, breakouts everyone else is also
  buying. Losing fills get worse slippage than winning fills, an asymmetry
  a flat model cannot capture.
- **Fees are modeled, spreads only partly.** 0.26% taker per side is
  Kraken's worst tier (pessimistic, good), but the bid-ask spread itself
  widens in volatile hours, and OHLC candles hide it entirely.
- **Latency and partial fills.** Paper fills are instant and complete at
  the bar open. A real order arrives tens to hundreds of milliseconds
  later, may fill partially, or may miss a fast move entirely. Candle data
  also arrives late: the "next bar open" the backtest trades at is not the
  price you would actually get at that wall-clock moment.
- **Survivorship in the data.** BTC and ETH are the pairs everyone tests
  because they survived and went up. A strategy validated only on
  survivors inherits their luck.
- **Regime dependence.** 2021 trends, the 2022 crash, and 2024–2026 chop
  each reward different behavior. A walk-forward that spans them is more
  honest than a single backtest, but nothing guarantees the next regime
  resembles any window you tested.
- **Paper discipline is free.** The simulator never widens a stop, never
  doubles down, never hesitates. Live, the operator is part of the system.

If a strategy doesn't clearly beat buy-and-hold out-of-sample *after* these
haircuts — and as of the data above, neither bundled strategy does — the
correct trade is no trade.

## Ground rules baked into the code

- No real orders, no API keys, no withdrawal logic — these are absent by
  construction, not disabled by a flag someone could flip.
- Pessimistic tie-breaks everywhere a bar is ambiguous.
- Realism beats better-looking numbers: costs are worst-tier, fills are
  next-bar, and the report says "DID NOT beat buy-and-hold" in so many
  words when that is what happened.
