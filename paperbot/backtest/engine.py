"""Event-driven backtester.

Processes bars strictly in time order with no vectorized shortcuts that
could leak future data:

  close of bar t-1: strategy sees rows <= t-1, emits orders
  open  of bar t:   orders fill at bar t's open (with slippage + fee)
  during bar t:     stops / take-profits checked against bar t's high/low
  close of bar t:   mark to market, kill-switch check, next signals

The pessimistic conventions (stop-before-TP, gap-through-stop fills at the
open) live in paperbot.execution and are shared with the paper engine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from paperbot.config import CostConfig, RiskConfig
from paperbot.execution import (
    Account,
    PendingOrder,
    Trade,
    stop_exit_reference,
    take_profit_exit_reference,
)
from paperbot.strategies.base import Strategy

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Everything needed to judge a run honestly."""

    equity_curve: pd.Series  # equity at each bar close, UTC index
    trades: list[Trade]
    starting_balance: float
    halted: bool
    params: dict[str, Any] = field(default_factory=dict)
    symbols: list[str] = field(default_factory=list)


def run_backtest(
    strategy: Strategy,
    data: dict[str, pd.DataFrame],
    starting_balance: float,
    risk: RiskConfig,
    costs: CostConfig,
    trade_from: pd.Timestamp | None = None,
    liquidate_at_end: bool = False,
) -> BacktestResult:
    """Run `strategy` over `data` (symbol -> raw OHLCV DataFrame).

    DataFrames must have a UTC DatetimeIndex and columns
    ts/open/high/low/close/volume (as produced by CandleStore.load).

    trade_from: bars before this timestamp are used only to warm up
        indicators — no signals fire and they are excluded from the
        reported equity curve. Used by walk-forward test windows.
    liquidate_at_end: close any open position at the final bar's close
        (with full costs), so consecutive walk-forward windows can be
        chained without positions leaking across the boundary.
    """
    prepared: dict[str, list[Any]] = {}
    index = None
    for symbol, df in data.items():
        pdf = strategy.prepare(df)
        if index is None:
            index = pdf.index
        else:
            index = index.union(pdf.index)
        prepared[symbol] = pdf

    if index is None or len(index) == 0:
        raise ValueError("no data supplied to backtest")

    # Align every symbol to the shared clock; missing bars become NaN rows
    # that we skip. itertuples is ~10x faster than iterrows.
    rows: dict[str, list[Any]] = {}
    for symbol, pdf in prepared.items():
        aligned = pdf.reindex(index)
        rows[symbol] = list(aligned.itertuples(index=False))
    ts_ms = [int(t.value // 1_000_000) for t in index]

    account = Account(cash=starting_balance)
    pending: list[PendingOrder] = []
    equity_points: list[float] = []
    flatten_all = False

    def bar_ok(row: Any) -> bool:
        return row is not None and not pd.isna(row.open)

    for i in range(len(index)):
        ts = ts_ms[i]
        bars = {s: rows[s][i] for s in rows}
        marks = {
            s: (b.close if bar_ok(b) else None) for s, b in bars.items()
        }
        # Carry last known mark for symbols missing this bar.
        for s in list(marks):
            if marks[s] is None:
                prev = next(
                    (rows[s][j].close for j in range(i - 1, -1, -1) if bar_ok(rows[s][j])),
                    None,
                )
                if prev is None:
                    del marks[s]
                else:
                    marks[s] = prev

        # ---- 1. Open of bar: fill pending orders (exits before entries). ----
        if flatten_all:
            for pos in list(account.positions.values()):
                b = bars.get(pos.symbol)
                if bar_ok(b):
                    account.close_position(pos, ts, b.open, "kill switch", costs)
            pending = []
            if not account.positions:
                flatten_all = False
        else:
            for order in sorted(pending, key=lambda o: o.side != "exit"):
                b = bars.get(order.symbol)
                if not bar_ok(b):
                    continue  # no bar this hour; order stays pending
                if order.side == "exit":
                    pos = account.positions.get(Account.key(order.symbol, order.strategy))
                    if pos is not None:
                        account.close_position(pos, ts, b.open, order.reason, costs)
                    pending = [o for o in pending if o is not order]
                elif order.side == "enter" and order.intent is not None:
                    account.open_long(
                        order.symbol, order.strategy, ts, b.open, order.intent, marks, risk, costs
                    )
                    pending = [o for o in pending if o is not order]

        # ---- 2. Intrabar: stops first (pessimistic), then take-profits. ----
        for pos in list(account.positions.values()):
            b = bars.get(pos.symbol)
            if not bar_ok(b):
                continue
            if b.low <= pos.stop_price:
                ref = stop_exit_reference(b.open, pos.stop_price)
                account.close_position(pos, ts, ref, "stop hit", costs)
            elif pos.take_profit is not None and b.high >= pos.take_profit:
                ref = take_profit_exit_reference(b.open, pos.take_profit)
                account.close_position(pos, ts, ref, "take profit", costs)

        # ---- 3. Close of bar: mark, kill switch, then next signals. ----
        for pos in account.positions.values():
            b = bars.get(pos.symbol)
            if bar_ok(b):
                pos.highest_close = max(pos.highest_close, b.close)

        equity_points.append(account.equity(marks))

        if account.check_kill_switch(marks, risk):
            flatten_all = True
            pending = []
            continue
        if account.halted:
            continue
        if trade_from is not None and index[i] < trade_from:
            continue

        for symbol, b in bars.items():
            if not bar_ok(b):
                continue
            k = Account.key(symbol, strategy.name)
            pos = account.positions.get(k)
            if pos is None:
                if any(o.symbol == symbol and o.side == "enter" for o in pending):
                    continue
                intent = strategy.should_enter(b)
                if intent is not None:
                    pending.append(
                        PendingOrder(symbol, strategy.name, "enter", intent, intent.reason)
                    )
            else:
                action = strategy.manage(b, pos)
                if action is None:
                    continue
                if action.exit_market:
                    if not any(
                        o.symbol == symbol and o.side == "exit" and o.strategy == strategy.name
                        for o in pending
                    ):
                        pending.append(
                            PendingOrder(symbol, strategy.name, "exit", None, action.reason)
                        )
                elif action.new_stop is not None and action.new_stop > pos.stop_price:
                    pos.stop_price = action.new_stop

    if liquidate_at_end and account.positions:
        last_ts = ts_ms[-1]
        for pos in list(account.positions.values()):
            ref = next(
                (rows[pos.symbol][j].close for j in range(len(index) - 1, -1, -1)
                 if bar_ok(rows[pos.symbol][j])),
                pos.entry_price,
            )
            account.close_position(pos, last_ts, ref, "end of window", costs)
        equity_points[-1] = account.equity({})

    curve = pd.Series(equity_points, index=index, name="equity")
    if trade_from is not None:
        curve = curve[curve.index >= trade_from]
    return BacktestResult(
        equity_curve=curve,
        trades=list(account.trades),
        starting_balance=starting_balance,
        halted=account.halted,
        params=dict(strategy.params),
        symbols=list(data.keys()),
    )
