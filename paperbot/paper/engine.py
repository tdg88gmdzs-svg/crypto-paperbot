"""Paper trading engine: applies closed candles to the persistent account.

Uses the exact same execution rules as the backtester (see
paperbot.execution): orders created at a bar's close fill at the NEXT
bar's open, stops are checked intrabar and assumed to fire before
take-profits, all fills pay taker fee + slippage.

One deliberate difference: when the kill switch trips live, positions are
flattened immediately at the current close instead of the next open —
live you would market-out now, not in an hour.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from paperbot.config import Config
from paperbot.execution import (
    Account,
    PendingOrder,
    stop_exit_reference,
    take_profit_exit_reference,
)
from paperbot.paper.state import PaperState
from paperbot.strategies.base import Strategy

log = logging.getLogger(__name__)

# Bars of history handed to strategy.prepare() each evaluation — enough
# for EMA(200) plus margin.
PREPARE_WINDOW = 500


class PaperEngine:
    """Holds the live paper account and processes one closed bar at a time."""

    def __init__(self, cfg: Config, strategies: list[Strategy], state: PaperState) -> None:
        self.cfg = cfg
        self.strategies = strategies
        self.state = state
        self.account, self.pending = state.load_account(cfg.paper.starting_balance)
        self.last_marks: dict[str, float] = {}

    def process_bar(self, symbol: str, history: pd.DataFrame) -> None:
        """Apply the newest closed bar for `symbol`.

        `history` is the candle DataFrame up to and including that bar
        (CandleStore.load output). Executes pending orders at this bar's
        open, checks stops intrabar, then evaluates strategies at the close.
        """
        if history.empty:
            return
        bar = history.iloc[-1]
        ts = int(bar["ts"])
        self.last_marks[symbol] = float(bar["close"])
        marks = dict(self.last_marks)

        # 1. Fill pending orders for this symbol at the bar open.
        still_pending: list[PendingOrder] = []
        for order in sorted(self.pending, key=lambda o: o.side != "exit"):
            if order.symbol != symbol:
                still_pending.append(order)
                continue
            if order.side == "exit":
                pos = self.account.positions.get(Account.key(order.symbol, order.strategy))
                if pos is not None:
                    t = self.account.close_position(pos, ts, float(bar["open"]), order.reason, self.cfg.costs)
                    self.state.append_trade(t)
                    log.info("EXIT  %s %s qty=%.6f @ %.2f pnl=%+.2f (%s)",
                             t.symbol, t.strategy, t.qty, t.exit_price, t.pnl, t.exit_reason)
            elif order.intent is not None and not self.account.halted:
                pos = self.account.open_long(
                    order.symbol, order.strategy, ts, float(bar["open"]),
                    order.intent, marks, self.cfg.risk, self.cfg.costs,
                )
                if pos is not None:
                    log.info("ENTER %s %s qty=%.6f @ %.2f stop=%.2f (%s)",
                             pos.symbol, pos.strategy, pos.qty, pos.entry_price,
                             pos.stop_price, order.reason)
        self.pending = still_pending

        # 2. Intrabar stop / take-profit checks for this symbol.
        for pos in list(self.account.positions.values()):
            if pos.symbol != symbol:
                continue
            if float(bar["low"]) <= pos.stop_price:
                ref = stop_exit_reference(float(bar["open"]), pos.stop_price)
                t = self.account.close_position(pos, ts, ref, "stop hit", self.cfg.costs)
                self.state.append_trade(t)
                log.info("STOP  %s %s @ %.2f pnl=%+.2f", t.symbol, t.strategy, t.exit_price, t.pnl)
            elif pos.take_profit is not None and float(bar["high"]) >= pos.take_profit:
                ref = take_profit_exit_reference(float(bar["open"]), pos.take_profit)
                t = self.account.close_position(pos, ts, ref, "take profit", self.cfg.costs)
                self.state.append_trade(t)
                log.info("TP    %s %s @ %.2f pnl=%+.2f", t.symbol, t.strategy, t.exit_price, t.pnl)

        # 3. Bar close: trailing anchors, kill switch, new signals.
        for pos in self.account.positions.values():
            if pos.symbol == symbol:
                pos.highest_close = max(pos.highest_close, float(bar["close"]))

        if self.account.check_kill_switch(marks, self.cfg.risk):
            # Live: flatten NOW at current marks rather than next open.
            for pos in list(self.account.positions.values()):
                mark = marks.get(pos.symbol, pos.entry_price)
                t = self.account.close_position(pos, ts, mark, "kill switch", self.cfg.costs)
                self.state.append_trade(t)
                log.critical("KILL SWITCH FLATTEN %s %s @ %.2f pnl=%+.2f",
                             t.symbol, t.strategy, t.exit_price, t.pnl)
            self.pending = []

        if not self.account.halted:
            window = history.tail(PREPARE_WINDOW)
            for strat in self.strategies:
                prepared = strat.prepare(window)
                row = next(prepared.tail(1).itertuples(index=False))
                key = Account.key(symbol, strat.name)
                pos = self.account.positions.get(key)
                if pos is None:
                    if any(o.symbol == symbol and o.strategy == strat.name and o.side == "enter"
                           for o in self.pending):
                        continue
                    intent = strat.should_enter(row)
                    if intent is not None:
                        self.pending.append(
                            PendingOrder(symbol, strat.name, "enter", intent, intent.reason))
                        log.info("SIGNAL enter %s %s (%s)", symbol, strat.name, intent.reason)
                else:
                    action = strat.manage(row, pos)
                    if action is not None:
                        if action.exit_market and not any(
                            o.symbol == symbol and o.strategy == strat.name and o.side == "exit"
                            for o in self.pending
                        ):
                            self.pending.append(
                                PendingOrder(symbol, strat.name, "exit", None, action.reason))
                            log.info("SIGNAL exit %s %s (%s)", symbol, strat.name, action.reason)
                        elif action.new_stop is not None and action.new_stop > pos.stop_price:
                            pos.stop_price = action.new_stop

        equity = self.account.equity(marks)
        self.state.append_equity(ts, equity)
        self.state.save(self.account, self.pending)

    def daily_summary(self) -> str:
        """Plain-English snapshot for the daily log."""
        marks = dict(self.last_marks)
        equity = self.account.equity(marks)
        dd = 1.0 - equity / self.account.peak_equity if self.account.peak_equity else 0.0
        day_ago = (pd.Timestamp.utcnow() - pd.Timedelta(days=1)).value // 1_000_000
        recent = [t for t in self.account.trades if t.exit_ts >= day_ago]
        lines = [
            "──── daily summary ────",
            f"equity ${equity:,.2f} (peak ${self.account.peak_equity:,.2f}, "
            f"drawdown {100 * dd:.1f}%)",
            f"open positions: {len(self.account.positions)}",
        ]
        for pos in self.account.positions.values():
            mark = marks.get(pos.symbol, pos.entry_price)
            upnl = pos.qty * (mark - pos.entry_price)
            lines.append(
                f"  {pos.symbol} [{pos.strategy}] qty={pos.qty:.6f} "
                f"entry={pos.entry_price:.2f} stop={pos.stop_price:.2f} upnl={upnl:+.2f}"
            )
        lines.append(f"trades closed last 24h: {len(recent)} "
                     f"(net {sum(t.pnl for t in recent):+,.2f})")
        if self.account.halted:
            lines.append("ACCOUNT IS HALTED (kill switch). Manual reset required.")
        return "\n".join(lines)
