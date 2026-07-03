"""Shared execution and risk logic.

Both the backtester and the live paper engine use exactly this code, so a
fill in a backtest and a fill in paper trading obey the same (pessimistic)
rules:

- All fills are taker fills: fee applied on entry AND exit notional.
- Slippage always moves against you: buys fill above the reference price,
  sells below it.
- Signals fire on a bar's close; fills happen at the NEXT bar's open.
- If a bar gaps through a stop, the fill is at the (worse) open, not the
  stop price.
- If a bar touches both the stop and the take-profit, the stop is assumed
  to have been hit first (pessimistic).

Risk rules (the "hard rules, enforced in one place"):
- Position size risks at most `risk_per_trade` of current equity, computed
  off the stop distance.
- No leverage: notional is capped by available cash.
- At most `max_positions` concurrent positions.
- Kill switch: if equity drops `kill_switch_drawdown` from its peak, the
  account is flattened and halted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from paperbot.config import CostConfig, RiskConfig
from paperbot.strategies.base import EntryIntent

log = logging.getLogger(__name__)


@dataclass
class Position:
    """An open long position."""

    symbol: str
    strategy: str
    qty: float
    entry_price: float  # actual fill price incl. slippage
    entry_ts: int  # epoch ms of the fill bar's open
    stop_price: float
    take_profit: float | None
    entry_fee: float  # USD paid on entry
    highest_close: float  # for trailing stops


@dataclass
class Trade:
    """A completed round trip."""

    symbol: str
    strategy: str
    qty: float
    entry_ts: int
    entry_price: float
    exit_ts: int
    exit_price: float
    fees: float  # total USD fees, both sides
    pnl: float  # net USD P&L after fees and slippage
    exit_reason: str


@dataclass
class PendingOrder:
    """An instruction created at a bar close, to be filled at the next open."""

    symbol: str
    strategy: str
    side: str  # "enter" | "exit"
    intent: EntryIntent | None = None  # for entries
    reason: str = ""


def buy_fill_price(reference: float, slippage: float) -> float:
    """Price actually paid when buying at `reference` (slippage against you)."""
    return reference * (1.0 + slippage)


def sell_fill_price(reference: float, slippage: float) -> float:
    """Price actually received when selling at `reference`."""
    return reference * (1.0 - slippage)


def stop_exit_reference(bar_open: float, stop_price: float) -> float:
    """Reference price for a stop triggered within a bar.

    If the bar opened at/below the stop (gap through), you get the open;
    otherwise you get the stop level. Slippage is applied on top by the
    caller. Never better than the stop.
    """
    return min(bar_open, stop_price)


def take_profit_exit_reference(bar_open: float, tp_price: float) -> float:
    """Reference price for a take-profit triggered within a bar.

    A TP is a limit-style exit: you get at least the TP. If the bar gapped
    open above it, you get the (better) open.
    """
    return max(bar_open, tp_price)


def size_position(
    equity: float,
    cash: float,
    est_entry_price: float,
    stop_distance: float,
    risk: RiskConfig,
    costs: CostConfig,
) -> float:
    """Return quantity for a new long, or 0.0 if it can't be sized.

    Risk-based size: risking `risk_per_trade` of equity over the stop
    distance. Capped so entry notional + fee never exceeds available cash
    (spot account, no leverage).
    """
    if stop_distance <= 0 or est_entry_price <= 0 or equity <= 0:
        return 0.0
    risk_dollars = equity * risk.risk_per_trade
    qty = risk_dollars / stop_distance
    # Cash cap: qty * price * (1 + fee) <= cash
    max_affordable = cash / (est_entry_price * (1.0 + costs.taker_fee))
    qty = min(qty, max_affordable)
    return max(qty, 0.0)


@dataclass
class Account:
    """Cash + positions + realized trades. Shared by backtest and paper."""

    cash: float
    positions: dict[str, Position] = field(default_factory=dict)  # key: symbol|strategy
    trades: list[Trade] = field(default_factory=list)
    peak_equity: float = 0.0
    halted: bool = False

    def __post_init__(self) -> None:
        self.peak_equity = max(self.peak_equity, self.cash)

    @staticmethod
    def key(symbol: str, strategy: str) -> str:
        return f"{symbol}|{strategy}"

    def equity(self, marks: dict[str, float]) -> float:
        """Cash plus positions marked at `marks[symbol]`."""
        total = self.cash
        for pos in self.positions.values():
            mark = marks.get(pos.symbol, pos.entry_price)
            total += pos.qty * mark
        return total

    def open_long(
        self,
        symbol: str,
        strategy: str,
        ts: int,
        reference_price: float,
        intent: EntryIntent,
        marks: dict[str, float],
        risk: RiskConfig,
        costs: CostConfig,
    ) -> Position | None:
        """Fill an entry at `reference_price` (a bar open). Applies slippage,
        fee, risk sizing, and the max-positions rule. Returns the position
        or None if the entry was rejected/unaffordable."""
        if self.halted:
            return None
        if len(self.positions) >= risk.max_positions:
            log.debug("entry rejected (%s %s): max positions reached", symbol, strategy)
            return None
        k = self.key(symbol, strategy)
        if k in self.positions:
            return None
        fill = buy_fill_price(reference_price, costs.slippage)
        qty = size_position(
            equity=self.equity(marks),
            cash=self.cash,
            est_entry_price=fill,
            stop_distance=intent.stop_distance,
            risk=risk,
            costs=costs,
        )
        notional = qty * fill
        if notional < 10.0:  # dust guard: exchanges reject sub-$10 orders anyway
            return None
        fee = notional * costs.taker_fee
        self.cash -= notional + fee
        pos = Position(
            symbol=symbol,
            strategy=strategy,
            qty=qty,
            entry_price=fill,
            entry_ts=ts,
            stop_price=fill - intent.stop_distance,
            take_profit=(fill + intent.take_profit_distance)
            if intent.take_profit_distance
            else None,
            entry_fee=fee,
            highest_close=fill,
        )
        self.positions[k] = pos
        return pos

    def close_position(
        self,
        pos: Position,
        ts: int,
        reference_price: float,
        reason: str,
        costs: CostConfig,
    ) -> Trade:
        """Fill an exit at `reference_price`. Applies slippage and fee,
        realizes the trade, and frees the slot."""
        fill = sell_fill_price(reference_price, costs.slippage)
        proceeds = pos.qty * fill
        fee = proceeds * costs.taker_fee
        self.cash += proceeds - fee
        trade = Trade(
            symbol=pos.symbol,
            strategy=pos.strategy,
            qty=pos.qty,
            entry_ts=pos.entry_ts,
            entry_price=pos.entry_price,
            exit_ts=ts,
            exit_price=fill,
            fees=pos.entry_fee + fee,
            pnl=proceeds - fee - (pos.qty * pos.entry_price + pos.entry_fee),
            exit_reason=reason,
        )
        self.trades.append(trade)
        del self.positions[self.key(pos.symbol, pos.strategy)]
        return trade

    def check_kill_switch(self, marks: dict[str, float], risk: RiskConfig) -> bool:
        """Update peak equity; return True the moment drawdown from peak
        breaches the kill-switch threshold. Caller must then flatten."""
        eq = self.equity(marks)
        self.peak_equity = max(self.peak_equity, eq)
        if self.halted:
            return False
        if eq <= self.peak_equity * (1.0 - risk.kill_switch_drawdown):
            self.halted = True
            log.critical(
                "KILL SWITCH: equity %.2f is %.1f%% below peak %.2f — flattening and halting",
                eq,
                100.0 * (1.0 - eq / self.peak_equity),
                self.peak_equity,
            )
            return True
        return False
