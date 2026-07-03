"""Performance metrics computed from an equity curve and trade list.

Sharpe is annualized from daily returns with sqrt(365) — crypto trades
every day. Buy-and-hold pays the same fees and slippage as the strategy,
so the comparison isn't tilted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from paperbot.config import CostConfig
from paperbot.execution import Trade, buy_fill_price, sell_fill_price


@dataclass(frozen=True)
class Stats:
    start: str
    end: str
    years: float
    final_equity: float
    total_return_pct: float
    cagr_pct: float
    max_drawdown_pct: float
    sharpe: float
    win_rate_pct: float
    profit_factor: float
    num_trades: int
    total_fees: float
    buy_hold_return_pct: float
    halted: bool

    def beats_buy_and_hold(self) -> bool:
        return self.total_return_pct > self.buy_hold_return_pct


def max_drawdown(curve: pd.Series) -> float:
    """Max peak-to-trough drawdown as a positive fraction."""
    running_peak = curve.cummax()
    dd = 1.0 - curve / running_peak
    return float(dd.max()) if len(dd) else 0.0


def sharpe_ratio(curve: pd.Series) -> float:
    """Annualized Sharpe (rf=0) from daily-resampled equity returns."""
    daily = curve.resample("1D").last().dropna()
    rets = daily.pct_change().dropna()
    if len(rets) < 2 or rets.std() == 0:
        return 0.0
    return float(rets.mean() / rets.std() * math.sqrt(365.0))


def buy_and_hold_return(data: dict[str, pd.DataFrame], costs: CostConfig) -> float:
    """Equal-weight buy at first open, hold, sell at last close.

    Pays taker fee + slippage on both sides, same as the strategy.
    Returns a fraction (0.5 == +50%).
    """
    if not data:
        return 0.0
    legs = []
    for df in data.values():
        if df.empty:
            continue
        entry = buy_fill_price(float(df["open"].iloc[0]), costs.slippage)
        exit_ = sell_fill_price(float(df["close"].iloc[-1]), costs.slippage)
        gross = exit_ / entry
        net = gross * (1.0 - costs.taker_fee) / (1.0 + costs.taker_fee)
        legs.append(net - 1.0)
    return sum(legs) / len(legs) if legs else 0.0


def compute_stats(
    curve: pd.Series,
    trades: list[Trade],
    starting_balance: float,
    data: dict[str, pd.DataFrame],
    costs: CostConfig,
    halted: bool = False,
) -> Stats:
    """Compute the full honest-report stat block for one run."""
    final = float(curve.iloc[-1]) if len(curve) else starting_balance
    total_ret = final / starting_balance - 1.0
    years = max((curve.index[-1] - curve.index[0]).total_seconds() / (365.25 * 86400), 1e-9)
    cagr = (final / starting_balance) ** (1.0 / years) - 1.0 if final > 0 else -1.0

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    gross_win = sum(t.pnl for t in wins)
    gross_loss = -sum(t.pnl for t in losses)
    pf = gross_win / gross_loss if gross_loss > 0 else (math.inf if gross_win > 0 else 0.0)

    return Stats(
        start=str(curve.index[0].date()) if len(curve) else "",
        end=str(curve.index[-1].date()) if len(curve) else "",
        years=round(years, 2),
        final_equity=round(final, 2),
        total_return_pct=round(100 * total_ret, 2),
        cagr_pct=round(100 * cagr, 2),
        max_drawdown_pct=round(100 * max_drawdown(curve), 2),
        sharpe=round(sharpe_ratio(curve), 2),
        win_rate_pct=round(100 * len(wins) / len(trades), 1) if trades else 0.0,
        profit_factor=round(pf, 2) if math.isfinite(pf) else float("inf"),
        num_trades=len(trades),
        total_fees=round(sum(t.fees for t in trades), 2),
        buy_hold_return_pct=round(100 * buy_and_hold_return(data, costs), 2),
        halted=halted,
    )


def format_stats(stats: Stats, title: str) -> str:
    """Plain-English report block. States plainly when buy-and-hold wins."""
    lines = [
        f"── {title} " + "─" * max(0, 60 - len(title)),
        f"  Period               {stats.start} → {stats.end}  ({stats.years} yrs)",
        f"  Final equity         ${stats.final_equity:,.2f}",
        f"  Total return         {stats.total_return_pct:+.2f}%",
        f"  CAGR                 {stats.cagr_pct:+.2f}%",
        f"  Max drawdown         {stats.max_drawdown_pct:.2f}%",
        f"  Sharpe (daily, ann.) {stats.sharpe:.2f}",
        f"  Win rate             {stats.win_rate_pct:.1f}%",
        f"  Profit factor        {stats.profit_factor}",
        f"  Trades               {stats.num_trades}",
        f"  Fees paid            ${stats.total_fees:,.2f}",
        f"  Buy & hold return    {stats.buy_hold_return_pct:+.2f}%",
    ]
    if stats.halted:
        lines.append("  ⚠ KILL SWITCH FIRED during this run — account was flattened and halted.")
    if stats.beats_buy_and_hold():
        lines.append(
            f"  Verdict: strategy beat buy-and-hold by "
            f"{stats.total_return_pct - stats.buy_hold_return_pct:+.2f} pp after costs."
        )
    else:
        lines.append(
            f"  Verdict: strategy DID NOT beat buy-and-hold "
            f"({stats.total_return_pct:+.2f}% vs {stats.buy_hold_return_pct:+.2f}%) after costs."
        )
    return "\n".join(lines)
