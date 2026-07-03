"""Walk-forward analysis.

For each rolling window:
  1. Grid-search parameters on `train_months` of data (in-sample).
  2. Run the winning parameters on the following `test_months`
     (out-of-sample), starting from the equity the previous test window
     ended with, so the stitched OOS curve compounds realistically.
  3. Roll forward by `step_months`.

In-sample numbers are reported separately and should be treated as
marketing; the stitched out-of-sample result is the honest one.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from paperbot.backtest.engine import BacktestResult, run_backtest
from paperbot.backtest.metrics import Stats, compute_stats, format_stats, max_drawdown
from paperbot.config import Config
from paperbot.execution import Trade
from paperbot.strategies.base import Strategy

log = logging.getLogger(__name__)

# Bars fed to the test run before trading starts, purely to warm up
# indicators (EMA200 etc.). 450 1h-bars ≈ 19 days.
WARMUP_BARS = 450


@dataclass
class WindowResult:
    train_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    chosen_params: dict[str, Any]
    used_defaults: bool
    train_score: float
    train_trades: int
    oos_return_pct: float
    oos_trades: int
    halted: bool


@dataclass
class WalkForwardResult:
    windows: list[WindowResult]
    oos_curve: pd.Series
    oos_trades: list[Trade]
    oos_stats: Stats
    is_stats_mean_return_pct: float
    strategy_name: str
    objective: str
    report: str = ""


def _score(result: BacktestResult, objective: str) -> float:
    """Objective for the train-window optimizer."""
    curve = result.equity_curve
    total_ret = float(curve.iloc[-1]) / result.starting_balance - 1.0
    if objective == "return_over_maxdd":
        return total_ret / max(max_drawdown(curve), 0.01)
    if objective == "profit_factor":
        wins = sum(t.pnl for t in result.trades if t.pnl > 0)
        losses = -sum(t.pnl for t in result.trades if t.pnl <= 0)
        return wins / losses if losses > 0 else (10.0 if wins > 0 else 0.0)
    if objective == "sharpe":
        from paperbot.backtest.metrics import sharpe_ratio

        return sharpe_ratio(curve)
    raise ValueError(f"unknown objective: {objective}")


def _slice(data: dict[str, pd.DataFrame], start: pd.Timestamp, end: pd.Timestamp,
           warmup_bars: int = 0) -> dict[str, pd.DataFrame]:
    out = {}
    for sym, df in data.items():
        sub = df[df.index < end]
        cutoff_pos = sub.index.searchsorted(start)
        sub = sub.iloc[max(0, cutoff_pos - warmup_bars):]
        if len(sub):
            out[sym] = sub
    return out


def run_walkforward(
    strategy_cls: type[Strategy],
    data: dict[str, pd.DataFrame],
    cfg: Config,
) -> WalkForwardResult:
    """Run the full walk-forward analysis and build a plain-English report."""
    wf = cfg.walkforward
    section = cfg.strategies[strategy_cls.name]
    grid = strategy_cls.param_grid(section)
    defaults = strategy_cls.default_params(section)

    first = min(df.index[0] for df in data.values())
    last = max(df.index[-1] for df in data.values())

    windows: list[WindowResult] = []
    oos_curves: list[pd.Series] = []
    oos_trades: list[Trade] = []
    is_returns: list[float] = []
    any_halted = False
    equity = cfg.backtest_starting_balance

    train_start = first
    while True:
        test_start = train_start + pd.DateOffset(months=wf.train_months)
        test_end = test_start + pd.DateOffset(months=wf.test_months)
        if test_start >= last:
            break
        test_end = min(test_end, last + pd.Timedelta(hours=1))

        train_data = _slice(data, train_start, test_start)
        if not train_data or min(len(df) for df in train_data.values()) < WARMUP_BARS * 2:
            train_start += pd.DateOffset(months=wf.step_months)
            continue

        # ---- In-sample optimization ----
        best_params, best_score, best_trades = None, -math.inf, 0
        for params in grid:
            res = run_backtest(
                strategy_cls(params), train_data, cfg.backtest_starting_balance,
                cfg.risk, cfg.costs, liquidate_at_end=True,
            )
            score = _score(res, wf.objective)
            if score > best_score:
                best_params, best_score, best_trades = params, score, len(res.trades)

        used_defaults = False
        if best_params is None or best_trades < wf.min_train_trades:
            # Too few trades to trust the optimizer — fall back to defaults.
            best_params, used_defaults = defaults, True
        is_returns.append(best_score)

        # ---- Out-of-sample test (chained equity) ----
        test_data = _slice(data, test_start, test_end, warmup_bars=WARMUP_BARS)
        oos = run_backtest(
            strategy_cls(best_params), test_data, equity, cfg.risk, cfg.costs,
            trade_from=test_start, liquidate_at_end=True,
        )
        oos_ret = float(oos.equity_curve.iloc[-1]) / equity - 1.0
        windows.append(
            WindowResult(
                train_start=train_start,
                test_start=test_start,
                test_end=test_end,
                chosen_params={k: best_params[k] for k in sorted(section.get("grid") or best_params)},
                used_defaults=used_defaults,
                train_score=round(best_score, 3) if math.isfinite(best_score) else 0.0,
                train_trades=best_trades,
                oos_return_pct=round(100 * oos_ret, 2),
                oos_trades=len(oos.trades),
                halted=oos.halted,
            )
        )
        oos_curves.append(oos.equity_curve)
        oos_trades.extend(oos.trades)
        any_halted = any_halted or oos.halted
        equity = float(oos.equity_curve.iloc[-1])
        log.info(
            "window %s→%s: params=%s OOS return %+.2f%% (%d trades)",
            test_start.date(), test_end.date(), windows[-1].chosen_params, 100 * oos_ret, len(oos.trades),
        )

        train_start += pd.DateOffset(months=wf.step_months)

    if not oos_curves:
        raise ValueError("not enough data for a single walk-forward window")

    stitched = pd.concat(oos_curves)
    stitched = stitched[~stitched.index.duplicated(keep="last")].sort_index()
    oos_span_data = _slice(data, stitched.index[0], stitched.index[-1] + pd.Timedelta(hours=1))
    oos_stats = compute_stats(
        stitched, oos_trades, cfg.backtest_starting_balance, oos_span_data, cfg.costs,
        halted=any_halted,
    )

    result = WalkForwardResult(
        windows=windows,
        oos_curve=stitched,
        oos_trades=oos_trades,
        oos_stats=oos_stats,
        is_stats_mean_return_pct=round(float(pd.Series(is_returns).mean()), 3) if is_returns else 0.0,
        strategy_name=strategy_cls.name,
        objective=wf.objective,
    )
    result.report = _format_report(result, cfg)
    return result


def _format_report(r: WalkForwardResult, cfg: Config) -> str:
    lines = [
        "",
        "=" * 66,
        f"WALK-FORWARD REPORT — strategy: {r.strategy_name}",
        f"train {cfg.walkforward.train_months}mo / test {cfg.walkforward.test_months}mo "
        f"/ step {cfg.walkforward.step_months}mo, objective: {r.objective}",
        "=" * 66,
        "",
        "Per-window out-of-sample results (the only numbers that matter):",
        f"{'test period':<26}{'params':<34}{'OOS ret':>9}{'trades':>8}",
    ]
    for w in r.windows:
        period = f"{w.test_start.date()} → {w.test_end.date()}"
        params = ", ".join(f"{k}={v}" for k, v in sorted(w.chosen_params.items()))
        if w.used_defaults:
            params += " (defaults)"
        halted_mark = "  ⚠halt" if w.halted else ""
        lines.append(
            f"{period:<26}{params:<34}{w.oos_return_pct:>+8.2f}%{w.oos_trades:>8}{halted_mark}"
        )
    n_halted = sum(1 for w in r.windows if w.halted)
    if n_halted:
        lines.append("")
        lines.append(
            f"NOTE: the kill switch fired in {n_halted}/{len(r.windows)} test windows. "
            "Each window restarts with a fresh drawdown peak, so the stitched curve keeps "
            "trading; a persistent live account would have halted permanently at the FIRST "
            "-15% drawdown and stopped there. The stitched number therefore understates "
            "nothing — it shows what repeatedly re-enabling a losing bot would cost."
        )
    lines.append("")
    lines.append(format_stats(r.oos_stats, "STITCHED OUT-OF-SAMPLE (honest number)"))
    lines.append("")
    lines.append(
        f"In-sample mean objective score was {r.is_stats_mean_return_pct} — reported only to "
        "show how much the optimizer flattered itself; do not use it to judge the strategy."
    )
    return "\n".join(lines)
