"""`bot report`: current paper-account stats and recent trades."""

from __future__ import annotations

from datetime import datetime, timezone

from paperbot.backtest.metrics import max_drawdown, sharpe_ratio
from paperbot.config import Config
from paperbot.paper.state import PaperState


def _fmt_ts(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def build_report(cfg: Config) -> str:
    """Assemble the paper-account report as plain text."""
    if not cfg.paper.state_db.exists():
        return "No paper state found. Run `bot paper` first."
    state = PaperState(cfg.paper.state_db)
    try:
        account, pending = state.load_account(cfg.paper.starting_balance)
        curve = state.equity_curve()
        lines = ["═══ PAPER ACCOUNT REPORT ═══"]
        if curve.empty:
            lines.append("No equity history yet (the loop hasn't processed a candle).")
            lines.append(f"cash: ${account.cash:,.2f}")
            return "\n".join(lines)

        equity = float(curve.iloc[-1])
        start = cfg.paper.starting_balance
        dd_now = 1.0 - equity / account.peak_equity if account.peak_equity else 0.0
        wins = [t for t in account.trades if t.pnl > 0]
        lines += [
            f"as of                {curve.index[-1]:%Y-%m-%d %H:%M} UTC",
            f"equity               ${equity:,.2f}  (started ${start:,.2f}, "
            f"{100 * (equity / start - 1):+.2f}%)",
            f"peak equity          ${account.peak_equity:,.2f}  "
            f"(current drawdown {100 * dd_now:.1f}%)",
            f"max drawdown         {100 * max_drawdown(curve):.2f}%",
            f"sharpe (daily, ann.) {sharpe_ratio(curve):.2f}",
            f"closed trades        {len(account.trades)} "
            f"(win rate {100 * len(wins) / len(account.trades):.1f}%)"
            if account.trades else "closed trades        0",
            f"open positions       {len(account.positions)}",
        ]
        for pos in account.positions.values():
            lines.append(
                f"  {pos.symbol} [{pos.strategy}] qty={pos.qty:.6f} "
                f"entry={pos.entry_price:.2f} stop={pos.stop_price:.2f} "
                f"since {_fmt_ts(pos.entry_ts)}"
            )
        if pending:
            lines.append(f"pending orders       {len(pending)}")
        if account.halted:
            lines.append("⚠ ACCOUNT HALTED by kill switch. Delete the state DB to reset.")

        recent = state.recent_trades(15)
        if recent:
            lines.append("")
            lines.append("recent trades (newest first):")
            for t in recent:
                lines.append(
                    f"  {_fmt_ts(t.exit_ts)}  {t.symbol:<9} {t.strategy:<12} "
                    f"pnl {t.pnl:+9.2f}  fees {t.fees:6.2f}  {t.exit_reason}"
                )
        return "\n".join(lines)
    finally:
        state.close()
