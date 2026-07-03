"""Command-line interface.

  bot backtest    --strategy trendfollow --from 2021-01-01
  bot walkforward --strategy trendfollow
  bot paper
  bot report
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer

from paperbot.backtest.engine import run_backtest
from paperbot.backtest.metrics import compute_stats, format_stats
from paperbot.config import Config, load_config
from paperbot.data import CandleStore, ensure_history, make_exchange
from paperbot.logutil import setup_logging
from paperbot.strategies import STRATEGIES

app = typer.Typer(help="crypto-paperbot: paper trading only, no real orders — ever.")

CONFIG_OPT = typer.Option("config.yaml", "--config", "-c", help="Path to config.yaml")


def _load(config_path: str) -> Config:
    cfg = load_config(config_path)
    setup_logging(cfg.log_dir, cfg.log_level)
    return cfg


def _parse_date(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def _load_data(cfg: Config, start: datetime, end: Optional[datetime] = None):
    """Download/refresh cached candles for all configured symbols."""
    store = CandleStore(cfg.data.cache_db)
    exchange = make_exchange(cfg.data.history_exchange)
    data = {}
    for symbol in cfg.data.symbols:
        df = ensure_history(store, exchange, symbol, cfg.data.timeframe, start, end)
        if df.empty:
            typer.echo(f"warning: no data for {symbol}", err=True)
        else:
            data[symbol] = df
    store.close()
    return data


def _resolve_strategy(name: str):
    if name not in STRATEGIES:
        typer.echo(f"unknown strategy {name!r}; available: {', '.join(STRATEGIES)}", err=True)
        raise typer.Exit(1)
    return STRATEGIES[name]


@app.command()
def backtest(
    strategy: str = typer.Option("trendfollow", help="Strategy name"),
    from_: str = typer.Option("2021-01-01", "--from", help="Start date (ISO)"),
    to: Optional[str] = typer.Option(None, "--to", help="End date (ISO), default now"),
    config: str = CONFIG_OPT,
) -> None:
    """Run a single backtest with default parameters over the full range."""
    cfg = _load(config)
    cls = _resolve_strategy(strategy)
    start = _parse_date(from_)
    end = _parse_date(to) if to else None
    data = _load_data(cfg, start, end)
    if not data:
        raise typer.Exit(1)

    strat = cls(cls.default_params(cfg.strategies[strategy]))
    result = run_backtest(
        strat, data, cfg.backtest_starting_balance, cfg.risk, cfg.costs,
        liquidate_at_end=True,
    )
    stats = compute_stats(
        result.equity_curve, result.trades, cfg.backtest_starting_balance,
        data, cfg.costs, halted=result.halted,
    )
    typer.echo(format_stats(stats, f"BACKTEST {strategy} (in-sample, default params)"))
    typer.echo(
        "\nNote: a single full-period backtest is in-sample by definition. "
        "Run `bot walkforward` for the honest out-of-sample view."
    )


@app.command()
def walkforward(
    strategy: str = typer.Option("trendfollow", help="Strategy name"),
    from_: str = typer.Option("2021-01-01", "--from", help="Start date (ISO)"),
    to: Optional[str] = typer.Option(None, "--to", help="End date (ISO), default now"),
    config: str = CONFIG_OPT,
) -> None:
    """Walk-forward analysis: optimize on 12mo, test on the next 3mo, roll."""
    from paperbot.walkforward import run_walkforward

    cfg = _load(config)
    cls = _resolve_strategy(strategy)
    data = _load_data(cfg, _parse_date(from_), _parse_date(to) if to else None)
    if not data:
        raise typer.Exit(1)
    result = run_walkforward(cls, data, cfg)
    typer.echo(result.report)


@app.command()
def paper(
    strategy: Optional[list[str]] = typer.Option(
        None, "--strategy", help="Strategy name(s); default: all registered"
    ),
    once: bool = typer.Option(
        False, "--once", help="Process new candles once and exit (for schedulers)"
    ),
    config: str = CONFIG_OPT,
) -> None:
    """Run the live paper-trading loop (Ctrl-C to stop), or one tick with --once."""
    from paperbot.paper.runner import run_paper_loop, run_paper_once

    cfg = _load(config)
    names = list(strategy) if strategy else list(STRATEGIES)
    for n in names:
        _resolve_strategy(n)
    if once:
        n_bars = run_paper_once(cfg, names)
        typer.echo(f"tick complete: {n_bars} bar(s) processed.")
        return
    try:
        asyncio.run(run_paper_loop(cfg, names))
    except KeyboardInterrupt:
        typer.echo("stopped. State is saved; restart `bot paper` to resume.")


@app.command()
def dashboard(
    out: str = typer.Option("docs", help="Output directory for the static site"),
    config: str = CONFIG_OPT,
) -> None:
    """Generate the static HTML dashboard (docs/index.html for GitHub Pages)."""
    from paperbot.dashboard import build_dashboard

    cfg = _load(config)
    path = build_dashboard(cfg, Path(out))
    typer.echo(f"dashboard written to {path}")


@app.command()
def report(config: str = CONFIG_OPT) -> None:
    """Print current paper-account stats and recent trades."""
    from paperbot.report import build_report

    cfg = _load(config)
    typer.echo(build_report(cfg))


if __name__ == "__main__":
    app()
