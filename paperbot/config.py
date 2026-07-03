"""Typed access to config.yaml.

Loads the YAML once and exposes plain dataclasses so the rest of the code
gets attribute access and type hints instead of dict-key spelunking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    live_exchange: str
    history_exchange: str
    symbols: list[str]
    timeframe: str
    entry_timeframe: str
    cache_db: Path


@dataclass(frozen=True)
class CostConfig:
    """Fees/slippage as fractions (0.0026, not 0.26)."""

    taker_fee: float
    slippage: float


@dataclass(frozen=True)
class RiskConfig:
    risk_per_trade: float  # fraction of equity, e.g. 0.01
    max_positions: int
    kill_switch_drawdown: float  # fraction, e.g. 0.15


@dataclass(frozen=True)
class PaperConfig:
    starting_balance: float
    state_db: Path
    candle_grace_seconds: int


@dataclass(frozen=True)
class WalkForwardConfig:
    train_months: int
    test_months: int
    step_months: int
    objective: str
    min_train_trades: int


@dataclass(frozen=True)
class Config:
    data: DataConfig
    costs: CostConfig
    risk: RiskConfig
    paper: PaperConfig
    walkforward: WalkForwardConfig
    backtest_starting_balance: float
    strategies: dict[str, dict[str, Any]] = field(default_factory=dict)
    log_dir: Path = Path("logs")
    log_level: str = "INFO"
    root: Path = Path(".")


def load_config(path: str | Path = "config.yaml") -> Config:
    """Load and validate config.yaml, converting percentages to fractions."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    root = path.parent.resolve()

    d = raw["data"]
    c = raw["costs"]
    r = raw["risk"]
    p = raw["paper"]
    w = raw["walkforward"]

    return Config(
        data=DataConfig(
            live_exchange=d["live_exchange"],
            history_exchange=d["history_exchange"],
            symbols=list(d["symbols"]),
            timeframe=d["timeframe"],
            entry_timeframe=d["entry_timeframe"],
            cache_db=root / d["cache_db"],
        ),
        costs=CostConfig(
            taker_fee=c["taker_fee_pct"] / 100.0,
            slippage=c["slippage_pct"] / 100.0,
        ),
        risk=RiskConfig(
            risk_per_trade=r["risk_per_trade_pct"] / 100.0,
            max_positions=int(r["max_positions"]),
            kill_switch_drawdown=r["kill_switch_drawdown_pct"] / 100.0,
        ),
        paper=PaperConfig(
            starting_balance=float(p["starting_balance_usd"]),
            state_db=root / p["state_db"],
            candle_grace_seconds=int(p["candle_grace_seconds"]),
        ),
        walkforward=WalkForwardConfig(
            train_months=int(w["train_months"]),
            test_months=int(w["test_months"]),
            step_months=int(w["step_months"]),
            objective=str(w["objective"]),
            min_train_trades=int(w["min_train_trades"]),
        ),
        backtest_starting_balance=float(raw["backtest"]["starting_balance_usd"]),
        strategies=raw.get("strategies", {}),
        log_dir=root / raw["logging"]["dir"],
        log_level=str(raw["logging"]["level"]),
        root=root,
    )
