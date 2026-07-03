"""Shared test fixtures: a tiny synthetic market and simple configs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from paperbot.config import CostConfig, RiskConfig


@pytest.fixture()
def costs() -> CostConfig:
    return CostConfig(taker_fee=0.0026, slippage=0.0005)


@pytest.fixture()
def zero_costs() -> CostConfig:
    return CostConfig(taker_fee=0.0, slippage=0.0)


@pytest.fixture()
def risk() -> RiskConfig:
    return RiskConfig(risk_per_trade=0.01, max_positions=2, kill_switch_drawdown=0.15)


def make_df(opens, highs, lows, closes, start="2024-01-01") -> pd.DataFrame:
    """Build a candle DataFrame in CandleStore.load format."""
    n = len(opens)
    idx = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "ts": (idx.view(np.int64) // 1_000_000),
            "open": np.asarray(opens, dtype=float),
            "high": np.asarray(highs, dtype=float),
            "low": np.asarray(lows, dtype=float),
            "close": np.asarray(closes, dtype=float),
            "volume": np.ones(n),
        },
        index=idx,
    )
