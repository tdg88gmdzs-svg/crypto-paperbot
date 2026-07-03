"""Causal technical indicators.

Every function here uses only data at or before each row (rolling/EWM are
causal by construction), so a strategy reading row i's indicator values in
the backtester cannot look ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average (standard span parameterization)."""
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's Average True Range from columns high/low/close."""
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    # Wilder smoothing == EWM with alpha = 1/period.
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI. Returns values in [0, 100]."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # All-gain windows (avg_loss == 0) are RSI 100 by convention.
    return out.fillna(100.0).where(avg_gain.notna() & avg_loss.notna())


def donchian_upper(high: pd.Series, period: int) -> pd.Series:
    """Highest high of the *previous* `period` bars (excludes current bar,
    so a close above it is a genuine breakout, not self-comparison)."""
    return high.rolling(period, min_periods=period).max().shift(1)


def donchian_lower(low: pd.Series, period: int) -> pd.Series:
    """Lowest low of the previous `period` bars (excludes current bar)."""
    return low.rolling(period, min_periods=period).min().shift(1)
