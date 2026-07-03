"""TrendFollow: EMA regime filter + Donchian breakout + ATR trailing stop.

Long when the fast EMA is above the slow EMA (uptrend regime) and the close
breaks above the prior N-bar Donchian upper channel. Exit on a chandelier
trailing stop (highest close since entry minus k*ATR) or when the regime
flips bearish.
"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from paperbot.indicators import atr, donchian_upper, ema
from paperbot.strategies.base import EntryIntent, ManageAction, Strategy


class TrendFollow(Strategy):
    name = "trendfollow"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        out = df.copy()
        out["ema_fast"] = ema(out["close"], int(p["ema_fast"]))
        out["ema_slow"] = ema(out["close"], int(p["ema_slow"]))
        out["donch_up"] = donchian_upper(out["high"], int(p["donchian_period"]))
        out["atr"] = atr(out, int(p["atr_period"]))
        return out

    def should_enter(self, row: Any) -> EntryIntent | None:
        if pd.isna(row.ema_slow) or pd.isna(row.donch_up) or pd.isna(row.atr):
            return None
        if row.atr <= 0:
            return None
        uptrend = row.ema_fast > row.ema_slow
        breakout = row.close > row.donch_up
        if uptrend and breakout:
            return EntryIntent(
                stop_distance=float(self.params["atr_stop_mult"]) * row.atr,
                reason=f"donchian({int(self.params['donchian_period'])}) breakout in uptrend",
            )
        return None

    def manage(self, row: Any, position: Any) -> ManageAction | None:
        if not pd.isna(row.ema_slow) and row.ema_fast < row.ema_slow:
            return ManageAction(exit_market=True, reason="regime flipped bearish")
        if pd.isna(row.atr):
            return None
        trail = position.highest_close - float(self.params["atr_stop_mult"]) * row.atr
        if trail > position.stop_price:
            return ManageAction(new_stop=trail, reason="trail stop raised")
        return None

    @classmethod
    def default_params(cls, cfg_section: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "ema_fast": cfg_section["ema_fast"],
            "ema_slow": cfg_section["ema_slow"],
            "donchian_period": cfg_section["donchian_period"],
            "atr_period": cfg_section["atr_period"],
            "atr_stop_mult": cfg_section["atr_stop_mult"],
        }
