"""MeanRevert: RSI(2) dip-buying inside an uptrend.

Long when price is above the long EMA (uptrend) and RSI(2) is deeply
oversold. Fixed take-profit and fixed ATR stop set at entry; also exits
when RSI recovers above the exit threshold.
"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from paperbot.indicators import atr, ema, rsi
from paperbot.strategies.base import EntryIntent, ManageAction, Strategy


class MeanRevert(Strategy):
    name = "meanrevert"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        out = df.copy()
        out["ema_trend"] = ema(out["close"], int(p["ema_trend"]))
        out["rsi"] = rsi(out["close"], int(p["rsi_period"]))
        out["atr"] = atr(out, int(p["atr_period"]))
        return out

    def should_enter(self, row: Any) -> EntryIntent | None:
        if pd.isna(row.ema_trend) or pd.isna(row.rsi) or pd.isna(row.atr):
            return None
        if row.atr <= 0:
            return None
        if row.close > row.ema_trend and row.rsi < float(self.params["rsi_entry"]):
            return EntryIntent(
                stop_distance=float(self.params["atr_stop_mult"]) * row.atr,
                take_profit_distance=float(self.params["take_profit_atr_mult"]) * row.atr,
                reason=f"rsi({int(self.params['rsi_period'])})={row.rsi:.1f} dip in uptrend",
            )
        return None

    def manage(self, row: Any, position: Any) -> ManageAction | None:
        if not pd.isna(row.rsi) and row.rsi > float(self.params["rsi_exit"]):
            return ManageAction(exit_market=True, reason="rsi recovered")
        return None

    @classmethod
    def default_params(cls, cfg_section: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "ema_trend": cfg_section["ema_trend"],
            "rsi_period": cfg_section["rsi_period"],
            "rsi_entry": cfg_section["rsi_entry"],
            "rsi_exit": cfg_section["rsi_exit"],
            "atr_period": cfg_section["atr_period"],
            "atr_stop_mult": cfg_section["atr_stop_mult"],
            "take_profit_atr_mult": cfg_section["take_profit_atr_mult"],
        }
