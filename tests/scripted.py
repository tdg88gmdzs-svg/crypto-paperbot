"""A deterministic strategy for driving the engine in tests."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from paperbot.strategies.base import EntryIntent, ManageAction, Strategy


class ScriptStrategy(Strategy):
    """Enters on bar numbers in `enter_bars`, exits on `exit_bars`.

    params:
      enter_bars: set[int]      bar indices whose CLOSE emits an entry signal
      exit_bars: set[int]       bar indices whose CLOSE emits a market exit
      stop_distance: float      initial stop distance in price units
      tp_distance: float|None   optional take-profit distance
    """

    name = "script"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["bar_no"] = np.arange(len(out))
        return out

    def should_enter(self, row: Any) -> EntryIntent | None:
        if int(row.bar_no) in self.params["enter_bars"]:
            return EntryIntent(
                stop_distance=float(self.params["stop_distance"]),
                take_profit_distance=self.params.get("tp_distance"),
                reason="scripted entry",
            )
        return None

    def manage(self, row: Any, position: Any) -> ManageAction | None:
        if int(row.bar_no) in self.params.get("exit_bars", set()):
            return ManageAction(exit_market=True, reason="scripted exit")
        return None

    @classmethod
    def default_params(cls, cfg_section: Mapping[str, Any]) -> dict[str, Any]:
        return dict(cfg_section)
