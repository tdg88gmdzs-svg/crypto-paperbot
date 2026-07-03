"""Strategy base class.

A strategy is deliberately dumb about money: it only ever expresses
"enter long with this stop distance" or "exit". Position sizing, fees,
slippage, and risk limits belong to the engine/risk layer, so a strategy
cannot cheat its way to better numbers.

Long-only by design: this simulates spot trading with no margin.
"""

from __future__ import annotations

import abc
import itertools
from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd


@dataclass(frozen=True)
class EntryIntent:
    """A request to open a long position at the next bar's open.

    stop_distance: initial stop distance in price units (engine sets
        stop = fill_price - stop_distance). Also drives position sizing.
    take_profit_distance: optional fixed TP distance in price units
        (engine sets tp = fill_price + take_profit_distance).
    """

    stop_distance: float
    take_profit_distance: float | None = None
    reason: str = ""


@dataclass(frozen=True)
class ManageAction:
    """What to do with an open position after a bar closes.

    exit_market: liquidate at the next bar's open.
    new_stop: raise the stop to this level (engine ignores lowering —
        stops only ever tighten).
    """

    exit_market: bool = False
    new_stop: float | None = None
    reason: str = ""


class Strategy(abc.ABC):
    """Base class for all strategies.

    Lifecycle per (symbol, run):
      1. `prepare(df)` adds indicator columns. Must be causal: row i may
         only depend on rows <= i.
      2. For each closed bar, the engine calls `should_enter(row)` when
         flat, or `manage(row, position)` when in a position. `row` is a
         namedtuple of the prepared bar (from DataFrame.itertuples).
    """

    name: str = "base"

    def __init__(self, params: Mapping[str, Any]) -> None:
        self.params = dict(params)

    @abc.abstractmethod
    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of `df` with indicator columns added (causal only)."""

    @abc.abstractmethod
    def should_enter(self, row: Any) -> EntryIntent | None:
        """Called on each bar close while flat. Return an EntryIntent to
        request a fill at the next bar's open, or None."""

    @abc.abstractmethod
    def manage(self, row: Any, position: Any) -> ManageAction | None:
        """Called on each bar close while holding `position` (a
        paperbot.execution.Position). May tighten the stop or exit."""

    @classmethod
    @abc.abstractmethod
    def default_params(cls, cfg_section: Mapping[str, Any]) -> dict[str, Any]:
        """Default parameters from this strategy's config.yaml section."""

    @classmethod
    def param_grid(cls, cfg_section: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Cartesian product of the `grid` block over the defaults.

        Used by walk-forward optimization. Small on purpose: every extra
        dimension is another chance to overfit.
        """
        base = cls.default_params(cfg_section)
        grid: Mapping[str, list[Any]] = cfg_section.get("grid", {})
        if not grid:
            return [base]
        keys = sorted(grid)
        combos = []
        for values in itertools.product(*(grid[k] for k in keys)):
            p = dict(base)
            p.update(dict(zip(keys, values)))
            combos.append(p)
        return combos
