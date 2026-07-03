"""Pluggable strategies. Register new strategies in STRATEGIES."""

from __future__ import annotations

from paperbot.strategies.base import EntryIntent, ManageAction, Strategy
from paperbot.strategies.meanrevert import MeanRevert
from paperbot.strategies.trendfollow import TrendFollow

STRATEGIES: dict[str, type[Strategy]] = {
    TrendFollow.name: TrendFollow,
    MeanRevert.name: MeanRevert,
}

__all__ = ["Strategy", "EntryIntent", "ManageAction", "TrendFollow", "MeanRevert", "STRATEGIES"]
