"""Fill-logic tests: the places where a backtester silently fakes profits."""

from __future__ import annotations

import pytest

from paperbot.backtest.engine import run_backtest
from paperbot.execution import (
    buy_fill_price,
    sell_fill_price,
    stop_exit_reference,
    take_profit_exit_reference,
)
from tests.conftest import make_df
from tests.scripted import ScriptStrategy


def flat_market(n: int, price: float = 100.0):
    return make_df([price] * n, [price] * n, [price] * n, [price] * n)


def test_entry_fills_at_next_bar_open_not_signal_bar(zero_costs, risk):
    """Signal at bar 1's close must fill at bar 2's open — never bar 1."""
    df = make_df(
        opens=[100, 100, 110, 110, 110],
        highs=[100, 100, 110, 110, 110],
        lows=[100, 100, 110, 110, 110],
        closes=[100, 100, 110, 110, 110],
    )
    strat = ScriptStrategy({"enter_bars": {1}, "stop_distance": 50.0})
    res = run_backtest(strat, {"X": df}, 10_000, risk, zero_costs, liquidate_at_end=True)
    assert len(res.trades) == 1
    trade = res.trades[0]
    # Bar 1 closed at 100 but the fill must be bar 2's open of 110.
    assert trade.entry_price == pytest.approx(110.0)


def test_entry_slippage_and_fee_charged(costs, risk):
    df = flat_market(5)
    strat = ScriptStrategy({"enter_bars": {1}, "stop_distance": 50.0})
    res = run_backtest(strat, {"X": df}, 10_000, risk, costs, liquidate_at_end=True)
    t = res.trades[0]
    assert t.entry_price == pytest.approx(buy_fill_price(100.0, costs.slippage))
    # Both sides pay the 0.26% taker fee on their notional.
    expected_fees = (
        t.qty * t.entry_price * costs.taker_fee + t.qty * t.exit_price * costs.taker_fee
    )
    assert t.fees == pytest.approx(expected_fees)


def test_flat_market_round_trip_loses_exactly_costs(costs, risk):
    """In a flat market the only P&L is fees + slippage — and it's negative."""
    df = flat_market(6)
    strat = ScriptStrategy({"enter_bars": {1}, "exit_bars": {3}, "stop_distance": 50.0})
    res = run_backtest(strat, {"X": df}, 10_000, risk, costs)
    t = res.trades[0]
    assert t.pnl < 0
    gross_entry = t.qty * t.entry_price + t.qty * t.entry_price * costs.taker_fee
    gross_exit = t.qty * t.exit_price - t.qty * t.exit_price * costs.taker_fee
    assert t.pnl == pytest.approx(gross_exit - gross_entry)


def test_stop_fills_at_stop_when_touched_intrabar(zero_costs, risk):
    """Bar trades through the stop: fill at the stop level, not the low."""
    df = make_df(
        opens=[100, 100, 100, 100],
        highs=[100, 100, 100, 100],
        lows=[100, 100, 100, 80],  # bar 3 dives to 80
        closes=[100, 100, 100, 95],
    )
    strat = ScriptStrategy({"enter_bars": {1}, "stop_distance": 10.0})  # stop at 90
    res = run_backtest(strat, {"X": df}, 10_000, risk, zero_costs)
    t = res.trades[0]
    assert t.exit_reason == "stop hit"
    assert t.exit_price == pytest.approx(90.0)


def test_stop_gap_through_fills_at_worse_open(zero_costs, risk):
    """Bar OPENS below the stop: you get the open, not the stop price."""
    df = make_df(
        opens=[100, 100, 100, 70],  # bar 3 gaps open far below the 90 stop
        highs=[100, 100, 100, 75],
        lows=[100, 100, 100, 65],
        closes=[100, 100, 100, 72],
    )
    strat = ScriptStrategy({"enter_bars": {1}, "stop_distance": 10.0})
    res = run_backtest(strat, {"X": df}, 10_000, risk, zero_costs)
    t = res.trades[0]
    assert t.exit_reason == "stop hit"
    assert t.exit_price == pytest.approx(70.0)  # the gap open, NOT 90


def test_stop_beats_take_profit_when_both_touched(zero_costs, risk):
    """Pessimistic tie-break: a bar spanning both stop and TP takes the stop."""
    df = make_df(
        opens=[100, 100, 100, 100],
        highs=[100, 100, 100, 130],  # TP (110) touched...
        lows=[100, 100, 100, 85],  # ...but so is the stop (90)
        closes=[100, 100, 100, 100],
    )
    strat = ScriptStrategy({"enter_bars": {1}, "stop_distance": 10.0, "tp_distance": 10.0})
    res = run_backtest(strat, {"X": df}, 10_000, risk, zero_costs)
    t = res.trades[0]
    assert t.exit_reason == "stop hit"
    assert t.exit_price == pytest.approx(90.0)


def test_take_profit_gap_up_fills_at_better_open(zero_costs, risk):
    """TP is limit-like: a gap open above it fills at the open."""
    df = make_df(
        opens=[100, 100, 100, 125],
        highs=[100, 100, 100, 126],
        lows=[100, 100, 100, 120],
        closes=[100, 100, 100, 124],
    )
    strat = ScriptStrategy({"enter_bars": {1}, "stop_distance": 50.0, "tp_distance": 10.0})
    res = run_backtest(strat, {"X": df}, 10_000, risk, zero_costs)
    t = res.trades[0]
    assert t.exit_reason == "take profit"
    assert t.exit_price == pytest.approx(125.0)


def test_exit_signal_fills_next_open(zero_costs, risk):
    df = make_df(
        opens=[100, 100, 100, 100, 90, 90],
        highs=[100, 100, 100, 100, 90, 90],
        lows=[100, 100, 100, 100, 90, 90],
        closes=[100, 100, 100, 100, 90, 90],
    )
    strat = ScriptStrategy({"enter_bars": {1}, "exit_bars": {3}, "stop_distance": 50.0})
    res = run_backtest(strat, {"X": df}, 10_000, risk, zero_costs)
    t = res.trades[0]
    # Exit signal at bar 3's close (price 100) fills at bar 4's open (90).
    assert t.exit_price == pytest.approx(90.0)


def test_slippage_directions(zero_costs):
    assert buy_fill_price(100.0, 0.001) == pytest.approx(100.1)
    assert sell_fill_price(100.0, 0.001) == pytest.approx(99.9)
    assert stop_exit_reference(bar_open=95.0, stop_price=90.0) == 90.0
    assert stop_exit_reference(bar_open=85.0, stop_price=90.0) == 85.0
    assert take_profit_exit_reference(bar_open=105.0, tp_price=110.0) == 110.0
    assert take_profit_exit_reference(bar_open=115.0, tp_price=110.0) == 115.0
