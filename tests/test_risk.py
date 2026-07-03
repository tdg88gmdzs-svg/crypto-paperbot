"""Risk-rule tests: sizing, position limits, and the kill switch."""

from __future__ import annotations

import pytest

from paperbot.backtest.engine import run_backtest
from paperbot.config import CostConfig, RiskConfig
from paperbot.execution import Account, size_position
from paperbot.strategies.base import EntryIntent
from tests.conftest import make_df
from tests.scripted import ScriptStrategy


def test_size_risks_one_percent_of_equity(zero_costs, risk):
    """$10,000 equity, 1% risk, $50 stop distance → qty 2 ($100 at risk)."""
    qty = size_position(
        equity=10_000, cash=10_000, est_entry_price=1_000,
        stop_distance=50.0, risk=risk, costs=zero_costs,
    )
    assert qty == pytest.approx(2.0)
    assert qty * 50.0 == pytest.approx(10_000 * 0.01)


def test_size_capped_by_cash_no_leverage(costs, risk):
    """A tight stop must not let notional exceed available cash."""
    qty = size_position(
        equity=10_000, cash=10_000, est_entry_price=100.0,
        stop_distance=0.10, risk=risk, costs=costs,  # naive qty would be 1000
    )
    assert qty * 100.0 * (1 + costs.taker_fee) <= 10_000 + 1e-6


def test_size_zero_on_bad_inputs(zero_costs, risk):
    assert size_position(10_000, 10_000, 100.0, 0.0, risk, zero_costs) == 0.0
    assert size_position(10_000, 10_000, -5.0, 10.0, risk, zero_costs) == 0.0
    assert size_position(0.0, 0.0, 100.0, 10.0, risk, zero_costs) == 0.0


def test_stop_loss_realizes_about_one_percent(zero_costs, risk):
    """End-to-end: stopped-out trade loses ~1% of starting equity."""
    df = make_df(
        opens=[100, 100, 100, 100],
        highs=[100, 100, 100, 100],
        lows=[100, 100, 100, 90],  # clean stop hit at 90, no gap
        closes=[100, 100, 100, 95],
    )
    strat = ScriptStrategy({"enter_bars": {1}, "stop_distance": 10.0})
    res = run_backtest(strat, {"X": df}, 10_000, risk, zero_costs)
    t = res.trades[0]
    assert t.pnl == pytest.approx(-100.0)  # exactly 1% of $10k with zero costs


def test_max_positions_enforced(zero_costs, risk):
    """Third concurrent entry must be rejected (max_positions=2)."""
    account = Account(cash=100_000)
    marks = {"A": 100.0, "B": 100.0, "C": 100.0}
    intent = EntryIntent(stop_distance=10.0)
    assert account.open_long("A", "s", 0, 100.0, intent, marks, risk, zero_costs)
    assert account.open_long("B", "s", 0, 100.0, intent, marks, risk, zero_costs)
    assert account.open_long("C", "s", 0, 100.0, intent, marks, risk, zero_costs) is None
    assert len(account.positions) == 2


def test_no_duplicate_position_same_symbol_strategy(zero_costs, risk):
    account = Account(cash=100_000)
    marks = {"A": 100.0}
    intent = EntryIntent(stop_distance=10.0)
    assert account.open_long("A", "s", 0, 100.0, intent, marks, risk, zero_costs)
    assert account.open_long("A", "s", 1, 100.0, intent, marks, risk, zero_costs) is None


def test_kill_switch_flattens_and_halts(zero_costs):
    """15% drawdown from peak → flatten everything, halt, ignore new signals."""
    risk = RiskConfig(risk_per_trade=0.5, max_positions=2, kill_switch_drawdown=0.15)
    df = make_df(
        opens=[100, 100, 100, 75, 75, 75, 75],
        highs=[100, 100, 100, 75, 75, 75, 75],
        lows=[100, 100, 100, 75, 75, 75, 75],  # never touches the 50 stop
        closes=[100, 100, 75, 75, 75, 75, 75],  # bar 2 close: -25% on the position
        # enter_bars 4 below: a signal AFTER the halt that must be ignored
    )
    strat = ScriptStrategy({"enter_bars": {0, 4}, "stop_distance": 50.0})
    res = run_backtest(strat, {"X": df}, 10_000, risk, zero_costs)
    assert res.halted
    # Position (100 units, all-in at 100) marked at 75 → equity 7500 = -25% → trip.
    assert len(res.trades) == 1
    assert res.trades[0].exit_reason == "kill switch"
    assert res.trades[0].exit_price == pytest.approx(75.0)  # next bar open
    # After the halt, the bar-4 signal must NOT create a trade.
    assert len(res.trades) == 1
    final_equity = res.equity_curve.iloc[-1]
    assert final_equity == pytest.approx(10_000 - 100 * 25.0)


def test_kill_switch_peak_tracks_highs(zero_costs):
    """Drawdown is measured from the PEAK, not the starting balance."""
    risk = RiskConfig(risk_per_trade=0.01, max_positions=2, kill_switch_drawdown=0.15)
    account = Account(cash=10_000)
    assert not account.check_kill_switch({"A": 0.0}, risk)
    account.cash = 13_000  # simulate gains → peak rises
    assert not account.check_kill_switch({}, risk)
    account.cash = 11_200  # only -13.8% from peak 13000
    assert not account.check_kill_switch({}, risk)
    account.cash = 11_000  # -15.4% from peak → trip
    assert account.check_kill_switch({}, risk)
    assert account.halted


def test_entry_rejected_when_halted(zero_costs, risk):
    account = Account(cash=10_000)
    account.halted = True
    intent = EntryIntent(stop_distance=10.0)
    assert account.open_long("A", "s", 0, 100.0, intent, {"A": 100.0}, risk, zero_costs) is None
