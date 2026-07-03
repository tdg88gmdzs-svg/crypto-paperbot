"""SQLite persistence for the paper account.

Everything the bot needs to survive a restart lives here: cash, open
positions, pending orders, the full trade log, and the equity curve.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from paperbot.execution import Account, PendingOrder, Position, Trade
from paperbot.strategies.base import EntryIntent

_SCHEMA = """
CREATE TABLE IF NOT EXISTS account (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    cash REAL NOT NULL,
    peak_equity REAL NOT NULL,
    halted INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS positions (
    key TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    strategy TEXT NOT NULL,
    qty REAL NOT NULL,
    entry_price REAL NOT NULL,
    entry_ts INTEGER NOT NULL,
    stop_price REAL NOT NULL,
    take_profit REAL,
    entry_fee REAL NOT NULL,
    highest_close REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS pending_orders (
    symbol TEXT NOT NULL,
    strategy TEXT NOT NULL,
    side TEXT NOT NULL,
    stop_distance REAL,
    take_profit_distance REAL,
    reason TEXT
);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    strategy TEXT NOT NULL,
    qty REAL NOT NULL,
    entry_ts INTEGER NOT NULL,
    entry_price REAL NOT NULL,
    exit_ts INTEGER NOT NULL,
    exit_price REAL NOT NULL,
    fees REAL NOT NULL,
    pnl REAL NOT NULL,
    exit_reason TEXT
);
CREATE TABLE IF NOT EXISTS equity_curve (
    ts INTEGER PRIMARY KEY,
    equity REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS cursor (
    symbol TEXT PRIMARY KEY,
    last_ts INTEGER NOT NULL
);
"""


class PaperState:
    """Load/save the paper Account and its history."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ---- account -------------------------------------------------------

    def load_account(self, starting_balance: float) -> tuple[Account, list[PendingOrder]]:
        """Load the persisted account, initializing it on first run."""
        row = self._conn.execute("SELECT cash, peak_equity, halted FROM account WHERE id=1").fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO account (id, cash, peak_equity, halted) VALUES (1, ?, ?, 0)",
                (starting_balance, starting_balance),
            )
            self._conn.commit()
            return Account(cash=starting_balance), []

        account = Account(cash=row[0], peak_equity=row[1], halted=bool(row[2]))
        for p in self._conn.execute(
            "SELECT symbol, strategy, qty, entry_price, entry_ts, stop_price,"
            " take_profit, entry_fee, highest_close FROM positions"
        ):
            pos = Position(
                symbol=p[0], strategy=p[1], qty=p[2], entry_price=p[3], entry_ts=p[4],
                stop_price=p[5], take_profit=p[6], entry_fee=p[7], highest_close=p[8],
            )
            account.positions[Account.key(pos.symbol, pos.strategy)] = pos
        for t in self._conn.execute(
            "SELECT symbol, strategy, qty, entry_ts, entry_price, exit_ts, exit_price,"
            " fees, pnl, exit_reason FROM trades ORDER BY id"
        ):
            account.trades.append(Trade(*t))

        pending = [
            PendingOrder(
                symbol=o[0],
                strategy=o[1],
                side=o[2],
                intent=EntryIntent(stop_distance=o[3], take_profit_distance=o[4])
                if o[2] == "enter"
                else None,
                reason=o[5] or "",
            )
            for o in self._conn.execute(
                "SELECT symbol, strategy, side, stop_distance, take_profit_distance,"
                " reason FROM pending_orders"
            )
        ]
        return account, pending

    def save(self, account: Account, pending: list[PendingOrder]) -> None:
        """Persist current account snapshot atomically."""
        with self._conn:
            self._conn.execute(
                "UPDATE account SET cash=?, peak_equity=?, halted=? WHERE id=1",
                (account.cash, account.peak_equity, int(account.halted)),
            )
            self._conn.execute("DELETE FROM positions")
            self._conn.executemany(
                "INSERT INTO positions VALUES (?,?,?,?,?,?,?,?,?,?)",
                [
                    (k, p.symbol, p.strategy, p.qty, p.entry_price, p.entry_ts,
                     p.stop_price, p.take_profit, p.entry_fee, p.highest_close)
                    for k, p in account.positions.items()
                ],
            )
            self._conn.execute("DELETE FROM pending_orders")
            self._conn.executemany(
                "INSERT INTO pending_orders VALUES (?,?,?,?,?,?)",
                [
                    (o.symbol, o.strategy, o.side,
                     o.intent.stop_distance if o.intent else None,
                     o.intent.take_profit_distance if o.intent else None,
                     o.reason)
                    for o in pending
                ],
            )

    def append_trade(self, t: Trade) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO trades (symbol, strategy, qty, entry_ts, entry_price,"
                " exit_ts, exit_price, fees, pnl, exit_reason) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (t.symbol, t.strategy, t.qty, t.entry_ts, t.entry_price,
                 t.exit_ts, t.exit_price, t.fees, t.pnl, t.exit_reason),
            )

    def append_equity(self, ts_ms: int, equity: float) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO equity_curve VALUES (?, ?)", (ts_ms, equity)
            )

    def get_cursor(self, symbol: str) -> int | None:
        """Open-time (ms) of the last candle processed for `symbol`."""
        row = self._conn.execute(
            "SELECT last_ts FROM cursor WHERE symbol=?", (symbol,)
        ).fetchone()
        return row[0] if row else None

    def set_cursor(self, symbol: str, ts_ms: int) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO cursor VALUES (?, ?)", (symbol, ts_ms)
            )

    # ---- reads for reporting -------------------------------------------

    def equity_curve(self) -> pd.Series:
        df = pd.read_sql_query("SELECT ts, equity FROM equity_curve ORDER BY ts", self._conn)
        if df.empty:
            return pd.Series(dtype=float)
        idx = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return pd.Series(df["equity"].values, index=idx, name="equity")

    def recent_trades(self, n: int = 20) -> list[Trade]:
        rows = self._conn.execute(
            "SELECT symbol, strategy, qty, entry_ts, entry_price, exit_ts, exit_price,"
            " fees, pnl, exit_reason FROM trades ORDER BY id DESC LIMIT ?",
            (n,),
        ).fetchall()
        return [Trade(*r) for r in rows]
