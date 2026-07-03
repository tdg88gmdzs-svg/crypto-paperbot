"""Market data layer: public OHLCV via ccxt, cached in SQLite.

Public endpoints only — no API keys exist anywhere in this project.

Two sources, on purpose:
- history_exchange (default binance): supports paginated multi-year history.
- live_exchange (default kraken): used by the paper loop for fresh candles.
  Kraken's public OHLC endpoint returns at most ~720 recent candles per
  timeframe, which is useless for backfill but fine for live updates.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import pandas as pd

log = logging.getLogger(__name__)

TIMEFRAME_MS: dict[str, int] = {
    "15m": 15 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS candles (
    symbol    TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    ts        INTEGER NOT NULL,  -- epoch ms, candle open time (UTC)
    open      REAL NOT NULL,
    high      REAL NOT NULL,
    low       REAL NOT NULL,
    close     REAL NOT NULL,
    volume    REAL NOT NULL,
    source    TEXT NOT NULL,
    PRIMARY KEY (symbol, timeframe, ts)
);
"""


class CandleStore:
    """SQLite-backed OHLCV cache keyed by (symbol, timeframe, open-time)."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def upsert(self, symbol: str, timeframe: str, rows: list[list[float]], source: str) -> int:
        """Insert or replace candles. `rows` are ccxt-style [ts, o, h, l, c, v]."""
        if not rows:
            return 0
        self._conn.executemany(
            "INSERT OR REPLACE INTO candles VALUES (?,?,?,?,?,?,?,?,?)",
            [(symbol, timeframe, int(r[0]), r[1], r[2], r[3], r[4], r[5], source) for r in rows],
        )
        self._conn.commit()
        return len(rows)

    def load(
        self,
        symbol: str,
        timeframe: str,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> pd.DataFrame:
        """Return candles as a DataFrame indexed by UTC timestamp, ascending."""
        query = "SELECT ts, open, high, low, close, volume FROM candles WHERE symbol=? AND timeframe=?"
        params: list[object] = [symbol, timeframe]
        if start_ms is not None:
            query += " AND ts >= ?"
            params.append(start_ms)
        if end_ms is not None:
            query += " AND ts < ?"
            params.append(end_ms)
        query += " ORDER BY ts"
        df = pd.read_sql_query(query, self._conn, params=params)
        if df.empty:
            return df
        df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df.set_index("timestamp")

    def latest_ts(self, symbol: str, timeframe: str) -> int | None:
        row = self._conn.execute(
            "SELECT MAX(ts) FROM candles WHERE symbol=? AND timeframe=?", (symbol, timeframe)
        ).fetchone()
        return row[0]

    def earliest_ts(self, symbol: str, timeframe: str) -> int | None:
        row = self._conn.execute(
            "SELECT MIN(ts) FROM candles WHERE symbol=? AND timeframe=?", (symbol, timeframe)
        ).fetchone()
        return row[0]


def make_exchange(name: str) -> ccxt.Exchange:
    """Instantiate a ccxt exchange for PUBLIC data only (no credentials)."""
    ex = getattr(ccxt, name)({"enableRateLimit": True})
    return ex


def fetch_ohlcv_paginated(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int | None = None,
    page_limit: int = 1000,
) -> list[list[float]]:
    """Fetch candles [since_ms, until_ms) by paging forward through history.

    Stops when the exchange stops advancing (end of available history) or
    the target end is reached. Drops the still-forming candle at the tail.
    """
    tf_ms = TIMEFRAME_MS[timeframe]
    now_ms = int(time.time() * 1000)
    end_ms = min(until_ms or now_ms, now_ms)
    # Never include the currently-forming candle.
    last_closed_open = ((now_ms // tf_ms) - 1) * tf_ms
    all_rows: list[list[float]] = []
    cursor = since_ms
    while cursor < end_ms:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=page_limit)
        if not batch:
            break
        batch = [r for r in batch if since_ms <= r[0] < end_ms and r[0] <= last_closed_open]
        if not batch:
            break
        all_rows.extend(batch)
        next_cursor = batch[-1][0] + tf_ms
        if next_cursor <= cursor:  # exchange not advancing; avoid infinite loop
            break
        cursor = next_cursor
    # Dedupe on timestamp, keep last.
    seen: dict[int, list[float]] = {int(r[0]): r for r in all_rows}
    return [seen[k] for k in sorted(seen)]


def ensure_history(
    store: CandleStore,
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime | None = None,
) -> pd.DataFrame:
    """Make sure the cache covers [start, end); download only what's missing.

    Returns the cached candles for the range. Gap detection is simple
    head/tail extension — good enough because we always download forward
    in contiguous pages.
    """
    start_ms = int(start.replace(tzinfo=start.tzinfo or timezone.utc).timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000) if end else int(time.time() * 1000)
    tf_ms = TIMEFRAME_MS[timeframe]

    have_first = store.earliest_ts(symbol, timeframe)
    have_last = store.latest_ts(symbol, timeframe)

    if have_first is None:
        log.info("cache empty for %s %s — downloading from %s", symbol, timeframe, start)
        rows = fetch_ohlcv_paginated(exchange, symbol, timeframe, start_ms, end_ms)
        store.upsert(symbol, timeframe, rows, exchange.id)
        log.info("downloaded %d candles for %s %s", len(rows), symbol, timeframe)
    else:
        if start_ms < have_first - tf_ms:
            rows = fetch_ohlcv_paginated(exchange, symbol, timeframe, start_ms, have_first)
            store.upsert(symbol, timeframe, rows, exchange.id)
            log.info("backfilled %d earlier candles for %s %s", len(rows), symbol, timeframe)
        if end_ms > have_last + tf_ms:
            rows = fetch_ohlcv_paginated(exchange, symbol, timeframe, have_last + tf_ms, end_ms)
            store.upsert(symbol, timeframe, rows, exchange.id)
            log.info("extended %d newer candles for %s %s", len(rows), symbol, timeframe)

    return store.load(symbol, timeframe, start_ms, end_ms)


def fetch_latest_closed(
    exchange: ccxt.Exchange, symbol: str, timeframe: str, lookback: int = 300
) -> list[list[float]]:
    """Fetch the most recent closed candles (for the live paper loop)."""
    tf_ms = TIMEFRAME_MS[timeframe]
    now_ms = int(time.time() * 1000)
    last_closed_open = ((now_ms // tf_ms) - 1) * tf_ms
    rows = exchange.fetch_ohlcv(symbol, timeframe, limit=lookback)
    return [r for r in rows if r[0] <= last_closed_open]
