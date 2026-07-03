"""Async paper-trading loop and one-shot tick.

Two ways to run the same engine:
- `run_paper_loop`: long-running local process that wakes shortly after
  each candle closes.
- `run_paper_once`: process any newly closed candles and exit. Used by
  scheduled runners (e.g. an hourly GitHub Actions job) where the process
  is ephemeral but the state DB persists.

Both are restart-safe: the per-symbol cursor of the last processed candle
lives in the state DB, so missed bars are replayed in order on the next
run, and a fresh candle cache never causes a history replay.
"""

from __future__ import annotations

import asyncio
import logging
import time

from paperbot.config import Config
from paperbot.data import TIMEFRAME_MS, CandleStore, fetch_latest_closed, make_exchange
from paperbot.paper.engine import PaperEngine
from paperbot.paper.state import PaperState
from paperbot.strategies import STRATEGIES
from paperbot.strategies.base import Strategy

log = logging.getLogger(__name__)

# Kraken serves at most ~720 candles; take them all so indicators
# (EMA 200 etc.) have enough history even on a fresh cache.
LIVE_LOOKBACK = 720


def build_strategies(cfg: Config, names: list[str]) -> list[Strategy]:
    """Instantiate strategies with their default config.yaml parameters."""
    out: list[Strategy] = []
    for name in names:
        cls = STRATEGIES[name]
        out.append(cls(cls.default_params(cfg.strategies[name])))
    return out


def process_new_candles(cfg: Config, engine: PaperEngine, exchange, store: CandleStore) -> int:
    """Fetch fresh closed candles and run the engine over any unseen ones.

    Returns the number of bars processed. Synchronous: callers in async
    contexts wrap it in a thread.
    """
    tf = cfg.data.timeframe
    tf_ms = TIMEFRAME_MS[tf]
    processed = 0
    for symbol in cfg.data.symbols:
        rows = fetch_latest_closed(exchange, symbol, tf, LIVE_LOOKBACK)
        if not rows:
            log.warning("%s: no candles returned", symbol)
            continue
        store.upsert(symbol, tf, rows, exchange.id)
        cursor = engine.state.get_cursor(symbol)
        if cursor is None:
            # First run ever for this symbol: start from the newest closed
            # bar instead of replaying whatever history is in the cache.
            cursor = rows[-2][0] if len(rows) >= 2 else 0
            log.info("%s: initializing cursor at %d (no history replay)", symbol, cursor)
        new_ts = sorted(r[0] for r in rows if r[0] > cursor)
        for ts in new_ts:
            history = store.load(symbol, tf, end_ms=ts + tf_ms)
            engine.process_bar(symbol, history)
            engine.state.set_cursor(symbol, ts)
            processed += 1
        if new_ts:
            log.info(
                "%s: processed %d new bar(s), equity $%.2f",
                symbol, len(new_ts), engine.account.equity(engine.last_marks),
            )
    return processed


def _make_parts(cfg: Config, strategy_names: list[str]):
    exchange = make_exchange(cfg.data.live_exchange)
    store = CandleStore(cfg.data.cache_db)
    state = PaperState(cfg.paper.state_db)
    engine = PaperEngine(cfg, build_strategies(cfg, strategy_names), state)
    return exchange, store, engine


def run_paper_once(cfg: Config, strategy_names: list[str]) -> int:
    """One tick: process new candles, persist state, exit. For schedulers."""
    exchange, store, engine = _make_parts(cfg, strategy_names)
    if engine.account.halted:
        log.critical("account is HALTED by the kill switch; nothing to do. "
                     "Delete %s to reset the paper account.", cfg.paper.state_db)
        return 0
    n = process_new_candles(cfg, engine, exchange, store)
    if engine.last_marks:
        log.info("\n%s", engine.daily_summary())
    return n


async def run_paper_loop(cfg: Config, strategy_names: list[str]) -> None:
    """Main unattended loop. Ctrl-C to stop; state persists."""
    tf = cfg.data.timeframe
    tf_ms = TIMEFRAME_MS[tf]
    exchange, store, engine = _make_parts(cfg, strategy_names)

    log.info(
        "paper loop starting: %s on %s (%s), equity state in %s",
        ", ".join(strategy_names), ", ".join(cfg.data.symbols), tf, cfg.paper.state_db,
    )
    if engine.account.halted:
        log.critical("account is HALTED by the kill switch; refusing to trade. "
                     "Delete %s to reset the paper account.", cfg.paper.state_db)
        return

    last_summary_day: int = -1
    while True:
        try:
            await asyncio.to_thread(process_new_candles, cfg, engine, exchange, store)

            if engine.account.halted:
                log.critical("halted by kill switch — exiting paper loop.")
                return

            # Daily summary once per UTC day.
            day = int(time.time() // 86400)
            if day != last_summary_day and engine.last_marks:
                log.info("\n%s", engine.daily_summary())
                last_summary_day = day

        except Exception:  # noqa: BLE001 — keep the loop alive on API hiccups
            log.exception("error in paper loop; retrying next candle")

        # Sleep until just after the next candle close.
        now_ms = int(time.time() * 1000)
        next_close = (now_ms // tf_ms + 1) * tf_ms
        sleep_s = (next_close - now_ms) / 1000 + cfg.paper.candle_grace_seconds
        log.debug("sleeping %.0fs until next %s candle", sleep_s, tf)
        await asyncio.sleep(sleep_s)
