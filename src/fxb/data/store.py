"""SQLite candle store for FXB.

Schema::

    candles(symbol TEXT, timeframe TEXT, ts_open INTEGER,      -- UTC epoch SECONDS
            open REAL, high REAL, low REAL, close REAL,        -- BID prices
            tick_volume INTEGER, spread REAL)                  -- spread in POINTS

Price convention (global rule): stored OHLC are **bid** prices and each bar
carries its ``spread`` in points, so the ask side is reconstructible as
``ask = bid + spread * point_size`` with ``point_size = pip_size / 10``
(see :mod:`fxb.core.instruments`). Only fully closed bars are ever stored —
:data:`fxb.data.validate.CANDLE_COLUMNS` defines the frame schema and
:func:`fxb.data.validate.validate_candles` runs before every write.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from types import TracebackType
from typing import Optional

import pandas as pd

from fxb.core.logger import get_logger
from fxb.core.timeutil import ensure_utc
from fxb.data.validate import CANDLE_COLUMNS, validate_candles

log = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS candles (
    symbol      TEXT    NOT NULL,
    timeframe   TEXT    NOT NULL,
    ts_open     INTEGER NOT NULL,   -- UTC epoch seconds (bar OPEN time)
    open        REAL    NOT NULL,
    high        REAL    NOT NULL,
    low         REAL    NOT NULL,
    close       REAL    NOT NULL,
    tick_volume INTEGER NOT NULL,
    spread      REAL    NOT NULL,
    PRIMARY KEY (symbol, timeframe, ts_open)
);
CREATE INDEX IF NOT EXISTS ix_candles_lookup
    ON candles (symbol, timeframe, ts_open);
"""


class CandleStore:
    """Idempotent SQLite store of closed candles (context-manager supported)."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    def __enter__(self) -> "CandleStore":
        return self

    def __exit__(self, exc_type: Optional[type[BaseException]],
                 exc: Optional[BaseException], tb: Optional[TracebackType]) -> None:
        self.close()

    # -- writes ------------------------------------------------------------
    def upsert_candles(self, symbol: str, timeframe: str, df: pd.DataFrame) -> int:
        """Insert-or-replace closed candles; idempotent, single transaction.

        Validates ``df`` first (:class:`~fxb.data.validate.CandleValidationError`
        on failure). Returns the number of rows written.
        """
        validate_candles(df)
        if len(df) == 0:
            return 0
        sym = symbol.strip().upper()
        tf = timeframe.strip().upper()
        rows = []
        for rec in df.itertuples(index=False):
            ts = ensure_utc(rec.ts_open)
            rows.append((
                sym, tf, int(ts.timestamp()),
                float(rec.open), float(rec.high), float(rec.low), float(rec.close),
                int(rec.tick_volume), float(rec.spread),
            ))
        with self._conn:  # single transaction; rolls back on exception
            self._conn.executemany(
                "INSERT OR REPLACE INTO candles "
                "(symbol, timeframe, ts_open, open, high, low, close, tick_volume, spread) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        log.debug("upserted %d %s %s candles into %s", len(rows), sym, tf, self.db_path)
        return len(rows)

    def delete_range(self, symbol: str, timeframe: str,
                     start: Optional[pd.Timestamp] = None,
                     end: Optional[pd.Timestamp] = None) -> int:
        """Delete stored bars with ``start <= ts_open < end`` (None = unbounded)."""
        sym = symbol.strip().upper()
        tf = timeframe.strip().upper()
        sql = "DELETE FROM candles WHERE symbol = ? AND timeframe = ?"
        params: list = [sym, tf]
        if start is not None:
            sql += " AND ts_open >= ?"
            params.append(int(ensure_utc(start).timestamp()))
        if end is not None:
            sql += " AND ts_open < ?"
            params.append(int(ensure_utc(end).timestamp()))
        with self._conn:
            cur = self._conn.execute(sql, params)
        return cur.rowcount

    # -- reads -------------------------------------------------------------
    def get_candles(self, symbol: str, timeframe: str,
                    start: Optional[pd.Timestamp] = None,
                    end: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """Return candles indexed by UTC ``ts_open``, sorted ascending.

        ``start`` inclusive, ``end`` exclusive; either may be omitted. Columns
        follow :data:`fxb.data.validate.CANDLE_COLUMNS` (minus the index).
        """
        sym = symbol.strip().upper()
        tf = timeframe.strip().upper()
        sql = ("SELECT ts_open, open, high, low, close, tick_volume, spread "
               "FROM candles WHERE symbol = ? AND timeframe = ?")
        params: list = [sym, tf]
        if start is not None:
            sql += " AND ts_open >= ?"
            params.append(int(ensure_utc(start).timestamp()))
        if end is not None:
            sql += " AND ts_open < ?"
            params.append(int(ensure_utc(end).timestamp()))
        sql += " ORDER BY ts_open ASC"
        df = pd.read_sql_query(sql, self._conn, params=params)
        if len(df):
            df["ts_open"] = pd.to_datetime(df["ts_open"], unit="s", utc=True)
        else:
            df["ts_open"] = pd.to_datetime([], utc=True)
        df = df.set_index("ts_open").sort_index()
        df.index.name = "ts_open"
        return df[[c for c in CANDLE_COLUMNS if c != "ts_open"]]

    def last_ts(self, symbol: str, timeframe: str) -> Optional[pd.Timestamp]:
        """Open time of the newest stored bar, or ``None`` when empty."""
        row = self._conn.execute(
            "SELECT MAX(ts_open) FROM candles WHERE symbol = ? AND timeframe = ?",
            (symbol.strip().upper(), timeframe.strip().upper()),
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return pd.Timestamp(int(row[0]), unit="s", tz="UTC")

    def count(self, symbol: str, timeframe: str) -> int:
        """Number of stored bars for ``(symbol, timeframe)``."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM candles WHERE symbol = ? AND timeframe = ?",
            (symbol.strip().upper(), timeframe.strip().upper()),
        ).fetchone()
        return int(row[0])
