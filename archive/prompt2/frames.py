"""OHLCV candle frame conventions shared across FXB.

Schema (a *candle frame*):
    ts_open : pd.Timestamp (tz-aware UTC)  -- bar OPEN time (index-like column)
    open, high, low, close : float         -- BID prices
    volume  : float                        -- tick volume
    spread  : float                        -- spread in POINTS at bar close

Helper ``last_closed_htf_index`` implements the prompt-1 rule: an HTF bar is
usable only once its close time has passed relative to the primary-timeframe
"now" (which is itself the close time of the last closed primary bar).
"""
from __future__ import annotations

import pandas as pd

from fxb.core.timeutil import bar_close_time, tf_to_timedelta, to_utc

CANDLE_COLUMNS = ["ts_open", "open", "high", "low", "close", "volume", "spread"]


def make_candle_frame(records) -> pd.DataFrame:
    """Build a validated candle frame from row dicts / tuples."""
    df = pd.DataFrame(list(records), columns=CANDLE_COLUMNS)
    df["ts_open"] = df["ts_open"].map(to_utc)
    return df.sort_values("ts_open").reset_index(drop=True)


def validate_candles(df: pd.DataFrame, timeframe: str) -> None:
    """Raise ValueError if the frame violates conventions (used by tests/providers)."""
    for col in CANDLE_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"candle frame missing column '{col}'")
    if df["ts_open"].dt.tz is None:
        raise ValueError("ts_open must be tz-aware UTC")
    if not df["ts_open"].is_monotonic_increasing:
        raise ValueError("ts_open must be sorted ascending")
    bad = (df["high"] < df[["open", "close", "low"]].max(axis=1)) | (
        df["low"] > df[["open", "close", "high"]].min(axis=1)
    )
    if bad.any():
        raise ValueError("OHLC invariant violated (high<max or low>min)")
    if (df["spread"] < 0).any():
        raise ValueError("spread (points) must be >= 0")


def last_closed_htf_index(htf_df: pd.DataFrame, htf_tf: str, now) -> int:
    """Index position of the LAST FULLY CLOSED HTF bar at or before ``now``.

    ``now`` is the close time of the last closed primary bar. An HTF bar with
    ts_open T covers [T, T+dur); it is closed iff T + dur <= now.

    Returns -1 when no HTF bar is closed yet.
    """
    if htf_df is None or len(htf_df) == 0:
        return -1
    t = to_utc(now)
    closes = htf_df["ts_open"] + tf_to_timedelta(htf_tf)
    mask = closes <= t
    if not bool(mask.any()):
        return -1
    return int(mask.values.nonzero()[0][-1])


def bar_slice_upto(df: pd.DataFrame, timeframe: str, now):
    """Boolean positions of bars in ``df`` that are fully closed at or before now."""
    closes = df["ts_open"].map(bar_close_time, timeframe=timeframe)
    return (closes <= to_utc(now)).values
