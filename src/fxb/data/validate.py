"""Candle schema and validation for FXB.

Standard candle DataFrame columns (this is THE schema everywhere in FXB)::

    ts_open      tz-aware UTC timestamp  -- the bar OPEN time (primary key)
    open, high,  float bid prices        -- BID side only
    low, close
    tick_volume  int64                   -- broker tick volume (0 if unknown)
    spread       float                   -- spread in POINTS at bar close

Ask price is never stored: ``ask = bid + spread * point_size`` where
``point_size = pip_size / 10`` (see :mod:`fxb.core.instruments`).

Only fully closed bars are ever validated/stored/returned: a bar is closed
when ``now_utc >= ts_open + timeframe_duration``.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from fxb.core.timeutil import NaiveDatetimeError, ensure_utc, timeframe_to_timedelta

#: Required columns of a standard candle frame (order matters for output).
CANDLE_COLUMNS: List[str] = [
    "ts_open", "open", "high", "low", "close", "tick_volume", "spread",
]

_PRICE_COLS = ("open", "high", "low", "close")


class CandleValidationError(ValueError):
    """Raised when a candle DataFrame violates the schema or sanity rules."""


def validate_candles(df: pd.DataFrame) -> None:
    """Validate ``df`` against the candle schema; raise on ANY violation.

    Checks: required columns present; ``ts_open`` tz-aware UTC; strictly
    increasing ``ts_open``; no duplicates; no NaN/inf anywhere; numeric
    prices > 0; ``high >= max(open, close)``; ``low <= min(open, close)``;
    ``spread >= 0``. An empty frame with the right columns is valid.
    """
    problems: List[str] = []

    missing = [c for c in CANDLE_COLUMNS if c not in df.columns]
    if missing:
        problems.append(f"missing columns: {missing}")
        raise CandleValidationError("; ".join(problems))

    if df["ts_open"].isna().any():
        problems.append("ts_open contains NaN")
    else:
        try:
            [ensure_utc(t) for t in df["ts_open"]]
        except NaiveDatetimeError as exc:
            problems.append(f"ts_open not tz-aware UTC: {exc}")

    ts = pd.DatetimeIndex(pd.to_datetime(df["ts_open"]))
    diffs = ts.to_series().diff().dropna()
    if (diffs <= pd.Timedelta(0)).any():
        problems.append("ts_open must be strictly increasing (no duplicates)")

    for col in list(_PRICE_COLS) + ["tick_volume", "spread"]:
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype="float64")
        if np.isnan(vals).any():
            problems.append(f"{col}: NaN values present")
        elif np.isinf(vals).any():
            problems.append(f"{col}: infinite values present")

    o = pd.to_numeric(df["open"], errors="coerce")
    h = pd.to_numeric(df["high"], errors="coerce")
    l = pd.to_numeric(df["low"], errors="coerce")
    c = pd.to_numeric(df["close"], errors="coerce")
    sp = pd.to_numeric(df["spread"], errors="coerce")

    if ((o <= 0) | (h <= 0) | (l <= 0) | (c <= 0)).fillna(False).any():
        problems.append("prices must be > 0")
    bad_high = (h < np.maximum(o, c)).fillna(False)
    if bad_high.any():
        problems.append(f"high < max(open, close) in {int(bad_high.sum())} row(s)")
    bad_low = (l > np.minimum(o, c)).fillna(False)
    if bad_low.any():
        problems.append(f"low > min(open, close) in {int(bad_low.sum())} row(s)")
    if (sp < 0).fillna(False).any():
        problems.append("spread must be >= 0")

    if problems:
        raise CandleValidationError("; ".join(problems))


def find_gaps(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Return a frame of expected-but-missing bar opens within the data span.

    Uses :func:`fxb.core.timeutil.expected_bar_times`, so weekend-closed
    periods are NOT flagged. Columns: ``ts_open`` (UTC), ``tf``. Empty frame
    (correct dtypes) when there are no gaps.
    """
    from fxb.core.timeutil import expected_bar_times

    validate_candles(df)
    out_cols = {"ts_open": pd.Series(pd.to_datetime([], utc=True)), "tf": pd.Series(dtype="object")}
    if len(df) == 0:
        return pd.DataFrame(out_cols)

    opens = pd.DatetimeIndex(pd.to_datetime(df["ts_open"])).tz_convert("UTC")
    dur = pd.Timedelta(timeframe_to_timedelta(tf))
    start = opens.min()
    end = opens.max() + dur  # last bar's close: it exists, but expect won't flag it
    expected = expected_bar_times(tf, start, end)
    have = set(opens)
    missing = [t for t in expected if t not in have]
    if not missing:
        return pd.DataFrame(out_cols)
    return pd.DataFrame({"ts_open": pd.DatetimeIndex(missing), "tf": [tf] * len(missing)})


def drop_unclosed_bar(df: pd.DataFrame, tf: str, now_utc: pd.Timestamp) -> pd.DataFrame:
    """Return ``df`` without its final bar if that bar has not closed yet.

    A bar is closed when ``now_utc >= ts_open + duration``. Only the LAST row
    is examined (providers may include exactly one forming bar).
    """
    if len(df) == 0:
        return df.copy()
    dur = timeframe_to_timedelta(tf)
    now = ensure_utc(now_utc)
    last_open = ensure_utc(df["ts_open"].iloc[-1])
    if last_open + pd.Timedelta(dur) > now:
        return df.iloc[:-1].copy()
    return df.copy()
