"""Shared test fixtures: build standard-schema candle frames."""
from __future__ import annotations

from typing import Iterable, List, Optional

import pandas as pd

COLUMNS = ["ts_open", "open", "high", "low", "close", "tick_volume", "spread"]


def make_candles(opens: Iterable[pd.Timestamp], o=1.0, h=1.1, l=0.9, c=1.05,
                 vol: int = 10, spread: float = 2.0) -> pd.DataFrame:
    """Build a valid candle frame from an iterable of UTC-aware bar opens."""
    ts = pd.DatetimeIndex(pd.to_datetime(list(opens))).tz_convert("UTC")
    n = len(ts)
    return pd.DataFrame({
        "ts_open": ts,
        "open": [float(o)] * n,
        "high": [float(h)] * n,
        "low": [float(l)] * n,
        "close": [float(c)] * n,
        "tick_volume": [int(vol)] * n,
        "spread": [float(spread)] * n,
    }, columns=COLUMNS)


def grid(tf_minutes: int, start: str, n: int) -> List[pd.Timestamp]:
    """``n`` consecutive bar opens ``tf_minutes`` apart starting at ``start`` (UTC)."""
    s = pd.Timestamp(start, tz="UTC")
    return [s + pd.Timedelta(minutes=tf_minutes * i) for i in range(n)]
