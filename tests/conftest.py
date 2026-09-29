"""Shared test fixtures: build standard-schema candle frames.

Also installs a pytest ``caplog`` default: FXB modules use per-module loggers
without propagate=False, so tests that assert on WARNING text need the root
handler enabled at WARNING level by default (individual tests may override).
"""
from __future__ import annotations

import logging
from typing import Iterable, List, Optional

import pandas as pd
import pytest

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


@pytest.fixture(autouse=True)
def _root_caplog_level(caplog):
    """Default caplog capture level: WARNING (tests can request lower)."""
    caplog.set_level(logging.WARNING)

# Make the tests directory importable as a package (``import helpers``) so
# test modules can share builders without duplicating them.
import os as _os
import sys as _sys

_TESTS_DIR = _os.path.dirname(_os.path.abspath(__file__))
if _TESTS_DIR not in _sys.path:
    _sys.path.insert(0, _TESTS_DIR)

from helpers import (  # noqa: E402,F401
    candles_from_ohlc,
    synth_gapped,
    synth_mixed,
    synth_range,
    synth_series,
    synth_trend,
    truncation_cut_points,
)
