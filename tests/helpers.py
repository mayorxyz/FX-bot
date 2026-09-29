"""Shared helpers for indicator tests: synthetic candle frames.

All builders return standard-schema frames: UTC DatetimeIndex ``ts_open``
(bar OPEN time), columns open/high/low/close/tick_volume/spread, bid prices.
Series are deterministic given the seed so truncation-invariance failures
are reproducible.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import pytest


def candles_from_ohlc(rows, start="2024-03-04 00:00", tf_minutes=15) -> pd.DataFrame:
    """Build a frame from (open, high, low, close) tuples on a UTC grid."""
    s = pd.Timestamp(start, tz="UTC")
    idx = pd.DatetimeIndex([s + pd.Timedelta(minutes=tf_minutes * i)
                            for i in range(len(rows))], name="ts_open")
    arr = np.asarray(rows, dtype="float64")
    df = pd.DataFrame(
        {"open": arr[:, 0], "high": arr[:, 1], "low": arr[:, 2],
         "close": arr[:, 3], "tick_volume": 10.0, "spread": 2.0},
        index=idx,
    )
    return df


def synth_series(n: int = 600, kind: str = "mixed", seed: int = 7) -> pd.DataFrame:
    """Deterministic synthetic price series with realistic OHLC bars.

    kind:
      * "trend"   — steady drift up with noise.
      * "range"   — mean-reverting oscillation around a level.
      * "gapped"  — mostly smooth with seeded jump gaps (displacements).
      * "mixed"   — trend segment, then range, then gapped bursts.
    Bars never violate high >= max(o,c), low <= min(o,c); prices stay > 0.
    """
    rng = np.random.default_rng(seed)
    if kind == "trend":
        segs = [("trend", n)]
    elif kind == "range":
        segs = [("range", n)]
    elif kind == "gapped":
        segs = [("gapped", n)]
    elif kind == "mixed":
        a, b = n // 3, n // 3
        segs = [("trend", a), ("range", b), ("gapped", n - a - b)]
    else:
        raise ValueError(kind)
    base = 1.10
    rows = []
    prev_close = base
    t = 0
    for kind_name, length in segs:
        for _ in range(length):
            if kind_name == "trend":
                step = 0.0004 + rng.normal(0.0, 0.0002)
            elif kind_name == "range":
                step = -0.15 * (prev_close - base) * 0.001 + rng.normal(0.0, 0.00025)
            else:  # gapped
                step = rng.normal(0.0, 0.0002)
                if rng.random() < 0.02:
                    step += rng.choice([-1.0, 1.0]) * rng.uniform(0.0015, 0.003)
            o = prev_close
            c = o + step
            hi_wick = abs(rng.normal(0.0, 0.00015))
            lo_wick = abs(rng.normal(0.0, 0.00015))
            h = max(o, c) + hi_wick
            l = min(o, c) - lo_wick
            rows.append((o, h, l, c))
            prev_close = c
            t += 1
    df = candles_from_ohlc(rows, start="2024-03-04 00:00", tf_minutes=15)
    assert df["low"].min() > 0
    return df


CUT_POINTS_KW = dict(n=600)


def truncation_cut_points(m: int, k: int = 20, seed: int = 11) -> List[int]:
    """Sorted unique random cut lengths in [warmup, m] for invariance tests."""
    rng = np.random.default_rng(seed)
    lo = min(80, m // 4)
    pts = set(int(x) for x in rng.integers(lo, m + 1, size=k * 3))
    out = sorted(pts)[:k]
    if m not in out:
        out.append(m)
    return out


@pytest.fixture(scope="session")
def synth_mixed() -> pd.DataFrame:
    return synth_series(600, "mixed", seed=7)


@pytest.fixture(scope="session")
def synth_trend() -> pd.DataFrame:
    return synth_series(600, "trend", seed=8)


@pytest.fixture(scope="session")
def synth_range() -> pd.DataFrame:
    return synth_series(600, "range", seed=9)


@pytest.fixture(scope="session")
def synth_gapped() -> pd.DataFrame:
    return synth_series(600, "gapped", seed=10)
