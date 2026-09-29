"""Shared causal helpers for the indicator modules.

All helpers here obey the global look-ahead rule: the value produced for row
``t`` depends only on rows ``<= t`` of the input frame. Candle frames use the
standard schema (UTC DatetimeIndex ``ts_open``, bid prices, ``tick_volume``,
``spread`` in points). Pure numpy/pandas; no TA-Lib.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def reindex_like(out: pd.DataFrame, like: pd.DataFrame) -> pd.DataFrame:
    """Align ``out`` to the exact index of ``like`` (missing rows become NaN).

    Guarantees every indicator returns a frame with the SAME index as its
    input, which is what makes truncation-invariance tests meaningful.
    """
    return out.reindex(like.index)


def _as_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Bar-open timestamps as a UTC DatetimeIndex, from index or column."""
    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
    else:
        idx = pd.DatetimeIndex(pd.to_datetime(df["ts_open"]))
    if idx.tz is None:
        raise ValueError("indicator input must carry tz-aware UTC timestamps")
    return idx


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Elementwise True Range using ONLY current and prior bars (causal).

    ``TR[0] = high[0] - low[0]`` (no previous close exists);
    ``TR[t] = max(h-l, |h-c[t-1]|, |l-c[t-1]|)``.
    """
    tr = np.empty(len(high), dtype="float64")
    if len(tr) == 0:
        return tr
    tr[0] = high[0] - low[0]
    prev_close = close[:-1]
    hl = high[1:] - low[1:]
    hc = np.abs(high[1:] - prev_close)
    lc = np.abs(low[1:] - prev_close)
    tr[1:] = np.maximum(np.maximum(hl, hc), lc)
    return tr


def wilder_smooth(values: np.ndarray, n: int) -> np.ndarray:
    """Wilder's smoothing (RMA) of ``values`` with period ``n``.

    The first value appears at position ``n - 1`` and equals the simple mean
    of the first ``n`` inputs; afterwards ``S[t] = (S[t-1]*(n-1) + v[t]) / n``.
    Positions before ``n - 1`` are NaN. Fully causal: ``S[t]`` uses inputs
    up to and including ``v[t]`` only.
    """
    out = np.full(len(values), np.nan, dtype="float64")
    if len(values) < n or n <= 0:
        return out
    seed = float(np.mean(values[:n]))
    out[n - 1] = seed
    for i in range(n, len(values)):
        out[i] = (out[i - 1] * (n - 1) + float(values[i])) / n
    return out
