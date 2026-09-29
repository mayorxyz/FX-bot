"""Premium / discount classification of a price within a range.

Pure helper module (no bar iteration), so the global look-ahead rule is
trivially satisfied: :func:`pd_zone` only sees the numbers you hand it, and
:func:`last_confirmed_swing_range` derives its range exclusively from swings
already CONFIRMED at or before the queried bar (structure.swings stamps are
respected).

Definitions (documented):
  * midpoint = 50% of [range_low, range_high].
  * PREMIUM   — price strictly above the midpoint.
  * DISCOUNT  — price strictly below the midpoint.
  * EQUILIBRIUM — price equal to the midpoint within ``tol`` (absolute price
    units; default 0.0 means exact equality only).
  * Degenerate/invalid ranges (high <= low, NaN) -> "UNKNOWN".
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from fxb.indicators._util import _as_index
from fxb.indicators.structure import swings

PREMIUM = "PREMIUM"
DISCOUNT = "DISCOUNT"
EQUILIBRIUM = "EQUILIBRIUM"
UNKNOWN = "UNKNOWN"


def pd_zone(price: float, range_low: float, range_high: float,
            tol: float = 0.0) -> str:
    """Classify ``price`` against the 50% midpoint of [range_low, range_high]."""
    vals = np.asarray([price, range_low, range_high], dtype="float64")
    if not np.all(np.isfinite(vals)):
        return UNKNOWN
    if range_high <= range_low:
        return UNKNOWN
    mid = 0.5 * (range_low + range_high)
    if abs(price - mid) <= tol:
        return EQUILIBRIUM
    return PREMIUM if price > mid else DISCOUNT


def last_confirmed_swing_range(df: pd.DataFrame, t, left: int = 3,
                               right: int = 3) -> Optional[dict]:
    """Range of the most recent confirmed swing PAIR known at bar ``t``.

    Scans :func:`fxb.indicators.structure.swings` confirmations up to bar
    ``t`` (inclusive) and returns the pair formed by the latest confirmed
    swing high and the latest confirmed swing low whose ORIGINAL bars bound a
    completed leg (low-high or high-low ordering irrelevant; we take min/max
    as the range). Returns ``None`` while fewer than two opposite-side
    confirmed swings exist.

    Result dict: ``{"range_low", "range_high", "midpoint", "high_ts",
    "low_ts"}`` where ``high_ts``/``low_ts`` are the ORIGINAL swing bar times
    (not the confirmation stamps). Uses only bars <= t.
    """
    idx = _as_index(df)
    if isinstance(t, (int, np.integer)):
        pos = int(t)
    else:
        pos = int(idx.get_loc(pd.Timestamp(t)))
    if pos < 0 or pos >= len(idx):
        raise IndexError(f"bar position {pos} out of range")
    sw = swings(df, left=left, right=right)
    hi_price = hi_ts = lo_price = lo_ts = None
    for k in range(pos + 1):
        ph = sw["swing_high_price"].iloc[k]
        pl = sw["swing_low_price"].iloc[k]
        if ph == ph:  # non-NaN
            hi_price = float(ph)
            hi_ts = sw["swing_high_bar_ts"].iloc[k]
        if pl == pl:
            lo_price = float(pl)
            lo_ts = sw["swing_low_bar_ts"].iloc[k]
    if hi_price is None or lo_price is None:
        return None
    low = min(hi_price, lo_price)
    high = max(hi_price, lo_price)
    if high <= low:
        return None
    return {
        "range_low": low,
        "range_high": high,
        "midpoint": 0.5 * (low + high),
        "high_ts": hi_ts,
        "low_ts": lo_ts,
    }
