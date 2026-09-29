"""Market regime classification: TREND / RANGE / VOLATILE.

Built exclusively from the causal primitives in :mod:`fxb.indicators.basic`
(ADX, Kaufman efficiency ratio, ATR percentile), so row ``t`` of the output
depends only on bars ``<= t`` and truncation invariance holds wherever the
inputs themselves have values (during indicator warm-up every input is NaN
and the label falls through to ``cfg.below_warmup``, documented below).

No Hurst exponent (deliberately excluded — unreliable at bar scale).

Classification order (documented; first match wins):
  1. VOLATILE — ``atr_pct >= cfg.volatile_atr_pct`` (chop/expansion risk).
  2. TREND    — ``adx >= cfg.trend_adx`` AND ``er >= cfg.trend_er``.
  3. RANGE    — ``adx <= cfg.range_adx`` AND ``er <= cfg.range_er``.
  4. else     — ``cfg.below_warmup`` ("RANGE" by default) as the neutral
     fallback for mixed or warm-up states.

Pure numpy/pandas; no TA-Lib.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fxb.indicators._util import _as_index
from fxb.indicators.basic import adx, atr_percentile, efficiency_ratio


@dataclass(frozen=True)
class RegimeConfig:
    """Thresholds for :func:`regime` (all inclusive comparisons).

    Attributes
    ----------
    adx_n, er_n, atr_n, atr_window :
        Parameter sets forwarded to the underlying indicators.
    trend_adx :
        ADX level at/above which directional strength counts as trending
        (classic Wilder threshold, 25.0).
    trend_er :
        Efficiency-ratio floor (0..1) required together with ``trend_adx``
        for a TREND label (0.30: net move must be >= 30% of path length).
    range_adx :
        ADX ceiling for RANGE (20.0).
    range_er :
        ER ceiling for RANGE (0.20).
    volatile_atr_pct :
        Rolling-percentile rank (0..100) of ATR at/above which the market is
        labelled VOLATILE regardless of directionality (90.0).
    below_warmup :
        Label emitted while any needed input is still NaN (indicator warm-up)
        or when thresholds disagree. Default "RANGE" (neutral).
    """
    adx_n: int = 14
    er_n: int = 20
    atr_n: int = 14
    atr_window: int = 500
    trend_adx: float = 25.0
    trend_er: float = 0.30
    range_adx: float = 20.0
    range_er: float = 0.20
    volatile_atr_pct: float = 90.0
    below_warmup: str = "RANGE"


def regime(df: pd.DataFrame, cfg: RegimeConfig | None = None) -> pd.DataFrame:
    """Per-bar regime label plus its inputs, point-in-time.

    Same index as ``df``. Columns: ``regime`` ("TREND"/"RANGE"/"VOLATILE"),
    ``adx``, ``er``, ``atr_pct`` (the raw inputs used for the decision; NaN
    during warm-up).
    """
    if cfg is None:
        cfg = RegimeConfig()
    idx = _as_index(df)
    a = adx(df, n=cfg.adx_n)["adx"].to_numpy(dtype="float64")
    e = efficiency_ratio(df, n=cfg.er_n)["er"].to_numpy(dtype="float64")
    p = atr_percentile(df, n=cfg.atr_n, window=cfg.atr_window)["atr_pct"].to_numpy(dtype="float64")
    m = len(idx)
    labels = np.empty(m, dtype=object)
    for t in range(m):
        av, ev, pv = a[t], e[t], p[t]
        if np.isfinite(pv) and pv >= cfg.volatile_atr_pct:
            labels[t] = "VOLATILE"
        elif np.isfinite(av) and np.isfinite(ev) \
                and av >= cfg.trend_adx and ev >= cfg.trend_er:
            labels[t] = "TREND"
        elif np.isfinite(av) and np.isfinite(ev) \
                and av <= cfg.range_adx and ev <= cfg.range_er:
            labels[t] = "RANGE"
        else:
            labels[t] = cfg.below_warmup
    return pd.DataFrame(
        {"regime": labels, "adx": a, "er": e, "atr_pct": p}, index=df.index
    )
