"""Basic causal indicators: ATR, ADX, efficiency ratio, ATR percentile, displacement.

Look-ahead rule: every output column at row ``t`` uses only rows ``<= t``.
Wilder smoothing seeds at position ``n-1`` (NaN before). All functions return
a DataFrame with the same index as the input. Pure numpy/pandas; no TA-Lib.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fxb.indicators._util import reindex_like, true_range, wilder_smooth


def atr(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    """Wilder-smoothed Average True Range.

    Columns: ``atr``. NaN until ``n`` TR values exist (first value at row
    ``n - 1``, equal to the mean of the first ``n`` true ranges).
    """
    high = df["high"].to_numpy(dtype="float64")
    low = df["low"].to_numpy(dtype="float64")
    close = df["close"].to_numpy(dtype="float64")
    tr = true_range(high, low, close)
    out = pd.DataFrame({"atr": wilder_smooth(tr, n)}, index=df.index)
    return out


def adx(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    """Wilder ADX with +DI / -DI.

    Procedure (all steps causal):
      1. up-move   = high[t] - high[t-1]; down-move = low[t-1] - low[t].
         +DM = up-move when up-move > down-move and up-move > 0 else 0;
         -DM = down-move when down-move > up-move and down-move > 0 else 0.
      2. Wilder-smooth TR, +DM, -DM over ``n`` bars (seed = sum of first n),
         so the smoothed values start at row ``n - 1``.
      3. +DI = 100 * smooth(+DM)/smooth(TR); -DI mirrored. NaN before row
         ``n - 1``.
      4. DX  = 100 * |+DI - -DI| / (+DI + -DI) (0 when both DI are 0).
      5. ADX = Wilder mean-of-DX seeded with the average of the first ``n``
         DX values (which start at row ``n - 1``), so the first ADX appears
         at row ``2n - 2``.

    Columns: ``adx``, ``plus_di``, ``minus_di``.
    """
    high = df["high"].to_numpy(dtype="float64")
    low = df["low"].to_numpy(dtype="float64")
    close = df["close"].to_numpy(dtype="float64")
    m = len(df)
    tr = true_range(high, low, close)

    up = np.diff(high, prepend=high[:1])
    dn = -np.diff(low, prepend=low[:1])  # low[t-1] - low[t]
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    plus_dm[0] = 0.0
    minus_dm[0] = 0.0

    def _wilder_sums(values: np.ndarray) -> np.ndarray:
        """Wilder's original recursive sums: seed = sum(first n), NaN before."""
        out = np.full(m, np.nan)
        if m < n:
            return out
        acc = float(np.sum(values[:n]))
        out[n - 1] = acc
        for i in range(n, m):
            acc = acc - acc / n + float(values[i])
            out[i] = acc
        return out

    s_tr = _wilder_sums(tr)
    s_pdm = _wilder_sums(plus_dm)
    s_mdm = _wilder_sums(minus_dm)

    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = np.where(s_tr > 0, 100.0 * s_pdm / s_tr, np.nan)
        minus_di = np.where(s_tr > 0, 100.0 * s_mdm / s_tr, np.nan)
        di_sum = plus_di + minus_di
        dx = np.where(di_sum > 0, 100.0 * np.abs(plus_di - minus_di) / di_sum, 0.0)
    dx[~(s_tr > 0)] = np.nan  # undefined DI -> undefined DX

    adx_vals = np.full(m, np.nan)
    first_dx = n - 1               # first non-NaN DX position
    seed_pos = first_dx + n - 1    # first ADX position = 2n - 2
    if m >= seed_pos + 1:
        seed_dx = dx[first_dx:first_dx + n]
        if not np.all(np.isfinite(seed_dx)):
            # fall back: skip leading NaNs, still deterministic & causal
            valid = np.flatnonzero(np.isfinite(dx))
            if len(valid) >= n:
                first_dx = int(valid[0])
                seed_pos = first_dx + n - 1
                seed_dx = dx[first_dx:first_dx + n]
        if m >= seed_pos + 1 and np.all(np.isfinite(seed_dx)):
            acc = float(np.mean(seed_dx))
            adx_vals[seed_pos] = acc
            for i in range(seed_pos + 1, m):
                d = dx[i] if np.isfinite(dx[i]) else 0.0
                acc = (acc * (n - 1) + d) / n
                adx_vals[i] = acc

    out = pd.DataFrame(
        {"adx": adx_vals, "plus_di": plus_di, "minus_di": minus_di},
        index=df.index,
    )
    return out


def efficiency_ratio(close: pd.Series | pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Kaufman Efficiency Ratio: |net change| / sum of |bar-to-bar changes|.

    ``ER[t] = |c[t] - c[t-n]| / sum(|c[i] - c[i-1]|, i in t-n+1..t)``.
    NaN before row ``n``; 0 when the denominator is 0 (flat market).
    Accepts a close Series or a candle DataFrame. Column: ``er``.
    """
    if isinstance(close, pd.DataFrame):
        c = close["close"]
        index = close.index
    else:
        c = close
        index = close.index
    cv = c.to_numpy(dtype="float64")
    m = len(cv)
    net = np.full(m, np.nan)
    vol = np.full(m, np.nan)
    step = np.abs(np.diff(cv))          # step[i] = |c[i+1] - c[i]|
    if m > n:
        net[n:] = np.abs(cv[n:] - cv[:-n])
        # rolling sum of the last n steps ending at row t: steps t-n .. t-1
        csum = np.concatenate(([0.0], np.cumsum(step)))
        vol[n:] = csum[n:] - csum[:m - n]
    er = np.where(vol > 0, net / vol, 0.0)
    er[:n] = np.nan
    er[np.isnan(net)] = np.nan
    return pd.DataFrame({"er": er}, index=index)


def _rolling_percentile_rank(values: pd.Series, window: int) -> pd.Series:
    """Trailing-window percentile rank (<= value) via pandas, no look-ahead."""
    def _rank(arr: np.ndarray) -> float:
        v = arr[-1]
        if not np.isfinite(v):
            return np.nan
        hist = arr[np.isfinite(arr)]
        if len(hist) < 2:
            return np.nan
        return 100.0 * float(np.count_nonzero(hist <= v)) / len(hist)

    return values.rolling(window, min_periods=min(window, 20)).apply(
        _rank, raw=True
    )


def atr_percentile(df: pd.DataFrame, n: int = 14, window: int = 500) -> pd.DataFrame:
    """Rolling percentile rank of ATR within its trailing ``window`` history.

    ``atr_pct[t] = 100 * (# {ATR[s]} <= ATR[t]) / count`` over the non-NaN
    ATR values in rows ``t-window+1 .. t`` (inclusive, so only bars <= t).
    Documented warm-up rule: the value stays NaN until at least
    ``min(window, 20)`` valid ATR samples exist in the trailing window; once
    fewer than ``window`` samples have accumulated it is computed on what is
    available (so short histories still get a number after 20 ATR bars).

    Column: ``atr_pct`` (0..100).
    """
    vals = atr(df, n=n)["atr"]
    out = _rolling_percentile_rank(vals, window)
    return pd.DataFrame({"atr_pct": out}, index=df.index)


def displacement(df: pd.DataFrame, atr_mult: float = 1.5,
                 body_ratio: float = 0.6) -> pd.DataFrame:
    """Displacement candle flag: big body vs PRIOR bar's ATR, strong close.

    A candle at ``t`` is a displacement when ALL hold (only rows <= t used):
      - body = |close - open| >= ``atr_mult`` * ATR[t-1] (prior bar's ATR),
      - range = high - low > 0 and body / range >= ``body_ratio``,
      - direction from the candle itself: close > open -> +1 (bullish),
        close < open -> -1 (bearish); doji bodies never flag.
    Columns: ``displacement`` (bool), ``direction`` (+1 bullish / -1 bearish /
    0 none). NaN ATR before the seed -> no flags there.
    """
    o = df["open"].to_numpy(dtype="float64")
    h = df["high"].to_numpy(dtype="float64")
    l = df["low"].to_numpy(dtype="float64")
    c = df["close"].to_numpy(dtype="float64")
    a = atr(df)["atr"].to_numpy(dtype="float64")
    m = len(df)
    prior_atr = np.concatenate(([np.nan], a[:-1]))
    body = np.abs(c - o)
    rng = h - l
    with np.errstate(invalid="ignore"):
        big_body = body >= (atr_mult * prior_atr)
        strong = np.where(rng > 0, body / np.where(rng > 0, rng, 1.0), 0.0) >= body_ratio
        ok = np.isfinite(prior_atr) & big_body & strong & (rng > 0)
        bull = ok & (c > o)
        bear = ok & (c < o)
    direction = np.zeros(m, dtype="int64")
    direction[bear] = -1
    direction[bull] = 1
    flag = bull | bear
    return pd.DataFrame(
        {"displacement": flag, "direction": direction}, index=df.index
    )
