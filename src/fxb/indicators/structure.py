"""Fractal swings and market structure (BOS / CHoCH), strictly point-in-time.

Look-ahead rule: a swing at bar ``i`` is only KNOWABLE after ``right`` further
bars exist, so it is stamped at its CONFIRMATION row ``i + right``; the
original bar time travels in ``swing_*_bar_ts``. Before confirmation every
value is NaN. Consequently all structure outputs satisfy truncation
invariance: ``f(df.iloc[:k]) == f(df).iloc[:k]``.

Break convention (documented): BOS/CHoCH breaks are CLOSE-based — a break is
registered when a bar's CLOSE crosses the level of the most recent CONFIRMED
swing in the relevant direction. Wicks alone never break structure.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from fxb.indicators._util import _as_index


def swings(df: pd.DataFrame, left: int = 3, right: int = 3) -> pd.DataFrame:
    """Fractal swing highs/lows stamped at their confirmation bar.

    A swing HIGH sits at bar ``i`` when ``high[i]`` is strictly greater than
    the ``left`` preceding bars' highs AND strictly greater than the ``right``
    following bars' highs (strict > on both sides; equal-height neighbours do
    not form a swing). It becomes knowable at bar ``i + right`` and is
    therefore written to the output ONLY on that confirmation row, together
    with ``swing_high_bar_ts`` naming the original bar. Swing lows mirror it.

    Columns: ``swing_high_price``, ``swing_high_bar_ts``, ``swing_low_price``,
    ``swing_low_bar_ts``. All NaN before confirmation. Same index as input.
    """
    idx = _as_index(df)
    h = df["high"].to_numpy(dtype="float64")
    l = df["low"].to_numpy(dtype="float64")
    m = len(idx)
    sh_price: List[Optional[float]] = [np.nan] * m
    sh_ts: List[Optional[pd.Timestamp]] = [None] * m
    sl_price: List[Optional[float]] = [np.nan] * m
    sl_ts: List[Optional[pd.Timestamp]] = [None] * m
    for i in range(left, m - right):
        window_l = slice(i - left, i)
        window_r = slice(i + 1, i + 1 + right)
        if h[i] > h[window_l].max() and h[i] > h[window_r].max():
            conf = i + right
            sh_price[conf] = float(h[i])
            sh_ts[conf] = idx[i]
        if l[i] < l[window_l].min() and l[i] < l[window_r].min():
            conf = i + right
            sl_price[conf] = float(l[i])
            sl_ts[conf] = idx[i]
    out = pd.DataFrame(
        {
            "swing_high_price": sh_price,
            "swing_high_bar_ts": pd.array(sh_ts, dtype="datetime64[ns, UTC]"),
            "swing_low_price": sl_price,
            "swing_low_bar_ts": pd.array(sl_ts, dtype="datetime64[ns, UTC]"),
        },
        index=df.index,
    )
    return out


def market_structure(df: pd.DataFrame, left: int = 3, right: int = 3) -> pd.DataFrame:
    """Trend state with Break Of Structure (BOS) and Change Of Character (CHoCH).

    Algorithm (causal; each decision at bar ``t`` sees only bars <= t):
      * Maintain the last CONFIRMED swing high / swing low (from
        :func:`swings`, applied on rows <= t, plus their prices and ts).
      * Trend state starts at 0 (no confirmed swings yet). It flips to +1 on
        the first bullish break, -1 on the first bearish break.
      * Bullish break: ``close[t] > last_swing_high``. If trend was +1 this
        is a BOS (continuation); if trend was -1 it is a CHoCH/MSS (and the
        trend then flips to +1); from 0 it establishes +1 (counted as BOS).
      * Bearish break mirrors it against ``last_swing_low``.
      * Only one event per bar; if both sides break on the same close the
        direction matching the PREVIOUS trend wins (the opposing side would
        be its own CHoCH next time). Documented tie-break.
      * After a bullish break the swing-high reference resets to the newest
        confirmed swing high formed at-or-before the break bar (may be none
        until a new one confirms); same mirrored for lows. The opposite-side
        level is never reset by an aligned break.

    Columns (per-bar state): ``trend`` (+1/-1/0), ``bos`` (bool), ``choch``
    (bool), ``broken_level`` (price of the swing broken at this bar, else
    NaN), ``break_direction`` (+1/-1/0 at this bar), ``last_swing_high``,
    ``last_swing_low`` (prices of the latest confirmed swings known so far).
    """
    sw = swings(df, left=left, right=right)
    idx = _as_index(df)
    close = df["close"].to_numpy(dtype="float64")
    m = len(df)

    trend = np.zeros(m, dtype="int64")
    bos = np.zeros(m, dtype=bool)
    choch = np.zeros(m, dtype=bool)
    broken_level = np.full(m, np.nan)
    break_dir = np.zeros(m, dtype="int64")
    last_sh = np.full(m, np.nan)
    last_sl = np.full(m, np.nan)

    cur_sh: Optional[float] = None
    cur_sl: Optional[float] = None
    tr = 0
    # pending resets: set to True when the corresponding side just broke,
    # meaning the level should drop to the NEXT confirmed swing of that side
    reset_h = False
    reset_l = False

    for t in range(m):
        p_h = sw["swing_high_price"].iloc[t]
        p_l = sw["swing_low_price"].iloc[t]
        has_h = not (p_h != p_h)  # NaN check without pandas dependency quirks
        has_l = not (p_l != p_l)
        if has_h:
            if reset_h:
                cur_sh = float(p_h)   # first confirmed high after the break
                reset_h = False
            elif cur_sh is None or float(p_h) >= cur_sh:
                # keep the most recent confirmed swing high; higher price also
                # promotes when no reset pending (levels only ever come from
                # confirmed swings)
                cur_sh = float(p_h)
            else:
                cur_sh = float(p_h)   # most recent confirmed wins
        if has_l:
            if reset_l:
                cur_sl = float(p_l)
                reset_l = False
            else:
                cur_sl = float(p_l)

        c = close[t]
        event_dir = 0
        level = np.nan
        up_break = cur_sh is not None and c > cur_sh
        dn_break = cur_sl is not None and c < cur_sl
        if up_break and dn_break:
            # Tie-break: honor the side aligned with the previous trend.
            event_dir = 1 if tr >= 0 else -1
        elif up_break:
            event_dir = 1
        elif dn_break:
            event_dir = -1

        if event_dir == 1:
            level = float(cur_sh)
            if tr == -1:
                choch[t] = True
                tr = 1
            else:
                bos[t] = True
                tr = 1
            reset_h = True
        elif event_dir == -1:
            level = float(cur_sl)
            if tr == 1:
                choch[t] = True
                tr = -1
            else:
                bos[t] = True
                tr = -1
            reset_l = True

        trend[t] = tr
        break_dir[t] = event_dir
        broken_level[t] = level
        last_sh[t] = cur_sh if cur_sh is not None else np.nan
        last_sl[t] = cur_sl if cur_sl is not None else np.nan

    return pd.DataFrame(
        {
            "trend": trend,
            "bos": bos,
            "choch": choch,
            "broken_level": broken_level,
            "break_direction": break_dir,
            "last_swing_high": last_sh,
            "last_swing_low": last_sl,
        },
        index=df.index,
    )
