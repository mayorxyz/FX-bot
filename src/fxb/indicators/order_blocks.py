"""Order blocks derived from displacement legs that break structure (BOS/CHoCH).

Look-ahead rule: an order block only becomes KNOWABLE when the displacement
leg it belongs to actually breaks structure, so every event is stamped at the
BOS/CHoCH CONFIRMATION bar; ``ob_bar_ts`` carries the original candle time.
Before confirmation there is no row at all (the event table simply has no
entry with ``created_ts > t``), and :func:`active_obs_at` replays only bars
``<= t``. Consequence: outputs are truncation-invariant —
``order_blocks(df.iloc[:k], ...) == order_blocks(df, ...)[created_ts <= k-th]``.

Definitions (documented):
  * BULLISH OB = the last DOWN candle (close < open) immediately before the
    first bullish displacement candle of a leg whose close finally breaks the
    last confirmed swing HIGH upward (BOS or CHoCH per ``structure``).
  * BEARISH OB mirrors it (last UP candle before a bearish displacement leg
    closing below the last confirmed swing LOW).
  * Zone bounds default to the OB candle BODY (max/min(open, close));
    ``use_wicks=True`` uses high/low instead.
  * MITIGATED when price later trades INTO the zone (bullish: low <= top;
    bearish: high >= bottom). INVALIDATED when a CLOSE goes beyond the far
    side (bullish: close < bottom; bearish: close > top). A mitigated block is
    retired from the active set; invalidation may still be recorded later.

Pure numpy/pandas; no TA-Lib. Follows the conventions of fvg.py / structure.py.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from fxb.indicators._util import _as_index
from fxb.indicators.basic import displacement


def order_blocks(df: pd.DataFrame, structure: pd.DataFrame, lookback: int = 10,
                 atr_mult: float = 1.5, body_ratio: float = 0.6,
                 use_wicks: bool = False) -> pd.DataFrame:
    """Detect order blocks; return an EVENT table (like fair_value_gaps).

    Parameters
    ----------
    df : standard candle frame (UTC DatetimeIndex ``ts_open``, bid OHLC).
    structure : output of :func:`fxb.indicators.structure.market_structure`
        on the same frame (columns ``bos``, ``choch``, ``break_direction``).
    lookback : how many bars before the break to search for the displacement
        leg's first candle (default 10).
    atr_mult, body_ratio : passed through to :func:`displacement`.
    use_wicks : zone from the OB candle's high/low instead of its body.

    Output columns (one row per event, in creation order):
    ``created_ts`` (stamp = BOS/CHoCH confirmation bar open time),
    ``ob_bar_ts`` (original OB candle open time), ``direction`` (+1/-1),
    ``top``, ``bottom``, ``mitigated_idx``, ``mitigated_ts``,
    ``invalidated_idx``, ``invalidated_ts``. The ``*_idx`` columns are integer
    positions into ``df`` (NaN until the condition occurs); they make the
    replay in :func:`active_obs_at` exact and cheap.
    """
    idx = _as_index(df)
    o = df["open"].to_numpy(dtype="float64")
    h = df["high"].to_numpy(dtype="float64")
    l = df["low"].to_numpy(dtype="float64")
    c = df["close"].to_numpy(dtype="float64")
    m = len(idx)

    disp = displacement(df, atr_mult=atr_mult, body_ratio=body_ratio)
    dflag = disp["displacement"].to_numpy(dtype=bool)
    ddir = disp["direction"].to_numpy(dtype="int64")

    for col in ("bos", "choch", "break_direction"):
        if col not in structure.columns:
            raise ValueError(f"structure frame missing required column {col!r}")
    event = (structure["bos"].astype(bool) | structure["choch"].astype(bool))
    ev = event.to_numpy(dtype=bool)
    bdir = structure["break_direction"].to_numpy(dtype="int64")

    rows: List[dict] = []
    used_legs: set = set()
    for t in range(m):
        if not ev[t] or bdir[t] == 0:
            continue
        direction = int(bdir[t])
        # find the leg: scan back from t to the first displacement candle of
        # this direction; keep the EARLIEST consecutive one (start of move)
        leg_start: Optional[int] = None
        j = t
        while j >= max(0, t - lookback):
            if dflag[j] and ddir[j] == direction:
                leg_start = j
            elif dflag[j]:
                break  # opposite-direction displacement ends the leg
            j -= 1
        if leg_start is None:
            continue
        if leg_start in used_legs:
            continue  # one OB per displacement leg
        # OB candle: last candle BEFORE the leg with an opposing body
        ob_i: Optional[int] = None
        k = leg_start - 1
        while k >= max(0, leg_start - 1 - lookback):
            if direction == 1 and c[k] < o[k]:
                ob_i = k
                break
            if direction == -1 and c[k] > o[k]:
                ob_i = k
                break
            k -= 1
        if ob_i is None:
            continue
        used_legs.add(leg_start)
        if use_wicks:
            top, bottom = float(h[ob_i]), float(l[ob_i])
        else:
            top = float(max(o[ob_i], c[ob_i]))
            bottom = float(min(o[ob_i], c[ob_i]))
        # forward pass for mitigation / invalidation using bars AFTER stamp t
        mit_i: Optional[int] = None
        inv_i: Optional[int] = None
        for s in range(t + 1, m):
            if mit_i is None:
                touched = (l[s] <= top) if direction == 1 else (h[s] >= bottom)
                if touched:
                    mit_i = s
            if inv_i is None:
                broken = (c[s] < bottom) if direction == 1 else (c[s] > top)
                if broken:
                    inv_i = s
            if mit_i is not None and inv_i is not None:
                break
        rows.append({
            "created_ts": idx[t],
            "ob_bar_ts": idx[ob_i],
            "direction": direction,
            "top": top,
            "bottom": bottom,
            "mitigated_idx": mit_i if mit_i is not None else np.nan,
            "mitigated_ts": idx[mit_i] if mit_i is not None else pd.NaT,
            "invalidated_idx": inv_i if inv_i is not None else np.nan,
            "invalidated_ts": idx[inv_i] if inv_i is not None else pd.NaT,
        })

    cols = ["created_ts", "ob_bar_ts", "direction", "top", "bottom",
            "mitigated_idx", "mitigated_ts", "invalidated_idx", "invalidated_ts"]
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows, columns=cols)
    return out.reset_index(drop=True)


def active_obs_at(events: pd.DataFrame, df: pd.DataFrame, t) -> List[dict]:
    """Order blocks ACTIVE (unmitigated, uninvalidated) at bar ``t``.

    Uses only bars ``<= t``: an event exists once its ``created_ts`` position
    is ``<= t``; it leaves the active set from its mitigation bar onward (or
    from its invalidation bar if that came first). Returns a list of dicts
    with ``id, direction, top, bottom, midpoint, created_ts, ob_bar_ts``.
    """
    idx = _as_index(df)
    if isinstance(t, (int, np.integer)):
        pos = int(t)
    else:
        pos = int(idx.get_loc(pd.Timestamp(t)))
    if pos < 0 or pos >= len(idx):
        raise IndexError(f"bar position {pos} out of range")
    active: List[dict] = []
    if events is None or len(events) == 0:
        return active
    for eid, r in events.iterrows():
        created = idx.get_loc(pd.Timestamp(r["created_ts"]))
        if created > pos:
            continue
        mit = r["mitigated_idx"]
        inv = r["invalidated_idx"]
        retire = np.inf
        if pd.notna(mit):
            retire = min(retire, int(mit))
        if pd.notna(inv):
            retire = min(retire, int(inv))
        if pos >= retire:
            continue
        active.append({
            "id": int(eid),
            "direction": int(r["direction"]),
            "top": float(r["top"]),
            "bottom": float(r["bottom"]),
            "midpoint": 0.5 * (float(r["top"]) + float(r["bottom"])),
            "created_ts": r["created_ts"],
            "ob_bar_ts": r["ob_bar_ts"],
        })
    return active
