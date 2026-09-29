"""Fair Value Gaps (3-candle imbalances) with point-in-time state tracking.

Definition (documented): using bars ``i-2, i-1, i`` (all known at bar ``i``'s
close — no future data):
  * BULLISH FVG when ``low[i] > high[i-2]`` -> gap between ``high[i-2]``
    (bottom) and ``low[i]`` (top).
  * BEARISH FVG when ``high[i] < low[i-2]`` -> gap between ``high[i]``
    (bottom) and ``low[i-2]`` (top).
The event is stamped at bar ``i`` (its close), which is knowable.

Mitigation semantics used by :func:`fvg_state` (bars <= t only):
  * ``mitigated_ts``   — first bar whose trade touches INTO the gap
    (bullish: ``low <= top``; bearish: ``high >= bottom``), strictly after
    creation. Once touched the gap is considered MITIGATED (no longer active).
  * ``fully_filled_ts`` — first bar whose CLOSE crosses the far edge
    (bullish: ``close <= bottom``; bearish: ``close >= top``).
  * ``invalidated_ts`` — first bar whose CLOSE closes beyond the ORIGIN edge
    against the gap direction (bullish: ``close > top`` before any touch —
    price jumped over the gap; bearish: ``close < bottom``); informational.

Pure numpy/pandas; no TA-Lib; same-index outputs for state frames.
"""
from __future__ import annotations

import bisect
from typing import List, Optional

import numpy as np
import pandas as pd

from fxb.indicators._util import _as_index
from fxb.indicators.basic import atr, displacement


def fair_value_gaps(df: pd.DataFrame, min_size_atr: float = 0.1) -> pd.DataFrame:
    """Detect 3-candle imbalances; return an EVENT table (not a per-bar frame).

    Columns: ``created_ts`` (bar ``i`` open time = confirmation stamp),
    ``direction`` (+1 bullish / -1 bearish), ``top``, ``bottom``,
    ``midpoint``, ``size`` (price units), ``size_atr`` (size / ATR at bar
    ``i``, NaN while ATR has no value yet), ``displacement_flag`` (True when
    the MIDDLE candle ``i-1`` is a displacement candle).

    Filter: keep events with ``size >= min_size_atr * ATR[i]``. While ATR is
    still NaN (warm-up) the size filter cannot run and the event IS kept
    (documented choice: never silently drop gaps during warm-up; downstream
    code can require ``size_atr``).
    """
    idx = _as_index(df)
    h = df["high"].to_numpy(dtype="float64")
    l = df["low"].to_numpy(dtype="float64")
    a = atr(df)["atr"].to_numpy(dtype="float64")
    disp = displacement(df)["displacement"].to_numpy(dtype=bool)
    m = len(idx)
    rows: List[dict] = []
    for i in range(2, m):
        direction = 0
        top = bottom = np.nan
        if l[i] > h[i - 2]:          # bullish imbalance
            direction, top, bottom = 1, float(l[i]), float(h[i - 2])
        elif h[i] < l[i - 2]:        # bearish imbalance
            direction, top, bottom = -1, float(l[i - 2]), float(h[i])
        if direction == 0:
            continue
        size = top - bottom
        size_atr = size / a[i] if np.isfinite(a[i]) and a[i] > 0 else np.nan
        if np.isfinite(size_atr):
            if size_atr < min_size_atr:
                continue
        # ATR not available yet -> keep (documented warm-up behaviour)
        rows.append(
            {
                "created_ts": idx[i],
                "direction": direction,
                "top": top,
                "bottom": bottom,
                "midpoint": 0.5 * (top + bottom),
                "size": size,
                "size_atr": size_atr,
                "displacement_flag": bool(disp[i - 1]),
            }
        )
    cols = ["created_ts", "direction", "top", "bottom", "midpoint",
            "size", "size_atr", "displacement_flag"]
    if not rows:
        return pd.DataFrame(columns=cols)
    ev = pd.DataFrame(rows, columns=cols)
    ev["created_idx"] = [int(idx.get_loc(ts)) for ts in ev["created_ts"]]
    return ev.reset_index(drop=True)


class FvgState:
    """Per-bar snapshot of which FVGs are ACTIVE (unmitigated) at bar ``t``."""

    def __init__(self, t: int, ts_open: pd.Timestamp, active_ids: List[int],
                 mitigated_ids: List[int], filled_ids: List[int]):
        self.t = t
        self.ts_open = ts_open
        #: ids (row labels of the events table) of gaps created and untouched.
        self.active_ids = active_ids
        self.mitigated_ids = mitigated_ids      # touched into the gap so far
        self.fully_filled_ids = filled_ids      # closed through the far edge

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"FvgState(t={self.t}, ts_open={self.ts_open}, "
                f"active={len(self.active_ids)}, mitigated={len(self.mitigated_ids)}, "
                f"filled={len(self.fully_filled_ids)})")


def fvg_state(df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Per-bar FVG state computed with bars ``<= t`` only.

    Same index as ``df``. Columns:
      * ``n_active`` — number of unmitigated gaps alive at bar ``t``
        (created at some bar ``c <= t`` and not yet touched into on any bar
        ``c+1 .. t``),
      * ``n_mitigated`` / ``n_filled`` — cumulative counts of gaps that have
        been touched / fully filled up to ``t``,
      * ``active_top`` / ``active_bottom`` — zone bounds of the most recently
        created still-active gap (NaN when none),
      * ``has_active_bull`` / ``has_active_bear`` — booleans.
    """
    idx = _as_index(df)
    h = df["high"].to_numpy(dtype="float64")
    l = df["low"].to_numpy(dtype="float64")
    c = df["close"].to_numpy(dtype="float64")
    m = len(idx)

    n_active = np.zeros(m, dtype="int64")
    n_mit = np.zeros(m, dtype="int64")
    n_fill = np.zeros(m, dtype="int64")
    act_top = np.full(m, np.nan)
    act_bot = np.full(m, np.nan)
    has_bull = np.zeros(m, dtype=bool)
    has_bear = np.zeros(m, dtype=bool)

    events = events if len(events) else pd.DataFrame(columns=["created_idx"])
    created_by_bar: dict[int, list] = {}
    for _, r in events.iterrows():
        created_by_bar.setdefault(int(r["created_idx"]), []).append(r)

    live: list[dict] = []      # dicts with id/dir/top/bottom
    mit_count = 0
    fill_count = 0
    filled_ids = set()
    for t in range(m):
        for r in created_by_bar.get(t, []):
            live.append({"id": int(r.name) if isinstance(r.name, int) else None,
                         "dir": int(r["direction"]), "top": float(r["top"]),
                         "bottom": float(r["bottom"])})
        still_live = []
        for g in live:
            if g["dir"] == 1:
                touched = l[t] <= g["top"]
                filled = c[t] <= g["bottom"]
            else:
                touched = h[t] >= g["bottom"]
                filled = c[t] >= g["top"]
            if touched and g["id"] not in filled_ids:
                mit_count += 1
                if filled:
                    filled_ids.add(g["id"])
                    fill_count += 1
                continue  # mitigated -> no longer active
            if filled and g["id"] not in filled_ids:
                # close crossed the far edge without a recorded touch is
                # impossible (fill implies touch); defensive branch anyway
                mit_count += 1
                filled_ids.add(g["id"])
                fill_count += 1
                continue
            still_live.append(g)
        live = still_live
        n_active[t] = len(live)
        n_mit[t] = mit_count
        n_fill[t] = fill_count
        if live:
            g = live[-1]
            act_top[t] = g["top"]
            act_bot[t] = g["bottom"]
            has_bull[t] = g["dir"] == 1
            has_bear[t] = g["dir"] == -1
    return pd.DataFrame(
        {
            "n_active": n_active,
            "n_mitigated": n_mit,
            "n_filled": n_fill,
            "active_top": act_top,
            "active_bottom": act_bot,
            "has_active_bull": has_bull,
            "has_active_bear": has_bear,
        },
        index=df.index,
    )


def active_fvgs_at(df: pd.DataFrame, events: pd.DataFrame, t) -> List[FvgState]:
    """FVG status at bar ``t`` using ONLY bars ``<= t``.

    ``t`` may be an integer position or a timestamp present in ``df``.
    Returns a single-element list with the :class:`FvgState` snapshot plus a
    detailed ``gaps`` list (each gap as a dict with id/direction/top/bottom/
    midpoint). Kept as a list for API symmetry with multi-snapshot callers.
    """
    idx = _as_index(df)
    if isinstance(t, (int, np.integer)):
        pos = int(t)
    else:
        pos = int(idx.get_loc(pd.Timestamp(t)))
    if pos < 0 or pos >= len(idx):
        raise IndexError(f"bar position {pos} out of range")
    st_df = fvg_state(df.iloc[: pos + 1], events)
    st = st_df.iloc[pos]
    # rebuild the exact live-gap details from events (causal replay)
    h = df["high"].to_numpy(dtype="float64")
    l = df["low"].to_numpy(dtype="float64")
    c = df["close"].to_numpy(dtype="float64")
    gaps = []
    for eid, r in events.iterrows():
        ci = int(r["created_idx"])
        if ci > pos:
            continue
        d = int(r["direction"])
        top, bot = float(r["top"]), float(r["bottom"])
        touched = False
        for k in range(ci + 1, pos + 1):
            if (d == 1 and l[k] <= top) or (d == -1 and h[k] >= bot):
                touched = True
                break
        if not touched:
            gaps.append({"id": int(eid), "direction": d, "top": top,
                         "bottom": bot, "midpoint": 0.5 * (top + bot)})
    snap = FvgState(pos, idx[pos], [g["id"] for g in gaps], [], [])
    snap.gaps = gaps  # type: ignore[attr-defined]
    return [snap]
