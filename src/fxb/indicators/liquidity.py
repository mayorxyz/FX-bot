"""Session ranges, prior-day levels, equal highs/lows and liquidity sweeps.

All outputs obey the global look-ahead rule: per-bar frames are indexed like
the input and row ``t`` uses only bars ``<= t``; event tables stamp rows at
their CONFIRMATION bar and carry the original bar time separately.

Documented session definitions (UTC hours, coarse approximation — DST shifts
are intentionally NOT applied to these three boxes; see sessions.py for the
NY-wall-clock killzones):
  * ASIA   [00:00, 08:00) UTC
  * LONDON [07:00, 16:00) UTC
  * NEWYORK [12:00, 21:00) UTC
Sessions overlap by design (London/Asia 07-08, NY/London 12-16). A bar belongs
to a session when its OPEN time falls inside the box. "Completed" values for a
session on date D become available only from the first bar whose open time is
at or after the session's end hour on D (i.e. after every constituent bar of
D has closed).

Prior-day levels use the FXB daily anchor: a trading day runs 17:00 -> next
17:00 America/New_York (DST-safe via zoneinfo); the completed day's OHLC is
available only from the first bar opening at/after 17:00 NY.

Pure numpy/pandas; no TA-Lib.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from fxb.core.timeutil import _MARKET_TZ, ensure_utc
from fxb.indicators._util import _as_index
from fxb.indicators.basic import atr
from fxb.indicators.structure import swings

#: session name -> inclusive start hour / exclusive end hour (UTC)
SESSION_WINDOWS = {
    "asia": (0, 8),
    "london": (7, 16),
    "newyork": (12, 21),
}

#: Trading-day anchor in New York wall-clock time (open hour of the FX day).
DAILY_ANCHOR_NY_HOUR = 17


def _session_mask(idx: pd.DatetimeIndex, name: str) -> np.ndarray:
    lo, hi = SESSION_WINDOWS[name]
    hours = idx.hour.to_numpy()
    days = idx.normalize()
    end = days + pd.Timedelta(hours=hi)
    return (hours >= lo) & (hours < hi) & (idx < end) & (idx >= days + pd.Timedelta(hours=lo))


def session_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Developing and completed high/low per session, point-in-time per bar.

    Same index as ``df``. For each session S in {asia, london, newyork}:
      * ``S_dev_high`` / ``S_dev_low`` — max/min over all bars of S's CURRENT
        calendar-day window that opened at or before this bar (NaN until the
        first bar of that window appears).
      * ``S_high`` / ``S_low`` — the COMPLETED session of the most recent
        fully-past day, exposed from the first bar whose open time is at or
        after that session's end hour (bars <= t only).
    """
    idx = _as_index(df)
    h = df["high"].to_numpy(dtype="float64")
    l = df["low"].to_numpy(dtype="float64")
    m = len(idx)
    out: dict[str, np.ndarray] = {}
    for name in SESSION_WINDOWS:
        out[f"{name}_dev_high"] = np.full(m, np.nan)
        out[f"{name}_dev_low"] = np.full(m, np.nan)
        out[f"{name}_high"] = np.full(m, np.nan)
        out[f"{name}_low"] = np.full(m, np.nan)

    lo_np, hi_np = idx.hour.to_numpy(), None
    day_np = idx.normalize().to_numpy()
    # per-session accumulation across sorted bars
    state = {name: {"day": None, "dh": -np.inf, "dl": np.inf,
                    "ch": None, "cl": None} for name in SESSION_WINDOWS}
    for t in range(m):
        ts = idx[t]
        hour = ts.hour
        day = ts.normalize()
        for name, (lo, hi) in SESSION_WINDOWS.items():
            st = state[name]
            # roll into a new developing day window
            if st["day"] is None or day != st["day"]:
                # previous day's session, if it had data, becomes completed
                if st["day"] is not None and np.isfinite(st["dh"]):
                    st["ch"], st["cl"] = float(st["dh"]), float(st["dl"])
                st["day"] = day
                st["dh"], st["dl"] = -np.inf, np.inf
            if lo <= hour < hi:
                st["dh"] = max(st["dh"], float(h[t]))
                st["dl"] = min(st["dl"], float(l[t]))
            if np.isfinite(st["dh"]):
                out[f"{name}_dev_high"][t] = st["dh"]
                out[f"{name}_dev_low"][t] = st["dl"]
            # completed value is published once this bar is past the session
            # end on a later day than the stored completed session's day
            if st["ch"] is not None and (hour >= hi or day > st["day"]):
                # only publish when we are strictly beyond the window that
                # produced ch/cl: either same day after hi, or a newer day
                publish = (day == st["day"] and hour >= hi) or (day > st["day"])
                if publish:
                    out[f"{name}_high"][t] = st["ch"]
                    out[f"{name}_low"][t] = st["cl"]
    return pd.DataFrame(out, index=df.index)


def _fx_day_anchor(ts_utc: pd.Timestamp) -> pd.Timestamp:
    """Open time (UTC) of the FX trading day containing bar open ``ts_utc``.

    The day starting at 17:00 America/New_York covers [anchor, anchor+24h)
    except across DST transitions where it is 23/25 hours long; anchoring is
    done in NY wall-clock space so both cases fall out naturally.
    """
    ts = ensure_utc(ts_utc)
    ny = ts.tz_convert(_MARKET_TZ)
    if ny.hour >= DAILY_ANCHOR_NY_HOUR:
        anchor_ny = ny.normalize() + pd.Timedelta(hours=DAILY_ANCHOR_NY_HOUR)
    else:
        prev = (ny - pd.Timedelta(days=1)).normalize()
        anchor_ny = prev + pd.Timedelta(hours=DAILY_ANCHOR_NY_HOUR)
    return anchor_ny.tz_convert("UTC")


def prior_day_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Previous FX-trading-day HIGH/LOW/OPEN/CLOSE, available after its close.

    Days are anchored at 17:00 America/New_York (handles DST). A completed
    day's levels appear on every bar whose open time is at or after the day's
    anchor-close instant (next day 17:00 NY); during the day itself the
    columns are NaN — never the live partial day.

    Columns: ``pdh``, ``pdl``, ``pdo``, ``pdc``, ``prior_day_start_ts`` (the
    anchor-open of the day the levels describe).
    """
    idx = _as_index(df)
    o = df["open"].to_numpy(dtype="float64")
    h = df["high"].to_numpy(dtype="float64")
    l = df["low"].to_numpy(dtype="float64")
    c = df["close"].to_numpy(dtype="float64")
    m = len(idx)
    pdh = np.full(m, np.nan)
    pdl = np.full(m, np.nan)
    pdo = np.full(m, np.nan)
    pdc = np.full(m, np.nan)
    pds = pd.array([pd.NaT] * m, dtype="datetime64[ns, UTC]")

    # group bars by their FX day anchor
    anchors = [_fx_day_anchor(ts) for ts in idx]
    order = sorted(set(anchors))
    agg: dict[pd.Timestamp, dict] = {}
    for a in order:
        pass
    # single forward pass: complete days only (a day is complete once the
    # NEXT bar's anchor differs AND that next bar opens at/after its close)
    cur_a = anchors[0]
    cur_o = float(o[0]); cur_h = float(h[0]); cur_l = float(l[0]); cur_c = float(c[0])
    last_pub = None  # (levels tuple, anchor) available for publication
    day_close_of = lambda a: a + pd.Timedelta(days=1)  # nominal; refined below
    for t in range(m):
        a = anchors[t]
        if a != cur_a:
            # cur_a's day just ended at the previous bar; its close instant is
            # the next day's anchor (17:00 NY boundary, DST-correct by
            # construction since anchors are computed in NY wall clock).
            close_instant = a  # next anchor == this day's close instant
            last_pub = (cur_o, cur_h, cur_l, cur_c, cur_a, close_instant)
            cur_a, cur_o, cur_h, cur_l, cur_c = a, float(o[t]), float(h[t]), float(l[t]), float(c[t])
        else:
            cur_h = max(cur_h, float(h[t]))
            cur_l = min(cur_l, float(l[t]))
            cur_c = float(c[t])
        if last_pub is not None and idx[t] >= last_pub[5]:
            pdh[t], pdl[t], pdo[t], pdc[t] = last_pub[1], last_pub[2], last_pub[0], last_pub[3]
            pds[t] = last_pub[4]
    return pd.DataFrame(
        {"pdh": pdh, "pdl": pdl, "pdo": pdo, "pdc": pdc,
         "prior_day_start_ts": pds},
        index=df.index,
    )


def equal_highs_lows(df: pd.DataFrame, tolerance_atr: float = 0.1,
                     min_touches: int = 2, left: int = 3, right: int = 3) -> pd.DataFrame:
    """Clusters of confirmed swing highs/lows within ``tolerance_atr`` of ATR.

    Swings come from :func:`fxb.indicators.structure.swings` (already stamped
    at confirmation). Clustering is causal: at confirmation bar ``t`` only
    swings confirmed at-or-before ``t`` are considered; a level is emitted
    when the cluster reaches ``min_touches`` members (price tolerance =
    ``tolerance_atr * ATR[t]``). Each additional touch re-stamps the level on
    the touching swing's confirmation bar.

    Event-table columns: ``created_ts`` (confirmation stamp of the bar that
    formed/re-stamped the EQH/EQL), ``side`` ("EQH"/"EQL"), ``level``
    (cluster mean price), ``touches`` (count so far), ``touch_bar_ts_list``
    (list of original swing bar times). Truncation invariant: events depend
    only on confirmations at-or-before their stamp.
    """
    sw = swings(df, left=left, right=right)
    idx = _as_index(df)
    a = atr(df)["atr"].to_numpy(dtype="float64")
    m = len(idx)
    rows: List[dict] = []
    clusters: dict[str, dict] = {}
    for side, price_col, ts_col in (("EQH", "swing_high_price", "swing_high_bar_ts"),
                                    ("EQL", "swing_low_price", "swing_low_bar_ts")):
        clusters[side] = {"level": None, "touches": [], "last_stamp": None}
    for t in range(m):
        for side, price_col, ts_col in (("EQH", "swing_high_price", "swing_high_bar_ts"),
                                        ("EQL", "swing_low_price", "swing_low_bar_ts")):
            p = sw[price_col].iloc[t]
            if p != p:  # NaN
                continue
            clus = clusters[side]
            tol = tolerance_atr * a[t] if np.isfinite(a[t]) and a[t] > 0 else np.nan
            near = (clus["level"] is not None and np.isfinite(tol)
                    and abs(float(p) - clus["level"]) <= tol)
            if near:
                clus["touches"].append(idx[t])
                prices = [tp for tp in clus["touches"]]
                # mean of member prices stored alongside touches
                clus["prices"].append(float(p))
                clus["level"] = float(np.mean(clus["prices"]))
            else:
                clus["touches"] = [idx[t]]
                clus["prices"] = [float(p)]
                clus["level"] = float(p)
            if len(clus["touches"]) >= max(2, min_touches):
                rows.append({
                    "created_ts": idx[t],
                    "side": side,
                    "level": clus["level"],
                    "touches": len(clus["touches"]),
                    "touch_bar_ts_list": list(clus["touches"]),
                })
    cols = ["created_ts", "side", "level", "touches", "touch_bar_ts_list"]
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values("created_ts").reset_index(drop=True)


def sweeps(df: pd.DataFrame, levels: pd.DataFrame) -> pd.DataFrame:
    """Stop-run events: bar pierces a level but CLOSE returns inside.

    Parameters
    ----------
    levels : DataFrame with a ``level`` column (e.g. built from
        :func:`equal_highs_lows` output or session-completed highs/lows).
        Optional ``side`` column ("HIGH"-type level swept upward, "LOW"-type
        swept downward); defaults to testing both directions per level.

    Semantics (documented, causal): an UP-sweep of level L at bar ``t`` needs
    ``high[t] > L`` and ``close[t] <= L``; if the close does not return on
    ``t`` but does on ``t+1`` (``close[t+1] <= L`` and no earlier close was
    already below L after the pierce), the event is stamped at ``t+1``.
    DOWN-sweep mirrors it (``low[t] < L`` then close back ``>= L``). Piercing
    alone (close stays beyond) is a break, not a sweep, and is ignored.

    Output event table: ``sweep_ts`` (stamp bar time), ``swept_level``,
    ``side`` ("UP"/"DOWN"), ``pierced_ts`` (original bar time of the pierce).
    """
    idx = _as_index(df)
    h = df["high"].to_numpy(dtype="float64")
    l = df["low"].to_numpy(dtype="float64")
    c = df["close"].to_numpy(dtype="float64")
    m = len(idx)
    if "level" not in levels.columns:
        raise ValueError("levels frame must contain a 'level' column")
    sides = (levels["side"].astype(str).tolist()
             if "side" in levels.columns else [None] * len(levels))
    rows: List[dict] = []
    for lv, want_side in zip(levels["level"].astype(float).tolist(), sides):
        test_up = want_side in (None, "HIGH", "EQH", "UP")
        test_dn = want_side in (None, "LOW", "EQL", "DOWN")
        if test_up:
            t = 0
            while t < m:
                if h[t] > lv:
                    # find first close back inside at t or t+1
                    if c[t] <= lv:
                        rows.append({"sweep_ts": idx[t], "swept_level": lv,
                                     "side": "UP", "pierced_ts": idx[t]})
                        t += 1
                        continue
                    if t + 1 < m and c[t + 1] <= lv and c[t] > lv:
                        rows.append({"sweep_ts": idx[t + 1], "swept_level": lv,
                                     "side": "UP", "pierced_ts": idx[t]})
                        t += 2
                        continue
                    # close stayed outside: skip ahead past this excursion
                    j = t
                    while j < m and c[j] > lv:
                        j += 1
                    t = j
                    continue
                t += 1
        if test_dn:
            t = 0
            while t < m:
                if l[t] < lv:
                    if c[t] >= lv:
                        rows.append({"sweep_ts": idx[t], "swept_level": lv,
                                     "side": "DOWN", "pierced_ts": idx[t]})
                        t += 1
                        continue
                    if t + 1 < m and c[t + 1] >= lv and c[t] < lv:
                        rows.append({"sweep_ts": idx[t + 1], "swept_level": lv,
                                     "side": "DOWN", "pierced_ts": idx[t]})
                        t += 2
                        continue
                    j = t
                    while j < m and c[j] < lv:
                        j += 1
                    t = j
                    continue
                t += 1
    cols = ["sweep_ts", "swept_level", "side", "pierced_ts"]
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values("sweep_ts").reset_index(drop=True)
