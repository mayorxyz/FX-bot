"""Aggregate lower-timeframe candles into higher-timeframe candles.

Rules (global):
- ``open`` = first, ``high`` = max, ``low`` = min, ``close`` = last,
  ``tick_volume`` = sum, ``spread`` = last of the constituent LTF bars.
- **A partial HTF bar is NEVER emitted**: an HTF bucket is returned only when
  every expected LTF bar inside it (per :func:`fxb.core.timeutil.expected_bar_times`,
  i.e. excluding the weekend-closed period) is present in the input.
- Bucket grid ("anchor"):
    * H1 / H4 / D1 with a fixed UTC hour anchor -> epoch-aligned buckets
      (H4 at 00/04/08/12/16/20 UTC by default).
    * D1 anchored to a wall-clock time in a tz database zone, e.g. the default
      ``"17:00 America/New_York"`` — buckets run from one NY-17:00 to the next,
      so the UTC boundary shifts automatically across DST changes.
- ``allow_missing_frac`` (default 0.0 = strict): a bucket is emitted when at
  most ``allow_missing_frac`` of its expected LTF bars are missing; with the
  default any missing bar blocks emission and a WARNING is logged with the
  window start and the missing-bar count. Buckets with ZERO expected bars
  (fully weekend-closed windows) are skipped silently.
"""
from __future__ import annotations

import logging
import re
from datetime import timedelta
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

from fxb.core.config import load_settings
from fxb.core.timeutil import ensure_utc, expected_bar_times, timeframe_to_timedelta
from fxb.data.validate import CANDLE_COLUMNS, validate_candles

log = logging.getLogger(__name__)

_ANCHOR_RE = re.compile(r"^(\d{1,2}):(\d{2})\s+(.+)$")


def parse_anchor(anchor: Optional[str]) -> Tuple[int, Optional[ZoneInfo]]:
    """Parse an anchor string into ``(hour_of_day_utc_or_local, zone_or_None)``.

    ``"4 UTC"`` / ``"04:00 UTC"`` -> epoch-aligned buckets offset by 4 h UTC;
    ``"20 UTC"`` -> daily buckets at 20:00 UTC;
    ``"17:00 America/New_York"`` -> wall-clock anchored buckets.
    ``None`` resolves the configured default ``D1_ANCHOR``.
    """
    if anchor is None:
        anchor = load_settings().d1_anchor
    text = anchor.strip()
    m = _ANCHOR_RE.match(text)
    if not m:
        bare = re.match(r"^(\d{1,2})\s+UTC$", text, flags=re.IGNORECASE)
        if not bare:
            raise ValueError(
                f"invalid anchor {anchor!r}; expected e.g. '4 UTC' or '17:00 America/New_York'"
            )
        hour, minute = int(bare.group(1)), 0
        if hour > 23:
            raise ValueError(f"anchor hour out of range: {hour}")
        return hour % 24, None
    hour, minute, zone = int(m.group(1)), int(m.group(2)), m.group(3).strip()
    if minute != 0:
        raise ValueError("only whole-hour anchors are supported")
    if hour > 23:
        raise ValueError(f"anchor hour out of range: {hour}")
    if zone.upper() == "UTC":
        return hour % 24, None
    return hour, ZoneInfo(zone)


def _fixed_offset_hours(target_tf: str) -> int:
    """Grid offset (hours UTC) for intraday targets: H4 aligns to 00/04/08/..."""
    tf = target_tf.strip().upper()
    fixed = {"M5": 0, "M15": 0, "M30": 0, "H1": 0, "H4": 4}
    if tf not in fixed:
        raise ValueError(f"unsupported target timeframe {target_tf!r}")
    return fixed[tf]


def _bucket_opens_fixed(start: pd.Timestamp, end: pd.Timestamp,
                        tf_dur: timedelta, hour: int) -> List[pd.Timestamp]:
    """Epoch-aligned HTF opens covering ``[start, end)`` with grid offset ``hour`` UTC."""
    step_s = int(tf_dur.total_seconds())
    offset_s = hour * 3600
    t0 = int(start.timestamp())
    # first grid point at or before start (so the bucket containing start is included)
    first = ((t0 - offset_s) // step_s) * step_s + offset_s
    opens: List[pd.Timestamp] = []
    t = first
    while t < int(end.timestamp()):
        opens.append(pd.Timestamp(t, unit="s", tz="UTC"))
        t += step_s
    return opens


def _bucket_opens_zone(start: pd.Timestamp, end: pd.Timestamp,
                       tf_dur: timedelta, hour: int, zone: ZoneInfo) -> List[pd.Timestamp]:
    """Wall-clock-anchored daily opens: iterate local dates, convert each
    ``hour:00`` in ``zone`` to UTC (handles DST via fold=0, earliest occurrence)."""
    local_start = start.tz_convert(zone).normalize()
    local_end = end.tz_convert(zone).normalize() + pd.Timedelta(days=3)
    opens: List[pd.Timestamp] = []
    day = local_start - pd.Timedelta(days=2)
    while day <= local_end:
        naive = day.replace(hour=hour).tz_localize(None)  # plain wall-clock value
        aware = pd.Timestamp(naive).tz_localize(zone, ambiguous=False, nonexistent="shift_forward")
        opens.append(aware.tz_convert("UTC"))
        day = day + pd.Timedelta(days=1)
    idx = [int(o.timestamp()) for o in opens]
    lo, hi = int(start.timestamp()), int(end.timestamp())
    keep = []
    for i, s in enumerate(idx):
        # include bucket whose open is before hi and close after lo
        if s < hi and s + int(tf_dur.total_seconds()) > lo:
            keep.append(opens[i])
    return keep


def resample_ohlc(df_lower: pd.DataFrame, target_tf: str,
                  anchor: Optional[str] = None,
                  allow_missing_frac: float = 0.0) -> pd.DataFrame:
    """Resample closed LTF candles (standard schema) into closed HTF candles.

    ``target_tf`` supports M5/M15/M30/H1/H4/D1. Incomplete buckets (any
    expected LTF bar missing, e.g. a truncated final bucket or data gaps) are
    dropped entirely — a partial HTF bar is never returned. Raises
    :class:`~fxb.data.validate.CandleValidationError` on invalid input.

    ``anchor`` selects the bucket grid: intraday targets default to the UTC
    hour-aligned grid (H4 at 00/04/...); D1 defaults to the configured
    ``D1_ANCHOR`` (default "17:00 America/New_York", DST-aware). A fixed-UTC
    daily grid can be requested with e.g. ``anchor="20 UTC"``.

    ``allow_missing_frac`` (default 0.0, strict): maximum fraction of a
    bucket's EXPECTED LTF bars that may be missing while still emitting the
    bucket. With the default, any missing bar blocks emission and a WARNING
    naming the window start and the missing count is logged. Buckets with no
    expected bars at all (fully weekend-closed windows) are skipped silently.
    """
    if not 0.0 <= allow_missing_frac < 1.0:
        raise ValueError("allow_missing_frac must be in [0.0, 1.0)")
    validate_candles(df_lower)
    if len(df_lower) == 0:
        empty = pd.DataFrame({c: pd.Series(dtype="float64") for c in CANDLE_COLUMNS})
        empty["ts_open"] = pd.to_datetime([], utc=True)
        return empty[CANDLE_COLUMNS]

    tgt = target_tf.strip().upper()
    tgt_dur = timeframe_to_timedelta(tgt)
    src_step = pd.to_datetime(df_lower["ts_open"]).diff().dropna().max()
    ltf_first = ensure_utc(df_lower["ts_open"].iloc[0])
    ltf_last_close = ensure_utc(df_lower["ts_open"].iloc[-1]) + src_step

    if tgt == "D1":
        hour, zone = parse_anchor(anchor)  # None -> config default
    else:
        if anchor is not None:
            raise ValueError("explicit anchors are only supported for D1 resampling")
        hour, zone = _fixed_offset_hours(tgt), None

    if zone is None:
        buckets = _bucket_opens_fixed(ltf_first, ltf_last_close, tgt_dur, hour)
    else:
        buckets = _bucket_opens_zone(ltf_first, ltf_last_close, tgt_dur, hour, zone)

    rows: List[dict] = []
    ts_all = pd.DatetimeIndex(pd.to_datetime(df_lower["ts_open"])).tz_convert("UTC")
    for b_open in buckets:
        b_close = b_open + pd.Timedelta(tgt_dur)
        mask = (ts_all >= b_open) & (ts_all < b_close)
        sub = df_lower.loc[mask]
        if len(sub) == 0:
            continue
        # Completeness: every EXPECTED LTF bar in this bucket must be present.
        # expected_bar_times only yields bars that fully fit inside the
        # window and excludes weekend-closed periods, so a fully closed
        # window (zero expected bars) is skipped silently below.
        expected = [t for t in expected_bar_times_from(src_step, b_open, b_close)
                    if t >= b_open and t + src_step <= b_close]
        n_expected = len(expected)
        if n_expected == 0:
            continue  # market-closed window: not expected, skip silently
        have = set(ts_all[mask])
        missing = [t for t in expected if t not in have]
        frac_missing = len(missing) / n_expected
        if frac_missing > allow_missing_frac:
            log.warning(
                "resample %s->%s: incomplete HTF window starting %s "
                "(%d of %d expected LTF bars missing); not emitted",
                str(src_step), tgt, b_open.isoformat(), len(missing), n_expected,
            )
            continue  # partial bucket -> never emitted
        rows.append({
            "ts_open": b_open,
            "open": float(sub["open"].iloc[0]),
            "high": float(sub["high"].max()),
            "low": float(sub["low"].min()),
            "close": float(sub["close"].iloc[-1]),
            "tick_volume": int(sub["tick_volume"].sum()),
            "spread": float(sub["spread"].iloc[-1]),
        })
    if not rows:
        empty = pd.DataFrame({c: pd.Series(dtype="object") for c in CANDLE_COLUMNS})
        empty["ts_open"] = pd.to_datetime([], utc=True)
        return empty[CANDLE_COLUMNS]
    out = pd.DataFrame(rows)[CANDLE_COLUMNS]
    out = out.sort_values("ts_open").reset_index(drop=True)
    return out


def expected_bar_times_from(step: pd.Timedelta, start: pd.Timestamp,
                            end: pd.Timestamp) -> List[pd.Timestamp]:
    """Expected LTF opens inside ``[start, end)`` honoring market closure.

    ``step`` is the LTF bar duration; it is mapped back to its timeframe name
    so :func:`fxb.core.timeutil.expected_bar_times` can exclude weekend bars.
    """
    minutes = int(step.total_seconds() // 60)
    names = {1: "M1", 5: "M5", 15: "M15", 30: "M30", 60: "H1", 240: "H4", 1440: "D1"}
    if minutes not in names:
        raise ValueError(f"cannot resample from unsupported source step {step}")
    return expected_bar_times(names[minutes], start, end)


def last_closed_htf_index(df_ltf: pd.DataFrame, df_htf: pd.DataFrame,
                          htf_tf: str) -> List[int]:
    """Map each LTF row to the index of the LAST fully-closed HTF bar.

    For every LTF bar ``i`` (its CLOSE time), returns the positional index
    into ``df_htf`` of the newest HTF bar whose own close time is <= that
    LTF close, or ``-1`` if none. Later steps use this to fetch HTF context
    without look-ahead. Assumes both frames sorted ascending by ``ts_open``.
    """
    validate_candles(df_htf)
    dur = pd.Timedelta(timeframe_to_timedelta(htf_tf))
    htf_opens = [ensure_utc(t) for t in df_htf["ts_open"]]
    htf_closes = [o + dur for o in htf_opens]
    out: List[int] = []
    j = -1
    for ts in df_ltf["ts_open"]:
        ltf_close = ensure_utc(ts) + _infer_ltf_duration(df_ltf)
        while j + 1 < len(htf_closes) and htf_closes[j + 1] <= ltf_close:
            j += 1
        out.append(j)
    return out


def _infer_ltf_duration(df_ltf: pd.DataFrame) -> pd.Timedelta:
    diffs = pd.to_datetime(df_ltf["ts_open"]).diff().dropna()
    if len(diffs) == 0:
        raise ValueError("need at least two LTF bars to infer its duration")
    return pd.Timedelta(diffs.mode().iloc[0])
