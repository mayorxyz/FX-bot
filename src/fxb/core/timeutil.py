"""UTC time helpers for FXB.

Global conventions (non-negotiable):
- Every timestamp handled by FXB is a timezone-aware UTC ``datetime`` or
  ``pandas.Timestamp``. Naive datetimes are rejected with :class:`NaiveDatetimeError`.
- A candle's timestamp ``ts_open`` is its OPEN time; close time =
  ``ts_open + timeframe_to_timedelta(tf)``. A bar is closed when
  ``now_utc >= ts_open + duration``.
- Forex market-hours model (approximation, documented): the spot FX market is
  CLOSED from Friday 17:00 America/New_York (inclusive) to Sunday 17:00
  America/New_York (exclusive), and OPEN otherwise. New York DST is honored
  via :mod:`zoneinfo`, so the UTC boundaries shift between winter (22:00Z)
  and summer (21:00Z). Public holidays are NOT modeled.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List
from zoneinfo import ZoneInfo

import pandas as pd

UTC = timezone.utc

#: IANA zone used for the weekly market-open/close wall-clock schedule.
MARKET_TZ_NAME = "America/New_York"
_MARKET_TZ = ZoneInfo(MARKET_TZ_NAME)

#: Weekly close (Friday) and reopen (Sunday) hour in NEW YORK wall-clock time.
#: The market is closed on [Fri 17:00 NY, Sun 17:00 NY).
MARKET_CLOSE_FRIDAY_NY_HOUR = 17
MARKET_OPEN_SUNDAY_NY_HOUR = 17

#: Back-compat aliases (deprecated): the schedule is now defined in New York
#: wall-clock time, not fixed UTC hours. Kept so older imports keep working.
MARKET_CLOSE_FRIDAY_UTC_HOUR = MARKET_CLOSE_FRIDAY_NY_HOUR
MARKET_OPEN_SUNDAY_UTC_HOUR = MARKET_OPEN_SUNDAY_NY_HOUR


def _to_utc(ny_ts: pd.Timestamp) -> pd.Timestamp:
    """Convert a New-York wall-clock Timestamp to UTC (DST-correct).

    ``ny_ts`` may be naive (interpreted as New York wall-clock time) or
    already carry the New York zone (as produced by :meth:`tz_convert`); in
    that case it is converted directly. Localizing an already-aware stamp is
    impossible, so the two cases must be handled separately.
    """
    ts = pd.Timestamp(ny_ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize(_MARKET_TZ, nonexistent="shift_forward", ambiguous=False)
    elif str(ts.tzinfo) != MARKET_TZ_NAME:
        raise ValueError(f"expected New York wall-clock timestamp, got tz {ts.tzinfo!r}")
    return ts.tz_convert("UTC")


def _week_closed_window(ref: pd.Timestamp) -> List[tuple[pd.Timestamp, pd.Timestamp]]:
    """Candidate weekly closed windows [c0, c1) (UTC pairs) for ``ref``'s week.

    Windows run Friday 17:00 -> Sunday 17:00 *New York wall-clock time*;
    converting those instants to UTC honors DST automatically (winter
    closure = Fri 22:00Z .. Sun 22:00Z; summer = Fri 21:00Z .. Sun 21:00Z).

    The anchor is the most recent Friday **in New York local date terms** at
    or before ``ref``. Because a UTC instant can sit in a different weekday
    than its New York counterpart around the Sunday reopen (e.g. winter
    Sun 00:30Z = Sat 19:30 NY — still inside the PREVIOUS Friday's closure),
    the window anchored on the current local Friday AND the one anchored on
    the previous Friday are both returned whenever the two UTC/NY weekday
    readings disagree around that boundary. Callers treat the result as a
    candidate set: an instant is "closed" iff it lies inside ANY of them.
    """
    ts = ensure_utc(ref)
    local = ts.tz_convert(_MARKET_TZ)
    days_since_fri = (local.dayofweek - 4) % 7   # Mon..Sun -> 4,5,6,0,1,2,3
    fri = local.normalize() - pd.Timedelta(days=int(days_since_fri))
    anchors = [fri]
    if int(local.dayofweek) != (int(ts.dayofweek) - int(days_since_fri)) % 7:
        # UTC and New York disagree about which calendar day we are in (the
        # only way that happens is near midnight/weekend boundaries): the
        # closure anchored on the PREVIOUS local Friday may still be in
        # effect (e.g. winter Sun 00:30Z = Sat 19:30 NY). Include it too.
        anchors.append(fri - pd.Timedelta(days=7))
    out: List[tuple[pd.Timestamp, pd.Timestamp]] = []
    for f in anchors:
        c0 = _to_utc(f.replace(hour=MARKET_CLOSE_FRIDAY_NY_HOUR))
        c1 = _to_utc((f + pd.Timedelta(days=2)).replace(hour=MARKET_OPEN_SUNDAY_NY_HOUR))
        out.append((c0, c1))
    return out


def closed_windows_around(ref: pd.Timestamp) -> List[tuple[pd.Timestamp, pd.Timestamp]]:
    """Candidate weekly closed windows that could contain/intersect ``ref``.

    Returns the distinct [Fri 17:00 NY, Sun 17:00 NY) windows (as UTC
    ``(c0, c1)`` pairs) found by probing ``ref`` and ref ± one week; each
    probe may itself yield two candidate anchors around the Sunday/DST
    boundary (see :func:`_week_closed_window`). This covers every boundary
    case: Friday evening, Saturday, and Sunday morning before the reopen
    under either UTC-day alignment (e.g. winter Sun 00:30Z belongs to the
    *previous* local week's window while Sun 23:00Z reopens the new one).
    """
    ts = ensure_utc(ref)
    out: List[tuple[pd.Timestamp, pd.Timestamp]] = []
    seen: set[tuple[float, float]] = set()
    for shift_days in (0, -7, 7):
        for c0, c1 in _week_closed_window(ts + pd.Timedelta(days=int(shift_days))):
            key = (float(c0.timestamp()), float(c1.timestamp()))
            if key not in seen:
                seen.add(key)
                out.append((c0, c1))
    return out


class NaiveDatetimeError(ValueError):
    """Raised when a naive (tz-unaware) datetime is passed where UTC is required."""


def ensure_utc(dt: datetime | pd.Timestamp) -> pd.Timestamp:
    """Return ``dt`` as a tz-aware UTC :class:`pandas.Timestamp`.

    Raises :class:`NaiveDatetimeError` for naive input; converts other aware
    zones to UTC.
    """
    ts = pd.Timestamp(dt)
    if ts.tzinfo is None or ts.tzinfo.utcoffset(ts) is None:
        raise NaiveDatetimeError(
            f"naive datetime {dt!r} rejected: all timestamps must be tz-aware UTC"
        )
    return ts.tz_convert("UTC")


def timeframe_to_timedelta(tf: str) -> timedelta:
    """Convert a timeframe string (M1, M5, M15, M30, H1, H4, D1) to a timedelta."""
    key = tf.strip().upper()
    minutes = {
        "M1": 1,
        "M5": 5,
        "M15": 15,
        "M30": 30,
        "H1": 60,
        "H4": 240,
        "D1": 1440,
    }
    if key not in minutes:
        raise ValueError(f"unknown timeframe {tf!r}; supported: {sorted(minutes)}")
    return timedelta(minutes=minutes[key])


def bar_close_time(tf: str, ts_open: datetime | pd.Timestamp) -> pd.Timestamp:
    """Close time of the bar that opens at ``ts_open`` on timeframe ``tf``."""
    return ensure_utc(ts_open) + pd.Timedelta(timeframe_to_timedelta(tf))


def is_forex_market_open(dt: datetime | pd.Timestamp) -> bool:
    """True if the (approximated) spot FX market is open at instant ``dt``.

    CLOSED from Friday 17:00 America/New_York (inclusive) to Sunday 17:00
    America/New_York (exclusive); OPEN otherwise. Pure clock approximation:
    holidays and half-days are ignored. Because the schedule is evaluated in
    New York wall-clock time, the UTC boundary shifts with DST (22:00 UTC in
    winter, 21:00 UTC in summer).
    """
    ts = ensure_utc(dt)
    for c0, c1 in closed_windows_around(ts):
        if c0 <= ts < c1:
            return False
    return True


def expected_bar_times(tf: str, start: datetime | pd.Timestamp,
                       end: datetime | pd.Timestamp) -> List[pd.Timestamp]:
    """Expected OPEN times of fully-closed bars in ``[start, end)`` for ``tf``.

    - Grid anchored at the Unix epoch (exact minute multiples), which aligns
      with midnight UTC for all supported timeframes.
    - Bars whose full lifetime would touch the weekend-closed period
      (Friday 17:00 .. Sunday 17:00 America/New_York) are excluded, so gap
      detection never flags the weekend.
    - Only bars already CLOSED at ``end`` are expected (``close <= end``).
    """
    start_ts = ensure_utc(start)
    end_ts = ensure_utc(end)
    dur = pd.Timedelta(timeframe_to_timedelta(tf))
    if end_ts <= start_ts:
        return []
    # round start up onto the epoch-aligned grid
    epoch_s = int(start_ts.timestamp())
    step_s = int(dur.total_seconds())
    first_open_s = ((epoch_s + step_s - 1) // step_s) * step_s
    out: List[pd.Timestamp] = []
    t = first_open_s
    while t + step_s <= int(end_ts.timestamp()):
        open_ts = pd.Timestamp(t, unit="s", tz="UTC")
        close_ts = open_ts + dur
        # exclude bars touching the weekly closed window [Fri 17:00 NY, Sun 17:00 NY)
        if not (_touches_closed_window(open_ts, close_ts)):
            out.append(open_ts)
        t += step_s
    return out


def _touches_closed_window(open_ts: pd.Timestamp, close_ts: pd.Timestamp) -> bool:
    """True if the bar lifetime [open, close] overlaps any weekly closed window.

    A bar is excluded when it would trade into the closed period
    [Fri 17:00 NY, Sun 17:00 NY). Windows for the bar's local week and the
    previous/next weeks are checked so early-Sunday bars (whose UTC date can
    fall in either NY week) are handled correctly.
    """
    for c0, c1 in closed_windows_around(open_ts):
        if open_ts < c1 and close_ts > c0:
            return True
    return False


# ---------------------------------------------------------------------------
# Compatibility helpers used by core.frames / signals / backtest modules.
# Thin aliases over the canonical functions above; kept here so those modules
# never re-implement the market-hours or timezone logic.
# ---------------------------------------------------------------------------

#: Alias of :func:`is_forex_market_open` (canonical name for older callers).
is_market_open = is_forex_market_open


def to_utc(dt: datetime | pd.Timestamp) -> pd.Timestamp:
    """Alias of :func:`ensure_utc`."""
    return ensure_utc(dt)


def tf_to_timedelta(tf: str) -> pd.Timedelta:
    """Return :func:`timeframe_to_timedelta` as a ``pandas.Timedelta``."""
    return pd.Timedelta(timeframe_to_timedelta(tf))


def tf_minutes(tf: str) -> int:
    """Timeframe duration in whole minutes."""
    return int(timeframe_to_timedelta(tf).total_seconds() // 60)


def crosses_weekend(start: datetime | pd.Timestamp, end: datetime | pd.Timestamp) -> bool:
    """True if the interval ``[start, end]`` contains any weekly closed window.

    Used to flag trades held across the weekend closure
    [Fri 17:00 NY, Sun 17:00 NY). Endpoints exactly touching a boundary count
    as crossing only if the closed period lies strictly inside the span.
    """
    t0 = ensure_utc(start)
    t1 = ensure_utc(end)
    if t1 <= t0:
        return False
    for c0, c1 in closed_windows_around(t0 + (t1 - t0) / 2):
        if t0 < c1 and t1 > c0:
            return True
    return False


def session_of(ts: datetime | pd.Timestamp) -> str:
    """Trading-session tag by UTC hour of ``ts`` (documented approximation).

    ASIA [0, 7), LONDON [7, 12), OVERLAP [12, 17), NEWYORK [17, 22),
    CLOSED otherwise (the weekend-closure hours where no fill should occur).
    DST shifts are ignored at this coarse granularity on purpose.
    """
    t = ensure_utc(ts)
    h = t.hour
    if 0 <= h < 7:
        return "ASIA"
    if 7 <= h < 12:
        return "LONDON"
    if 12 <= h < 17:
        return "OVERLAP"
    if 17 <= h < 22:
        return "NEWYORK"
    return "CLOSED"