"""UTC time helpers for FXB.

Global conventions (non-negotiable):
- Every timestamp handled by FXB is a timezone-aware UTC ``datetime`` or
  ``pandas.Timestamp``. Naive datetimes are rejected with :class:`NaiveDatetimeError`.
- A candle's timestamp ``ts_open`` is its OPEN time; close time =
  ``ts_open + timeframe_to_timedelta(tf)``. A bar is closed when
  ``now_utc >= ts_open + duration``.
- Forex market-hours model (approximation, documented): the spot FX market is
  open continuously from Sunday 22:00 UTC to Friday 22:00 UTC and closed in
  between. Public holidays are NOT modeled.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List

import pandas as pd

UTC = timezone.utc

#: Weekly open (Sunday) and close (Friday) hour in UTC for the FX market.
MARKET_OPEN_SUNDAY_UTC_HOUR = 22
MARKET_CLOSE_FRIDAY_UTC_HOUR = 22

_WEEK_SECONDS = 7 * 24 * 3600
#: Epoch day 0 (1970-01-01) was a Thursday, so shift the weekly modulo by 4
#: days to make it Monday-based; Friday 22:00 UTC is then the start of the
#: weekly closed window.
_CLOSED_START_SEC = (4 * 24 + MARKET_CLOSE_FRIDAY_UTC_HOUR) * 3600  # Mon-based Fri 22:00


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

    Closed from Friday 22:00 UTC (inclusive) to Sunday 22:00 UTC (exclusive).
    This is a pure clock approximation: holidays and half-days are ignored.
    """
    ts = ensure_utc(dt)
    # Monday-based week position (epoch day 0 was a Thursday -> +4 days shift).
    sec_in_week = ((int(ts.timestamp()) + 4 * 86_400) % _WEEK_SECONDS)
    return sec_in_week < _CLOSED_START_SEC


def expected_bar_times(tf: str, start: datetime | pd.Timestamp,
                       end: datetime | pd.Timestamp) -> List[pd.Timestamp]:
    """Expected OPEN times of fully-closed bars in ``[start, end)`` for ``tf``.

    - Grid anchored at the Unix epoch (exact minute multiples), which aligns
      with midnight UTC for all supported timeframes.
    - Bars whose full lifetime would touch the weekend-closed period
      (Friday 22:00 UTC .. Sunday 22:00 UTC) are excluded, so gap detection
      never flags the weekend.
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
        # exclude bars touching the weekly closed window [Fri 22:00, Sun 22:00)
        if not (_touches_closed_window(open_ts, close_ts)):
            out.append(open_ts)
        t += step_s
    return out


def _touches_closed_window(open_ts: pd.Timestamp, close_ts: pd.Timestamp) -> bool:
    """True if [open, close] overlaps a weekly closed window.

    The closed window runs Friday 22:00 UTC (inclusive) to Sunday 22:00 UTC
    (exclusive). Both the window of the week containing ``open_ts`` and the
    previous week's window are checked so that early-Sunday bars are handled.
    """
    day = open_ts.normalize()
    # most recent Friday on-or-before the bar's opening date
    fri = day - pd.Timedelta(days=(open_ts.weekday() - 4) % 7)
    fri_close = fri + pd.Timedelta(hours=MARKET_CLOSE_FRIDAY_UTC_HOUR)
    sun_open = fri_close + pd.Timedelta(days=2)
    for shift_days in (-7, 0):
        c0 = fri_close + pd.Timedelta(days=shift_days)
        c1 = sun_open + pd.Timedelta(days=shift_days)
        if open_ts < c1 and close_ts > c0:
            return True
    return False
