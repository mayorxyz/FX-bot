"""UTC time helpers for FXB.

Conventions (prompt 1):
- All timestamps are tz-aware UTC ``pandas.Timestamp`` / ``datetime`` objects.
- ``ts_open`` is the OPEN time of a bar; close time = ts_open + timeframe duration.
- FX market hours model: open Sunday 22:00 UTC, closed Friday 22:00 UTC ->
  Sunday 22:00 UTC (daily session boundary at 22:00 UTC).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

UTC = timezone.utc

#: Hour (UTC) at which the FX market closes on Friday and opens on Sunday.
MARKET_OPEN_UTC_HOUR = 22
MARKET_CLOSE_UTC_HOUR = 22

#: Session boundaries by UTC hour (documented approximation used for trade tagging).
#:   ASIA     00:00-07:00 UTC (Tokyo; Sydney early)
#:   LONDON   07:00-12:00 UTC
#:   OVERLAP  12:00-17:00 UTC (London/New York overlap)
#:   NEWYORK  17:00-22:00 UTC
#:   CLOSED   22:00-00:00 UTC (weekend/rollover gap region)
SESSION_ASIA = "ASIA"
SESSION_LONDON = "LONDON"
SESSION_NEWYORK = "NEWYORK"
SESSION_OVERLAP = "OVERLAP"
SESSION_CLOSED = "CLOSED"


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


def to_utc(ts) -> pd.Timestamp:
    """Coerce anything timestamp-like to a tz-aware UTC pandas Timestamp."""
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize(UTC)
    else:
        t = t.tz_convert(UTC)
    return t


def tf_to_timedelta(timeframe: str) -> timedelta:
    """Convert an FXB timeframe string ('M5', 'H1', 'D1', ...) to a timedelta."""
    tf = timeframe.strip().upper()
    unit = tf[0]
    n = int(tf[1:])
    if unit == "M":
        return timedelta(minutes=n)
    if unit == "H":
        return timedelta(hours=n)
    if unit == "D":
        return timedelta(days=n)
    if unit == "W":
        return timedelta(weeks=n)
    raise ValueError(f"Unknown timeframe '{timeframe}' (expected M<n>, H<n>, D<n>, W<n>)")


def tf_minutes(timeframe: str) -> int:
    return int(tf_to_timedelta(timeframe).total_seconds() // 60)


def tf_to_pandas_freq(timeframe: str) -> str:
    """Pandas resample frequency alias for an FXB timeframe string."""
    tf = timeframe.strip().upper()
    unit, n = tf[0], tf[1:]
    if unit == "M":
        return f"{n}min"
    if unit == "H":
        return f"{n}h"
    if unit == "D":
        return f"{n}D"
    if unit == "W":
        return f"{n}W"
    raise ValueError(f"Unknown timeframe '{timeframe}'")


def is_tf_higher(primary: str, other: str) -> bool:
    return tf_minutes(other) > tf_minutes(primary)


def bar_close_time(ts_open, timeframe: str) -> pd.Timestamp:
    return to_utc(ts_open) + tf_to_timedelta(timeframe)


def is_market_open(ts) -> bool:
    """Weekend model: market is open from Sunday 22:00 UTC to Friday 22:00 UTC."""
    t = to_utc(ts)
    wd = t.weekday()  # Mon=0 ... Sun=6
    h = t.hour
    if wd == 5:  # Saturday
        return False
    if wd == 6:  # Sunday: open only after 22:00
        return h >= MARKET_OPEN_UTC_HOUR
    if wd == 4:  # Friday: closed from 22:00
        return h < MARKET_CLOSE_UTC_HOUR
    return True


def crosses_weekend(ts_a, ts_b) -> bool:
    """True if the interval [ts_a, ts_b) contains any market-closed weekend window
    (Friday 22:00 UTC -> Sunday 22:00 UTC)."""
    a, b = to_utc(ts_a), to_utc(ts_b)
    if b <= a:
        return False
    # walk day by day; each calendar day D defines the closed window
    # [D->Friday of that week 22:00 UTC, +48h].
    day = a.normalize()
    end = b.normalize()
    while day <= end:
        days_from_fri = (day.weekday() - 4) % 7
        fri_close = day - pd.Timedelta(days=int(days_from_fri)) + pd.Timedelta(
            hours=MARKET_CLOSE_UTC_HOUR
        )
        sun_open = fri_close + pd.Timedelta(days=2)
        if _overlaps(a, b, fri_close, sun_open):
            return True
        day += pd.Timedelta(days=1)
    return False


def _overlaps(a, b, c, d) -> bool:
    """Do intervals (a,b) and (c,d) overlap?"""
    return a < d and c < b


def session_of(ts) -> str:
    """Trading session tag by UTC hour (see table above)."""
    t = to_utc(ts)
    h = t.hour
    if not is_market_open(t):
        return SESSION_CLOSED
    if 0 <= h < 7:
        return SESSION_ASIA
    if 7 <= h < 12:
        return SESSION_LONDON
    if 12 <= h < 17:
        return SESSION_OVERLAP
    return SESSION_NEWYORK
