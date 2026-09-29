"""NY wall-clock session tags and killzones (DST-safe via zoneinfo).

Distinct from the coarse UTC boxes in liquidity.py: killzones are defined in
NEW YORK local time so they track US DST automatically.

Definitions (documented):
  * London killzone : 02:00 - 05:00 NY (inclusive start, exclusive end).
  * New York killzone: 07:00 - 10:00 NY (inclusive start, exclusive end).
  * ``session_tag(ts)`` broad labels by NY hour: LONDON [02,07),
    NEWYORK [07,16), ASIA/OTHER [16,02) spanning midnight; a bar inside both
    killzone-adjacent hours still gets one tag only (hours decide).
  * ``minutes_since_session_open`` counts minutes since the CURRENT killzone
    opened; NaN outside any killzone.
All functions accept tz-aware UTC timestamps (anything else raises via
:func:`fxb.core.timeutil.ensure_utc`).
"""
from __future__ import annotations

import math
from datetime import datetime

import pandas as pd

from fxb.core.timeutil import _MARKET_TZ, ensure_utc

#: killzones in NY wall-clock hours: name -> (start_hour, end_hour)
KILLZONES = {
    "LONDON_KZ": (2, 5),
    "NEWYORK_KZ": (7, 10),
}


def _ny_time(ts: datetime | pd.Timestamp) -> pd.Timestamp:
    """Convert an aware timestamp to America/New_York wall clock."""
    return ensure_utc(ts).tz_convert(_MARKET_TZ)


def session_tag(ts: datetime | pd.Timestamp) -> str:
    """Broad NY-local session label for instant ``ts``.

    LONDON  [02:00, 07:00) NY, NEWYORK [07:00, 16:00) NY, ASIA otherwise
    ([16:00, 02:00) NY wrapping midnight — includes the Pacific session and
    the late-NY roll-off). Returns "CLOSED_WEEKEND" inside the FX weekly
    closure if the market-hours model says so (kept simple: hour-based only,
    weekday checked via the NY calendar day).
    """
    ny = _ny_time(ts)
    # weekend closure check in NY terms: Fri >= 17:00 or Sat, or Sun < 17:00
    dow = ny.dayofweek
    hour = ny.hour + ny.minute / 60.0 + ny.second / 3600.0
    if (dow == 4 and hour >= 17.0) or dow == 5 or (dow == 6 and hour < 17.0):
        return "CLOSED_WEEKEND"
    if 2 <= ny.hour < 7:
        return "LONDON"
    if 7 <= ny.hour < 16:
        return "NEWYORK"
    return "ASIA"


def is_killzone(ts: datetime | pd.Timestamp) -> bool:
    """True when ``ts`` lies inside any killzone window (NY wall clock)."""
    return killzone_name(ts) is not None


def killzone_name(ts: datetime | pd.Timestamp) -> str | None:
    """Name of the active killzone at ``ts`` ("LONDON_KZ"/"NEWYORK_KZ") or None."""
    ny = _ny_time(ts)
    h = ny.hour
    m = ny.minute
    s = ny.second
    for name, (lo, hi) in KILLZONES.items():
        if lo <= h < hi:
            return name
    return None


def minutes_since_session_open(ts: datetime | pd.Timestamp) -> float:
    """Minutes elapsed since the active killzone opened; NaN outside killzones.

    DST-safe: the comparison happens entirely in NY wall-clock space, so the
    value measures local-session minutes even across a DST transition day.
    """
    name = killzone_name(ts)
    if name is None:
        return float("nan")
    lo, _hi = KILLZONES[name]
    ny = _ny_time(ts)
    delta = ny - (ny.normalize() + pd.Timedelta(hours=lo))
    return delta.total_seconds() / 60.0
