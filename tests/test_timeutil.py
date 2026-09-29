"""Tests for fxb.core.timeutil."""
import pandas as pd
import pytest

from fxb.core.timeutil import (
    NaiveDatetimeError,
    bar_close_time,
    ensure_utc,
    expected_bar_times,
    is_forex_market_open,
    timeframe_to_timedelta,
)


def test_ensure_utc_accepts_aware_and_converts():
    ny = pd.Timestamp("2024-06-07 18:00", tz="America/New_York")
    out = ensure_utc(ny)
    assert str(out.tz) == "UTC"
    assert out == pd.Timestamp("2024-06-07 22:00", tz="UTC")


def test_ensure_utc_rejects_naive():
    with pytest.raises(NaiveDatetimeError):
        ensure_utc(pd.Timestamp("2024-06-07 18:00"))


def test_timeframe_to_timedelta_all_supported():
    assert timeframe_to_timedelta("M1") == pd.Timedelta(minutes=1)
    assert timeframe_to_timedelta("M5") == pd.Timedelta(minutes=5)
    assert timeframe_to_timedelta("M15") == pd.Timedelta(minutes=15)
    assert timeframe_to_timedelta("M30") == pd.Timedelta(minutes=30)
    assert timeframe_to_timedelta("H1") == pd.Timedelta(hours=1)
    assert timeframe_to_timedelta("H4") == pd.Timedelta(hours=4)
    assert timeframe_to_timedelta("D1") == pd.Timedelta(days=1)
    with pytest.raises(ValueError):
        timeframe_to_timedelta("W1")


def test_market_closed_weekend_window():
    # Friday 21:59 UTC open, 22:00 closed (inclusive)
    assert is_forex_market_open(pd.Timestamp("2024-06-07 21:59", tz="UTC")) is True
    assert is_forex_market_open(pd.Timestamp("2024-06-07 22:00", tz="UTC")) is False
    # Saturday / Sunday daytime closed
    assert is_forex_market_open(pd.Timestamp("2024-06-08 12:00", tz="UTC")) is False
    assert is_forex_market_open(pd.Timestamp("2024-06-09 21:59", tz="UTC")) is False
    # Sunday 22:00 reopens; Monday normal hours open
    assert is_forex_market_open(pd.Timestamp("2024-06-09 22:00", tz="UTC")) is True
    assert is_forex_market_open(pd.Timestamp("2024-06-10 08:00", tz="UTC")) is True


def test_bar_close_time():
    ts = bar_close_time("H1", pd.Timestamp("2024-06-10 08:00", tz="UTC"))
    assert ts == pd.Timestamp("2024-06-10 09:00", tz="UTC")


def test_expected_bar_times_skips_weekend_entirely():
    start = pd.Timestamp("2024-06-07 20:00", tz="UTC")   # Friday evening
    end = pd.Timestamp("2024-06-10 00:00", tz="UTC")     # Monday 00:00
    bars = expected_bar_times("H1", start, end)
    # Fri 20:00 and 21:00 bars are fully inside the market window.
    assert bars[0] == start
    assert bars[1] == pd.Timestamp("2024-06-07 21:00", tz="UTC")
    # Then nothing until Sunday 22:00 (bars touching the closed window excluded).
    assert bars[2] == pd.Timestamp("2024-06-09 22:00", tz="UTC")
    assert all(is_forex_market_open(t) for t in bars)
    # Only bars already CLOSED before `end` are expected.
    assert bars[-1] == pd.Timestamp("2024-06-09 23:00", tz="UTC")


def test_expected_bar_times_excludes_unclosed_tail():
    start = pd.Timestamp("2024-06-10 08:00", tz="UTC")
    end = pd.Timestamp("2024-06-10 09:30", tz="UTC")
    bars = expected_bar_times("H1", start, end)
    assert bars == [start]  # the 09:00 bar would close at 10:00 > end


def test_expected_bar_times_m15_friday_boundary():
    start = pd.Timestamp("2024-06-07 21:30", tz="UTC")
    end = pd.Timestamp("2024-06-07 23:00", tz="UTC")
    bars = expected_bar_times("M15", start, end)
    # 21:30 bar closes 21:45 (open); 21:45 bar closes 22:00 exactly at close -> allowed;
    # 22:00+ bars touch the closed window -> excluded.
    assert bars == [pd.Timestamp("2024-06-07 21:30", tz="UTC"),
                    pd.Timestamp("2024-06-07 21:45", tz="UTC")]
