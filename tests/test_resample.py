"""Tests for fxb.data.resample (strict HTF completeness, anchors, no look-ahead)."""
import pandas as pd
import pytest

from conftest import grid, make_candles
from fxb.data.resample import last_closed_htf_index, parse_anchor, resample_ohlc


def m15_frame(opens, o=None):
    """M15 frame with per-bar-varying close so aggregation is checkable."""
    n = len(opens)
    if o is None:
        o = list(range(1, n + 1))
    return pd.DataFrame({
        "ts_open": pd.DatetimeIndex(pd.to_datetime(list(opens))).tz_convert("UTC"),
        "open": [float(i) for i in o],
        "high": [float(i) + 0.5 for i in o],
        "low": [float(i) - 0.5 for i in o],
        "close": [float(i) + 0.1 for i in o],
        "tick_volume": [10] * n,
        "spread": [2.0] * n,
    })


# ---------------------------------------------------------------- M15 -> H1
def test_m15_to_h1_aggregation_correct():
    df = m15_frame(grid(15, "2024-06-10 08:00", 4))  # full hour
    h1 = resample_ohlc(df, "H1")
    assert len(h1) == 1
    row = h1.iloc[0]
    assert row["ts_open"] == pd.Timestamp("2024-06-10 08:00", tz="UTC")
    assert row["open"] == 1.0            # first
    assert row["high"] == pytest.approx(4.5)   # max of highs
    assert row["low"] == pytest.approx(0.5)    # min of lows
    assert row["close"] == pytest.approx(4.1)  # last
    assert row["tick_volume"] == 40      # sum
    assert row["spread"] == 2.0          # last


def test_partial_h1_not_emitted():
    df = m15_frame(grid(15, "2024-06-10 08:00", 3))  # missing 08:45 bar
    h1 = resample_ohlc(df, "H1")
    assert len(h1) == 0


def test_missing_interior_bar_blocks_bucket():
    df = m15_frame(grid(15, "2024-06-10 08:00", 4)).drop(index=2).reset_index(drop=True)
    assert len(resample_ohlc(df, "H1")) == 0


def test_allow_missing_frac_tolerates_one_of_four():
    df = m15_frame(grid(15, "2024-06-10 08:00", 4)).drop(index=2).reset_index(drop=True)
    out = resample_ohlc(df, "H1", allow_missing_frac=0.25)
    assert len(out) == 1  # 1/4 missing <= allowed fraction


# ---------------------------------------------------------------- M15 -> H4
def test_m15_to_h4_full_window():
    opens = grid(15, "2024-06-10 08:00", 16)  # 08:00..11:45 -> one H4 (08:00 window)
    h4 = resample_ohlc(m15_frame(opens), "H4")
    assert len(h4) == 1
    assert h4["ts_open"].iloc[0] == pd.Timestamp("2024-06-10 08:00", tz="UTC")
    assert h4["tick_volume"].iloc[0] == 160


def test_regression_partial_h4_window_not_emitted():
    """REGRESSION: a partial H4 window must never be emitted."""
    opens = grid(15, "2024-06-10 08:00", 15)  # 15 of 16 bars: missing 11:45
    h4 = resample_ohlc(m15_frame(opens), "H4")
    assert len(h4) == 0
    # but the same data minus nothing else -> adding the last bar emits it
    h4_full = resample_ohlc(m15_frame(grid(15, "2024-06-10 08:00", 16)), "H4")
    assert len(h4_full) == 1


def test_h4_grid_aligned_to_utc_multiples_of_four():
    # 08:00 start sits exactly on the 00/04/08/... grid
    opens = grid(15, "2024-06-10 04:00", 32)  # 04:00..11:45 -> two H4 windows
    h4 = resample_ohlc(m15_frame(opens), "H4")
    assert list(h4["ts_open"]) == [pd.Timestamp("2024-06-10 04:00", tz="UTC"),
                                   pd.Timestamp("2024-06-10 08:00", tz="UTC")]


# ---------------------------------------------------------------- D1 anchor / DST
def _m15_range(start_utc: pd.Timestamp, end_utc: pd.Timestamp):
    """All M15 opens with open >= start and close <= end (contiguous grid)."""
    out = []
    t = start_utc
    while t + pd.Timedelta(minutes=15) <= end_utc:
        out.append(t)
        t += pd.Timedelta(minutes=15)
    return out


def test_d1_anchor_default_is_ny_17():
    hour, zone = parse_anchor(None)
    assert hour == 17
    assert str(zone) == "America/New_York"


def test_d1_boundaries_across_dst_change():
    """REGRESSION: D1 windows spanning the US DST spring-forward (2024-03-10).

    NY 17:00 is UTC 22:00 in winter (EST, UTC-5) and UTC 21:00 in summer
    (EDT, UTC-4), so the UTC boundary shifts by one hour across the change.
    The window Mar-08 22:00Z -> Mar-10 21:00Z (Fri 17:00 EST -> Sun 17:00 EDT)
    spans the transition and contains exactly 96 expected M15 bars (the
    market-closed weekend hours are excluded from the expectation).
    """
    fri_win = pd.Timestamp("2024-03-08 22:00", tz="UTC")   # Fri 17:00 EST
    sun_win = pd.Timestamp("2024-03-10 21:00", tz="UTC")   # Sun 17:00 EDT
    mon_win = pd.Timestamp("2024-03-11 21:00", tz="UTC")   # Mon 17:00 EDT

    opens = _m15_range(fri_win, sun_win) + _m15_range(sun_win, mon_win)
    df = m15_frame(opens)
    d1 = resample_ohlc(df, "D1")

    boundaries = list(d1["ts_open"])
    assert fri_win in boundaries          # correct pre-DST boundary (22:00Z)
    assert sun_win in boundaries          # correct post-DST boundary (21:00Z)
    # Bar counts: the winter Fri->Sun window and the DST-spanning Sun->Mon
    # window each contain exactly 96 EXPECTED M15 bars (market-closed hours
    # are excluded from the expectation, so neither is flagged incomplete).
    row_fri = d1[d1["ts_open"] == fri_win].iloc[0]
    assert row_fri["tick_volume"] == 960          # 96 bars * 10 vol
    row_sun = d1[d1["ts_open"] == sun_win].iloc[0]
    assert row_sun["tick_volume"] == 960          # 23h clock span - 1h DST = 96 bars
    # Consecutive emitted boundaries differ by 24h except across the DST shift (23h):
    diffs = pd.Series(boundaries).diff().dropna()
    assert all((d == pd.Timedelta(hours=24)) or (d == pd.Timedelta(hours=23))
               or (d == pd.Timedelta(hours=25)) for d in diffs)


def test_d1_summer_window_bar_count():
    """A fully supplied summer (EDT) day window emits with its real bar count."""
    start = pd.Timestamp("2024-03-11 21:00", tz="UTC")   # Mon 17:00 EDT
    end = pd.Timestamp("2024-03-12 21:00", tz="UTC")     # Tue 17:00 EDT
    d1 = resample_ohlc(m15_frame(_m15_range(start, end)), "D1")
    assert len(d1) == 1
    assert d1["tick_volume"].iloc[0] == 960  # 96 M15 bars * 10


def test_weekend_windows_skipped_silently(caplog):
    """Fully weekend-closed D1/H4 windows have ZERO expected bars and must be
    skipped without any 'incomplete' warning."""
    thu = _m15_range(pd.Timestamp("2024-06-06 00:00", tz="UTC"),
                     pd.Timestamp("2024-06-07 00:00", tz="UTC"))
    fri = _m15_range(pd.Timestamp("2024-06-07 00:00", tz="UTC"),
                     pd.Timestamp("2024-06-07 22:00", tz="UTC"))
    sun = _m15_range(pd.Timestamp("2024-06-09 22:00", tz="UTC"),
                     pd.Timestamp("2024-06-10 00:00", tz="UTC"))
    df = m15_frame(thu + fri + sun)
    h4 = resample_ohlc(df, "H4")
    # No H4 window may contain a bar that touches the closed weekend.
    for ts in h4["ts_open"]:
        close = ts + pd.Timedelta(hours=4)
        assert not (ts < pd.Timestamp("2024-06-09 22:00", tz="UTC") and
                    close > pd.Timestamp("2024-06-07 22:00", tz="UTC"))
    incomplete_msgs = [r.getMessage() for r in caplog.records if "incomplete" in r.getMessage()]
    # Only genuine gaps (partial weekday windows at the edges) may warn; the
    # pure weekend windows themselves never appear as warnings.
    for msg in incomplete_msgs:
        assert "2024-06-08" not in msg and "2024-06-09T0" not in msg


# ---------------------------------------------------------- last_closed_htf_index
def test_last_closed_htf_index_never_returns_future_htf():
    m15 = m15_frame(grid(15, "2024-06-10 08:00", 12))  # 08:00..10:45 opens
    h1 = resample_ohlc(m15, "H1")                     # 2 closed hours (08, 09)
    idx = last_closed_htf_index(m15, h1, "H1")
    assert len(idx) == len(m15)
    h1_opens = list(h1["ts_open"])
    for i, j in enumerate(idx):
        ltf_close = m15["ts_open"].iloc[i] + pd.Timedelta(minutes=15)
        if j >= 0:
            htf_close = h1_opens[j] + pd.Timedelta(hours=1)
            assert htf_close <= ltf_close  # THE no-look-ahead guarantee
    # First LTF bar (closes 08:15) sees no closed H1 yet.
    assert idx[0] == -1
    # Bar closing exactly at 09:00 (08:45 open) sees the 08:00 H1.
    assert idx[3] == 0
    # Last bar (closes 11:00) sees both H1s.
    assert idx[-1] == len(h1) - 1


def test_last_closed_htf_index_empty_htf():
    m15 = m15_frame(grid(15, "2024-06-10 08:00", 3))
    empty = m15.iloc[0:0]
    idx = last_closed_htf_index(m15, empty, "H1")
    assert idx == [-1, -1, -1]
