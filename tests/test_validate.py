"""Tests for fxb.data.validate."""
import numpy as np
import pandas as pd
import pytest

from conftest import grid, make_candles
from fxb.data.validate import (
    CandleValidationError,
    drop_unclosed_bar,
    find_gaps,
    validate_candles,
)


def df5():
    return make_candles(grid(15, "2024-06-10 08:00", 5))


def test_valid_frame_passes():
    validate_candles(df5())  # must not raise


def test_empty_frame_with_columns_passes():
    validate_candles(make_candles([]))


def test_missing_column_raises():
    df = df5().drop(columns=["spread"])
    with pytest.raises(CandleValidationError, match="missing columns"):
        validate_candles(df)


def test_naive_timestamp_raises():
    df = df5()
    df["ts_open"] = df["ts_open"].dt.tz_localize(None)
    with pytest.raises(CandleValidationError, match="tz-aware UTC"):
        validate_candles(df)


def test_non_utc_tz_is_converted_ok():
    df = df5()
    df["ts_open"] = df["ts_open"].dt.tz_convert("America/New_York")
    validate_candles(df)  # aware non-UTC is acceptable input, converted internally


def test_duplicate_and_unsorted_raise():
    df = df5()
    dup = pd.concat([df, df.iloc[[2]]], ignore_index=True)
    with pytest.raises(CandleValidationError, match="strictly increasing"):
        validate_candles(dup)


def test_nan_raises():
    df = df5()
    df.loc[1, "close"] = np.nan
    with pytest.raises(CandleValidationError, match="NaN"):
        validate_candles(df)


def test_inf_raises():
    df = df5()
    df.loc[1, "high"] = np.inf
    with pytest.raises(CandleValidationError, match="infinite"):
        validate_candles(df)


def test_high_below_max_oc_raises():
    df = df5()
    df.loc[0, "high"] = 0.95  # < close 1.05
    with pytest.raises(CandleValidationError, match=r"high < max\(open, close\)"):
        validate_candles(df)


def test_low_above_min_oc_raises():
    df = df5()
    df.loc[0, "low"] = 1.02  # > open 1.0
    with pytest.raises(CandleValidationError, match=r"low > min\(open, close\)"):
        validate_candles(df)


def test_nonpositive_price_raises():
    df = df5()
    df.loc[0, "open"] = 0.0
    with pytest.raises(CandleValidationError, match="prices must be > 0"):
        validate_candles(df)


def test_negative_spread_raises():
    df = df5()
    df.loc[0, "spread"] = -1.0
    with pytest.raises(CandleValidationError, match="spread must be >= 0"):
        validate_candles(df)


def test_find_gaps_flags_missing_weekday_bar():
    df = df5().drop(index=2).reset_index(drop=True)
    gaps = find_gaps(df, "M15")
    assert len(gaps) == 1
    assert gaps["ts_open"].iloc[0] == pd.Timestamp("2024-06-10 08:30", tz="UTC")


def test_find_gaps_no_false_positive_on_clean_data():
    assert len(find_gaps(df5(), "M15")) == 0


def test_find_gaps_ignores_weekend():
    opens = [pd.Timestamp(t, tz="UTC") for t in [
        "2024-06-07 21:00", "2024-06-07 21:15", "2024-06-07 21:30", "2024-06-07 21:45",
        "2024-06-09 22:00", "2024-06-09 22:15",
    ]]
    gaps = find_gaps(make_candles(opens), "M15")
    assert len(gaps) == 0  # the whole weekend gap must NOT be flagged


def test_drop_unclosed_bar_removes_forming_last_bar():
    df = df5()
    now = df["ts_open"].iloc[-1] + pd.Timedelta(minutes=10)  # last bar not closed yet
    out = drop_unclosed_bar(df, "M15", now)
    assert len(out) == len(df) - 1
    assert out["ts_open"].iloc[-1] == df["ts_open"].iloc[-2]


def test_drop_unclosed_bar_keeps_all_when_closed():
    df = df5()
    now = df["ts_open"].iloc[-1] + pd.Timedelta(minutes=15)  # exactly closed
    out = drop_unclosed_bar(df, "M15", now)
    assert len(out) == len(df)
