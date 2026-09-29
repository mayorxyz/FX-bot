"""Tests for fxb.indicators.structure: confirmation stamps, BOS/CHoCH, invariance."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxb.indicators.structure import market_structure, swings
from helpers import candles_from_ohlc, synth_series, truncation_cut_points


def _pyramid_series():
    """Hand-built series with ONE clean swing high at a known index.

    Bars rise to a peak then fall; left=right=3 so the peak needs 3 lower
    highs on each side. Peak placed at index 8.
    """
    rows = []
    heights = [1.0000, 1.0002, 1.0004, 1.0006, 1.0008, 1.0010, 1.0012,
               1.0014, 1.0020,          # <- peak bar body/high at index 8
               1.0014, 1.0012, 1.0010, 1.0008, 1.0006, 1.0004]
    for i, hh in enumerate(heights):
        o = hh - 0.0002
        c = hh - 0.0001
        l = hh - 0.0005
        rows.append((o, hh, l, c))
    return candles_from_ohlc(rows)


def test_swing_high_confirmation_row_and_original_ts():
    df = _pyramid_series()
    sw = swings(df, left=3, right=3)
    peak_idx = 8
    peak_price = df["high"].iloc[peak_idx]
    # nothing confirmed before peak+right
    assert sw["swing_high_price"].iloc[: peak_idx + 3].isna().all()
    # confirmed exactly `right` bars later
    conf_idx = peak_idx + 3
    assert sw["swing_high_price"].iloc[conf_idx] == pytest.approx(peak_price)
    assert sw["swing_high_bar_ts"].iloc[conf_idx] == df.index[peak_idx]
    # only that one row carries it
    nn = sw["swing_high_price"].notna()
    assert nn.idxmax() == df.index[conf_idx]
    assert nn.to_numpy().sum() == 1


def test_swing_low_mirror():
    rows = []
    lows = [1.0100, 1.0090, 1.0080, 1.0070, 1.0060, 1.0050, 1.0040,
            1.0030, 1.0010,             # trough at index 8
            1.0030, 1.0040, 1.0050, 1.0060, 1.0070, 1.0080]
    for ll in lows:
        rows.append((ll + 0.0002, ll + 0.0005, ll, ll + 0.0001))
    df = candles_from_ohlc(rows)
    sw = swings(df, left=3, right=3)
    conf_idx = 8 + 3
    assert sw["swing_low_price"].iloc[conf_idx] == pytest.approx(df["low"].iloc[8])
    assert sw["swing_low_bar_ts"].iloc[conf_idx] == df.index[8]


def _bos_choch_frame():
    """Series engineered for: downtrend -> CHoCH (bullish break) -> BOS.

    Structure (left=right=2 used in test):
      * swing low L1 at idx 4 (deep), swing high H1 at idx 9
      * price makes lower low L2 at idx 14 (trend still down)
      * strong rally closes above H1 at idx 19  -> CHoCH (+1)
      * continuation closes above next confirmed high at idx 25 -> BOS (+1)
    """
    rows = [
        (1.010, 1.011, 1.009, 1.010),   # 0
        (1.009, 1.010, 1.008, 1.009),   # 1
        (1.008, 1.009, 1.007, 1.008),   # 2
        (1.007, 1.008, 1.006, 1.007),   # 3
        (1.006, 1.007, 1.000, 1.001),   # 4  swing low L1=1.000
        (1.002, 1.003, 1.001, 1.002),   # 5
        (1.003, 1.004, 1.002, 1.003),   # 6
        (1.004, 1.005, 1.003, 1.004),   # 7
        (1.005, 1.006, 1.004, 1.005),   # 8
        (1.006, 1.010, 1.005, 1.009),   # 9  swing high H1=1.010
        (1.008, 1.009, 1.007, 1.008),   # 10
        (1.006, 1.007, 1.005, 1.006),   # 11
        (1.004, 1.005, 1.003, 1.004),   # 12
        (1.002, 1.003, 1.001, 1.002),   # 13
        (1.001, 1.002, 0.995, 0.996),   # 14 swing low L2=0.995
        (0.997, 0.998, 0.996, 0.997),   # 15
        (0.998, 0.999, 0.997, 0.998),   # 16
        (1.000, 1.001, 0.999, 1.000),   # 17
        (1.005, 1.006, 1.004, 1.005),   # 18
        (1.011, 1.012, 1.010, 1.0115),  # 19 close > H1 -> CHoCH
        (1.011, 1.012, 1.010, 1.011),   # 20
        (1.012, 1.013, 1.011, 1.012),   # 21
        (1.013, 1.016, 1.012, 1.015),   # 22 swing high H2=1.016
        (1.014, 1.015, 1.013, 1.014),   # 23
        (1.013, 1.014, 1.012, 1.013),   # 24
        (1.017, 1.018, 1.016, 1.0175),  # 25 close > H2 -> BOS
        (1.017, 1.018, 1.016, 1.017),   # 26
        (1.016, 1.017, 1.015, 1.016),   # 27
    ]
    return candles_from_ohlc(rows)


def test_bos_then_choch_sequence():
    df = _bos_choch_frame()
    ms = market_structure(df, left=2, right=2)
    # trend must be DOWN (-1) before the CHoCH: verify via earlier breaks
    # first bearish break establishes -1 somewhere at/before idx ~ 17
    pre = ms["trend"].iloc[:19].to_numpy()
    assert (pre == -1).any(), "expected a down-trend state before CHoCH"
    # CHoCH bullish event occurs when close crosses H1 after its confirmation
    choch_rows = ms.index[ms["choch"] & (ms["break_direction"] == 1)]
    assert len(choch_rows) >= 1
    first_choch = df.index.get_loc(choch_rows[0])
    assert first_choch >= 19  # cannot fire before the rally bar's data exists
    # trend flips to +1 at the CHoCH and stays
    assert ms["trend"].iloc[first_choch] == 1
    # a later bullish BOS exists (continuation beyond next confirmed high)
    bos_rows = ms.index[ms["bos"] & (ms["break_direction"] == 1)]
    later_bos = [df.index.get_loc(ts) for ts in bos_rows if df.index.get_loc(ts) > first_choch]
    assert later_bos, "expected a bullish BOS after the CHoCH"
    # broken_level equals a previously CONFIRMED swing high price
    lvl = ms["broken_level"].iloc[first_choch]
    assert lvl in set(ms["last_swing_high"].dropna().round(6).tolist())


def test_breaks_are_close_based_not_wick():
    # wick pokes above the level but close stays below -> no break
    rows = [
        (1.005, 1.006, 1.004, 1.005),
        (1.004, 1.005, 1.003, 1.004),
        (1.003, 1.010, 1.002, 1.003),   # idx2 huge upper wick, close low
        (1.002, 1.003, 1.001, 1.002),
        (1.001, 1.002, 1.000, 1.001),
        (1.002, 1.003, 1.001, 1.002),
        (1.003, 1.004, 1.002, 1.003),
    ]
    df = candles_from_ohlc(rows)
    ms = market_structure(df, left=2, right=2)
    # swing high of 1.010 confirms at idx 4; no close ever exceeds 1.010
    assert not ms["bos"].any() and not ms["choch"].any()


@pytest.mark.parametrize("kind", ["trend", "range", "gapped", "mixed"])
def test_structure_truncation_invariance(kind):
    df = synth_series(600, kind, seed={"trend": 8, "range": 9,
                                       "gapped": 10, "mixed": 7}[kind])
    cuts = truncation_cut_points(len(df), k=20, seed=13)
    full_sw = swings(df)
    full_ms = market_structure(df)
    for k in cuts:
        sub = df.iloc[:k]
        pd.testing.assert_frame_equal(swings(sub), full_sw.iloc[:k])
        pd.testing.assert_frame_equal(market_structure(sub), full_ms.iloc[:k])
