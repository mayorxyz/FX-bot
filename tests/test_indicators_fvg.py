"""Tests for fxb.indicators.fvg: hand-built gaps, mitigation/fill, invariance."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxb.indicators.fvg import active_fvgs_at, fair_value_gaps, fvg_state
from helpers import candles_from_ohlc, synth_series, truncation_cut_points


def _bullish_fvg_frame():
    """Bars 0..2 create a bullish FVG at bar 2: low[2] > high[0]."""
    rows = [
        (1.000, 1.001, 0.999, 1.0005),   # i-2: high 1.001
        (1.003, 1.006, 1.002, 1.0055),   # i-1: displacement candle
        (1.005, 1.007, 1.004, 1.0065),   # i:   low 1.004 > 1.001 -> gap
        (1.006, 1.008, 1.005, 1.007),
    ]
    return candles_from_ohlc(rows)


def test_bullish_fvg_created_and_stamped_at_bar_i():
    df = _bullish_fvg_frame()
    ev = fair_value_gaps(df, min_size_atr=0.0)
    assert len(ev) == 1
    r = ev.iloc[0]
    assert int(r["direction"]) == 1
    assert r["created_ts"] == df.index[2]           # stamped at close of bar i
    assert r["top"] == pytest.approx(df["low"].iloc[2])
    assert r["bottom"] == pytest.approx(df["high"].iloc[0])
    assert r["midpoint"] == pytest.approx((r["top"] + r["bottom"]) / 2)
    assert r["size"] == pytest.approx(r["top"] - r["bottom"])


def _bearish_fvg_frame():
    rows = [
        (1.006, 1.007, 1.005, 1.0055),   # i-2: low 1.005
        (1.003, 1.004, 1.001, 1.0015),   # i-1: big down candle
        (1.002, 1.003, 1.000, 1.001),    # i:   high 1.003 < low[i-2]=1.005
    ]
    return candles_from_ohlc(rows)


def test_bearish_fvg():
    df = _bearish_fvg_frame()
    ev = fair_value_gaps(df, min_size_atr=0.0)
    assert len(ev) == 1
    r = ev.iloc[0]
    assert int(r["direction"]) == -1
    assert r["top"] == pytest.approx(df["low"].iloc[0])
    assert r["bottom"] == pytest.approx(df["high"].iloc[2])


def test_fvg_mitigated_then_filled():
    # bull gap (top=1.004, bottom=1.001) then touch then full fill
    rows = [
        (1.000, 1.001, 0.999, 1.0005),
        (1.003, 1.006, 1.002, 1.0055),
        (1.005, 1.007, 1.004, 1.0065),   # gap created at idx 2
        (1.006, 1.007, 1.005, 1.006),    # idx3 low 1.005 > top 1.004: untouched
        (1.005, 1.006, 1.003, 1.0035),   # idx4 low 1.003 <= 1.004 MITIGATED
        (1.003, 1.004, 1.000, 1.0005),   # idx5 close 1.0005 <= bottom FULL FILLED
    ]
    df = candles_from_ohlc(rows)
    ev = fair_value_gaps(df, min_size_atr=0.0)
    st = fvg_state(df, ev)
    assert st["n_active"].iloc[2] == 1
    assert st["n_active"].iloc[3] == 1
    assert st["n_active"].iloc[4] == 0      # mitigated -> retired
    assert st["n_mitigated"].iloc[4] == 1
    assert st["has_active_bull"].iloc[3]
    assert not st["has_active_bull"].iloc[4]
    snaps = active_fvgs_at(df, ev, 3)
    assert len(snaps[0].active_ids) == 1
    snaps4 = active_fvgs_at(df, ev, 4)
    assert len(snaps4[0].active_ids) == 0


def test_min_size_atr_filter_drops_small_gaps():
    # 14 flat bars to seed ATR (~0.0006), then a tiny 0.0002 gap vs a big one
    rows = [(1.000, 1.0004, 0.9998, 1.0002)] * 14
    rows += [
        (1.0002, 1.0003, 1.0001, 1.0002),    # idx14: high 1.0003
        (1.0005, 1.0007, 1.0004, 1.0006),    # idx15: middle candle
        (1.0006, 1.0008, 1.0005, 1.0007),    # idx16: low 1.0005 > 1.0003 -> gap 0.0002
        (1.0006, 1.0007, 1.0005, 1.0006),    # idx17: overlaps both edges -> no gap
        (1.0010, 1.0012, 1.0009, 1.0011),    # idx18: middle candle big
        (1.0014, 1.0015, 1.0013, 1.0014),    # idx19: low 1.0013 < 1.0008? no:
        #   bullish check low[19]=1.0013 > high[17]=1.0007 -> gap 0.0006 (big)
    ]
    df = candles_from_ohlc(rows)
    ev_all = fair_value_gaps(df, min_size_atr=0.0)
    ev_big = fair_value_gaps(df, min_size_atr=0.9)
    assert len(ev_all) == 2
    assert len(ev_big) == 1
    assert ev_big.iloc[0]["created_ts"] == df.index[19]


@pytest.mark.parametrize("kind", ["trend", "range", "gapped", "mixed"])
def test_fvg_truncation_invariance(kind):
    df = synth_series(600, kind, seed={"trend": 8, "range": 9,
                                       "gapped": 10, "mixed": 7}[kind])
    cuts = truncation_cut_points(len(df), k=20, seed=17)
    full_ev = fair_value_gaps(df)
    full_state = fvg_state(df, full_ev)
    for k in cuts:
        sub = df.iloc[:k]
        ev_sub = fair_value_gaps(sub)
        # events table: identical up to the cut (all stamps are <= k-1)
        pd.testing.assert_frame_equal(ev_sub.reset_index(drop=True),
                                      full_ev.reset_index(drop=True).iloc[:len(ev_sub)])
        pd.testing.assert_frame_equal(fvg_state(sub, ev_sub), full_state.iloc[:k])
