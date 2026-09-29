"""Tests for fxb.indicators.basic: hand-checked values + truncation invariance."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxb.indicators.basic import (
    adx,
    atr,
    atr_percentile,
    displacement,
    efficiency_ratio,
)
from helpers import candles_from_ohlc, synth_series, truncation_cut_points


# ---------------------------------------------------------------------------
# Wilder ATR cross-check against an explicit reference loop computed in-test.
# ---------------------------------------------------------------------------
def _ref_atr(high, low, close, n=14):
    trs = []
    for i in range(len(high)):
        if i == 0:
            trs.append(high[i] - low[i])
        else:
            trs.append(max(high[i] - low[i], abs(high[i] - close[i - 1]),
                           abs(low[i] - close[i - 1])))
    out = [np.nan] * len(trs)
    if len(trs) < n:
        return out
    seed = sum(trs[:n]) / n
    out[n - 1] = seed
    prev = seed
    for i in range(n, len(trs)):
        prev = (prev * (n - 1) + trs[i]) / n
        out[i] = prev
    return out


def test_atr_matches_reference_loop(synth_mixed):
    df = synth_mixed.iloc[:120]
    got = atr(df, n=14)["atr"].to_numpy()
    exp = np.array(_ref_atr(df["high"].tolist(), df["low"].tolist(),
                            df["close"].tolist(), n=14))
    np.testing.assert_allclose(got, exp, rtol=0, atol=1e-12, equal_nan=True)


def test_atr_first_value_position_and_nan_before(synth_trend):
    df = synth_trend.iloc[:50]
    got = atr(df, n=14)["atr"]
    assert got.iloc[:13].isna().all()
    assert np.isfinite(got.iloc[13])


# ---------------------------------------------------------------------------
# Wilder ADX cross-check with an explicit loop (Wilder sums variant).
# ---------------------------------------------------------------------------
def _ref_adx(high, low, close, n=14):
    m = len(high)
    trs, pdms, mdms = [], [], []
    for i in range(m):
        if i == 0:
            trs.append(high[i] - low[i]); pdms.append(0.0); mdms.append(0.0)
        else:
            trs.append(max(high[i] - low[i], abs(high[i] - close[i - 1]),
                           abs(low[i] - close[i - 1])))
            up = high[i] - high[i - 1]
            dn = low[i - 1] - low[i]
            pdms.append(up if (up > dn and up > 0) else 0.0)
            mdms.append(dn if (dn > up and dn > 0) else 0.0)
    s_tr = [np.nan] * m; s_p = [np.nan] * m; s_m = [np.nan] * m
    if m < n:
        return None
    a_tr, a_p, a_m = sum(trs[:n]), sum(pdms[:n]), sum(mdms[:n])
    s_tr[n - 1], s_p[n - 1], s_m[n - 1] = a_tr, a_p, a_m
    for i in range(n, m):
        a_tr += trs[i] - a_tr / n
        a_p += pdms[i] - a_p / n
        a_m += mdms[i] - a_m / n
        s_tr[i], s_p[i], s_m[i] = a_tr, a_p, a_m
    pdi = [100 * p / t if np.isfinite(t) and t > 0 else np.nan
           for p, t in zip(s_p, s_tr)]
    mdi = [100 * mm / t if np.isfinite(t) and t > 0 else np.nan
           for mm, t in zip(s_m, s_tr)]
    dx = []
    for p, q in zip(pdi, mdi):
        if not (np.isfinite(p) and np.isfinite(q)):
            dx.append(np.nan)
        elif p + q > 0:
            dx.append(100 * abs(p - q) / (p + q))
        else:
            dx.append(0.0)
    adxv = [np.nan] * m
    first = n - 1
    seed_pos = first + n - 1
    if m >= seed_pos + 1:
        acc = sum(dx[first:first + n]) / n
        adxv[seed_pos] = acc
        for i in range(seed_pos + 1, m):
            d = dx[i] if np.isfinite(dx[i]) else 0.0
            acc = (acc * (n - 1) + d) / n
            adxv[i] = acc
    return np.array(adxv), np.array(pdi), np.array(mdi)


def test_adx_matches_reference_loop(synth_mixed):
    df = synth_mixed.iloc[:160]
    got = adx(df, n=14)
    ref = _ref_adx(df["high"].tolist(), df["low"].tolist(), df["close"].tolist(), 14)
    a_exp, p_exp, m_exp = ref
    np.testing.assert_allclose(got["adx"].to_numpy(), a_exp,
                               rtol=1e-12, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(got["plus_di"].to_numpy(), p_exp,
                               rtol=1e-12, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(got["minus_di"].to_numpy(), m_exp,
                               rtol=1e-12, atol=1e-12, equal_nan=True)


def test_adx_first_value_at_2n_minus_2(synth_range):
    df = synth_range.iloc[:60]
    got = adx(df, n=14)["adx"]
    assert got.iloc[:26].isna().all()          # rows 0..25 NaN
    assert np.isfinite(got.iloc[26])           # first at 2*14-2 = 26


# ---------------------------------------------------------------------------
# Efficiency ratio: hand check.
# ---------------------------------------------------------------------------
def test_efficiency_ratio_hand_computed():
    # closes 1..6 rising by 1 each step except a reversal: net/sum
    closes = pd.Series([1.0, 2.0, 3.0, 2.0, 3.0, 4.0])
    got = efficiency_ratio(closes, n=3)["er"]
    # at row 3: |c3-c0| / (|2-1|+|3-2|+|2-3|) = 1/3
    assert got.iloc[3] == pytest.approx(1.0 / 3.0)
    # at row 5: net |c5-c2| = |4-3| = 1; steps rows 3,4,5: |2-3|+|3-2|+|4-3|
    #        = 1+1+1 = 3  -> ER = 1/3
    assert got.iloc[5] == pytest.approx(1.0 / 3.0)
    assert got.iloc[:3].isna().all()


def test_efficiency_ratio_flat_is_zero():
    closes = pd.Series([1.5] * 30)
    got = efficiency_ratio(closes, n=20)["er"]
    assert got.iloc[20:].fillna(0.0).eq(0.0).all()


# ---------------------------------------------------------------------------
# ATR percentile bounds.
# ---------------------------------------------------------------------------
def test_atr_percentile_bounds(synth_mixed):
    got = atr_percentile(synth_mixed, n=14, window=100)["atr_pct"]
    v = got.dropna()
    assert ((v >= 0) & (v <= 100)).all()
    assert got.isna().iloc[:14].all()  # before ATR exists -> NaN


# ---------------------------------------------------------------------------
# Displacement: hand-built fixture.
# ---------------------------------------------------------------------------
def test_displacement_hand_built():
    # 15 small bars to seed ATR, then one big bullish body vs prior ATR
    rows = [(1.0, 1.0005, 0.9995, 1.0002)] * 15
    # prior ATR ~ 0.001 (range of flat bars ~0.001 after TR from gaps)
    rows.append((1.0, 1.0008, 0.9992, 1.0004))
    # big bull candle: body 0.004 >= 1.5*ATR_prior, body/range 0.004/0.004=1
    rows.append((1.0004, 1.0044, 1.0004, 1.0044))
    df = candles_from_ohlc(rows)
    got = displacement(df, atr_mult=1.5, body_ratio=0.6)
    assert not got["displacement"].iloc[:16].any()
    assert bool(got["displacement"].iloc[16])
    assert int(got["direction"].iloc[16]) == 1


def test_displacement_bearish_direction(synth_gapped):
    got = displacement(synth_gapped)
    bear_rows = got.index[got["direction"] == -1]
    if len(bear_rows):
        t = synth_gapped.index.get_loc(bear_rows[0])
        bar = synth_gapped.iloc[t]
        assert bar["close"] < bar["open"]


# ---------------------------------------------------------------------------
# Truncation invariance (>= 20 seeded cut points, all four series kinds).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["trend", "range", "gapped", "mixed"])
def test_basic_truncation_invariance(kind):
    df = synth_series(600, kind, seed={"trend": 8, "range": 9,
                                       "gapped": 10, "mixed": 7}[kind])
    cuts = truncation_cut_points(len(df), k=20, seed=11)
    full = atr(df)
    full_adx = adx(df)
    full_er = efficiency_ratio(df)
    full_pct = atr_percentile(df, n=14, window=200)
    full_disp = displacement(df)
    for k in cuts:
        sub = df.iloc[:k]
        pd.testing.assert_frame_equal(atr(sub), full.iloc[:k])
        pd.testing.assert_frame_equal(adx(sub), full_adx.iloc[:k])
        pd.testing.assert_frame_equal(efficiency_ratio(sub), full_er.iloc[:k])
        pd.testing.assert_frame_equal(
            atr_percentile(sub, n=14, window=200), full_pct.iloc[:k])
        pd.testing.assert_frame_equal(displacement(sub), full_disp.iloc[:k])
