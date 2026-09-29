"""FXB analysis indicators (pure numpy/pandas, no TA-Lib).

Global look-ahead rule (applies to every module here):
- Each function returns a DataFrame with the SAME index as the input frame.
- The value at row ``t`` may use ONLY rows ``<= t`` (causal / point-in-time).
- Anything that needs future bars to confirm (swings, order blocks) is
  stamped at its CONFIRMATION bar and carries the original bar time in a
  separate column; before confirmation the value is NaN/False.
- Truncation invariance: ``f(df.iloc[:k]) == f(df).iloc[:k]`` for every
  prefix length ``k``. Tests enforce this on seeded synthetic series.

Candle schema (standard frames): tz-aware UTC DatetimeIndex ``ts_open``
(bar OPEN time), columns ``open, high, low, close`` (bid prices),
``tick_volume``, ``spread`` (points). Spot forex has NO real volume: never
use CVD/delta/order-flow concepts.

Modules: basic (ATR/ADX/ER/ATR-pct/displacement), structure (swings/BOS/CHoCH),
fvg (fair value gaps + state), order_blocks, liquidity (session ranges,
prior-day levels, EQH/EQL, sweeps), regime, sessions (NY killzones),
premium_discount.
"""

from fxb.indicators import (  # noqa: F401
    basic,
    fvg,
    liquidity,
    order_blocks,
    premium_discount,
    regime,
    sessions,
    structure,
)
