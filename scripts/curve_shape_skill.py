"""Energy curve-SHAPE diagnostics (batch-2 candidates #2a/#2b).

Two declared signals on the EIA c1..c4 panel, distinct from brick #2's LEVEL
of slope:
  2a SLOPE MOMENTUM: 63bd change in the smoothed front slope. Hypothesis:
     a steepening curve (rising backwardation) is bullish before the level
     itself peaks — trade the derivative, not the level.
  2b CURVATURE (fly): ln c1 - 2 ln c2 + ln c3. No strong prior on sign
     (two-sided) — the front kink may mean near-term stress (bullish) or
     richness (bearish); the gate decides.

DECLARED (part of the batch-2 family: 13 cells, gate |t| > ~2.9 + stability):
  cells = {slope-mom, fly} x TS x {5bd, 21bd} = 4 of the 13
  construction: sign(signal), equal weight, same roll-repaired returns as
  brick #2's gauntlet.

Usage: python scripts/curve_shape_skill.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tscarry_skill import load_asset, detect_rolls_and_returns, ASSETS, SMOOTH_D

MOM_LOOKBACK = 63
HORIZONS = [5, 21]


def bucket(ts):
    lo = ts.year - (ts.year - 1985) % 3
    return f"{lo}-{lo + 2}"


def run_cell(name, S, R, h):
    F = R.shift(-1).rolling(h, min_periods=h).sum().reindex(S.index).shift(-(h - 1))
    step = S.index[::h]
    Ss, Fs = S.reindex(step), F.reindex(step)
    port = (np.sign(Ss) * Fs).mean(axis=1, skipna=True).dropna()
    t = port.mean() / (port.std(ddof=1) / np.sqrt(len(port)))
    ann = 252 / h
    print(f"  {name:11s} h={h:2d}bd: mean={port.mean() * ann:+.1%}/yr  t={t:+.2f}  "
          f"n={len(port)}", flush=True)
    if h == 21:
        g = port.groupby(port.index.map(bucket))
        pos = sum(1 for b, v in g if len(v) >= 8 and v.mean() > 0)
        tot = sum(1 for b, v in g if len(v) >= 8)
        print(f"{'':25s}buckets positive: {pos}/{tot}", flush=True)
    return t


def main():
    slope, fly, rets = {}, {}, {}
    print(f"{'='*72}\n  CURVE-SHAPE SKILL  (slope-momentum + fly; gate |t|>2.9)\n{'='*72}",
          flush=True)
    for key in ASSETS:
        df = load_asset(key)
        if df is None:
            continue
        r, _ = detect_rolls_and_returns(df)
        rets[key] = r
        s = (df["c1"] / df["c2"] - 1.0).rolling(SMOOTH_D, min_periods=1).mean()
        slope[key] = s - s.shift(MOM_LOOKBACK)
        if "c3" in df.columns:
            fly[key] = (np.log(df["c1"]) - 2 * np.log(df["c2"])
                        + np.log(df["c3"])).rolling(SMOOTH_D, min_periods=1).mean()
    R = pd.DataFrame(rets).sort_index()
    SM = pd.DataFrame(slope).reindex(R.index)
    FL = pd.DataFrame(fly).reindex(R.index)

    for h in HORIZONS:
        run_cell("slope-mom", SM, R, h)
    for h in HORIZONS:
        run_cell("fly", FL, R, h)

    print(f"\n  GATE: |t| > ~2.9 (batch-2 family) + stability; fly is two-sided.",
          flush=True)


if __name__ == "__main__":
    main()
