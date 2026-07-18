"""Energy futures TIME-SERIES MOMENTUM diagnostic (batch-2 candidate #1).

Hypothesis: trailing 12-month excess return predicts continuation (Moskowitz-
Ooi-Pedersen TSMOM) — the canonical managed-futures signal, in the asset class
where it is actually documented (crypto TSMOM died in this gauntlet June 10).
A pass would add a MOMENTUM-family brick, decorrelated from both carry bricks
by signal type — valuable precisely because carry crashes in risk-off and
momentum tends to pay then.

DECLARED (batch-2 family: 13 cells total -> gate |t| > ~2.9 + bucket stability):
  signal  = sign of trailing 252bd sum of roll-repaired front excess log-returns
  cells   = TS equal-weight x horizons {5bd, 21bd}   (2 cells of the 13)
  data    = EIA roll-repaired energy panel (same returns as brick #2's gauntlet)

Usage: python scripts/tsmom_commod_skill.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tscarry_skill import load_asset, detect_rolls_and_returns, ASSETS

LOOKBACK = 252
HORIZONS = [5, 21]


def bucket(ts):
    lo = ts.year - (ts.year - 1985) % 3
    return f"{lo}-{lo + 2}"


def main():
    rets = {}
    print(f"{'='*72}\n  ENERGY TSMOM SKILL  (12m lookback; batch-2 gate |t|>2.9)\n{'='*72}",
          flush=True)
    for key in ASSETS:
        df = load_asset(key)
        if df is None:
            continue
        r, _ = detect_rolls_and_returns(df)
        rets[key] = r
    R = pd.DataFrame(rets).sort_index()
    S = R.rolling(LOOKBACK, min_periods=int(LOOKBACK * 0.8)).sum()   # 12m excess ret

    for h in HORIZONS:
        F = R.shift(-1).rolling(h, min_periods=h).sum().reindex(S.index).shift(-(h - 1))
        step = S.index[::h]
        Ss, Fs = S.reindex(step), F.reindex(step)
        port = (np.sign(Ss) * Fs).mean(axis=1, skipna=True).dropna()
        t = port.mean() / (port.std(ddof=1) / np.sqrt(len(port)))
        ann = 252 / h
        print(f"\n  h={h:2d}bd TS: mean={port.mean() * ann:+.1%}/yr  t={t:+.2f}  "
              f"n={len(port)}", flush=True)
        per = (np.sign(Ss) * Fs)
        line = []
        for k in per.columns:
            v = per[k].dropna()
            if len(v) > 30:
                tt = v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))
                line.append(f"{k}:{tt:+.1f}")
        print(f"         per-asset t: {'  '.join(line)}", flush=True)
        if h == 21:
            g = port.groupby(port.index.map(bucket))
            bl = [f"{b}:{v.mean() * 12:+.0%}" for b, v in g if len(v) >= 8]
            pos = sum(1 for b, v in g if len(v) >= 8 and v.mean() > 0)
            tot = sum(1 for b, v in g if len(v) >= 8)
            print(f"         3y buckets: {'  '.join(bl)}", flush=True)
            print(f"         positive buckets: {pos}/{tot}", flush=True)

    print(f"\n  GATE: |t| > ~2.9 (batch-2 family of 13 cells) + stability.", flush=True)


if __name__ == "__main__":
    main()
