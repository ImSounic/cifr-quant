"""Winter-premium seasonality diagnostic (batch-2 candidate #3 — ONE cell).

Hypothesis (declared, not fished): heating-demand commodities (natgas, heating
oil) earn a positive excess return in the pre-winter/winter window Sep 1-Jan 31
— utilities and distributors buy price insurance ahead of heating season, and
whoever holds the long side collects the hedging premium.

Deliberately ONE cell: the equal-weight NG+HO book held long Sep-Jan and flat
otherwise, one observation per winter (non-overlapping by construction).
Testing 6 assets x 12 months would be a 72-cell fishing trip; this is the
opposite. Membership in batch-2 family (13 cells, gate |t| > ~2.9 + stability).

Usage: python scripts/seasonality_skill.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tscarry_skill import load_asset, detect_rolls_and_returns

ASSETS = ["natgas", "heatoil"]


def main():
    print(f"{'='*70}\n  WINTER SEASONALITY SKILL  (long NG+HO Sep1-Jan31; 1 cell)\n{'='*70}",
          flush=True)
    rets = {}
    for key in ASSETS:
        df = load_asset(key)
        r, _ = detect_rolls_and_returns(df)
        rets[key] = r
    R = pd.DataFrame(rets).sort_index()
    book = R.mean(axis=1, skipna=True)

    in_window = R.index.month.isin([9, 10, 11, 12, 1])
    # label each winter by its starting year (Jan belongs to prior Sep's winter)
    winter_year = np.where(R.index.month == 1, R.index.year - 1, R.index.year)

    w = pd.Series(book[in_window].to_numpy(),
                  index=winter_year[in_window]).groupby(level=0).sum()
    w = w[w.index >= 1994]                       # both assets live
    o = pd.Series(book[~in_window].to_numpy(),
                  index=R.index.year[~in_window]).groupby(level=0).sum()
    o = o[o.index >= 1994]

    tw = w.mean() / (w.std(ddof=1) / np.sqrt(len(w)))
    to = o.mean() / (o.std(ddof=1) / np.sqrt(len(o)))
    print(f"\n  Winter (Sep-Jan) book return: {w.mean():+.1%}/winter  t={tw:+.2f}  "
          f"n={len(w)} winters  positive {int((w > 0).sum())}/{len(w)}", flush=True)
    print(f"  Rest of year (control):       {o.mean():+.1%}/period  t={to:+.2f}  "
          f"n={len(o)}", flush=True)
    d = w.mean() - o.mean()
    se = np.sqrt(w.var(ddof=1) / len(w) + o.var(ddof=1) / len(o))
    print(f"  Winter-minus-rest spread:     {d:+.1%}  t={d / se:+.2f}", flush=True)
    print(f"\n  GATE: |t| > ~2.9 (batch-2 family of 13 cells).", flush=True)


if __name__ == "__main__":
    main()
