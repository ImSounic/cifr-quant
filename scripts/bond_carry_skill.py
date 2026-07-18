"""US TREASURY TERM-STRUCTURE CARRY diagnostic (batch-2 candidate #5).

Hypothesis: the term spread (long yield minus cash rate) predicts bond excess
returns — a steep curve pays you to hold duration (carry + rolldown); an
inverted curve pays you to be short. The payer: issuers/borrowers demanding
long-term funding certainty.

DATA (FRED, keyless, daily): DGS2/DGS5/DGS10/DGS30 constant-maturity yields
(1962/1977-present) + DGS3MO cash rate. Bond returns approximated from CMT
yields (zero-coupon approx): r ~= y*dt - tau*dy — standard for diagnostics.

DECLARED (batch-2 family: 13 cells, gate |t| > ~2.9 + stability):
  signal = term spread y_tau - y_3m, sign construction, equal weight across
  tenors {2,5,10,30}; cells = horizons {21bd, 63bd} (bonds are slow).

Usage: python scripts/bond_carry_skill.py
"""

import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.base_config import DATA_RAW_DIR

BOND_DIR = DATA_RAW_DIR / "bonds"
TENORS = {"DGS2": 2, "DGS5": 5, "DGS10": 10, "DGS30": 30}
CASH = "DGS3MO"
HORIZONS = [21, 63]


def fred(series):
    p = BOND_DIR / f"{series}.csv"
    if not p.exists():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(urllib.request.urlopen(req, timeout=60).read())
        time.sleep(0.5)
    df = pd.read_csv(p)
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna().set_index("date")["value"] / 100.0


def main():
    print(f"{'='*72}\n  BOND TERM-STRUCTURE CARRY SKILL  (FRED CMT; gate |t|>2.9)\n{'='*72}",
          flush=True)
    cash = fred(CASH)
    y, r_ex, sig = {}, {}, {}
    for s, tau in TENORS.items():
        ys = fred(s)
        print(f"  {s} ({tau}y): {ys.index[0].date()} .. {ys.index[-1].date()}",
              flush=True)
        # daily excess return of holding the tau-year zero (approx), vs cash
        dy = ys.diff()
        r = ys.shift(1) / 252 - tau * dy - cash.reindex(ys.index).ffill() / 252
        r_ex[s] = r
        sig[s] = ys - cash.reindex(ys.index).ffill()
    R = pd.DataFrame(r_ex).sort_index()
    S = pd.DataFrame(sig).reindex(R.index)

    for h in HORIZONS:
        F = R.shift(-1).rolling(h, min_periods=h).sum().reindex(S.index).shift(-(h - 1))
        step = S.index[::h]
        Ss, Fs = S.reindex(step), F.reindex(step)
        port = (np.sign(Ss) * Fs).mean(axis=1, skipna=True).dropna()
        t = port.mean() / (port.std(ddof=1) / np.sqrt(len(port)))
        ann = 252 / h
        print(f"\n  h={h:2d}bd TS: mean={port.mean() * ann:+.1%}/yr  t={t:+.2f}  "
              f"n={len(port)}", flush=True)
        g = port.groupby((port.index.year // 5) * 5)
        bl = [f"{b}s:{v.mean() * ann / (21 / h):+.0%}" for b, v in g if len(v) >= 12]
        pos = sum(1 for b, v in g if len(v) >= 12 and v.mean() > 0)
        tot = sum(1 for b, v in g if len(v) >= 12)
        print(f"         5y buckets positive: {pos}/{tot}", flush=True)
        if h == 21:
            print(f"         buckets: {'  '.join(bl)}", flush=True)

    print(f"\n  GATE: |t| > ~2.9 (batch-2 family of 13 cells) + stability.", flush=True)


if __name__ == "__main__":
    main()
