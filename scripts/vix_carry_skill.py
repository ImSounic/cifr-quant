"""VIX TERM-STRUCTURE CARRY diagnostic (batch-2 candidate #6).

Hypothesis: when the VIX curve is in contango (VIX3M > VIX), rolling short
vol collects the insurance premium equity investors pay for crash protection;
in backwardation the premium flips. The payer: portfolio insurance buyers.
KNOWN DANGER: the short-vol side has catastrophic left tails (Feb 2018
"Volmageddon") — this candidate exists partly to see the gauntlet price that.

DATA (free): CBOE VIX & VIX3M daily index history (CSV, keyless);
SVXY (short VIX-futures ETF, 2011-) via yfinance as the TRADEABLE return —
using a real instrument's returns bakes in roll costs and the Feb 2018
leverage change (-1x -> -0.5x; sign construction is robust to leverage,
recorded as a caveat).

DECLARED (batch-2 family: 13 cells, gate |t| > ~2.9 + stability):
  signal = sign(VIX3M - VIX) at close t  (contango -> long SVXY, i.e. short
  vol; backwardation -> short SVXY)
  cells = TS x horizons {5bd, 21bd}  (2 of the 13)

Usage: python scripts/vix_carry_skill.py
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

VIX_DIR = DATA_RAW_DIR / "vix"
HORIZONS = [5, 21]


def cboe(name):
    p = VIX_DIR / f"{name}.csv"
    if not p.exists():
        url = f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{name}.csv"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(urllib.request.urlopen(req, timeout=60).read())
        time.sleep(0.5)
    df = pd.read_csv(p)
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["close"]


def main():
    print(f"{'='*72}\n  VIX TERM-STRUCTURE CARRY SKILL  (contango -> short vol; "
          f"gate |t|>2.9)\n{'='*72}", flush=True)
    vix = cboe("VIX_History")
    vix3m = cboe("VIX3M_History")

    import yfinance as yf
    import warnings
    warnings.filterwarnings("ignore")
    svxy = yf.Ticker("SVXY").history(period="max", interval="1d",
                                     auto_adjust=True)["Close"]
    svxy.index = svxy.index.tz_localize(None)
    r = svxy.pct_change()

    basis = (vix3m - vix).reindex(r.index).ffill(limit=3)
    print(f"  VIX/VIX3M: {vix.index[0].date()}..; SVXY: {r.index[0].date()} .. "
          f"{r.index[-1].date()} ({len(r)}d)", flush=True)
    print(f"  Contango frequency: {(basis > 0).mean():.0%} of days", flush=True)

    for h in HORIZONS:
        F = r.shift(-1).rolling(h, min_periods=h).sum().shift(-(h - 1))
        step = r.index[::h]
        s, f = basis.reindex(step), F.reindex(step)
        port = (np.sign(s) * f).dropna()
        t = port.mean() / (port.std(ddof=1) / np.sqrt(len(port)))
        ann = 252 / h
        print(f"\n  h={h:2d}bd TS: mean={port.mean() * ann:+.1%}/yr  t={t:+.2f}  "
              f"n={len(port)}", flush=True)
        # tail check — the construction-matched question for short vol
        print(f"         worst period: {port.min():+.1%}   5th pct: "
              f"{port.quantile(0.05):+.1%}   skew: {port.skew():+.1f}", flush=True)
        g = port.groupby((port.index.year // 3) * 3)
        bl = [f"{b}:{v.mean() * ann:+.0%}/yr" for b, v in g if len(v) >= 8]
        pos = sum(1 for b, v in g if len(v) >= 8 and v.mean() > 0)
        tot = sum(1 for b, v in g if len(v) >= 8)
        print(f"         3y buckets: {'  '.join(bl)}  -> positive {pos}/{tot}",
              flush=True)

    print(f"\n  GATE: |t| > ~2.9 (batch-2 family) + stability + acceptable tails.",
          flush=True)


if __name__ == "__main__":
    main()
