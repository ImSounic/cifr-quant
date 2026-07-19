"""PHASE 3.1 RETEST — attention-continuation & sentiment-momentum on
PRE-2022 data (declared before fetching).

Origin: the July 19 attention/sentiment diagnostic (attention_skill.py) FAILED
its declared contrarian hypotheses, but both came out INVERTED with internal
consistency on 2022-2026 perp data: attention spikes CONTINUE (XS t=-2.00 on
the contrarian construction) and fear precedes further losses. Sign flips
after seeing data are forbidden — but an inverted lead may earn a NEW
pre-declared test on data it has never seen.

INDEPENDENT DATA: Binance SPOT daily klines 2017-01-01..2021-12-31 (never used
by any diagnostic in this repo) + the same Wikipedia pageviews (2015-) and
Fear&Greed (2018-) series over that window only.

DECLARED CELLS (3, gate |t| > 2.5, sign must MATCH the 2022-26 observation):
  1. XS attention-CONTINUATION (long top-third shock / short bottom-third),
     h=1d;  2. same, h=5d;
  3. FNG MOMENTUM on BTC (long greed / short fear), h=5d.
Window hard-capped at 2021-12-31. Same shock construction (28d), same t+1
alignment.

Usage: python scripts/attention_retest.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.base_config import DATA_RAW_DIR
from attention_skill import ARTICLES, fetch_wiki, fetch_fng, SHOCK_LB, tstat, buckets

SPOT_DIR = DATA_RAW_DIR / "spot_daily"
END = "2021-12-31"


def fetch_spot_daily(sym):
    safe = sym.replace("/", "_").lower()
    p = SPOT_DIR / f"{safe}_1d.csv"
    if not p.exists():
        import ccxt
        ex = ccxt.binance({"enableRateLimit": True})
        since = ex.parse8601("2017-01-01T00:00:00Z")
        end_ms = ex.parse8601("2022-01-01T00:00:00Z")
        candles = []
        while since < end_ms:
            batch = ex.fetch_ohlcv(sym, "1d", since=since, limit=1000)
            if not batch:
                break
            candles.extend(batch)
            since = batch[-1][0] + 1
            time.sleep(ex.rateLimit / 1000)
        if not candles:
            return None
        df = pd.DataFrame(candles, columns=["ts", "o", "h", "l", "close", "v"])
        df["timestamps"] = pd.to_datetime(df["ts"], unit="ms")
        p.parent.mkdir(parents=True, exist_ok=True)
        df[["timestamps", "close"]].to_csv(p, index=False)
    df = pd.read_csv(p)
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    s = df.set_index("timestamps")["close"]
    return s[s.index <= END]


def main():
    print(f"{'='*72}\n  PHASE 3.1 RETEST — pre-2022 spot panel  (3 cells, |t|>2.5, "
          f"declared signs)\n{'='*72}", flush=True)
    views, rets = {}, {}
    for sym in ARTICLES:
        try:
            v = fetch_wiki(ARTICLES[sym])
        except Exception:
            continue
        px = fetch_spot_daily(sym)
        if px is None or len(px) < 200:
            print(f"  {sym}: no/short spot history pre-2022 — excluded", flush=True)
            continue
        views[sym] = v[v.index <= END]
        rets[sym] = px.pct_change()
        print(f"  {sym}: spot {px.index[0].date()} .. {px.index[-1].date()}", flush=True)

    V = pd.DataFrame(views).sort_index()
    R = pd.DataFrame(rets).reindex(V.index)
    lv = np.log1p(V)
    shock = lv - lv.rolling(SHOCK_LB).mean()
    ANN = 365

    # cells 1-2: XS CONTINUATION (long spiking / short quiet)
    for h in (1, 5):
        F = R.shift(-1).rolling(h, min_periods=h).sum().shift(-(h - 1))
        rows = []
        for t in shock.index[::h]:
            if t not in F.index:
                continue
            s, f = shock.loc[t].dropna(), F.loc[t]
            s = s[f.reindex(s.index).notna()]
            if len(s) < 6:
                continue
            k = max(len(s) // 3, 1)
            order = s.sort_values()
            rows.append((t, f[order.index[-k:]].mean() - f[order.index[:k]].mean()))
        p = pd.Series(dict(rows)).sort_index()
        t_, n = tstat(p)
        bl, pos, tot = buckets(p, ANN / h, min_n=40)
        print(f"\n  XS continuation h={h}d: {p.mean() * ANN / h:+.1%}/yr  t={t_:+.2f}  "
              f"n={n}", flush=True)
        print(f"      buckets: {bl}  -> positive {pos}/{tot}", flush=True)

    # cell 3: FNG momentum on BTC (long greed / short fear), 2018-02..2021
    fng = fetch_fng()
    fng = fng[fng.index <= END]
    sig = np.sign(fng - 50.0).reindex(R.index).ffill(limit=2)
    h = 5
    fb = R["BTC/USDT"].shift(-1).rolling(h, min_periods=h).sum().shift(-(h - 1))
    p = (sig.reindex(R.index[::h]) * fb.reindex(R.index[::h])).dropna()
    t_, n = tstat(p)
    bl, pos, tot = buckets(p, ANN / h, min_n=30)
    print(f"\n  FNG momentum h=5d (long greed/short fear): {p.mean() * ANN / h:+.1%}/yr  "
          f"t={t_:+.2f}  n={n}", flush=True)
    print(f"      buckets: {bl}  -> positive {pos}/{tot}", flush=True)

    print(f"\n  DECLARED RULE: pass = |t|>2.5 with sign matching the 2022-26\n"
          f"  observation (continuation positive / momentum positive) + stability.",
          flush=True)


if __name__ == "__main__":
    main()
