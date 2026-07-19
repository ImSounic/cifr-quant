"""PHASE 3.1 — text-adjacent attention & sentiment diagnostics.

The first Phase-3 candidates: numerical features derived from what people READ
and FEEL, not from prices. (GDELT news-tone is the third planned source —
rate-limited from the dev IP at build time; fetch retries there are in
fetch_attention-style pacing and can run from the HPC.)

DECLARED (Phase-3 batch 1: 5 cells, gate |t| > ~2.6 + stability):
  A. ATTENTION SHOCKS (Wikipedia pageviews, daily, 2015-):
     shock_i(t) = ln(1+views_t) - mean_28d(ln(1+views)).
     Hypothesis (Barber-Odean / Da-Engelberg-Gao): attention spikes mark
     retail buying climaxes -> NEGATIVE forward returns.
     Cells: XS (long bottom-third shock, short top-third) x {1d, 5d};
            TS Bitcoin sign(-shock) x {5d}.
     Fixed article list below; 404s excluded by availability (no post-hoc picks).
  B. SENTIMENT CONTRARIAN (alternative.me Fear & Greed, daily, 2018-):
     sign(50 - value): long fear, short greed, on BTC. Cells: {5d, 21d}.
  Signals at day t predict returns from t+1 (day-t views complete at t+1 00:00
  UTC; F&G shifted the same way — conservative).

Usage: python scripts/attention_skill.py     # fetches/caches its own data
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.base_config import DATA_RAW_DIR

ATT_DIR = DATA_RAW_DIR / "attention"
DERIVS_DIR = DATA_RAW_DIR / "derivs"
UA = {"User-Agent": "cifr-quant-research (contact: rajakkaraju@gmail.com)"}

ARTICLES = {
    "BTC/USDT": "Bitcoin",
    "ETH/USDT": "Ethereum",
    "BNB/USDT": "BNB",
    "SOL/USDT": "Solana_(blockchain_platform)",
    "XRP/USDT": "XRP",
    "ADA/USDT": "Cardano_(blockchain_platform)",
    "AVAX/USDT": "Avalanche_(blockchain_platform)",
    "DOGE/USDT": "Dogecoin",
    "DOT/USDT": "Polkadot_(blockchain_platform)",
    "LINK/USDT": "Chainlink_(blockchain)",
    "UNI/USDT": "Uniswap",
    "ATOM/USDT": "Cosmos_(blockchain)",
    "LTC/USDT": "Litecoin",
    "NEAR/USDT": "NEAR_Protocol",
    "MATIC/USDT": "Polygon_(blockchain_platform)",
}
SHOCK_LB = 28
GATE_NOTE = "5 cells -> |t| > ~2.6 + stability"


def fetch_wiki(article):
    p = ATT_DIR / f"wiki_{article}.json"
    if not p.exists():
        url = (f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
               f"en.wikipedia/all-access/user/{article}/daily/20150701/20260718")
        req = urllib.request.Request(url, headers=UA)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(urllib.request.urlopen(req, timeout=60).read())
        time.sleep(0.5)
    d = json.loads(p.read_text())
    items = d.get("items", [])
    if not items:
        raise ValueError("no items")
    s = pd.Series({pd.Timestamp(i["timestamp"][:8]): float(i["views"]) for i in items})
    return s.sort_index()


def fetch_fng():
    p = ATT_DIR / "fng.json"
    if not p.exists():
        req = urllib.request.Request("https://api.alternative.me/fng/?limit=0&format=json",
                                     headers=UA)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(urllib.request.urlopen(req, timeout=60).read())
    d = json.loads(p.read_text())
    s = pd.Series({pd.Timestamp(int(x["timestamp"]), unit="s").normalize(): float(x["value"])
                   for x in d["data"]})
    return s.sort_index()


def daily_closes(sym):
    safe = sym.replace("/", "_").lower()
    p = DERIVS_DIR / f"{safe}_perp_1h.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    px = df.set_index("timestamps")["close"].resample("D").last()
    return px


def tstat(x):
    x = x.dropna()
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))), len(x)


def buckets(p, ann, min_n=60):
    g = p.groupby((p.index.year // 3) * 3)
    line = [f"{b}:{v.mean() * ann:+.0%}" for b, v in g if len(v) >= min_n]
    pos = sum(1 for b, v in g if len(v) >= min_n and v.mean() > 0)
    tot = sum(1 for b, v in g if len(v) >= min_n)
    return "  ".join(line), pos, tot


def main():
    print(f"{'='*72}\n  PHASE 3.1 — ATTENTION & SENTIMENT SKILL  ({GATE_NOTE})\n{'='*72}",
          flush=True)

    views, rets = {}, {}
    for sym, art in ARTICLES.items():
        try:
            v = fetch_wiki(art)
        except Exception as e:
            print(f"  no wiki data {sym} ({art}): {str(e)[:50]} — excluded", flush=True)
            continue
        px = daily_closes(sym)
        if px is None:
            print(f"  no price data {sym} — excluded", flush=True)
            continue
        views[sym] = v
        rets[sym] = px.pct_change()
    print(f"  panel: {len(views)} coins with views+prices", flush=True)

    V = pd.DataFrame(views).sort_index()
    R = pd.DataFrame(rets).reindex(V.index)
    lv = np.log1p(V)
    shock = lv - lv.rolling(SHOCK_LB).mean()

    ANN = 365
    # --- A: XS cells (hypothesis: HIGH shock -> negative, so long low / short high)
    for h in (1, 5):
        F = R.shift(-1).rolling(h, min_periods=h).sum().shift(-(h - 1))
        step = shock.index[::h]
        rows = []
        for t in step:
            s, f = shock.loc[t].dropna(), F.loc[t] if t in F.index else None
            if f is None:
                continue
            s = s[f.reindex(s.index).notna()]
            if len(s) < 6:
                continue
            k = max(len(s) // 3, 1)
            order = s.sort_values()
            rows.append((t, f[order.index[:k]].mean() - f[order.index[-k:]].mean()))
        p = pd.Series(dict(rows)).sort_index()
        t_, n = tstat(p)
        bl, pos, tot = buckets(p, ANN / h, min_n=40)
        print(f"\n  A/XS h={h}d (long quiet / short spiking): "
              f"{p.mean() * ANN / h:+.1%}/yr  t={t_:+.2f}  n={n}", flush=True)
        print(f"      3y buckets: {bl}  -> positive {pos}/{tot}", flush=True)

    # --- A: TS BTC cell
    h = 5
    fb = R["BTC/USDT"].shift(-1).rolling(h, min_periods=h).sum().shift(-(h - 1))
    sb = -shock["BTC/USDT"]
    step = sb.index[::h]
    p = (np.sign(sb.reindex(step)) * fb.reindex(step)).dropna()
    t_, n = tstat(p)
    print(f"\n  A/TS-BTC h=5d (sign of -shock): {p.mean() * ANN / h:+.1%}/yr  "
          f"t={t_:+.2f}  n={n}", flush=True)

    # --- B: Fear&Greed contrarian on BTC
    fng = fetch_fng()
    sig = np.sign(50.0 - fng).reindex(R.index).ffill(limit=2)
    for h in (5, 21):
        fb = R["BTC/USDT"].shift(-1).rolling(h, min_periods=h).sum().shift(-(h - 1))
        step = R.index[::h]
        p = (sig.reindex(step) * fb.reindex(step)).dropna()
        t_, n = tstat(p)
        bl, pos, tot = buckets(p, ANN / h, min_n=24 if h == 21 else 40)
        print(f"\n  B/FNG h={h}d (long fear / short greed): {p.mean() * ANN / h:+.1%}/yr  "
              f"t={t_:+.2f}  n={n}", flush=True)
        print(f"      3y buckets: {bl}  -> positive {pos}/{tot}", flush=True)

    print(f"\n  GATE: {GATE_NOTE}.", flush=True)


if __name__ == "__main__":
    main()
