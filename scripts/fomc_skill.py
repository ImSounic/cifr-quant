"""PHASE 3.2 — FOMC event-window diagnostic (macro events as a feature).

Hypothesis (Lucca-Moench pre-FOMC drift, documented in equities; crypto
versions reported post-2019): risk assets earn abnormal returns around FOMC
announcements — the market pays a premium for bearing announcement risk.

DATA: FOMC announcement dates parsed from federalreserve.gov (minutes-link
dates, cached); BTC/ETH daily returns spliced from Binance spot (2017-21,
the attention_retest panel) + perp 1h->daily (2022-); EIA energy panel.

DECLARED (3 cells, gate |t| > 2.5 + per-year stability):
  window = announcement day t and t-1 (UTC daily bars; announcement is 19:00
  UTC on t, so day-t's bar includes the reaction — this is an EVENT-WINDOW
  test, not pure pre-drift; declared as such).
  1. crypto (BTC+ETH avg) event-window daily mean vs 0
  2. crypto event-window mean MINUS non-event mean
  3. energy panel (roll-repaired, 2017-2024) event vs non-event

Usage: python scripts/fomc_skill.py
"""

import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.base_config import DATA_RAW_DIR
from attention_skill import daily_closes
from attention_retest import fetch_spot_daily

MACRO_DIR = DATA_RAW_DIR / "macro"
UA = {"User-Agent": "Mozilla/5.0 (cifr-quant research)"}


def fomc_dates():
    p = MACRO_DIR / "fomc_dates.csv"
    if not p.exists():
        urls = ["https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"] + [
            f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{y}.htm"
            for y in (2017, 2018, 2019)]
        dates = set()
        for u in urls:
            try:
                html = urllib.request.urlopen(
                    urllib.request.Request(u, headers=UA), timeout=60).read().decode(
                    "utf-8", "ignore")
                dates.update(re.findall(r"fomcminutes(\d{8})", html))
            except Exception as e:
                print(f"  (fetch {u.rsplit('/', 1)[-1]}: {str(e)[:50]})", flush=True)
            time.sleep(1)
        p.parent.mkdir(parents=True, exist_ok=True)
        pd.Series(sorted(dates)).to_csv(p, index=False, header=["date"])
    d = pd.read_csv(p, dtype=str)["date"]
    return pd.to_datetime(d, format="%Y%m%d")


def crypto_daily(sym):
    spot = fetch_spot_daily(sym)          # capped at 2021-12-31
    perp = daily_closes(sym)              # 2022+
    px = pd.concat([spot, perp[perp.index > spot.index[-1]]]).sort_index()
    return px.pct_change()


def report(ev, nv, label):
    t_ev = ev.mean() / (ev.std(ddof=1) / np.sqrt(len(ev)))
    d = ev.mean() - nv.mean()
    se = np.sqrt(ev.var(ddof=1) / len(ev) + nv.var(ddof=1) / len(nv))
    print(f"\n  {label}", flush=True)
    print(f"    event-window mean {ev.mean():+.3%}/d (t={t_ev:+.2f}, n={len(ev)})  "
          f"vs non-event {nv.mean():+.3%}/d", flush=True)
    print(f"    difference {d:+.3%}/d  t={d / se:+.2f}", flush=True)
    yr = ev.groupby(ev.index.year).mean()
    pos = int((yr > 0).sum())
    print(f"    per-year event-window sign: positive {pos}/{len(yr)}  "
          f"({'  '.join(f'{y}:{v:+.2%}' for y, v in yr.items())})", flush=True)
    return t_ev, d / se


def main():
    print(f"{'='*72}\n  FOMC EVENT-WINDOW SKILL  (3 cells, |t|>2.5)\n{'='*72}",
          flush=True)
    ann = fomc_dates()
    print(f"  {len(ann)} FOMC announcements {ann.iloc[0].date()} .. "
          f"{ann.iloc[-1].date()}", flush=True)
    ev_days = set(ann) | set(ann - pd.Timedelta(days=1))

    # crypto book
    r = pd.concat([crypto_daily("BTC/USDT"), crypto_daily("ETH/USDT")], axis=1)
    book = r.mean(axis=1, skipna=True).dropna()
    m = book.index.isin(ev_days)
    t1, t2 = report(book[m], book[~m], "CRYPTO (BTC+ETH):")

    # energy panel
    from tscarry_skill import load_asset, detect_rolls_and_returns
    e = {}
    for key in ("wti", "natgas", "heatoil", "rbob"):
        df = load_asset(key)
        rr, _ = detect_rolls_and_returns(df)
        e[key] = np.exp(rr) - 1
    eb = pd.DataFrame(e).mean(axis=1, skipna=True).dropna()
    eb = eb[eb.index >= "2017-01-01"]
    m = eb.index.isin(ev_days)
    t3, _ = report(eb[m], eb[~m], "ENERGY (4-asset avg, 2017-2024):")

    print(f"\n  GATE: |t| > 2.5 on the declared cells (crypto vs 0; crypto "
          f"difference; energy difference) + per-year stability.", flush=True)


if __name__ == "__main__":
    main()
