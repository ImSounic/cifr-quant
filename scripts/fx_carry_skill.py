"""G10 FX CARRY diagnostic (batch-2 candidate #4).

Hypothesis: currencies with high short rates outperform low-rate currencies
vs USD (uncovered interest parity fails; the classic carry premium). The
payer: borrowers who insist on funding in low-rate currencies and investors
who over-hedge them. The most-documented carry premium in existence.

DATA (free, keyless): FRED fredgraph.csv —
  spots: daily USD crosses (DEX* series), back to the 1970s
  rates: OECD 3-month interbank per country (IR3TIB01*M156N, monthly; several
         were discontinued 2023-24 with the OECD MEI revamp — the panel ends
         where the rates end, and that recency gap is recorded honestly)

DECLARED (batch-2 family: 13 cells, gate |t| > ~2.9 + stability):
  monthly panel; carry_i = (rate_i - rate_US) using the PRIOR month's print
  (publication-lag safe); return = spot log-change + carry/12.
  cells (2 of 13):
    XS: long top-3 / short bottom-3 carry, equal weight, monthly
    TS: per currency, long vs USD if carry>0 else short, equal weight

Usage: python scripts/fx_carry_skill.py          # fetches if data missing
"""

import io
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.base_config import DATA_RAW_DIR

FX_DIR = DATA_RAW_DIR / "fx"

# ccy: (FRED spot series, spot is USD-per-ccy?, FRED 3m rate series)
UNIVERSE = {
    "EUR": ("DEXUSEU", True,  "IR3TIB01EZM156N"),
    "JPY": ("DEXJPUS", False, "IR3TIB01JPM156N"),
    "GBP": ("DEXUSUK", True,  "IR3TIB01GBM156N"),
    "CAD": ("DEXCAUS", False, "IR3TIB01CAM156N"),
    "AUD": ("DEXUSAL", True,  "IR3TIB01AUM156N"),
    "CHF": ("DEXSZUS", False, "IR3TIB01CHM156N"),
    "SEK": ("DEXSDUS", False, "IR3TIB01SEM156N"),
    "NOK": ("DEXNOUS", False, "IR3TIB01NOM156N"),
    "NZD": ("DEXUSNZ", True,  "IR3TIB01NZM156N"),
}
US_RATE = "IR3TIB01USM156N"


def fred(series):
    p = FX_DIR / f"{series}.csv"
    if not p.exists():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        raw = urllib.request.urlopen(req, timeout=60).read()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
        time.sleep(0.5)
    df = pd.read_csv(p)
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna().set_index("date")["value"]


def main():
    print(f"{'='*72}\n  G10 FX CARRY SKILL  (FRED spots + OECD 3m rates; gate |t|>2.9)\n{'='*72}",
          flush=True)
    spot_m, carry = {}, {}
    us = fred(US_RATE)
    us_m = us.resample("ME").last()
    print(f"  US 3m rate: {us.index[0].date()} .. {us.index[-1].date()}", flush=True)
    for ccy, (spot_s, usd_per_ccy, rate_s) in UNIVERSE.items():
        try:
            sp = fred(spot_s)
            rt = fred(rate_s)
        except Exception as e:
            print(f"  SKIP {ccy}: {str(e)[:60]}", flush=True)
            continue
        sp = sp if usd_per_ccy else 1.0 / sp
        m = np.log(sp.resample("ME").last())
        rate_m = rt.resample("ME").last()
        spot_m[ccy] = m
        carry[ccy] = (rate_m - us_m)
        print(f"  {ccy}: spot {sp.index[0].year}-{sp.index[-1].year}, "
              f"rate {rt.index[0].year}-{rt.index[-1].year}", flush=True)

    S = pd.DataFrame(spot_m)
    C = pd.DataFrame(carry).reindex(S.index)
    # next-month total return of being long ccy vs USD; carry uses PRIOR month print
    ret_fx = S.diff().shift(-1)
    Cl = C.shift(1)
    R = ret_fx.add(Cl / 100 / 12, fill_value=np.nan)
    R = R.shift(0)

    both = Cl.notna() & R.notna()
    live_months = both.sum(axis=1)
    ok = live_months >= 6
    print(f"\n  Panel: {int(ok.sum())} usable months "
          f"({live_months[ok].index[0].date()} .. {live_months[ok].index[-1].date()})",
          flush=True)

    # --- XS cell
    xs = []
    for t in R.index[ok]:
        c, r = Cl.loc[t].dropna(), R.loc[t]
        c = c[r.reindex(c.index).notna()]
        if len(c) < 6:
            continue
        order = c.sort_values()
        xs.append((t, r[order.index[-3:]].mean() - r[order.index[:3]].mean()))
    xs = pd.Series(dict(xs)).sort_index()
    t_xs = xs.mean() / (xs.std(ddof=1) / np.sqrt(len(xs)))
    print(f"\n  XS (top3-bottom3): {xs.mean() * 12:+.1%}/yr  t={t_xs:+.2f}  n={len(xs)}",
          flush=True)

    # --- TS cell
    ts = (np.sign(Cl) * R).mean(axis=1, skipna=True)[ok].dropna()
    t_ts = ts.mean() / (ts.std(ddof=1) / np.sqrt(len(ts)))
    print(f"  TS (sign of carry): {ts.mean() * 12:+.1%}/yr  t={t_ts:+.2f}  n={len(ts)}",
          flush=True)

    # stability: 5y buckets (monthly data, ~30y)
    for name, p in [("XS", xs), ("TS", ts)]:
        g = p.groupby((p.index.year // 5) * 5)
        bl = [f"{b}s:{v.mean() * 12:+.0%}" for b, v in g if len(v) >= 24]
        pos = sum(1 for b, v in g if len(v) >= 24 and v.mean() > 0)
        tot = sum(1 for b, v in g if len(v) >= 24)
        print(f"  {name} 5y buckets: {'  '.join(bl)}  -> positive {pos}/{tot}", flush=True)

    print(f"\n  GATE: |t| > ~2.9 (batch-2 family of 13 cells) + stability.", flush=True)


if __name__ == "__main__":
    main()
