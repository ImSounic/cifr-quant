"""FX CARRY — pre-declared independent RETEST (follow-up to the batch-2
near-miss: G10 XS t=+2.66 with 9/9 positive 5y buckets, below the 13-cell
family gate).

RULES (declared before any data was touched — see PROJECT_STATE):
  Cell 1 (PRIMARY, EM/extended panel): fixed candidate list = currencies with
    both a FRED daily H.10 spot and an OECD 3m rate, outside the original nine:
    MXN BRL ZAR KRW INR CNY DKK. Availability from this fixed list decides
    membership — no post-hoc selection. Same construction as the original:
    monthly, prior-month rate print, XS top-2/bottom-2 (>=6 live) or
    top-1/bottom-1 (4-5 live).
  Cell 2 (SECONDARY, early-G10): the original nine, 1971..1982-01 — the decade
    BEFORE the original panel began (>=4 live instead of >=6).
  PASS = primary t >= +2.5 with majority-positive 5y buckets AND secondary
  positive sign. Then FX carry = brick #3 on joint evidence. Tails reported
  (peso risk); the earned backtest must then use realistic EM spreads.

Usage: python scripts/fx_carry_retest.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fx_carry_skill import fred, UNIVERSE as G10_UNIVERSE, US_RATE

EM_UNIVERSE = {
    "MXN": ("DEXMXUS", False, "IR3TIB01MXM156N"),
    "BRL": ("DEXBZUS", False, "IR3TIB01BRM156N"),
    "ZAR": ("DEXSFUS", False, "IR3TIB01ZAM156N"),
    "KRW": ("DEXKOUS", False, "IR3TIB01KRM156N"),
    "INR": ("DEXINUS", False, "IR3TIB01INM156N"),
    "CNY": ("DEXCHUS", False, "IR3TIB01CNM156N"),
    "DKK": ("DEXDNUS", False, "IR3TIB01DKM156N"),
}


def build_panel(universe):
    spot_m, carry = {}, {}
    us_m = fred(US_RATE).resample("ME").last()
    for ccy, (spot_s, usd_per_ccy, rate_s) in universe.items():
        try:
            sp = fred(spot_s)
            rt = fred(rate_s)
        except Exception as e:
            print(f"  no data {ccy} ({str(e)[:50]}) — excluded by availability",
                  flush=True)
            continue
        sp = sp if usd_per_ccy else 1.0 / sp
        spot_m[ccy] = np.log(sp.resample("ME").last())
        carry[ccy] = fred(rate_s).resample("ME").last() - us_m
        print(f"  {ccy}: spot {sp.index[0].year}-{sp.index[-1].year}, "
              f"rate {rt.index[0].year}-{rt.index[-1].year}", flush=True)
    S = pd.DataFrame(spot_m)
    C = pd.DataFrame(carry).reindex(S.index)
    R = S.diff().shift(-1).add(C.shift(1) / 100 / 12, fill_value=np.nan)
    return C.shift(1), R


def xs_cell(Cl, R, min_live, label, start=None, end=None):
    rows = []
    for t in R.index:
        if start is not None and t < pd.Timestamp(start):
            continue
        if end is not None and t > pd.Timestamp(end):
            continue
        c, r = Cl.loc[t].dropna(), R.loc[t]
        c = c[r.reindex(c.index).notna()]
        if len(c) < min_live:
            continue
        k = 2 if len(c) >= 6 else 1
        order = c.sort_values()
        rows.append((t, r[order.index[-k:]].mean() - r[order.index[:k]].mean()))
    p = pd.Series(dict(rows)).sort_index()
    if len(p) < 24:
        print(f"\n  {label}: only {len(p)} months — insufficient", flush=True)
        return None
    t_stat = p.mean() / (p.std(ddof=1) / np.sqrt(len(p)))
    print(f"\n  {label}: {p.mean() * 12:+.1%}/yr  t={t_stat:+.2f}  n={len(p)} months "
          f"({p.index[0].date()} .. {p.index[-1].date()})", flush=True)
    print(f"    tails: worst month {p.min():+.1%}   5th pct {p.quantile(0.05):+.1%}   "
          f"skew {p.skew():+.1f}", flush=True)
    g = p.groupby((p.index.year // 5) * 5)
    bl = [f"{b}s:{v.mean() * 12:+.0%}" for b, v in g if len(v) >= 24]
    pos = sum(1 for b, v in g if len(v) >= 24 and v.mean() > 0)
    tot = sum(1 for b, v in g if len(v) >= 24)
    print(f"    5y buckets: {'  '.join(bl)}  -> positive {pos}/{tot}", flush=True)
    return {"t": t_stat, "mean_ann": p.mean() * 12, "n": len(p),
            "buckets_pos": pos, "buckets_tot": tot}


def main():
    print(f"{'='*72}\n  FX CARRY RETEST  (pre-declared: EM primary + early-G10 "
          f"secondary)\n{'='*72}", flush=True)

    print("\nPrimary panel (EM/extended, fixed list):", flush=True)
    Cl_em, R_em = build_panel(EM_UNIVERSE)
    em = xs_cell(Cl_em, R_em, min_live=4, label="PRIMARY  EM XS carry")

    print("\nSecondary panel (original G10, pre-1982 window):", flush=True)
    Cl_g, R_g = build_panel(G10_UNIVERSE)
    early = xs_cell(Cl_g, R_g, min_live=4, label="SECONDARY early-G10 XS carry",
                    end="1982-01-31")

    print(f"\n{'='*72}\n  DECLARED PASS RULE: primary t>=+2.5 with majority-positive "
          f"buckets\n  AND secondary positive sign.", flush=True)
    if em and early:
        ok = (em["t"] >= 2.5 and em["buckets_pos"] > em["buckets_tot"] / 2
              and early["t"] > 0)
        print(f"  VERDICT: {'PASS — FX carry graduates on joint evidence' if ok else 'FAIL'}",
              flush=True)


if __name__ == "__main__":
    main()
