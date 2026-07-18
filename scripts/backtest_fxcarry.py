"""FX CARRY earned backtest (gate passed July 18 via pre-declared retest:
G10 t=+2.66 9/9 buckets; EM t=+4.03 6/6; early-G10 t=+2.05).

Construction FROZEN from the validated diagnostics — two books, as tested:
  G10 book: top-3 long / bottom-3 short by prior-month rate differential
  EM book:  top-2/bottom-2 (top-1/bottom-1 when 4-5 live)
Each scaled to GROSS 1.0 (0.5 per leg), monthly rebalance. What this run adds
is reality: per-side costs on turnover — G10 0.05%, EM 0.20% (EM spreads are
real; the diagnostic was gross). Carry accrual is in the returns already
(prior-month rate differential / 12). Spot-plus-rate is an idealization of a
rolled-forward implementation; the cost knob absorbs the roll friction.

Trial ledger: 1 config + 1 pessimistic scenario (costs x2). No tuning.

Usage:
    python scripts/backtest_fxcarry.py
    python scripts/backtest_fxcarry.py --cost-mult 2   # pessimistic
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.base_config import RESULTS_DIR
from fx_carry_skill import UNIVERSE as G10_UNIVERSE
from fx_carry_retest import EM_UNIVERSE, build_panel

COST = {"g10": 0.0005, "em": 0.0020}


def run_book(Cl, R, min_live, kmax, cost_side, label):
    w_prev = pd.Series(dtype=float)
    rows = []
    for t in R.index:
        c, r = Cl.loc[t].dropna(), R.loc[t]
        c = c[r.reindex(c.index).notna()]
        if len(c) < min_live:
            continue
        k = kmax if len(c) >= 3 * kmax else 1
        order = c.sort_values()
        w = pd.Series(0.0, index=c.index)
        w[order.index[-k:]] = 0.5 / k
        w[order.index[:k]] = -0.5 / k
        turn = float((w.subtract(w_prev, fill_value=0.0)).abs().sum())
        gross = float((w * r.reindex(w.index)).sum())
        rows.append({"t": t, "gross": gross, "cost": turn * cost_side,
                     "net": gross - turn * cost_side, "turnover": turn})
        w_prev = w
    df = pd.DataFrame(rows).set_index("t")
    return df


def stats(df, label):
    r = df["net"]
    years = len(r) / 12
    eq = (1 + r).cumprod()
    ann = eq.iloc[-1] ** (1 / years) - 1
    sharpe = r.mean() / r.std(ddof=1) * np.sqrt(12)
    dd = float(((eq - eq.cummax()) / eq.cummax()).min())
    gross_ann = (1 + df["gross"]).prod() ** (1 / years) - 1
    print(f"  {label:22s} {ann:+6.1%}/yr net ({gross_ann:+.1%} gross)  "
          f"Sharpe {sharpe:+.2f}  maxDD {dd:+.1%}  turnover {df['turnover'].mean():.2f}/mo  "
          f"({years:.1f}y)", flush=True)
    pos = sum(1 for y, v in r.groupby(r.index.year) if len(v) >= 10 and (1 + v).prod() > 1)
    tot = sum(1 for y, v in r.groupby(r.index.year) if len(v) >= 10)
    print(f"  {'':22s} positive years: {pos}/{tot}", flush=True)
    return {"ann_net": ann, "ann_gross": gross_ann, "sharpe": sharpe,
            "max_dd": dd, "pos_years": pos, "n_years": tot}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost-mult", type=float, default=1.0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    print(f"{'='*72}\n  FX CARRY EARNED BACKTEST  (costs x{args.cost_mult:.0f}: "
          f"G10 {COST['g10'] * args.cost_mult:.2%}/side, "
          f"EM {COST['em'] * args.cost_mult:.2%}/side)\n{'='*72}", flush=True)

    Cl_g, R_g = build_panel(G10_UNIVERSE)
    Cl_e, R_e = build_panel(EM_UNIVERSE)

    print(flush=True)
    g10 = run_book(Cl_g, R_g, min_live=6, kmax=3,
                   cost_side=COST["g10"] * args.cost_mult, label="g10")
    em = run_book(Cl_e, R_e, min_live=4, kmax=2,
                  cost_side=COST["em"] * args.cost_mult, label="em")
    s_g = stats(g10, "G10 book (top3/bot3)")
    s_e = stats(em, "EM book (top2/bot2)")

    both = pd.DataFrame({"g10": g10["net"], "em": em["net"]}).dropna()
    rho = both["g10"].corr(both["em"])
    combo = both.mean(axis=1)
    print(f"\n  G10-EM correlation: {rho:+.2f} "
          f"(overlap {both.index[0].date()} .. {both.index[-1].date()})", flush=True)
    s_c = stats(pd.DataFrame({"net": combo, "gross": combo,
                              "turnover": 0}, index=combo.index), "50/50 combined")

    out = RESULTS_DIR / "backtest"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    with open(out / f"fxcarry_backtest{tag}.json", "w") as f:
        json.dump({"config": {"cost_mult": args.cost_mult, "cost": COST},
                   "g10": s_g, "em": s_e, "corr": rho, "combined": s_c},
                  f, indent=2, default=str)
    print(f"\nSaved to {out / f'fxcarry_backtest{tag}.json'}", flush=True)


if __name__ == "__main__":
    main()
