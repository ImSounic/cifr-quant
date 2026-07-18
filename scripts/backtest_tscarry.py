"""Term-structure carry EARNED backtest (gate passed July 18, 2026 — t=+3.35).

Construction FROZEN from the diagnostic (tscarry_skill.py — single source of
truth: signal, smoothing, returns and roll handling are imported from it):
  - signal: sign of 5d-smoothed (c1/c2 - 1) at close t  (backwardation -> long)
  - book:   equal notional across live assets (4-6 over 1985-2024), TS only
  - cadence: rebalance every 5 business days (the horizon cell that passed)

What this run ADDS is reality, not tuning:
  - execution cost on traded notional at every rebalance (default 0.03%/side,
    conservative for CL, ~fair for NG/HO/RB)
  - roll cost: 2 sides x cost whenever a held contract rolls (~monthly/asset)
  - per-year decomposition, Sharpe, maxDD, cost drag (gross vs net)

Trial ledger: this is config #1 for this signal. Cost knobs exist for scenario
robustness (0.03% vs 0.06%), NOT for parameter search. Any change to signal
construction is a NEW strategy that must re-earn the gate.

Usage:
    python scripts/backtest_tscarry.py
    python scripts/backtest_tscarry.py --cost-per-side 0.0006   # pessimistic scenario
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
from tscarry_skill import load_asset, detect_rolls_and_returns, ASSETS, SMOOTH_D


def build_panel():
    sig, rets, rolls = {}, {}, {}
    for key in ASSETS:
        df = load_asset(key)
        if df is None:
            continue
        r, rl = detect_rolls_and_returns(df)
        rets[key] = r
        rolls[key] = rl
        sig[key] = (df["c1"] / df["c2"] - 1.0).rolling(SMOOTH_D, min_periods=1).mean()
    S = pd.DataFrame(sig).sort_index()
    R = pd.DataFrame(rets).reindex(S.index)
    L = pd.DataFrame(rolls).reindex(S.index).fillna(False)
    return S, R, L


def run(S, R, L, rebalance_every=5, cost_per_side=0.0003):
    idx = S.index
    assets = list(S.columns)
    w = pd.Series(0.0, index=assets)
    rows = []
    for i in range(len(idx)):
        t = idx[i]
        # daily gross pnl of the held book (log->simple per asset)
        gross = 0.0
        for a in assets:
            r = R.at[t, a]
            if w[a] != 0.0 and np.isfinite(r):
                gross += w[a] * (np.exp(r) - 1.0)
        # roll cost: pay the spread twice on each held asset's roll day
        roll_cost = sum(2.0 * abs(w[a]) * cost_per_side
                        for a in assets if w[a] != 0.0 and L.at[t, a])
        # rebalance at close every N days
        turn = 0.0
        if i % rebalance_every == 0:
            live = [a for a in assets
                    if np.isfinite(S.at[t, a]) and np.isfinite(R.at[t, a])]
            new_w = pd.Series(0.0, index=assets)
            if live:
                for a in live:
                    new_w[a] = np.sign(S.at[t, a]) / len(live)
            turn = float((new_w - w).abs().sum())
            w = new_w
        cost = turn * cost_per_side + roll_cost
        rows.append({"timestamp": t, "gross": gross, "cost": cost,
                     "net": gross - cost, "turnover": turn,
                     "n_live": int((w != 0).sum())})
    df = pd.DataFrame(rows).set_index("timestamp")
    df["equity"] = (1.0 + df["net"]).cumprod()
    return df


def report(df, cost_per_side):
    rets = df["net"]
    years = len(rets) / 252
    total = df["equity"].iloc[-1] - 1.0
    ann = (1.0 + total) ** (1 / years) - 1.0
    sharpe = rets.mean() / rets.std(ddof=1) * np.sqrt(252)
    eq = df["equity"]
    dd = float(((eq - eq.cummax()) / eq.cummax()).min())
    gross_ann = (1.0 + df["gross"]).prod() ** (1 / years) - 1.0

    print(f"\n{'='*68}\n  TS-CARRY BACKTEST  rebal=5bd  cost={cost_per_side:.2%}/side  "
          f"{df.index[0].date()} .. {df.index[-1].date()} (~{years:.1f}y)\n{'='*68}",
          flush=True)
    print(f"  Ann. return:  {ann:+.1%} net   ({gross_ann:+.1%} gross — cost drag "
          f"{gross_ann - ann:.1%}/yr)", flush=True)
    print(f"  Sharpe:       {sharpe:+.2f}", flush=True)
    print(f"  Max drawdown: {dd:+.1%}", flush=True)
    print(f"  Avg turnover: {df['turnover'][df['turnover'] > 0].mean():.2f} gross/rebal  "
          f"(sign flips are rare — carry persists)", flush=True)
    print(f"  Live assets:  {df['n_live'].min()}..{df['n_live'].max()}", flush=True)

    print(f"\n  Per-year:", flush=True)
    pos = tot = 0
    for y, ydf in df.groupby(df.index.year):
        if len(ydf) < 100:
            continue
        yr = ydf["net"]
        ys = yr.mean() / yr.std(ddof=1) * np.sqrt(252) if yr.std(ddof=1) > 0 else np.nan
        net = (1 + yr).prod() - 1
        tot += 1
        pos += net > 0
        print(f"    {y}: net={net:+7.1%}  Sharpe={ys:+5.2f}  "
              f"cost={ydf['cost'].sum():.2%}", flush=True)
    print(f"\n  Positive years: {pos}/{tot}", flush=True)
    return {"ann_net": ann, "ann_gross": gross_ann, "sharpe": sharpe, "max_dd": dd,
            "years": years, "pos_years": pos, "n_years": tot}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebalance-every", type=int, default=5)
    ap.add_argument("--cost-per-side", type=float, default=0.0003)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    S, R, L = build_panel()
    print(f"Panel: {len(S)} days x {S.shape[1]} assets "
          f"({S.index[0].date()} .. {S.index[-1].date()})", flush=True)
    df = run(S, R, L, rebalance_every=args.rebalance_every,
             cost_per_side=args.cost_per_side)
    summary = report(df, args.cost_per_side)

    out_dir = RESULTS_DIR / "backtest"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    df["equity"].to_csv(out_dir / f"equity_tscarry{tag}.csv", header=["equity"])
    with open(out_dir / f"tscarry_backtest{tag}.json", "w") as f:
        json.dump({"config": vars(args), "summary": summary}, f, indent=2, default=str)
    print(f"\nSaved to {out_dir / f'tscarry_backtest{tag}.json'}", flush=True)


if __name__ == "__main__":
    main()
