"""Short-term reversal backtest (earned via momo_skill.py, June 10 2026).

What earned it: 7d-lookback -> 1d-forward XS rank IC -0.024, pooled t=-2.86
(clears the 2.6 multiplicity bar), negative 4/5 years (2023 t=-2.20, 2024
t=-2.47; 2022 flat +0.44). Found by scan (hypothesis was momentum), so treated
with the marginal-pass discount: cost-honest backtest, declared configs, low
expectations.

Strategy: each day at 00:00, rank assets by trailing 7d return. LONG the K
biggest LOSERS, SHORT the K biggest WINNERS, dollar-neutral, equal weight.
Hold one day. Funding cashflows on the held perp positions are modeled
(overlap with funding ranks measured at +0.017 so they should net ~0 — we
model them anyway). Costs on |Δw| only. Optional band hysteresis.

Usage:
    python scripts/backtest_reversal.py --tag rev_maker --cost-per-side 0.0003
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.base_config import DATA_RAW_DIR, RESULTS_DIR

DERIVS_DIR = DATA_RAW_DIR / "derivs"


def load_daily():
    """Daily close panel + daily-summed funding panel from the derivs files."""
    from configs.crypto_universe import get_crypto_configs
    symbols = list(get_crypto_configs(tiers=(1, 2)).keys())

    closes, funding = {}, {}
    for sym in symbols:
        safe = sym.replace("/", "_").lower()
        p_path = DERIVS_DIR / f"{safe}_perp_1h.csv"
        f_path = DERIVS_DIR / f"{safe}_funding.csv"
        if not (p_path.exists() and f_path.exists()):
            print(f"  SKIP {sym}: missing derivs data", flush=True)
            continue
        p = pd.read_csv(p_path)
        p["timestamps"] = pd.to_datetime(p["timestamps"])
        closes[sym] = p.set_index("timestamps")["close"].resample("1D").last()
        f = pd.read_csv(f_path)
        f["timestamps"] = pd.to_datetime(f["timestamps"]).dt.round("h")
        funding[sym] = f.set_index("timestamps")["funding_rate"].resample("1D").sum()
    return pd.DataFrame(closes).sort_index(), pd.DataFrame(funding).sort_index()


def run(close, fund_daily, *, lookback=7, k=3, exit_band=0,
        cost_per_side=0.0003, capital=100_000.0):
    sig = close / close.shift(lookback) - 1.0      # trailing return (known at t)
    assets = list(close.columns)
    last_w = pd.Series(0.0, index=assets)

    rows = []
    for i in range(lookback, len(close) - 1):
        t = close.index[i]
        s_row = sig.iloc[i]
        px_t = close.iloc[i]
        usable = s_row.dropna().index.intersection(px_t.dropna().index)
        if len(usable) < 2 * k + 1:
            continue

        order = list(s_row[usable].sort_values().index)   # losers first
        rank_of = {a: r for r, a in enumerate(order)}
        n = len(order)

        prev_longs = [a for a in assets if last_w.get(a, 0) > 0 and a in rank_of]
        prev_shorts = [a for a in assets if last_w.get(a, 0) < 0 and a in rank_of]
        keep_l = sorted([a for a in prev_longs if rank_of[a] < k + exit_band],
                        key=lambda a: rank_of[a])[:k]
        keep_s = sorted([a for a in prev_shorts if rank_of[a] >= n - k - exit_band],
                        key=lambda a: -rank_of[a])[:k]

        longs = list(keep_l)                              # LONG the losers
        for a in order:
            if len(longs) >= k:
                break
            if a not in longs and a not in keep_s:
                longs.append(a)
        shorts = list(keep_s)                             # SHORT the winners
        for a in reversed(order):
            if len(shorts) >= k:
                break
            if a not in shorts and a not in longs:
                shorts.append(a)

        w = pd.Series(0.0, index=assets)
        w[longs] = 0.5 / k
        w[shorts] = -0.5 / k

        turnover = float((w - last_w).abs().sum())
        cost = turnover * cost_per_side

        ret = (close.iloc[i + 1] / px_t - 1.0).reindex(assets).fillna(0.0)
        price_pnl = float((w * ret).sum())
        t1 = close.index[i + 1]
        f_row = (fund_daily.loc[t1].reindex(assets).fillna(0.0)
                 if t1 in fund_daily.index else pd.Series(0.0, index=assets))
        fund_pnl = float(-(w * f_row).sum())

        rows.append({"timestamp": t, "price_pnl": price_pnl, "funding_pnl": fund_pnl,
                     "cost": cost, "net": price_pnl + fund_pnl - cost,
                     "turnover": turnover})
        last_w = w

    df = pd.DataFrame(rows).set_index("timestamp")
    df["equity"] = capital * (1.0 + df["net"]).cumprod()
    return df


def report(df, args):
    ppy = 365
    rets = df["net"]
    years = len(rets) / ppy
    total = df["equity"].iloc[-1] / df["equity"].iloc[0] - 1.0
    ann = (1.0 + total) ** (1.0 / max(years, 1e-9)) - 1.0
    sharpe = rets.mean() / rets.std(ddof=1) * np.sqrt(ppy) if rets.std(ddof=1) > 0 else float("nan")
    eq = df["equity"]
    dd = float(((eq - eq.cummax()) / eq.cummax()).min())

    print(f"\n{'='*64}\n  REVERSAL BACKTEST  lb={args.lookback}d K={args.k} "
          f"band={args.exit_band} cost={args.cost_per_side:.2%}/side  "
          f"({years:.1f}y)\n{'='*64}", flush=True)
    print(f"  Total return:   {total:+.1%}   Ann.: {ann:+.1%}", flush=True)
    print(f"  Sharpe:         {sharpe:+.2f}", flush=True)
    print(f"  Max drawdown:   {dd:+.1%}", flush=True)
    print(f"  Avg turnover:   {df['turnover'].mean():.2f} gross/day", flush=True)
    print(f"  PnL decomposition:", flush=True)
    print(f"    price PnL:   {df['price_pnl'].sum():+.2%}", flush=True)
    print(f"    funding PnL: {df['funding_pnl'].sum():+.2%}", flush=True)
    print(f"    costs:       {-df['cost'].sum():+.2%}", flush=True)
    print(f"\n  Per-year:", flush=True)
    for year, ydf in df.groupby(df.index.year):
        yr = ydf["net"]
        ys = yr.mean() / yr.std(ddof=1) * np.sqrt(ppy) if yr.std(ddof=1) > 0 else float("nan")
        print(f"    {year}: net={yr.sum():+7.2%}  Sharpe={ys:+5.2f}  "
              f"price={ydf['price_pnl'].sum():+7.2%}  funding={ydf['funding_pnl'].sum():+6.2%}  "
              f"cost={-ydf['cost'].sum():+6.2%}", flush=True)
    return {"total_return": total, "annualized": ann, "sharpe": sharpe, "max_dd": dd,
            "avg_turnover": float(df["turnover"].mean()),
            "price_pnl": float(df["price_pnl"].sum()),
            "funding_pnl": float(df["funding_pnl"].sum()),
            "costs": float(df["cost"].sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback", type=int, default=7)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--exit-band", type=int, default=1)
    ap.add_argument("--cost-per-side", type=float, default=0.0003)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    close, fund_daily = load_daily()
    print(f"Panel: {close.shape[0]} days × {close.shape[1]} assets", flush=True)

    df = run(close, fund_daily, lookback=args.lookback, k=args.k,
             exit_band=args.exit_band, cost_per_side=args.cost_per_side)
    summary = report(df, args)

    out_dir = RESULTS_DIR / "backtest"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    df["equity"].to_csv(out_dir / f"equity_reversal{tag}.csv", header=["equity"])
    with open(out_dir / f"reversal_backtest{tag}.json", "w") as f:
        json.dump({"config": vars(args), "summary": summary}, f, indent=2, default=str)
    print(f"\nSaved to {out_dir / f'reversal_backtest{tag}.json'}", flush=True)


if __name__ == "__main__":
    main()
