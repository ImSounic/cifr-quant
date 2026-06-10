"""Cross-sectional funding-carry backtest (the EARNED backtest — gate passed).

Diagnostic results that earned it (carry_skill.py, June 10 2026):
  8h XS IC −0.020, t=−4.83 pooled; negative ALL 5 years, |t|>2 in 4/5.

Strategy: at each 8h funding event, rank assets by funding rate; SHORT the
top-K highest-funding perps, LONG the bottom-K lowest-funding perps,
dollar-neutral, equal weight per leg. Both legs monetize the same effect twice:
shorts collect the high funding AND capture predicted underperformance.

PnL per 8h period = price PnL + funding cashflows − turnover costs:
  funding_pnl = −Σ w_i · rate_i(t+1)   (long pays positive funding, short receives)
  costs charged only on |Δw| (membership changes slowly — funding is persistent)
Optional hysteresis (--buffer): incumbents keep their slot unless they fall
more than `buffer` ranks past the cutoff — cuts turnover/cost drag.

Reports: equity, annualized return/Sharpe/maxDD, per-year breakdown, turnover,
and the price-vs-funding PnL decomposition (how much is prediction vs collection).

Usage:
    python scripts/backtest_carry.py                 # K=3, 8h rebalance
    python scripts/backtest_carry.py --k 4 --buffer 2 --rebalance-every 3
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
EVENTS_PER_YEAR = 3 * 365


def load_panels():
    """Aligned (events × assets) panels of funding rates and perp closes."""
    from configs.crypto_universe import get_crypto_configs
    symbols = list(get_crypto_configs(tiers=(1, 2)).keys())

    funding, closes = {}, {}
    for sym in symbols:
        safe = sym.replace("/", "_").lower()
        f_path = DERIVS_DIR / f"{safe}_funding.csv"
        p_path = DERIVS_DIR / f"{safe}_perp_1h.csv"
        if not (f_path.exists() and p_path.exists()):
            print(f"  SKIP {sym}: no derivs data", flush=True)
            continue
        f = pd.read_csv(f_path)
        # funding timestamps carry ms jitter (…00.006) — round to the hour grid
        f["timestamps"] = pd.to_datetime(f["timestamps"]).dt.round("h")
        f = f.drop_duplicates(subset="timestamps").set_index("timestamps")["funding_rate"]
        p = pd.read_csv(p_path)
        p["timestamps"] = pd.to_datetime(p["timestamps"])
        p = p.drop_duplicates(subset="timestamps").set_index("timestamps")["close"]
        funding[sym] = f
        closes[sym] = p

    fund_panel = pd.DataFrame(funding).sort_index()
    # perp close sampled AT each funding event time
    close_panel = pd.DataFrame({s: closes[s].reindex(fund_panel.index, method="ffill")
                                for s in funding}).sort_index()
    return fund_panel, close_panel


def run(fund, close, *, k=3, exit_band=0, smooth=1, rebalance_every=1,
        cost_per_side=0.0008, capital=100_000.0):
    """Walk the 8h event grid. Position w formed at event t (using rates paid up
    to and including t — known then) is held until t+rebalance_every; funding
    accrues at each intermediate event; costs on |Δw|.

    smooth:    rank on the trailing mean of the last `smooth` funding events
               (funding is persistent; single prints churn the ranks).
    exit_band: band hysteresis — enter on reaching top/bottom K, exit only when
               falling outside top/bottom (K + exit_band).
    """
    times = fund.index
    assets = list(fund.columns)
    signal = fund.rolling(smooth, min_periods=1).mean() if smooth > 1 else fund

    rows = []
    last_w = pd.Series(0.0, index=assets)
    for i in range(0, len(times) - 1, rebalance_every):
        t = times[i]
        sig_t = signal.iloc[i]
        px_t = close.iloc[i]
        usable = sig_t.dropna().index.intersection(px_t.dropna().index)
        if len(usable) < 2 * k + 1:
            continue

        order = list(sig_t[usable].sort_values().index)   # low funding first
        rank_of = {a: r for r, a in enumerate(order)}
        n = len(order)

        prev_longs = [a for a in assets if last_w.get(a, 0) > 0 and a in rank_of]
        prev_shorts = [a for a in assets if last_w.get(a, 0) < 0 and a in rank_of]

        # incumbents survive while inside the exit band; best survivors first
        keep_l = sorted([a for a in prev_longs if rank_of[a] < k + exit_band],
                        key=lambda a: rank_of[a])[:k]
        keep_s = sorted([a for a in prev_shorts if rank_of[a] >= n - k - exit_band],
                        key=lambda a: -rank_of[a])[:k]

        longs = list(keep_l)
        for a in order:                                   # fill from the top
            if len(longs) >= k:
                break
            if a not in longs and a not in keep_s:
                longs.append(a)
        shorts = list(keep_s)
        for a in reversed(order):                         # fill from the bottom
            if len(shorts) >= k:
                break
            if a not in shorts and a not in longs:
                shorts.append(a)

        w = pd.Series(0.0, index=assets)
        w[longs] = 0.5 / k                          # 0.5 gross per leg, dollar-neutral
        w[shorts] = -0.5 / k

        turnover = float((w - last_w).abs().sum())
        cost = turnover * cost_per_side

        j = min(i + rebalance_every, len(times) - 1)
        px_t1 = close.iloc[j]
        ret = (px_t1 / px_t - 1.0).reindex(assets).fillna(0.0)
        price_pnl = float((w * ret).sum())

        # funding accrues at each event strictly after t up to and incl. t_j
        fund_pnl = 0.0
        for m in range(i + 1, j + 1):
            fund_pnl += float(-(w * fund.iloc[m].reindex(assets).fillna(0.0)).sum())

        rows.append({"timestamp": t, "price_pnl": price_pnl, "funding_pnl": fund_pnl,
                     "cost": cost, "net": price_pnl + fund_pnl - cost,
                     "turnover": turnover})
        last_w = w

    df = pd.DataFrame(rows).set_index("timestamp")
    df["equity"] = capital * (1.0 + df["net"]).cumprod()
    return df


def report(df, k, rebalance_every, cost_per_side):
    ppy = EVENTS_PER_YEAR / rebalance_every
    rets = df["net"]
    years = len(rets) / ppy
    total = df["equity"].iloc[-1] / df["equity"].iloc[0] - 1.0
    ann = (1.0 + total) ** (1.0 / max(years, 1e-9)) - 1.0
    sharpe = rets.mean() / rets.std(ddof=1) * np.sqrt(ppy) if rets.std(ddof=1) > 0 else float("nan")
    eq = df["equity"]
    dd = float(((eq - eq.cummax()) / eq.cummax()).min())

    print(f"\n{'='*64}\n  CARRY BACKTEST  K={k} rebal={rebalance_every}ev "
          f"cost={cost_per_side:.2%}/side  periods={len(df)} (~{years:.1f}y)\n{'='*64}", flush=True)
    print(f"  Total return:   {total:+.1%}   Ann.: {ann:+.1%}", flush=True)
    print(f"  Sharpe:         {sharpe:+.2f}", flush=True)
    print(f"  Max drawdown:   {dd:+.1%}", flush=True)
    print(f"  Avg turnover:   {df['turnover'].mean():.2f} gross/period", flush=True)
    g = len(df)
    print(f"  PnL decomposition (sum over backtest, in return units):", flush=True)
    print(f"    price PnL:   {df['price_pnl'].sum():+.2%}   (the prediction leg)", flush=True)
    print(f"    funding PnL: {df['funding_pnl'].sum():+.2%}   (the collection leg)", flush=True)
    print(f"    costs:       {-df['cost'].sum():+.2%}", flush=True)

    print(f"\n  Per-year:", flush=True)
    for year, ydf in df.groupby(df.index.year):
        yr = ydf["net"]
        ys = yr.mean() / yr.std(ddof=1) * np.sqrt(ppy) if yr.std(ddof=1) > 0 else float("nan")
        print(f"    {year}: net={yr.sum():+7.2%}  Sharpe={ys:+5.2f}  "
              f"price={ydf['price_pnl'].sum():+7.2%}  funding={ydf['funding_pnl'].sum():+7.2%}  "
              f"cost={-ydf['cost'].sum():+6.2%}", flush=True)
    return {"total_return": total, "annualized": ann, "sharpe": sharpe, "max_dd": dd,
            "avg_turnover": float(df["turnover"].mean()),
            "price_pnl": float(df["price_pnl"].sum()),
            "funding_pnl": float(df["funding_pnl"].sum()),
            "costs": float(df["cost"].sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--exit-band", type=int, default=0,
                    help="Band hysteresis: exit only when outside top/bottom K+band")
    ap.add_argument("--smooth", type=int, default=1,
                    help="Rank on trailing mean of last N funding events (9 = 3 days)")
    ap.add_argument("--rebalance-every", type=int, default=1, help="In 8h events (3=daily)")
    ap.add_argument("--cost-per-side", type=float, default=0.0008,
                    help="Fee + slippage per side on traded notional (taker ~0.0008, maker ~0.0003)")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    fund, close = load_panels()
    print(f"Panel: {fund.shape[0]} events × {fund.shape[1]} assets "
          f"({fund.index[0]} .. {fund.index[-1]})", flush=True)

    df = run(fund, close, k=args.k, exit_band=args.exit_band, smooth=args.smooth,
             rebalance_every=args.rebalance_every, cost_per_side=args.cost_per_side)
    summary = report(df, args.k, args.rebalance_every, args.cost_per_side)

    out_dir = RESULTS_DIR / "backtest"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    df["equity"].to_csv(out_dir / f"equity_carry{tag}.csv", header=["equity"])
    with open(out_dir / f"carry_backtest{tag}.json", "w") as f:
        json.dump({"config": vars(args), "summary": summary}, f, indent=2, default=str)
    print(f"\nSaved to {out_dir / f'carry_backtest{tag}.json'}", flush=True)


if __name__ == "__main__":
    main()
