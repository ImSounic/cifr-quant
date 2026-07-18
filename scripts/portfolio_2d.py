"""Phase 2D — the portfolio layer: combine the firm's frozen bricks under
vol-targeted sizing.

In plain terms: each strategy ("brick") is good but wild — funding carry drew
down 30%, term-structure carry 60% (natgas dominates an equal-notional book).
This layer never touches the SIGNALS (frozen; changes re-earn the gate). It only
changes SIZING, using the one thing v1 proved predictable: volatility persists
(IC 0.61). Quiet assets get more weight, wild ones less, and each brick is
levered up/down weekly so its risk stays near a constant target. Two ~zero-
correlation bricks at equal risk then add, not average — the portfolio Sharpe
exceeds either brick's.

DECLARED CONFIG (one config — the risk layer was pre-declared in the plan
June 10; constants chosen a priori, not swept):
  VOL_LOOKBACK = 63 trading days (one quarter) of trailing realized vol
  TARGET_VOL   = 10%/yr per brick, and 10%/yr for the combined portfolio
  LEV_CAP      = 3x on any vol-targeting scalar (sanity bound)
  REBAL        = weekly (5bd), matching brick #2's cadence
  Across bricks: 50/50 risk split.

Bricks (imported from their frozen sources):
  #1 funding carry  — backtest_carry.run() with the frozen v2_maker_8h config
                      (K=3, smooth=9, exit band 2, maker 3bp), 8h events -> daily
  #2 TS carry       — tscarry_skill signals/returns; sign(slope) direction with
                      inverse-vol weights instead of equal notional

Outputs: vol-targeted per-brick stats (full history), overlap-window brick
correlation, combined portfolio equity/stats/per-year. Saves
results/portfolio/phase2d_*.{json,csv}.

Usage:
    python scripts/portfolio_2d.py
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.base_config import RESULTS_DIR
from backtest_carry import load_panels, run as run_carry
from tscarry_skill import load_asset, detect_rolls_and_returns, ASSETS, SMOOTH_D

VOL_LOOKBACK = 63
TARGET_VOL = 0.10
LEV_CAP = 3.0
REBAL = 5
ANN = 252


def stats(rets, label):
    rets = rets.dropna()
    years = len(rets) / ANN
    eq = (1 + rets).cumprod()
    total = eq.iloc[-1] - 1
    ann = (1 + total) ** (1 / years) - 1
    sharpe = rets.mean() / rets.std(ddof=1) * np.sqrt(ANN)
    dd = float(((eq - eq.cummax()) / eq.cummax()).min())
    vol = rets.std(ddof=1) * np.sqrt(ANN)
    print(f"  {label:34s} {ann:+6.1%}/yr  Sharpe {sharpe:+.2f}  "
          f"vol {vol:5.1%}  maxDD {dd:+.1%}  ({years:.1f}y)", flush=True)
    return {"ann": ann, "sharpe": sharpe, "vol": vol, "max_dd": dd, "years": years}


def vol_target_stream(rets, label):
    """Scale a daily return stream to TARGET_VOL using trailing realized vol,
    recomputed weekly, capped at LEV_CAP. Scalar at t uses vol through t-1."""
    vol = rets.rolling(VOL_LOOKBACK).std().shift(1) * np.sqrt(ANN)
    lev = (TARGET_VOL / vol).clip(upper=LEV_CAP)
    lev = lev.iloc[::REBAL].reindex(rets.index).ffill()   # weekly recompute
    out = (rets * lev).dropna()
    print(f"  {label}: avg leverage {lev.mean():.2f}x "
          f"(cap {LEV_CAP:.0f}x hit {100 * (lev >= LEV_CAP).mean():.0f}% of days)",
          flush=True)
    return out


def brick1_daily():
    """Frozen funding-carry brick, 8h events compounded to BUSINESS-day returns.
    Crypto trades 7d/week; weekend PnL is folded into the next business day so
    both bricks live on the same 252-day grid (dropping weekends would silently
    discard ~2/7 of the funding stream)."""
    fund, close = load_panels()
    df = run_carry(fund, close, k=3, exit_band=2, smooth=9,
                   rebalance_every=1, cost_per_side=0.0003)
    r8 = df["net"]
    daily = (1 + r8).groupby(r8.index.date).prod() - 1
    daily.index = pd.to_datetime(daily.index)
    bidx = pd.bdate_range(daily.index.min(), daily.index.max())
    pos = np.clip(np.searchsorted(bidx, daily.index, side="left"), 0, len(bidx) - 1)
    folded = (1 + daily).groupby(bidx[pos]).prod() - 1
    return folded


def brick2_daily_volweighted():
    """Frozen TS-carry signal with inverse-vol (risk-parity) weights.
    Direction = sign(5d-smoothed slope) — unchanged. Weight magnitude
    ~ 1/vol_i so natgas stops dominating. Costs 3bp/side as in the earned
    backtest (turnover in vol-weighted units), monthly roll costs included."""
    sig, rets, rolls = {}, {}, {}
    for key in ASSETS:
        df = load_asset(key)
        if df is None:
            continue
        r, rl = detect_rolls_and_returns(df)
        rets[key], rolls[key] = r, rl
        sig[key] = (df["c1"] / df["c2"] - 1.0).rolling(SMOOTH_D, min_periods=1).mean()
    S = pd.DataFrame(sig).sort_index()
    R = pd.DataFrame(rets).reindex(S.index)
    Rs = np.exp(R) - 1.0                                  # simple returns
    L = pd.DataFrame(rolls).reindex(S.index).astype("boolean").fillna(False)
    V = Rs.rolling(VOL_LOOKBACK).std().shift(1) * np.sqrt(ANN)  # per-asset vol

    idx = S.index
    assets = list(S.columns)
    w = pd.Series(0.0, index=assets)
    out = []
    cost_rate = 0.0003
    for i in range(len(idx)):
        t = idx[i]
        gross = float((w * Rs.loc[t].fillna(0.0)).sum())
        roll_cost = sum(2.0 * abs(w[a]) * cost_rate
                        for a in assets if w[a] != 0.0 and bool(L.at[t, a]))
        turn = 0.0
        if i % REBAL == 0:
            live = [a for a in assets if np.isfinite(S.at[t, a])
                    and np.isfinite(Rs.at[t, a]) and np.isfinite(V.at[t, a])
                    and V.at[t, a] > 0.01]
            new_w = pd.Series(0.0, index=assets)
            for a in live:
                # each asset sized to contribute TARGET_VOL/N under independence
                new_w[a] = np.sign(S.at[t, a]) * (TARGET_VOL / len(live)) / V.at[t, a]
            turn = float((new_w - w).abs().sum())
            w = new_w
        out.append({"t": t, "net": gross - roll_cost - turn * cost_rate})
    r = pd.DataFrame(out).set_index("t")["net"]
    return r[r.index >= r.first_valid_index()]


def brick3_daily():
    """Frozen FX-carry brick (G10 top3/bot3 + EM top2/bot2, 50/50 across the
    live books), marked DAILY: weights set at month-end from the prior-month
    rate print and held through the month; daily return = spot log-change +
    carry accrual/252; costs (5bp G10 / 20bp EM per side) on the monthly
    weight changes. Before 1997 (EM panel starts) the brick is the G10 book
    alone."""
    from fx_carry_skill import fred, UNIVERSE as G10U, US_RATE
    from fx_carry_retest import EM_UNIVERSE

    us_m = fred(US_RATE).resample("ME").last()
    books = []
    for universe, min_live, kmax, cost in [(G10U, 6, 3, 0.0005),
                                           (EM_UNIVERSE, 4, 2, 0.0020)]:
        spots, carr = {}, {}
        for ccy, (spot_s, usd_per_ccy, rate_s) in universe.items():
            try:
                sp, rt = fred(spot_s), fred(rate_s)
            except Exception:
                continue
            sp = sp if usd_per_ccy else 1.0 / sp
            spots[ccy] = np.log(sp)
            carr[ccy] = rt.resample("ME").last() - us_m
        S = pd.DataFrame(spots).sort_index()
        Cl = pd.DataFrame(carr).shift(1)              # publication-lag safe

        w_by_month = {}
        for t in Cl.index:
            c = Cl.loc[t].dropna()
            c = c[[a for a in c.index if a in S.columns]]
            if len(c) < min_live:
                continue
            k = kmax if len(c) >= 3 * kmax else 1
            order = c.sort_values()
            w = pd.Series(0.0, index=S.columns)
            w[order.index[-k:]] = 0.5 / k
            w[order.index[:k]] = -0.5 / k
            w_by_month[t] = w
        W = pd.DataFrame(w_by_month).T
        Wd = W.reindex(S.index, method="ffill").shift(1).fillna(0.0)
        dS = S.diff()
        carry_d = Cl.reindex(S.index, method="ffill") / 100 / 252
        ret = (Wd * (dS + carry_d).fillna(0.0)).sum(axis=1)
        net = ret - Wd.diff().abs().sum(axis=1) * cost
        books.append(net[Wd.abs().sum(axis=1) > 0])
    b3 = pd.concat(books, axis=1).mean(axis=1, skipna=True)
    return b3.dropna()


def main():
    print(f"{'='*74}\n  PHASE 2D — PORTFOLIO LAYER  (declared: {VOL_LOOKBACK}d vol, "
          f"{TARGET_VOL:.0%} target, {LEV_CAP:.0f}x cap, {REBAL}bd rebal)\n{'='*74}",
          flush=True)

    print("\nBrick #1 (funding carry, frozen) — regenerating from data…", flush=True)
    b1 = brick1_daily()
    print("Brick #2 (TS carry, frozen signal, inverse-vol weights)…", flush=True)
    b2_raw = brick2_daily_volweighted()
    print("Brick #3 (FX carry, frozen books, daily-marked)…", flush=True)
    b3 = brick3_daily()

    print(f"\n  RAW bricks:", flush=True)
    raw = {"b1": b1, "b2": b2_raw, "b3": b3}
    s_raw = {k: stats(v, f"{k} raw") for k, v in raw.items()}

    print(f"\n  VOL-TARGETED to {TARGET_VOL:.0%}:", flush=True)
    vt = {k: vol_target_stream(v, k) for k, v in raw.items()}
    s_vt = {k: stats(v, f"{k} vol-targeted") for k, v in vt.items()}

    # ---- pairwise correlations (each on its own maximal overlap)
    V = pd.DataFrame(vt)
    print(f"\n  Correlations (daily, pairwise maximal overlap):", flush=True)
    print(V.corr(min_periods=120).round(2).to_string(), flush=True)

    # ---- the firm's curve: equal risk across LIVE bricks each day
    combo = V.mean(axis=1, skipna=True).dropna()
    n_live = V.notna().sum(axis=1)
    print(f"\n  COMBINED (equal risk across live bricks; composition timeline):",
          flush=True)
    for n in (1, 2, 3):
        m = n_live == n
        if m.any():
            print(f"    {n} brick(s): {V.index[m][0].date()} .. {V.index[m][-1].date()}"
                  f"  ({int(m.sum())}d)", flush=True)
    sc = stats(combo, "combined, un-retargeted")
    combo_vt = vol_target_stream(combo, "combined")
    scv = stats(combo_vt, f"combined, retargeted {TARGET_VOL:.0%}")

    # strict 3-way window for the record
    all3 = V.dropna()
    if len(all3) > 120:
        c3 = all3.mean(axis=1)
        print(f"\n  Strict 3-brick window:", flush=True)
        s3 = stats(c3, f"all-3 window")
    else:
        s3 = None

    print(f"\n  Per-year (combined, retargeted, last 12y):", flush=True)
    for y, v in combo_vt.groupby(combo_vt.index.year):
        if len(v) < 60 or y < 2014:
            continue
        print(f"    {y}: net={(1 + v).prod() - 1:+7.1%}  "
              f"Sharpe={v.mean() / v.std(ddof=1) * np.sqrt(ANN):+5.2f}  "
              f"bricks={int(n_live.reindex(v.index).max())}", flush=True)

    out_dir = RESULTS_DIR / "portfolio"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, s in [("b1_vt", vt["b1"]), ("b2_vt", vt["b2"]), ("b3_vt", vt["b3"]),
                    ("combined_vt", combo_vt)]:
        (1 + s).cumprod().to_csv(out_dir / f"phase2d_equity_{name}.csv",
                                 header=["equity"])
    with open(out_dir / "phase2d_summary.json", "w") as f:
        json.dump({"config": {"vol_lookback": VOL_LOOKBACK, "target_vol": TARGET_VOL,
                              "lev_cap": LEV_CAP, "rebal": REBAL, "n_bricks": 3},
                   "raw": s_raw, "vol_targeted": s_vt,
                   "corr": V.corr(min_periods=120).to_dict(),
                   "combined": sc, "combined_vt": scv, "all3_window": s3},
                  f, indent=2, default=str)
    print(f"\nSaved to {out_dir}/phase2d_*.json/csv", flush=True)


if __name__ == "__main__":
    main()
