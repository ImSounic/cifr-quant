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


def main():
    print(f"{'='*74}\n  PHASE 2D — PORTFOLIO LAYER  (declared: {VOL_LOOKBACK}d vol, "
          f"{TARGET_VOL:.0%} target, {LEV_CAP:.0f}x cap, {REBAL}bd rebal)\n{'='*74}",
          flush=True)

    print("\nBrick #1 (funding carry, frozen) — regenerating from data…", flush=True)
    b1 = brick1_daily()
    print("Brick #2 (TS carry, frozen signal, inverse-vol weights)…", flush=True)
    b2_raw = brick2_daily_volweighted()

    print(f"\n  RAW bricks:", flush=True)
    s1_raw = stats(b1, "brick1 funding carry (as frozen)")
    s2_raw = stats(b2_raw, "brick2 ts-carry (inverse-vol book)")

    print(f"\n  VOL-TARGETED to {TARGET_VOL:.0%}:", flush=True)
    b1_vt = vol_target_stream(b1, "brick1")
    b2_vt = vol_target_stream(b2_raw, "brick2")
    s1 = stats(b1_vt, "brick1 vol-targeted")
    s2 = stats(b2_vt, "brick2 vol-targeted")

    # ---- correlation & combination on the overlap
    both = pd.DataFrame({"b1": b1_vt, "b2": b2_vt}).dropna()
    if len(both) < 60:
        print(f"\n  Overlap too short ({len(both)}d) — no combined book.", flush=True)
        return
    rho_d = both["b1"].corr(both["b2"])
    wk = both.resample("W").sum()
    rho_w = wk["b1"].corr(wk["b2"])
    print(f"\n  Overlap: {both.index[0].date()} .. {both.index[-1].date()} "
          f"({len(both)}d)  corr daily {rho_d:+.2f} / weekly {rho_w:+.2f}", flush=True)

    combo = 0.5 * both["b1"] + 0.5 * both["b2"]
    print(f"\n  COMBINED (50/50 risk):", flush=True)
    sc = stats(combo, "combined, un-retargeted")
    combo_vt = vol_target_stream(combo, "combined")
    scv = stats(combo_vt, f"combined, retargeted {TARGET_VOL:.0%}")

    print(f"\n  Per-year (combined, retargeted):", flush=True)
    for y, v in combo_vt.groupby(combo_vt.index.year):
        if len(v) < 60:
            continue
        print(f"    {y}: net={(1 + v).prod() - 1:+7.1%}  "
              f"Sharpe={v.mean() / v.std(ddof=1) * np.sqrt(ANN):+5.2f}", flush=True)

    out_dir = RESULTS_DIR / "portfolio"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, s in [("b1_vt", b1_vt), ("b2_vt", b2_vt), ("combined_vt", combo_vt)]:
        (1 + s).cumprod().to_csv(out_dir / f"phase2d_equity_{name}.csv",
                                 header=["equity"])
    with open(out_dir / "phase2d_summary.json", "w") as f:
        json.dump({"config": {"vol_lookback": VOL_LOOKBACK, "target_vol": TARGET_VOL,
                              "lev_cap": LEV_CAP, "rebal": REBAL},
                   "brick1_raw": s1_raw, "brick2_raw": s2_raw,
                   "brick1_vt": s1, "brick2_vt": s2,
                   "corr_daily": rho_d, "corr_weekly": rho_w,
                   "combined": sc, "combined_vt": scv},
                  f, indent=2, default=str)
    print(f"\nSaved to {out_dir}/phase2d_*.json/csv", flush=True)


if __name__ == "__main__":
    main()
