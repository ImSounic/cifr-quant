"""Measure the RAW forecasting skill of the Kronos ensemble, decoupled from any
trading strategy.

Trading PnL conflates "is the forecaster good?" with "is the strategy good?".
This script answers ONLY the first question, off the cached forecasts + raw data
(CPU, no GPU). For every cached (symbol, rebalance_t) it compares the model's
predicted move to the realised move at the forecast horizon and reports:

  - Information Coefficient (Spearman rank corr of predicted vs realised return),
    pooled and per asset.
  - Directional hit rate (does sign(prediction) match sign(realised move)?),
    overall, per asset, and BUCKETED BY CONFIDENCE — the key test of whether the
    model's own confidence stratifies outcomes (if it does, the signal is real
    and we're diluting it; if it's flat ~50%, confidence is noise).
  - Mean signed edge per forecast (realised_return in the predicted direction),
    before costs — the theoretical ceiling for a long/short bet at this horizon.

Usage:
    python scripts/forecast_skill.py --market all
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Kronos"))

from configs.base_config import RESULTS_DIR
from src.backtest.grid import locate_context
from src.model.forecast_cache import load_forecast_cache
from scripts.backtest_portfolio import load_crypto_assets, load_commodity_assets


def _spearman(a, b):
    if len(a) < 3:
        return float("nan")
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def collect(market, dfs, pred_len, forecasts, lookback=512):
    rows = []
    for (sym, t), fb in forecasts.items():
        if sym not in dfs:
            continue
        pl = pred_len[sym]
        ctx, fut = locate_context(dfs[sym], t, lookback, pl)
        if fut is None:
            continue
        entry = float(fb.entry_price)
        if entry <= 0:
            continue
        realised = float(fut["close"].iloc[-1])
        realised_ret = (realised - entry) / entry
        pred_ret = (float(fb.q50) - entry) / entry          # median-based
        pred_ret_mean = (float(fb.mean) - entry) / entry    # mean-based
        # Directional hit uses the cache's own majority-vote direction.
        if fb.direction == "long":
            hit = realised_ret > 0
            signed_edge = realised_ret
        else:
            hit = realised_ret < 0
            signed_edge = -realised_ret
        rows.append(dict(symbol=sym, confidence=float(fb.confidence),
                         pred_ret=pred_ret, pred_ret_mean=pred_ret_mean,
                         realised_ret=realised_ret, hit=bool(hit),
                         signed_edge=signed_edge))
    return pd.DataFrame(rows)


def report(market, df):
    print(f"\n{'='*68}\n  FORECAST SKILL: {market.upper()}  (n={len(df)})\n{'='*68}", flush=True)
    if df.empty:
        print("  no matched forecasts", flush=True)
        return

    ic = _spearman(df["pred_ret"].to_numpy(), df["realised_ret"].to_numpy())
    ic_mean = _spearman(df["pred_ret_mean"].to_numpy(), df["realised_ret"].to_numpy())
    hit = df["hit"].mean()
    edge = df["signed_edge"].mean()
    print(f"  Pooled IC (median pred):  {ic:+.4f}", flush=True)
    print(f"  Pooled IC (mean pred):    {ic_mean:+.4f}", flush=True)
    print(f"  Directional hit rate:     {hit:.1%}   (50% = no skill)", flush=True)
    print(f"  Mean signed edge/fcast:   {edge:+.4%}   (pre-cost ceiling)", flush=True)

    print("\n  Hit rate by confidence bucket (does confidence stratify?):", flush=True)
    bins = [0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 1.01]
    labels = ["0.50-0.55", "0.55-0.60", "0.60-0.65", "0.65-0.70", "0.70-0.80", "0.80-1.0"]
    df = df.copy()
    df["bucket"] = pd.cut(df["confidence"], bins=bins, labels=labels, right=False)
    for lab in labels:
        sub = df[df["bucket"] == lab]
        if len(sub):
            print(f"    {lab}:  n={len(sub):5d}  hit={sub['hit'].mean():.1%}  "
                  f"edge={sub['signed_edge'].mean():+.4%}", flush=True)

    print("\n  Per-asset (sorted by hit rate):", flush=True)
    g = df.groupby("symbol").agg(n=("hit", "size"), hit=("hit", "mean"),
                                 edge=("signed_edge", "mean"),
                                 ic=("pred_ret", lambda s: _spearman(
                                     s.to_numpy(), df.loc[s.index, "realised_ret"].to_numpy())))
    g = g.sort_values("hit", ascending=False)
    for sym, r in g.iterrows():
        print(f"    {sym:12s}  n={int(r['n']):4d}  hit={r['hit']:.1%}  "
              f"edge={r['edge']:+.4%}  IC={r['ic']:+.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["crypto", "commodity", "all"], default="all")
    args = ap.parse_args()

    forecasts_dir = RESULTS_DIR / "forecasts"
    markets = ["crypto", "commodity"] if args.market == "all" else [args.market]

    for market in markets:
        if market == "crypto":
            from configs.crypto_universe import get_crypto_configs
            symbols = list(get_crypto_configs(tiers=(1, 2)).keys())
            dfs, pred_len = load_crypto_assets(symbols)
        else:
            from configs.commodity_universe import get_commodity_configs
            keys = list(get_commodity_configs(categories=("precious", "energy")).keys())
            dfs, pred_len = load_commodity_assets(keys)
        if not dfs:
            print(f"  No data for {market}", flush=True)
            continue
        forecasts = load_forecast_cache(market, forecasts_dir)
        df = collect(market, dfs, pred_len, forecasts)
        report(market, df)


if __name__ == "__main__":
    main()
