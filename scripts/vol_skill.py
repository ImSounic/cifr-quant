"""Does Kronos predict VOLATILITY (even though it can't predict direction)?

Returns are near-unpredictable, but volatility clusters and is forecastable -
the one robust stylized fact in finance. Our cache already contains a volatility
forecast we never evaluated: the quantile spread (q95-q05). This script tests
whether that predicted spread rank-correlates with REALISED volatility over the
forecast horizon. CPU-only, off the existing cache + data.

If the IC here is meaningfully positive (and unlike the directional ICs, it
should be), the model has real skill - at predicting risk, not direction - which
reframes the project toward vol-targeting / risk products / vol trading.

Usage:
    python scripts/vol_skill.py --market all
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
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 3:
        return float("nan")
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def collect(dfs, pred_len, forecasts, lookback=512):
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
        # Predicted volatility proxy: the forecast interval width.
        pred_width = (float(fb.q95) - float(fb.q05)) / entry
        # Also a "context vol" baseline: realised vol of the lookback window
        # (this is what a naive GARCH-ish/persistence model would use).
        ctx_close = ctx["close"].to_numpy(dtype=float)
        ctx_rets = np.diff(np.log(np.clip(ctx_close[-pl - 1:], 1e-12, None)))
        ctx_vol = float(np.std(ctx_rets)) if len(ctx_rets) > 1 else 0.0
        # Realised volatility over the horizon.
        closes = np.concatenate([[entry], fut["close"].to_numpy(dtype=float)])
        rets = np.diff(np.log(np.clip(closes, 1e-12, None)))
        realised_vol = float(np.std(rets))
        realised_range = float((fut["high"].max() - fut["low"].min()) / entry)
        rows.append(dict(symbol=sym, timestamp=pd.Timestamp(t),
                         pred_width=pred_width, ctx_vol=ctx_vol,
                         realised_vol=realised_vol, realised_range=realised_range))
    return pd.DataFrame(rows)


def xs_ic(df, col_pred, col_real, min_assets=3):
    ics = []
    for _, g in df.groupby("timestamp"):
        if len(g) >= min_assets:
            ic = _spearman(g[col_pred].to_numpy(), g[col_real].to_numpy())
            if np.isfinite(ic):
                ics.append(ic)
    if len(ics) < 5:
        return None
    ics = np.asarray(ics)
    t = ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics))) if ics.std(ddof=1) > 0 else float("nan")
    return ics.mean(), t, len(ics)


def report(market, df):
    print(f"\n{'='*68}\n  VOLATILITY SKILL: {market.upper()}  (n={len(df)})\n{'='*68}", flush=True)
    if df.empty:
        print("  no matched forecasts", flush=True)
        return
    pw, rv, rr = df["pred_width"], df["realised_vol"], df["realised_range"]
    print(f"  Pooled IC  pred_width vs realised_vol:    "
          f"{_spearman(pw.to_numpy(), rv.to_numpy()):+.4f}", flush=True)
    print(f"  Pooled IC  pred_width vs realised_range:  "
          f"{_spearman(pw.to_numpy(), rr.to_numpy()):+.4f}", flush=True)
    print(f"  Baseline   ctx_vol   vs realised_vol:     "
          f"{_spearman(df['ctx_vol'].to_numpy(), rv.to_numpy()):+.4f}   "
          f"(naive persistence - the bar to beat)", flush=True)

    xs = xs_ic(df, "pred_width", "realised_vol")
    if xs:
        m, t, n = xs
        print(f"  Cross-sectional vol IC: mean={m:+.4f}  t-stat={t:+.2f}  periods={n}", flush=True)

    print("\n  Per-asset (pred_width vs realised_vol IC):", flush=True)
    for sym, g in df.groupby("symbol"):
        ic = _spearman(g["pred_width"].to_numpy(), g["realised_vol"].to_numpy())
        print(f"    {sym:12s}  n={len(g):4d}  IC={ic:+.3f}", flush=True)


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
        df = collect(dfs, pred_len, forecasts)
        report(market, df)


if __name__ == "__main__":
    main()
