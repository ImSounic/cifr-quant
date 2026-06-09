"""Measure forecasting skill AS A FUNCTION OF HORIZON (GPU, run on HPC).

The final-step diagnostic (scripts/forecast_skill.py) showed ~zero skill at the
48-bar crypto / 6-bar commodity horizon. But skill almost always concentrates at
SHORT horizons and decays. This re-runs the ensemble and, for every forecast,
records predicted vs realised return at EACH step h=1..pred_len, then reports IC
and directional hit rate per horizon.

If short-horizon IC is meaningfully positive (say >0.03 and rising as h->1), the
project pivots to short-horizon trading. If it's flat ~0 at every horizon, the
model has no extractable directional edge on this data.

Lean by design: fewer paths + a thinned grid (--stride) so it finishes fast.

Usage:
    python scripts/horizon_skill.py --market all --n-paths 20 --stride 96
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Kronos"))

from src.backtest.grid import compute_rebalance_grid, compute_test_window, locate_context
from src.model.build import build_market_ensemble
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


def run_market(market, dfs, pred_len, ensemble, *, n_paths, stride,
               lookback=512, test_days=90):
    pl_max = max(pred_len.values())
    test_start, test_end = compute_test_window(dfs, test_days)
    rebalance_times, ref_sym = compute_rebalance_grid(dfs, pl_max, stride, test_start, test_end)
    print(f"\n[{market}] horizons up to {pl_max}  windows~{len(rebalance_times)}/asset  "
          f"ref={ref_sym}  n_paths={n_paths} stride={stride}", flush=True)

    # Accumulate predicted/realised return per horizon step.
    pred_by_h = {h: [] for h in range(1, pl_max + 1)}
    real_by_h = {h: [] for h in range(1, pl_max + 1)}

    for r, t in enumerate(rebalance_times):
        for s, df in dfs.items():
            pl = pred_len[s]
            ctx, fut = locate_context(df, t, lookback, pl)
            if ctx is None:
                continue
            entry = float(ctx["close"].iloc[-1])
            if entry <= 0:
                continue
            try:
                res = ensemble.predict_with_quantiles(
                    df=ctx[["open", "high", "low", "close", "volume", "amount"]],
                    x_timestamp=ctx["timestamps"], y_timestamp=fut["timestamps"],
                    pred_len=pl, n_paths=n_paths)
            except Exception as e:
                print(f"    {t} {s} failed: {e}", flush=True)
                continue
            pred_path = np.asarray(res["close_q50"], dtype=float)   # median over horizon
            real_path = fut["close"].to_numpy(dtype=float)
            for h in range(1, pl + 1):
                pred_by_h[h].append((pred_path[h - 1] - entry) / entry)
                real_by_h[h].append((real_path[h - 1] - entry) / entry)
        if r % 10 == 0:
            print(f"    [{r+1}/{len(rebalance_times)}] {t}", flush=True)

    print(f"\n  Skill by horizon ({market}):", flush=True)
    print(f"  {'h':>4}  {'n':>6}  {'IC':>8}  {'hit':>7}  {'mean|pred|':>10}", flush=True)
    for h in range(1, pl_max + 1):
        p = np.asarray(pred_by_h[h]); rl = np.asarray(real_by_h[h])
        if len(p) < 10:
            continue
        ic = _spearman(p, rl)
        hit = float(np.mean(np.sign(p) == np.sign(rl)))
        print(f"  {h:>4}  {len(p):>6}  {ic:>+8.4f}  {hit:>6.1%}  {np.mean(np.abs(p)):>10.4%}",
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["crypto", "commodity", "all"], default="all")
    ap.add_argument("--n-paths", type=int, default=20)
    ap.add_argument("--stride", type=int, default=96,
                    help="Candles between sampled rebalance points (larger = faster)")
    ap.add_argument("--test-days", type=int, default=90)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        else:
            device = "cpu"
    print(f"Device: {device}", flush=True)

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
        ensemble = build_market_ensemble(market, device=device)
        run_market(market, dfs, pred_len, ensemble,
                   n_paths=args.n_paths, stride=args.stride, test_days=args.test_days)


if __name__ == "__main__":
    main()
