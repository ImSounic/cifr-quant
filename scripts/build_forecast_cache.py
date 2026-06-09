"""Build the Kronos forecast cache for the portfolio backtest (GPU, run on HPC).

Runs the SAME ensemble CQR was calibrated on over the walk-forward grid and
dumps one forecast row per (symbol, rebalance_t) to
results/forecasts/{market}_forecasts.csv. After this, strategy A/B
(scripts/backtest_portfolio.py) is CPU-only and needs no GPU.

Usage:
    python scripts/build_forecast_cache.py --market all --n-paths 30
"""

import sys
import argparse
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Kronos"))

from configs.base_config import RESULTS_DIR
from src.model.build import build_market_ensemble
from src.model.forecast_cache import build_forecast_cache

# Reuse the exact asset loaders from the backtest driver.
from scripts.backtest_portfolio import load_crypto_assets, load_commodity_assets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["crypto", "commodity", "all"], default="all")
    ap.add_argument("--n-paths", type=int, default=30)
    ap.add_argument("--test-days", type=int, default=90)
    ap.add_argument("--step-size", type=int, default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"Device: {device}", flush=True)

    forecasts_dir = RESULTS_DIR / "forecasts"
    markets = ["crypto", "commodity"] if args.market == "all" else [args.market]

    for market in markets:
        print(f"\n{'='*64}\n  BUILD FORECAST CACHE: {market.upper()}\n{'='*64}", flush=True)
        if market == "crypto":
            from configs.crypto_universe import get_crypto_configs
            symbols = list(get_crypto_configs(tiers=(1, 2)).keys())
            dfs, pred_len = load_crypto_assets(symbols)
        else:
            from configs.commodity_universe import get_commodity_configs
            keys = list(get_commodity_configs(categories=("precious", "energy")).keys())
            dfs, pred_len = load_commodity_assets(keys)

        if not dfs:
            print(f"  No data for {market}, skipping.", flush=True)
            continue

        ensemble = build_market_ensemble(market, device=device)
        build_forecast_cache(
            market=market, ensemble=ensemble,
            asset_dfs=dfs, asset_pred_len=pred_len,
            forecasts_dir=forecasts_dir,
            n_paths=args.n_paths, step_size=args.step_size,
            test_days=args.test_days,
        )

    print("\nForecast cache build complete.", flush=True)


if __name__ == "__main__":
    main()
