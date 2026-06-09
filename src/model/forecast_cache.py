"""Build and load the Kronos forecast cache.

The cache is the key to fast strategy iteration: run the (expensive, GPU)
ensemble ONCE over the whole walk-forward grid and dump one row per
(symbol, rebalance_t). Strategy A/B then runs CPU-only off the cache with no
model in the loop.

Cache format: a plain CSV per market at results/forecasts/{market}_forecasts.csv
(rows are tiny — at most ~n_assets * n_rebalances) plus a sidecar meta JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.backtest.grid import compute_rebalance_grid, compute_test_window, locate_context
from src.backtest.strategy_api import ForecastBundle


CACHE_COLUMNS = [
    "symbol", "timestamp", "entry_price",
    "q05", "q25", "q50", "q75", "q95", "mean",
    "direction", "confidence", "pred_len",
]


def _cache_paths(forecasts_dir: Path, market: str) -> Tuple[Path, Path]:
    return (forecasts_dir / f"{market}_forecasts.csv",
            forecasts_dir / f"{market}_forecasts_meta.json")


def build_forecast_cache(
    market: str,
    ensemble,
    asset_dfs: Dict[str, pd.DataFrame],
    asset_pred_len: Dict[str, int],
    forecasts_dir: Path,
    *,
    lookback: int = 512,
    n_paths: int = 30,
    step_size: int = None,
    test_days: int = 90,
    verbose: bool = True,
) -> Path:
    """Run the ensemble over the walk-forward grid and write the cache CSV."""
    symbols = list(asset_dfs.keys())
    pred_len = max(asset_pred_len[s] for s in symbols)
    step = step_size or pred_len

    test_start, test_end = compute_test_window(asset_dfs, test_days)
    rebalance_times, ref_sym = compute_rebalance_grid(
        asset_dfs, pred_len, step, test_start, test_end)

    if verbose:
        print(f"[{market}] cache build: test {test_start.date()}..{test_end.date()}  "
              f"ref={ref_sym}  rebalances={len(rebalance_times)}  assets={len(symbols)}  "
              f"pred_len={pred_len} step={step} n_paths={n_paths}", flush=True)

    rows = []
    for r, t in enumerate(rebalance_times):
        for s in symbols:
            pl = asset_pred_len[s]
            context, future = locate_context(asset_dfs[s], t, lookback, pl)
            if context is None:
                continue
            try:
                res = ensemble.predict_with_quantiles(
                    df=context[["open", "high", "low", "close", "volume", "amount"]],
                    x_timestamp=context["timestamps"],
                    y_timestamp=future["timestamps"],
                    pred_len=pl, n_paths=n_paths,
                )
            except Exception as e:
                if verbose:
                    print(f"    {t} {s} predict failed: {e}", flush=True)
                continue
            rows.append({
                "symbol": s,
                "timestamp": pd.Timestamp(t).isoformat(),
                "entry_price": float(res["entry_price"]),
                "q05": float(res["close_q05"][-1]),
                "q25": float(res["close_q25"][-1]),
                "q50": float(res["close_q50"][-1]),
                "q75": float(res["close_q75"][-1]),
                "q95": float(res["close_q95"][-1]),
                "mean": float(res["close_mean"][-1]),
                "direction": str(res["direction"]),
                "confidence": float(res["directional_confidence"]),
                "pred_len": int(pl),
            })
        if verbose and (r % 10 == 0 or r == len(rebalance_times) - 1):
            print(f"    [{r+1}/{len(rebalance_times)}] {t}  rows so far={len(rows)}", flush=True)

    forecasts_dir.mkdir(parents=True, exist_ok=True)
    csv_path, meta_path = _cache_paths(forecasts_dir, market)
    pd.DataFrame(rows, columns=CACHE_COLUMNS).to_csv(csv_path, index=False)
    with open(meta_path, "w") as f:
        json.dump({
            "market": market, "lookback": lookback, "n_paths": n_paths,
            "step": step, "pred_len_grid": pred_len, "test_days": test_days,
            "test_start": test_start.isoformat(), "test_end": test_end.isoformat(),
            "ref_symbol": ref_sym, "n_rebalances": len(rebalance_times),
            "n_rows": len(rows), "symbols": symbols,
        }, f, indent=2)
    if verbose:
        print(f"[{market}] wrote {len(rows)} forecast rows -> {csv_path}", flush=True)
    return csv_path


def load_forecast_cache(market: str, forecasts_dir: Path
                        ) -> Dict[Tuple[str, pd.Timestamp], ForecastBundle]:
    """Load a market's cache into {(symbol, timestamp): ForecastBundle}."""
    csv_path, _ = _cache_paths(forecasts_dir, market)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Forecast cache not found at {csv_path}. Run build_forecast_cache.py first.")
    df = pd.read_csv(csv_path)
    out: Dict[Tuple[str, pd.Timestamp], ForecastBundle] = {}
    for _, row in df.iterrows():
        ts = pd.Timestamp(row["timestamp"])
        out[(row["symbol"], ts)] = ForecastBundle(
            symbol=row["symbol"], timestamp=ts,
            entry_price=float(row["entry_price"]),
            q05=float(row["q05"]), q25=float(row["q25"]), q50=float(row["q50"]),
            q75=float(row["q75"]), q95=float(row["q95"]), mean=float(row["mean"]),
            direction=str(row["direction"]), confidence=float(row["confidence"]),
            pred_len=int(row["pred_len"]),
        )
    return out
