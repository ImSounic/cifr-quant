"""Multi-asset dual-market portfolio backtest (walk-forward, CQR-gated).

Runs a joint risk-parity portfolio per market on the held-out test window
(last 90 days), then combines the two markets into a single daily top-line.

Uses the SAME ensemble that CQR was calibrated on (zero-shot + 2 finetunes per
market) and applies the per-asset CQR corrections from
`results/cqr/cqr_calibrations.json` to widen the q05/q95 bands used for SL/TP
and risk-parity sizing.

Usage:
    python scripts/backtest_portfolio.py --market all --n-paths 30
    python scripts/backtest_portfolio.py --market crypto --n-paths 30 --capital 100000
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Kronos"))

from configs.base_config import DATA_RAW_DIR, RESULTS_DIR
from src.backtest.costs import COST_MODELS
from src.backtest.metrics import evaluate_trades
from src.backtest.portfolio_engine import run_market_backtest

COMMODITY_KEY_TO_FILE = {
    "XAU/USD": "xau_usd", "XAG/USD": "xag_usd", "XPT/USD": "xpt_usd",
    "CRUDE_OIL": "wti_usd", "BRENT_OIL": "brent_usd",
    "NATURAL_GAS": "ng_usd", "COPPER": "copper_usd",
}


def load_cqr(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"CQR calibrations not found at {path}. Run calibrate_cqr_multi.py first.")
    with open(path) as f:
        data = json.load(f)
    return {k: float(v["correction"]) for k, v in data.items()}


def load_crypto_assets(symbols):
    dfs, pred_len = {}, {}
    from configs.crypto_universe import get_crypto_configs
    cfgs = get_crypto_configs(tiers=(1, 2))
    for sym in symbols:
        safe = sym.replace("/", "_").lower()
        path = DATA_RAW_DIR / "crypto" / f"{safe}_15m.csv"
        if not path.exists():
            print(f"  SKIP {sym}: no data at {path}", flush=True)
            continue
        df = pd.read_csv(path)
        df["timestamps"] = pd.to_datetime(df["timestamps"])
        dfs[sym] = df
        pred_len[sym] = cfgs[sym].pred_len
    return dfs, pred_len


def load_commodity_assets(keys):
    dfs, pred_len = {}, {}
    from configs.commodity_universe import get_commodity_configs
    cfgs = get_commodity_configs(categories=("precious", "energy"))
    for key in keys:
        file_key = COMMODITY_KEY_TO_FILE.get(key, key.replace("/", "_").lower())
        path = DATA_RAW_DIR / "commodity" / f"{file_key}_4h.csv"
        if not path.exists():
            print(f"  SKIP {key}: no data at {path}", flush=True)
            continue
        df = pd.read_csv(path)
        df["timestamps"] = pd.to_datetime(df["timestamps"])
        dfs[key] = df
        pred_len[key] = cfgs[key].pred_len
    return dfs, pred_len


def market_metrics(result, n_trials=1):
    """PerformanceReport for one market from its rebalance equity curve."""
    eq = result.equity.values.astype(float)
    eq = np.concatenate([[result.initial_capital], eq])  # prepend start
    trade_pnls = np.array([t.pnl_pct for t in result.trades]) if result.trades else np.array([])
    test_years = max((result.test_end - result.test_start).days / 365.25, 1e-9)
    n_ret = max(len(eq) - 1, 1)
    ppy = n_ret / test_years
    return evaluate_trades(trade_pnls, eq, periods_per_year=ppy, n_trials=n_trials)


def daily_equity(result) -> pd.Series:
    """Resample a market's per-rebalance equity to a daily curve."""
    s = result.equity.sort_index()
    if s.empty:
        return s
    daily = s.resample("1D").last().ffill()
    return daily


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["crypto", "commodity", "all"], default="all")
    ap.add_argument("--n-paths", type=int, default=30)
    ap.add_argument("--capital", type=float, default=100_000.0,
                    help="Total capital, split evenly across the active markets")
    ap.add_argument("--min-confidence", type=float, default=0.55)
    ap.add_argument("--max-position-pct", type=float, default=0.10)
    ap.add_argument("--max-drawdown-halt", type=float, default=0.25)
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

    from src.model.build import build_market_ensemble

    corrections = load_cqr(RESULTS_DIR / "cqr" / "cqr_calibrations.json")
    print(f"Loaded {len(corrections)} CQR corrections", flush=True)

    markets = ["crypto", "commodity"] if args.market == "all" else [args.market]
    per_market_capital = args.capital / len(markets)

    results = {}
    reports = {}

    for market in markets:
        print(f"\n{'='*64}\n  {market.upper()} PORTFOLIO BACKTEST\n{'='*64}", flush=True)
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
        res = run_market_backtest(
            market=market, ensemble=ensemble,
            asset_dfs=dfs, asset_pred_len=pred_len, corrections=corrections,
            cost_model=COST_MODELS[market],
            n_paths=args.n_paths, step_size=args.step_size,
            test_days=args.test_days, min_confidence=args.min_confidence,
            max_position_pct=args.max_position_pct,
            initial_capital=per_market_capital,
            max_drawdown_halt=args.max_drawdown_halt,
        )
        results[market] = res
        reports[market] = market_metrics(res)
        print(f"\n--- {market} result ---\n{reports[market]}", flush=True)

    # Combined daily top-line.
    combined_report = None
    if results:
        dailies = {m: daily_equity(r) for m, r in results.items()}
        full_idx = None
        for s in dailies.values():
            full_idx = s.index if full_idx is None else full_idx.union(s.index)
        combined = pd.Series(0.0, index=full_idx)
        for m, s in dailies.items():
            aligned = s.reindex(full_idx).ffill().fillna(results[m].initial_capital)
            combined = combined.add(aligned, fill_value=results[m].initial_capital)
        all_trades = [t.pnl_pct for r in results.values() for t in r.trades]
        eq = combined.values.astype(float)
        combined_report = evaluate_trades(
            np.array(all_trades) if all_trades else np.array([]),
            eq, periods_per_year=365.0, n_trials=1)
        print(f"\n{'='*64}\n  COMBINED PORTFOLIO (daily top-line)\n{'='*64}", flush=True)
        print(combined_report, flush=True)

    # Persist.
    out_dir = RESULTS_DIR / "backtest"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"config": vars(args), "markets": {}}
    for m, r in results.items():
        rep = reports[m]
        summary["markets"][m] = {
            "initial_capital": r.initial_capital,
            "final_capital": r.final_capital,
            "n_rebalances": r.n_rebalances,
            "n_trades": rep.n_trades,
            "total_return": rep.total_return,
            "annualized_return": rep.annualized_return,
            "sharpe": rep.sharpe_ratio,
            "max_drawdown": rep.max_drawdown,
            "calmar": rep.calmar_ratio,
            "win_rate": rep.win_rate,
            "profit_factor": rep.profit_factor,
            "test_start": str(r.test_start.date()),
            "test_end": str(r.test_end.date()),
            "per_asset": r.per_asset,
        }
        r.equity.to_csv(out_dir / f"equity_{m}.csv", header=["equity"])
    if combined_report is not None:
        summary["combined"] = {
            "total_return": combined_report.total_return,
            "annualized_return": combined_report.annualized_return,
            "sharpe": combined_report.sharpe_ratio,
            "max_drawdown": combined_report.max_drawdown,
            "calmar": combined_report.calmar_ratio,
            "win_rate": combined_report.win_rate,
            "profit_factor": combined_report.profit_factor,
            "n_trades": combined_report.n_trades,
        }
    with open(out_dir / "portfolio_backtest.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved backtest summary to {out_dir / 'portfolio_backtest.json'}", flush=True)


if __name__ == "__main__":
    main()
