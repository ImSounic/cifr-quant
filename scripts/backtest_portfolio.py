"""Multi-asset dual-market portfolio backtest (walk-forward, CQR-gated).

CPU-only: reads cached Kronos forecasts (results/forecasts/, built on HPC by
scripts/build_forecast_cache.py) and runs an injected STRATEGY + SIZER over the
walk-forward grid, then combines the two markets into a single daily top-line.

Because forecasts are cached, swapping --strategy / --sizer is instant and needs
no GPU — this is how we A/B strategies.

Usage:
    python scripts/build_forecast_cache.py --market all      # once, on HPC (GPU)
    python scripts/backtest_portfolio.py  --market all --strategy directional_momentum
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Kronos"))

from configs.base_config import DATA_RAW_DIR, RESULTS_DIR
from src.backtest.costs import COST_MODELS
from src.backtest.metrics import evaluate_trades
from src.backtest.portfolio_engine import run_market_backtest
from src.backtest.strategies import DirectionalMomentum, RegimeGatedTrend, MeanReversion
from src.backtest.sizing import InverseWidthRiskParity, EqualWeight
from src.model.forecast_cache import load_forecast_cache
from src.regime.classifier import RegimeClassifier

COMMODITY_KEY_TO_FILE = {
    "XAU/USD": "xau_usd", "XAG/USD": "xag_usd", "XPT/USD": "xpt_usd",
    "CRUDE_OIL": "wti_usd", "BRENT_OIL": "brent_usd",
    "NATURAL_GAS": "ng_usd", "COPPER": "copper_usd",
}


REGIME_AWARE = {"regime_gated_trend", "mean_reversion"}


def build_strategy(name, min_confidence):
    """Return (strategy, needs_regime_classifier)."""
    if name == "directional_momentum":
        return DirectionalMomentum(min_confidence=min_confidence), False
    if name == "regime_gated_trend":
        return RegimeGatedTrend(min_confidence=min_confidence), True
    if name == "mean_reversion":
        return MeanReversion(), True
    raise ValueError(f"Unknown strategy '{name}'. Available: "
                     "directional_momentum, regime_gated_trend, mean_reversion")


def build_sizer(name, max_position_pct):
    if name == "inverse_width_risk_parity":
        return InverseWidthRiskParity(max_position_pct=max_position_pct)
    if name == "equal_weight":
        return EqualWeight(max_position_pct=max_position_pct)
    raise ValueError(f"Unknown sizer '{name}'.")


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
    s = result.equity.sort_index()
    if s.empty:
        return s
    return s.resample("1D").last().ffill()


def exit_mix(result) -> dict:
    mix = {"tp": 0, "sl": 0, "timeout": 0}
    for t in result.trades:
        mix[t.exit_reason] = mix.get(t.exit_reason, 0) + 1
    return mix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["crypto", "commodity", "all"], default="all")
    ap.add_argument("--strategy", default="directional_momentum")
    ap.add_argument("--sizer", default="inverse_width_risk_parity")
    ap.add_argument("--capital", type=float, default=100_000.0,
                    help="Total capital, split evenly across the active markets")
    ap.add_argument("--min-confidence", type=float, default=0.55)
    ap.add_argument("--max-position-pct", type=float, default=0.10)
    ap.add_argument("--max-drawdown-halt", type=float, default=0.25)
    ap.add_argument("--test-days", type=int, default=90)
    ap.add_argument("--step-size", type=int, default=None)
    ap.add_argument("--tag", default=None, help="Suffix for output files (for A/B runs)")
    # Regime thresholds (tunable for the A/B sweep; Hurst is conservative on weak
    # drift, so the strict 0.55 default may need lowering to let trends through).
    ap.add_argument("--hurst-trend", type=float, default=0.55)
    ap.add_argument("--adx-trend", type=float, default=25.0)
    args = ap.parse_args()

    forecasts_dir = RESULTS_DIR / "forecasts"
    corrections = load_cqr(RESULTS_DIR / "cqr" / "cqr_calibrations.json")
    print(f"Loaded {len(corrections)} CQR corrections", flush=True)
    print(f"Strategy={args.strategy}  Sizer={args.sizer}", flush=True)

    markets = ["crypto", "commodity"] if args.market == "all" else [args.market]
    per_market_capital = args.capital / len(markets)

    results, reports = {}, {}

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

        forecasts = load_forecast_cache(market, forecasts_dir)
        print(f"  Loaded {len(forecasts)} cached forecasts for {market}", flush=True)

        strategy, needs_regime = build_strategy(args.strategy, args.min_confidence)
        sizer = build_sizer(args.sizer, args.max_position_pct)
        regime_clf = (RegimeClassifier(hurst_trend=args.hurst_trend, adx_trend=args.adx_trend)
                      if needs_regime else None)
        if needs_regime:
            print(f"  Regime classifier: Hurst+ADX composite "
                  f"(hurst_trend={args.hurst_trend}, adx_trend={args.adx_trend})", flush=True)

        res = run_market_backtest(
            market=market,
            asset_dfs=dfs, asset_pred_len=pred_len,
            cost_model=COST_MODELS[market],
            forecasts=forecasts, corrections=corrections,
            strategy=strategy, sizer=sizer,
            regime_classifier=regime_clf,
            step_size=args.step_size, test_days=args.test_days,
            initial_capital=per_market_capital,
            max_drawdown_halt=args.max_drawdown_halt,
        )
        results[market] = res
        reports[market] = market_metrics(res)
        print(f"\n--- {market} result ({res.strategy_name}/{res.sizer_name}) ---", flush=True)
        print(reports[market], flush=True)
        print(f"  Exit mix: {exit_mix(res)}", flush=True)

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
    tag = f"_{args.tag}" if args.tag else ""
    summary = {"config": vars(args), "markets": {}}
    for m, r in results.items():
        rep = reports[m]
        summary["markets"][m] = {
            "strategy": r.strategy_name, "sizer": r.sizer_name,
            "initial_capital": r.initial_capital, "final_capital": r.final_capital,
            "n_rebalances": r.n_rebalances, "n_trades": rep.n_trades,
            "total_return": rep.total_return, "annualized_return": rep.annualized_return,
            "sharpe": rep.sharpe_ratio, "max_drawdown": rep.max_drawdown,
            "calmar": rep.calmar_ratio, "win_rate": rep.win_rate,
            "profit_factor": rep.profit_factor, "exit_mix": exit_mix(r),
            "test_start": str(r.test_start.date()), "test_end": str(r.test_end.date()),
            "per_asset": r.per_asset,
        }
        r.equity.to_csv(out_dir / f"equity_{m}{tag}.csv", header=["equity"])
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
    out_json = out_dir / f"portfolio_backtest{tag}.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved backtest summary to {out_json}", flush=True)


if __name__ == "__main__":
    main()
