"""Multi-asset CQR calibration for the dual-market portfolio.

For each asset, runs ensemble predictions on a calibration window,
computes conformity scores, and saves the CQR correction term.
These corrections are used at inference time to widen prediction
intervals for guaranteed coverage.

Usage:
    python scripts/calibrate_cqr_multi.py --market crypto --n-paths 30 --step-size 48
    python scripts/calibrate_cqr_multi.py --market commodity --n-paths 30 --step-size 6
    python scripts/calibrate_cqr_multi.py --market all
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import timedelta

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Kronos"))

from configs.base_config import DATA_RAW_DIR, CHECKPOINTS_DIR, RESULTS_DIR


def load_raw_data(market: str, symbol_key: str) -> pd.DataFrame:
    """Load raw OHLCV CSV for an asset."""
    if market == "crypto":
        safe = symbol_key.replace("/", "_").lower()
        path = DATA_RAW_DIR / "crypto" / f"{safe}_15m.csv"
    else:
        safe = symbol_key.replace("/", "_").lower()
        path = DATA_RAW_DIR / "commodity" / f"{safe}_4h.csv"

    if not path.exists():
        raise FileNotFoundError(f"No data at {path}")

    df = pd.read_csv(path)
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    return df


def split_calibration(df: pd.DataFrame, cal_months: int = 3, val_months: int = 3):
    """
    Extract calibration window from the data.

    Layout: [...train...][cal_months][val_months][test_months=3]
    We use the calibration window for CQR.
    """
    end = df["timestamps"].max()
    test_start = end - timedelta(days=90)       # Last 3 months = test
    val_start = test_start - timedelta(days=90)  # 3 months before = val
    cal_start = val_start - timedelta(days=90)   # 3 months before = cal

    cal_df = df[(df["timestamps"] >= cal_start) & (df["timestamps"] < val_start)].reset_index(drop=True)
    val_df = df[(df["timestamps"] >= val_start) & (df["timestamps"] < test_start)].reset_index(drop=True)

    # Need enough history before cal window for context
    train_end = cal_start
    train_df = df[df["timestamps"] < train_end].reset_index(drop=True)

    return train_df, cal_df, val_df


def build_ensemble(market: str, device: str):
    """Build the appropriate ensemble for a market."""
    from model import Kronos, KronosTokenizer, KronosPredictor
    from src.model.ensemble import EnsemblePredictor

    predictors = []
    names = []
    weights = []

    # Zero-shot (shared)
    print("  Loading zero-shot Kronos-base...")
    zs_tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    zs_mdl = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    zs_mdl = zs_mdl.to(device).eval()
    predictors.append(KronosPredictor(zs_mdl, zs_tok, max_context=512))
    names.append("zero_shot")
    weights.append(1.0)

    if market == "crypto":
        # BTC-finetuned v1 + v2
        for exp in ["cifr-btc", "cifr-btc-v2"]:
            tok_path = CHECKPOINTS_DIR / exp / "tokenizer" / "best_model"
            mdl_path = CHECKPOINTS_DIR / exp / "predictor" / "best_model"
            if tok_path.exists() and mdl_path.exists():
                print(f"  Loading {exp}...")
                ft_tok = KronosTokenizer.from_pretrained(str(tok_path))
                ft_mdl = Kronos.from_pretrained(str(mdl_path))
                ft_mdl = ft_mdl.to(device).eval()
                predictors.append(KronosPredictor(ft_mdl, ft_tok, max_context=512))
                names.append(exp)
                weights.append(1.0)
            else:
                print(f"  WARNING: {exp} checkpoint not found, skipping")

    elif market == "commodity":
        # XAU-finetuned v1 + v2
        for exp in ["cifr-xau", "cifr-xau-v2"]:
            tok_path = CHECKPOINTS_DIR / exp / "tokenizer" / "best_model"
            mdl_path = CHECKPOINTS_DIR / exp / "predictor" / "best_model"
            if tok_path.exists() and mdl_path.exists():
                print(f"  Loading {exp}...")
                ft_tok = KronosTokenizer.from_pretrained(str(tok_path))
                ft_mdl = Kronos.from_pretrained(str(mdl_path))
                ft_mdl = ft_mdl.to(device).eval()
                predictors.append(KronosPredictor(ft_mdl, ft_tok, max_context=512))
                names.append(exp)
                weights.append(1.0)
            else:
                print(f"  WARNING: {exp} checkpoint not found, skipping")

    ensemble = EnsemblePredictor(predictors, weights, names)
    print(f"  Ensemble: {ensemble}")
    return ensemble


def calibrate_asset(
    ensemble,
    train_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    val_df: pd.DataFrame,
    pred_len: int,
    lookback: int = 512,
    n_paths: int = 30,
    step_size: int = None,
    coverage: float = 0.90,
):
    """
    Run CQR calibration for a single asset.

    1. Roll through calibration window, generate ensemble predictions
    2. Compute conformity scores
    3. Find correction term for target coverage
    4. Validate on val set
    """
    from src.risk.cqr import calibrate as cqr_calibrate

    if step_size is None:
        step_size = pred_len  # Non-overlapping windows

    # Combine train tail + cal for rolling context
    context_data = pd.concat([train_df.tail(lookback), cal_df]).reset_index(drop=True)

    y_true_list = []
    q_lower_list = []
    q_upper_list = []

    total_windows = (len(cal_df) - pred_len) // step_size
    print(f"    Rolling {total_windows} calibration windows (step={step_size})...")

    for i in range(0, len(cal_df) - pred_len, step_size):
        # Context: lookback candles ending at cal position i
        ctx_end = lookback + i
        ctx_start = ctx_end - lookback

        if ctx_start < 0 or ctx_end + pred_len > len(context_data):
            continue

        context = context_data.iloc[ctx_start:ctx_end]
        future = context_data.iloc[ctx_end:ctx_end + pred_len]

        if len(future) < pred_len:
            break

        x_df = context[["open", "high", "low", "close", "volume", "amount"]]
        x_ts = context["timestamps"]
        y_ts = future["timestamps"]

        try:
            result = ensemble.predict_with_quantiles(
                df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
                pred_len=pred_len, n_paths=n_paths,
            )

            # Use final-step close quantiles
            q_lo = result["close_q05"][-1]
            q_hi = result["close_q95"][-1]
            y_actual = future["close"].iloc[-1]

            y_true_list.append(y_actual)
            q_lower_list.append(q_lo)
            q_upper_list.append(q_hi)
        except Exception as e:
            print(f"    Window {i} failed: {e}")
            continue

    if len(y_true_list) < 5:
        print(f"    WARNING: Only {len(y_true_list)} calibration samples, too few for CQR")
        return None

    y_true = np.array(y_true_list)
    q_lower = np.array(q_lower_list)
    q_upper = np.array(q_upper_list)

    # Calibrate
    cal_result = cqr_calibrate(y_true, q_lower, q_upper, coverage=coverage)
    print(f"    {cal_result}")

    # Validate on val set if enough data
    if len(val_df) > lookback + pred_len:
        val_context = pd.concat([cal_df.tail(lookback), val_df]).reset_index(drop=True)
        val_y = []
        val_lo = []
        val_hi = []

        val_windows = min(20, (len(val_df) - pred_len) // step_size)
        for i in range(0, val_windows * step_size, step_size):
            ctx_end = lookback + i
            ctx_start = ctx_end - lookback
            if ctx_end + pred_len > len(val_context):
                break

            context = val_context.iloc[ctx_start:ctx_end]
            future = val_context.iloc[ctx_end:ctx_end + pred_len]
            if len(future) < pred_len:
                break

            try:
                result = ensemble.predict_with_quantiles(
                    df=context[["open", "high", "low", "close", "volume", "amount"]],
                    x_timestamp=context["timestamps"],
                    y_timestamp=future["timestamps"],
                    pred_len=pred_len, n_paths=n_paths,
                )
                val_y.append(future["close"].iloc[-1])
                val_lo.append(result["close_q05"][-1] - cal_result.correction)
                val_hi.append(result["close_q95"][-1] + cal_result.correction)
            except:
                continue

        if val_y:
            val_y = np.array(val_y)
            val_lo = np.array(val_lo)
            val_hi = np.array(val_hi)
            val_coverage = float(np.mean((val_y >= val_lo) & (val_y <= val_hi)))
            print(f"    Validation coverage: {val_coverage:.1%} (target: {coverage:.0%})")

    return cal_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=["crypto", "commodity", "all"], default="all")
    parser.add_argument("--n-paths", type=int, default=30)
    parser.add_argument("--coverage", type=float, default=0.90)
    parser.add_argument("--step-size", type=int, default=None,
                        help="Step size between calibration windows (default: pred_len)")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    results = {}

    # --- CRYPTO ---
    if args.market in ("crypto", "all"):
        print(f"\n{'='*60}")
        print("  CRYPTO CQR CALIBRATION")
        print(f"{'='*60}")

        from configs.crypto_universe import get_crypto_configs
        configs = get_crypto_configs(tiers=(1, 2))
        ensemble = build_ensemble("crypto", device)

        for symbol, cfg in configs.items():
            print(f"\n  [{symbol}]")
            try:
                df = load_raw_data("crypto", symbol)
                print(f"    Data: {len(df)} candles ({df['timestamps'].iloc[0].date()} to {df['timestamps'].iloc[-1].date()})")

                train_df, cal_df, val_df = split_calibration(df)
                print(f"    Cal window: {len(cal_df)} candles, Val: {len(val_df)} candles")

                step = args.step_size or cfg.pred_len
                cal_result = calibrate_asset(
                    ensemble, train_df, cal_df, val_df,
                    pred_len=cfg.pred_len, n_paths=args.n_paths,
                    step_size=step, coverage=args.coverage,
                )

                if cal_result:
                    results[symbol] = {
                        "market": "crypto",
                        "correction": cal_result.correction,
                        "coverage_target": cal_result.coverage_target,
                        "coverage_achieved": cal_result.coverage_achieved,
                        "n_calibration": cal_result.n_calibration,
                    }
            except Exception as e:
                print(f"    FAILED: {e}")

    # --- COMMODITIES ---
    if args.market in ("commodity", "all"):
        print(f"\n{'='*60}")
        print("  COMMODITY CQR CALIBRATION")
        print(f"{'='*60}")

        from configs.commodity_universe import get_commodity_configs
        configs = get_commodity_configs(categories=("precious", "energy"))
        ensemble = build_ensemble("commodity", device)

        # Map config keys to data file names
        key_to_file = {
            "XAU/USD": "xau_usd",
            "XAG/USD": "xag_usd",
            "XPT/USD": "xpt_usd",
            "CRUDE_OIL": "wti_usd",
            "BRENT_OIL": "brent_usd",
            "NATURAL_GAS": "ng_usd",
            "COPPER": "copper_usd",
        }

        for key, cfg in configs.items():
            file_key = key_to_file.get(key, key.replace("/", "_").lower())
            print(f"\n  [{key} → {cfg.instrument}]")
            try:
                # Try loading from commodity dir
                path = DATA_RAW_DIR / "commodity" / f"{file_key}_4h.csv"
                if not path.exists():
                    # Try the old xau path
                    path = DATA_RAW_DIR / "xau" / "gold_4h.csv" if "xau" in file_key else path

                if not path.exists():
                    print(f"    SKIP: no data at {path}")
                    continue

                df = pd.read_csv(path)
                df["timestamps"] = pd.to_datetime(df["timestamps"])
                print(f"    Data: {len(df)} candles ({df['timestamps'].iloc[0].date()} to {df['timestamps'].iloc[-1].date()})")

                train_df, cal_df, val_df = split_calibration(df)
                print(f"    Cal window: {len(cal_df)} candles, Val: {len(val_df)} candles")

                step = args.step_size or cfg.pred_len
                cal_result = calibrate_asset(
                    ensemble, train_df, cal_df, val_df,
                    pred_len=cfg.pred_len, n_paths=args.n_paths,
                    step_size=step, coverage=args.coverage,
                )

                if cal_result:
                    results[key] = {
                        "market": "commodity",
                        "instrument": cfg.instrument,
                        "correction": cal_result.correction,
                        "coverage_target": cal_result.coverage_target,
                        "coverage_achieved": cal_result.coverage_achieved,
                        "n_calibration": cal_result.n_calibration,
                    }
            except Exception as e:
                print(f"    FAILED: {e}")

    # Save results
    output_dir = RESULTS_DIR / "cqr"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "cqr_calibrations.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nSaved CQR calibrations to {output_path}")

    # Summary
    print(f"\n{'='*60}")
    print("  CQR CALIBRATION SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Asset':<15} {'Market':<10} {'Correction':>12} {'Coverage':>10} {'N':>6}")
    print(f"  {'-'*55}")
    for asset, r in results.items():
        print(f"  {asset:<15} {r['market']:<10} {r['correction']:>+12.4f} {r['coverage_achieved']:>9.1%} {r['n_calibration']:>6}")


if __name__ == "__main__":
    main()
