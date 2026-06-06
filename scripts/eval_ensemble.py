"""
Ensemble evaluation — the real SOTA benchmark.

Compares: zero-shot, finetuned, ensemble (ZS+FT), and multi-seed ensemble.
Uses 50+ evaluation windows with bootstrap confidence intervals.

Usage:
    python scripts/eval_ensemble.py --markets eur xau --n-windows 50 --n-paths 50
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Kronos"))

from model import Kronos, KronosTokenizer, KronosPredictor


# ─── Config ───

MARKETS = {
    "btc": {"name": "BTC/USDT", "timeframe": "15m", "pred_len": 48,
            "lookback": 400, "data_file": "data/processed/btc/validation.csv", "exp_name": "cifr-btc"},
    "eur": {"name": "EUR/USD", "timeframe": "1h", "pred_len": 24,
            "lookback": 400, "data_file": "data/processed/eur/validation.csv", "exp_name": "cifr-eur"},
    "xau": {"name": "XAU/USD", "timeframe": "4h", "pred_len": 6,
            "lookback": 200, "data_file": "data/processed/xau/validation.csv", "exp_name": "cifr-xau"},
}

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results" / "ensemble"


# ─── Metrics ───

def compute_metrics(pred_returns, actual_returns, pred_closes, actual_closes):
    """Compute all metrics with bootstrap confidence intervals."""
    pred = np.array(pred_returns)
    actual = np.array(actual_returns)
    pred_c = np.array(pred_closes)
    actual_c = np.array(actual_closes)

    if len(pred) < 3:
        return {}

    ic, _ = stats.pearsonr(pred, actual)
    rank_ic, _ = stats.spearmanr(pred, actual)
    dir_acc = float(np.mean(np.sign(pred) == np.sign(actual)))
    rmse = float(np.sqrt(np.mean((pred_c - actual_c) ** 2)))
    mae = float(np.mean(np.abs(pred_c - actual_c)))

    # Bootstrap CIs for directional accuracy and IC
    n_boot = 1000
    ic_boots = []
    da_boots = []
    for _ in range(n_boot):
        idx = np.random.choice(len(pred), len(pred), replace=True)
        if len(np.unique(pred[idx])) < 3:
            continue
        try:
            ic_b, _ = stats.pearsonr(pred[idx], actual[idx])
            ic_boots.append(ic_b)
        except:
            pass
        da_boots.append(float(np.mean(np.sign(pred[idx]) == np.sign(actual[idx]))))

    return {
        "ic": float(ic) if not np.isnan(ic) else 0.0,
        "ic_ci_lo": float(np.percentile(ic_boots, 2.5)) if ic_boots else 0.0,
        "ic_ci_hi": float(np.percentile(ic_boots, 97.5)) if ic_boots else 0.0,
        "rank_ic": float(rank_ic) if not np.isnan(rank_ic) else 0.0,
        "directional_accuracy": dir_acc,
        "da_ci_lo": float(np.percentile(da_boots, 2.5)) if da_boots else 0.0,
        "da_ci_hi": float(np.percentile(da_boots, 97.5)) if da_boots else 0.0,
        "rmse": rmse,
        "mae": mae,
        "n_windows": len(pred),
        "mean_pred_return": float(np.mean(pred)),
        "mean_actual_return": float(np.mean(actual)),
        "hit_rate_long": float(np.mean(actual[pred > 0] > 0)) if np.sum(pred > 0) > 0 else 0.0,
        "hit_rate_short": float(np.mean(actual[pred < 0] < 0)) if np.sum(pred < 0) > 0 else 0.0,
    }


# ─── Evaluation Engine ───

def run_evaluation(predictors_dict, market_key, config, n_windows=50, n_paths=5):
    """
    Run rolling evaluation with multiple models simultaneously.

    Args:
        predictors_dict: {'model_name': predictor_or_list}
        market_key: 'btc', 'eur', 'xau'
        config: market config dict
        n_windows: evaluation windows
        n_paths: Monte Carlo paths per prediction (sample_count)

    Returns:
        dict of {model_name: metrics_dict}
    """
    data_path = PROJECT_ROOT / config["data_file"]
    df = pd.read_csv(data_path, parse_dates=["timestamps"])

    pred_len = config["pred_len"]
    lookback = config["lookback"]

    total_available = len(df) - lookback - pred_len
    if total_available <= 0:
        print(f"  Not enough data for {config['name']}")
        return {}

    step = max(1, total_available // n_windows)
    actual_windows = min(n_windows, total_available // step)

    print(f"\n{'='*70}")
    print(f"  {config['name']} ({config['timeframe']}) — {actual_windows} windows, {n_paths} paths")
    print(f"  Data: {len(df)} candles, lookback={lookback}, pred_len={pred_len}")
    print(f"{'='*70}")

    # Collect results per model
    results = {name: {"pred_returns": [], "actual_returns": [],
                       "pred_closes": [], "actual_closes": []}
               for name in predictors_dict}

    for i in range(actual_windows):
        start_idx = i * step
        end_idx = start_idx + lookback

        context = df.iloc[start_idx:end_idx]
        future = df.iloc[end_idx:end_idx + pred_len]
        if len(future) < pred_len:
            break

        x_df = context[["open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)
        x_ts = context["timestamps"].reset_index(drop=True)
        y_ts = future["timestamps"].reset_index(drop=True)

        actual_close = future["close"].iloc[-1]
        entry_close = context["close"].iloc[-1]
        actual_return = (actual_close - entry_close) / entry_close

        status_parts = []
        for name, predictor_info in predictors_dict.items():
            # Handle ensemble (list of predictors with weights)
            if isinstance(predictor_info, list):
                # Ensemble: weighted average
                preds_weighted = []
                total_w = 0
                for pred, w in predictor_info:
                    try:
                        p = pred.predict(df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
                                         pred_len=pred_len, T=1.0, top_p=0.9,
                                         sample_count=n_paths)
                        preds_weighted.append((p["close"].iloc[-1], w))
                        total_w += w
                    except:
                        continue
                if not preds_weighted:
                    continue
                pred_close = sum(pc * w for pc, w in preds_weighted) / total_w
            else:
                # Single predictor
                try:
                    pred_df = predictor_info.predict(
                        df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
                        pred_len=pred_len, T=1.0, top_p=0.9,
                        sample_count=n_paths,
                    )
                    pred_close = pred_df["close"].iloc[-1]
                except Exception as e:
                    continue

            pred_return = (pred_close - entry_close) / entry_close
            results[name]["pred_returns"].append(pred_return)
            results[name]["actual_returns"].append(actual_return)
            results[name]["pred_closes"].append(pred_close)
            results[name]["actual_closes"].append(actual_close)

            direction = "+" if np.sign(pred_return) == np.sign(actual_return) else "-"
            status_parts.append(f"{name}:{direction}")

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{actual_windows}] {' '.join(status_parts)}")

    # Compute metrics
    all_metrics = {}
    for name in results:
        if results[name]["pred_returns"]:
            m = compute_metrics(
                results[name]["pred_returns"], results[name]["actual_returns"],
                results[name]["pred_closes"], results[name]["actual_closes"],
            )
            m["model"] = name
            m["market"] = config["name"]
            all_metrics[name] = m

    return all_metrics


def load_model_to_predictor(model_path, tokenizer_path, device):
    """Load a single model into a KronosPredictor."""
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
    model = Kronos.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    return KronosPredictor(model, tokenizer, max_context=512)


# ─── Main ───

def main():
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--markets", nargs="+", default=["eur", "xau"])
    parser.add_argument("--n-windows", type=int, default=50)
    parser.add_argument("--n-paths", type=int, default=10)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    all_results = []

    for mk in args.markets:
        if mk not in MARKETS:
            continue
        config = MARKETS[mk]
        exp_name = config["exp_name"]

        # Build predictors dict
        predictors = {}

        # 1. Zero-shot
        print(f"Loading zero-shot model...")
        zs_pred = load_model_to_predictor(
            "NeoQuasar/Kronos-base", "NeoQuasar/Kronos-Tokenizer-base", device)
        predictors["zero_shot"] = zs_pred

        # 2. Finetuned v1
        tok_v1 = CHECKPOINTS_DIR / exp_name / "tokenizer" / "best_model"
        mdl_v1 = CHECKPOINTS_DIR / exp_name / "predictor" / "best_model"
        if tok_v1.exists() and mdl_v1.exists():
            print(f"Loading finetuned v1 ({exp_name})...")
            ft_pred = load_model_to_predictor(str(mdl_v1), str(tok_v1), device)
            predictors["finetuned_v1"] = ft_pred

            # 3. Ensemble (ZS + FT, equal weight)
            predictors["ensemble_eq"] = [(zs_pred, 1.0), (ft_pred, 1.0)]

        # 4. Finetuned v2 (multi-seed or improved hyperparams)
        exp_v2 = exp_name + "-v2"
        tok_v2 = CHECKPOINTS_DIR / exp_v2 / "tokenizer" / "best_model"
        mdl_v2 = CHECKPOINTS_DIR / exp_v2 / "predictor" / "best_model"
        if tok_v2.exists() and mdl_v2.exists():
            print(f"Loading finetuned v2 ({exp_v2})...")
            ft2_pred = load_model_to_predictor(str(mdl_v2), str(tok_v2), device)
            predictors["finetuned_v2"] = ft2_pred

            # 5. Full ensemble (ZS + FT_v1 + FT_v2)
            if "finetuned_v1" in predictors:
                predictors["ensemble_full"] = [
                    (zs_pred, 1.0), (ft_pred, 1.0), (ft2_pred, 1.0)
                ]

        # Run evaluation
        metrics = run_evaluation(predictors, mk, config,
                                  n_windows=args.n_windows, n_paths=args.n_paths)

        # Print results
        print(f"\n{'='*90}")
        print(f"  {config['name']} RESULTS")
        print(f"{'='*90}")
        print(f"  {'Model':<18} {'IC':>8} {'IC 95%CI':>16} {'RankIC':>8} "
              f"{'Dir.Acc':>8} {'DA 95%CI':>16} {'RMSE':>10}")
        print(f"  {'-'*86}")

        for name, m in sorted(metrics.items()):
            ic_ci = f"[{m['ic_ci_lo']:+.3f},{m['ic_ci_hi']:+.3f}]"
            da_ci = f"[{m['da_ci_lo']:.1%},{m['da_ci_hi']:.1%}]"
            print(f"  {name:<18} {m['ic']:>+8.4f} {ic_ci:>16} {m['rank_ic']:>+8.4f} "
                  f"{m['directional_accuracy']:>7.1%} {da_ci:>16} {m['rmse']:>10.4f}")
            all_results.append(m)

        # Cleanup GPU memory
        del predictors
        torch.cuda.empty_cache() if device == "cuda" else None

    # Save
    if all_results:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_results).to_csv(RESULTS_DIR / "ensemble_metrics.csv", index=False)
        print(f"\nSaved -> {RESULTS_DIR / 'ensemble_metrics.csv'}")


if __name__ == "__main__":
    main()
