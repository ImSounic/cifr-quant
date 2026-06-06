"""
Finetuned model evaluation — compare against zero-shot baseline.

Runs finetuned Kronos checkpoints on validation data and computes
the same metrics as zero_shot_baseline.py for direct comparison.

Usage:
    python scripts/finetuned_eval.py                    # All available markets
    python scripts/finetuned_eval.py --markets eur xau  # Specific markets
"""

import sys
import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Kronos"))

from model import Kronos, KronosTokenizer, KronosPredictor


# ─── Configuration ───

MARKETS = {
    "btc": {
        "name": "BTC/USDT",
        "timeframe": "15m",
        "pred_len": 48,
        "lookback": 400,
        "data_file": "data/processed/btc/validation.csv",
        "exp_name": "cifr-btc",
    },
    "eur": {
        "name": "EUR/USD",
        "timeframe": "1h",
        "pred_len": 24,
        "lookback": 400,
        "data_file": "data/processed/eur/validation.csv",
        "exp_name": "cifr-eur",
    },
    "xau": {
        "name": "XAU/USD",
        "timeframe": "4h",
        "pred_len": 6,
        "lookback": 200,
        "data_file": "data/processed/xau/validation.csv",
        "exp_name": "cifr-xau",
    },
}

N_EVAL_WINDOWS = 20
SAMPLE_COUNT = 5
RESULTS_DIR = PROJECT_ROOT / "results" / "finetuned"

# Checkpoints saved by train_sequential.py (via Kronos/finetune_csv/)
# Symlink: cifr-quant/checkpoints -> cifr-quant/Kronos/checkpoints
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"


# ─── Metrics ───

def compute_ic(predicted, actual):
    if len(predicted) < 3:
        return 0.0
    corr, _ = stats.pearsonr(predicted, actual)
    return float(corr) if not np.isnan(corr) else 0.0


def compute_rank_ic(predicted, actual):
    if len(predicted) < 3:
        return 0.0
    corr, _ = stats.spearmanr(predicted, actual)
    return float(corr) if not np.isnan(corr) else 0.0


def compute_directional_accuracy(predicted, actual):
    if len(predicted) == 0:
        return 0.0
    return float(np.mean(np.sign(predicted) == np.sign(actual)))


def compute_rmse(predicted, actual):
    return float(np.sqrt(np.mean((predicted - actual) ** 2)))


def compute_mae(predicted, actual):
    return float(np.mean(np.abs(predicted - actual)))


# ─── Evaluation ───

def load_finetuned_model(market_key, config, device):
    """Load finetuned tokenizer + predictor for a market."""
    exp_name = config["exp_name"]
    ckpt_dir = CHECKPOINTS_DIR / exp_name

    tok_path = ckpt_dir / "tokenizer" / "best_model"
    mdl_path = ckpt_dir / "predictor" / "best_model"

    if not tok_path.exists():
        raise FileNotFoundError(f"No finetuned tokenizer at {tok_path}")
    if not mdl_path.exists():
        raise FileNotFoundError(f"No finetuned predictor at {mdl_path}")

    print(f"  Tokenizer: {tok_path}")
    print(f"  Predictor: {mdl_path}")

    tokenizer = KronosTokenizer.from_pretrained(str(tok_path))
    model = Kronos.from_pretrained(str(mdl_path))
    model = model.to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Loaded: {param_count/1e6:.1f}M params on {device}")

    return KronosPredictor(model, tokenizer, max_context=512)


def evaluate_market(predictor, market_key, config, label="Finetuned"):
    """Rolling evaluation on validation data."""
    data_path = PROJECT_ROOT / config["data_file"]
    df = pd.read_csv(data_path, parse_dates=["timestamps"])

    pred_len = config["pred_len"]
    lookback = config["lookback"]
    name = config["name"]

    print(f"\n{'='*60}")
    print(f"Evaluating {name} ({config['timeframe']}) — {label}")
    print(f"Data: {len(df)} candles, lookback={lookback}, pred_len={pred_len}")
    print(f"{'='*60}")

    total_available = len(df) - lookback - pred_len
    if total_available <= 0:
        print(f"  ERROR: Not enough data. Need {lookback + pred_len}, have {len(df)}")
        return {}

    step = max(1, total_available // N_EVAL_WINDOWS)
    n_windows = min(N_EVAL_WINDOWS, total_available // step)

    all_pred_returns = []
    all_actual_returns = []
    all_pred_closes = []
    all_actual_closes = []
    sample_predictions = []

    for i in range(n_windows):
        start_idx = i * step
        end_idx = start_idx + lookback

        context = df.iloc[start_idx:end_idx]
        x_df = context[["open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)
        x_ts = context["timestamps"].reset_index(drop=True)

        future = df.iloc[end_idx:end_idx + pred_len]
        if len(future) < pred_len:
            break
        y_ts = future["timestamps"].reset_index(drop=True)

        try:
            pred_df = predictor.predict(
                df=x_df,
                x_timestamp=x_ts,
                y_timestamp=y_ts,
                pred_len=pred_len,
                T=1.0,
                top_p=0.9,
                sample_count=SAMPLE_COUNT,
            )
        except Exception as e:
            print(f"  Window {i+1}: prediction failed — {e}")
            continue

        pred_close = pred_df["close"].iloc[-1]
        actual_close = future["close"].iloc[-1]
        entry_close = context["close"].iloc[-1]

        pred_return = (pred_close - entry_close) / entry_close
        actual_return = (actual_close - entry_close) / entry_close

        all_pred_returns.append(pred_return)
        all_actual_returns.append(actual_return)
        all_pred_closes.append(pred_close)
        all_actual_closes.append(actual_close)

        if len(sample_predictions) < 3:
            sample_predictions.append({
                "context_close": context["close"].values,
                "actual_close": future["close"].values,
                "pred_close": pred_df["close"].values,
            })

        direction = "✓" if np.sign(pred_return) == np.sign(actual_return) else "✗"
        print(f"  Window {i+1}/{n_windows}: pred={pred_return:+.4f} actual={actual_return:+.4f} {direction}")

    if not all_pred_returns:
        print("  No successful predictions!")
        return {}

    pred_arr = np.array(all_pred_returns)
    actual_arr = np.array(all_actual_returns)
    pred_close_arr = np.array(all_pred_closes)
    actual_close_arr = np.array(all_actual_closes)

    metrics = {
        "market": name,
        "timeframe": config["timeframe"],
        "model": label,
        "n_windows": len(pred_arr),
        "ic": compute_ic(pred_arr, actual_arr),
        "rank_ic": compute_rank_ic(pred_arr, actual_arr),
        "directional_accuracy": compute_directional_accuracy(pred_arr, actual_arr),
        "rmse": compute_rmse(pred_close_arr, actual_close_arr),
        "mae": compute_mae(pred_close_arr, actual_close_arr),
        "mean_pred_return": float(np.mean(pred_arr)),
        "mean_actual_return": float(np.mean(actual_arr)),
    }

    print(f"\n  Results ({label}):")
    print(f"    IC:                   {metrics['ic']:+.4f}")
    print(f"    Rank IC:              {metrics['rank_ic']:+.4f}")
    print(f"    Directional Accuracy: {metrics['directional_accuracy']:.1%}")
    print(f"    RMSE:                 {metrics['rmse']:.4f}")
    print(f"    MAE:                  {metrics['mae']:.4f}")

    if sample_predictions:
        plot_samples(market_key, name, sample_predictions, label)

    return metrics


def plot_samples(market_key, name, samples, label):
    """Plot sample predictions vs actual."""
    fig, axes = plt.subplots(1, len(samples), figsize=(6 * len(samples), 4))
    if len(samples) == 1:
        axes = [axes]

    for idx, (ax, sample) in enumerate(zip(axes, samples)):
        ctx_len = len(sample["context_close"])
        pred_len = len(sample["pred_close"])

        show_ctx = min(100, ctx_len)
        x_ctx = range(show_ctx)
        x_pred = range(show_ctx, show_ctx + pred_len)

        ax.plot(x_ctx, sample["context_close"][-show_ctx:],
                color="steelblue", linewidth=1, label="History")
        ax.plot(x_pred, sample["actual_close"],
                color="black", linewidth=1.5, label="Actual")
        ax.plot(x_pred, sample["pred_close"],
                color="red", linewidth=1.5, linestyle="--", label="Predicted")
        ax.axvline(x=show_ctx, color="gray", linestyle=":", alpha=0.5)
        ax.set_title(f"{name} — {label} Sample {idx+1}", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = RESULTS_DIR / f"{market_key}_finetuned_samples.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot -> {out_path}")


def plot_comparison(all_metrics):
    """Bar chart comparing zero-shot vs finetuned metrics."""
    markets = sorted(set(m["market"] for m in all_metrics))
    metric_names = ["ic", "rank_ic", "directional_accuracy"]
    labels = {"ic": "IC", "rank_ic": "Rank IC", "directional_accuracy": "Dir. Accuracy"}

    fig, axes = plt.subplots(1, len(metric_names), figsize=(5 * len(metric_names), 5))

    for ax, metric_name in zip(axes, metric_names):
        x = np.arange(len(markets))
        width = 0.35

        zs_vals = []
        ft_vals = []
        for mkt in markets:
            zs = [m for m in all_metrics if m["market"] == mkt and m["model"] == "Zero-Shot"]
            ft = [m for m in all_metrics if m["market"] == mkt and m["model"] == "Finetuned"]
            zs_vals.append(zs[0][metric_name] if zs else 0)
            ft_vals.append(ft[0][metric_name] if ft else 0)

        ax.bar(x - width/2, zs_vals, width, label="Zero-Shot", color="steelblue", alpha=0.8)
        ax.bar(x + width/2, ft_vals, width, label="Finetuned", color="coral", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(markets, fontsize=10)
        ax.set_title(labels[metric_name], fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        ax.axhline(y=0, color="black", linewidth=0.5)

    plt.tight_layout()
    out_path = RESULTS_DIR / "zero_shot_vs_finetuned.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nComparison plot -> {out_path}")


# ─── Main ───

def main():
    import torch

    parser = argparse.ArgumentParser(description="Evaluate finetuned Kronos checkpoints")
    parser.add_argument("--markets", nargs="+", default=None,
                        help="Markets to evaluate (default: all with checkpoints)")
    parser.add_argument("--compare", action="store_true",
                        help="Also run zero-shot for comparison")
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # Which markets to evaluate
    target_markets = args.markets or list(MARKETS.keys())
    available = []
    for mk in target_markets:
        if mk not in MARKETS:
            print(f"Unknown market: {mk}")
            continue
        exp = MARKETS[mk]["exp_name"]
        tok = CHECKPOINTS_DIR / exp / "tokenizer" / "best_model"
        mdl = CHECKPOINTS_DIR / exp / "predictor" / "best_model"
        if tok.exists() and mdl.exists():
            available.append(mk)
        else:
            print(f"Skipping {mk}: checkpoint not found at {CHECKPOINTS_DIR / exp}")

    if not available:
        print("No finetuned checkpoints found. Run finetuning first.")
        return

    print(f"\nEvaluating finetuned models for: {', '.join(available)}")

    all_metrics = []

    # Run zero-shot baseline if --compare
    if args.compare:
        print(f"\n{'#'*60}")
        print("ZERO-SHOT BASELINE")
        print(f"{'#'*60}")

        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
        model = model.to(device)
        model.eval()
        zs_predictor = KronosPredictor(model, tokenizer, max_context=512)

        for mk in available:
            metrics = evaluate_market(zs_predictor, mk, MARKETS[mk], label="Zero-Shot")
            if metrics:
                all_metrics.append(metrics)

        del model, tokenizer, zs_predictor
        torch.cuda.empty_cache() if device == "cuda" else None

    # Run finetuned evaluation
    print(f"\n{'#'*60}")
    print("FINETUNED EVALUATION")
    print(f"{'#'*60}")

    for mk in available:
        config = MARKETS[mk]
        print(f"\nLoading finetuned model for {config['name']}...")
        try:
            predictor = load_finetuned_model(mk, config, device)
        except Exception as e:
            print(f"  Failed to load: {e}")
            continue

        metrics = evaluate_market(predictor, mk, config, label="Finetuned")
        if metrics:
            all_metrics.append(metrics)

        del predictor
        torch.cuda.empty_cache() if device == "cuda" else None

    # Summary
    if all_metrics:
        print(f"\n{'='*80}")
        print("EVALUATION SUMMARY")
        print(f"{'='*80}")
        print(f"{'Market':<12} {'Model':<12} {'IC':>8} {'RankIC':>8} {'Dir.Acc':>8} {'RMSE':>10} {'MAE':>10}")
        print("-" * 80)
        for m in sorted(all_metrics, key=lambda x: (x["market"], x["model"])):
            print(f"{m['market']:<12} {m['model']:<12} {m['ic']:>+8.4f} {m['rank_ic']:>+8.4f} "
                  f"{m['directional_accuracy']:>7.1%} {m['rmse']:>10.4f} {m['mae']:>10.4f}")
        print(f"{'='*80}")

        # Save
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        results_df = pd.DataFrame(all_metrics)
        out_path = RESULTS_DIR / "finetuned_metrics.csv"
        results_df.to_csv(out_path, index=False)
        print(f"\nSaved metrics -> {out_path}")

        if args.compare:
            plot_comparison(all_metrics)


if __name__ == "__main__":
    main()
