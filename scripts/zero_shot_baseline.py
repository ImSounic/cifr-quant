"""
Zero-shot baseline evaluation for all 3 markets.

Runs pre-trained Kronos-base (no finetuning) on validation data
to establish baseline metrics before finetuning.

Usage:
    python scripts/zero_shot_baseline.py

Runs on Mac Mini CPU/MPS — no GPU required (~4GB memory).
"""

import sys
import os
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
        "pred_len": 48,      # 12 hours
        "lookback": 400,
        "data_file": "data/processed/btc/validation.csv",
    },
    "eur": {
        "name": "EUR/USD",
        "timeframe": "1h",
        "pred_len": 24,      # 1 day
        "lookback": 400,
        "data_file": "data/processed/eur/validation.csv",
    },
    "xau": {
        "name": "XAU/USD",
        "timeframe": "4h",
        "pred_len": 6,       # 1 day
        "lookback": 200,     # Smaller: only 385 candles in validation
        "data_file": "data/processed/xau/validation.csv",
    },
}

LOOKBACK = 400          # Default (overridden per-market above)
N_EVAL_WINDOWS = 20     # Number of rolling evaluation windows per market
SAMPLE_COUNT = 5        # Paths to average for point forecast (speed vs accuracy)
RESULTS_DIR = PROJECT_ROOT / "results" / "zero_shot"


# ─── Metrics ───

def compute_ic(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Information Coefficient: Pearson correlation between predicted and actual returns."""
    if len(predicted) < 3:
        return 0.0
    corr, _ = stats.pearsonr(predicted, actual)
    return float(corr) if not np.isnan(corr) else 0.0


def compute_rank_ic(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Rank IC: Spearman rank correlation between predicted and actual returns."""
    if len(predicted) < 3:
        return 0.0
    corr, _ = stats.spearmanr(predicted, actual)
    return float(corr) if not np.isnan(corr) else 0.0


def compute_directional_accuracy(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Fraction of predictions with correct direction (up/down)."""
    if len(predicted) == 0:
        return 0.0
    correct = np.sign(predicted) == np.sign(actual)
    return float(np.mean(correct))


def compute_rmse(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((predicted - actual) ** 2)))


def compute_mae(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(predicted - actual)))


# ─── Evaluation ───

def evaluate_market(
    predictor: KronosPredictor,
    market_key: str,
    config: dict,
) -> dict:
    """
    Run rolling zero-shot evaluation on a single market.

    Slides a window through validation data:
    [lookback context] → [predict pred_len candles] → compare to actual
    """
    data_path = PROJECT_ROOT / config["data_file"]
    df = pd.read_csv(data_path, parse_dates=["timestamps"])

    pred_len = config["pred_len"]
    lookback = config.get("lookback", LOOKBACK)
    name = config["name"]

    print(f"\n{'='*60}")
    print(f"Evaluating {name} ({config['timeframe']}) — Zero Shot")
    print(f"Data: {len(df)} candles, lookback={lookback}, pred_len={pred_len}")
    print(f"{'='*60}")

    # Calculate step size to get N_EVAL_WINDOWS evenly spaced windows
    total_available = len(df) - lookback - pred_len
    if total_available <= 0:
        print(f"  ERROR: Not enough data. Need {LOOKBACK + pred_len}, have {len(df)}")
        return {}

    step = max(1, total_available // N_EVAL_WINDOWS)
    n_windows = min(N_EVAL_WINDOWS, total_available // step)

    all_pred_returns = []
    all_actual_returns = []
    all_pred_closes = []
    all_actual_closes = []
    sample_predictions = []  # Store a few for visualization

    for i in range(n_windows):
        start_idx = i * step
        end_idx = start_idx + lookback

        # Context
        context = df.iloc[start_idx:end_idx]
        x_df = context[["open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)
        x_ts = context["timestamps"].reset_index(drop=True)

        # Future (ground truth)
        future = df.iloc[end_idx:end_idx + pred_len]
        if len(future) < pred_len:
            break
        y_ts = future["timestamps"].reset_index(drop=True)

        # Predict
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

        # Compare at final prediction timestep
        pred_close = pred_df["close"].iloc[-1]
        actual_close = future["close"].iloc[-1]
        entry_close = context["close"].iloc[-1]

        pred_return = (pred_close - entry_close) / entry_close
        actual_return = (actual_close - entry_close) / entry_close

        all_pred_returns.append(pred_return)
        all_actual_returns.append(actual_return)
        all_pred_closes.append(pred_close)
        all_actual_closes.append(actual_close)

        # Store first 3 for visualization
        if len(sample_predictions) < 3:
            sample_predictions.append({
                "context_close": context["close"].values,
                "actual_close": future["close"].values,
                "pred_close": pred_df["close"].values,
                "timestamps_ctx": context["timestamps"].values,
                "timestamps_fut": future["timestamps"].values,
            })

        direction = "✓" if np.sign(pred_return) == np.sign(actual_return) else "✗"
        print(f"  Window {i+1}/{n_windows}: pred={pred_return:+.4f} actual={actual_return:+.4f} {direction}")

    if not all_pred_returns:
        print("  No successful predictions!")
        return {}

    # Compute metrics
    pred_arr = np.array(all_pred_returns)
    actual_arr = np.array(all_actual_returns)
    pred_close_arr = np.array(all_pred_closes)
    actual_close_arr = np.array(all_actual_closes)

    metrics = {
        "market": name,
        "timeframe": config["timeframe"],
        "n_windows": len(pred_arr),
        "ic": compute_ic(pred_arr, actual_arr),
        "rank_ic": compute_rank_ic(pred_arr, actual_arr),
        "directional_accuracy": compute_directional_accuracy(pred_arr, actual_arr),
        "rmse": compute_rmse(pred_close_arr, actual_close_arr),
        "mae": compute_mae(pred_close_arr, actual_close_arr),
        "mean_pred_return": float(np.mean(pred_arr)),
        "mean_actual_return": float(np.mean(actual_arr)),
    }

    print(f"\n  Results:")
    print(f"    IC:                   {metrics['ic']:+.4f}")
    print(f"    Rank IC:              {metrics['rank_ic']:+.4f}")
    print(f"    Directional Accuracy: {metrics['directional_accuracy']:.1%}")
    print(f"    RMSE:                 {metrics['rmse']:.4f}")
    print(f"    MAE:                  {metrics['mae']:.4f}")

    # Save visualization
    if sample_predictions:
        plot_samples(market_key, name, sample_predictions)

    return metrics


def plot_samples(market_key: str, name: str, samples: list):
    """Plot sample predictions vs actual for visual inspection."""
    fig, axes = plt.subplots(1, len(samples), figsize=(6 * len(samples), 4))
    if len(samples) == 1:
        axes = [axes]

    for idx, (ax, sample) in enumerate(zip(axes, samples)):
        ctx_len = len(sample["context_close"])
        pred_len = len(sample["pred_close"])

        # Plot last 100 context candles + prediction
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
        ax.set_title(f"{name} — Sample {idx+1}", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = RESULTS_DIR / f"{market_key}_zero_shot_samples.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot → {out_path}")


# ─── Main ───

def main():
    import torch

    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Device: {device}")
    print(f"Loading Kronos-base (102M params)...")

    # Load pre-trained model (no finetuning)
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    model = model.to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {param_count/1e6:.1f}M params on {device}")

    predictor = KronosPredictor(model, tokenizer, max_context=512)

    # Evaluate all markets
    all_metrics = []
    for market_key, config in MARKETS.items():
        metrics = evaluate_market(predictor, market_key, config)
        if metrics:
            all_metrics.append(metrics)

    # Summary table
    if all_metrics:
        print(f"\n{'='*70}")
        print("ZERO-SHOT BASELINE SUMMARY")
        print(f"{'='*70}")
        print(f"{'Market':<12} {'IC':>8} {'RankIC':>8} {'Dir.Acc':>8} {'RMSE':>10} {'MAE':>10}")
        print("-" * 70)
        for m in all_metrics:
            print(f"{m['market']:<12} {m['ic']:>+8.4f} {m['rank_ic']:>+8.4f} "
                  f"{m['directional_accuracy']:>7.1%} {m['rmse']:>10.4f} {m['mae']:>10.4f}")
        print(f"{'='*70}")

        # Save results
        results_df = pd.DataFrame(all_metrics)
        out_path = RESULTS_DIR / "zero_shot_metrics.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(out_path, index=False)
        print(f"\nSaved metrics → {out_path}")


if __name__ == "__main__":
    main()
