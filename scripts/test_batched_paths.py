"""Local sanity check for batched Monte-Carlo path generation.

Verifies that `src.model.batched_inference.predict_paths` (all paths in one
batched forward pass) is:
  1. shape/format-correct (list of n_paths DataFrames, right columns/index),
  2. distributionally equivalent to the original per-path
     `predict(sample_count=1)` loop (same mean/std of final-step close to
     within Monte-Carlo noise), and
  3. faster on GPU.

Uses ONLY the zero-shot Kronos-base model, so it runs without the HPC-only
finetuned checkpoints. Intended for the laptop `mlenv` GPU as a small check
before committing/resubmitting.

Run:
    python scripts/test_batched_paths.py
"""

import sys
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Kronos"))


def make_synthetic_window(n_ctx=256, freq="15min", seed=0):
    """Build a plausible synthetic OHLCV context window."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.002, n_ctx)
    close = 100.0 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.001, n_ctx)))
    low = close * (1 - np.abs(rng.normal(0, 0.001, n_ctx)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    vol = np.abs(rng.normal(1000, 100, n_ctx))
    amount = vol * close
    ts = pd.date_range("2025-01-01", periods=n_ctx, freq=freq)
    df = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": vol, "amount": amount, "timestamps": ts,
    })
    return df


def main():
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"Device: {device}")

    from model import Kronos, KronosTokenizer, KronosPredictor
    from src.model.batched_inference import predict_paths

    print("Loading zero-shot Kronos-base...")
    tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    mdl = Kronos.from_pretrained("NeoQuasar/Kronos-base").to(device).eval()
    predictor = KronosPredictor(mdl, tok, max_context=512)

    pred_len = 6
    n_paths = 64
    df = make_synthetic_window(n_ctx=256)

    ctx = df.iloc[:-pred_len]
    fut = df.iloc[-pred_len:]
    x_df = ctx[["open", "high", "low", "close", "volume", "amount"]]
    x_ts = ctx["timestamps"]
    y_ts = fut["timestamps"]

    # --- Batched ---
    torch.manual_seed(123)
    t0 = time.time()
    batched = predict_paths(predictor, x_df, x_ts, y_ts, pred_len=pred_len, n_paths=n_paths)
    if device == "cuda":
        torch.cuda.synchronize()
    t_batched = time.time() - t0

    # --- Looped (original behaviour) ---
    torch.manual_seed(123)
    t0 = time.time()
    looped = []
    for _ in range(n_paths):
        looped.append(predictor.predict(
            df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
            pred_len=pred_len, sample_count=1, verbose=False))
    if device == "cuda":
        torch.cuda.synchronize()
    t_looped = time.time() - t0

    # --- Correctness checks ---
    print("\n=== Format checks ===")
    assert isinstance(batched, list), "predict_paths must return a list"
    assert len(batched) == n_paths, f"expected {n_paths} paths, got {len(batched)}"
    cols = ["open", "high", "low", "close", "volume", "amount"]
    assert list(batched[0].columns) == cols, f"bad columns: {list(batched[0].columns)}"
    assert len(batched[0]) == pred_len, f"bad pred_len: {len(batched[0])}"
    assert list(batched[0].index) == list(y_ts), "index must equal y_timestamp"
    print(f"  OK: {len(batched)} paths, shape {batched[0].shape}, cols {cols}")

    b_final = np.array([p["close"].iloc[-1] for p in batched])
    l_final = np.array([p["close"].iloc[-1] for p in looped])

    print("\n=== Distributional equivalence (final-step close) ===")
    print(f"  batched: mean={b_final.mean():.4f} std={b_final.std():.4f} "
          f"[{b_final.min():.3f}, {b_final.max():.3f}]")
    print(f"  looped : mean={l_final.mean():.4f} std={l_final.std():.4f} "
          f"[{l_final.min():.3f}, {l_final.max():.3f}]")
    mean_diff = abs(b_final.mean() - l_final.mean())
    std_diff = abs(b_final.std() - l_final.std())
    pooled_std = max(np.std(np.concatenate([b_final, l_final])), 1e-6)
    print(f"  |mean diff|={mean_diff:.4f}  |std diff|={std_diff:.4f}  "
          f"(pooled std={pooled_std:.4f})")
    # Means should agree within a few standard errors given MC noise.
    se = pooled_std / np.sqrt(n_paths)
    ok_mean = mean_diff < 5 * se
    ok_std = std_diff < 0.5 * pooled_std
    print(f"  mean within 5*SE ({5*se:.4f}): {'OK' if ok_mean else 'CHECK'}")
    print(f"  std within 50% of pooled:      {'OK' if ok_std else 'CHECK'}")

    print("\n=== Speed ===")
    print(f"  looped : {t_looped:.2f}s ({t_looped/n_paths*1000:.1f} ms/path)")
    print(f"  batched: {t_batched:.2f}s ({t_batched/n_paths*1000:.1f} ms/path)")
    if t_batched > 0:
        print(f"  speedup: {t_looped/t_batched:.1f}x")

    print("\nDONE." + ("" if (ok_mean and ok_std) else "  (review CHECK items above)"))


if __name__ == "__main__":
    main()
