"""Batched Monte-Carlo path generation for Kronos.

The vendored ``auto_regressive_inference`` in Kronos already expands a single
series into ``sample_count`` independent stochastic rollouts and runs them as
ONE batched forward pass — but it then *averages* them away
(``preds = np.mean(preds, axis=1)``). Our CQR/ensemble code needs the
individual paths to form quantile bands, so the existing ensemble worked
around this by calling ``predict(sample_count=1)`` in a Python loop — i.e.
N separate forward passes, one path each. On a Lovelace L40S that runs the
GPU at batch-size-1 while it could comfortably run batch-size-N.

This module replicates the autoregressive loop verbatim (so results are
statistically identical) but returns the un-averaged paths, letting us draw
all N Monte-Carlo paths for a window in a single batched forward pass.

We deliberately keep this in the tracked ``src/`` tree rather than editing the
vendored ``Kronos/`` clone (which is a gitlink and would not propagate to the
HPC via the parent repo).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from model.kronos import sample_from_logits, calc_time_stamps


@torch.no_grad()
def _auto_regressive_inference_paths(
    tokenizer,
    model,
    x,
    x_stamp,
    y_stamp,
    max_context,
    pred_len,
    clip=5,
    T=1.0,
    top_k=0,
    top_p=0.9,
    sample_count=1,
):
    """Identical to Kronos ``auto_regressive_inference`` but WITHOUT the final
    mean over ``sample_count``.

    Returns:
        np.ndarray of shape (B, sample_count, total_seq_len, feat).
    """
    x = torch.clip(x, -clip, clip)
    device = x.device

    # Expand each series into `sample_count` independent rollouts (batched).
    x = x.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x.size(1), x.size(2)).to(device)
    x_stamp = x_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x_stamp.size(1), x_stamp.size(2)).to(device)
    y_stamp = y_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, y_stamp.size(1), y_stamp.size(2)).to(device)

    x_token = tokenizer.encode(x, half=True)

    initial_seq_len = x.size(1)
    batch_size = x_token[0].size(0)
    total_seq_len = initial_seq_len + pred_len
    full_stamp = torch.cat([x_stamp, y_stamp], dim=1)

    generated_pre = x_token[0].new_empty(batch_size, pred_len)
    generated_post = x_token[1].new_empty(batch_size, pred_len)

    pre_buffer = x_token[0].new_zeros(batch_size, max_context)
    post_buffer = x_token[1].new_zeros(batch_size, max_context)
    buffer_len = min(initial_seq_len, max_context)
    if buffer_len > 0:
        start_idx = max(0, initial_seq_len - max_context)
        pre_buffer[:, :buffer_len] = x_token[0][:, start_idx:start_idx + buffer_len]
        post_buffer[:, :buffer_len] = x_token[1][:, start_idx:start_idx + buffer_len]

    for i in range(pred_len):
        current_seq_len = initial_seq_len + i
        window_len = min(current_seq_len, max_context)

        if current_seq_len <= max_context:
            input_tokens = [pre_buffer[:, :window_len], post_buffer[:, :window_len]]
        else:
            input_tokens = [pre_buffer, post_buffer]

        context_end = current_seq_len
        context_start = max(0, context_end - max_context)
        current_stamp = full_stamp[:, context_start:context_end, :].contiguous()

        s1_logits, context = model.decode_s1(input_tokens[0], input_tokens[1], current_stamp)
        s1_logits = s1_logits[:, -1, :]
        sample_pre = sample_from_logits(s1_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)

        s2_logits = model.decode_s2(context, sample_pre)
        s2_logits = s2_logits[:, -1, :]
        sample_post = sample_from_logits(s2_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)

        generated_pre[:, i] = sample_pre.squeeze(-1)
        generated_post[:, i] = sample_post.squeeze(-1)

        if current_seq_len < max_context:
            pre_buffer[:, current_seq_len] = sample_pre.squeeze(-1)
            post_buffer[:, current_seq_len] = sample_post.squeeze(-1)
        else:
            pre_buffer.copy_(torch.roll(pre_buffer, shifts=-1, dims=1))
            post_buffer.copy_(torch.roll(post_buffer, shifts=-1, dims=1))
            pre_buffer[:, -1] = sample_pre.squeeze(-1)
            post_buffer[:, -1] = sample_post.squeeze(-1)

    full_pre = torch.cat([x_token[0], generated_pre], dim=1)
    full_post = torch.cat([x_token[1], generated_post], dim=1)

    context_start = max(0, total_seq_len - max_context)
    input_tokens = [
        full_pre[:, context_start:total_seq_len].contiguous(),
        full_post[:, context_start:total_seq_len].contiguous(),
    ]
    z = tokenizer.decode(input_tokens, half=True)
    z = z.reshape(-1, sample_count, z.size(1), z.size(2))  # (B, sample_count, seq, feat)
    return z.cpu().numpy()


def predict_paths(
    predictor,
    df: pd.DataFrame,
    x_timestamp,
    y_timestamp,
    pred_len: int,
    n_paths: int,
    T: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.9,
):
    """Draw ``n_paths`` independent Monte-Carlo paths in a single batched
    forward pass for one ``KronosPredictor``.

    Mirrors ``KronosPredictor.predict`` normalization/denormalization exactly,
    so each returned path is distributionally identical to a single
    ``predict(sample_count=1)`` call — just generated in parallel.

    Returns:
        list[pd.DataFrame] of length ``n_paths``, each with columns
        [open, high, low, close, volume, amount] indexed by ``y_timestamp``.
    """
    price_cols = predictor.price_cols
    vol_col = predictor.vol_col
    amt_vol = predictor.amt_vol
    clip = predictor.clip
    device = predictor.device

    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if not all(col in df.columns for col in price_cols):
        raise ValueError(f"Price columns {price_cols} not found in DataFrame.")

    df = df.copy()
    if vol_col not in df.columns:
        df[vol_col] = 0.0
        df[amt_vol] = 0.0
    if amt_vol not in df.columns and vol_col in df.columns:
        df[amt_vol] = df[vol_col] * df[price_cols].mean(axis=1)

    if df[price_cols + [vol_col, amt_vol]].isnull().values.any():
        raise ValueError("Input DataFrame contains NaN values in price or volume columns.")

    x_time_df = calc_time_stamps(x_timestamp)
    y_time_df = calc_time_stamps(y_timestamp)

    x = df[price_cols + [vol_col, amt_vol]].values.astype(np.float32)
    x_stamp = x_time_df.values.astype(np.float32)
    y_stamp = y_time_df.values.astype(np.float32)

    x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
    x = (x - x_mean) / (x_std + 1e-5)
    x = np.clip(x, -clip, clip)

    x = x[np.newaxis, :]
    x_stamp = x_stamp[np.newaxis, :]
    y_stamp = y_stamp[np.newaxis, :]

    x_tensor = torch.from_numpy(x.astype(np.float32)).to(device)
    x_stamp_tensor = torch.from_numpy(x_stamp.astype(np.float32)).to(device)
    y_stamp_tensor = torch.from_numpy(y_stamp.astype(np.float32)).to(device)

    preds = _auto_regressive_inference_paths(
        predictor.tokenizer, predictor.model,
        x_tensor, x_stamp_tensor, y_stamp_tensor,
        predictor.max_context, pred_len,
        clip, T, top_k, top_p, sample_count=n_paths,
    )

    preds = preds[0]                      # (n_paths, total_seq_len, feat)
    preds = preds[:, -pred_len:, :]       # (n_paths, pred_len, feat)
    preds = preds * (x_std + 1e-5) + x_mean

    cols = price_cols + [vol_col, amt_vol]
    return [pd.DataFrame(preds[p], columns=cols, index=y_timestamp) for p in range(preds.shape[0])]
