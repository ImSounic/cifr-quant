"""Conformalized Quantile Regression (CQR) for calibrated prediction intervals.

CQR provides distribution-free, finite-sample coverage guarantees for
prediction intervals. It corrects model overconfidence by computing
conformity scores on a held-out calibration set.

Reference: Romano et al. (2019) "Conformalized Quantile Regression" NeurIPS.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class CQRCalibration:
    """Stores CQR calibration results."""
    correction: float           # Additive correction term
    coverage_target: float      # Desired coverage (e.g., 0.90)
    coverage_achieved: float    # Actual coverage on calibration set
    n_calibration: int          # Number of calibration samples
    conformity_scores: np.ndarray  # All computed scores

    def __repr__(self) -> str:
        return (
            f"CQRCalibration:\n"
            f"  Target coverage:   {self.coverage_target:.1%}\n"
            f"  Achieved coverage: {self.coverage_achieved:.1%}\n"
            f"  Correction term:   {self.correction:+.6f}\n"
            f"  Calibration size:  {self.n_calibration}"
        )


def calibrate(
    y_true: np.ndarray,
    q_lower: np.ndarray,
    q_upper: np.ndarray,
    coverage: float = 0.90,
) -> CQRCalibration:
    """
    Compute CQR calibration from a held-out calibration set.

    For each calibration sample, compute the conformity score:
        score_i = max(q_lower_i - y_i, y_i - q_upper_i)

    Positive scores indicate the true value fell outside the interval.
    The correction term is the ceil((1-alpha)(1 + 1/n))-th quantile
    of the conformity scores.

    Args:
        y_true: Actual observed values (n_cal,)
        q_lower: Predicted lower bounds (n_cal,) — e.g., 5th percentile
        q_upper: Predicted upper bounds (n_cal,) — e.g., 95th percentile
        coverage: Desired coverage level (default 0.90)

    Returns:
        CQRCalibration with the correction term
    """
    n = len(y_true)
    assert len(q_lower) == n and len(q_upper) == n, "All arrays must have same length"

    # Compute conformity scores
    scores = np.maximum(q_lower - y_true, y_true - q_upper)

    # Compute correction: quantile level with finite-sample correction
    alpha = 1 - coverage
    quantile_level = np.ceil((1 - alpha) * (n + 1)) / n
    quantile_level = min(quantile_level, 1.0)  # Clip to valid range

    correction = float(np.quantile(scores, quantile_level))

    # Check achieved coverage with this correction
    adjusted_lower = q_lower - correction
    adjusted_upper = q_upper + correction
    achieved = float(np.mean((y_true >= adjusted_lower) & (y_true <= adjusted_upper)))

    return CQRCalibration(
        correction=correction,
        coverage_target=coverage,
        coverage_achieved=achieved,
        n_calibration=n,
        conformity_scores=scores,
    )


def adjust_levels(
    stop_loss: float,
    take_profit: float,
    calibration: CQRCalibration,
) -> tuple[float, float]:
    """
    Adjust SL/TP levels using CQR correction.

    Widens the interval by the correction term to achieve
    the target coverage guarantee.

    Args:
        stop_loss: Raw stop loss level (lower bound)
        take_profit: Raw take profit level (upper bound)
        calibration: CQR calibration result

    Returns:
        Tuple of (adjusted_stop_loss, adjusted_take_profit)
    """
    # Widen the interval: lower bound goes down, upper bound goes up
    adjusted_sl = stop_loss - calibration.correction
    adjusted_tp = take_profit + calibration.correction

    return adjusted_sl, adjusted_tp


def build_calibration_dataset(
    predictor,      # CifrPredictor
    cal_df,         # Calibration DataFrame with timestamps, OHLCV
    lookback: int,  # Context window size (512)
    pred_len: int,  # Prediction horizon
    n_paths: int,   # Monte Carlo paths per prediction
    sl_quantile: float = 0.05,
    tp_quantile: float = 0.95,
    step_size: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build arrays for CQR calibration by rolling through calibration data.

    At each step:
    1. Take lookback candles as context
    2. Generate n_paths forecast trajectories
    3. Compute quantile bounds
    4. Record actual outcome

    Args:
        predictor: CifrPredictor instance
        cal_df: Calibration period data
        lookback: Number of historical candles (512)
        pred_len: Prediction horizon
        n_paths: Paths per prediction
        sl_quantile: Lower quantile
        tp_quantile: Upper quantile
        step_size: Roll forward by this many candles between predictions

    Returns:
        Tuple of (y_true, q_lower, q_upper) arrays
    """
    from src.model.sampler import sample_paths

    y_true_list = []
    q_lower_list = []
    q_upper_list = []

    total_steps = (len(cal_df) - lookback - pred_len) // step_size

    for i in range(0, len(cal_df) - lookback - pred_len, step_size):
        # Context window
        context = cal_df.iloc[i:i + lookback]
        x_ts = context["timestamps"]

        # Future window (ground truth)
        future = cal_df.iloc[i + lookback:i + lookback + pred_len]
        y_ts = future["timestamps"]

        if len(future) < pred_len:
            break

        # Generate paths
        x_df = context[["open", "high", "low", "close", "volume", "amount"]]
        paths = sample_paths(
            predictor, x_df, x_ts, y_ts,
            pred_len=pred_len, n_paths=n_paths,
            verbose=False,
        )

        # Extract quantiles at final timestep
        final_lows = paths.lows[:, -1]
        final_highs = paths.highs[:, -1]

        q_lo = np.quantile(final_lows, sl_quantile)
        q_hi = np.quantile(final_highs, tp_quantile)
        y_actual = future["close"].iloc[-1]

        y_true_list.append(y_actual)
        q_lower_list.append(q_lo)
        q_upper_list.append(q_hi)

    return (
        np.array(y_true_list),
        np.array(q_lower_list),
        np.array(q_upper_list),
    )
