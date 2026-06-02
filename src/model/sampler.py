"""Monte Carlo multi-path sampling for probabilistic forecasting.

CRITICAL DESIGN NOTE:
Kronos's predict(sample_count=N) generates N paths and AVERAGES them
into a single DataFrame. To get individual trajectories for quantile
analysis, we call predict(sample_count=1) N times.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm

from src.model.predictor import CifrPredictor


@dataclass
class ForecastPaths:
    """Container for multiple sampled forecast trajectories."""
    paths: list[pd.DataFrame]     # N individual forecast DataFrames
    timestamps: pd.Series         # Shared prediction timestamps
    n_paths: int = 0

    def __post_init__(self):
        self.n_paths = len(self.paths)

    @property
    def closes(self) -> np.ndarray:
        """Extract close prices as (n_paths, pred_len) array."""
        return np.array([p["close"].values for p in self.paths])

    @property
    def highs(self) -> np.ndarray:
        """Extract high prices as (n_paths, pred_len) array."""
        return np.array([p["high"].values for p in self.paths])

    @property
    def lows(self) -> np.ndarray:
        """Extract low prices as (n_paths, pred_len) array."""
        return np.array([p["low"].values for p in self.paths])

    @property
    def opens(self) -> np.ndarray:
        """Extract open prices as (n_paths, pred_len) array."""
        return np.array([p["open"].values for p in self.paths])

    def mean_forecast(self) -> pd.DataFrame:
        """Average across all paths."""
        combined = pd.concat(self.paths)
        return combined.groupby(combined.index).mean()

    def quantile_at(self, q: float, field: str = "close") -> np.ndarray:
        """Get quantile across paths for a given field."""
        data = np.array([p[field].values for p in self.paths])
        return np.quantile(data, q, axis=0)

    def directional_confidence(self, current_price: float) -> float:
        """Fraction of paths showing positive return at final timestep."""
        final_closes = self.closes[:, -1]
        return float(np.mean(final_closes > current_price))

    def path_dispersion(self) -> float:
        """Variance of path endpoints (proxy for regime uncertainty)."""
        final_closes = self.closes[:, -1]
        return float(np.var(final_closes))

    def trend_strength(self) -> float:
        """
        Consistency of directional movement across paths.
        1.0 = all paths trend same direction, 0.0 = random.
        """
        # Compute per-path slope (simple linear regression)
        slopes = []
        x = np.arange(self.closes.shape[1])
        for i in range(self.n_paths):
            y = self.closes[i]
            slope = np.polyfit(x, y, 1)[0]
            slopes.append(slope)

        slopes = np.array(slopes)
        # Fraction of slopes that agree with majority direction
        pos_frac = np.mean(slopes > 0)
        return float(max(pos_frac, 1 - pos_frac))


def sample_paths(
    predictor: CifrPredictor,
    df: pd.DataFrame,
    x_timestamp: pd.Series,
    y_timestamp: pd.Series,
    pred_len: int,
    n_paths: int = 30,
    temperature: float = 1.0,
    top_p: float = 0.9,
    verbose: bool = True,
) -> ForecastPaths:
    """
    Generate N independent forecast paths via repeated sampling.

    Each call to predict(sample_count=1) with T > 0 produces a
    stochastically different trajectory due to temperature-controlled
    autoregressive sampling.

    Args:
        predictor: CifrPredictor instance
        df: Historical OHLCV data
        x_timestamp: Historical timestamps
        y_timestamp: Future timestamps to predict
        pred_len: Number of candles to predict
        n_paths: Number of Monte Carlo paths (default 30)
        temperature: Sampling temperature (higher = more diverse paths)
        top_p: Nucleus sampling threshold
        verbose: Show progress bar

    Returns:
        ForecastPaths container with N individual trajectories
    """
    paths = []
    iterator = range(n_paths)
    if verbose:
        iterator = tqdm(iterator, desc="Sampling paths")

    for _ in iterator:
        path_df = predictor.predict_single(
            df=df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            temperature=temperature,
            top_p=top_p,
        )
        paths.append(path_df)

    return ForecastPaths(paths=paths, timestamps=y_timestamp)
