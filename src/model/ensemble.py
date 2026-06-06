"""
Ensemble predictor — combines zero-shot and finetuned Kronos models.

In quant, ensembling uncorrelated signals is the single highest-impact
technique for improving Sharpe ratio (Grinold's Law: IR = IC * sqrt(Breadth)).
Two models with IC=0.05 each but low correlation produce a combined IC >> 0.05.
"""

import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass, field

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Kronos"))


@dataclass
class EnsembleConfig:
    """Configuration for ensemble prediction."""
    n_paths: int = 50                    # Monte Carlo paths per model
    temperature: float = 1.0
    top_p: float = 0.9
    weight_method: str = "equal"         # 'equal', 'inverse_loss', 'learned'
    models: List[str] = field(default_factory=lambda: ["zero_shot", "finetuned"])


class EnsemblePredictor:
    """
    Combines predictions from multiple Kronos models.

    Strategies:
    - Equal weight: simple average (surprisingly hard to beat)
    - Inverse val loss: weight by 1/val_loss (better model gets more weight)
    - Stacked: use calibration set to learn optimal weights
    """

    def __init__(self, predictors: list, weights: Optional[list] = None, names: Optional[list] = None):
        """
        Args:
            predictors: List of KronosPredictor instances
            weights: Optional weights for each predictor (default: equal)
            names: Optional names for logging
        """
        self.predictors = predictors
        self.names = names or [f"model_{i}" for i in range(len(predictors))]

        if weights is None:
            self.weights = [1.0 / len(predictors)] * len(predictors)
        else:
            total = sum(weights)
            self.weights = [w / total for w in weights]

    @classmethod
    def from_market(cls, market: str, device: str = "auto",
                    checkpoints_dir: Optional[Path] = None,
                    use_zero_shot: bool = True,
                    use_finetuned: bool = True,
                    val_losses: Optional[dict] = None):
        """
        Build ensemble for a market from available models.

        Args:
            market: 'btc', 'eur', or 'xau'
            device: torch device
            checkpoints_dir: where finetuned checkpoints live
            use_zero_shot: include zero-shot base model
            use_finetuned: include finetuned model
            val_losses: dict of {'zero_shot': loss, 'finetuned': loss} for weighting
        """
        from model import Kronos, KronosTokenizer, KronosPredictor

        if checkpoints_dir is None:
            checkpoints_dir = PROJECT_ROOT / "checkpoints"

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        predictors = []
        names = []
        weights = []

        # Zero-shot model
        if use_zero_shot:
            print(f"Loading zero-shot Kronos-base...")
            tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
            model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
            model = model.to(device)
            model.eval()
            predictors.append(KronosPredictor(model, tokenizer, max_context=512))
            names.append("zero_shot")
            weights.append(1.0)

        # Finetuned model
        if use_finetuned:
            exp_names = {"btc": "cifr-btc", "eur": "cifr-eur", "xau": "cifr-xau"}
            exp_name = exp_names.get(market, f"cifr-{market}")
            tok_path = checkpoints_dir / exp_name / "tokenizer" / "best_model"
            mdl_path = checkpoints_dir / exp_name / "predictor" / "best_model"

            if tok_path.exists() and mdl_path.exists():
                print(f"Loading finetuned {exp_name}...")
                tokenizer = KronosTokenizer.from_pretrained(str(tok_path))
                model = Kronos.from_pretrained(str(mdl_path))
                model = model.to(device)
                model.eval()
                predictors.append(KronosPredictor(model, tokenizer, max_context=512))
                names.append("finetuned")
                weights.append(1.0)
            else:
                print(f"No finetuned checkpoint for {market}, using zero-shot only")

        if not predictors:
            raise RuntimeError("No models loaded")

        # Weight by inverse val loss if provided
        if val_losses and len(predictors) > 1:
            for i, name in enumerate(names):
                if name in val_losses:
                    weights[i] = 1.0 / val_losses[name]

        return cls(predictors, weights, names)

    def predict_single(self, df, x_timestamp, y_timestamp, pred_len,
                       T=1.0, top_p=0.9, sample_count=1):
        """
        Single prediction from ensemble — weighted average of all models.

        Returns averaged DataFrame (same format as KronosPredictor.predict).
        """
        all_preds = []
        for pred, w, name in zip(self.predictors, self.weights, self.names):
            try:
                result = pred.predict(
                    df=df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
                    pred_len=pred_len, T=T, top_p=top_p, sample_count=sample_count,
                )
                all_preds.append((result, w))
            except Exception as e:
                print(f"  {name} prediction failed: {e}")
                continue

        if not all_preds:
            raise RuntimeError("All ensemble models failed")

        # Weighted average
        total_weight = sum(w for _, w in all_preds)
        combined = all_preds[0][0].copy()
        numeric_cols = combined.select_dtypes(include=[np.number]).columns

        combined[numeric_cols] = 0.0
        for result, w in all_preds:
            combined[numeric_cols] += result[numeric_cols] * (w / total_weight)

        return combined

    def sample_paths(self, df, x_timestamp, y_timestamp, pred_len,
                     n_paths=50, T=1.0, top_p=0.9):
        """
        Generate Monte Carlo paths from all ensemble members.

        Each model contributes paths proportional to its weight.
        """
        all_paths = []

        for pred, w, name in zip(self.predictors, self.weights, self.names):
            # Allocate paths proportional to weight
            model_paths = max(1, int(n_paths * w / sum(self.weights)))

            for _ in range(model_paths):
                try:
                    result = pred.predict(
                        df=df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
                        pred_len=pred_len, T=T, top_p=top_p, sample_count=1,
                    )
                    all_paths.append(result)
                except Exception:
                    continue

        if not all_paths:
            raise RuntimeError("No paths generated from ensemble")

        return all_paths

    def predict_with_quantiles(self, df, x_timestamp, y_timestamp, pred_len,
                                n_paths=50, T=1.0, top_p=0.9,
                                quantiles=(0.05, 0.25, 0.50, 0.75, 0.95)):
        """
        Full probabilistic forecast with quantile bands from ensemble paths.

        Returns:
            dict with 'median', 'mean', and quantile DataFrames,
            plus 'paths' list and 'directional_confidence' float.
        """
        paths = self.sample_paths(df, x_timestamp, y_timestamp, pred_len,
                                  n_paths=n_paths, T=T, top_p=top_p)

        # Stack close prices
        closes = np.array([p["close"].values for p in paths])
        highs = np.array([p["high"].values for p in paths])
        lows = np.array([p["low"].values for p in paths])

        # Entry price (last known close)
        entry = df["close"].iloc[-1]

        # Directional confidence
        final_closes = closes[:, -1]
        up_count = np.sum(final_closes > entry)
        directional_confidence = max(up_count, len(final_closes) - up_count) / len(final_closes)
        direction = "long" if up_count > len(final_closes) / 2 else "short"

        # Quantile bands
        result = {
            "entry_price": entry,
            "direction": direction,
            "directional_confidence": directional_confidence,
            "n_paths": len(paths),
            "n_models": len(self.predictors),
            "model_names": self.names,
            "model_weights": self.weights,
        }

        for q in quantiles:
            result[f"close_q{int(q*100):02d}"] = np.quantile(closes, q, axis=0)
            result[f"high_q{int(q*100):02d}"] = np.quantile(highs, q, axis=0)
            result[f"low_q{int(q*100):02d}"] = np.quantile(lows, q, axis=0)

        result["close_mean"] = np.mean(closes, axis=0)
        result["close_median"] = np.median(closes, axis=0)

        # Stop loss from low quantile, take profit from high quantile
        result["stop_loss"] = np.min(result["low_q05"])       # Worst case low
        result["take_profit"] = np.max(result["high_q95"])    # Best case high

        return result

    def __repr__(self):
        parts = [f"{n}(w={w:.2f})" for n, w in zip(self.names, self.weights)]
        return f"EnsemblePredictor([{', '.join(parts)}])"
