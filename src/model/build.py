"""Shared ensemble builder.

Both CQR calibration (`scripts/calibrate_cqr_multi.py`) and the portfolio
backtest MUST use the identical ensemble, otherwise the per-asset CQR
corrections won't match the model that produced the quantile bands. This
module is the single source of truth for which models go into each market's
ensemble:

    crypto    : zero-shot + cifr-btc  + cifr-btc-v2
    commodity : zero-shot + cifr-xau  + cifr-xau-v2
"""

import sys
import torch
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Kronos"))

# Finetuned checkpoints per market, in ensemble order.
MARKET_FINETUNES = {
    "crypto": ["cifr-btc", "cifr-btc-v2"],
    "commodity": ["cifr-xau", "cifr-xau-v2"],
}


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_market_ensemble(market: str, device: str = "auto",
                          checkpoints_dir: Optional[Path] = None):
    """Build the zero-shot + finetuned ensemble for a market.

    Mirrors the ensemble used during CQR calibration exactly.
    """
    from model import Kronos, KronosTokenizer, KronosPredictor
    from src.model.ensemble import EnsemblePredictor

    if checkpoints_dir is None:
        checkpoints_dir = PROJECT_ROOT / "checkpoints"
    device = resolve_device(device)

    predictors, names, weights = [], [], []

    # Zero-shot (shared base)
    print("  Loading zero-shot Kronos-base...", flush=True)
    zs_tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    zs_mdl = Kronos.from_pretrained("NeoQuasar/Kronos-base").to(device).eval()
    predictors.append(KronosPredictor(zs_mdl, zs_tok, max_context=512))
    names.append("zero_shot")
    weights.append(1.0)

    for exp in MARKET_FINETUNES.get(market, []):
        tok_path = checkpoints_dir / exp / "tokenizer" / "best_model"
        mdl_path = checkpoints_dir / exp / "predictor" / "best_model"
        if tok_path.exists() and mdl_path.exists():
            print(f"  Loading {exp}...", flush=True)
            ft_tok = KronosTokenizer.from_pretrained(str(tok_path))
            ft_mdl = Kronos.from_pretrained(str(mdl_path)).to(device).eval()
            predictors.append(KronosPredictor(ft_mdl, ft_tok, max_context=512))
            names.append(exp)
            weights.append(1.0)
        else:
            print(f"  WARNING: {exp} checkpoint not found, skipping", flush=True)

    ensemble = EnsemblePredictor(predictors, weights, names)
    print(f"  {market} ensemble: {ensemble}", flush=True)
    return ensemble
