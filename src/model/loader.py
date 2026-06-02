"""Load Kronos models and tokenizers from HuggingFace or local checkpoints."""

import sys
from pathlib import Path
from typing import Optional
import torch


def setup_kronos_path():
    """Add Kronos repo to Python path if installed locally."""
    # Try importing directly first (pip installed)
    try:
        from model import Kronos, KronosTokenizer, KronosPredictor
        return
    except ImportError:
        pass

    # Look for local clone
    candidates = [
        Path(__file__).parent.parent.parent / "kronos",  # cifr-quant/kronos/
        Path(__file__).parent.parent.parent / "Kronos",
        Path.home() / "Kronos",
    ]
    for p in candidates:
        if (p / "model").exists():
            sys.path.insert(0, str(p))
            return

    raise ImportError(
        "Kronos not found. Either:\n"
        "  1. pip install kronos-model-arch\n"
        "  2. git clone https://github.com/shiyu-coder/Kronos.git into project root\n"
        "  3. Set KRONOS_PATH environment variable"
    )


def load_model(
    model_name: str = "NeoQuasar/Kronos-base",
    tokenizer_name: str = "NeoQuasar/Kronos-Tokenizer-base",
    device: str = "auto",
    local_model_path: Optional[str] = None,
    local_tokenizer_path: Optional[str] = None,
):
    """
    Load Kronos model and tokenizer.

    Args:
        model_name: HuggingFace model name or local path
        tokenizer_name: HuggingFace tokenizer name or local path
        device: Device to load model on ('auto', 'cuda', 'cpu', 'mps')
        local_model_path: Override with local finetuned checkpoint path
        local_tokenizer_path: Override with local finetuned tokenizer path

    Returns:
        Tuple of (model, tokenizer, device_str)
    """
    setup_kronos_path()
    from model import Kronos, KronosTokenizer

    # Resolve device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Load tokenizer
    tok_path = local_tokenizer_path or tokenizer_name
    print(f"Loading tokenizer from {tok_path}...")
    tokenizer = KronosTokenizer.from_pretrained(tok_path)

    # Load model
    mdl_path = local_model_path or model_name
    print(f"Loading model from {mdl_path}...")
    model = Kronos.from_pretrained(mdl_path)
    model = model.to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {param_count/1e6:.1f}M params on {device}")

    return model, tokenizer, device


def load_finetuned(market: str, checkpoints_dir: Optional[Path] = None):
    """
    Load a finetuned checkpoint for a specific market.

    Args:
        market: One of 'btc', 'eur', 'xau'
        checkpoints_dir: Base checkpoints directory

    Returns:
        Tuple of (model, tokenizer, device)
    """
    if checkpoints_dir is None:
        from configs.base_config import CHECKPOINTS_DIR
        checkpoints_dir = CHECKPOINTS_DIR

    # Kronos train_sequential.py saves to: {base_path}/{exp_name}/{component}/best_model/
    exp_names = {"btc": "cifr-btc", "eur": "cifr-eur", "xau": "cifr-xau"}
    exp_name = exp_names.get(market, f"cifr-{market}")
    ckpt_dir = checkpoints_dir / exp_name

    if not ckpt_dir.exists():
        raise FileNotFoundError(
            f"No finetuned checkpoint found at {ckpt_dir}. "
            f"Run finetuning first or use load_model() for zero-shot."
        )

    # Kronos saves best checkpoints to {component}/best_model/
    tok_path = ckpt_dir / "tokenizer" / "best_model"
    mdl_path = ckpt_dir / "predictor" / "best_model"

    if not tok_path.exists():
        tok_path = None  # Fall back to base tokenizer
    if not mdl_path.exists():
        mdl_path = None  # Fall back to base model

    return load_model(
        local_model_path=str(mdl_path) if mdl_path else "NeoQuasar/Kronos-base",
        local_tokenizer_path=str(tok_path) if tok_path else "NeoQuasar/Kronos-Tokenizer-base",
    )
