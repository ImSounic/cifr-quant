"""Base configuration shared across all market experiments."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Paths
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"


@dataclass
class KronosConfig:
    """Kronos model configuration."""
    model_name: str = "NeoQuasar/Kronos-base"
    tokenizer_name: str = "NeoQuasar/Kronos-Tokenizer-base"
    max_context: int = 512
    device: str = "cuda"


@dataclass
class SamplingConfig:
    """Monte Carlo sampling configuration for probabilistic forecasting."""
    n_paths: int = 30
    temperature: float = 1.0
    top_p: float = 0.9


@dataclass
class QuantileConfig:
    """Quantile risk management configuration."""
    stop_loss_quantile: float = 0.05     # 5th percentile
    take_profit_quantile: float = 0.95   # 95th percentile
    confidence_band_lo: float = 0.25     # 25th percentile
    confidence_band_hi: float = 0.75     # 75th percentile
    cqr_coverage: float = 0.90           # 90% coverage guarantee
    min_directional_confidence: float = 0.60  # 60% of paths must agree


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    initial_capital: float = 100_000.0
    max_position_pct: float = 0.10       # Max 10% of capital per trade
    max_drawdown_halt: float = 0.15      # Halt if drawdown > 15%


@dataclass
class MarketConfig:
    """Per-market configuration. Subclassed per instrument."""
    instrument: str = ""
    timeframe: str = ""
    candles_per_year: int = 0
    pred_len: int = 0
    data_years: int = 2

    # Cost model
    spread_pct: float = 0.0
    slippage_pct: float = 0.0
    commission_pct: float = 0.0

    # Data
    has_volume: bool = True
    data_source: str = ""

    @property
    def total_cost_pct(self) -> float:
        return self.spread_pct + self.slippage_pct + self.commission_pct

    # Splits
    test_months: int = 3
    val_months: int = 3
    cal_months: int = 3  # CQR calibration
