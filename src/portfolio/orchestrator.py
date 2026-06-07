"""Portfolio orchestrator — multi-asset prediction and allocation.

This is the core engine for the dual-market multi-asset strategy.
It coordinates Kronos predictions across all assets in both markets,
applies CQR-calibrated uncertainty, and outputs position sizing.

Architecture:
    For each market, we have:
    - One finetuned checkpoint (BTC for crypto, XAU for commodities)
    - One zero-shot Kronos-base model
    - Ensemble combines both for every asset in the universe

    The key insight: finetuning on BTC captures crypto-specific patterns
    (momentum cascades, leverage liquidations, 24/7 microstructure) that
    transfer to all crypto assets. Similarly, XAU finetuning captures
    commodity macro dynamics (real rates, central bank flows, risk-off).
"""

import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Kronos"))


@dataclass
class AssetSignal:
    """Trading signal for a single asset."""
    symbol: str
    market: str                    # 'crypto' or 'commodity'
    direction: str                 # 'long', 'short', or 'flat'
    directional_confidence: float  # 0.5 to 1.0
    predicted_return: float        # Expected return over pred_len
    stop_loss: float               # CQR-adjusted stop price
    take_profit: float             # CQR-adjusted take price
    entry_price: float             # Current price
    interval_width: float          # Uncertainty band width (for position sizing)
    n_paths: int                   # Monte Carlo paths used
    ic_estimate: float = 0.0       # Historical IC for this asset (if available)


@dataclass
class PortfolioAllocation:
    """Portfolio-level allocation across all assets."""
    signals: List[AssetSignal]
    timestamp: pd.Timestamp
    total_capital: float
    positions: Dict[str, float] = field(default_factory=dict)  # symbol → position_size_usd
    weights: Dict[str, float] = field(default_factory=dict)     # symbol → weight (0 to 1)
    regime: str = "unknown"        # 'trending', 'mean_reverting', 'volatile'


class PortfolioOrchestrator:
    """
    Coordinates multi-asset predictions and portfolio allocation.

    Usage:
        orch = PortfolioOrchestrator.build(
            crypto_assets=['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
            commodity_assets=['XAU/USD', 'XAG/USD'],
            device='cuda'
        )
        allocation = orch.generate_signals(
            crypto_data={'BTC/USDT': df_btc, 'ETH/USDT': df_eth, ...},
            commodity_data={'XAU/USD': df_xau, ...},
            capital=100_000.0
        )
    """

    def __init__(
        self,
        crypto_ensemble=None,     # EnsemblePredictor for crypto
        commodity_ensemble=None,  # EnsemblePredictor for commodities
        crypto_assets: List[str] = None,
        commodity_assets: List[str] = None,
        n_paths: int = 50,
        min_confidence: float = 0.55,
        max_position_pct: float = 0.10,
        risk_parity: bool = True,
    ):
        self.crypto_ensemble = crypto_ensemble
        self.commodity_ensemble = commodity_ensemble
        self.crypto_assets = crypto_assets or []
        self.commodity_assets = commodity_assets or []
        self.n_paths = n_paths
        self.min_confidence = min_confidence
        self.max_position_pct = max_position_pct
        self.risk_parity = risk_parity

    @classmethod
    def build(
        cls,
        crypto_assets: List[str] = None,
        commodity_assets: List[str] = None,
        checkpoints_dir: Optional[Path] = None,
        device: str = "auto",
        n_paths: int = 50,
    ):
        """
        Build orchestrator with ensembles for both markets.

        Loads:
        - Zero-shot Kronos-base (shared across all assets)
        - BTC-finetuned checkpoint (applied to all crypto)
        - XAU-finetuned checkpoint (applied to all commodities)
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

        # Load zero-shot model (shared)
        print("Loading zero-shot Kronos-base...")
        zs_tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        zs_model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
        zs_model = zs_model.to(device).eval()
        zs_predictor = KronosPredictor(zs_model, zs_tokenizer, max_context=512)

        # Build crypto ensemble
        crypto_ensemble = None
        if crypto_assets:
            crypto_predictors = [zs_predictor]
            crypto_names = ["zero_shot"]
            crypto_weights = [1.0]

            # Load BTC-finetuned checkpoint
            for exp in ["cifr-btc"]:
                tok = checkpoints_dir / exp / "tokenizer" / "best_model"
                mdl = checkpoints_dir / exp / "predictor" / "best_model"
                if tok.exists() and mdl.exists():
                    print(f"Loading crypto finetuned ({exp})...")
                    ft_tok = KronosTokenizer.from_pretrained(str(tok))
                    ft_mdl = Kronos.from_pretrained(str(mdl))
                    ft_mdl = ft_mdl.to(device).eval()
                    crypto_predictors.append(KronosPredictor(ft_mdl, ft_tok, max_context=512))
                    crypto_names.append(f"finetuned_{exp}")
                    crypto_weights.append(1.0)

            from src.model.ensemble import EnsemblePredictor
            crypto_ensemble = EnsemblePredictor(
                crypto_predictors, crypto_weights, crypto_names
            )
            print(f"Crypto ensemble: {crypto_ensemble}")

        # Build commodity ensemble
        commodity_ensemble = None
        if commodity_assets:
            commodity_predictors = [zs_predictor]
            commodity_names = ["zero_shot"]
            commodity_weights = [1.0]

            # Load XAU-finetuned checkpoints (v1 and v2)
            for exp in ["cifr-xau", "cifr-xau-v2"]:
                tok = checkpoints_dir / exp / "tokenizer" / "best_model"
                mdl = checkpoints_dir / exp / "predictor" / "best_model"
                if tok.exists() and mdl.exists():
                    print(f"Loading commodity finetuned ({exp})...")
                    ft_tok = KronosTokenizer.from_pretrained(str(tok))
                    ft_mdl = Kronos.from_pretrained(str(mdl))
                    ft_mdl = ft_mdl.to(device).eval()
                    commodity_predictors.append(KronosPredictor(ft_mdl, ft_tok, max_context=512))
                    commodity_names.append(f"finetuned_{exp}")
                    commodity_weights.append(1.0)

            from src.model.ensemble import EnsemblePredictor
            commodity_ensemble = EnsemblePredictor(
                commodity_predictors, commodity_weights, commodity_names
            )
            print(f"Commodity ensemble: {commodity_ensemble}")

        return cls(
            crypto_ensemble=crypto_ensemble,
            commodity_ensemble=commodity_ensemble,
            crypto_assets=crypto_assets or [],
            commodity_assets=commodity_assets or [],
            n_paths=n_paths,
        )

    def predict_asset(
        self,
        symbol: str,
        market: str,
        df: pd.DataFrame,
        x_timestamp: pd.Series,
        y_timestamp: pd.Series,
        pred_len: int,
    ) -> Optional[AssetSignal]:
        """
        Generate signal for a single asset using the appropriate ensemble.

        Args:
            symbol: Asset symbol (e.g., 'BTC/USDT')
            market: 'crypto' or 'commodity'
            df: OHLCV DataFrame (context window)
            x_timestamp: Context timestamps
            y_timestamp: Future timestamps
            pred_len: Prediction horizon in candles

        Returns:
            AssetSignal or None if prediction fails
        """
        ensemble = self.crypto_ensemble if market == "crypto" else self.commodity_ensemble
        if ensemble is None:
            return None

        try:
            result = ensemble.predict_with_quantiles(
                df=df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=pred_len,
                n_paths=self.n_paths,
            )

            entry = result["entry_price"]
            predicted_close = result["close_median"][-1]
            predicted_return = (predicted_close - entry) / entry

            return AssetSignal(
                symbol=symbol,
                market=market,
                direction=result["direction"],
                directional_confidence=result["directional_confidence"],
                predicted_return=float(predicted_return),
                stop_loss=float(result["stop_loss"]),
                take_profit=float(result["take_profit"]),
                entry_price=float(entry),
                interval_width=float(result["close_q95"][-1] - result["close_q05"][-1]) / entry,
                n_paths=result["n_paths"],
            )
        except Exception as e:
            print(f"  Prediction failed for {symbol}: {e}")
            return None

    def generate_signals(
        self,
        crypto_data: Dict[str, dict] = None,
        commodity_data: Dict[str, dict] = None,
        capital: float = 100_000.0,
    ) -> PortfolioAllocation:
        """
        Generate signals for all assets and compute portfolio allocation.

        Args:
            crypto_data: {symbol: {'df': DataFrame, 'x_ts': Series, 'y_ts': Series, 'pred_len': int}}
            commodity_data: Same format
            capital: Total portfolio capital

        Returns:
            PortfolioAllocation with signals and position sizes
        """
        signals = []

        # Crypto signals
        if crypto_data:
            for symbol, data in crypto_data.items():
                signal = self.predict_asset(
                    symbol=symbol,
                    market="crypto",
                    df=data["df"],
                    x_timestamp=data["x_ts"],
                    y_timestamp=data["y_ts"],
                    pred_len=data["pred_len"],
                )
                if signal:
                    signals.append(signal)

        # Commodity signals
        if commodity_data:
            for symbol, data in commodity_data.items():
                signal = self.predict_asset(
                    symbol=symbol,
                    market="commodity",
                    df=data["df"],
                    x_timestamp=data["x_ts"],
                    y_timestamp=data["y_ts"],
                    pred_len=data["pred_len"],
                )
                if signal:
                    signals.append(signal)

        # Filter by minimum confidence
        tradeable = [s for s in signals if s.directional_confidence >= self.min_confidence]

        # Position sizing
        positions, weights = self._size_positions(tradeable, capital)

        return PortfolioAllocation(
            signals=signals,
            timestamp=pd.Timestamp.now(),
            total_capital=capital,
            positions=positions,
            weights=weights,
        )

    def _size_positions(
        self,
        signals: List[AssetSignal],
        capital: float,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Size positions using inverse-volatility (risk parity) weighting.

        Narrow prediction intervals → larger position (more confident).
        Wide prediction intervals → smaller position (less confident).

        Each position is capped at max_position_pct of capital.
        """
        if not signals:
            return {}, {}

        positions = {}
        weights = {}

        if self.risk_parity:
            # Inverse interval width weighting
            inv_widths = {s.symbol: 1.0 / max(s.interval_width, 0.001) for s in signals}
            total_inv = sum(inv_widths.values())

            for s in signals:
                raw_weight = inv_widths[s.symbol] / total_inv
                # Cap at max position
                capped_weight = min(raw_weight, self.max_position_pct)
                weights[s.symbol] = capped_weight
                positions[s.symbol] = capped_weight * capital
        else:
            # Equal weight
            n = len(signals)
            per_asset = min(1.0 / n, self.max_position_pct)
            for s in signals:
                weights[s.symbol] = per_asset
                positions[s.symbol] = per_asset * capital

        return positions, weights

    def summary(self, allocation: PortfolioAllocation) -> str:
        """Pretty-print portfolio allocation."""
        lines = [
            f"\n{'='*70}",
            f"  PORTFOLIO ALLOCATION — {allocation.timestamp.strftime('%Y-%m-%d %H:%M')}",
            f"  Capital: ${allocation.total_capital:,.0f}",
            f"{'='*70}",
        ]

        # Group by market
        for market in ["crypto", "commodity"]:
            market_signals = [s for s in allocation.signals if s.market == market]
            if not market_signals:
                continue

            lines.append(f"\n  {market.upper()} ({len(market_signals)} assets)")
            lines.append(f"  {'Symbol':<12} {'Dir':>5} {'Conf':>6} {'Pred.Ret':>9} {'Width':>8} {'Position':>10}")
            lines.append(f"  {'-'*56}")

            for s in sorted(market_signals, key=lambda x: -x.directional_confidence):
                pos = allocation.positions.get(s.symbol, 0)
                active = "→" if s.directional_confidence >= self.min_confidence else " "
                lines.append(
                    f" {active}{s.symbol:<12} {s.direction:>5} {s.directional_confidence:>5.1%} "
                    f"{s.predicted_return:>+8.4f} {s.interval_width:>7.4f} ${pos:>9,.0f}"
                )

        tradeable = sum(1 for s in allocation.signals if s.directional_confidence >= self.min_confidence)
        deployed = sum(allocation.positions.values())
        lines.append(f"\n  Tradeable: {tradeable}/{len(allocation.signals)} assets")
        lines.append(f"  Capital deployed: ${deployed:,.0f} ({deployed/allocation.total_capital:.1%})")

        return "\n".join(lines)
