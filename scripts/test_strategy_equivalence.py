"""Regression check: the refactored strategy+sizer pipeline must reproduce the
ORIGINAL inline engine math on identical forecasts.

Exact P&L of job 511139 can't be reproduced (the prior run's MC paths weren't
saved), so we test LOGICAL equivalence: given the same ForecastBundles + CQR
corrections, `DirectionalMomentum` + `InverseWidthRiskParity` must produce the
exact same gated set, SL/TP levels, and position sizes as the old formula.

CPU-only, no model, no data. Run on the laptop:
    python scripts/test_strategy_equivalence.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.strategy_api import AssetDecision, ForecastBundle
from src.backtest.strategies import DirectionalMomentum
from src.backtest.sizing import InverseWidthRiskParity


def old_inline(forecasts, corrections, min_confidence, max_position_pct, capital):
    """Verbatim re-implementation of the ORIGINAL engine's gating + sizing."""
    candidates = []
    for fb in forecasts:
        corr = corrections.get(fb.symbol, 0.0)
        entry = float(fb.entry_price)
        lower = float(fb.q05) - corr
        upper = float(fb.q95) + corr
        width = (upper - lower) / entry if entry > 0 else np.inf
        conf = float(fb.confidence)
        direction = fb.direction
        if conf < min_confidence or direction not in ("long", "short"):
            continue
        candidates.append(dict(symbol=fb.symbol, direction=direction, lower=lower,
                               upper=upper, entry=entry, width=max(width, 1e-4)))
    if not candidates:
        return {}
    inv = np.array([1.0 / c["width"] for c in candidates])
    raw_w = inv / inv.sum()
    weights = np.minimum(raw_w, max_position_pct)
    out = {}
    for c, w in zip(candidates, weights):
        if c["direction"] == "long":
            sl, tp = c["lower"], c["upper"]
        else:
            sl, tp = c["upper"], c["lower"]
        out[c["symbol"]] = dict(direction=c["direction"], sl=sl, tp=tp,
                                pos_usd=float(w) * capital)
    return out


def new_pipeline(forecasts, corrections, min_confidence, max_position_pct, capital):
    decisions = [AssetDecision(symbol=fb.symbol, context_df=pd.DataFrame(),
                               forecast=fb.with_correction(corrections.get(fb.symbol, 0.0)))
                 for fb in forecasts]
    strat = DirectionalMomentum(min_confidence=min_confidence)
    sizer = InverseWidthRiskParity(max_position_pct=max_position_pct)
    intents = strat.generate_intents(decisions)
    sizes = sizer.size(intents, capital)
    out = {}
    for it in intents:
        out[it.symbol] = dict(direction=it.direction, sl=it.stop, tp=it.target,
                              pos_usd=sizes[it.symbol])
    return out


def main():
    rng = np.random.default_rng(0)
    forecasts = []
    for i in range(20):
        entry = float(rng.uniform(10, 50000))
        spread = entry * rng.uniform(0.005, 0.05)
        up = rng.random() > 0.5
        forecasts.append(ForecastBundle(
            symbol=f"A{i}", timestamp=pd.Timestamp("2026-01-01"),
            entry_price=entry,
            q05=entry - spread, q25=entry - spread / 2, q50=entry,
            q75=entry + spread / 2, q95=entry + spread, mean=entry,
            direction="long" if up else "short",
            confidence=float(rng.uniform(0.50, 0.95)), pred_len=6,
        ))
    corrections = {f"A{i}": float(rng.uniform(0, 100)) for i in range(20)}
    cfg = dict(min_confidence=0.55, max_position_pct=0.10, capital=50_000.0)

    old = old_inline(forecasts, corrections, **cfg)
    new = new_pipeline(forecasts, corrections, **cfg)

    assert set(old) == set(new), f"gated set differs: {set(old) ^ set(new)}"
    max_err = 0.0
    for sym in old:
        for k in ("sl", "tp", "pos_usd"):
            err = abs(old[sym][k] - new[sym][k])
            max_err = max(max_err, err)
            assert err < 1e-9, f"{sym}.{k}: old={old[sym][k]} new={new[sym][k]}"
        assert old[sym]["direction"] == new[sym]["direction"]

    print(f"OK: {len(old)} gated positions match exactly "
          f"(max |diff| across sl/tp/pos_usd = {max_err:.2e})")
    print("Refactor is logically equivalent to the original engine.")


if __name__ == "__main__":
    main()
