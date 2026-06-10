"""Time-series momentum (TSMOM): Phase 2C candidate #3.

Each asset is timed against ITS OWN history: hold LONG while its trailing
L-day return is positive, SHORT while negative (Moskowitz-Ooi-Pedersen
structure — the managed-futures classic, documented in crypto too).

Structurally different from both existing candidates:
  - carry: flow signal, dollar-neutral
  - XS momentum/reversal: relative ranks, dollar-neutral (failed)
  - TSMOM: own-history timing, NET exposure varies (long book in bulls, short
    in bears) — a third risk profile, expected correlated with market beta in
    level but with timing alpha if real.

Construction-matched by nature: the statistic tested IS the strategy's gross
return — mean of sign(trailing ret) × forward return, per asset, equal-weight
portfolio. t-stats on non-overlapping periods. Includes per-year stability,
per-asset sign consistency, a buy-and-hold baseline (TSMOM must beat passive
long — otherwise it's just beta), and an estimated cost drag from sign flips.

Usage:
    python scripts/tsmom_skill.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.base_config import DATA_RAW_DIR

DERIVS_DIR = DATA_RAW_DIR / "derivs"
LOOKBACKS = {"7d": 7, "14d": 14, "28d": 28, "90d": 90}
HORIZONS = {"1d": 1, "3d": 3, "7d": 7}


def _tstat(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 20 or x.std(ddof=1) == 0:
        return float("nan"), len(x)
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))), len(x)


def load_close():
    from configs.crypto_universe import get_crypto_configs
    symbols = list(get_crypto_configs(tiers=(1, 2)).keys())
    closes = {}
    for sym in symbols:
        safe = sym.replace("/", "_").lower()
        p_path = DERIVS_DIR / f"{safe}_perp_1h.csv"
        if not p_path.exists():
            print(f"  SKIP {sym}: no perp data", flush=True)
            continue
        p = pd.read_csv(p_path)
        p["timestamps"] = pd.to_datetime(p["timestamps"])
        closes[sym] = p.set_index("timestamps")["close"].resample("1D").last()
    return pd.DataFrame(closes).sort_index()


def main():
    close = load_close()
    print(f"Daily panel: {close.shape[0]} days × {close.shape[1]} assets "
          f"({close.index[0].date()} .. {close.index[-1].date()})", flush=True)

    fwd = {h: close.shift(-d) / close - 1.0 for h, d in HORIZONS.items()}

    print(f"\n{'='*72}\n  TIME-SERIES MOMENTUM SKILL\n{'='*72}", flush=True)
    print("\n  Equal-weight TSMOM portfolio gross return (sign(trailing) × fwd):",
          flush=True)
    print(f"  {'lookback':>9s}" + "".join(
        f"  {h+'/d':>9s} {h+'_t':>7s}" for h in HORIZONS), flush=True)

    port_series = {}
    for lname, lb in LOOKBACKS.items():
        sign = np.sign(close / close.shift(lb) - 1.0)
        line = f"  {lname:>9s}"
        for h, d in HORIZONS.items():
            rets, dates = [], []
            for i in range(lb, len(close) - d, d):          # non-overlapping
                s_row = sign.iloc[i]
                r_row = fwd[h].iloc[i]
                m = s_row.notna() & r_row.notna()
                if m.sum() >= 8:
                    # per-day units for comparability across horizons
                    rets.append(float((s_row[m] * r_row[m]).mean()) / d)
                    dates.append(close.index[i])
            port_series[(lname, h)] = pd.Series(rets, index=dates)
            t, n = _tstat(np.asarray(rets))
            line += f"  {np.nanmean(rets):>+9.4%} {t:>+7.2f}"
        print(line, flush=True)

    # Baseline: passive equal-weight long (TSMOM must beat this to be timing alpha).
    print("\n  Baseline — passive equal-weight LONG (per-day):", flush=True)
    bh = []
    for i in range(0, len(close) - 1):
        r_row = fwd["1d"].iloc[i]
        if r_row.notna().sum() >= 8:
            bh.append(float(r_row.dropna().mean()))
    bh = np.asarray(bh)
    t, n = _tstat(bh)
    print(f"    mean={np.nanmean(bh):+.4%}/day  t={t:+.2f}  n={n}", flush=True)

    print("\n  Per-year stability (1d horizon, per-day gross):", flush=True)
    for lname in LOOKBACKS:
        s = port_series[(lname, "1d")]
        line = f"  {lname:>9s}:"
        for year, ys in s.groupby(s.index.year):
            if len(ys) >= 30:
                t, n = _tstat(ys.to_numpy())
                line += f"  {year}: {ys.mean():+.3%} (t={t:+.2f})"
        print(line, flush=True)

    # Per-asset sign consistency (28d, 1d horizon).
    print("\n  Per-asset TSMOM gross (28d lookback, 1d horizon):", flush=True)
    sign28 = np.sign(close / close.shift(28) - 1.0)
    for sym in close.columns:
        x = (sign28[sym] * fwd["1d"][sym]).dropna()
        if len(x) >= 100:
            t, n = _tstat(x.to_numpy())
            print(f"    {sym:12s} mean={x.mean():+.4%}/day  t={t:+.2f}", flush=True)

    # Cost reality: sign flips per asset per day -> turnover estimate.
    print("\n  Cost drag estimate (sign flips => 2.0 turnover each):", flush=True)
    for lname, lb in LOOKBACKS.items():
        sign = np.sign(close / close.shift(lb) - 1.0)
        flips = (sign != sign.shift(1)).sum().sum() / max(len(close) - lb, 1)
        # flips per day across the book; each flip turns over 2/n_assets gross
        n_assets = close.shape[1]
        daily_turnover = flips * 2.0 / n_assets
        print(f"    {lname:>6s}: ~{flips:.2f} flips/day -> turnover ~{daily_turnover:.2f} "
              f"gross/day -> maker drag ~{daily_turnover * 0.0003 * 365:.1%}/yr", flush=True)

    print("\n  GATE: portfolio |t| > ~2.6 (12 cells), per-year stability, AND must", flush=True)
    print("  beat the passive-long baseline (else it's just beta).", flush=True)


if __name__ == "__main__":
    main()
