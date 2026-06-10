"""Slow cross-sectional momentum: Phase 2C signal candidate #2.

Hypothesis (documented in crypto academic literature): over 1-4 week lookbacks,
recent relative WINNERS keep outperforming recent relative LOSERS => XS rank IC
between trailing return and forward return is POSITIVE. (If it comes out
reliably negative instead, that's short-term reversal — also tradeable, but a
different book.)

Why this candidate: structurally different from carry (price signal vs flow
signal => likely decorrelated brick), and slow by construction (weekly-ish
turnover => no cost battle like carry v1).

Measures, off the perp 1h klines already on disk (CPU, seconds):
  1. XS rank IC grid: lookbacks {7, 14, 28, 28skip7} days × forward horizons
     {1, 3, 7} days. t-stats on NON-OVERLAPPING daily periods (subsampled by
     horizon). 12 cells => multiple testing: we demand sign CONSISTENCY across
     the grid + per-year stability, not one lucky cell (Bonferroni-ish bar for
     the chosen cell: |t| > ~2.6).
  2. Per-year stability for every lookback at the 7d horizon.
  3. Decorrelation vs carry: mean |spearman| between momentum ranks and
     trailing-3d-funding ranks (want LOW => independent brick).

Usage:
    python scripts/momo_skill.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.base_config import DATA_RAW_DIR

DERIVS_DIR = DATA_RAW_DIR / "derivs"
LOOKBACKS = {"7d": (7, 0), "14d": (14, 0), "28d": (28, 0), "28-7d": (28, 7)}
HORIZONS = {"1d": 1, "3d": 3, "7d": 7}
MIN_ASSETS = 8


def _spearman(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 3:
        return float("nan")
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _tstat(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 20 or x.std(ddof=1) == 0:
        return float("nan"), len(x)
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))), len(x)


def load_daily_panel():
    """Daily 00:00 close panel from the perp 1h files, plus 3d-mean funding."""
    from configs.crypto_universe import get_crypto_configs
    symbols = list(get_crypto_configs(tiers=(1, 2)).keys())

    closes, funding = {}, {}
    for sym in symbols:
        safe = sym.replace("/", "_").lower()
        p_path = DERIVS_DIR / f"{safe}_perp_1h.csv"
        f_path = DERIVS_DIR / f"{safe}_funding.csv"
        if not p_path.exists():
            print(f"  SKIP {sym}: no perp data", flush=True)
            continue
        p = pd.read_csv(p_path)
        p["timestamps"] = pd.to_datetime(p["timestamps"])
        daily = p.set_index("timestamps")["close"].resample("1D").last()
        closes[sym] = daily
        if f_path.exists():
            f = pd.read_csv(f_path)
            f["timestamps"] = pd.to_datetime(f["timestamps"]).dt.round("h")
            fr = f.set_index("timestamps")["funding_rate"].rolling(9, min_periods=3).mean()
            funding[sym] = fr.resample("1D").last()
    return pd.DataFrame(closes).sort_index(), pd.DataFrame(funding).sort_index()


def main():
    close, fund3d = load_daily_panel()
    print(f"Daily panel: {close.shape[0]} days × {close.shape[1]} assets "
          f"({close.index[0].date()} .. {close.index[-1].date()})", flush=True)

    # Precompute signals and forward returns.
    signals = {}
    for name, (lb, skip) in LOOKBACKS.items():
        signals[name] = close.shift(skip) / close.shift(lb) - 1.0
    fwd = {h: close.shift(-d) / close - 1.0 for h, d in HORIZONS.items()}

    print(f"\n{'='*72}\n  XS MOMENTUM SKILL  (hypothesis: POSITIVE IC)\n{'='*72}", flush=True)
    print(f"\n  Pooled XS rank IC grid (t-stats on non-overlapping days):", flush=True)
    print(f"  {'lookback':>9s}" + "".join(f"  {h+'_IC':>8s} {h+'_t':>7s}" for h in HORIZONS),
          flush=True)

    ic_series = {}   # (lookback, horizon) -> list of daily ICs (non-overlapping)
    for lname, sig in signals.items():
        line = f"  {lname:>9s}"
        for h, d in HORIZONS.items():
            ics, dates = [], []
            for i in range(0, len(close), d):           # non-overlapping
                s_row = sig.iloc[i]
                r_row = fwd[h].iloc[i]
                m = s_row.notna() & r_row.notna()
                if m.sum() >= MIN_ASSETS:
                    ic = _spearman(s_row[m], r_row[m])
                    if np.isfinite(ic):
                        ics.append(ic); dates.append(close.index[i])
            ic_series[(lname, h)] = pd.Series(ics, index=dates)
            t, n = _tstat(np.asarray(ics))
            line += f"  {np.nanmean(ics):>+8.3f} {t:>+7.2f}"
        print(line, flush=True)

    for hz in ("1d", "7d"):
        print(f"\n  Per-year stability ({hz} horizon):", flush=True)
        for lname in LOOKBACKS:
            s = ic_series[(lname, hz)]
            line = f"  {lname:>9s}:"
            for year, ys in s.groupby(s.index.year):
                if len(ys) >= 20:
                    t, n = _tstat(ys.to_numpy())
                    line += f"  {year}: {ys.mean():+.3f} (t={t:+.2f})"
            print(line, flush=True)

    # Construction-matched tail check: what a K=3 tails portfolio ACTUALLY trades.
    # (Lesson from the reversal backtest: rank IC measures the whole cross-section;
    # a K portfolio trades only the tails, which can behave oppositely.)
    print("\n  Tail spread check (bottom-3 minus top-3 by 7d trailing ret, 1d fwd):",
          flush=True)
    sig7 = signals["7d"]
    spreads, sdates = [], []
    for i in range(len(close) - 1):
        s_row = sig7.iloc[i]
        r_row = fwd["1d"].iloc[i]
        m = s_row.notna() & r_row.notna()
        if m.sum() >= MIN_ASSETS:
            order = s_row[m].sort_values().index
            spread = float(r_row[order[:3]].mean() - r_row[order[-3:]].mean())
            spreads.append(spread); sdates.append(close.index[i])
    sp = pd.Series(spreads, index=sdates)
    t, n = _tstat(sp.to_numpy())
    print(f"    pooled: mean={sp.mean():+.4%}/day  t={t:+.2f}  n={n}  "
          f"(positive = reversal tradeable in the tails)", flush=True)
    for year, ys in sp.groupby(sp.index.year):
        if len(ys) >= 30:
            t, n = _tstat(ys.to_numpy())
            print(f"    {year}: mean={ys.mean():+.4%}/day  t={t:+.2f}", flush=True)

    # Decorrelation vs carry signal.
    print("\n  Overlap with the carry book (momentum rank vs 3d-funding rank):", flush=True)
    overlaps = []
    sig28 = signals["28-7d"]
    common = [c for c in sig28.columns if c in fund3d.columns]
    for i in range(0, len(close), 7):
        s_row = sig28.iloc[i][common]
        ts = close.index[i]
        if ts in fund3d.index:
            f_row = fund3d.loc[ts, common]
            m = s_row.notna() & f_row.notna()
            if m.sum() >= MIN_ASSETS:
                ic = _spearman(s_row[m], f_row[m])
                if np.isfinite(ic):
                    overlaps.append(ic)
    if overlaps:
        print(f"    mean signal-rank correlation: {np.mean(overlaps):+.3f}  "
              f"(|·| < ~0.3 => substantially independent brick)", flush=True)

    print("\n  GATE: 12 grid cells => demand sign consistency across the grid AND", flush=True)
    print("  per-year stability; chosen cell needs |t| > ~2.6 (multiplicity-adjusted).",
          flush=True)


if __name__ == "__main__":
    main()
