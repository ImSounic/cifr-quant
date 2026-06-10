"""Funding-rate carry: the first v2 signal diagnostic.

Hypothesis (documented in the literature): persistently HIGH funding predicts
NEGATIVE forward perp returns (crowded longs paying through the nose), and vice
versa. Separately, short-perp/long-spot HARVESTS funding directly, which is
income independent of direction.

This script measures, per asset and cross-sectionally:
  1. Predictive IC: spearman(funding_rate, forward perp return) at 8h/1d/3d
     horizons. Carry hypothesis => NEGATIVE IC. t-stats computed on
     NON-OVERLAPPING samples only (subsampled by horizon) — no autocorrelation
     cheating.
  2. Cross-sectional IC: at each funding event, rank assets by funding vs their
     forward returns. Mean XS IC + t-stat across events.
  3. Harvest economics: annualized funding income from always-short-perp (hedged
     with spot) and from a conditional variant (only when trailing 30d funding
     percentile > 0.7), before basis/costs.

GATE (the v2 constitution): a backtest is built ONLY if |t| > 2 on the
predictive ICs, or the harvest income is economically significant after cost
estimates. CPU-only, runs anywhere the data exists.

Usage:
    python scripts/carry_skill.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.base_config import DATA_RAW_DIR

DERIVS_DIR = DATA_RAW_DIR / "derivs"
HORIZONS = {"8h": 1, "1d": 3, "3d": 9}      # in 8h funding events
PCTL_WINDOW = 90                            # trailing events for percentile (~30d)


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
    if len(x) < 5 or x.std(ddof=1) == 0:
        return float("nan"), len(x)
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))), len(x)


def load_asset(sym):
    safe = sym.replace("/", "_").lower()
    f_path = DERIVS_DIR / f"{safe}_funding.csv"
    p_path = DERIVS_DIR / f"{safe}_perp_1h.csv"
    if not (f_path.exists() and p_path.exists()):
        return None
    fund = pd.read_csv(f_path)
    fund["timestamps"] = pd.to_datetime(fund["timestamps"])
    perp = pd.read_csv(p_path)
    perp["timestamps"] = pd.to_datetime(perp["timestamps"])
    return fund.sort_values("timestamps").reset_index(drop=True), \
           perp.sort_values("timestamps").reset_index(drop=True)


def build_events(fund, perp):
    """One row per funding event with forward perp returns at each horizon."""
    pts = perp["timestamps"].values
    pclose = perp["close"].to_numpy(dtype=float)

    def close_at(ts):
        i = np.searchsorted(pts, ts, side="right") - 1
        return pclose[i] if i >= 0 else np.nan

    rows = []
    ets = fund["timestamps"].values
    rates = fund["funding_rate"].to_numpy(dtype=float)
    pctl = fund["funding_rate"].rolling(PCTL_WINDOW).apply(
        lambda w: (w <= w.iloc[-1]).mean(), raw=False).to_numpy()

    for i in range(len(fund)):
        t = ets[i]
        c0 = close_at(t)
        if not np.isfinite(c0) or c0 <= 0:
            continue
        row = {"timestamp": fund["timestamps"].iloc[i],
               "funding": rates[i], "funding_pctl": pctl[i]}
        ok = False
        for name, n_ev in HORIZONS.items():
            t1 = t + np.timedelta64(8 * n_ev, "h")
            c1 = close_at(t1)
            # require the horizon end to actually exist in the data
            if np.isfinite(c1) and pts[-1] >= t1:
                row[f"ret_{name}"] = (c1 - c0) / c0
                ok = True
            else:
                row[f"ret_{name}"] = np.nan
        if ok:
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    from configs.crypto_universe import get_crypto_configs
    symbols = list(get_crypto_configs(tiers=(1, 2)).keys())

    per_asset = {}
    for sym in symbols:
        loaded = load_asset(sym)
        if loaded is None:
            print(f"  SKIP {sym}: no derivs data", flush=True)
            continue
        fund, perp = loaded
        ev = build_events(fund, perp)
        if len(ev) < 100:
            print(f"  SKIP {sym}: only {len(ev)} events", flush=True)
            continue
        per_asset[sym] = ev

    if not per_asset:
        print("No data — run scripts/fetch_derivs.py first.")
        return

    print(f"\n{'='*72}\n  FUNDING-CARRY SKILL  ({len(per_asset)} assets)\n{'='*72}", flush=True)

    # ---------- 1. Time-series predictive IC per asset ----------
    print("\n  Per-asset time-series IC (funding vs fwd return; hypothesis: NEGATIVE)",
          flush=True)
    print(f"  {'asset':12s} {'n':>6s}" + "".join(
        f"  {h+'_IC':>8s} {h+'_t':>7s}" for h in HORIZONS), flush=True)
    pooled_t = {h: [] for h in HORIZONS}
    for sym, ev in per_asset.items():
        line = f"  {sym:12s} {len(ev):>6d}"
        for h, n_ev in HORIZONS.items():
            sub = ev.iloc[::n_ev]                       # non-overlapping
            ic = _spearman(sub["funding"], sub[f"ret_{h}"])
            # per-event signed contribution for the t-stat: sign-flipped product
            # of standardized ranks is overkill; use bootstrap-free approach:
            # t-stat of the per-sample products of centered ranks
            fr = pd.Series(sub["funding"]).rank()
            rr = pd.Series(sub[f"ret_{h}"]).rank()
            prod = ((fr - fr.mean()) * (rr - rr.mean())) / (fr.std(ddof=1) * rr.std(ddof=1))
            t, n = _tstat(prod.to_numpy())
            pooled_t[h].append(ic)
            line += f"  {ic:>+8.3f} {t:>+7.2f}"
        print(line, flush=True)

    print("\n  Mean per-asset IC across universe (sign consistency is the tell):", flush=True)
    for h in HORIZONS:
        ics = np.asarray(pooled_t[h], dtype=float)
        t, n = _tstat(ics)
        print(f"    {h}: mean IC={np.nanmean(ics):+.4f}  t(across {n} assets)={t:+.2f}  "
              f"negative in {np.mean(ics < 0):.0%} of assets", flush=True)

    # ---------- 2. Cross-sectional IC ----------
    merged = pd.concat([ev.assign(symbol=s) for s, ev in per_asset.items()])
    print("\n  Cross-sectional rank IC (rank assets by funding vs their fwd returns):",
          flush=True)
    for h, n_ev in HORIZONS.items():
        ics = []
        grouped = merged.groupby("timestamp")
        for i, (ts, g) in enumerate(grouped):
            if i % n_ev:                                 # non-overlapping periods
                continue
            if len(g) >= 5:
                ic = _spearman(g["funding"], g[f"ret_{h}"])
                if np.isfinite(ic):
                    ics.append(ic)
        t, n = _tstat(np.asarray(ics))
        print(f"    {h}: mean XS IC={np.nanmean(ics):+.4f}  t={t:+.2f}  periods={n}  "
              f"IC<0 in {np.mean(np.asarray(ics) < 0):.0%}", flush=True)

    # ---------- 3. Harvest economics ----------
    print("\n  Funding HARVEST (short perp / long spot collects positive funding):",
          flush=True)
    print(f"  {'asset':12s} {'mean_rate/8h':>13s} {'ann.yield':>10s} {'%pos':>6s} "
          f"{'cond.yield(p>0.7)':>18s}", flush=True)
    for sym, ev in per_asset.items():
        mean_rate = ev["funding"].mean()
        ann = mean_rate * 3 * 365
        pos = (ev["funding"] > 0).mean()
        cond = ev.loc[ev["funding_pctl"] > 0.7, "funding"]
        cond_ann = cond.mean() * 3 * 365 * (len(cond) / len(ev)) if len(cond) else 0.0
        print(f"  {sym:12s} {mean_rate:>+13.5%} {ann:>+10.2%} {pos:>6.0%} {cond_ann:>+18.2%}",
              flush=True)

    print("\n  GATE: build a backtest only if |t| > 2 above (predictive), or the", flush=True)
    print("  harvest yield comfortably exceeds round-trip costs (~0.2%) + basis risk.", flush=True)


if __name__ == "__main__":
    main()
