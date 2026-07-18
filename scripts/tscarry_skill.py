"""Commodity TERM-STRUCTURE CARRY diagnostic (the gauntlet, pre-backtest).

Hypothesis (declared before running): the front slope of the futures curve
predicts front-contract excess returns — backwardation (c1 > c2) is bullish,
contango bearish. The economic payer: hedgers paying speculators for price
insurance (normal backwardation), documented as one of the most persistent
commodity premia (Gorton-Rouwenhorst 2006; Koijen et al. 2018 "Carry").

DATA: EIA daily contract-1..4 panels (scripts/fetch_futures_curve.py),
6 energy assets, ~1985/1994..April 2024 (EIA discontinued the series then).
The live post-2024 window accumulates separately via --snapshot cron.

DECLARED CONSTRUCTION (single config — count trials, no tuning):
  signal  = 5-day trailing mean of (c1/c2 - 1), known at close t
  target  = forward h-day sum of daily front EXCESS returns starting t+1
  rolls   = detected via the joint column-shift signature: on a roll date every
            column steps forward (c1[t]~c2[t-1], c2[t]~c3[t-1], ...); pick
            stay/shift by total |log-diff|; >=15bd between rolls. Validity
            check printed: detected rolls must be ~12/yr/asset.
  cells   = 2 horizons (5bd, 21bd) x 2 constructions:
              TS: per asset, long if slope>0 else short, equal weight
              XS: rank across >=4 live assets, top-half long / bottom-half
                  short (construction-matched: what a portfolio would trade)
  GATE: 4 cells -> |t| > ~2.5 on non-overlapping samples, AND multi-decade
  bucket stability (no sign-flipping regimes), or the signal dies here.

Usage:
    python scripts/tscarry_skill.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.base_config import DATA_RAW_DIR

CURVE_DIR = DATA_RAW_DIR / "futures_curve"
ASSETS = ["wti", "natgas", "heatoil", "rbob", "gasoline", "propane"]
SMOOTH_D = 5
HORIZONS = [5, 21]
MIN_ROLL_GAP = 15


def load_asset(key):
    cols = {}
    for n in range(1, 5):
        p = CURVE_DIR / f"{key}_c{n}.csv"
        if p.exists():
            s = pd.read_csv(p)
            s["timestamps"] = pd.to_datetime(s["timestamps"])
            cols[f"c{n}"] = s.set_index("timestamps")["price"]
    if "c1" not in cols or "c2" not in cols:
        return None
    df = pd.DataFrame(cols).sort_index()
    return df.dropna(subset=["c1", "c2"])


def detect_rolls_and_returns(df):
    """Daily front excess log-returns with roll-day repair + roll flags."""
    idx = df.index
    r = pd.Series(np.nan, index=idx)
    rolls = pd.Series(False, index=idx)
    last_roll_i = -10**9
    vals = df.to_numpy()
    colmap = {c: i for i, c in enumerate(df.columns)}
    for i in range(1, len(idx)):
        if (idx[i] - idx[i - 1]).days > 7:
            continue
        stay, shift, pairs = 0.0, 0.0, 0
        for n in (1, 2, 3):
            a, b = f"c{n}", f"c{n + 1}"
            if a in colmap and b in colmap:
                cn_t, cn_p = vals[i, colmap[a]], vals[i - 1, colmap[a]]
                cnn_p = vals[i - 1, colmap[b]]
                if np.isfinite(cn_t) and np.isfinite(cn_p) and np.isfinite(cnn_p) \
                        and cn_p > 0 and cnn_p > 0 and cn_t > 0:
                    stay += abs(np.log(cn_t / cn_p))
                    shift += abs(np.log(cn_t / cnn_p))
                    pairs += 1
        if pairs == 0:
            continue
        c1_t, c1_p = vals[i, colmap["c1"]], vals[i - 1, colmap["c1"]]
        c2_p = vals[i - 1, colmap["c2"]]
        if shift < stay and (i - last_roll_i) >= MIN_ROLL_GAP:
            rolls.iloc[i] = True
            last_roll_i = i
            r.iloc[i] = np.log(c1_t / c2_p)
        else:
            r.iloc[i] = np.log(c1_t / c1_p)
    return r, rolls


def bucket_label(ts):
    y = ts.year
    lo = y - (y - 1985) % 3
    return f"{lo}-{lo + 2}"


def main():
    sig, fwd, rets = {}, {h: {} for h in HORIZONS}, {}
    print(f"{'='*72}\n  TERM-STRUCTURE CARRY SKILL  (EIA c1..c4 panels)\n{'='*72}",
          flush=True)
    for key in ASSETS:
        df = load_asset(key)
        if df is None:
            print(f"  SKIP {key}: no data", flush=True)
            continue
        r, rolls = detect_rolls_and_returns(df)
        years = max((df.index[-1] - df.index[0]).days / 365.25, 1e-9)
        rpy = rolls.sum() / years
        slope = (df["c1"] / df["c2"] - 1.0).rolling(SMOOTH_D, min_periods=1).mean()
        sig[key] = slope
        rets[key] = r
        for h in HORIZONS:
            fwd[h][key] = r.shift(-1).rolling(h, min_periods=h).sum().shift(-(h - 1))
        print(f"  {key:9s} {df.index[0].date()} .. {df.index[-1].date()}  "
              f"({len(df)} days)  detected rolls/yr: {rpy:4.1f} "
              f"{'OK' if 9 <= rpy <= 14 else '<-- CHECK'}", flush=True)

    keys = list(sig.keys())
    S = pd.DataFrame(sig)
    print(f"\n  Signal snapshot: median |slope| "
          f"{np.nanmedian(np.abs(S.to_numpy())):.2%}/month across panel", flush=True)

    results = {}
    for h in HORIZONS:
        F = pd.DataFrame(fwd[h]).reindex(S.index)
        step_idx = S.index[::h]                        # non-overlapping periods
        Ss, Fs = S.reindex(step_idx), F.reindex(step_idx)

        # --- TS cell: long backwardation / short contango, equal weight
        port_ts = (np.sign(Ss) * Fs).mean(axis=1, skipna=True).dropna()
        t_ts = port_ts.mean() / (port_ts.std(ddof=1) / np.sqrt(len(port_ts)))

        # --- XS cell: top-half long / bottom-half short among >=4 live assets
        xs_rows = []
        for t in Ss.index:
            row_s, row_f = Ss.loc[t].dropna(), Fs.loc[t]
            row_s = row_s[row_f.reindex(row_s.index).notna()]
            if len(row_s) < 4:
                continue
            k = len(row_s) // 2
            order = row_s.sort_values()
            longs, shorts = order.index[-k:], order.index[:k]
            xs_rows.append((t, row_f[longs].mean() - row_f[shorts].mean()))
        port_xs = pd.Series(dict(xs_rows)).sort_index()
        t_xs = (port_xs.mean() / (port_xs.std(ddof=1) / np.sqrt(len(port_xs)))
                if len(port_xs) > 10 else float("nan"))

        # per-asset IC (diagnostic color, not a gate cell)
        ics = []
        for k2 in keys:
            m = Ss[k2].notna() & Fs[k2].notna()
            if m.sum() > 30:
                ic = stats.spearmanr(Ss[k2][m], Fs[k2][m]).statistic
                ics.append(f"{k2}:{ic:+.03f}")
        ann = 252 / h
        print(f"\n  h={h:2d}bd  TS: mean={port_ts.mean():+.4%}/prd "
              f"(~{port_ts.mean() * ann:+.1%}/yr)  t={t_ts:+.2f}  n={len(port_ts)}",
              flush=True)
        print(f"         XS: mean={port_xs.mean():+.4%}/prd "
              f"(~{port_xs.mean() * ann:+.1%}/yr)  t={t_xs:+.2f}  n={len(port_xs)}",
              flush=True)
        print(f"         per-asset IC: {'  '.join(ics)}", flush=True)
        results[h] = (port_ts, port_xs)

    # --- stability: 3-year buckets on the TS and XS books, h=21
    print(f"\n  Stability (3y buckets, h=21bd):", flush=True)
    pts, pxs = results[21]
    for name, p in [("TS", pts), ("XS", pxs)]:
        buckets = p.groupby(p.index.map(bucket_label))
        line = []
        for b, v in buckets:
            if len(v) >= 8:
                line.append(f"{b}:{v.mean() * 12:+.1%}")
        print(f"    {name}: " + "  ".join(line), flush=True)
        pos = sum(1 for b, v in buckets if len(v) >= 8 and v.mean() > 0)
        tot = sum(1 for b, v in buckets if len(v) >= 8)
        print(f"        positive buckets: {pos}/{tot}", flush=True)

    print(f"\n  GATE: 4 declared cells (TS/XS x 5/21bd) -> |t| > ~2.5 AND bucket\n"
          f"  stability (no sign-flip regimes), else the candidate dies here.\n"
          f"  Costs/backtest come ONLY after a pass (methodology rule #1/#5).",
          flush=True)


if __name__ == "__main__":
    main()
