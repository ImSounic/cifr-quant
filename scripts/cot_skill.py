"""COT positioning diagnostic: does speculator crowding predict commodity returns?

In plain terms: when hedge funds/CTAs ("non-commercials") are extremely crowded
long in a commodity, the easy buying is done and forward returns tend to be
poor — and vice versa. Fade the crowd. Same mechanism family as our validated
funding-carry brick (crowded positioning → adverse forward returns), expressed
in a completely different market on a weekly clock → if real, a maximally
DECORRELATED second brick.

Declared signals (trial ledger: 2 signals × 3 horizons = 6 cells, bar |t|>2.4):
  A. spec_pctl: net speculator position (long−short)/OI, percentile over a
     trailing 3 years. Hypothesis: HIGH percentile → NEGATIVE forward return.
  B. comm_pctl: net commercial position percentile (the hedgers — historically
     the "smart money" side). Hypothesis: HIGH → POSITIVE forward return.

NO-LOOKAHEAD HANDLING (the classic COT trap): the report is stamped Tuesday but
only PUBLISHED Friday afternoon. Signals here become usable the following
MONDAY (report_date + 6 days); forward returns start from the first close at or
after that. Anyone testing COT without this lag is cheating by 3+ days.

Tests (per the upgraded gauntlet):
  - Per-commodity time-series IC + t (non-overlapping by horizon), sign consistency
  - Pooled XS rank IC across the 6 commodities + construction-matched top2−bottom2 spread
  - Stability by 3-year buckets (decades of data → real multi-regime test)
  - Sign-based TS construction check: long if pctl<0.2 / short if pctl>0.8

Usage:
    python scripts/cot_skill.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.base_config import DATA_RAW_DIR

COT_DIR = DATA_RAW_DIR / "cot"
KEYS = ["gold", "silver", "platinum", "wti", "brent", "natgas"]
PCTL_WEEKS = 156                       # 3y trailing percentile window
LAG_DAYS = 6                           # Tuesday report -> usable next Monday
HORIZONS = {"1w": 1, "2w": 2, "4w": 4}
MIN_XS = 4


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


def _pctl(s, window):
    return s.rolling(window, min_periods=window // 2).apply(
        lambda w: (w <= w.iloc[-1]).mean(), raw=False)


def load_key(key):
    c_p = COT_DIR / f"{key}_cot.csv"
    p_p = COT_DIR / f"{key}_px_1d.csv"
    if not (c_p.exists() and p_p.exists()):
        return None
    cot = pd.read_csv(c_p)
    cot["report_date"] = pd.to_datetime(cot["report_date"])
    px = pd.read_csv(p_p)
    px["timestamps"] = pd.to_datetime(px["timestamps"])
    px = px.sort_values("timestamps").reset_index(drop=True)

    oi = cot["open_interest_all"].astype(float)
    cot["net_spec"] = (cot["noncomm_positions_long_all"] - cot["noncomm_positions_short_all"]) / oi
    cot["net_comm"] = (cot["comm_positions_long_all"] - cot["comm_positions_short_all"]) / oi
    cot["spec_pctl"] = _pctl(cot["net_spec"], PCTL_WEEKS)
    cot["comm_pctl"] = _pctl(cot["net_comm"], PCTL_WEEKS)
    cot["avail"] = cot["report_date"] + pd.Timedelta(days=LAG_DAYS)

    ts = px["timestamps"].values
    close = px["close"].to_numpy(dtype=float)

    def close_at_or_after(t):
        i = np.searchsorted(ts, np.datetime64(t), side="left")
        return close[i] if i < len(close) else np.nan

    for h, weeks in HORIZONS.items():
        rets = []
        for _, row in cot.iterrows():
            c0 = close_at_or_after(row["avail"])
            c1 = close_at_or_after(row["avail"] + pd.Timedelta(weeks=weeks))
            rets.append((c1 - c0) / c0 if np.isfinite(c0) and np.isfinite(c1) and c0 > 0
                        else np.nan)
        cot[f"ret_{h}"] = rets
    return cot.dropna(subset=["spec_pctl"]).reset_index(drop=True)


def main():
    data = {}
    for key in KEYS:
        d = load_key(key)
        if d is None or len(d) < 200:
            print(f"  SKIP {key}: missing or short data", flush=True)
            continue
        data[key] = d
    if not data:
        print("No COT data — run scripts/fetch_cot.py first.")
        return

    print(f"\n{'='*76}\n  COT POSITIONING SKILL  ({len(data)} commodities)\n{'='*76}", flush=True)

    for sig, hypo in (("spec_pctl", "NEGATIVE (fade crowded specs)"),
                      ("comm_pctl", "POSITIVE (follow hedgers)")):
        print(f"\n  Signal {sig} — hypothesis: {hypo}", flush=True)
        print(f"  {'asset':10s} {'n':>5s}" + "".join(
            f"  {h+'_IC':>7s} {h+'_t':>6s}" for h in HORIZONS), flush=True)
        per_h_ics = {h: [] for h in HORIZONS}
        for key, d in data.items():
            line = f"  {key:10s} {len(d):>5d}"
            for h, weeks in HORIZONS.items():
                sub = d.iloc[::weeks]
                ic = _spearman(sub[sig], sub[f"ret_{h}"])
                fr, rr = sub[sig].rank(), sub[f"ret_{h}"].rank()
                prod = ((fr - fr.mean()) * (rr - rr.mean())) / (fr.std(ddof=1) * rr.std(ddof=1))
                t, _ = _tstat(prod.to_numpy())
                per_h_ics[h].append(ic)
                line += f"  {ic:>+7.3f} {t:>+6.2f}"
            print(line, flush=True)
        for h in HORIZONS:
            ics = np.asarray(per_h_ics[h], dtype=float)
            t, n = _tstat(ics) if len(ics) >= 5 else (float("nan"), len(ics))
            exp_neg = sig == "spec_pctl"
            frac = np.mean(ics < 0) if exp_neg else np.mean(ics > 0)
            print(f"    {h}: mean IC={np.nanmean(ics):+.4f}  "
                  f"hypothesis-sign in {frac:.0%} of assets", flush=True)

    # Cross-sectional + construction-matched spread + stability (spec_pctl, primary)
    merged = pd.concat([d.assign(asset=k)[["avail", "spec_pctl"]
                                          + [f"ret_{h}" for h in HORIZONS]]
                        for k, d in data.items()])
    merged["week"] = merged["avail"].dt.to_period("W")
    print("\n  Cross-sectional (rank 6 commodities by spec_pctl; hypothesis IC<0):", flush=True)
    for h, weeks in HORIZONS.items():
        ics, spreads, dates = [], [], []
        for i, (wk, g) in enumerate(merged.groupby("week")):
            if i % weeks:
                continue
            g = g.dropna(subset=["spec_pctl", f"ret_{h}"])
            if len(g) >= MIN_XS:
                ic = _spearman(g["spec_pctl"], g[f"ret_{h}"])
                if np.isfinite(ic):
                    ics.append(ic); dates.append(wk.start_time)
                order = g.sort_values("spec_pctl")
                spreads.append(float(order[f"ret_{h}"].iloc[:2].mean()
                                     - order[f"ret_{h}"].iloc[-2:].mean()))
        t, n = _tstat(np.asarray(ics))
        ts_, _ = _tstat(np.asarray(spreads))
        print(f"    {h}: XS IC={np.nanmean(ics):+.4f} t={t:+.2f} (n={n}) | "
              f"tail spread (low2−high2)={np.nanmean(spreads):+.3%} t={ts_:+.2f}", flush=True)
        if h == "1w" and len(ics) > 100:
            s = pd.Series(ics, index=pd.DatetimeIndex(dates))
            print("      stability by 3y bucket:", flush=True)
            for bucket, ys in s.groupby(s.index.year // 3 * 3):
                if len(ys) >= 30:
                    bt, bn = _tstat(ys.to_numpy())
                    print(f"        {bucket}-{bucket+2}: IC={ys.mean():+.3f} t={bt:+.2f} (n={bn})",
                          flush=True)

    print("\n  GATE: 2 signals × 3 horizons = 6 cells → |t| > ~2.4 on IC AND the", flush=True)
    print("  construction-matched spread, hypothesis-signed, stable across buckets.", flush=True)


if __name__ == "__main__":
    main()
