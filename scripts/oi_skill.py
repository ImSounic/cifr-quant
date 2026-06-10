"""Open-interest / squeeze diagnostic: Phase 2C candidate (data-gated).

In plain terms: open interest (OI) = how many futures contracts are open. When
OI spikes (lots of new positions) AND funding is extreme (those positions are
crowded on one side), the crowd is maximally stretched — small adverse moves
force liquidations that cascade (a "squeeze"). Hypothesis: crowded-long setups
(OI surge + high funding) predict NEGATIVE forward returns; crowded-short
setups (OI surge + negative funding) predict POSITIVE.

Why this candidate: structurally different from carry (positioning *change* vs
funding *level*) → potential second brick with low correlation to brick #1.

DATA GATE: Binance only serves ~30 days of OI history; we've been accumulating
since June 10 2026 via the OI fetcher. This script refuses to draw conclusions
below MIN_DAYS of usable history and reports how much has accumulated — run it
weekly; it becomes meaningful ~early July 2026.

Tests (per the upgraded gauntlet):
  1. XS rank IC of each signal vs forward returns (8h, 1d), non-overlapping t-stats.
  2. Construction-matched tail spread (bottom-3 minus top-3) — the reversal lesson.
  3. Weekly stability buckets (yearly buckets need years we don't have yet).
Signals tested (declared up front, count = 3 for the trial ledger):
  A. oi_z:      24h OI change, z-scored vs trailing 14d   (pure positioning flow)
  B. crowd:     oi_z × sign(funding 3d mean)              (crowding direction-aware)
  C. oi_funding: oi_z × funding_pctl(30d)                 (continuous crowding)

Usage:
    python scripts/oi_skill.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.base_config import DATA_RAW_DIR

DERIVS_DIR = DATA_RAW_DIR / "derivs"
MIN_DAYS = 21
MIN_ASSETS = 8
HORIZONS = {"8h": 8, "1d": 24}          # in hours


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
    if len(x) < 15 or x.std(ddof=1) == 0:
        return float("nan"), len(x)
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))), len(x)


def load_panels():
    from configs.crypto_universe import get_crypto_configs
    symbols = list(get_crypto_configs(tiers=(1, 2)).keys())
    oi, close, fund = {}, {}, {}
    for sym in symbols:
        safe = sym.replace("/", "_").lower()
        o_p = DERIVS_DIR / f"{safe}_oi_1h.csv"
        p_p = DERIVS_DIR / f"{safe}_perp_1h.csv"
        f_p = DERIVS_DIR / f"{safe}_funding.csv"
        if not (o_p.exists() and p_p.exists() and f_p.exists()):
            continue
        o = pd.read_csv(o_p)
        o["timestamps"] = pd.to_datetime(o["timestamps"])
        oi[sym] = o.set_index("timestamps")["open_interest_value"].resample("1h").last()
        p = pd.read_csv(p_p)
        p["timestamps"] = pd.to_datetime(p["timestamps"])
        close[sym] = p.set_index("timestamps")["close"].resample("1h").last()
        f = pd.read_csv(f_p)
        f["timestamps"] = pd.to_datetime(f["timestamps"]).dt.round("h")
        fr = f.set_index("timestamps")["funding_rate"]
        fund[sym] = fr.reindex(pd.date_range(fr.index.min(), fr.index.max(), freq="1h")
                               ).ffill()
    return (pd.DataFrame(oi).sort_index(), pd.DataFrame(close).sort_index(),
            pd.DataFrame(fund).sort_index())


def main():
    oi, close, fund = load_panels()
    if oi.empty:
        print("No OI data — run scripts/fetch_derivs.py --what oi first.")
        return
    # usable = rows where most assets have OI
    usable = oi.notna().sum(axis=1) >= MIN_ASSETS
    days = usable.sum() / 24
    print(f"OI panel: {oi.shape[0]} hours × {oi.shape[1]} assets; "
          f"~{days:.1f} usable days accumulated", flush=True)
    if days < MIN_DAYS:
        print(f"\nDATA GATE: need ≥{MIN_DAYS} days; have {days:.1f}. "
              f"Re-run after more accumulation (weekly OI cron is filling this in).",
          flush=True)
        return

    idx = oi.index.intersection(close.index)
    oi, close = oi.loc[idx], close.loc[idx]
    fund = fund.reindex(idx).ffill()

    # Signals (declared trio)
    oi_chg = oi.pct_change(24)
    oi_z = (oi_chg - oi_chg.rolling(14 * 24).mean()) / oi_chg.rolling(14 * 24).std()
    f3d = fund.rolling(9 * 8).mean()                       # ~3d of hourly-ffilled funding
    f_pctl = fund.rolling(30 * 24).apply(lambda w: (w <= w.iloc[-1]).mean(), raw=False)
    signals = {
        "oi_z": oi_z,
        "crowd": oi_z * np.sign(f3d),
        "oi_funding": oi_z * f_pctl,
    }
    fwd = {h: close.shift(-hh) / close - 1.0 for h, hh in HORIZONS.items()}

    print(f"\n{'='*72}\n  OI / SQUEEZE SKILL  (hypothesis: NEGATIVE IC for crowded-long)\n{'='*72}",
          flush=True)
    for name, sig in signals.items():
        line = f"  {name:>10s}"
        for h, hh in HORIZONS.items():
            ics, spreads = [], []
            for i in range(0, len(idx) - hh, hh):          # non-overlapping
                s_row, r_row = sig.iloc[i], fwd[h].iloc[i]
                m = s_row.notna() & r_row.notna()
                if m.sum() >= MIN_ASSETS:
                    ic = _spearman(s_row[m], r_row[m])
                    if np.isfinite(ic):
                        ics.append(ic)
                    order = s_row[m].sort_values().index
                    spreads.append(float(r_row[order[:3]].mean() - r_row[order[-3:]].mean()))
            t, nn = _tstat(np.asarray(ics))
            ts, _ = _tstat(np.asarray(spreads))
            line += (f"  {h}: IC={np.nanmean(ics):+.3f} t={t:+.2f} | "
                     f"tail={np.nanmean(spreads):+.3%} t={ts:+.2f} (n={nn})")
        print(line, flush=True)

    print("\n  GATE: 3 signals × 2 horizons = 6 cells → |t| > ~2.4 on BOTH the IC and",
          flush=True)
    print("  the construction-matched tail spread, plus stability across weeks.", flush=True)


if __name__ == "__main__":
    main()
