"""Shadow-vs-backtest attribution: is the funding-collection FAIL regime or bug?

The graduation checklist (paper_review.py) compares live funding/event against
the backtest's 5-YEAR mean (+0.0070%/event). But funding-spread levels are
regime-dependent, and 2026 funding is known-low — so a FAIL there is ambiguous:
either the live implementation diverged from the frozen strategy (bug), or the
machine is faithful and the current regime simply pays less (regime).

This tool disambiguates by running the FROZEN backtest (constants imported from
paper_trade_carry — single source of truth) on refreshed data and slicing it to
the exact shadow window, then pairing rows event-by-event:

  - same-window backtest funding/event  ≈ live funding/event  → REGIME (faithful)
  - same-window backtest funding/event >> live funding/event  → BUG (investigate)

Pairing detail: backtest row at event t books the funding accrued at t+8h on the
book formed at t; the live shadow row at event t books funding accrued AT t on
the book formed at t−8h. So backtest rows are shifted +8h before joining.

Read-only: touches no state, changes no strategy code.

Usage (HPC head node, AFTER refreshing data):
    python scripts/fetch_derivs.py --what funding
    python scripts/fetch_derivs.py --what perp
    python scripts/shadow_vs_backtest.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backtest_carry import load_panels, run
from paper_review import load_carry_history
from paper_trade_carry import K, SMOOTH, EXIT_BAND, COST_PER_SIDE, HIST_PATH


def main():
    if not HIST_PATH.exists():
        print("No shadow history — nothing to compare.")
        return
    live = load_carry_history(HIST_PATH)
    live["event"] = pd.to_datetime(live["event"])
    live = live.sort_values("event").set_index("event")

    fund, close = load_panels()
    print(f"Panel: {fund.shape[0]} events × {fund.shape[1]} assets "
          f"({fund.index[0]} .. {fund.index[-1]})", flush=True)
    if fund.index[-1] < live.index[-1] - pd.Timedelta(hours=16):
        print(f"\n  WARNING: derivs data ends {fund.index[-1]} but shadow runs to "
              f"{live.index[-1]} — run fetch_derivs first or the window is truncated.",
              flush=True)

    # Frozen config, full history (so smoothing + incumbency state is warm by
    # the time the shadow window starts), then shift +8h to align accounting.
    bt = run(fund, close, k=K, exit_band=EXIT_BAND, smooth=SMOOTH,
             rebalance_every=1, cost_per_side=COST_PER_SIDE)
    bt = bt.copy()
    bt.index = bt.index + pd.Timedelta(hours=8)

    full_fund_mean = bt["funding_pnl"].mean()

    both = live.join(bt, how="inner", lsuffix="_live", rsuffix="_bt")
    if len(both) < 10:
        print(f"\n  Only {len(both)} overlapping events — check data freshness.")
        return

    print(f"\n{'='*68}\n  SHADOW vs FROZEN BACKTEST — same {len(both)} events "
          f"({both.index[0]:%Y-%m-%d} .. {both.index[-1]:%Y-%m-%d})\n{'='*68}",
          flush=True)

    for col, label in [("funding_pnl", "funding/event"), ("price_pnl", "price/event"),
                       ("net", "net/event"), ("turnover", "turnover/event")]:
        lv = both[f"{col}_live"].astype(float)
        bv = both[f"{col}_bt"].astype(float)
        d = lv - bv
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        t = d.mean() / se if se and se > 0 else float("nan")
        corr = lv.corr(bv)
        print(f"  {label:15s} live {lv.mean():+.4%}  bt(same window) {bv.mean():+.4%}  "
              f"paired-t {t:+.1f}  corr {corr:+.2f}", flush=True)

    lw = both["funding_pnl_live"].astype(float).mean()
    bw = both["funding_pnl_bt"].astype(float).mean()
    print(f"\n  Reference: backtest funding/event FULL SAMPLE {full_fund_mean:+.4%} "
          f"(the checklist constant)", flush=True)
    print(f"{'='*68}", flush=True)
    if abs(lw - bw) < 0.5 * abs(full_fund_mean - bw) or abs(bw - full_fund_mean) > abs(lw - bw):
        print("  READ: same-window backtest is ALSO below the 5-yr constant →\n"
              "  the checklist FAIL is (at least partly) REGIME, not implementation.\n"
              "  Judge fidelity by the paired-t and corr above: |t|<2 and high corr\n"
              "  on funding = the live machine is faithfully executing the strategy.",
              flush=True)
    else:
        print("  READ: same-window backtest collects like the 5-yr constant but the\n"
              "  live book does not → suspect IMPLEMENTATION divergence (universe\n"
              "  drift, data gaps, symbol mapping). Investigate before extending.",
              flush=True)


if __name__ == "__main__":
    main()
