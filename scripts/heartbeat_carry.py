"""Heartbeat + tripwire monitor for the shadow carry book.

Protects the 30-day shadow run from its two silent failure modes:
  1. STALENESS — the paper-trade cron died and nobody noticed: alert if the
     newest row in carry_history.csv is older than 9 hours (events are 8h apart).
  2. TRIPWIRE — equity dropped >10% from its trailing-30-day peak (a ~2-sigma
     event vs the backtest distribution): alert + investigate before trusting
     anything else.

Alerts go to ntfy.sh — zero-signup push notifications. Setup (one time):
  1. Pick a hard-to-guess topic name, e.g. cifr-carry-x7k2m9 (topics are public
     namespaces — the name IS the secret).
  2. On your phone/browser: install ntfy / open https://ntfy.sh/<topic> and
     subscribe.
  3. Put the topic in the cron line via NTFY_TOPIC (see below).

Cron (hourly, on the half-hour so it never races the 8h trader):
  30 * * * * cd /home/s3702111/cifr-quant && NTFY_TOPIC=<topic> /home/s3702111/.conda/envs/trade/bin/python scripts/heartbeat_carry.py >> slurm/logs/heartbeat.log 2>&1

A 'heartbeat OK' line is logged every run; alerts are pushed only on problems
(re-alerts are rate-limited to once per 8h per condition via a state file).
"""

import os
import sys
import json
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.base_config import RESULTS_DIR

HIST_PATH = RESULTS_DIR / "paper" / "carry_history.csv"
ALERT_STATE = RESULTS_DIR / "paper" / "heartbeat_state.json"
STALE_AFTER_H = 9
TRIPWIRE_DD = 0.10
REALERT_AFTER_H = 8


def push(topic: str, title: str, message: str) -> bool:
    try:
        req = urllib.request.Request(
            f"https://ntfy.sh/{topic}",
            data=message.encode(),
            headers={"Title": title, "Priority": "high", "Tags": "warning"},
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status == 200
    except Exception as e:
        print(f"  ntfy push failed: {e}", flush=True)
        return False


def should_alert(state: dict, key: str, now: datetime) -> bool:
    last = state.get(key)
    if last is None:
        return True
    return (now - datetime.fromisoformat(last)) > timedelta(hours=REALERT_AFTER_H)


def main():
    topic = os.environ.get("NTFY_TOPIC", "").strip()
    now = datetime.now(timezone.utc)
    stamp = now.isoformat(timespec="seconds")

    state = json.loads(ALERT_STATE.read_text()) if ALERT_STATE.exists() else {}
    problems = []

    if not HIST_PATH.exists():
        problems.append(("no_history", "Shadow book has no history file",
                         f"{HIST_PATH} does not exist — has the paper trader ever run?"))
    else:
        import pandas as pd
        from paper_review import load_carry_history
        hist = load_carry_history(HIST_PATH)
        last_run = pd.Timestamp(hist["run_at"].iloc[-1])
        if last_run.tzinfo is None:
            last_run = last_run.tz_localize("UTC")
        age_h = (now - last_run.to_pydatetime()).total_seconds() / 3600
        if age_h > STALE_AFTER_H:
            problems.append(("stale", "Shadow carry book is STALE",
                             f"Last paper-trade run was {age_h:.1f}h ago "
                             f"(threshold {STALE_AFTER_H}h). Check the cron job."))

        eq = hist["equity"].astype(float)
        recent = eq.iloc[-90:]                       # trailing ~30 days (3/day)
        peak = recent.max()
        dd = (peak - eq.iloc[-1]) / peak if peak > 0 else 0.0
        if dd > TRIPWIRE_DD:
            problems.append(("tripwire", "Shadow carry book hit the DD tripwire",
                             f"Equity {eq.iloc[-1]:.4f} is {dd:.1%} below its trailing-30d "
                             f"peak {peak:.4f} (threshold {TRIPWIRE_DD:.0%}). Halt and "
                             f"investigate before trusting the strategy."))

        if not problems:
            print(f"[{stamp}] heartbeat OK  last_run={last_run.isoformat()}  "
                  f"equity={eq.iloc[-1]:.4f}  rows={len(hist)}", flush=True)

    for key, title, msg in problems:
        print(f"[{stamp}] PROBLEM {key}: {msg}", flush=True)
        if topic and should_alert(state, key, now):
            if push(topic, title, msg):
                state[key] = now.isoformat()
                print(f"  alert pushed to ntfy.sh/{topic}", flush=True)
        elif not topic:
            print("  (NTFY_TOPIC not set — logged only)", flush=True)

    ALERT_STATE.parent.mkdir(parents=True, exist_ok=True)
    ALERT_STATE.write_text(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
