# Operations Card — what runs, when, and how to check it

> Day-to-day reference. All times are **CEST** (the HPC head node's clock).
> Funding events are 00/08/16 UTC = 02/10/18 CEST.

## The schedule (crontab on hpc-head2, user s3702111)

| Time (CEST) | Job | Log |
|---|---|---|
| 02:05, 10:05, 18:05 daily | Shadow book trades (paper money) | `slurm/logs/paper_carry.log` |
| 02:10, 10:10, 18:10 daily | OKX demo book trades (real practice orders) | `slurm/logs/okx_demo_carry.log` |
| :30 every hour | Heartbeat watchdog → phone alert on problems | `slurm/logs/heartbeat.log` |
| Sunday 03:00 | OKX funding history archive | `slurm/logs/fetch_okx_funding.log` |

Phone alerts: ntfy app, topic `cifr-carry-rj76x2`. **Silence = healthy.**
An alert means: book stale >9h (cron died) or equity −10% from peak (halt & investigate).

## The routine

### Every few days (~30 seconds) — SSH to hpc-head2, then:
```bash
cd ~/cifr-quant
tail -3 slurm/logs/paper_carry.log slurm/logs/okx_demo_carry.log
```
Healthy looks like: one line per 8h cycle, recent timestamps, `net=` small numbers,
OKX line showing `fill_rate(last)=…%`.

### Weekly (~2 minutes)
```bash
# on the HPC — full status + early checklist read:
cd ~/cifr-quant && conda activate trade
python scripts/paper_review.py
column -s, -t < results/paper/carry_history.csv | tail -6
```
```powershell
# on the laptop (PowerShell) — backup the irreplaceable data:
scp -r s3702111@hpc-head2.ewi.utwente.nl:cifr-quant/results/paper W:\backups\cifr-quant\
scp -r s3702111@hpc-head2.ewi.utwente.nl:cifr-quant/data/raw/derivs_okx W:\backups\cifr-quant\
```

### Scheduled reviews (paste output to Claude)
- **~June 24** (day 14): `python scripts/paper_review.py` — funding collection,
  turnover, fill rates vs backtest.
- **~July 10** (day 30): same command — graduation decision (checklist, NOT the
  PnL sign). If PASS → live-capital conversation (venue: OKX; first: VPS migration).
- **Weekly from ~end June**: `python scripts/oi_skill.py` — reports its own data
  gate until OI history is thick enough, then delivers the next signal verdict.

## If something looks wrong

| Symptom | First move |
|---|---|
| Phone alert: STALE | `crontab -l` (still installed?), then `tail -20` the affected log for the error |
| Phone alert: TRIPWIRE | Do NOT tweak anything. `crontab -e`, comment out lines 1–2 (halt books), then full review with Claude |
| Log shows repeated `skip <SYM>` | Usually a venue/API hiccup for one asset — fine if others proceed |
| OKX `posSide` error | Demo account flipped to hedge mode — set One-way in demo trade settings |
| No new OKX fills reconciled | Check the demo account didn't expire/reset; rerun smoke test manually with keys |

## Standing rules
- The strategy config is **FROZEN** (constants in `scripts/paper_trade_carry.py`).
  Any "small improvement" = a new strategy that must re-earn its gate. Don't.
- Judge the experiment by the **checklist**, never by whether 2 weeks of PnL is up.
- Cron lines contain demo/testnet keys only — never put real-money keys in crontab.
