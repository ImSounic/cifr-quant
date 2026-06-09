# CIFR-QUANT: Project State

> **Purpose**: Cross-session memory for Claude. Read this file at the start of every new session.
> **Last updated**: June 8, 2026

---

## Quick Summary

Dual-market multi-asset algorithmic trading system using the Kronos financial foundation model (102M params, AAAI 2026). Two markets: **Crypto** (15-25 assets via Binance, 15m) and **Commodities** (6-8 assets via TwelveData, 4h). BTC-finetuned checkpoint transfers to all crypto; XAU-finetuned checkpoint transfers to all commodities. Ensemble predictions + CQR-calibrated risk parity position sizing. User is Raj (s3702111), UTwente student, building a quant firm.

---

## Current Status: Phase 4 — CQR Calibration Running, BTC v2 Complete

### Strategic Pivot (June 6-7, 2026)

Pivoted from **3-market × 1-asset** (BTC, EUR, XAU) to **2-market × many-assets** (Crypto + Commodities). Rationale:
- Kronos pre-training data heavily upweighted crypto (45 exchanges)
- XAU ensemble showed strong IC (+0.393), validating commodity finetuning transfer
- EUR/Forex dropped — no volume data, weekend gaps, weaker model fit
- Crypto × Commodities are decorrelated (24/7 momentum vs macro-driven)
- Breadth: 25-35 assets vs 3 dramatically improves portfolio IR via Grinold's Law

### All Finetuning Complete

| Market | Version | Val Loss | Time | Status |
|--------|---------|----------|------|--------|
| BTC/USDT | v1 (tok+pred) | tok: 0.0027 | tok: 2.6h, pred: ~2h | ✅ Complete |
| BTC/USDT | v2 (seed=137) | pred: 2.6672 | 9.6h (tok+pred) | ✅ Complete |
| EUR/USD | v1 (seed=42) | 1.6098 | 1h12m | ✅ Complete |
| EUR/USD | v2 (seed=137) | 1.5521 | 67min | ✅ Complete |
| XAU/USD | v1 | 0.9054 | 1h39m | ✅ Complete (overfitted) |
| XAU/USD | v2 (anti-overfit) | 1.2316 | 26min | ✅ Complete |

### Multi-Asset Data Fetched (June 8, 2026)

| Market | Assets | Candles Each | Source |
|--------|--------|-------------|--------|
| Crypto (15m) | 15 assets (tier 1+2) | ~155k (4.5 years) | Binance |
| Commodities (4h) | XAU: 9,584 | 6 years | TwelveData |
| Commodities (4h) | XAG, XPT, WTI, Brent, NatGas, Copper | ~2,800 each (2 years) | yfinance futures |

Data saved to `data/raw/crypto/` and `data/raw/commodity/`.

### CQR Calibration (June 8, 2026)

SLURM job 510442 submitted on L40S. Script: `scripts/calibrate_cqr_multi.py`. Runs ensemble predictions across all 21 assets with 50 MC paths, computes conformity scores for 90% coverage. Results save to `results/cqr/cqr_calibrations.json`.

**CHECK THIS FIRST** in next session — if `results/cqr/cqr_calibrations.json` is empty `{}`, the job failed. See Common Pitfalls below.

### Ensemble Evaluation Complete (50 windows, bootstrap CIs)

| Market | Best Model | Key Metric | Value |
|--------|-----------|------------|-------|
| XAU/USD | ensemble_full (ZS+v1+v2) | IC | **+0.393** |
| EUR/USD | ensemble_eq / ensemble_full | Dir. Accuracy | **60%** |
| BTC/USDT | finetuned_v1 | RMSE improvement | Yes, but IC/DA flat |

**Key finding**: XAU ensemble_full is the strongest signal. BTC v2 now complete (val_loss=2.6672) — crypto ensemble now has 3 models (ZS + BTC v1 + BTC v2).

---

## Dual-Market Multi-Asset Architecture

### Market Universes

**Crypto (Binance, 15m candles)**:
- Tier 1 (5 majors): BTC, ETH, BNB, SOL, XRP
- Tier 2 (10 large caps): ADA, AVAX, DOGE, DOT, LINK, MATIC, UNI, ATOM, LTC, NEAR
- Tier 3 (10 mid caps): APT, ARB, FIL, INJ, OP, SUI, TIA, SEI, AAVE, MKR
- Default: Tier 1+2 (15 assets). All use BTC-finetuned + zero-shot ensemble.

**Commodities (TwelveData, 4h candles)**:
- Precious: XAU/USD, XAG/USD, XPT/USD
- Energy: WTI, Brent, Natural Gas
- Industrial: Copper
- All use XAU-finetuned v1+v2 + zero-shot ensemble (3 models).

### Transfer Learning Strategy

BTC finetuning captures crypto-specific patterns (momentum cascades, leverage liquidations, 24/7 microstructure) → transfers to all crypto.
XAU finetuning captures commodity macro dynamics (real rates, central bank flows, risk-off) → transfers to all commodities.

### Portfolio Orchestrator (`src/portfolio/orchestrator.py`)

Core engine: `PortfolioOrchestrator.build()` loads shared zero-shot + market-specific finetuned models. Per-asset: 50 Monte Carlo paths → CQR-calibrated intervals → directional confidence + stop/take-profit → risk parity sizing (1/interval_width). Max 10% per position.

### Key Files Created for Multi-Asset

| File | Purpose |
|------|---------|
| `configs/crypto_universe.py` | 3-tier crypto asset definitions, costs, MarketConfig builder |
| `configs/commodity_universe.py` | Precious/energy/industrial definitions, costs, MarketConfig builder |
| `src/portfolio/__init__.py` | Portfolio module init |
| `src/portfolio/orchestrator.py` | PortfolioOrchestrator: multi-asset prediction + risk parity allocation |
| `scripts/fetch_universe.py` | Multi-asset data fetcher (Binance for crypto, TwelveData for commodities) |

### Critical Kronos Bug

`predict(sample_count=N)` **AVERAGES** N paths into one DataFrame. Must call `predict(sample_count=1)` N times for individual trajectories needed for quantile analysis.

### Checkpoint Path Issue (Fixed)

Training saves to `Kronos/checkpoints/` (because script cd's into `Kronos/finetune_csv/`). Loader expects `checkpoints/`. Fixed with symlink on HPC: `ln -s Kronos/checkpoints checkpoints`.

---

## HPC Environment

- **Cluster**: UTwente SLURM, head nodes `hpc-head1/2.ewi.utwente.nl`
- **Best GPUs in main-gpu**: Lovelace (L40S, 48GB) on hpc-node01-18
- **Conda env**: `trade` (has torch, transformers, etc.)
- **HuggingFace**: Model cached on head node, jobs use `HF_HUB_OFFLINE=1`
- **Conda activation**: `source $(conda info --base)/etc/profile.d/conda.sh && conda activate trade` (the direct path doesn't work)
- **QOS limit**: May restrict concurrent GPU jobs. Submit BTC first if blocked.

---

## Development Environment

Local dev/inference/backtest runs on the dev machine; all training/calibration runs on the HPC.

### HP Windows 11 laptop (current dev machine, set up June 8, 2026)

- **Repo**: `W:\cifr-quant` (W: is a `subst` of `C:\Work Drive`).
- **Python env**: system-wide venv `mlenv` at `C:\Work Drive\envs\mlenv`. Activate in PowerShell by typing `mlenv` (a profile function running `Activate.ps1`). Python 3.12.10.
- **GPU**: NVIDIA RTX PRO 1000 (Blackwell) Laptop GPU — `torch` cu128 build, `cuda.is_available() == True`. Local GPU inference works.
- **Deps**: `mlenv` ships torch/transformers/pandas/numpy/scipy/einops/numba/matplotlib/plotly/wandb/tqdm. Added for this project: `ccxt yfinance twelvedata anthropic python-dotenv`. **`vectorbt` and `mapie` are NOT installed and NOT needed** — the backtest engine (`src/backtest/`) and CQR (`src/risk/cqr.py`) are custom implementations; installing them would risk downgrading numpy 2.4.3 / pandas 3.0.1 in the shared venv.
- **Kronos**: tracked as a gitlink (mode 160000, commit `67b630e`) with **no `.gitmodules`**, so a fresh clone leaves `Kronos/` empty. Restore with: `git clone https://github.com/shiyu-coder/Kronos.git Kronos && cd Kronos && git checkout 67b630e`.
- **Run commands** with `PYTHONPATH` set to repo root + `Kronos`, e.g. PowerShell: `$env:PYTHONPATH="W:\cifr-quant;W:\cifr-quant\Kronos"`.
- **Not present locally** (gitignored, live on HPC): `data/` and `checkpoints/`. Sync from HPC before any local inference/backtest. Kronos-base/Tokenizer-base weights download from HuggingFace on first use (laptop has internet — do NOT set `HF_HUB_OFFLINE` locally).

### New-laptop setup checklist

1. `git clone https://github.com/ImSounic/cifr-quant.git`
2. Restore Kronos (see clone command above).
3. Activate `mlenv`; `pip install ccxt yfinance twelvedata anthropic python-dotenv` (rest already in mlenv).
4. Sync `data/` and `checkpoints/` from HPC (`~/cifr-quant/`) if running locally.
5. Create `.env` with `TWELVEDATA_API_KEY`, `ANTHROPIC_API_KEY` (see PROJECT_PLAN Environment Variables).

---

## File Map

### Configs

| File | Purpose |
|------|---------|
| `finetune/config_btc.yaml` | BTC: batch=128, tok=20 epochs, pred=10, LR=5e-6, skip_existing=true |
| `finetune/config_eur.yaml` | EUR v1: batch=32, tok=30, pred=15, LR=3e-6, seed=42 |
| `finetune/config_eur_v2.yaml` | EUR v2: batch=64, same hyperparams, seed=137 |
| `finetune/config_xau.yaml` | XAU v1: batch=32, tok=50, pred=20 — **OVERFITTED** |
| `finetune/config_xau_v2.yaml` | XAU v2: batch=64, tok=15, pred=5, LR halved, decay=0.15 |
| `configs/base_config.py` | `CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"` |

### SLURM Scripts

| File | Target GPU | Time Limit |
|------|-----------|------------|
| `slurm/finetune_btc.sh` | Ampere (A40) | 20h |
| `slurm/finetune_eur.sh` | A40 (ampere) — **outdated** | 8h |
| `slurm/finetune_eur_v2.sh` | Ampere (A40) | 8h |
| `slurm/finetune_xau.sh` | A40 (ampere) — **outdated** | 8h |
| `slurm/finetune_xau_v2.sh` | Ampere (A40) | 8h |
| `slurm/eval_ensemble.sh` | Ampere (A40) | 4h |

### Evaluation Scripts

| File | Purpose |
|------|---------|
| `scripts/zero_shot_baseline.py` | 20 rolling windows, IC/RankIC/dir accuracy/RMSE/MAE |
| `scripts/finetuned_eval.py` | Same metrics for finetuned checkpoints, `--compare` flag |
| `scripts/eval_ensemble.py` | 50+ windows, bootstrap 95% CIs, tests ZS/FT_v1/FT_v2/ensemble_eq/ensemble_full |

### Core Source

| File | Purpose |
|------|---------|
| `src/model/loader.py` | Load zero-shot or finetuned models. Maps market→exp_name (btc→cifr-btc) |
| `src/model/ensemble.py` | `EnsemblePredictor` — weighted average, Monte Carlo paths, quantile bands |
| `src/model/predictor.py` | KronosPredictor wrapper |
| `src/model/sampler.py` | Multi-path Monte Carlo sampling |
| `src/risk/cqr.py` | Conformalized Quantile Regression |
| `src/risk/quantile.py` | Empirical quantile extraction |
| `src/risk/position_sizer.py` | Adaptive position sizing |
| `src/backtest/engine.py` | Walk-forward backtest runner |
| `src/backtest/costs.py` | Transaction cost models |
| `src/backtest/metrics.py` | Sharpe, drawdown, Calmar, etc. |
| `src/strategy/llm_generator.py` | LLM strategy code generation |
| `src/strategy/executor.py` | Strategy execution engine |
| `src/strategy/context_builder.py` | Build context packs for LLM |

### Data

| Path | Contents |
|------|----------|
| `data/processed/btc/` | ~97k 15m candles, split into tokenizer_train.csv, validation.csv, test.csv |
| `data/processed/eur/` | ~23k 1h candles, same splits |
| `data/processed/xau/` | ~9.5k 4h candles, same splits |

---

## What's Next (In Order)

1. ✅ **Fetch multi-asset data** — 15 crypto (Binance) + 7 commodity (TwelveData/yfinance) done
2. ✅ **Create BTC v2 config + train** — seed=137, val_loss=2.6672, complete
3. 🔄 **CQR calibration** — Job 510442 (June 8) **TIMED OUT** at 12h with empty `{}` (crypto leg too heavy + results only written at end). Script now saves per-asset incrementally and resumes. **Run order**: `sbatch slurm/cqr_calibrate_commodity.sh` (fast), then `sbatch slurm/cqr_calibrate_crypto.sh` (resubmit to resume if it hits 12h). Accumulates in `results/cqr/cqr_calibrations.json`.
4. ✅ **Multi-asset walk-forward backtest** — `scripts/backtest_portfolio.py` + `src/backtest/portfolio_engine.py` + `src/model/build.py`. Ran June 9 (job 511139). **Results below.**
5. ⏳ **Regime detection + strategy redesign** — HIGHEST IMPACT. Backtest proved the current single directional strategy only works in trends. See "Strategy Roadmap" below.
6. ⏳ **LLM strategy layer** — Generate and evaluate strategy code
7. ⏳ **Final test set evaluation** — Touched exactly ONCE

---

## Backtest Results (June 9, 2026 — job 511139)

90-day walk-forward (2026-03 → 2026-06), $100k split 50/50, long & short, CQR-gated, batched MC paths. **Ran clean, full coverage, MATIC correctly excluded (data ends 2024).**

| Market | Return | Sharpe | Calmar | Max DD | Win% | PF | Trades |
|--------|--------|--------|--------|--------|------|----|--------|
| Crypto (15m) | **−23.4%** | **−3.28** | −2.46 | −26.9% | 46.6% | 0.81 | 2358 |
| Commodity (4h) | **+16.0%** | **+4.36** | 16.5 | −5.0% | 51.7% | 1.52 | 331 |
| **Combined** | **−5.1%** | **−1.20** | −1.50 | −12.6% | 47.3% | 0.91 | 2689 |

**Verdict:** Crypto is a decisive loser (double-negative: <50% win rate AND avg loss > avg win; likely tripped the 25% DD halt). Commodity looks excellent **but is unverified** — almost certainly trend-regime luck (energy trended hard: Crude +$2.5k, Brent +$3.1k) on a single window + small sample; Sharpe 4.36 is not believable as a sustained number. Same engine + same CQR on both markets → the asymmetry is **regime/market-structure driven, not a code bug**.

### Strategy we ran (the ONLY one so far)
Single **probabilistic-directional** strategy: ensemble of 30 Kronos MC paths → majority vote = direction + confidence; gate at conf ≥ 0.55 (too weak, ~16/30); SL/TP = CQR-widened q05/q95 band; inverse-interval-width risk parity (10% cap); intrabar SL/TP else timeout. Always-in when confident, symmetric long/short, single horizon.

---

## Strategy A/B Sweep #1 (June 9 2026, off forecast cache)

Refactor verified faithful: baseline reproduced crypto −23.7% / commodity +14.1%.
Then swept 4 strategies CPU-only off `results/forecasts/` (instant, no GPU):

| Strategy | Crypto | Commodity | Crypto exit mix (tp/sl/timeout) |
|----------|--------|-----------|---------------------------------|
| directional (baseline) | −23.7% | **+14.1%** | 0 / 31 / 2341 |
| regime_gated_trend strict | −2.1% (36 tr) | **0 trades** | 0 / 0 / 36 |
| regime_gated_trend relaxed | −5.8% (108 tr) | **0 trades** | 0 / 0 / 108 |
| mean_reversion (ATR exits) | −6.2% | +1.8% (Sh 1.20) | **106 / 195 / 25** |

**Two hard findings:**
1. **CQR band is INERT as SL/TP** — 0 take-profits in 2372 crypto trades; every
   trade rides 12h to timeout. The band is calibrated for distributional
   coverage, far too wide to trigger intrabar. Directional = pure horizon bet.
2. **Hurst+ADX regime gating is a NET NEGATIVE** — it makes 0 commodity trades
   (kills the only profitable book) because Hurst never labels trending energy
   as "trend." The signed-off composite is empirically wrong for these markets.
   ADX-only or vol-only gating may still be worth a look; Hurst is the problem.
3. **ATR exits WORK** — mean_reversion's ATR stops produced real tp/sl fills
   (106/195/25). Direction was wrong for crypto, but exits fired. → Phase C.

**Next experiment (Phase C, building):** `directional_atr` = winning direction
(momentum) + working exits (ATR TP/SL, RR 1.5) — the untested winning-direction
+ working-exits cell. Knobs: `--stop-atr-mult`, `--tp-atr-mult`.

## Strategy Roadmap (research-driven, June 9 2026)

Backtest + literature (López de Prado, regime-filter and vol-targeting research) point to four high-leverage changes. **Build as a pluggable strategy interface so we can backtest many and compare — don't hardcode one.**

1. **Regime filter (biggest lever).** Trend systems fail in ranges; MR fails in trends — exactly our crypto-vs-commodity split. Add Hurst exponent / ADX / realized-vol classifier per asset per rebalance. Only allow directional/momentum trades in trending regimes; switch to mean-reversion (or stand aside) in chop. This alone likely fixes crypto's bleed.
2. **Meta-labeling (López de Prado).** Keep Kronos as the *primary* (direction) model; train a secondary binary classifier (RF/GBM on vol, momentum, autocorr, RSI, spread, confidence) that decides **trade vs skip**. Documented lift: precision/accuracy up, false positives down — directly targets our 46.6% crypto win rate. Replaces the naive 0.55 confidence gate.
3. **Volatility-targeted stops & sizing.** Replace static CQR-band SL/TP with **ATR-based stops (1.5–2× ATR)** — research shows ~35% fewer premature stop-outs vs flat rules. Size to a **volatility target** (and/or fractional/half-Kelly using the model's own win-rate & payoff estimate), not just inverse interval width. "Vol is more predictable than direction" — lean on it.
4. **Strategy library to A/B test.** Candidates to implement behind one interface and run through `portfolio_engine`: (a) directional-momentum [current], (b) mean-reversion on the CQR band (fade q05/q95 touches), (c) breakout/trend-following gated by regime, (d) market-neutral long-top / short-bottom cross-sectional rank by expected return, (e) meta-labeled version of each. Compare on Sharpe, deflated-Sharpe, trade count, exit mix.

**Caveats to enforce next:** multi-window walk-forward (not one 90-day window), deflated Sharpe / multiple-testing haircut, and a flat/non-trending validation window to test the commodity regime-luck hypothesis before trusting +16%.

---

## Research Findings (Quant Firm Research, June 6 2026)

### Key Takeaways

1. **No public profitable Kronos trading results** — paper acknowledges backtesting is "simplified example, not production-ready"
2. **Top firms (RenTech, Two Sigma, Citadel)** use ensemble of weak uncorrelated signals, not single strong models
3. **Renaissance Medallion** ~35% annualized after fees — uses ensemble methods, alternative data, regime detection
4. **What kills ML trading**: overfitting, ignoring regime changes, not accounting for execution costs
5. **Realistic targets**: Sharpe 1.0-1.5 excellent, 2.0+ exceptional for retail quant

### Actionable Recommendations Applied

- Ensemble architecture (ZS + multi-seed FT) ✅
- Anti-overfitting for XAU (v2 config) ✅ 
- CQR for uncertainty quantification (code exists, needs calibration) ⏳
- Regime detection (not yet implemented) ❌
- Walk-forward backtest with costs (code exists, needs running) ⏳

---

## Common Pitfalls & Fixes

| Problem | Cause | Fix |
|---------|-------|-----|
| `FileNotFoundError` on data | Relative paths resolve wrong from `Kronos/finetune_csv/` | Use absolute paths in YAML: `/home/s3702111/cifr-quant/data/processed/...` |
| HuggingFace 401 | Wrong repo name or no auth | Repo is `NeoQuasar/Kronos-base` (not Tsinghua-AIR). Use `hf auth login` with read token |
| HuggingFace httpx closed | Compute nodes can't reach internet | Cache on head node with `snapshot_download()`, use `HF_HUB_OFFLINE=1` |
| Checkpoints not found | Training saves to `Kronos/checkpoints/`, loader looks in `checkpoints/` | Symlink: `ln -s Kronos/checkpoints checkpoints` |
| Conda activation fails | Direct path doesn't exist | Use: `source $(conda info --base)/etc/profile.d/conda.sh && conda activate trade` |
| XAU overfitting | 50 tokenizer + 20 predictor epochs on 9.5k candles | v2: 15+5 epochs, lower LR, higher weight decay |
| BTC killed at time limit | 8h too short for predictor training | Updated to 20h, `skip_existing: true` skips completed tokenizer |
| CQR job TIMEOUT, empty `{}` results (job 510442) | Crypto leg ~40h (15 assets × ~179 windows × ~48 MC calls × 48 steps) >> 12h wall, and results were only written at the very end so timeout lost everything. tee log also never appeared (`slurm/logs/` didn't exist) | `calibrate_cqr_multi.py` now saves per-asset incrementally + resumes (skips assets already in JSON; `--force` to redo). Split into `slurm/cqr_calibrate_commodity.sh` (cheap) + `slurm/cqr_calibrate_crypto.sh` (n_paths 30, `--step-size 96`, resubmit to resume). All SLURM scripts `mkdir -p logs` before tee |
| SLURM logs not visible | Compute nodes have delayed NFS sync | Logs appear only AFTER job completes. Use `tee` inside script for real-time, or `sacct -j JOBID` to check status |
| CQR tqdm error | Old tqdm version on HPC | Don't use `miniinterval` kwarg in tqdm constructor |
| TwelveData commodity failures | Free tier only supports XAU/USD | Use `scripts/fetch_commodities_yf.py` for other commodities via yfinance futures |
| Git merge conflicts on HPC | Local edits on HPC diverge from Mac | `git stash && git pull --rebase && git stash pop`, or `git checkout --theirs <file>` |

---

## Development Environment

### Primary: HP Laptop (as of June 8, 2026)

- Clone: `git clone https://github.com/ImSounic/cifr-quant.git`
- Python env: `conda create -n trade python=3.11 && conda activate trade`
- Install: `pip install torch transformers pandas numpy ccxt yfinance twelvedata tqdm scipy`
- Kronos submodule: `cd Kronos && pip install -e .` (or ensure Kronos/ is on PYTHONPATH)
- HuggingFace: `pip install huggingface_hub && huggingface-cli login` (read token)
- Data: `data/` is gitignored. Re-fetch with `python scripts/fetch_universe.py --market crypto --tiers 1 2`
- Checkpoints: `checkpoints/` is gitignored. For inference, download from HPC: `scp -r s3702111@hpc-head1.ewi.utwente.nl:~/cifr-quant/Kronos/checkpoints ./checkpoints/`
- SSH to HPC: `ssh s3702111@hpc-head1.ewi.utwente.nl`

### Training: UTwente HPC (SLURM)

- All finetuning and GPU-heavy eval runs on HPC
- Checkpoints live at `~/cifr-quant/Kronos/checkpoints/` on HPC, symlinked to `~/cifr-quant/checkpoints/`
- Use L40S (Lovelace) GPUs: `--gres=gpu:lovelace:1`

---

*This file is the single source of truth for project state. Update after every significant change.*
