# CIFR-QUANT: Project State

> **Purpose**: Cross-session memory for Claude. Read this file at the start of every new session.
> **Last updated**: June 6, 2026

---

## Quick Summary

Multi-market algorithmic trading system using the Kronos financial foundation model (102M params, AAAI 2026). Finetuning Kronos-base into 3 market-specialized checkpoints, combining ensemble predictions with CQR-calibrated risk management. User is Raj (s3702111), UTwente student, building a quant firm.

---

## Current Status: Phase 3.5 — Finetuning v2 + Ensemble

### What's Running Right Now

Three SLURM jobs submitted June 6, 2026 on UTwente HPC targeting **L40S (Lovelace)** GPUs:

| Job ID | Market | Config | Purpose | Expected Time |
|--------|--------|--------|---------|---------------|
| 510044 | BTC/USDT | `config_btc.yaml` | Predictor only (tokenizer done, `skip_existing: true`) | ~6-8h |
| 510045 | EUR/USD | `config_eur_v2.yaml` | Full retrain, seed=137 for ensemble diversity | ~1-1.5h |
| 510046 | XAU/USD | `config_xau_v2.yaml` | Anti-overfitting (fewer epochs, lower LR) | ~45min |

### Completed Finetuning (v1)

| Market | Status | Val Loss | Time | Notes |
|--------|--------|----------|------|-------|
| BTC/USDT | **Tokenizer done, predictor incomplete** | Tok: 0.0027 | Tok: 2.6h | Previous job killed at 8h limit. New job with 20h limit + L40S submitted. |
| EUR/USD | **Complete** | 1.6098 | 1h12m | Finetuning improved RMSE/MAE vs zero-shot |
| XAU/USD | **Complete but overfitted** | 0.9054 | 1h39m | 50 tok + 20 pred epochs on 9.5k candles caused overfitting. Zero-shot outperformed. |

### v2 Training Rationale

- **BTC**: Same config but on L40S with `skip_existing: true` (skips completed tokenizer), batch_size=128, 20h limit
- **EUR v2**: Identical hyperparams to v1 but seed=137 (v1 was seed=42). Purpose: multi-seed ensemble diversity per Grinold's Law
- **XAU v2**: Anti-overfitting — 15 tokenizer + 5 predictor epochs (v1 had 50+20), LR halved, weight_decay 0.1→0.15, train_ratio 0.9→0.85

---

## Evaluation Results So Far

### Zero-Shot Baseline (20 rolling windows)

| Market | IC | Rank IC | Dir. Accuracy | RMSE | MAE |
|--------|-----|---------|---------------|------|-----|
| BTC/USDT 15m | -0.342 | -0.107 | 60.0% | 3160.24 | 2116.78 |
| EUR/USD 1h | -0.264 | -0.059 | 40.0% | 0.0090 | 0.0066 |
| XAU/USD 4h | Not in CSV | | | | |

### Finetuned v1 vs Zero-Shot (from HPC eval)

**EUR/USD**: Finetuned improved RMSE and MAE over zero-shot. Direction accuracy comparable.
**XAU/USD**: Finetuned performed WORSE than zero-shot — clear overfitting from too many epochs on small dataset.

*(Exact finetuned numbers were displayed in terminal output but not saved to CSV. Need to re-run `finetuned_eval.py` after v2 training completes.)*

---

## Architecture & Key Technical Decisions

### Ensemble Approach (Informed by Quant Research)

Based on research into Renaissance, Two Sigma, and academic literature:

1. **Ensemble uncorrelated signals** — ZS + FT_v1 + FT_v2 combined via Grinold's Law (IR = IC × √Breadth)
2. **Regime detection needed** — HMM or volatility-based to adjust position sizing per regime
3. **CQR for uncertainty** — distribution-free coverage guarantees for prediction intervals
4. **Walk-forward backtest with costs** — maker/taker fees, slippage model, realistic position sizing
5. **Target Sharpe 1.0-1.5** — realistic for retail quant with 3 markets

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
| `slurm/finetune_btc.sh` | L40S (lovelace) | 20h |
| `slurm/finetune_eur.sh` | A40 (ampere) — **outdated** | 8h |
| `slurm/finetune_eur_v2.sh` | L40S (lovelace) | 8h |
| `slurm/finetune_xau.sh` | A40 (ampere) — **outdated** | 8h |
| `slurm/finetune_xau_v2.sh` | L40S (lovelace) | 8h |

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

1. **Wait for v2 training to complete** (jobs 510044-510046)
2. **Run `eval_ensemble.py`** — benchmark ZS vs FT_v1 vs FT_v2 vs ensemble combinations with bootstrap CIs
3. **CQR calibration** — calibrate prediction intervals on calibration split for each market
4. **Walk-forward backtest** — with realistic transaction costs (BTC: ~0.07%, EUR: ~1 pip, XAU: ~$0.40 per RT)
5. **Add regime detection** — HMM or volatility classifier to modulate position sizing
6. **LLM strategy layer** — generate and evaluate strategy code
7. **Final test set evaluation** — touched exactly ONCE

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

---

*This file is the single source of truth for project state. Update after every significant change.*
