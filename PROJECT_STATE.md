# CIFR-QUANT: Project State

> **Purpose**: Cross-session memory for Claude. Read this file at the start of every new session.
> **Last updated**: June 7, 2026

---

## Quick Summary

Dual-market multi-asset algorithmic trading system using the Kronos financial foundation model (102M params, AAAI 2026). Two markets: **Crypto** (15-25 assets via Binance, 15m) and **Commodities** (6-8 assets via TwelveData, 4h). BTC-finetuned checkpoint transfers to all crypto; XAU-finetuned checkpoint transfers to all commodities. Ensemble predictions + CQR-calibrated risk parity position sizing. User is Raj (s3702111), UTwente student, building a quant firm.

---

## Current Status: Phase 4 — Architecture Redesign Complete, Ready for Multi-Asset Data + CQR

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
| EUR/USD | v1 (seed=42) | 1.6098 | 1h12m | ✅ Complete |
| EUR/USD | v2 (seed=137) | 1.5521 | 67min | ✅ Complete |
| XAU/USD | v1 | 0.9054 | 1h39m | ✅ Complete (overfitted) |
| XAU/USD | v2 (anti-overfit) | 1.2316 | 26min | ✅ Complete |

### Ensemble Evaluation Complete (50 windows, bootstrap CIs)

| Market | Best Model | Key Metric | Value |
|--------|-----------|------------|-------|
| XAU/USD | ensemble_full (ZS+v1+v2) | IC | **+0.393** |
| EUR/USD | ensemble_eq / ensemble_full | Dir. Accuracy | **60%** |
| BTC/USDT | finetuned_v1 | RMSE improvement | Yes, but IC/DA flat |

**Key finding**: XAU ensemble_full is the strongest signal. BTC finetuning helps RMSE but needs v2 (different seed) for ensemble diversity. EUR ensembles best at direction.

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

1. **Fetch multi-asset data** — Run `scripts/fetch_universe.py` to download crypto (Binance) + commodity (TwelveData) OHLCV
2. **Create BTC v2 config** — Different seed for crypto ensemble diversity (currently only 1 BTC finetuned checkpoint)
3. **CQR calibration** — Per-asset calibration using ensemble predictions on calibration split
4. **Multi-asset walk-forward backtest** — Portfolio-level with risk parity sizing, realistic costs per market
5. **Regime detection** — HMM or volatility classifier, highest-impact addition
6. **LLM strategy layer** — Generate and evaluate strategy code
7. **Final test set evaluation** — Touched exactly ONCE

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
