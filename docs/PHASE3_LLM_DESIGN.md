# Phase 3.3 — LLM-Scored Headlines: pre-registered design

> Written July 19, 2026, BEFORE downloading any corpus or scoring any headline.
> Changes to this design after seeing data = a new trial in the ledger.

## Hypothesis

News events carry tradeable information that pre-aggregated sentiment scores
(killed: GDELT tone, Fear&Greed, attention counts) destroy — because the edge,
if any, is in WHAT happened (event type, severity, specificity), not in how
positive the words sound. An LLM can extract a structured event representation
that nothing cheaper can.

## Corpus

Kaggle historical crypto-news dataset(s), selected by coverage BEFORE any
scoring: require timestamped headlines, ≥2 years overlapping the price panel
(spot 2017-21 + perp 2022-26). Selection criteria are data-quality only
(dedup rate, timestamp sanity, volume/day) — never signal results.

## Scoring (Claude, haiku-class, batched)

Each headline is scored WITHOUT dates, sources, or any price context:
  - event_type: one of {regulation, hack/exploit, adoption/integration,
    ETF/institutional-flow, macro, protocol/technical, legal/enforcement,
    market-structure, noise/other}
  - severity: 0-3 (how consequential for holders of the tagged assets,
    judged EX-ANTE as of publication)
  - direction: -1/0/+1 (rational ex-ante reading for the tagged assets)
  - assets: which of the 15-coin universe it concerns (or MARKET-wide)

### Declared contamination caveat (the LLM hindsight problem)

The scoring model has knowledge of crypto history: for famous headlines it may
"know the outcome," which leaks look-ahead into any feature built from its
judgments and BIASES THE TEST TOWARD FALSE POSITIVES. Mitigations (declared):
prompts demand ex-ante framing and never mention dates; features lean on the
objective taxonomy (event_type, severity, asset tags) more than direction;
and any pass is discounted accordingly at review — a marginal pass on
LLM-scored features is NOT a brick.

## Features and cells (declared)

Daily per-coin score: sum of severity x direction over that day's headlines
tagged to the coin; MARKET-wide score analogous. Shock = score minus trailing
28d mean (same construction as the attention family, for comparability).

Cells (4, gate |t| > 2.65 + bucket stability):
  1. XS shock across coins, h=1d (long positive-news / short negative-news)
  2. XS shock, h=5d
  3. TS-BTC market-wide shock sign, h=5d
  4. EVENT-TYPE cell: hack/exploit + legal/enforcement events only, tagged
     coin, h=5d (the "hard news" subset least contaminated by tone)

Alignment: headlines of day t (UTC) predict returns from t+1. No same-day cell
(intraday timestamps in Kaggle corpora are unreliable; same-day tests would
manufacture look-ahead).

## Cost & budget

Haiku-class, ~20 headlines/request, ~30-50k headlines ≈ $5-15. One scoring
pass only; re-scoring with a different prompt = new trial, counted.

## Amendments (July 19, before scoring — recorded, not silent)

1. **Scorer = open-weights local model (Qwen2.5-7B-Instruct, Q4, via ollama on
   the RTX PRO 1000), not Claude.** Raj's call (no paid models). Side effect:
   a 7B model has memorized far LESS crypto history than a frontier model —
   the hindsight-contamination risk shrinks. Cost: $0. Temperature 0, one pass.
2. **Corpus selection executed per the declared criteria**:
   - INCLUDED: CoinTelegraph newsfeed (16,825 headlines, 2017-08..2020-04,
     clean published timestamps) + oliviervha Crypto News+ (31,037, 2021-10..
     2023-12). Total 47,862 headlines in the priced window.
   - EXCLUDED: kashnitsky 2013-18 (year-granularity only; URL date recovery
     22% — fails the timestamp criterion).
   - Known coverage hole 2020-04..2021-10, declared as-is.
   - oliviervha ships a 'sentiment' column — IGNORED (pre-scored sentiment is
     exactly the thing this design distrusts).
