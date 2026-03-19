# Strict Walk-Forward Meta-Filter Validation — Run Report

## 1) Run Objective

Validate whether the meta-labeling layer improves the **stable regime orchestrator** under strict out-of-sample walk-forward rules:

- meta features built on train fold only
- labels built on train fold only
- meta filter fit/calibrated on train fold only
- fitted filter applied to next test fold only
- all reported performance stitched from test folds only

This report covers the latest run after integrating strict walk-forward meta calibration into the research engine.

---

## 2) Data Used

Current run uses the repository’s sample dataset pipeline:

- source file: `data/sample_ohlcv.csv`
- generated symbols: `EURUSD`, `GBPUSD`, `USDJPY`, `AUDUSD`
- tested symbol/timeframe in prototype: `EURUSD`, `H1`
- bars per symbol generated: 2400

Execution assumptions include spread/slippage/commission costs from the existing cost model.

---

## 3) Validation Setup

## 3.1 Walk-forward structure

- train window: 800 bars
- test window: 200 bars
- folds: 7

## 3.2 Compared strategies (walk-forward strict)

1. **Stable Orchestrator (WF Unfiltered)**
2. **Stable Orchestrator + MetaFilter (WF)**

## 3.3 Meta filter setup

- model: `RuleBasedMetaFilter`
- target filter rate: 0.4 (quantile-calibrated, constrained 20–60% band)
- label horizon: 24 bars
- success threshold: 0.0002
- minimum train samples for fitting: 30

---

## 4) Aggregate OOS Results

Source: `outputs/wf_meta_filter_comparison.csv`

| Strategy | CAGR | Sharpe | Sortino | MaxDrawdown | ProfitFactor | WinRate | Expectancy | TradeCount | ExposurePct | AvgHoldingBars | AvgFilterRateByFold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Stable Orchestrator (WF Unfiltered) | -0.0518 | -1.4393 | -1.6469 | -0.0236 | 0.7517 | 0.5484 | -0.000289 | 31 | 0.5071 | 22.9032 | 0.0000 |
| Stable Orchestrator + MetaFilter (WF) | -0.0350 | -1.1033 | -1.0756 | -0.0164 | 0.8863 | 0.4681 | -0.000081 | 47 | 0.3857 | 11.4894 | 0.5485 |

### Key deltas (Filtered - Unfiltered)

- Sharpe: **+0.3360**
- CAGR: **+0.0168**
- Max Drawdown: **+0.0072** (less severe drawdown)
- Expectancy: **+0.0002088** (less negative / improved)
- Profit Factor: **+0.1346**

Interpretation: strict OOS results show a positive risk-adjusted shift from the meta layer in this run.

---

## 5) Fold-Level Results

Source: `outputs/wf_meta_filter_folds.csv`

Per-fold fields include:

- `meta_filter_rate`
- `test_Sharpe_unfiltered` / `test_Sharpe_filtered`
- `test_CAGR_unfiltered` / `test_CAGR_filtered`
- `test_MaxDrawdown_unfiltered` / `test_MaxDrawdown_filtered`
- `test_Expectancy_unfiltered` / `test_Expectancy_filtered`
- `meta_threshold`
- `meta_state` (serialized learned calibration state)

### Fold summary statistics

- fold count: **7**
- Sharpe improved in **5/7** folds (71.43%)
- Expectancy improved in **6/7** folds (85.71%)
- Drawdown improved in **5/7** folds (71.43%)
- best Sharpe delta (filtered - unfiltered): **+5.5116**
- worst Sharpe delta (filtered - unfiltered): **-6.2329**

This indicates improvement is frequent but not universal, with material regime sensitivity by fold.

---

## 6) Meta-Filter Stability Diagnostics

Source: `outputs/wf_meta_filter_diagnostics.csv`

- AvgFilterRateByFold: **0.5485**
- StdFilterRateByFold: **0.2977**
- AvgMetaThresholdByFold: **0.3971**
- StdMetaThresholdByFold: **0.0579**
- PctFoldsFilteredSharpeImproved: **0.7143**
- PctFoldsFilteredExpectancyImproved: **0.8571**
- PctFoldsFilteredDrawdownImproved: **0.7143**
- FoldCount: **7**

### Stability notes

- Threshold stability is reasonably tight (`std ≈ 0.058`) relative to score scale.
- Filter rate dispersion is wider (`std ≈ 0.298`), indicating fold-dependent selectivity.
- Overall, fold-wise improvement ratios are favorable for Sharpe/Expectancy/Drawdown.

---

## 7) Artifacts Produced

Strict WF meta artifacts:

1. `outputs/wf_meta_filter_comparison.csv`
2. `outputs/wf_meta_filter_folds.csv`
3. `outputs/wf_filtered_vs_unfiltered_equity.png`
4. `outputs/wf_meta_filter_diagnostics.csv`

Related full fold export:

- `outputs/walk_forward_orchestrated_stable_meta_folds.csv`

---

## 8) Conclusion

Under strict train-then-test walk-forward validation, the meta-filter layer shows **modest but meaningful OOS improvement** over the unfiltered stable orchestrator in this run:

- better Sharpe
- better CAGR
- improved drawdown
- improved expectancy in most folds

The next confidence step is repeating this same strict WF meta process across expanded real datasets/symbols/timeframes and summarizing fold-consistency across the broader universe.
