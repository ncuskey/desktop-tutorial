# Forex Strategy Research Lab — Consolidated Results Report

_Updated: 2026-03-19_

## 1) Purpose

This report combines the currently tracked research outcomes into one document, spanning:

- MVP baseline build + initial walk-forward results
- Specialist sleeves and regime-aware orchestration phase
- Strict walk-forward meta-filter validation (orchestrator track)
- TrendBreakout_V2 strategy-research track:
  - R1.2.x regime/state/quality gating
  - R1.3 / R1.3.1 trend gating
  - R1.4 meta-labeling (follow-through)

---

## 2) Baseline MVP (Initial Lab Validation)

Source: `FOREX_RESEARCH_LAB_REPORT.md`

### 2.1 Baseline performance snapshot

| Strategy | Regime | CAGR | Sharpe | MaxDD | Expectancy | Trade Count |
|---|---|---:|---:|---:|---:|---:|
| MA Crossover | In-sample | -0.0377 | -0.7371 | -0.0400 | -0.0004 | 40 |
| RSI Reversal | In-sample | -0.2204 | -5.2626 | -0.1016 | -0.0005 | 186 |
| MA Crossover | Walk-forward | -0.1700 | -4.6173 | -0.0490 | -0.0014 | 31 |
| RSI Reversal | Walk-forward | -0.1063 | -2.5496 | -0.0320 | -0.0002 | 85 |

### 2.2 Baseline conclusion

- The lab correctly rejected naive baselines after costs.
- Walk-forward + bootstrap showed no robust positive edge in that initial setup.

---

## 3) Specialist Sleeves + Regime-Aware Orchestration

Source: `SPECIALIST_ORCHESTRATION_REPORT.md`

### 3.1 Key walk-forward results

| Strategy | Sharpe | MaxDD | Expectancy | Trade Count |
|---|---:|---:|---:|---:|
| MA Baseline | -4.6173 | -0.0490 | -0.00139 | 31 |
| MA Specialist (TRENDING) | -6.5915 | -0.0410 | -0.00119 | 30 |
| RSI Baseline | -2.5496 | -0.0320 | -0.00025 | 85 |
| RSI Specialist (RANGING) | -1.7835 | -0.0187 | -0.00007 | 56 |
| Regime Specialist Orchestrated | -8.4898 | -0.0690 | -0.00041 | 112 |

### 3.2 Switching diagnostics (orchestrated track)

- SwitchCount: **536**
- Avg return after switch (next 24 bars): **-0.000839**
- Avg drawdown after switch (next 24 bars): **-0.002697**
- % beneficial switches: **47.20%**

### 3.3 Conclusion

- RSI sleeve specialization improved behavior vs RSI baseline.
- Orchestration underperformed specialists in the reported configuration.
- This motivated later stabilization and stricter gating/meta workflows.

---

## 4) Strict WF Meta-Filter Validation (Stable Orchestrator Track)

Source: `WF_META_VALIDATION_RUN_REPORT.md`

### 4.1 Aggregate strict OOS comparison

| Variant | CAGR | Sharpe | MaxDD | Expectancy | Trade Count | Avg Filter Rate |
|---|---:|---:|---:|---:|---:|---:|
| Stable Orchestrator (Unfiltered) | -0.0518 | -1.4393 | -0.0236 | -0.000289 | 31 | 0.0000 |
| Stable Orchestrator + MetaFilter | -0.0350 | -1.1033 | -0.0164 | -0.000081 | 47 | 0.5485 |

### 4.2 Fold-level improvement rates (from that run)

- Sharpe improved in **5/7 folds (71.43%)**
- Expectancy improved in **6/7 folds (85.71%)**
- Drawdown improved in **5/7 folds (71.43%)**

### 4.3 Conclusion

- On that strict-WF orchestrator run, meta-filtering produced directionally positive OOS shifts.

---

## 5) TrendBreakout_V2 Track — Latest R1.2.3 / R1.3 / R1.4 Outputs

Sources:

- `outputs/regime_gated_comparison.csv`
- `outputs/state_filter_diagnostics.csv`
- `outputs/gate_comparison_by_fold.csv`
- `outputs/gate_coverage_metrics.csv`
- `outputs/gate_statistical_tests.csv`
- `outputs/meta_gate_comparison.csv`
- `outputs/meta_coverage_metrics.csv`
- `outputs/meta_stat_tests.csv`
- `outputs/meta_feature_importance.csv`

## 5.1 R1.2.3 (Lookback state gate + quality gate) — latest comparison

### Per-symbol gated vs unfiltered

| Symbol | Variant | Sharpe | Expectancy | MaxDD | Trade Count | Positive Fold % |
|---|---|---:|---:|---:|---:|---:|
| AUDUSD | Gated | -2.1932 | 0.000358 | -0.03381 | 33 | 0.0667 |
| AUDUSD | Unfiltered | -1.4533 | 0.000619 | -0.02861 | 44 | 0.1333 |
| EURUSD | Gated | -0.7221 | 0.000465 | -0.01191 | 23 | 0.2667 |
| EURUSD | Unfiltered | -0.0543 | 0.000703 | -0.01722 | 52 | 0.4667 |
| GBPUSD | Gated | -1.1692 | 0.000233 | -0.01494 | 36 | 0.4000 |
| GBPUSD | Unfiltered | -1.6522 | 0.000226 | -0.02843 | 73 | 0.2667 |

### Coverage / gating diagnostics (overall mean)

- allowed_bar_pct: **0.2070**
- allowed_trade_pct: **0.5368**
- pct_passing_regime_filter: **0.2702**
- pct_passing_quality_filter: **0.7413**
- pct_passing_both: **0.2070**

Interpretation:
- Coverage is in a practical range (around 20% allowed bars).
- Outcomes remain mixed by symbol; GBPUSD benefited most directionally in this snapshot.

## 5.2 R1.3 / R1.3.1 (Trend gating, rank-based stabilization) — latest fold aggregates

### Fold delta summary

- median delta Sharpe: **0.0000**
- % folds with Sharpe improvement: **13.33%**
- median delta Expectancy: **0.0000**
- % folds with Expectancy improvement: **15.56%**
- median gated trades/fold: **3.0**
- % folds with gated trades >= 5: **6.67%**

### Coverage summary (ALL row)

- allowed_bar_pct: **0.5069**
- entry_pass_rate: **0.8091**
- trades_per_fold: **3.1111**
- effective_trade_coverage: **0.8091**
- zero_trade_folds_pct: **0.0444**

### Statistical tests (`gate_statistical_tests.csv`)

- `delta_sharpe` Wilcoxon p-value: **0.6092**
- `delta_expectancy` Wilcoxon p-value: **0.4955**
- `delta_max_dd` Wilcoxon p-value: **0.0330**
- `delta_trade_count` Wilcoxon p-value: **0.00056**

Interpretation:
- Structural stabilization reduced collapse risk (very low zero-trade fold share).
- Performance uplift remained weak/inconsistent; drawdown/trade-count shifts were more detectable than Sharpe/expectancy gains.

## 5.3 R1.4 (Meta-labeling follow-through) — latest fold aggregates

### Fold delta summary

- median delta Sharpe: **0.0000**
- median delta Expectancy: **0.0000**
- median delta TradeCount: **0.0000**
- % folds with Sharpe improvement: **6.67%**
- % folds with Expectancy improvement: **6.67%**
- % folds with MaxDD improvement: **6.67%**
- median meta trade count: **4.0**

### Coverage summary (ALL row)

- trades_per_fold: **3.5778**
- effective_trade_coverage: **0.9280**
- zero_trade_folds_pct: **0.0444**

### Statistical tests (`meta_stat_tests.csv`)

- `delta_sharpe` Wilcoxon p-value: **0.4631**
- `delta_expectancy` Wilcoxon p-value: **0.4631**
- `delta_max_dd` Wilcoxon p-value: **0.1088**
- `delta_trade_count` Wilcoxon p-value: **0.0231**

### Top meta feature importances (mean abs importance)

1. `early_mfe` (0.9243)
2. `early_mae` (0.9213)
3. `early_slope` (0.4552)
4. `early_return_1` (0.4321)
5. `vr_24` (0.4078)

Interpretation:
- The R1.4 framework and artifacts are complete and WF-safe.
- Early post-entry dynamics are the strongest predictors in current fits.
- Latest aggregate uplift is not yet consistent enough for promotion.

---

## 6) Combined Artifact Index

### Core reports

- `FOREX_RESEARCH_LAB_REPORT.md`
- `SPECIALIST_ORCHESTRATION_REPORT.md`
- `WF_META_VALIDATION_RUN_REPORT.md`
- `CONSOLIDATED_RESULTS_REPORT.md` (this file)

### Latest R1.2.x / R1.3 / R1.4 result files

- `outputs/regime_gated_comparison.csv`
- `outputs/regime_gated_fold_results.csv`
- `outputs/state_filter_diagnostics.csv`
- `outputs/trade_feature_dataset.csv`
- `outputs/trade_feature_bins.csv`
- `outputs/gate_model_scores.csv`
- `outputs/gate_rule_export.json`
- `outputs/gate_rule_export.md`
- `outputs/gate_comparison_by_fold.csv`
- `outputs/gate_coverage_metrics.csv`
- `outputs/gate_statistical_tests.csv`
- `outputs/meta_feature_dataset.csv`
- `outputs/meta_model_scores.csv`
- `outputs/meta_gate_comparison.csv`
- `outputs/meta_stat_tests.csv`
- `outputs/meta_feature_importance.csv`
- `outputs/meta_coverage_metrics.csv`
- `outputs/charts/*.png` (feature bins, interactions, rank/uplift/calibration/drift diagnostics)

---

## 7) Overall Program Status (Consolidated)

1. **Infrastructure status:** strong and modular (strategy runner, WF validation, gating/meta layers, diagnostics).
2. **Evidence so far:** edge is conditional/regime-sensitive; robust universal uplift is not yet established across all recent runs.
3. **Most consistent tactical signal:** risk/selectivity controls can improve drawdown/overtrading characteristics; return/Sharpe uplift remains fragile and symbol/fold dependent.
4. **Current best use:** continue single-strategy research testbed mode, prioritizing robustness and fold-stability over peak outcomes.

