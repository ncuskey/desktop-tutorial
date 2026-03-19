# Specialist Sleeves & Regime-Aware Orchestration Report

## 1) Objective

This report documents the new implementation that upgrades the Forex Strategy Research Lab from regime analysis to **regime-driven specialist deployment and orchestration**.

The goal was to test whether:

1. Specialist sleeves outperform unfiltered baselines
2. Regime-aware orchestration outperforms standalone specialists
3. Drawdowns improve even if returns remain modest
4. Trade count drops while expectancy improves

---

## 2) Implementation Summary (What Was Added)

## 2.1 Specialist sleeve combination logic

### `combinations/engine.py`
Added:

- `combine_specialist_sleeves(strategy_outputs, regime_series, regime_to_sleeve, fallback, default_strategy)`

Behavior:
- Routes active sleeve by regime label
- If no mapping is present, supports fallback modes:
  - `flat`
  - `previous_position`
  - `default_strategy`

### `combinations/__init__.py`
- Exported `combine_specialist_sleeves`

---

## 2.2 Regime-aware orchestrator

### `orchestrators/regime_specialist.py`
Added:

- `RegimeSpecialistOrchestrator`

Inputs:
- Dataframe with regime columns
- Sleeve signals dict
- `regime_to_sleeve` routing map
- Optional `sleeve_weights`
- Optional risk controls

Output:
- Final orchestrated signal series (float position-capable)

### `orchestrators/__init__.py`
- Exported `RegimeSpecialistOrchestrator`

---

## 2.3 New risk controls

### `execution/risk_controls.py`
Added:

- `apply_no_trade_filter_high_vol(...)`
  - blocks new entries in `HIGH_VOL` unless explicitly enabled
- `apply_volatility_targeting(...)`
  - scales signal inversely with ATR-normalized volatility
  - capped by `max_leverage` (default 1.0)

### `execution/__init__.py`
- Exported new risk control functions

---

## 2.4 Metrics extension

### `metrics/performance.py`
Extended:

- `compute_metrics(...)` now reports:
  - `ExposurePct`
  - `AvgHoldingBars`

- `compute_metrics_by_regime(...)` now includes:
  - `PnL`
  - `Sharpe`
  - `CAGR`
  - `MaxDrawdown`
  - `TradeCount`
  - `Bars`
  - `TimePct`

---

## 2.5 Execution/trade diagnostics extension

### `execution/simulator.py`
Trade extraction now includes:
- `holding_bars` per trade

This powers `AvgHoldingBars`.

---

## 2.6 Walk-forward extension

### `research/walk_forward.py`
Extended to preserve richer metrics while evaluating:
- filtered specialists
- orchestrated specialist strategy

Per-fold outputs include regime breakdown fields:
- `test_regime_return_breakdown`
- `test_regime_time_pct`

---

## 2.7 Switching diagnostics

### `research/switch_diagnostics.py`
Added:
- `compute_switch_diagnostics(regime_series, returns, n_bars=24)`

Outputs:
- `SwitchCount`
- `AvgReturnAfterSwitch`
- `AvgDrawdownAfterSwitch`
- `% switches improving next-N-bar expectancy`

### `research/__init__.py`
- Exported `compute_switch_diagnostics`

---

## 2.8 Prototype experiment wiring

### `prototype.py` now runs:

**A) Baselines**
- MA Baseline
- RSI Baseline

**B) Specialist sleeves**
- MA Specialist (TRENDING)
- RSI Specialist (RANGING)

**C) Regime-specialist orchestrated strategy**
- Regime routing:
  - `TRENDING_LOW_VOL -> trend_sleeve`
  - `TRENDING_MID_VOL -> trend_sleeve`
  - `RANGING_LOW_VOL -> mean_reversion_sleeve`
  - `RANGING_MID_VOL -> mean_reversion_sleeve`
  - high-vol regimes intentionally unmapped → flat fallback
- Vol targeting enabled
- High-vol entry blocking enabled

**D) Equal-weight filtered ensemble**
- 50/50 filtered MA + filtered RSI

**Walk-forward**
- Added walk-forward runs for:
  - MA specialist
  - RSI specialist
  - orchestrated specialist

---

## 3) New Output Artifacts

Generated in `outputs/`:

1. `specialist_vs_orchestrated_comparison.csv`
2. `regime_attribution.csv`
3. `orchestrated_equity_curve.png`
4. `pnl_by_regime.png`
5. `switch_diagnostics.csv`

Also generated for diagnostics:
- `walk_forward_ma_filtered_folds.csv`
- `walk_forward_rsi_filtered_folds.csv`
- `walk_forward_orchestrated_folds.csv`

---

## 4) Results (Detailed)

Source: `outputs/specialist_vs_orchestrated_comparison.csv`

## 4.1 In-sample comparison

| Strategy | CAGR | Sharpe | Sortino | MaxDD | ProfitFactor | WinRate | Expectancy | TradeCount | ExposurePct | AvgHoldingBars |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MA Baseline | -0.0377 | -0.7371 | -1.1910 | -0.0400 | 0.8243 | 0.3250 | -0.00036 | 40 | 0.9664 | 56.80 |
| MA Specialist (TRENDING) | -0.0428 | -1.2088 | -1.4190 | -0.0370 | 0.8708 | 0.3400 | -0.00016 | 50 | 0.5138 | 24.16 |
| RSI Baseline | -0.1614 | -3.7253 | -5.8325 | -0.0805 | 0.5539 | 0.3575 | -0.00038 | 207 | 0.8766 | 9.96 |
| RSI Specialist (RANGING) | -0.0770 | -2.6387 | -2.6409 | -0.0452 | 0.7289 | 0.3924 | -0.00016 | 158 | 0.3535 | 5.26 |
| Filtered Ensemble (Equal Weight) | -0.1107 | -2.4819 | -3.7936 | -0.0638 | 0.7150 | 0.3750 | -0.00023 | 200 | 0.8673 | 10.20 |
| Regime Specialist Orchestrated | -0.1579 | -4.3972 | -5.3979 | -0.0671 | 0.7810 | 0.3305 | -0.00011 | 239 | 0.5815 | 5.72 |

### Key in-sample deltas

- **MA Specialist vs MA Baseline**
  - Sharpe: **-0.4717** (worse)
  - MaxDD: **+0.0030** (improved / less negative)
  - Exposure: **-0.4526** (much lower)

- **RSI Specialist vs RSI Baseline**
  - Sharpe: **+1.0866** (improved)
  - MaxDD: **+0.0352** (improved)
  - TradeCount: **-49** (lower activity)
  - Expectancy improved (less negative): from `-0.00038` to `-0.00016`

- **Orchestrated vs best specialist**
  - Sharpe underperformed best specialist by **-3.1884**
  - Sharpe underperformed equal-weight filtered ensemble by **-1.9153**

---

## 4.2 Walk-forward comparison

| Strategy | CAGR | Sharpe | Sortino | MaxDD | ProfitFactor | WinRate | Expectancy | TradeCount | ExposurePct | AvgHoldingBars |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MA Baseline | -0.1700 | -4.6173 | -5.8451 | -0.0490 | 0.2640 | 0.2903 | -0.00139 | 31 | 0.5907 | 26.68 |
| MA Specialist (TRENDING) | -0.1568 | -6.5915 | -5.5894 | -0.0410 | 0.1611 | 0.4000 | -0.00119 | 30 | 0.2607 | 12.17 |
| RSI Baseline | -0.1063 | -2.5496 | -3.5804 | -0.0320 | 0.7130 | 0.3529 | -0.00025 | 85 | 0.7193 | 11.85 |
| RSI Specialist (RANGING) | -0.0464 | -1.7835 | -1.4618 | -0.0187 | 0.8848 | 0.3750 | -0.00007 | 56 | 0.2550 | 6.38 |
| Regime Specialist Orchestrated | -0.2587 | -8.4898 | -9.0385 | -0.0690 | 0.4256 | 0.2589 | -0.00041 | 112 | 0.4514 | 5.64 |

### Key walk-forward deltas

- **RSI Specialist vs RSI Baseline**
  - Sharpe: **+0.7660**
  - MaxDD: **+0.0132** (improved)
  - TradeCount: **-29**
  - Exposure reduced sharply: `0.7193 -> 0.2550`

- **Orchestrated vs RSI Specialist**
  - Sharpe: **-6.7063** (significantly worse)
  - MaxDD also worse (`-0.0690` vs `-0.0187`)

---

## 4.3 Regime attribution highlights

Source: `outputs/regime_attribution.csv`

Trend-regime Sharpe snapshot:

| Strategy | TRENDING Sharpe | RANGING Sharpe |
|---|---:|---:|
| MA Baseline | 0.4156 | -1.9869 |
| RSI Baseline | -5.3254 | -1.8513 |
| MA Specialist (TRENDING) | -0.7881 | -5.6255 |
| RSI Specialist (RANGING) | -6.5517 | -2.2556 |
| Regime Specialist Orchestrated | -4.2221 | -4.6302 |

Observations:

- MA baseline still shows its strongest behavior in TRENDING relative to RANGING.
- RSI baseline remains materially better in RANGING than TRENDING.
- Specialist filters reduced exposure and concentrated behavior but did not uniformly improve every regime metric.
- Orchestrated strategy currently suffers in both trend and range buckets in this configuration.

---

## 4.4 Switching diagnostics

Source: `outputs/switch_diagnostics.csv`

- SwitchCount: **536**
- AvgReturnAfterSwitch (next 24 bars): **-0.000839**
- AvgDrawdownAfterSwitch (next 24 bars): **-0.002697**
- `% switches improving next-24-bar expectancy`: **47.20%**

Interpretation:

- Regime transitions are frequent.
- Switches are not yet delivering positive post-switch expectancy on average.
- Current regime map/filters likely over-switch and/or route in noisy transitions.

---

## 5) Success Criteria Assessment

1. **Specialist sleeves outperform unfiltered baselines**  
   - **Partially true**:
     - RSI specialist improved vs RSI baseline (in-sample + walk-forward)
     - MA specialist reduced risk/exposure but Sharpe worsened

2. **Regime-aware orchestration outperforms standalone specialists**  
   - **Not met in this run**:
     - Orchestrated strategy underperformed specialists and ensemble on Sharpe and drawdown

3. **Drawdowns improve even if return remains similar**  
   - **True for specialist sleeves**, especially RSI specialist
   - **Not true for orchestrated strategy** in current settings

4. **Trade count drops while expectancy improves**  
   - **True for RSI specialist**
   - Mixed for MA specialist
   - Not true for orchestrated strategy (trade count increased)

---

## 6) Practical Conclusions

1. The architecture now supports robust specialist deployment and orchestration experiments.
2. Current data/configuration indicates a cleaner edge for **RSI as a range specialist**.
3. The orchestrated strategy requires further tuning before it can outperform specialists:
   - regime map granularity
   - transition smoothing/hysteresis
   - volatility-target calibration
   - switch dampening / minimum regime duration

---

## 7) Reproducibility

From repo root:

```bash
python3 -m pip install -r requirements.txt
python3 prototype.py
```

Generated artifacts are saved to `outputs/`.
