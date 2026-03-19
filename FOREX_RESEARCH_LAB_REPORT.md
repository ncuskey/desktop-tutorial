# Forex Strategy Research Lab — Build & Results Report

## 1) Executive Summary

This repository now includes a production-oriented **Forex Strategy Research Lab prototype** implemented in Python with modular components for data handling, strategy generation, execution simulation, evaluation metrics, robustness testing, and experiment tracking.

The initial MVP objective was completed:

1. Load sample OHLCV data (mock CSV)
2. Run two baseline strategies:
   - MA crossover (trend)
   - RSI reversal (mean reversion)
3. Execute trades with realistic costs (spread + slippage + commission)
4. Run walk-forward optimization/testing
5. Output metrics, equity/drawdown curves, parameter heatmap, and robustness artifacts

Current run results are **negative after costs** for both baseline strategies (in-sample and walk-forward), which is expected for many naive parameterizations and is exactly the kind of outcome this framework is designed to expose early.

---

## 2) Build Scope Delivered

### 2.1 Folder/Module Architecture

The lab is structured with clear separation of concerns:

- `data/`
  - `loader.py` — OHLCV loading, mock generation, timeframe resampling
  - `indicators.py` — MA, RSI, ATR, Bollinger, ADX
  - `costs.py` — cost model and attachment utilities
- `strategies/`
  - `trend.py` — MA crossover, Donchian breakout
  - `mean_reversion.py` — RSI reversal, Bollinger fade
  - `breakout.py` — range breakout, volatility expansion breakout
  - `carry.py` — interest differential proxy placeholder
- `execution/`
  - `simulator.py` — signal-to-position conversion, transaction costs, equity/drawdown, trade extraction
- `metrics/`
  - `performance.py` — CAGR, Sharpe, Sortino, Max Drawdown, Profit Factor, Win Rate, Expectancy, Trade Count
- `research/`
  - `walk_forward.py` — rolling train/test optimization + stitched OOS aggregation
  - `parameter_sweep.py` — grid and random sweep
  - `bootstrap.py` — bootstrap distribution + risk-of-ruin estimate
  - `tracking.py` — SQLite experiment logging
- `combinations/`
  - `engine.py` — confirmation, weighted ensemble, specialist sleeve
- `orchestrators/`
  - `rule_based.py` — ADX threshold switch
  - `performance_based.py` — rolling-Sharpe weighted allocation
  - `regime.py` — ATR/ADX regime classification and switching
- `prototype.py`
  - end-to-end runnable pipeline for the initial deliverable

### 2.2 Dependencies

- `pandas`
- `numpy`
- `matplotlib`

(`requirements.txt` added)

---

## 3) Validation & Research Principles Implemented

### 3.1 No-Lookahead Bias (Implemented)

The execution layer shifts signal application by one bar:

- Position used for return at bar `t` comes from signal at `t-1`.
- This prevents using information from the current bar close to trade that same bar's return.

### 3.2 Strict Train/Test Separation (Implemented)

Walk-forward engine:

- Optimizes parameters only on the **training window**.
- Applies best parameters only to the **next test window**.
- Repeats in rolling steps and aggregates out-of-sample test returns.

### 3.3 Walk-Forward Evaluation (Implemented)

- Rolling windows used in prototype:
  - train: `800` bars
  - test: `200` bars
- Number of folds realized on dataset: `7`

### 3.4 Bootstrap Robustness (Implemented)

- Bootstrap over stitched walk-forward returns
- Outputs distribution summaries:
  - Sharpe quantiles (P05/Median/P95)
  - drawdown quantile
  - risk of ruin proxy

---

## 4) Experiment Configuration (Current Run)

### 4.1 Data

- Source: generated mock OHLCV at `data/sample_ohlcv.csv`
- Symbols generated: EURUSD, GBPUSD, USDJPY, AUDUSD
- Prototype run evaluated:
  - symbol: `EURUSD`
  - timeframe: `H1`

### 4.2 Transaction Cost Model

- spread: `0.8 bps`
- slippage: `0.5 bps`
- commission: `0.3 bps`
- total one-way cost used during position changes: `1.6 bps`

### 4.3 Strategy Parameters in Baseline Run

- MA crossover baseline:
  - `fast=20`, `slow=80`
- RSI reversal baseline:
  - `oversold=30`, `overbought=70`, `exit_level=50`

### 4.4 Walk-Forward Search Spaces

- MA grid:
  - `fast: [10, 20, 30]`
  - `slow: [50, 80, 120]`
- RSI grid:
  - `oversold: [25, 30, 35]`
  - `overbought: [65, 70, 75]`
  - `exit_level: [50]`

---

## 5) Results

Source: `outputs/metrics_summary.csv`

## 5.1 Performance Metrics Table

| Strategy | CAGR | Sharpe | Sortino | Max Drawdown | Profit Factor | Win Rate | Expectancy | Trade Count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MA Crossover (In-Sample) | -0.0377 | -0.7371 | -1.1910 | -0.0400 | 0.8243 | 0.3250 | -0.0004 | 40 |
| RSI Reversal (In-Sample) | -0.2204 | -5.2626 | -8.1906 | -0.1016 | 0.4867 | 0.2903 | -0.0005 | 186 |
| MA Crossover (Walk-Forward) | -0.1700 | -4.6173 | -5.8451 | -0.0490 | 0.2640 | 0.2903 | -0.0014 | 31 |
| RSI Reversal (Walk-Forward) | -0.1063 | -2.5496 | -3.5804 | -0.0320 | 0.7130 | 0.3529 | -0.0002 | 85 |

### 5.2 Interpretation

- Both strategies are unprofitable in this current setup after realistic costs.
- Walk-forward confirms weak robustness; no evidence of a stable positive edge in this run.
- RSI has more trades and less negative OOS Sharpe than MA in this setup, but still negative.
- This is a useful result: the framework is functioning as intended by rejecting weak strategy variants.

---

## 6) Walk-Forward Diagnostics

Sources:

- `outputs/walk_forward_ma_folds.csv`
- `outputs/walk_forward_rsi_folds.csv`

### 6.1 Fold Count

- MA folds: `7`
- RSI folds: `7`

### 6.2 Fold-Level Sharpe Behavior

- MA average test-fold Sharpe: `-4.8583`
- RSI average test-fold Sharpe: `-1.9471`
- Positive-Sharpe folds:
  - MA: `1 / 7`
  - RSI: `1 / 7`

### 6.3 Most Frequently Selected Parameters

- MA:
  - `{'fast': 20, 'slow': 80}` selected 2 times
  - others selected once each
- RSI:
  - `{'oversold': 25, 'overbought': 75, 'exit_level': 50}` selected 3 times
  - others selected once each

### 6.4 Best Single Fold (by test Sharpe)

- MA best fold:
  - test Sharpe: `2.5793`
  - params: `{'fast': 10, 'slow': 50}`
- RSI best fold:
  - test Sharpe: `6.7569`
  - params: `{'oversold': 25, 'overbought': 75, 'exit_level': 50}`

These isolated positive folds did not generalize across the full rolling sequence.

---

## 7) Bootstrap Robustness

Source: `outputs/bootstrap_summary.json` (300 bootstrap samples)

### 7.1 MA Walk-Forward Returns Bootstrap

- Sharpe P05: `-7.9797`
- Sharpe Median: `-4.7873`
- Sharpe P95: `-1.4145`
- Drawdown quantile (reported): `-0.0746`
- Risk of ruin estimate: `0.0`

### 7.2 RSI Walk-Forward Returns Bootstrap

- Sharpe P05: `-5.7148`
- Sharpe Median: `-2.4925`
- Sharpe P95: `0.9767`
- Drawdown quantile (reported): `-0.0635`
- Risk of ruin estimate: `0.0`

### 7.3 Interpretation

- MA bootstrap distribution is consistently negative in this run.
- RSI has a wider distribution and occasional positive bootstrap outcomes, but median remains negative.
- Risk-of-ruin output is 0.0 under current threshold settings; this should be interpreted cautiously and calibrated for portfolio-level assumptions.

---

## 8) Parameter Robustness (MA Heatmap + Sweep)

Sources:

- `outputs/ma_parameter_sweep.csv`
- `outputs/ma_parameter_heatmap.png`

Top MA combinations by in-sample Sharpe (all still negative):

1. `{'fast': 20, 'slow': 80}` → Sharpe `-0.7371`
2. `{'fast': 30, 'slow': 80}` → Sharpe `-1.0213`
3. `{'fast': 30, 'slow': 120}` → Sharpe `-1.0302`

Conclusion: current MA search region does not show a robust positive basin in this dataset/cost regime.

---

## 9) Generated Outputs

All artifacts are under `outputs/`:

- `metrics_summary.csv`
- `walk_forward_ma_folds.csv`
- `walk_forward_rsi_folds.csv`
- `bootstrap_ma_distribution.csv`
- `bootstrap_rsi_distribution.csv`
- `bootstrap_summary.json`
- `ma_parameter_sweep.csv`
- `equity_drawdown_curves.png`
- `ma_parameter_heatmap.png`
- `experiments.sqlite`

---

## 10) Design Strengths in Current Build

1. **Modular and extensible architecture** (clean package boundaries)
2. **Validation-first pipeline** (walk-forward + bootstrap included from day one)
3. **Execution realism baseline** (spread/slippage/commission costs modeled)
4. **Experiment traceability** (SQLite tracker)
5. **Combination and orchestration scaffolding already available** for next phases

---

## 11) Known Gaps / Next Recommended Iteration

1. **Portfolio-level multi-symbol evaluation**
   - current prototype run is single symbol (`EURUSD`), though data and modules support multi-symbol expansion.
2. **Richer execution realism**
   - session-dependent spread, asymmetric slippage, latency models, stop/limit mechanics.
3. **More robust risk controls**
   - volatility targeting, max position limits, exposure/netting, kill switches.
4. **Regime-conditional parameter sets**
   - tie optimizer outputs to regime classifier states.
5. **Cross-validation variants**
   - anchored walk-forward, purged CV, embargo windows.
6. **Statistical significance overlays**
   - confidence intervals for metrics and false-discovery controls across large sweeps.

---

## 12) Reproducibility

From repo root:

```bash
python3 -m pip install -r requirements.txt
python3 prototype.py
```

Report inputs are refreshed from generated artifacts in `outputs/`.

---

## 13) Bottom Line

The build successfully establishes a robust research framework (not a toy backtester), with proper anti-overfitting guardrails and realistic cost modeling.  
The first tested baselines (MA crossover and RSI reversal) do **not** survive evaluation in this current configuration, which is the correct and expected behavior of a research lab focused on robust strategy discovery.
