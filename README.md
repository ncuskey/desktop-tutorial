# MapEngine Web App

This project uses React and TypeScript to render and manipulate fantasy maps in the browser. A WebWorker (`src/worker.ts`) performs heavy map generation tasks so the UI stays responsive.

## Getting Started

```bash
npm install
npm start      # launches CRA dev server
npm run build  # creates production build
npm test       # runs unit tests
```

## Public Assets

The sample map JSON is served from /maps/sample.map.json in the public folder.

## Forex Strategy Research Lab

This repository now also includes a standalone Python research framework under `forex_research_lab/` for evaluating FX strategies with walk-forward validation, bootstrap robustness checks, cost-aware execution, and experiment tracking.

### Prototype scope

- multi-symbol sample universe: EURUSD, GBPUSD, USDJPY, AUDUSD
- multi-timeframe support: H1, H4, D1
- initial working strategies:
  - trend: moving average crossover, Donchian breakout
  - mean reversion: RSI reversal, Bollinger fade
  - breakout: range breakout, volatility expansion breakout
  - carry: carry proxy placeholder
- combinations: confirmation, ensemble, specialist sleeves
- orchestration: rule-based, performance-based, ATR/ADX regime switching
- research workflows:
  - parameter sweeps
  - walk-forward optimization and testing
  - bootstrap robustness analysis
  - SQLite / dataframe experiment tracking

### Run the prototype

```bash
python3 -m pip install -r requirements.txt
python3 run_forex_research_lab.py
```

The script will:

1. generate deterministic sample FX CSV data in `sample_data/forex/`
2. resample it into H1/H4/D1 research frames
3. run cost-aware walk-forward experiments for:
   - MA crossover
   - RSI mean reversion
4. save artifacts to `artifacts/forex_strategy_research_lab/`

### Output artifacts

Each experiment directory includes:

- `equity_curve.csv` and `equity_curve.png`
- `drawdown_curve.csv` and `drawdown_curve.png`
- `metrics.csv` / `metrics.json`
- `walk_forward_splits.csv`
- `trades.csv`
- `parameter_robustness_heatmap.csv` / `parameter_robustness_heatmap.png`

The top-level output folder also includes:

- `metrics_summary.csv`
- `experiment_log.csv`
- `experiment_runs.sqlite`
