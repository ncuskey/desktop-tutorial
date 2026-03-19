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

## Forex Strategy Research Lab Prototype

This repository now also includes a self-contained Python package, `forex_research_lab/`, for quantitative Forex strategy research. The lab is designed around modular components so strategies, combinations, and orchestration policies can be evaluated without lookahead bias and with realistic transaction costs.

### Included modules

- `forex_research_lab/data/`: OHLCV loading, resampling, indicators, and spread/cost modeling
- `forex_research_lab/strategies/`: trend, mean reversion, breakout, and carry signal generators
- `forex_research_lab/execution/`: signal-to-trade conversion with spread, slippage, and commissions
- `forex_research_lab/metrics/`: portfolio and trade-level performance statistics
- `forex_research_lab/research/`: parameter sweeps, walk-forward optimization, bootstrap robustness, reporting, and experiment tracking
- `forex_research_lab/combinations/`: confirmation, ensemble, and specialist sleeve combinations
- `forex_research_lab/orchestrators/`: rule-based, performance-based, and regime-based switching

### Python setup

```bash
python3 -m pip install -e .
```

### Run the prototype

```bash
python3 -m forex_research_lab.prototype
```

Or:

```bash
python3 scripts/run_prototype.py
```

The prototype will:

1. Generate sample multi-symbol hourly FX OHLCV data on first run
2. Resample it into `H4` and `D1`
3. Run moving-average crossover and RSI mean reversion strategies
4. Apply spread, slippage, and commission assumptions
5. Perform walk-forward optimization and bootstrap robustness tests
6. Save metrics tables, equity/drawdown charts, experiment logs, and sweep outputs under `research_outputs/prototype/`
