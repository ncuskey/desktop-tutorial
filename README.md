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

This repository also contains a Python prototype research framework under `forex_research_lab/` for evaluating Forex strategies with:

- multi-symbol OHLCV loading and resampling
- indicators and cost models
- modular trend / mean-reversion / breakout / carry strategies
- realistic execution with spread, slippage, and commissions
- walk-forward testing, bootstrap robustness checks, and experiment tracking

Run the prototype with:

```bash
python3 scripts/run_forex_prototype.py
```

Outputs are written to `research_outputs/prototype/`.
