# Forex Strategy Research Lab

This package provides a modular quantitative research framework for Forex strategy evaluation.

## Module layout

- `data/` — loading, resampling, indicators, and cost model columns
- `strategies/` — trend / mean reversion / breakout / carry signal generators
- `execution/` — signal-to-position conversion, costs, trade extraction, equity curve
- `metrics/` — performance metrics
- `research/` — walk-forward, bootstrap, parameter sweep, and experiment tracking
- `combinations/` — confirmation, ensemble, and specialist sleeves
- `orchestrators/` — rule-based, performance-based, and regime switching
- `visualization/` — equity/drawdown/heatmap plotting helpers

## Prototype run

From repository root:

```bash
python3 scripts/run_prototype.py
```

Outputs are written to:

- `outputs/forex_research_prototype/metrics_table.csv`
- `outputs/forex_research_prototype/equity_curves.png`
- `outputs/forex_research_prototype/drawdown_curves.png`
- `outputs/forex_research_prototype/parameter_robustness_heatmap_ma.png`
- plus walk-forward/bootstrapping/experiment log artifacts
