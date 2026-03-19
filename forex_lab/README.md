# Forex Strategy Research Lab

Production-oriented quantitative research framework for FX strategy evaluation.

## Structure

- `data/`: load OHLCV, resample timeframes, indicators, transaction cost model.
- `strategies/`: modular strategies returning position series `(-1, 0, +1)`.
- `execution/`: convert signals into net returns/equity with spread/slippage/commission.
- `metrics/`: CAGR, Sharpe, Sortino, max drawdown, profit factor, win rate, expectancy, trade count.
- `research/`: walk-forward testing, parameter sweep, bootstrap robustness, experiment tracking.
- `combinations/`: confirmation, ensembles, specialist sleeves.
- `orchestrators/`: rule-based, performance-based allocation, and simple regime switching.
- `sample_data/`: generated synthetic OHLCV CSVs.
- `outputs/`: experiment artifacts (equity, drawdown, metrics, heatmaps, bootstrap files).

## Prototype Run

From repository root:

```bash
python scripts/run_forex_research_prototype.py
```

The script will:

1. Generate/load sample FX OHLCV data.
2. Run walk-forward evaluation for:
   - MA crossover trend strategy
   - RSI reversal mean-reversion strategy
3. Apply realistic transaction costs.
4. Compute and save metrics, equity/drawdown curves, and parameter robustness heatmap.
5. Save experiment logs to CSV + SQLite.
