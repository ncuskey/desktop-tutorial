# Forex Strategy Research Lab

A production-quality quantitative research platform for Forex strategy evaluation. Supports multi-symbol, multi-timeframe testing with realistic execution costs, walk-forward analysis, and bootstrap robustness validation.

## Features

- **Data**: OHLCV loading, resampling, indicators (MA, RSI, ATR, Bollinger, ADX), cost model
- **Strategies**: Trend (MA crossover, Donchian), Mean reversion (RSI, Bollinger), Breakout, Carry
- **Execution**: Spread, slippage, commission modeling; equity curve
- **Metrics**: CAGR, Sharpe, Sortino, Max Drawdown, Profit Factor, Win Rate, Expectancy
- **Research**: Walk-forward engine, bootstrap engine, parameter sweep, experiment tracking
- **Combinations**: Confirmation, ensemble, specialist sleeves
- **Orchestrators**: Rule-based, performance-based, regime classifier

## Quick Start

```bash
pip install -r requirements.txt
python run_prototype.py
```

## Structure

```
forex_lab/
├── data/           # Load, resample, indicators, costs
├── strategies/     # MA crossover, RSI, Donchian, etc.
├── execution/      # Trade execution, costs, equity
├── metrics/        # Performance metrics
├── research/       # Walk-forward, bootstrap, parameter sweep
├── combinations/   # Confirmation, ensemble, sleeves
├── orchestrators/  # Rule-based, performance, regime
└── run_prototype.py
```

## Validation Principles

- No lookahead bias
- Strict train/test separation
- Walk-forward testing (rolling windows)
- Bootstrap robustness testing
