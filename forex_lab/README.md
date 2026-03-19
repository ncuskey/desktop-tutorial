# Forex Strategy Research Lab

A modular, production-quality quantitative research framework for evaluating Forex trading strategies.

## Architecture

```
forex_lab/
├── data/               # Data loading, resampling, indicators, cost model
├── strategies/         # Individual strategy implementations
├── execution/          # Signal-to-trade conversion with realistic costs
├── metrics/            # Performance metrics (CAGR, Sharpe, Sortino, etc.)
├── research/           # Walk-forward, bootstrap, parameter sweep, experiment tracking
├── combinations/       # Strategy combination engines (confirmation, ensemble, sleeves)
├── orchestrators/      # Strategy orchestration (rule-based, performance-based, regime)
├── run_prototype.py    # Minimal working prototype
└── requirements.txt    # Python dependencies
```

## Quick Start

```bash
pip install -r forex_lab/requirements.txt
python forex_lab/run_prototype.py
```

## Modules

### data/
- `generate_sample_data()` — synthetic OHLCV for any major pair
- `load_ohlcv()` — load from CSV
- `resample_timeframe()` — resample to H1/H4/D1
- `add_indicators()` — SMA, EMA, RSI, ATR, Bollinger Bands, ADX, Donchian
- `attach_cost_model()` — spread, slippage, commission

### strategies/
Each implements `generate_signals(df, params) -> Series` returning {-1, 0, +1}.

| Strategy | Type | Description |
|----------|------|-------------|
| `MACrossover` | Trend | Moving average crossover |
| `DonchianBreakout` | Trend | Donchian channel breakout |
| `RSIReversal` | Mean Reversion | RSI overbought/oversold |
| `BollingerFade` | Mean Reversion | Bollinger band fade |
| `RangeBreakout` | Breakout | N-bar range breakout |
| `VolatilityExpansion` | Breakout | ATR expansion breakout |
| `CarryStrategy` | Carry | Interest rate differential proxy |

### execution/
- `execute_signals()` — converts signals into trades with spread, slippage, commission; produces equity curve and drawdown

### metrics/
- `compute_metrics()` — CAGR, Sharpe, Sortino, Max Drawdown, Profit Factor, Win Rate, Expectancy, Trade Count
- `metrics_table()` — side-by-side comparison DataFrame

### research/
- `WalkForwardEngine` — rolling train/test optimization with OOS aggregation
- `BootstrapEngine` — Monte Carlo resampling for robustness (Sharpe/DD distributions, risk of ruin)
- `ParameterSweep` — grid or random search over hyperparameters
- `ExperimentTracker` — SQLite-backed experiment logging

### combinations/
- `ConfirmationCombiner` — trade only when N strategies agree
- `EnsembleCombiner` — weighted signal averaging
- `SpecialistSleeves` — conditional strategy activation via filter functions

### orchestrators/
- `RuleBasedOrchestrator` — ADX-based trend/mean-reversion switching
- `PerformanceBasedOrchestrator` — rolling Sharpe-based strategy selection
- `RegimeOrchestrator` — ATR/ADX regime classification with strategy routing

## Validation Principles

- **No lookahead bias** — all signals use `.shift()` and rolling windows
- **Walk-forward testing** — rolling optimization prevents in-sample overfitting
- **Bootstrap validation** — Monte Carlo resampling estimates true performance distributions
- **Realistic costs** — spread, slippage (bps), and commission applied to every trade

## Supported Pairs & Timeframes

- Pairs: EURUSD, GBPUSD, USDJPY, AUDUSD
- Timeframes: H1, H4, D1
