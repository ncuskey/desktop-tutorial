# Forex Strategy Research Lab

A production-quality quantitative research framework for evaluating Forex trading strategies. Designed to test individual strategies, combinations, and orchestration layers while enforcing walk-forward validation and bootstrap robustness testing to avoid overfitting.

---

## Architecture

```
forex_lab/
├── data/
│   ├── loader.py          # OHLCV loading + synthetic GBM data generator
│   ├── resampler.py       # Multi-timeframe resampling (H1 → H4 → D1)
│   ├── indicators.py      # MA, RSI, ATR, Bollinger Bands, ADX, Donchian, MACD
│   └── costs.py           # Spread / slippage / commission cost model
│
├── strategies/
│   ├── base.py            # BaseStrategy interface
│   ├── trend.py           # MACrossover, DonchianBreakout
│   ├── mean_reversion.py  # RSIReversal, BollingerFade
│   ├── breakout.py        # RangeBreakout, VolatilityExpansionBreakout
│   └── carry.py           # CarryProxy (rate differential)
│
├── execution/
│   └── engine.py          # ExecutionEngine: signals → equity curve (with costs)
│
├── metrics/
│   └── performance.py     # CAGR, Sharpe, Sortino, MaxDD, Calmar, PF, Win Rate…
│
├── research/
│   ├── walk_forward.py    # Rolling train/optimise/test walk-forward engine
│   ├── bootstrap.py       # Block bootstrap + trade bootstrap robustness engine
│   ├── param_sweep.py     # Grid + random parameter search
│   └── experiment_tracker.py  # SQLite-backed experiment logging
│
├── combinations/
│   ├── confirmation.py    # Require N-of-M strategy agreement
│   ├── ensemble.py        # Weighted signal averaging
│   └── sleeves.py         # Conditionally activated specialist strategies
│
├── orchestrators/
│   ├── rule_based.py      # ADX/ATR dispatch rules
│   ├── performance_based.py # Rolling Sharpe-weighted allocation
│   └── regime.py          # ATR+ADX regime classifier + strategy switcher
│
├── visualization.py       # Equity curves, drawdowns, heatmaps, bootstrap charts
├── main.py                # Full prototype demonstration
└── requirements.txt
```

---

## Quick Start

```bash
cd forex_lab
pip install -r requirements.txt
python3 main.py
```

Outputs (charts + SQLite) are written to `./output/`.

---

## Using Real Data

```python
from data.loader import load_ohlcv
df = load_ohlcv("path/to/EURUSD_H4.csv")
```

CSV must have columns: `open, high, low, close` (volume optional).  
Index should be a parseable datetime column.

---

## Running a Single Strategy

```python
import sys; sys.path.insert(0, ".")

from data.loader import generate_synthetic_ohlcv
from data.indicators import add_indicators
from strategies.trend import MACrossover
from execution.engine import ExecutionEngine
from metrics.performance import compute_metrics

df = add_indicators(generate_synthetic_ohlcv("EURUSD"))
strategy = MACrossover()
signals = strategy.generate_signals(df, {"fast_period": 20, "slow_period": 50})

engine = ExecutionEngine()
result = engine.run(df, signals, "EURUSD", "ma_crossover")
metrics = compute_metrics(result.net_returns, result.trades)
print(metrics)
```

---

## Walk-Forward Analysis

```python
from research.walk_forward import WalkForwardEngine
from strategies.trend import MACrossover

wf = WalkForwardEngine(train_bars=2000, test_bars=500, optimise_metric="sharpe")
result = wf.run(
    df,
    MACrossover(),
    param_grid={
        "fast_period": [10, 20, 30],
        "slow_period": [40, 60, 80],
        "ma_type": ["sma", "ema"],
    },
    symbol="EURUSD",
)
print(result.oos_metrics)
print(result.summary())
```

---

## Bootstrap Robustness

```python
from research.bootstrap import BootstrapEngine

bs = BootstrapEngine(n_samples=1000, method="block", block_size=20)
bs_result = bs.run_on_returns(result.net_returns, result.trades)
print(bs_result.summary())
# p_value_zero tells you the probability the strategy is due to chance
```

---

## Adding a New Strategy

1. Create a class in `strategies/` inheriting from `BaseStrategy`.
2. Implement `generate_signals(df, params) -> pd.Series`.
3. Return values must be in `{-1, 0, +1}` aligned to `df.index`.
4. No future data access — use `.shift(1)` when referencing indicators.

```python
class MyStrategy(BaseStrategy):
    name = "my_strategy"

    def generate_signals(self, df, params):
        # your logic here
        signal = pd.Series(0, index=df.index, dtype="int8")
        # ... fill signal ...
        return self._clip_signals(signal)
```

---

## Supported Symbols

| Symbol | Default Spread | Pip Size |
|--------|---------------|----------|
| EURUSD | 1.0 pip       | 0.0001   |
| GBPUSD | 1.5 pip       | 0.0001   |
| USDJPY | 1.2 pip       | 0.01     |
| AUDUSD | 1.8 pip       | 0.0001   |

---

## Validation Principles

- **No lookahead bias**: all indicators use `.shift(1)` before signal generation; execution is delayed by 1 bar.
- **Walk-forward required**: static backtests overfit; rolling optimisation+testing better simulates live conditions.
- **Bootstrap validation**: strong backtests can still fail robustness tests; bootstrap distributions reveal whether performance is statistically significant.
- **Strict train/test separation**: walk-forward windows never overlap between train and test segments.

---

## Output Files

| File | Description |
|------|-------------|
| `equity_*.png` | Equity curve + drawdown chart |
| `wf_*.png` | Walk-forward per-window equity |
| `bootstrap_*.png` | Bootstrap metric distribution |
| `heatmap_*.png` | Parameter robustness heatmap |
| `experiments.db` | SQLite database of all logged runs |
