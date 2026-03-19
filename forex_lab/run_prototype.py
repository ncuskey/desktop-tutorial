#!/usr/bin/env python3
"""
Forex Strategy Research Lab - Minimal Working Prototype

1) Load sample OHLC data (mock)
2) Run MA crossover and RSI mean reversion
3) Execute trades with costs
4) Run walk-forward evaluation
5) Output metrics + equity curve

Run: pip install -e . && python forex_lab/run_prototype.py (from workspace root)
"""

from pathlib import Path

import pandas as pd

from forex_lab.data import (
    CostModel,
    compute_indicators,
    generate_mock_data,
)
from forex_lab.execution import ExecutionEngine
from forex_lab.metrics import compute_metrics
from forex_lab.research import WalkForwardEngine
from forex_lab.strategies import MACrossoverStrategy, RSIReversalStrategy


def main():
    print("=" * 60)
    print("Forex Strategy Research Lab - Prototype")
    print("=" * 60)

    # 1) Load sample OHLC data (mock)
    print("\n1) Generating mock OHLC data...")
    df = generate_mock_data(n_bars=5000, symbol="EURUSD", freq="1h", seed=42)
    print(f"   Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    # Compute indicators
    df = compute_indicators(df)
    print("   Indicators computed: MA, RSI, ATR, Bollinger, ADX")

    # Attach cost model
    cost_model = CostModel(spread_bps=10, slippage_bps=5)
    print(f"   Cost model: spread={cost_model.spread_bps} bps, slippage={cost_model.slippage_bps} bps")

    # 2) Run MA crossover and RSI mean reversion
    print("\n2) Running strategies...")

    ma_strategy = MACrossoverStrategy()
    ma_signals = ma_strategy.generate_signals(df, ma_strategy.default_params)
    print(f"   MA Crossover: {int((ma_signals != 0).sum())} non-flat bars")

    rsi_strategy = RSIReversalStrategy()
    rsi_signals = rsi_strategy.generate_signals(df, rsi_strategy.default_params)
    print(f"   RSI Reversal: {int((rsi_signals != 0).sum())} non-flat bars")

    # 3) Execute trades with costs
    print("\n3) Executing trades with costs...")

    engine = ExecutionEngine(cost_model=cost_model, initial_capital=100_000)

    ma_equity, ma_trades, ma_equity_df = engine.run(df, ma_signals, size=0.1)
    rsi_equity, rsi_trades, rsi_equity_df = engine.run(df, rsi_signals, size=0.1)

    ma_metrics = compute_metrics(ma_equity, ma_trades, periods_per_year=8760)
    rsi_metrics = compute_metrics(rsi_equity, rsi_trades, periods_per_year=8760)

    print(f"   MA Crossover: {ma_metrics.trade_count} trades, "
          f"Sharpe={ma_metrics.sharpe:.3f}, MaxDD={ma_metrics.max_drawdown:.2%}")
    print(f"   RSI Reversal: {rsi_metrics.trade_count} trades, "
          f"Sharpe={rsi_metrics.sharpe:.3f}, MaxDD={rsi_metrics.max_drawdown:.2%}")

    # 4) Walk-forward evaluation
    print("\n4) Walk-forward evaluation...")

    wf = WalkForwardEngine(
        train_bars=1000,
        test_bars=500,
        step_bars=500,
        optimization_metric="sharpe",
        execution_engine=engine,
    )

    ma_param_grid = {"fast": [5, 10, 20], "slow": [20, 30, 50]}
    wf_results_ma, wf_equity_ma, wf_trades_ma = wf.run(
        df, lambda: MACrossoverStrategy(), ma_param_grid
    )

    rsi_param_grid = {"period": [14], "threshold_low": [25, 30], "threshold_high": [70, 75]}
    wf_results_rsi, wf_equity_rsi, wf_trades_rsi = wf.run(
        df, lambda: RSIReversalStrategy(), rsi_param_grid
    )

    print(f"   MA Walk-forward: {len(wf_results_ma)} folds")
    for r in wf_results_ma[:3]:
        print(f"      Fold {r['fold']}: OOS Sharpe={r['oos_sharpe']:.3f}, "
              f"trades={r['oos_trades']}, params={r['best_params']}")
    if len(wf_results_ma) > 3:
        print(f"      ...")

    print(f"   RSI Walk-forward: {len(wf_results_rsi)} folds")
    for r in wf_results_rsi[:3]:
        print(f"      Fold {r['fold']}: OOS Sharpe={r['oos_sharpe']:.3f}, "
              f"trades={r['oos_trades']}, params={r['best_params']}")

    # 5) Output metrics + equity curve
    print("\n5) Outputs...")

    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)

    # Metrics table
    metrics_df = pd.DataFrame([
        {
            "strategy": "MA Crossover",
            "cagr": ma_metrics.cagr,
            "sharpe": ma_metrics.sharpe,
            "sortino": ma_metrics.sortino,
            "max_drawdown": ma_metrics.max_drawdown,
            "profit_factor": ma_metrics.profit_factor,
            "win_rate": ma_metrics.win_rate,
            "trade_count": ma_metrics.trade_count,
        },
        {
            "strategy": "RSI Reversal",
            "cagr": rsi_metrics.cagr,
            "sharpe": rsi_metrics.sharpe,
            "sortino": rsi_metrics.sortino,
            "max_drawdown": rsi_metrics.max_drawdown,
            "profit_factor": rsi_metrics.profit_factor,
            "win_rate": rsi_metrics.win_rate,
            "trade_count": rsi_metrics.trade_count,
        },
    ])
    metrics_path = out_dir / "metrics_table.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"   Metrics table: {metrics_path}")

    # Equity curves
    ma_equity_df.to_csv(out_dir / "equity_ma.csv")
    rsi_equity_df.to_csv(out_dir / "equity_rsi.csv")
    print(f"   Equity curves: {out_dir / 'equity_ma.csv'}, {out_dir / 'equity_rsi.csv'}")

    # Walk-forward results
    wf_df = pd.DataFrame(wf_results_ma + wf_results_rsi)
    wf_df.to_csv(out_dir / "walk_forward_results.csv", index=False)
    print(f"   Walk-forward results: {out_dir / 'walk_forward_results.csv'}")

    # Parameter robustness heatmap (MA crossover)
    from forex_lab.research import ParameterSweep
    sweep = ParameterSweep(execution_engine=engine)
    heatmap_df = sweep.grid_search(
        df, lambda: MACrossoverStrategy(),
        {"fast": [5, 10, 15], "slow": [20, 30, 40, 50]},
        metric="sharpe",
    )
    heatmap_df.to_csv(out_dir / "param_robustness_heatmap.csv", index=False)
    print(f"   Parameter robustness heatmap: {out_dir / 'param_robustness_heatmap.csv'}")

    print("\n" + "=" * 60)
    print("Prototype complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
