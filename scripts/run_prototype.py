#!/usr/bin/env python3
"""Run the minimal working Forex Strategy Research Lab prototype."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from forex_research_lab.data import (
    ExecutionCostModel,
    add_indicators,
    attach_cost_columns,
    ensure_sample_data,
    load_multi_symbol_data,
    resample_symbol_map,
)
from forex_research_lab.execution import periods_per_year_from_timeframe, run_backtest
from forex_research_lab.metrics import compute_metrics
from forex_research_lab.research import (
    ExperimentTracker,
    WalkForwardConfig,
    bootstrap_returns,
    grid_search,
    run_walk_forward,
    to_heatmap_matrix,
)
from forex_research_lab.strategies import (
    ma_crossover_generate_signals,
    rsi_reversal_generate_signals,
)
from forex_research_lab.visualization import (
    plot_drawdown_curves,
    plot_equity_curves,
    plot_heatmap,
)


matplotlib.use("Agg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Forex Strategy Research Lab prototype")
    parser.add_argument("--symbol", type=str, default="EURUSD", help="Symbol to evaluate for prototype")
    parser.add_argument("--timeframe", type=str, default="H1", choices=["H1", "H4", "D1"])
    parser.add_argument("--data-dir", type=str, default="sample_data")
    parser.add_argument("--output-dir", type=str, default="outputs/forex_research_prototype")
    parser.add_argument("--periods", type=int, default=6000, help="H1 bars to generate for sample datasets")
    parser.add_argument("--bootstrap-samples", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
    sample_data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load sample OHLC data (generated if missing)
    ensure_sample_data(data_dir=sample_data_dir, symbols=symbols, periods=args.periods)
    h1_data = load_multi_symbol_data(sample_data_dir, symbols=symbols, timeframe="H1")
    multi_tf_data = resample_symbol_map(h1_data, timeframes=["H1", "H4", "D1"])

    symbol = args.symbol.upper()
    timeframe = args.timeframe.upper()
    if symbol not in multi_tf_data:
        raise ValueError(f"Unknown symbol: {symbol}")

    df = multi_tf_data[symbol][timeframe]
    df = attach_cost_columns(df, spread_pips=1.0, symbol=symbol)
    df = add_indicators(df)
    periods_per_year = periods_per_year_from_timeframe(timeframe)

    cost_model = ExecutionCostModel(
        spread_pips=1.0,
        slippage_bps=0.4,
        commission_bps=0.1,
    )

    # 2) Initial strategy runs (MA crossover + RSI mean reversion)
    ma_params = {"fast_window": 20, "slow_window": 60}
    rsi_params = {"rsi_window": 14, "lower": 30, "upper": 70, "exit_level": 50}

    ma_signal = ma_crossover_generate_signals(df, ma_params)
    rsi_signal = rsi_reversal_generate_signals(df, rsi_params)

    ma_bt = run_backtest(df=df, signal=ma_signal, cost_model=cost_model)
    rsi_bt = run_backtest(df=df, signal=rsi_signal, cost_model=cost_model)

    ma_metrics = compute_metrics(
        returns=ma_bt.net_returns,
        equity_curve=ma_bt.equity_curve,
        drawdown_curve=ma_bt.drawdown_curve,
        trades=ma_bt.trades,
        periods_per_year=periods_per_year,
    )
    rsi_metrics = compute_metrics(
        returns=rsi_bt.net_returns,
        equity_curve=rsi_bt.equity_curve,
        drawdown_curve=rsi_bt.drawdown_curve,
        trades=rsi_bt.trades,
        periods_per_year=periods_per_year,
    )

    # 3) Walk-forward evaluation (strict rolling train/test)
    if timeframe == "H1":
        train_bars = 24 * 120
        test_bars = 24 * 30
    elif timeframe == "H4":
        train_bars = 6 * 120
        test_bars = 6 * 30
    else:
        train_bars = 252 * 2
        test_bars = 63

    wf_config = WalkForwardConfig(
        train_bars=train_bars,
        test_bars=test_bars,
        step_bars=test_bars,
        objective="sharpe",
    )

    ma_param_grid = {
        "fast_window": [10, 20, 30],
        "slow_window": [40, 60, 90],
    }
    rsi_param_grid = {
        "rsi_window": [10, 14, 20],
        "lower": [25, 30, 35],
        "upper": [65, 70, 75],
        "exit_level": [45, 50, 55],
    }

    wf_ma = run_walk_forward(
        df=df,
        strategy_fn=ma_crossover_generate_signals,
        param_grid=ma_param_grid,
        cost_model=cost_model,
        periods_per_year=periods_per_year,
        config=wf_config,
    )
    wf_rsi = run_walk_forward(
        df=df,
        strategy_fn=rsi_reversal_generate_signals,
        param_grid=rsi_param_grid,
        cost_model=cost_model,
        periods_per_year=periods_per_year,
        config=wf_config,
    )

    # 4) Bootstrap robustness testing on walk-forward OOS returns
    ma_bootstrap = bootstrap_returns(
        wf_ma.aggregate_returns,
        n_samples=args.bootstrap_samples,
        periods_per_year=periods_per_year,
    )
    rsi_bootstrap = bootstrap_returns(
        wf_rsi.aggregate_returns,
        n_samples=args.bootstrap_samples,
        periods_per_year=periods_per_year,
    )

    # 5) Parameter sweep artifact + heatmap
    ma_sweep = grid_search(
        df=df,
        strategy_fn=ma_crossover_generate_signals,
        param_grid=ma_param_grid,
        cost_model=cost_model,
        periods_per_year=periods_per_year,
    )
    heatmap = to_heatmap_matrix(
        ma_sweep,
        x_param="fast_window",
        y_param="slow_window",
        value_col="sharpe",
    )

    # 6) Experiment tracking (DataFrame + SQLite)
    tracker = ExperimentTracker(db_path=output_dir / "experiments.sqlite")
    tracker.log_run(
        run_name="prototype_static",
        strategy="trend_ma_crossover",
        params=ma_params,
        symbol=symbol,
        timeframe=timeframe,
        metrics=ma_metrics,
    )
    tracker.log_run(
        run_name="prototype_static",
        strategy="mean_reversion_rsi",
        params=rsi_params,
        symbol=symbol,
        timeframe=timeframe,
        metrics=rsi_metrics,
    )
    tracker.log_run(
        run_name="prototype_walk_forward",
        strategy="trend_ma_crossover",
        params={"grid": str(ma_param_grid)},
        symbol=symbol,
        timeframe=timeframe,
        metrics=wf_ma.aggregate_metrics,
    )
    tracker.log_run(
        run_name="prototype_walk_forward",
        strategy="mean_reversion_rsi",
        params={"grid": str(rsi_param_grid)},
        symbol=symbol,
        timeframe=timeframe,
        metrics=wf_rsi.aggregate_metrics,
    )

    # 7) Persist outputs
    metrics_rows = [
        {"experiment": "static_ma_crossover", "symbol": symbol, "timeframe": timeframe, **ma_metrics},
        {"experiment": "static_rsi_reversal", "symbol": symbol, "timeframe": timeframe, **rsi_metrics},
        {"experiment": "walk_forward_ma_crossover", "symbol": symbol, "timeframe": timeframe, **wf_ma.aggregate_metrics},
        {"experiment": "walk_forward_rsi_reversal", "symbol": symbol, "timeframe": timeframe, **wf_rsi.aggregate_metrics},
    ]
    metrics_table = pd.DataFrame(metrics_rows)
    metrics_table.to_csv(output_dir / "metrics_table.csv", index=False)

    tracker.to_dataframe().to_csv(output_dir / "experiment_log.csv", index=False)
    wf_ma.segment_table.to_csv(output_dir / "walk_forward_segments_ma.csv", index=False)
    wf_rsi.segment_table.to_csv(output_dir / "walk_forward_segments_rsi.csv", index=False)
    ma_sweep.to_csv(output_dir / "parameter_sweep_ma.csv", index=False)
    ma_bootstrap.distribution.to_csv(output_dir / "bootstrap_distribution_ma.csv", index=False)
    rsi_bootstrap.distribution.to_csv(output_dir / "bootstrap_distribution_rsi.csv", index=False)

    summary_payload = {
        "ma_bootstrap_summary": ma_bootstrap.summary,
        "rsi_bootstrap_summary": rsi_bootstrap.summary,
    }
    (output_dir / "bootstrap_summary.json").write_text(json.dumps(summary_payload, indent=2))

    plot_equity_curves(
        equity_curves={
            "MA Static": ma_bt.equity_curve,
            "RSI Static": rsi_bt.equity_curve,
            "MA Walk-Forward OOS": wf_ma.aggregate_equity_curve,
            "RSI Walk-Forward OOS": wf_rsi.aggregate_equity_curve,
        },
        output_path=output_dir / "equity_curves.png",
    )
    plot_drawdown_curves(
        drawdown_curves={
            "MA Static": ma_bt.drawdown_curve,
            "RSI Static": rsi_bt.drawdown_curve,
            "MA Walk-Forward OOS": wf_ma.aggregate_drawdown_curve,
            "RSI Walk-Forward OOS": wf_rsi.aggregate_drawdown_curve,
        },
        output_path=output_dir / "drawdown_curves.png",
    )
    plot_heatmap(
        matrix=heatmap,
        title="MA Crossover Robustness (Sharpe)",
        output_path=output_dir / "parameter_robustness_heatmap_ma.png",
        x_label="fast_window",
        y_label="slow_window",
    )

    print("Prototype run complete.")
    print(f"Output directory: {output_dir.resolve()}")
    print(metrics_table.to_string(index=False))


if __name__ == "__main__":
    main()
