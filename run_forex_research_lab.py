#!/usr/bin/env python3
"""Run the Forex Strategy Research Lab prototype end to end."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from forex_research_lab.data import (
    add_basic_indicators,
    attach_cost_model,
    ensure_sample_data,
    load_ohlcv_directory,
    prepare_multi_timeframe,
)
from forex_research_lab.research import ExperimentTracker, bootstrap_returns, run_walk_forward, save_experiment_outputs
from forex_research_lab.strategies import MovingAverageCrossoverStrategy, RSIReversalStrategy


DEFAULT_SYMBOLS = ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD")
DEFAULT_TIMEFRAMES = ("H1", "H4", "D1")
WALK_FORWARD_WINDOWS = {
    "H1": {"train_size": 24 * 120, "test_size": 24 * 30},
    "H4": {"train_size": 6 * 120, "test_size": 6 * 30},
    "D1": {"train_size": 180, "test_size": 60},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Forex Strategy Research Lab prototype")
    parser.add_argument("--symbols", nargs="*", default=list(DEFAULT_SYMBOLS), help="Symbols to include")
    parser.add_argument("--timeframes", nargs="*", default=list(DEFAULT_TIMEFRAMES), help="Timeframes to evaluate")
    parser.add_argument("--periods", type=int, default=24 * 365, help="Number of hourly bars to generate per symbol")
    parser.add_argument("--sample-data-dir", default="sample_data/forex", help="Directory for generated sample CSVs")
    parser.add_argument("--output-dir", default="artifacts/forex_strategy_research_lab", help="Directory for experiment outputs")
    parser.add_argument("--bootstrap-samples", type=int, default=250, help="Bootstrap sample count for the best experiment")
    return parser.parse_args()


def build_research_universe(
    sample_data_dir: Path,
    symbols: tuple[str, ...],
    timeframes: tuple[str, ...],
    periods: int,
) -> dict[str, dict[str, pd.DataFrame]]:
    ensure_sample_data(sample_data_dir, symbols=symbols, periods=periods)
    raw_data = load_ohlcv_directory(sample_data_dir)
    resampled = prepare_multi_timeframe(raw_data, timeframes=timeframes)

    universe: dict[str, dict[str, pd.DataFrame]] = {}
    for symbol, timeframe_map in resampled.items():
        if symbol not in symbols:
            continue
        universe[symbol] = {}
        for timeframe, dataframe in timeframe_map.items():
            enriched = add_basic_indicators(dataframe)
            universe[symbol][timeframe] = attach_cost_model(enriched, symbol=symbol)
    return universe


def run_prototype(
    universe: dict[str, dict[str, pd.DataFrame]],
    output_dir: Path,
    bootstrap_samples: int,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    tracker = ExperimentTracker(output_dir / "experiment_runs.sqlite")

    strategy_configs = [
        (
            MovingAverageCrossoverStrategy(),
            {"short_window": [10, 20, 30], "long_window": [50, 100, 150]},
        ),
        (
            RSIReversalStrategy(),
            {"window": [10, 14, 21], "entry_threshold": [25, 30, 35]},
        ),
    ]

    summary_rows: list[dict[str, float | str]] = []
    best_experiment: tuple[str, str, str, object] | None = None
    best_sharpe = float("-inf")

    for symbol, timeframe_map in universe.items():
        for timeframe, dataframe in timeframe_map.items():
            if timeframe not in WALK_FORWARD_WINDOWS:
                continue

            window_config = WALK_FORWARD_WINDOWS[timeframe]
            if len(dataframe) < window_config["train_size"] + window_config["test_size"]:
                continue

            for strategy, parameter_space in strategy_configs:
                result = run_walk_forward(
                    dataframe=dataframe,
                    strategy=strategy,
                    parameter_space=parameter_space,
                    train_size=window_config["train_size"],
                    test_size=window_config["test_size"],
                    step_size=window_config["test_size"],
                    symbol=symbol,
                    timeframe=timeframe,
                    tracker=tracker,
                )
                if result.split_summary.empty:
                    continue

                experiment_dir = output_dir / symbol / timeframe / strategy.name
                save_experiment_outputs(result, experiment_dir)

                summary_row: dict[str, float | str] = {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "strategy": strategy.name,
                    "splits": float(len(result.split_summary)),
                }
                summary_row.update(result.aggregate_metrics)
                summary_rows.append(summary_row)

                strategy_sharpe = float(result.aggregate_metrics.get("Sharpe", float("-inf")))
                if strategy_sharpe > best_sharpe:
                    best_sharpe = strategy_sharpe
                    best_experiment = (symbol, timeframe, strategy.name, result)

    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        raise RuntimeError("No experiments ran. Check the generated data length and walk-forward settings.")

    summary = summary.sort_values(by=["Sharpe", "CAGR"], ascending=[False, False], ignore_index=True)
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    tracker.to_frame().to_csv(output_dir / "experiment_log.csv", index=False)

    if best_experiment is not None:
        symbol, timeframe, strategy_name, result = best_experiment
        bootstrap = bootstrap_returns(result.test_returns, n_samples=bootstrap_samples)
        bootstrap_dir = output_dir / symbol / timeframe / strategy_name
        bootstrap.samples.to_csv(bootstrap_dir / "bootstrap_samples.csv", index=False)
        pd.DataFrame([bootstrap.summary]).to_csv(bootstrap_dir / "bootstrap_summary.csv", index=False)

    return summary


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    sample_data_dir = project_root / args.sample_data_dir
    output_dir = project_root / args.output_dir

    symbols = tuple(symbol.upper() for symbol in args.symbols)
    timeframes = tuple(timeframe.upper() for timeframe in args.timeframes)
    universe = build_research_universe(sample_data_dir, symbols=symbols, timeframes=timeframes, periods=args.periods)
    summary = run_prototype(universe, output_dir=output_dir, bootstrap_samples=args.bootstrap_samples)

    with pd.option_context("display.max_columns", None, "display.width", 160):
        print("Forex Strategy Research Lab prototype complete.")
        print(f"Artifacts written to: {output_dir}")
        print(summary.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
