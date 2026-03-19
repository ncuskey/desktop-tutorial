"""Runnable prototype for the Forex Strategy Research Lab."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from forex_research_lab.data import (
    attach_cost_model,
    compute_indicators,
    ensure_sample_ohlcv_csv,
    load_ohlcv_csv,
    resample_ohlcv,
)
from forex_research_lab.execution import run_backtest
from forex_research_lab.metrics import compute_metrics
from forex_research_lab.research import ExperimentTracker, ParameterSweep, bootstrap_returns, run_walk_forward
from forex_research_lab.research.reporting import (
    save_equity_drawdown_plot,
    save_metrics_table,
    save_parameter_heatmap,
)
from forex_research_lab.strategies import MovingAverageCrossoverStrategy, RSIReversalStrategy


TIMEFRAMES = ("H1", "H4", "D1")


def build_market_registry(sample_path: Path) -> tuple[dict[tuple[str, str], pd.DataFrame], pd.DataFrame]:
    raw = load_ohlcv_csv(sample_path)
    market_registry: dict[tuple[str, str], pd.DataFrame] = {}
    inventory_rows: list[dict] = []

    for timeframe in TIMEFRAMES:
        timeframe_frame = raw.copy() if timeframe == "H1" else resample_ohlcv(raw, timeframe)
        timeframe_frame = compute_indicators(attach_cost_model(timeframe_frame))

        for symbol, group in timeframe_frame.groupby("symbol", sort=True):
            prepared = group.reset_index(drop=True)
            market_registry[(symbol, timeframe)] = prepared
            inventory_rows.append(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "bars": len(prepared),
                    "start": prepared["timestamp"].iloc[0],
                    "end": prepared["timestamp"].iloc[-1],
                }
            )

    return market_registry, pd.DataFrame(inventory_rows)


def evaluate_strategy(
    label: str,
    strategy,
    df: pd.DataFrame,
    timeframe: str,
    baseline_params: dict,
    param_grid: dict[str, list],
    tracker: ExperimentTracker,
    output_dir: Path,
) -> pd.DataFrame:
    strategy_dir = output_dir / label
    strategy_dir.mkdir(parents=True, exist_ok=True)

    baseline_signals = strategy.generate_signals(df, baseline_params)
    baseline_result = run_backtest(df, baseline_signals)
    baseline_metrics = compute_metrics(baseline_result.frame, baseline_result.trades, timeframe=timeframe)
    save_equity_drawdown_plot(
        baseline_result.frame,
        strategy_dir / "baseline_equity_drawdown.png",
        title=f"{label} baseline",
    )
    save_metrics_table(baseline_metrics, strategy_dir / "baseline_metrics.csv")
    baseline_result.trades.to_csv(strategy_dir / "baseline_trades.csv", index=False)

    walk_forward = run_walk_forward(
        strategy=strategy,
        df=df,
        param_grid=param_grid,
        timeframe=timeframe,
        train_bars=800,
        test_bars=200,
        objective="sharpe",
        tracker=tracker,
        run_prefix=label,
    )
    if walk_forward.oos_frame.empty:
        raise ValueError(f"Walk-forward evaluation produced no out-of-sample bars for {label}")

    save_equity_drawdown_plot(
        walk_forward.oos_frame,
        strategy_dir / "walk_forward_equity_drawdown.png",
        title=f"{label} walk-forward out-of-sample",
    )
    save_metrics_table(walk_forward.aggregate_metrics, strategy_dir / "walk_forward_metrics.csv")
    walk_forward.folds.to_csv(strategy_dir / "walk_forward_folds.csv", index=False)
    walk_forward.oos_trades.to_csv(strategy_dir / "walk_forward_trades.csv", index=False)
    walk_forward.oos_frame.to_csv(strategy_dir / "walk_forward_returns.csv", index=False)
    walk_forward.sweep_results.to_csv(strategy_dir / "walk_forward_sweeps.csv", index=False)

    if {"short_window", "long_window"}.issubset(walk_forward.sweep_results.columns):
        first_fold = walk_forward.sweep_results.loc[walk_forward.sweep_results["fold"] == 0]
        if not first_fold.empty:
            save_parameter_heatmap(
                first_fold,
                strategy_dir / "parameter_heatmap.png",
                x="short_window",
                y="long_window",
                value="objective",
                title=f"{label} train-fold robustness heatmap",
            )

    bootstrap = bootstrap_returns(walk_forward.oos_frame["net_return"], timeframe=timeframe)
    bootstrap.samples.to_csv(strategy_dir / "bootstrap_samples.csv", index=False)
    save_metrics_table(bootstrap.summary, strategy_dir / "bootstrap_summary.csv")

    full_sweep = ParameterSweep().run(
        strategy=strategy,
        df=df,
        param_grid=param_grid,
        timeframe=timeframe,
        objective="sharpe",
    )
    full_sweep.results.to_csv(strategy_dir / "full_sample_sweep.csv", index=False)

    summary_rows = [
        {"strategy": label, "evaluation": "baseline", **baseline_metrics},
        {
            "strategy": label,
            "evaluation": "walk_forward_oos",
            **walk_forward.aggregate_metrics,
            **{f"bootstrap_{key}": value for key, value in bootstrap.summary.items()},
        },
    ]
    return pd.DataFrame(summary_rows)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sample_path = ensure_sample_ohlcv_csv(repo_root / "sample_data" / "fx_hourly_sample.csv")
    output_dir = repo_root / "research_outputs" / "prototype"
    output_dir.mkdir(parents=True, exist_ok=True)

    market_registry, inventory = build_market_registry(sample_path)
    inventory.to_csv(output_dir / "data_inventory.csv", index=False)

    symbol = "EURUSD"
    timeframe = "H1"
    prototype_df = market_registry[(symbol, timeframe)]

    tracker = ExperimentTracker(output_dir / "experiments.sqlite")
    strategy_summaries = [
        evaluate_strategy(
            label="ma_crossover",
            strategy=MovingAverageCrossoverStrategy(),
            df=prototype_df,
            timeframe=timeframe,
            baseline_params={"short_window": 20, "long_window": 100},
            param_grid={
                "short_window": [10, 20, 30, 40],
                "long_window": [60, 100, 140, 180],
            },
            tracker=tracker,
            output_dir=output_dir,
        ),
        evaluate_strategy(
            label="rsi_mean_reversion",
            strategy=RSIReversalStrategy(),
            df=prototype_df,
            timeframe=timeframe,
            baseline_params={
                "window": 14,
                "oversold": 30,
                "overbought": 70,
                "neutral_level": 50,
                "neutral_band": 5,
            },
            param_grid={
                "window": [10, 14, 21],
                "oversold": [25, 30, 35],
                "overbought": [65, 70, 75],
                "neutral_level": [50],
                "neutral_band": [3, 5],
            },
            tracker=tracker,
            output_dir=output_dir,
        ),
    ]

    tracker.to_frame().to_csv(output_dir / "experiment_log.csv", index=False)
    tracker.close()

    summary = pd.concat(strategy_summaries, ignore_index=True)
    summary.to_csv(output_dir / "prototype_summary.csv", index=False)

    print("Forex Strategy Research Lab prototype complete.")
    print(f"Sample data: {sample_path}")
    print(f"Outputs: {output_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
