"""Run the minimal Forex Strategy Research Lab prototype."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from forex_research_lab.data import (  # noqa: E402
    attach_cost_model,
    compute_basic_indicators,
    generate_mock_ohlcv,
    load_ohlcv_csv,
    resample_ohlcv,
)
from forex_research_lab.execution import evaluate_strategy  # noqa: E402
from forex_research_lab.metrics import compute_metrics  # noqa: E402
from forex_research_lab.research import (  # noqa: E402
    BootstrapEngine,
    ExperimentTracker,
    WalkForwardEngine,
    export_experiment_outputs,
)
from forex_research_lab.strategies import (  # noqa: E402
    MovingAverageCrossoverStrategy,
    RSIReversalStrategy,
)


@dataclass(slots=True)
class PrototypeExperiment:
    name: str
    strategy: object
    default_params: dict[str, float | int]
    param_grid: dict[str, list[float | int]]
    heatmap_x: str
    heatmap_y: str


def ensure_sample_data(sample_path: Path) -> Path:
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    if not sample_path.exists():
        sample = generate_mock_ohlcv(periods=24 * 360)
        sample.to_csv(sample_path, index=False)
    return sample_path


def run_prototype() -> None:
    sample_path = ensure_sample_data(REPO_ROOT / "sample_data" / "forex_mock_h1.csv")
    output_dir = REPO_ROOT / "research_outputs" / "prototype"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_h1 = attach_cost_model(load_ohlcv_csv(sample_path))
    h1 = compute_basic_indicators(raw_h1)
    h4 = compute_basic_indicators(resample_ohlcv(raw_h1, "4h"))
    d1 = compute_basic_indicators(resample_ohlcv(raw_h1, "1D"))

    tracker = ExperimentTracker(output_dir / "experiment_tracking.sqlite")
    walk_forward = WalkForwardEngine(train_bars=24 * 120, test_bars=24 * 30, step_bars=24 * 30)
    bootstrap = BootstrapEngine(n_bootstrap=200, seed=42)

    experiments = [
        PrototypeExperiment(
            name="ma_crossover",
            strategy=MovingAverageCrossoverStrategy(),
            default_params={"fast_window": 20, "slow_window": 60},
            param_grid={"fast_window": [10, 20, 30], "slow_window": [50, 70, 90]},
            heatmap_x="fast_window",
            heatmap_y="slow_window",
        ),
        PrototypeExperiment(
            name="rsi_reversal",
            strategy=RSIReversalStrategy(),
            default_params={"window": 14, "oversold": 30, "overbought": 70, "exit_level": 50},
            param_grid={
                "window": [10, 14, 18],
                "oversold": [25, 30, 35],
                "overbought": [65, 70, 75],
                "exit_level": [45, 50, 55],
            },
            heatmap_x="oversold",
            heatmap_y="overbought",
        ),
    ]

    overall_summary: list[dict[str, float | str]] = []

    for experiment in experiments:
        baseline = evaluate_strategy(
            h1,
            experiment.strategy,
            experiment.default_params,
            timeframe="H1",
            initial_capital=100_000.0,
        )
        walk_forward_result = walk_forward.run(
            h1,
            experiment.strategy,
            experiment.param_grid,
            timeframe="H1",
            initial_capital=100_000.0,
        )
        walk_forward_metrics = compute_metrics(
            returns=walk_forward_result.aggregated_frame["portfolio_return"],
            trades=walk_forward_result.aggregated_trades,
            timeframe="H1",
            initial_capital=100_000.0,
            equity=walk_forward_result.aggregated_frame["equity"],
        )

        strategy_output_dir = output_dir / experiment.name
        strategy_output_dir.mkdir(parents=True, exist_ok=True)

        metrics_table = pd.DataFrame(
            [
                {"strategy": experiment.name, "run_type": "baseline", **baseline.metrics},
                {"strategy": experiment.name, "run_type": "walk_forward", **walk_forward_metrics},
            ]
        )
        export_experiment_outputs(
            experiment_name=experiment.name,
            aggregated_frame=walk_forward_result.aggregated_frame,
            metrics_table=metrics_table,
            parameter_results=walk_forward_result.parameter_results,
            output_dir=strategy_output_dir,
            heatmap_x=experiment.heatmap_x,
            heatmap_y=experiment.heatmap_y,
        )

        parameter_results = walk_forward_result.parameter_results.copy()
        parameter_results["params_json"] = parameter_results["params"].apply(
            lambda item: json.dumps(item, sort_keys=True)
        )
        parameter_results.drop(columns=["params"]).to_csv(
            strategy_output_dir / "parameter_sweep_results.csv",
            index=False,
        )
        walk_forward_result.split_metrics.to_csv(strategy_output_dir / "walk_forward_splits.csv", index=False)
        walk_forward_result.aggregated_trades.to_csv(
            strategy_output_dir / "walk_forward_trades.csv",
            index=False,
        )

        bootstrap_distribution = bootstrap.run(
            returns=walk_forward_result.aggregated_frame["portfolio_return"],
            trades=walk_forward_result.aggregated_trades,
            timeframe="H1",
            initial_capital=100_000.0,
        )
        bootstrap_distribution.to_csv(strategy_output_dir / "bootstrap_distribution.csv", index=False)
        bootstrap_summary = pd.DataFrame(
            [
                {
                    "strategy": experiment.name,
                    "risk_of_ruin": float(bootstrap_distribution["ruined"].mean()),
                    "median_sharpe": float(bootstrap_distribution["sharpe"].median()),
                    "p05_sharpe": float(bootstrap_distribution["sharpe"].quantile(0.05)),
                    "median_max_drawdown": float(bootstrap_distribution["max_drawdown"].median()),
                    "p05_max_drawdown": float(bootstrap_distribution["max_drawdown"].quantile(0.05)),
                }
            ]
        )
        bootstrap_summary.to_csv(strategy_output_dir / "bootstrap_summary.csv", index=False)

        tracker.log_run(
            experiment_name="prototype_walk_forward",
            strategy=experiment.name,
            params={"search_space": experiment.param_grid},
            symbol="ALL",
            timeframe="H1",
            metrics=walk_forward_metrics,
        )

        overall_summary.append({"strategy": experiment.name, **walk_forward_metrics})

    pd.DataFrame(overall_summary).to_csv(output_dir / "prototype_summary.csv", index=False)
    tracker.to_frame().to_csv(output_dir / "experiment_log_snapshot.csv", index=False)

    print("Forex Strategy Research Lab prototype completed.")
    print(f"Sample H1 rows: {len(h1):,}")
    print(f"Resampled H4 rows: {len(h4):,}")
    print(f"Resampled D1 rows: {len(d1):,}")
    print(f"Outputs written to: {output_dir}")
    print(pd.DataFrame(overall_summary).round(4).to_string(index=False))


def main() -> None:
    run_prototype()


if __name__ == "__main__":
    main()
