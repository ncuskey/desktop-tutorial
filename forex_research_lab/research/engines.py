"""Research engines for validation and experiment management."""

from __future__ import annotations

import json
import sqlite3
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from forex_research_lab.execution.backtester import evaluate_strategy
from forex_research_lab.metrics.performance import compute_metrics
from forex_research_lab.types import BacktestArtifactPaths, WalkForwardResult, WalkForwardSplit

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


class ParameterSweepEngine:
    """Grid-search parameter combinations and rank outcomes."""

    def run(
        self,
        df: pd.DataFrame,
        strategy: Any,
        param_grid: dict[str, list[Any]],
        *,
        timeframe: str,
        objective: str = "sharpe",
    ) -> pd.DataFrame:
        if not param_grid:
            raise ValueError("param_grid must contain at least one parameter.")

        keys = list(param_grid)
        records: list[dict[str, Any]] = []

        for values in product(*(param_grid[key] for key in keys)):
            params = dict(zip(keys, values, strict=True))
            result = evaluate_strategy(df, strategy, params, timeframe=timeframe, initial_capital=100_000.0)
            record = {"params": params, **result.metrics}
            records.append(record)

        results = pd.DataFrame.from_records(records)
        if objective not in results.columns:
            raise ValueError(f"Objective '{objective}' is not available in sweep results.")
        return results.sort_values(objective, ascending=False).reset_index(drop=True)


class WalkForwardEngine:
    """Rolling train/test walk-forward evaluation."""

    def __init__(self, *, train_bars: int, test_bars: int, step_bars: int | None = None) -> None:
        self.train_bars = train_bars
        self.test_bars = test_bars
        self.step_bars = step_bars or test_bars
        self.sweep_engine = ParameterSweepEngine()

    def run(
        self,
        df: pd.DataFrame,
        strategy: Any,
        param_grid: dict[str, list[Any]],
        *,
        timeframe: str,
        objective: str = "sharpe",
        initial_capital: float = 100_000.0,
    ) -> WalkForwardResult:
        ordered = df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
        unique_timestamps = pd.Index(sorted(ordered["timestamp"].unique()))
        if len(unique_timestamps) < self.train_bars + self.test_bars:
            raise ValueError("Not enough bars for the configured walk-forward windows.")

        split_metadata: list[WalkForwardSplit] = []
        split_metric_records: list[dict[str, Any]] = []
        parameter_frames: list[pd.DataFrame] = []
        aggregated_frames: list[pd.DataFrame] = []
        aggregated_trades: list[pd.DataFrame] = []

        split_id = 0
        for start in range(0, len(unique_timestamps) - self.train_bars - self.test_bars + 1, self.step_bars):
            train_times = unique_timestamps[start : start + self.train_bars]
            test_times = unique_timestamps[start + self.train_bars : start + self.train_bars + self.test_bars]

            train_df = ordered[ordered["timestamp"].isin(train_times)].copy()
            test_df = ordered[ordered["timestamp"].isin(test_times)].copy()

            sweep = self.sweep_engine.run(
                train_df,
                strategy,
                param_grid,
                timeframe=timeframe,
                objective=objective,
            )
            sweep["split_id"] = split_id
            parameter_frames.append(sweep)

            best_row = sweep.iloc[0]
            best_params = dict(best_row["params"])
            test_result = evaluate_strategy(
                test_df,
                strategy,
                best_params,
                timeframe=timeframe,
                initial_capital=1.0,
            )

            split_frame = test_result.frame[["timestamp", "portfolio_return"]].copy()
            split_frame["split_id"] = split_id
            aggregated_frames.append(split_frame)

            split_trades = test_result.trades.copy()
            split_trades["split_id"] = split_id
            aggregated_trades.append(split_trades)

            split_metrics = {
                "split_id": split_id,
                "train_start": train_times[0],
                "train_end": train_times[-1],
                "test_start": test_times[0],
                "test_end": test_times[-1],
                "best_params": json.dumps(best_params, sort_keys=True),
                "train_objective": float(best_row[objective]),
                **test_result.metrics,
            }
            split_metric_records.append(split_metrics)
            split_metadata.append(
                WalkForwardSplit(
                    split_id=split_id,
                    train_start=pd.Timestamp(train_times[0]),
                    train_end=pd.Timestamp(train_times[-1]),
                    test_start=pd.Timestamp(test_times[0]),
                    test_end=pd.Timestamp(test_times[-1]),
                    best_params=best_params,
                    train_metric=float(best_row[objective]),
                )
            )
            split_id += 1

        aggregated_frame = pd.concat(aggregated_frames, ignore_index=True).sort_values("timestamp")
        aggregated_frame["equity"] = initial_capital * (1 + aggregated_frame["portfolio_return"]).cumprod()
        aggregated_frame["drawdown"] = aggregated_frame["equity"] / aggregated_frame["equity"].cummax() - 1

        trades = pd.concat(aggregated_trades, ignore_index=True) if aggregated_trades else pd.DataFrame()
        return WalkForwardResult(
            aggregated_frame=aggregated_frame.reset_index(drop=True),
            aggregated_trades=trades,
            split_metrics=pd.DataFrame.from_records(split_metric_records),
            splits=split_metadata,
            parameter_results=pd.concat(parameter_frames, ignore_index=True),
        )


class BootstrapEngine:
    """Bootstrap robustness testing over returns or trades."""

    def __init__(self, *, n_bootstrap: int = 250, seed: int = 42) -> None:
        self.n_bootstrap = n_bootstrap
        self.seed = seed

    def run(
        self,
        *,
        returns: pd.Series,
        trades: pd.DataFrame | None,
        timeframe: str,
        initial_capital: float = 100_000.0,
        ruin_threshold: float = 0.7,
    ) -> pd.DataFrame:
        source = (
            trades["trade_return"].dropna().astype(float).reset_index(drop=True)
            if trades is not None and not trades.empty and "trade_return" in trades.columns
            else returns.dropna().astype(float).reset_index(drop=True)
        )
        if source.empty:
            raise ValueError("Bootstrap requires at least one return or trade observation.")

        rng = np.random.default_rng(self.seed)
        records: list[dict[str, Any]] = []

        for sample_id in range(self.n_bootstrap):
            sample = source.iloc[rng.integers(0, len(source), len(source))].reset_index(drop=True)
            equity = initial_capital * (1 + sample).cumprod()
            sample_trades = pd.DataFrame({"trade_return": sample})
            metrics = compute_metrics(
                returns=sample,
                trades=sample_trades,
                timeframe=timeframe,
                initial_capital=initial_capital,
                equity=equity,
            )
            ruined = bool((equity <= initial_capital * ruin_threshold).any())
            records.append(
                {
                    "sample_id": sample_id,
                    "ruined": ruined,
                    **metrics,
                }
            )

        return pd.DataFrame.from_records(records)


class ExperimentTracker:
    """SQLite-backed experiment logger."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    logged_at TEXT NOT NULL,
                    experiment_name TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    metrics_json TEXT NOT NULL
                )
                """
            )

    def log_run(
        self,
        *,
        experiment_name: str,
        strategy: str,
        params: dict[str, Any],
        symbol: str,
        timeframe: str,
        metrics: dict[str, float],
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO experiments (
                    logged_at, experiment_name, strategy, params_json, symbol, timeframe, metrics_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pd.Timestamp.utcnow().isoformat(),
                    experiment_name,
                    strategy,
                    json.dumps(params, sort_keys=True),
                    symbol,
                    timeframe,
                    json.dumps(metrics, sort_keys=True),
                ),
            )

    def to_frame(self) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query("SELECT * FROM experiments ORDER BY run_id", conn)


def export_experiment_outputs(
    *,
    experiment_name: str,
    aggregated_frame: pd.DataFrame,
    metrics_table: pd.DataFrame,
    parameter_results: pd.DataFrame,
    output_dir: str | Path,
    heatmap_x: str | None = None,
    heatmap_y: str | None = None,
    heatmap_metric: str = "sharpe",
) -> BacktestArtifactPaths:
    """Persist research outputs to CSV and PNG files."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    safe_name = experiment_name.replace(" ", "_").lower()
    equity_curve_csv = output_path / f"{safe_name}_equity_curve.csv"
    drawdown_curve_csv = output_path / f"{safe_name}_drawdown_curve.csv"
    metrics_csv = output_path / f"{safe_name}_metrics.csv"
    heatmap_csv = output_path / f"{safe_name}_heatmap.csv"
    equity_curve_png = output_path / f"{safe_name}_equity_curve.png"
    drawdown_curve_png = output_path / f"{safe_name}_drawdown_curve.png"
    heatmap_png = output_path / f"{safe_name}_heatmap.png"

    aggregated_frame[["timestamp", "equity"]].to_csv(equity_curve_csv, index=False)
    aggregated_frame[["timestamp", "drawdown"]].to_csv(drawdown_curve_csv, index=False)
    metrics_table.to_csv(metrics_csv, index=False)

    plt.figure(figsize=(10, 4))
    plt.plot(aggregated_frame["timestamp"], aggregated_frame["equity"], label="Equity")
    plt.title(f"{experiment_name} Equity Curve")
    plt.xlabel("Timestamp")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig(equity_curve_png)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(aggregated_frame["timestamp"], aggregated_frame["drawdown"], label="Drawdown", color="tab:red")
    plt.title(f"{experiment_name} Drawdown Curve")
    plt.xlabel("Timestamp")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(drawdown_curve_png)
    plt.close()

    heatmap_frame = pd.DataFrame()
    if not parameter_results.empty and heatmap_x and heatmap_y:
        working = parameter_results.copy()
        working[heatmap_x] = working["params"].apply(lambda item: item.get(heatmap_x))
        working[heatmap_y] = working["params"].apply(lambda item: item.get(heatmap_y))
        heatmap_frame = working.pivot_table(
            index=heatmap_y,
            columns=heatmap_x,
            values=heatmap_metric,
            aggfunc="mean",
        ).sort_index().sort_index(axis=1)
        heatmap_frame.to_csv(heatmap_csv)

        plt.figure(figsize=(6, 5))
        plt.imshow(heatmap_frame.values, aspect="auto", origin="lower", cmap="viridis")
        plt.colorbar(label=heatmap_metric)
        plt.xticks(range(len(heatmap_frame.columns)), heatmap_frame.columns)
        plt.yticks(range(len(heatmap_frame.index)), heatmap_frame.index)
        plt.xlabel(heatmap_x)
        plt.ylabel(heatmap_y)
        plt.title(f"{experiment_name} Parameter Robustness")
        plt.tight_layout()
        plt.savefig(heatmap_png)
        plt.close()
    else:
        heatmap_frame.to_csv(heatmap_csv, index=False)
        heatmap_png = None

    return BacktestArtifactPaths(
        base_dir=output_path,
        equity_curve_csv=equity_curve_csv,
        drawdown_curve_csv=drawdown_curve_csv,
        metrics_csv=metrics_csv,
        heatmap_csv=heatmap_csv,
        experiment_log_db=output_path / "experiment_tracking.sqlite",
        equity_curve_png=equity_curve_png,
        drawdown_curve_png=drawdown_curve_png,
        heatmap_png=heatmap_png,
    )
