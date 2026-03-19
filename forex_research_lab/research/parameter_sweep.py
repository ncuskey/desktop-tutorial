"""Parameter search utilities."""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from forex_research_lab.execution.backtester import run_backtest
from forex_research_lab.metrics.performance import compute_performance_metrics
from forex_research_lab.strategies.base import BaseStrategy


def expand_parameter_grid(
    parameter_space: dict[str, list[Any]] | None,
    search_method: str = "grid",
    n_iter: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Expand a parameter dictionary into candidate configurations."""
    if not parameter_space:
        return [{}]

    keys = list(parameter_space.keys())
    value_product = list(product(*(parameter_space[key] for key in keys)))
    candidates = [dict(zip(keys, values, strict=True)) for values in value_product]

    if search_method == "grid":
        return candidates
    if search_method != "random":
        raise ValueError(f"Unsupported search method: {search_method}")

    if n_iter is None or n_iter >= len(candidates):
        return candidates

    generator = np.random.default_rng(seed)
    selected_indices = generator.choice(len(candidates), size=n_iter, replace=False)
    return [candidates[index] for index in selected_indices]


def _sort_results(results: pd.DataFrame, objective: str) -> pd.DataFrame:
    ascending = objective in {"Max Drawdown", "Risk of Ruin"}
    return results.sort_values(by=objective, ascending=ascending, ignore_index=True)


def run_parameter_sweep(
    dataframe: pd.DataFrame,
    strategy: BaseStrategy,
    parameter_space: dict[str, list[Any]] | None,
    objective: str = "Sharpe",
    search_method: str = "grid",
    n_iter: int | None = None,
    initial_capital: float = 100_000.0,
    symbol: str = "UNKNOWN",
    timeframe: str = "UNKNOWN",
    tracker: Any | None = None,
) -> pd.DataFrame:
    """Evaluate a parameter grid or random sample and return a ranked dataframe."""
    candidates = expand_parameter_grid(parameter_space, search_method=search_method, n_iter=n_iter)
    records: list[dict[str, Any]] = []

    for params in candidates:
        try:
            signal = strategy.generate_signals(dataframe, params=params)
        except ValueError:
            continue

        backtest = run_backtest(dataframe, signal, initial_capital=initial_capital)
        metrics = compute_performance_metrics(backtest.net_returns, backtest.equity_curve, backtest.trades)

        record: dict[str, Any] = {
            "strategy": strategy.name,
            "symbol": symbol,
            "timeframe": timeframe,
            "params": params,
        }
        record.update(params)
        record.update(metrics)
        records.append(record)

        if tracker is not None:
            tracker.log_run(
                strategy=strategy.name,
                params=params,
                symbol=symbol,
                timeframe=timeframe,
                metrics=metrics,
                context={"phase": "parameter_sweep"},
            )

    results = pd.DataFrame(records)
    if results.empty:
        return results
    return _sort_results(results, objective=objective)
