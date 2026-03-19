"""Parameter sweep utilities for systematic strategy optimization."""

from __future__ import annotations

from itertools import product
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from forex_research_lab.data.costs import ExecutionCostModel
from forex_research_lab.execution import run_backtest
from forex_research_lab.metrics import compute_metrics


StrategyFn = Callable[[pd.DataFrame, Mapping[str, float]], pd.Series]


def iter_param_grid(param_grid: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    keys = list(param_grid.keys())
    values = [param_grid[key] for key in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def random_param_samples(
    param_space: Mapping[str, Sequence[Any]],
    n_samples: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    keys = list(param_space.keys())
    out: list[dict[str, Any]] = []
    for _ in range(n_samples):
        sample: dict[str, Any] = {}
        for key in keys:
            choices = param_space[key]
            sample[key] = choices[int(rng.integers(0, len(choices)))]
        out.append(sample)
    return out


def _evaluate_once(
    df: pd.DataFrame,
    strategy_fn: StrategyFn,
    params: Mapping[str, Any],
    cost_model: ExecutionCostModel,
    periods_per_year: int,
) -> dict[str, Any]:
    signal = strategy_fn(df, params)
    bt = run_backtest(df=df, signal=signal, cost_model=cost_model)
    metrics = compute_metrics(
        returns=bt.net_returns,
        equity_curve=bt.equity_curve,
        drawdown_curve=bt.drawdown_curve,
        trades=bt.trades,
        periods_per_year=periods_per_year,
    )
    row = {**params, **metrics}
    return row


def grid_search(
    df: pd.DataFrame,
    strategy_fn: StrategyFn,
    param_grid: Mapping[str, Sequence[Any]],
    cost_model: ExecutionCostModel,
    periods_per_year: int,
) -> pd.DataFrame:
    rows = []
    for params in iter_param_grid(param_grid):
        rows.append(
            _evaluate_once(
                df=df,
                strategy_fn=strategy_fn,
                params=params,
                cost_model=cost_model,
                periods_per_year=periods_per_year,
            )
        )
    return pd.DataFrame(rows)


def random_search(
    df: pd.DataFrame,
    strategy_fn: StrategyFn,
    param_space: Mapping[str, Sequence[Any]],
    n_samples: int,
    cost_model: ExecutionCostModel,
    periods_per_year: int,
    seed: int = 42,
) -> pd.DataFrame:
    rows = []
    for params in random_param_samples(param_space, n_samples=n_samples, seed=seed):
        rows.append(
            _evaluate_once(
                df=df,
                strategy_fn=strategy_fn,
                params=params,
                cost_model=cost_model,
                periods_per_year=periods_per_year,
            )
        )
    return pd.DataFrame(rows)


def select_best_params(results: pd.DataFrame, objective: str = "sharpe") -> dict[str, Any]:
    if results.empty:
        raise ValueError("Parameter sweep results are empty")
    if objective not in results.columns:
        raise ValueError(f"Objective column '{objective}' not found")

    ranked = results.sort_values(by=objective, ascending=False)
    best = ranked.iloc[0].to_dict()
    metric_cols = {
        "cagr",
        "sharpe",
        "sortino",
        "max_drawdown",
        "profit_factor",
        "win_rate",
        "expectancy",
        "trade_count",
    }
    return {k: v for k, v in best.items() if k not in metric_cols}


def to_heatmap_matrix(
    results: pd.DataFrame,
    x_param: str,
    y_param: str,
    value_col: str = "sharpe",
) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    pivot = results.pivot_table(index=y_param, columns=x_param, values=value_col, aggfunc="mean")
    return pivot.sort_index().sort_index(axis=1)
