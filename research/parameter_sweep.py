from __future__ import annotations

from itertools import product
from typing import Callable

import numpy as np
import pandas as pd

from data.costs import CostModel
from execution.simulator import run_backtest
from metrics.performance import compute_metrics


def _iter_grid(param_grid: dict[str, list]) -> list[dict]:
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    return [dict(zip(keys, combo, strict=False)) for combo in product(*values)]


def grid_parameter_sweep(
    df: pd.DataFrame,
    strategy_fn: Callable[[pd.DataFrame, dict], pd.Series],
    param_grid: dict[str, list],
    cost_model: CostModel,
    timeframe: str = "H1",
) -> pd.DataFrame:
    rows: list[dict] = []
    for params in _iter_grid(param_grid):
        signal = strategy_fn(df, params)
        bt = run_backtest(df, signal, cost_model=cost_model)
        metrics = compute_metrics(bt.returns, bt.equity, bt.trades, timeframe=timeframe)
        rows.append({"params": params, **metrics})
    out = pd.DataFrame(rows)
    return out.sort_values("Sharpe", ascending=False).reset_index(drop=True)


def random_parameter_sweep(
    df: pd.DataFrame,
    strategy_fn: Callable[[pd.DataFrame, dict], pd.Series],
    param_space: dict[str, list],
    cost_model: CostModel,
    n_samples: int = 25,
    timeframe: str = "H1",
    seed: int = 13,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    keys = list(param_space.keys())
    for _ in range(n_samples):
        params = {k: rng.choice(param_space[k]) for k in keys}
        signal = strategy_fn(df, params)
        bt = run_backtest(df, signal, cost_model=cost_model)
        metrics = compute_metrics(bt.returns, bt.equity, bt.trades, timeframe=timeframe)
        rows.append({"params": params, **metrics})
    out = pd.DataFrame(rows)
    return out.sort_values("Sharpe", ascending=False).reset_index(drop=True)
