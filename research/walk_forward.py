from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from data.costs import CostModel
from execution.simulator import BacktestResult, run_backtest
from metrics.performance import compute_metrics
from research.parameter_sweep import grid_parameter_sweep


@dataclass
class WalkForwardResult:
    fold_results: pd.DataFrame
    combined_returns: pd.Series
    combined_equity: pd.Series
    combined_drawdown: pd.Series
    aggregate_metrics: dict[str, float]


def _backtest_with_params(
    df: pd.DataFrame,
    strategy_fn: Callable[[pd.DataFrame, dict], pd.Series],
    params: dict,
    cost_model: CostModel,
) -> BacktestResult:
    signal = strategy_fn(df, params)
    return run_backtest(df, signal, cost_model=cost_model)


def run_walk_forward(
    df: pd.DataFrame,
    strategy_fn: Callable[[pd.DataFrame, dict], pd.Series],
    param_grid: dict[str, list],
    train_bars: int,
    test_bars: int,
    cost_model: CostModel,
    timeframe: str = "H1",
    objective_metric: str = "Sharpe",
) -> WalkForwardResult:
    folds: list[dict] = []
    stitched_returns: list[np.ndarray] = []
    stitched_index: list = []

    i = 0
    while i + train_bars + test_bars <= len(df):
        train_df = df.iloc[i : i + train_bars].copy()
        test_df = df.iloc[i + train_bars : i + train_bars + test_bars].copy()

        sweep = grid_parameter_sweep(
            train_df,
            strategy_fn=strategy_fn,
            param_grid=param_grid,
            cost_model=cost_model,
            timeframe=timeframe,
        )
        best = sweep.iloc[0]
        best_params = best["params"]

        test_bt = _backtest_with_params(
            test_df, strategy_fn=strategy_fn, params=best_params, cost_model=cost_model
        )
        test_metrics = compute_metrics(
            test_bt.returns, test_bt.equity, test_bt.trades, timeframe=timeframe
        )

        folds.append(
            {
                "fold_start": train_df["timestamp"].iloc[0],
                "fold_train_end": train_df["timestamp"].iloc[-1],
                "fold_test_end": test_df["timestamp"].iloc[-1],
                "best_params": best_params,
                "train_objective": float(best[objective_metric]),
                **{f"test_{k}": v for k, v in test_metrics.items()},
            }
        )

        stitched_returns.append(test_bt.returns.to_numpy())
        stitched_index.extend(test_df["timestamp"].values.tolist())
        i += test_bars

    if not stitched_returns:
        raise ValueError("Not enough data for one train/test fold.")

    combined_returns = pd.Series(
        data=np.concatenate(stitched_returns),
        index=pd.to_datetime(stitched_index, utc=True),
        name="returns",
    ).sort_index()
    combined_equity = (1.0 + combined_returns).cumprod() * 100_000.0
    combined_drawdown = (combined_equity / combined_equity.cummax()) - 1.0

    aggregate_metrics = compute_metrics(
        combined_returns, combined_equity, pd.DataFrame(), timeframe=timeframe
    )

    return WalkForwardResult(
        fold_results=pd.DataFrame(folds),
        combined_returns=combined_returns,
        combined_equity=combined_equity,
        combined_drawdown=combined_drawdown,
        aggregate_metrics=aggregate_metrics,
    )
