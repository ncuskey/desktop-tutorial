from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from data.costs import CostModel
from research.walk_forward import WalkForwardResult, run_walk_forward


@dataclass
class PurgedWalkForwardResult:
    fold_results: pd.DataFrame
    combined_returns: pd.Series
    combined_equity: pd.Series
    combined_drawdown: pd.Series
    aggregate_metrics: dict[str, float]
    filtered_combined_returns: pd.Series | None = None
    filtered_combined_equity: pd.Series | None = None
    filtered_combined_drawdown: pd.Series | None = None
    filtered_aggregate_metrics: dict[str, float] | None = None
    meta_filter_diagnostics: dict[str, float] | None = None
    purge_bars: int = 0
    embargo_bars: int = 0


def run_purged_walk_forward(
    df: pd.DataFrame,
    strategy_fn: Callable[[pd.DataFrame, dict], pd.Series],
    param_grid: dict[str, list],
    train_bars: int,
    test_bars: int,
    cost_model: CostModel,
    purge_bars: int = 0,
    embargo_bars: int = 0,
    **kwargs: Any,
) -> PurgedWalkForwardResult:
    """
    Walk-forward evaluation with optional purge and embargo windows.

    - purge_bars removes bars between train and test.
    - embargo_bars skips bars between one fold's test end and next fold start.
    """
    if purge_bars < 0 or embargo_bars < 0:
        raise ValueError("purge_bars and embargo_bars must be >= 0")

    fold_results: list[pd.DataFrame] = []
    stitched_returns: list[pd.Series] = []
    stitched_filtered_returns: list[pd.Series] = []
    final_result: WalkForwardResult | None = None

    i = 0
    fold_id = 0
    while i + train_bars + purge_bars + test_bars <= len(df):
        train_df = df.iloc[i : i + train_bars].copy()
        test_start = i + train_bars + purge_bars
        test_df = df.iloc[test_start : test_start + test_bars].copy()
        one_fold_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

        result = run_walk_forward(
            df=one_fold_df,
            strategy_fn=strategy_fn,
            param_grid=param_grid,
            train_bars=train_bars,
            test_bars=test_bars,
            cost_model=cost_model,
            **kwargs,
        )
        fr = result.fold_results.copy()
        fr["fold_id"] = fold_id
        fr["purge_bars"] = int(purge_bars)
        fr["embargo_bars"] = int(embargo_bars)
        fold_results.append(fr)
        stitched_returns.append(result.combined_returns)
        if result.filtered_combined_returns is not None:
            stitched_filtered_returns.append(result.filtered_combined_returns)
        final_result = result

        i = i + test_bars + embargo_bars
        fold_id += 1

    if not fold_results or final_result is None:
        raise ValueError("Not enough data for one purged walk-forward fold.")

    combined_returns = pd.concat(stitched_returns).sort_index()
    combined_equity = (1.0 + combined_returns).cumprod() * 100_000.0
    combined_drawdown = (combined_equity / combined_equity.cummax()) - 1.0

    filtered_combined_returns = None
    filtered_combined_equity = None
    filtered_combined_drawdown = None
    if stitched_filtered_returns:
        filtered_combined_returns = pd.concat(stitched_filtered_returns).sort_index()
        filtered_combined_equity = (1.0 + filtered_combined_returns).cumprod() * 100_000.0
        filtered_combined_drawdown = (
            filtered_combined_equity / filtered_combined_equity.cummax()
        ) - 1.0

    # Keep aggregate metrics from most recent fold-level evaluation logic for consistency.
    aggregate_metrics = final_result.aggregate_metrics
    filtered_aggregate_metrics = final_result.filtered_aggregate_metrics
    meta_filter_diagnostics = final_result.meta_filter_diagnostics

    # Add run-level purge/embargo metadata into diagnostics.
    if meta_filter_diagnostics is not None:
        meta_filter_diagnostics = dict(meta_filter_diagnostics)
        meta_filter_diagnostics["PurgeBars"] = float(purge_bars)
        meta_filter_diagnostics["EmbargoBars"] = float(embargo_bars)

    return PurgedWalkForwardResult(
        fold_results=pd.concat(fold_results, ignore_index=True),
        combined_returns=combined_returns,
        combined_equity=combined_equity,
        combined_drawdown=combined_drawdown,
        aggregate_metrics=aggregate_metrics,
        filtered_combined_returns=filtered_combined_returns,
        filtered_combined_equity=filtered_combined_equity,
        filtered_combined_drawdown=filtered_combined_drawdown,
        filtered_aggregate_metrics=filtered_aggregate_metrics,
        meta_filter_diagnostics=meta_filter_diagnostics,
        purge_bars=int(purge_bars),
        embargo_bars=int(embargo_bars),
    )
