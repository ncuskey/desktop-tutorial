"""Walk-forward optimization with strict train/test separation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from forex_research_lab.execution.backtester import run_backtest
from forex_research_lab.metrics.performance import compute_performance_metrics
from forex_research_lab.research.parameter_sweep import run_parameter_sweep
from forex_research_lab.strategies.base import BaseStrategy


@dataclass
class WalkForwardResult:
    """Aggregate output for a walk-forward experiment."""

    strategy_name: str
    symbol: str
    timeframe: str
    test_returns: pd.Series
    equity_curve: pd.Series
    drawdown_curve: pd.Series
    aggregate_metrics: dict[str, float]
    split_summary: pd.DataFrame
    search_history: list[pd.DataFrame]
    trades: pd.DataFrame


def run_walk_forward(
    dataframe: pd.DataFrame,
    strategy: BaseStrategy,
    parameter_space: dict[str, list[Any]] | None,
    train_size: int,
    test_size: int,
    step_size: int | None = None,
    objective: str = "Sharpe",
    initial_capital: float = 100_000.0,
    symbol: str = "UNKNOWN",
    timeframe: str = "UNKNOWN",
    tracker: Any | None = None,
) -> WalkForwardResult:
    """Run rolling walk-forward optimization without leaking future data."""
    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be positive")

    step = step_size or test_size
    split_records: list[dict[str, Any]] = []
    search_history: list[pd.DataFrame] = []
    test_return_segments: list[pd.Series] = []
    trade_frames: list[pd.DataFrame] = []

    split_number = 0
    max_offset = len(dataframe) - train_size - test_size
    for start in range(0, max(0, max_offset) + 1, step):
        train_end = start + train_size
        test_end = train_end + test_size

        train_frame = dataframe.iloc[start:train_end]
        test_frame = dataframe.iloc[train_end:test_end]
        if train_frame.empty or test_frame.empty:
            continue

        sweep_results = run_parameter_sweep(
            dataframe=train_frame,
            strategy=strategy,
            parameter_space=parameter_space,
            objective=objective,
            initial_capital=initial_capital,
            symbol=symbol,
            timeframe=timeframe,
            tracker=tracker,
        )
        if sweep_results.empty:
            continue

        sweep_results = sweep_results.copy()
        sweep_results["split_id"] = split_number
        search_history.append(sweep_results)

        best_row = sweep_results.iloc[0]
        best_params = dict(best_row["params"])

        train_signal = strategy.generate_signals(train_frame, params=best_params)
        train_backtest = run_backtest(train_frame, train_signal, initial_capital=initial_capital)
        train_metrics = compute_performance_metrics(train_backtest.net_returns, train_backtest.equity_curve, train_backtest.trades)

        full_history = dataframe.iloc[:test_end]
        full_signal = strategy.generate_signals(full_history, params=best_params)
        initial_position = float(full_signal.iloc[train_end - 1]) if train_end > 0 else 0.0
        previous_close = float(dataframe["close"].iloc[train_end - 1]) if train_end > 0 else None
        test_signal = full_signal.iloc[train_end:test_end]

        test_backtest = run_backtest(
            dataframe=test_frame,
            position=test_signal,
            initial_capital=initial_capital,
            initial_position=initial_position,
            previous_close=previous_close,
        )
        test_metrics = compute_performance_metrics(test_backtest.net_returns, test_backtest.equity_curve, test_backtest.trades)

        test_return_segments.append(test_backtest.net_returns.rename(f"split_{split_number}"))
        if not test_backtest.trades.empty:
            trade_frame = test_backtest.trades.copy()
            trade_frame["split_id"] = split_number
            trade_frames.append(trade_frame)

        split_records.append(
            {
                "split_id": split_number,
                "train_start": train_frame.index[0],
                "train_end": train_frame.index[-1],
                "test_start": test_frame.index[0],
                "test_end": test_frame.index[-1],
                "best_params": best_params,
                "train_objective": float(best_row[objective]),
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"test_{key}": value for key, value in test_metrics.items()},
            }
        )

        if tracker is not None:
            tracker.log_run(
                strategy=strategy.name,
                params=best_params,
                symbol=symbol,
                timeframe=timeframe,
                metrics=test_metrics,
                context={"phase": "walk_forward_test", "split_id": split_number},
            )

        split_number += 1

    if test_return_segments:
        aggregate_returns = pd.concat(test_return_segments).sort_index()
        aggregate_returns = aggregate_returns[~aggregate_returns.index.duplicated(keep="last")]
        aggregate_equity = initial_capital * (1.0 + aggregate_returns).cumprod()
        aggregate_drawdown = aggregate_equity.div(aggregate_equity.cummax()).fillna(1.0) - 1.0
    else:
        aggregate_returns = pd.Series(dtype=float)
        aggregate_equity = pd.Series(dtype=float)
        aggregate_drawdown = pd.Series(dtype=float)

    aggregate_trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    aggregate_metrics = compute_performance_metrics(aggregate_returns, aggregate_equity, aggregate_trades)

    return WalkForwardResult(
        strategy_name=strategy.name,
        symbol=symbol,
        timeframe=timeframe,
        test_returns=aggregate_returns,
        equity_curve=aggregate_equity,
        drawdown_curve=aggregate_drawdown,
        aggregate_metrics=aggregate_metrics,
        split_summary=pd.DataFrame(split_records),
        search_history=search_history,
        trades=aggregate_trades,
    )
