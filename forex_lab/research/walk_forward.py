from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

import pandas as pd

from forex_lab.execution import BacktestResult, backtest_signals
from forex_lab.metrics import compute_metrics


@dataclass
class WalkForwardResult:
    aggregated_result: BacktestResult
    fold_results: pd.DataFrame
    best_params_by_fold: List[Dict]
    aggregate_metrics: Dict[str, float]


def _score_sharpe(df: pd.DataFrame, signal_fn: Callable, params: Dict, timeframe: str) -> float:
    signal = signal_fn(df, params)
    result = backtest_signals(df, signal)
    metrics = compute_metrics(result.equity_curve, result.returns, result.trades, timeframe=timeframe)
    return metrics["Sharpe"]


def run_walk_forward(
    df: pd.DataFrame,
    signal_fn: Callable[[pd.DataFrame, Dict], pd.Series],
    param_candidates: Sequence[Dict],
    train_bars: int,
    test_bars: int,
    timeframe: str = "H1",
) -> WalkForwardResult:
    if train_bars <= 0 or test_bars <= 0:
        raise ValueError("train_bars and test_bars must be positive")
    if len(df) < (train_bars + test_bars):
        raise ValueError("Not enough rows for a single walk-forward fold")

    fold_rows = []
    best_params = []
    stitched_returns = []
    stitched_gross = []
    stitched_costs = []
    stitched_pos = []
    stitched_trades = []

    fold_id = 0
    start = 0
    while start + train_bars + test_bars <= len(df):
        train_df = df.iloc[start : start + train_bars]
        test_df = df.iloc[start + train_bars : start + train_bars + test_bars]

        best = None
        best_score = float("-inf")
        for params in param_candidates:
            score = _score_sharpe(train_df, signal_fn=signal_fn, params=params, timeframe=timeframe)
            if score > best_score:
                best_score = score
                best = params
        assert best is not None

        test_signal = signal_fn(test_df, best)
        test_result = backtest_signals(test_df, test_signal)
        test_metrics = compute_metrics(
            test_result.equity_curve,
            test_result.returns,
            test_result.trades,
            timeframe=timeframe,
        )

        fold_rows.append(
            {
                "fold_id": fold_id,
                "train_start": train_df.index[0],
                "train_end": train_df.index[-1],
                "test_start": test_df.index[0],
                "test_end": test_df.index[-1],
                "best_params": best,
                **test_metrics,
            }
        )
        best_params.append(best)
        stitched_returns.append(test_result.returns)
        stitched_gross.append(test_result.gross_returns)
        stitched_costs.append(test_result.costs)
        stitched_pos.append(test_result.effective_position)
        if not test_result.trades.empty:
            trades = test_result.trades.copy()
            trades["fold_id"] = fold_id
            stitched_trades.append(trades)

        fold_id += 1
        start += test_bars

    all_returns = pd.concat(stitched_returns).sort_index()
    all_gross = pd.concat(stitched_gross).sort_index()
    all_costs = pd.concat(stitched_costs).sort_index()
    all_position = pd.concat(stitched_pos).sort_index()
    equity_curve = (1 + all_returns).cumprod() * 100_000.0
    drawdown_curve = equity_curve / equity_curve.cummax() - 1
    all_trades = (
        pd.concat(stitched_trades, ignore_index=True)
        if stitched_trades
        else pd.DataFrame(columns=["entry_time", "exit_time", "side", "bars", "trade_return", "fold_id"])
    )
    aggregated_result = BacktestResult(
        equity_curve=equity_curve,
        drawdown_curve=drawdown_curve,
        returns=all_returns,
        gross_returns=all_gross,
        costs=all_costs,
        effective_position=all_position,
        trades=all_trades,
    )

    aggregate_metrics = compute_metrics(
        aggregated_result.equity_curve,
        aggregated_result.returns,
        aggregated_result.trades,
        timeframe=timeframe,
    )
    return WalkForwardResult(
        aggregated_result=aggregated_result,
        fold_results=pd.DataFrame(fold_rows),
        best_params_by_fold=best_params,
        aggregate_metrics=aggregate_metrics,
    )
