"""Walk-forward optimization and evaluation engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from forex_research_lab.data.costs import ExecutionCostModel
from forex_research_lab.execution import run_backtest
from forex_research_lab.metrics import compute_metrics
from forex_research_lab.research.parameter_sweep import grid_search, select_best_params


StrategyFn = Callable[[pd.DataFrame, Mapping[str, float]], pd.Series]


@dataclass(frozen=True)
class WalkForwardConfig:
    train_bars: int
    test_bars: int
    step_bars: int | None = None
    objective: str = "sharpe"


@dataclass
class WalkForwardResult:
    segment_table: pd.DataFrame
    aggregate_returns: pd.Series
    aggregate_equity_curve: pd.Series
    aggregate_drawdown_curve: pd.Series
    aggregate_trades: pd.DataFrame
    aggregate_metrics: dict[str, float]


def run_walk_forward(
    df: pd.DataFrame,
    strategy_fn: StrategyFn,
    param_grid: Mapping[str, Sequence[Any]],
    cost_model: ExecutionCostModel,
    periods_per_year: int,
    config: WalkForwardConfig,
) -> WalkForwardResult:
    """
    Run rolling train/test windows:
    1) optimize on train
    2) evaluate chosen parameters on the next test segment
    """

    step = config.step_bars or config.test_bars
    start = 0
    segment_id = 0

    segment_rows: list[dict[str, Any]] = []
    all_test_returns: list[pd.Series] = []
    all_test_trades: list[pd.DataFrame] = []

    while start + config.train_bars + config.test_bars <= len(df):
        train_end = start + config.train_bars
        test_end = train_end + config.test_bars

        train_df = df.iloc[start:train_end]
        test_df = df.iloc[train_end:test_end]

        train_results = grid_search(
            df=train_df,
            strategy_fn=strategy_fn,
            param_grid=param_grid,
            cost_model=cost_model,
            periods_per_year=periods_per_year,
        )
        best_params = select_best_params(train_results, objective=config.objective)
        best_train_row = train_results.sort_values(by=config.objective, ascending=False).iloc[0]

        test_signal = strategy_fn(test_df, best_params)
        test_bt = run_backtest(df=test_df, signal=test_signal, cost_model=cost_model)
        test_metrics = compute_metrics(
            returns=test_bt.net_returns,
            equity_curve=test_bt.equity_curve,
            drawdown_curve=test_bt.drawdown_curve,
            trades=test_bt.trades,
            periods_per_year=periods_per_year,
        )

        segment_record: dict[str, Any] = {
            "segment_id": segment_id,
            "train_start": train_df.index[0],
            "train_end": train_df.index[-1],
            "test_start": test_df.index[0],
            "test_end": test_df.index[-1],
            "best_params": str(best_params),
            f"train_{config.objective}": float(best_train_row[config.objective]),
        }
        segment_record.update({f"test_{k}": v for k, v in test_metrics.items()})
        segment_rows.append(segment_record)

        all_test_returns.append(test_bt.net_returns)
        if not test_bt.trades.empty:
            trades = test_bt.trades.copy()
            trades["segment_id"] = segment_id
            all_test_trades.append(trades)

        start += step
        segment_id += 1

    if not all_test_returns:
        raise ValueError("No walk-forward windows were generated. Increase dataset size or shrink windows.")

    aggregate_returns = pd.concat(all_test_returns).sort_index()
    if aggregate_returns.index.has_duplicates:
        aggregate_returns = aggregate_returns.groupby(level=0).mean()

    aggregate_equity_curve = (1.0 + aggregate_returns).cumprod() * 100_000.0
    aggregate_drawdown_curve = aggregate_equity_curve / aggregate_equity_curve.cummax() - 1.0

    aggregate_trades = pd.concat(all_test_trades, ignore_index=True) if all_test_trades else pd.DataFrame()
    aggregate_metrics = compute_metrics(
        returns=aggregate_returns,
        equity_curve=aggregate_equity_curve,
        drawdown_curve=aggregate_drawdown_curve,
        trades=aggregate_trades,
        periods_per_year=periods_per_year,
    )

    segment_table = pd.DataFrame(segment_rows)
    return WalkForwardResult(
        segment_table=segment_table,
        aggregate_returns=aggregate_returns,
        aggregate_equity_curve=aggregate_equity_curve,
        aggregate_drawdown_curve=aggregate_drawdown_curve,
        aggregate_trades=aggregate_trades,
        aggregate_metrics=aggregate_metrics,
    )
