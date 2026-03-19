"""Walk-forward optimization engine."""

from __future__ import annotations

from dataclasses import dataclass
from json import dumps

import pandas as pd

from forex_research_lab.metrics import compute_metrics

from .sweep import ParameterSweep


@dataclass(slots=True)
class WalkForwardResult:
    folds: pd.DataFrame
    oos_frame: pd.DataFrame
    oos_trades: pd.DataFrame
    aggregate_metrics: dict[str, float]
    sweep_results: pd.DataFrame


def _extract_trades_from_frame(frame: pd.DataFrame) -> pd.DataFrame:
    active = frame[frame["position"] != 0].copy()
    if active.empty:
        return pd.DataFrame(
            columns=[
                "trade_id",
                "entry_time",
                "exit_time",
                "direction",
                "bars",
                "return_pct",
                "pnl",
            ]
        )

    active["trade_id"] = active["position"].ne(active["position"].shift(1)).cumsum()
    trades = []
    for trade_id, group in active.groupby("trade_id", sort=True):
        equity_before = (
            frame.loc[group.index[0] - 1, "equity"]
            if group.index[0] > frame.index.min()
            else frame["equity"].iloc[0]
        )
        equity_after = group["equity"].iloc[-1]
        trades.append(
            {
                "trade_id": int(trade_id),
                "entry_time": group["timestamp"].iloc[0],
                "exit_time": group["timestamp"].iloc[-1],
                "direction": int(group["position"].iloc[0]),
                "bars": int(len(group)),
                "return_pct": float((1.0 + group["net_return"]).prod() - 1.0),
                "pnl": float(equity_after - equity_before),
            }
        )
    return pd.DataFrame(trades)


def run_walk_forward(
    strategy,
    df: pd.DataFrame,
    param_grid: dict[str, list],
    timeframe: str,
    train_bars: int,
    test_bars: int,
    objective: str = "sharpe",
    tracker=None,
    run_prefix: str = "walk_forward",
) -> WalkForwardResult:
    """Optimize on rolling train windows and evaluate on the next test window."""

    sweep = ParameterSweep()
    fold_rows: list[dict] = []
    all_oos_frames: list[pd.DataFrame] = []
    all_sweep_results: list[pd.DataFrame] = []

    fold_number = 0
    start = 0
    while start + train_bars + test_bars <= len(df):
        train_end = start + train_bars
        test_end = train_end + test_bars
        train_df = df.iloc[start:train_end].reset_index(drop=True)
        test_df = df.iloc[train_end:test_end].reset_index(drop=True)
        context_df = df.iloc[start:test_end].reset_index(drop=True)

        sweep_result = sweep.run(
            strategy=strategy,
            df=train_df,
            param_grid=param_grid,
            timeframe=timeframe,
            objective=objective,
        )
        if not sweep_result.best_params:
            break

        all_sweep_results.append(
            sweep_result.results.assign(fold=fold_number, split="train")
        )

        train_signals = strategy.generate_signals(train_df, sweep_result.best_params)
        from forex_research_lab.execution import run_backtest

        train_result = run_backtest(train_df, train_signals)
        train_metrics = compute_metrics(train_result.frame, train_result.trades, timeframe=timeframe)

        context_signals = strategy.generate_signals(context_df, sweep_result.best_params)
        context_result = run_backtest(context_df, context_signals)
        test_mask = context_result.frame["timestamp"].isin(test_df["timestamp"])
        oos_frame = context_result.frame.loc[test_mask].copy()
        oos_frame["fold"] = fold_number
        oos_trades = _extract_trades_from_frame(oos_frame)
        oos_metrics = compute_metrics(oos_frame, oos_trades, timeframe=timeframe)

        best_params_text = dumps(sweep_result.best_params, sort_keys=True)
        fold_rows.append(
            {
                "fold": fold_number,
                "train_start": train_df["timestamp"].iloc[0],
                "train_end": train_df["timestamp"].iloc[-1],
                "test_start": test_df["timestamp"].iloc[0],
                "test_end": test_df["timestamp"].iloc[-1],
                "best_params": best_params_text,
                "train_objective": float(train_metrics.get(objective, 0.0)),
                "test_objective": float(oos_metrics.get(objective, 0.0)),
            }
        )
        all_oos_frames.append(oos_frame)

        if tracker is not None:
            run_id = f"{run_prefix}_fold_{fold_number}"
            tracker.log_run(
                run_id=run_id,
                strategy=strategy.name,
                params=best_params_text,
                symbol=str(df["symbol"].iloc[0]),
                timeframe=timeframe,
                split="train",
                metrics=train_metrics,
            )
            tracker.log_run(
                run_id=run_id,
                strategy=strategy.name,
                params=best_params_text,
                symbol=str(df["symbol"].iloc[0]),
                timeframe=timeframe,
                split="test",
                metrics=oos_metrics,
            )

        fold_number += 1
        start += test_bars

    if not all_oos_frames:
        empty = pd.DataFrame()
        return WalkForwardResult(
            folds=pd.DataFrame(fold_rows),
            oos_frame=empty,
            oos_trades=empty,
            aggregate_metrics={},
            sweep_results=pd.DataFrame(),
        )

    oos_frame = pd.concat(all_oos_frames, ignore_index=True)
    oos_trades = _extract_trades_from_frame(oos_frame)
    aggregate_metrics = compute_metrics(oos_frame, oos_trades, timeframe=timeframe)
    sweep_results = pd.concat(all_sweep_results, ignore_index=True)

    return WalkForwardResult(
        folds=pd.DataFrame(fold_rows),
        oos_frame=oos_frame,
        oos_trades=oos_trades,
        aggregate_metrics=aggregate_metrics,
        sweep_results=sweep_results,
    )
