"""Walk-forward optimization and testing engine."""

from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from forex_lab.execution.engine import ExecutionEngine, Trade
from forex_lab.metrics.compute import Metrics, compute_metrics


class WalkForwardEngine:
    """
    Rolling train/test walk-forward analysis.

    - Optimize params on train window
    - Test on next out-of-sample segment
    - Aggregate results across all folds
    """

    def __init__(
        self,
        train_bars: int,
        test_bars: int,
        step_bars: Optional[int] = None,
        optimization_metric: str = "sharpe",
        execution_engine: Optional[ExecutionEngine] = None,
    ):
        self.train_bars = train_bars
        self.test_bars = test_bars
        self.step_bars = step_bars or test_bars
        self.optimization_metric = optimization_metric
        self.execution_engine = execution_engine or ExecutionEngine()

    def run(
        self,
        df: pd.DataFrame,
        strategy_factory: Callable[[], Any],
        param_grid: dict[str, list[Any]],
        optimize_fn: Optional[Callable[[pd.DataFrame, dict], float]] = None,
    ) -> tuple[list[dict], pd.DataFrame, list[Trade]]:
        """
        Run walk-forward evaluation.

        Args:
            df: OHLCV with indicators
            strategy_factory: Callable that returns strategy instance
            param_grid: Dict of param name -> list of values
            optimize_fn: Optional custom optimization (df, params) -> score.
                If None, uses strategy.generate_signals + execution + metric.

        Returns:
            results: List of dicts with fold info, best params, metrics
            equity_curve: Concatenated OOS equity curve
            all_trades: All OOS trades
        """
        n = len(df)
        results = []
        all_equity = []
        all_trades: list[Trade] = []
        test_start_idx = 0

        while test_start_idx + self.train_bars + self.test_bars <= n:
            train_end = test_start_idx + self.train_bars
            test_end = train_end + self.test_bars

            train_df = df.iloc[test_start_idx:train_end].copy()
            test_df = df.iloc[train_end:test_end].copy()

            # Optimize on train
            best_params, best_score = self._optimize(
                train_df,
                strategy_factory,
                param_grid,
                optimize_fn,
            )

            # Test OOS
            strategy = strategy_factory()
            signals = strategy.generate_signals(test_df, best_params)
            equity, trades, _ = self.execution_engine.run(test_df, signals)
            metrics = compute_metrics(equity, trades)

            results.append(
                {
                    "fold": len(results),
                    "train_start": df.index[test_start_idx],
                    "train_end": df.index[train_end - 1],
                    "test_start": df.index[train_end],
                    "test_end": df.index[test_end - 1],
                    "best_params": best_params,
                    "best_train_score": best_score,
                    "oos_sharpe": metrics.sharpe,
                    "oos_cagr": metrics.cagr,
                    "oos_max_dd": metrics.max_drawdown,
                    "oos_trades": metrics.trade_count,
                }
            )

            all_equity.append(equity)
            all_trades.extend(trades)

            test_start_idx += self.step_bars

        equity_curve = pd.concat(all_equity) if all_equity else pd.Series(dtype=float)
        return results, equity_curve, all_trades

    def _optimize(
        self,
        train_df: pd.DataFrame,
        strategy_factory: Callable[[], Any],
        param_grid: dict[str, list[Any]],
        optimize_fn: Optional[Callable],
    ) -> tuple[dict, float]:
        """Grid search over param_grid, return best params and score."""
        from itertools import product

        keys = list(param_grid.keys())
        values = list(param_grid.values())
        best_score = -np.inf
        best_params = {}

        for combo in product(*values):
            params = dict(zip(keys, combo))
            if optimize_fn:
                score = optimize_fn(train_df, params)
            else:
                strategy = strategy_factory()
                signals = strategy.generate_signals(train_df, params)
                equity, trades, _ = self.execution_engine.run(train_df, signals)
                metrics = compute_metrics(equity, trades)
                score = getattr(metrics, self.optimization_metric, 0.0)

            if score > best_score:
                best_score = score
                best_params = params

        return best_params, best_score
