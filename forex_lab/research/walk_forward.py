"""Walk-forward optimization engine.

Rolling train/test windows to avoid overfitting.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from itertools import product
from typing import Any, Callable

from ..strategies.base import Strategy
from ..execution.engine import ExecutionEngine
from ..metrics.calculator import compute_metrics


class WalkForwardEngine:
    """Rolling walk-forward analysis.

    Splits data into sequential train/test windows, optimizes parameters
    on each training window, and evaluates on the immediately following
    test window. This avoids the look-ahead bias inherent in full-sample
    optimization.
    """

    def __init__(
        self,
        strategy: Strategy,
        execution_engine: ExecutionEngine | None = None,
        train_bars: int = 1000,
        test_bars: int = 250,
        step_bars: int | None = None,
        optimize_metric: str = "sharpe",
        periods_per_year: int = 252 * 6,
    ):
        self.strategy = strategy
        self.engine = execution_engine or ExecutionEngine()
        self.train_bars = train_bars
        self.test_bars = test_bars
        self.step_bars = step_bars or test_bars
        self.optimize_metric = optimize_metric
        self.periods_per_year = periods_per_year

    def run(
        self,
        df: pd.DataFrame,
        param_grid: dict[str, list[Any]] | None = None,
    ) -> WalkForwardResult:
        """Execute walk-forward analysis over the full dataset.

        Returns aggregated out-of-sample results plus per-window details.
        """
        if param_grid is None:
            param_grid = self.strategy.param_grid()

        param_combos = self._expand_grid(param_grid) if param_grid else [self.strategy.default_params()]
        n = len(df)
        windows = []
        oos_equity_parts = []
        oos_returns_parts = []

        start = 0
        while start + self.train_bars + self.test_bars <= n:
            train_end = start + self.train_bars
            test_end = min(train_end + self.test_bars, n)

            train_df = df.iloc[start:train_end]
            test_df = df.iloc[train_end:test_end]

            best_params, best_score = self._optimize(train_df, param_combos)

            signals = self.strategy.generate_signals(test_df, best_params)
            exec_result = self.engine.run(test_df, signals)

            oos_metrics = compute_metrics(
                exec_result["net_returns"],
                periods_per_year=self.periods_per_year,
            )

            windows.append(
                {
                    "train_start": train_df.index[0],
                    "train_end": train_df.index[-1],
                    "test_start": test_df.index[0],
                    "test_end": test_df.index[-1],
                    "best_params": best_params,
                    "train_score": best_score,
                    **{f"oos_{k}": v for k, v in oos_metrics.items()},
                }
            )

            oos_returns_parts.append(exec_result["net_returns"])
            oos_equity_parts.append(exec_result["equity"])

            start += self.step_bars

        oos_returns = pd.concat(oos_returns_parts) if oos_returns_parts else pd.Series(dtype=float)
        aggregate_metrics = compute_metrics(oos_returns, periods_per_year=self.periods_per_year)

        return WalkForwardResult(
            windows=pd.DataFrame(windows),
            oos_returns=oos_returns,
            aggregate_metrics=aggregate_metrics,
        )

    def _optimize(
        self, train_df: pd.DataFrame, param_combos: list[dict[str, Any]]
    ) -> tuple[dict[str, Any], float]:
        best_score = -np.inf
        best_params = param_combos[0]

        for params in param_combos:
            signals = self.strategy.generate_signals(train_df, params)
            exec_result = self.engine.run(train_df, signals)
            metrics = compute_metrics(
                exec_result["net_returns"],
                periods_per_year=self.periods_per_year,
            )
            score = metrics.get(self.optimize_metric, 0.0)
            if score > best_score:
                best_score = score
                best_params = params

        return best_params, best_score

    @staticmethod
    def _expand_grid(param_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        return [dict(zip(keys, combo)) for combo in product(*values)]


class WalkForwardResult:
    """Container for walk-forward results."""

    def __init__(
        self,
        windows: pd.DataFrame,
        oos_returns: pd.Series,
        aggregate_metrics: dict[str, float],
    ):
        self.windows = windows
        self.oos_returns = oos_returns
        self.aggregate_metrics = aggregate_metrics

    def summary(self) -> pd.DataFrame:
        """Return a summary DataFrame of aggregate OOS metrics."""
        return pd.DataFrame.from_dict(
            self.aggregate_metrics, orient="index", columns=["Value"]
        )
