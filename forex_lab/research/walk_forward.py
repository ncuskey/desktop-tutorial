"""Walk-forward analysis engine.

Splits data into rolling train/test windows, optimises parameters on the
training portion, then evaluates out-of-sample on the test portion.  Results
are aggregated across all folds to produce robust performance estimates.
"""

from __future__ import annotations

from itertools import product
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..execution import execute_signals
from ..metrics import compute_metrics
from ..strategies.base import Strategy


class WalkForwardEngine:
    """Rolling walk-forward optimiser and evaluator."""

    def __init__(
        self,
        strategy: Strategy,
        param_grid: dict[str, list[Any]],
        train_size: int,
        test_size: int,
        step_size: int | None = None,
        optimise_metric: str = "sharpe",
        freq: str = "h",
    ):
        self.strategy = strategy
        self.param_grid = param_grid
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size or test_size
        self.optimise_metric = optimise_metric
        self.freq = freq

    def _generate_param_combos(self) -> list[dict[str, Any]]:
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        return [dict(zip(keys, combo)) for combo in product(*values)]

    def _optimise_on_window(self, df_train: pd.DataFrame) -> dict[str, Any]:
        """Find the best parameter combination on the training window."""
        combos = self._generate_param_combos()
        best_score = -np.inf
        best_params: dict[str, Any] = combos[0]

        for params in combos:
            signals = self.strategy.generate_signals(df_train, params)
            result = execute_signals(df_train, signals)
            m = compute_metrics(result, self.freq)
            score = m.get(self.optimise_metric, 0.0)
            if score > best_score:
                best_score = score
                best_params = params

        return best_params

    def run(self, df: pd.DataFrame) -> dict[str, Any]:
        """Execute the walk-forward analysis.

        Returns a dict with per-fold results and aggregated OOS metrics.
        """
        n = len(df)
        folds: list[dict[str, Any]] = []
        oos_returns: list[pd.Series] = []

        start = 0
        while start + self.train_size + self.test_size <= n:
            train_end = start + self.train_size
            test_end = train_end + self.test_size

            df_train = df.iloc[start:train_end].copy()
            df_test = df.iloc[train_end:test_end].copy()

            best_params = self._optimise_on_window(df_train)

            signals = self.strategy.generate_signals(df_test, best_params)
            result = execute_signals(df_test, signals)
            fold_metrics = compute_metrics(result, self.freq)

            folds.append(
                {
                    "train_start": df_train.index[0],
                    "train_end": df_train.index[-1],
                    "test_start": df_test.index[0],
                    "test_end": df_test.index[-1],
                    "best_params": best_params,
                    "oos_metrics": fold_metrics,
                }
            )
            oos_returns.append(result["net_returns"])

            start += self.step_size

        if not oos_returns:
            return {"folds": [], "aggregate_metrics": {}}

        all_oos = pd.concat(oos_returns)
        agg_equity = 100_000 * np.cumprod(1 + all_oos.values)
        peak = np.maximum.accumulate(agg_equity)
        drawdown = (agg_equity - peak) / peak

        agg_result = pd.DataFrame(
            {
                "net_returns": all_oos.values,
                "equity": agg_equity,
                "drawdown": drawdown,
                "position": np.ones(len(all_oos)),
            },
            index=all_oos.index,
        )
        agg_metrics = compute_metrics(agg_result, self.freq)

        return {
            "folds": folds,
            "aggregate_metrics": agg_metrics,
            "oos_equity": agg_result[["equity", "drawdown"]],
        }
