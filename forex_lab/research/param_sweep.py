"""Parameter sweep — grid or random search with results storage."""

from __future__ import annotations

import pandas as pd
import numpy as np
from itertools import product
from typing import Any

from ..strategies.base import Strategy
from ..execution.engine import ExecutionEngine
from ..metrics.calculator import compute_metrics


class ParameterSweep:
    """Grid or random search over strategy parameters."""

    def __init__(
        self,
        strategy: Strategy,
        execution_engine: ExecutionEngine | None = None,
        periods_per_year: int = 252 * 6,
    ):
        self.strategy = strategy
        self.engine = execution_engine or ExecutionEngine()
        self.periods_per_year = periods_per_year

    def grid_search(
        self,
        df: pd.DataFrame,
        param_grid: dict[str, list[Any]] | None = None,
    ) -> pd.DataFrame:
        """Run exhaustive grid search. Returns DataFrame of all results."""
        if param_grid is None:
            param_grid = self.strategy.param_grid()

        combos = self._expand_grid(param_grid)
        return self._evaluate(df, combos)

    def random_search(
        self,
        df: pd.DataFrame,
        param_ranges: dict[str, tuple[float, float]],
        n_samples: int = 100,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Run random parameter search over continuous ranges."""
        rng = np.random.default_rng(seed)
        combos = []
        for _ in range(n_samples):
            params = {}
            for key, (lo, hi) in param_ranges.items():
                if isinstance(lo, int) and isinstance(hi, int):
                    params[key] = int(rng.integers(lo, hi + 1))
                else:
                    params[key] = rng.uniform(lo, hi)
            combos.append(params)
        return self._evaluate(df, combos)

    def _evaluate(
        self, df: pd.DataFrame, param_combos: list[dict[str, Any]]
    ) -> pd.DataFrame:
        results = []
        for params in param_combos:
            signals = self.strategy.generate_signals(df, params)
            exec_result = self.engine.run(df, signals)
            metrics = compute_metrics(
                exec_result["net_returns"],
                periods_per_year=self.periods_per_year,
            )
            results.append({**params, **metrics})
        return pd.DataFrame(results)

    @staticmethod
    def _expand_grid(param_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        return [dict(zip(keys, combo)) for combo in product(*values)]
