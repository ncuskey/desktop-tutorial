"""
Parameter sweep (grid search and random search) for strategy optimisation.

Results are stored as a DataFrame for downstream analysis and heatmap
visualisation.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from data.indicators import add_indicators
from execution.engine import ExecutionEngine
from metrics.performance import compute_metrics, MetricsResult


class ParameterSweep:
    """Grid or random parameter search over a strategy.

    Parameters
    ----------
    method:
        'grid'   — exhaustive grid search.
        'random' — random sample n_samples parameter combinations.
    n_samples:
        Number of random combinations to evaluate (only for 'random').
    optimise_metric:
        Primary metric to rank results ('sharpe', 'cagr', etc.).
    """

    def __init__(
        self,
        method: str = "grid",
        n_samples: int = 100,
        optimise_metric: str = "sharpe",
    ) -> None:
        self.method = method
        self.n_samples = n_samples
        self.optimise_metric = optimise_metric
        self._engine = ExecutionEngine()

    def run(
        self,
        df: pd.DataFrame,
        strategy,
        param_grid: dict[str, list],
        symbol: str = "EURUSD",
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Run parameter sweep.

        Parameters
        ----------
        df:
            OHLCV DataFrame (indicators will be added internally).
        strategy:
            Strategy instance.
        param_grid:
            Dict mapping parameter name → list of candidate values.
        symbol:
            FX pair.
        verbose:
            Show progress bar.

        Returns
        -------
        DataFrame with one row per parameter combination, sorted by
        optimise_metric descending.
        """
        df = add_indicators(df)
        combinations = self._build_combinations(param_grid)

        if verbose:
            combinations = tqdm(combinations, desc=f"Param sweep [{strategy.name}]")

        rows = []
        for params in combinations:
            signals = strategy.generate_signals(df, params)
            result = self._engine.run(df, signals, symbol, strategy.name, params)
            metrics = compute_metrics(result.net_returns, result.trades)
            row = {**params, **metrics.to_dict()}
            row["n_bars"] = len(df)
            rows.append(row)

        if not rows:
            return pd.DataFrame()

        results = pd.DataFrame(rows)
        results = results.sort_values(self.optimise_metric, ascending=False)
        return results.reset_index(drop=True)

    def top_n(self, results: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        return results.head(n)

    def robustness_ratio(self, results: pd.DataFrame, metric: str = "sharpe") -> float:
        """Fraction of parameter combos with positive metric value."""
        if metric not in results.columns or len(results) == 0:
            return 0.0
        return float((results[metric] > 0).mean())

    # ------------------------------------------------------------------

    def _build_combinations(self, param_grid: dict[str, list]) -> list[dict]:
        keys = list(param_grid.keys())
        all_values = list(param_grid.values())

        if self.method == "grid":
            return [
                dict(zip(keys, combo))
                for combo in itertools.product(*all_values)
            ]

        # Random search
        rng = np.random.default_rng(seed=42)
        combos = set()
        result = []
        attempts = 0
        max_attempts = self.n_samples * 10

        while len(result) < self.n_samples and attempts < max_attempts:
            combo = tuple(rng.choice(v) for v in all_values)
            if combo not in combos:
                combos.add(combo)
                result.append(dict(zip(keys, combo)))
            attempts += 1

        return result
