"""Parameter sweep — grid or random search over strategy hyperparameters."""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from ..execution import execute_signals
from ..metrics import compute_metrics
from ..strategies.base import Strategy


class ParameterSweep:
    """Sweep strategy parameters and collect metrics for each combination."""

    def __init__(
        self,
        strategy: Strategy,
        param_grid: dict[str, list[Any]],
        freq: str = "h",
        mode: str = "grid",
        n_random: int = 50,
        seed: int = 42,
    ):
        self.strategy = strategy
        self.param_grid = param_grid
        self.freq = freq
        self.mode = mode
        self.n_random = n_random
        self.seed = seed

    def _grid_combos(self) -> list[dict[str, Any]]:
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        return [dict(zip(keys, combo)) for combo in product(*values)]

    def _random_combos(self) -> list[dict[str, Any]]:
        rng = np.random.default_rng(self.seed)
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        combos = []
        for _ in range(self.n_random):
            combo = {k: rng.choice(v) for k, v in zip(keys, values)}
            combos.append(combo)
        return combos

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the sweep and return a DataFrame of params + metrics."""
        combos = self._grid_combos() if self.mode == "grid" else self._random_combos()
        rows: list[dict[str, Any]] = []

        for params in combos:
            signals = self.strategy.generate_signals(df, params)
            result = execute_signals(df, signals)
            m = compute_metrics(result, self.freq)
            row = {**params, **m}
            rows.append(row)

        return pd.DataFrame(rows)
