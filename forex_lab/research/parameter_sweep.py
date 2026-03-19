"""Parameter grid and random search."""

from typing import Any, Callable, Optional

import pandas as pd

from forex_lab.execution.engine import ExecutionEngine
from forex_lab.metrics.compute import Metrics, compute_metrics


class ParameterSweep:
    """
    Grid or random search over strategy parameters.
    Stores all results for analysis.
    """

    def __init__(
        self,
        execution_engine: Optional[ExecutionEngine] = None,
    ):
        self.execution_engine = execution_engine or ExecutionEngine()
        self.results: list[dict] = []

    def grid_search(
        self,
        df: pd.DataFrame,
        strategy_factory: Callable[[], Any],
        param_grid: dict[str, list[Any]],
        metric: str = "sharpe",
    ) -> pd.DataFrame:
        """
        Exhaustive grid search. Returns DataFrame of all results.
        """
        from itertools import product

        keys = list(param_grid.keys())
        values = list(param_grid.values())
        self.results = []

        for combo in product(*values):
            params = dict(zip(keys, combo))
            strategy = strategy_factory()
            signals = strategy.generate_signals(df, params)
            equity, trades, _ = self.execution_engine.run(df, signals)
            metrics = compute_metrics(equity, trades)

            row = {
                "params": str(params),
                "cagr": metrics.cagr,
                "sharpe": metrics.sharpe,
                "sortino": metrics.sortino,
                "max_drawdown": metrics.max_drawdown,
                "profit_factor": metrics.profit_factor,
                "win_rate": metrics.win_rate,
                "expectancy": metrics.expectancy,
                "trade_count": metrics.trade_count,
                "total_return": metrics.total_return,
                "volatility": metrics.volatility,
            }
            self.results.append(row)

        return pd.DataFrame(self.results)
