"""Performance-based orchestrator — allocate to strategies with best recent Sharpe."""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Any

from ..strategies.base import Strategy
from ..execution.engine import ExecutionEngine
from ..metrics.calculator import compute_metrics


class PerformanceBasedOrchestrator:
    """Allocate capital to strategies proportionally to their recent performance.

    Uses a rolling lookback window to estimate each strategy's Sharpe ratio,
    then weights signals accordingly.
    """

    def __init__(
        self,
        strategies: list[Strategy],
        params_list: list[dict[str, Any]] | None = None,
        lookback: int = 500,
        rebalance_every: int = 50,
        execution_engine: ExecutionEngine | None = None,
        periods_per_year: int = 252 * 6,
    ):
        self.strategies = strategies
        self.params_list = params_list or [s.default_params() for s in strategies]
        self.lookback = lookback
        self.rebalance_every = rebalance_every
        self.engine = execution_engine or ExecutionEngine()
        self.periods_per_year = periods_per_year

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate performance-weighted combined signals."""
        n = len(df)
        all_signals = {}
        all_returns = {}

        for i, (strat, params) in enumerate(zip(self.strategies, self.params_list)):
            sig = strat.generate_signals(df, params)
            all_signals[i] = sig
            exec_result = self.engine.run(df, sig)
            all_returns[i] = exec_result["net_returns"]

        combined = pd.Series(0.0, index=df.index)
        weights = np.ones(len(self.strategies)) / len(self.strategies)

        for t in range(self.lookback, n, self.rebalance_every):
            sharpes = []
            for i in range(len(self.strategies)):
                window_returns = all_returns[i].iloc[max(0, t - self.lookback):t]
                m = compute_metrics(window_returns, periods_per_year=self.periods_per_year)
                sharpes.append(max(m["sharpe"], 0.0))

            total = sum(sharpes)
            if total > 0:
                weights = np.array([s / total for s in sharpes])
            else:
                weights = np.ones(len(self.strategies)) / len(self.strategies)

            end = min(t + self.rebalance_every, n)
            for i in range(len(self.strategies)):
                combined.iloc[t:end] += weights[i] * all_signals[i].iloc[t:end]

        signal = pd.Series(0.0, index=df.index)
        signal[combined > 0.3] = 1.0
        signal[combined < -0.3] = -1.0
        return signal
