"""Performance-based: allocate to strategies with best recent Sharpe."""

from typing import Any, Callable, List, Optional

import numpy as np
import pandas as pd

from forex_lab.execution.engine import ExecutionEngine
from forex_lab.metrics.compute import compute_metrics


class PerformanceBasedOrchestrator:
    """
    Allocate capital to strategies with best recent performance.
    """

    def __init__(
        self,
        strategies: List[Callable],
        lookback_bars: int = 252,
        execution_engine: Optional[ExecutionEngine] = None,
    ):
        self.strategies = strategies  # list of (strategy_factory, param_grid) or similar
        self.lookback_bars = lookback_bars
        self.execution_engine = execution_engine or ExecutionEngine()

    def run(
        self,
        df: pd.DataFrame,
        strategy_factories: List[Callable],
        params_list: List[dict],
    ) -> pd.Series:
        """
        At each bar, compute rolling Sharpe for each strategy, pick best.
        Simplified: use fixed lookback, pick strategy with highest Sharpe.
        """
        from forex_lab.strategies.base import BaseStrategy

        n = len(df)
        signals_list = []
        for factory, params in zip(strategy_factories, params_list):
            strat = factory()
            sig = strat.generate_signals(df, params)
            signals_list.append(sig.reindex(df.index).fillna(0).astype(int))

        # Rolling: for each bar, look back, run each strategy, get Sharpe, pick best
        position = np.zeros(n)
        for i in range(self.lookback_bars, n):
            window = df.iloc[i - self.lookback_bars : i]
            best_sharpe = -np.inf
            best_sig = 0
            for j, sig in enumerate(signals_list):
                sig_window = sig.iloc[i - self.lookback_bars : i]
                equity, trades, _ = self.execution_engine.run(window, sig_window)
                m = compute_metrics(equity, trades)
                if m.sharpe > best_sharpe:
                    best_sharpe = m.sharpe
                    best_sig = sig.iloc[i]
            position[i] = best_sig

        return pd.Series(position, index=df.index)
