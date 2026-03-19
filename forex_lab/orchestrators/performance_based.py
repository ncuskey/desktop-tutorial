"""
Performance-based orchestrator.

Allocates signal weight to strategies proportionally to their recent
rolling Sharpe ratio.  Strategies with negative recent Sharpe receive
zero allocation (capital protection).

Uses a lookback window to estimate recent performance, then rebalances
at each step_bars interval.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from data.indicators import add_indicators
from execution.engine import ExecutionEngine
from metrics.performance import compute_metrics


class PerformanceBasedOrchestrator:
    """Dynamically allocate to strategies by recent rolling Sharpe.

    Parameters
    ----------
    lookback_bars:
        Window of bars used to estimate recent Sharpe for each strategy.
    rebalance_every:
        Rebalance allocation every N bars.
    min_sharpe:
        Minimum Sharpe to receive any allocation (default 0.0).
    """

    name = "performance_based_orchestrator"

    def __init__(
        self,
        lookback_bars: int = 500,
        rebalance_every: int = 100,
        min_sharpe: float = 0.0,
    ) -> None:
        self.lookback_bars = lookback_bars
        self.rebalance_every = rebalance_every
        self.min_sharpe = min_sharpe
        self._engine = ExecutionEngine()

    def generate_signals(
        self,
        df: pd.DataFrame,
        strategies: list,
        params_list: list[dict[str, Any]],
        symbol: str = "EURUSD",
    ) -> pd.Series:
        """Generate dynamically-weighted ensemble signal.

        At each rebalance point, compute rolling Sharpe for each strategy
        over the prior lookback window.  Allocate proportionally.

        Returns
        -------
        pd.Series of int8 {-1, 0, +1}.
        """
        df = add_indicators(df)

        # Pre-compute full signal series for each strategy
        all_signals = [s.generate_signals(df, p) for s, p in zip(strategies, params_list)]

        combined = pd.Series(0.0, index=df.index)
        current_weights = np.ones(len(strategies)) / len(strategies)

        for i in range(len(df)):
            # Rebalance periodically
            if i > self.lookback_bars and i % self.rebalance_every == 0:
                current_weights = self._compute_weights(
                    df.iloc[i - self.lookback_bars : i],
                    strategies,
                    params_list,
                    all_signals,
                    i - self.lookback_bars,
                    symbol,
                )

            if i >= self.lookback_bars:
                avg_signal = sum(
                    w * float(all_signals[j].iloc[i])
                    for j, w in enumerate(current_weights)
                )
                if avg_signal >= 0.3:
                    combined.iloc[i] = 1.0
                elif avg_signal <= -0.3:
                    combined.iloc[i] = -1.0

        return combined.astype("int8")

    # ------------------------------------------------------------------

    def _compute_weights(
        self,
        window_df: pd.DataFrame,
        strategies: list,
        params_list: list[dict],
        all_signals: list[pd.Series],
        window_start_idx: int,
        symbol: str,
    ) -> np.ndarray:
        """Compute allocation weights from recent Sharpe ratios."""
        sharpes = []
        for j, (strat, params) in enumerate(zip(strategies, params_list)):
            window_signals = all_signals[j].iloc[window_start_idx : window_start_idx + len(window_df)]
            result = self._engine.run(window_df, window_signals, symbol, strat.name, params)
            m = compute_metrics(result.net_returns, result.trades)
            sharpes.append(max(m.sharpe, self.min_sharpe))

        sharpes = np.array(sharpes)
        total = sharpes.sum()
        if total <= 0:
            return np.ones(len(strategies)) / len(strategies)
        return sharpes / total
