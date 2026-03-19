"""Ensemble: average or weighted signals."""

from typing import List, Optional

import numpy as np
import pandas as pd

from forex_lab.strategies.base import BaseStrategy


class EnsembleCombiner:
    """
    Average or weighted combination of strategy signals.
    Output: +1 if avg > threshold, -1 if avg < -threshold, else 0.
    """

    def __init__(
        self,
        strategies: List[BaseStrategy],
        weights: Optional[List[float]] = None,
        threshold: float = 0.0,
    ):
        self.strategies = strategies
        self.weights = weights or [1.0 / len(strategies)] * len(strategies)
        self.threshold = threshold

    def combine(
        self,
        df: pd.DataFrame,
        params_list: List[dict],
    ) -> pd.Series:
        """
        Combine signals: weighted average, then threshold.
        """
        if len(params_list) != len(self.strategies):
            raise ValueError("params_list length must match strategies")

        signals = []
        for strat, params in zip(self.strategies, params_list):
            sig = strat.generate_signals(df, params)
            signals.append(sig.reindex(df.index).fillna(0).astype(int))

        stacked = pd.concat(signals, axis=1)
        weights = np.array(self.weights)[: stacked.shape[1]]
        weights = weights / weights.sum()
        avg = (stacked.values * weights).sum(axis=1)

        position = np.where(avg > self.threshold, 1, np.where(avg < -self.threshold, -1, 0))
        return pd.Series(position, index=df.index)
