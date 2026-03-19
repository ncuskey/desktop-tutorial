"""Specialist sleeves: activate strategies conditionally (e.g. ADX filter)."""

from typing import Callable, List, Optional

import pandas as pd

from forex_lab.strategies.base import BaseStrategy


class SpecialistSleevesCombiner:
    """
    Use different strategies based on market condition.
    E.g. high ADX -> trend strategy, low ADX -> mean reversion.
    """

    def __init__(
        self,
        strategies: List[BaseStrategy],
        selector: Callable[[pd.DataFrame, int], int],
    ):
        """
        selector(df, i) -> index of strategy to use at row i.
        """
        self.strategies = strategies
        self.selector = selector

    def combine(
        self,
        df: pd.DataFrame,
        params_list: List[dict],
    ) -> pd.Series:
        """
        At each bar, use selector to pick strategy, then emit its signal.
        """
        if len(params_list) != len(self.strategies):
            raise ValueError("params_list length must match strategies")

        signals = []
        for strat, params in zip(self.strategies, params_list):
            sig = strat.generate_signals(df, params)
            signals.append(sig.reindex(df.index).fillna(0).astype(int))

        stacked = pd.concat(signals, axis=1)
        n = len(df)
        position = []
        for i in range(n):
            idx = self.selector(df, i)
            idx = min(idx, stacked.shape[1] - 1)
            position.append(stacked.iloc[i, idx])

        return pd.Series(position, index=df.index)
