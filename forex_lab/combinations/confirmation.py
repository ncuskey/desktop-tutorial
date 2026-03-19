"""Confirmation: only trade when multiple strategies agree."""

from typing import List, Optional

import numpy as np
import pandas as pd

from forex_lab.strategies.base import BaseStrategy


class ConfirmationCombiner:
    """
    Trade only when all (or threshold) strategies agree on direction.
    """

    def __init__(self, strategies: List[BaseStrategy], min_agreement: Optional[int] = None):
        self.strategies = strategies
        self.min_agreement = min_agreement or len(strategies)

    def combine(
        self,
        df: pd.DataFrame,
        params_list: List[dict],
    ) -> pd.Series:
        """
        Combine signals: output +1 only if >= min_agreement long, -1 if >= min_agreement short.
        """
        if len(params_list) != len(self.strategies):
            raise ValueError("params_list length must match strategies")

        signals = []
        for strat, params in zip(self.strategies, params_list):
            sig = strat.generate_signals(df, params)
            signals.append(sig.reindex(df.index).fillna(0).astype(int))

        stacked = pd.concat(signals, axis=1)
        long_votes = (stacked == 1).sum(axis=1)
        short_votes = (stacked == -1).sum(axis=1)

        position = np.where(long_votes >= self.min_agreement, 1, np.where(short_votes >= self.min_agreement, -1, 0))
        return pd.Series(position, index=df.index)
