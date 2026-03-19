"""
Confirmation combiner — only trades when multiple strategies agree.

Signal rules:
  - Long  (+1) when all (or >= threshold) strategies signal long.
  - Short (-1) when all (or >= threshold) strategies signal short.
  - Flat  ( 0) otherwise.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from data.indicators import add_indicators


class ConfirmationCombiner:
    """Combine N strategy signals requiring consensus before trading.

    Parameters
    ----------
    threshold:
        Fraction of strategies that must agree (default 1.0 = all agree).
        e.g. 0.67 means ≥2/3 must agree.
    """

    name = "confirmation"

    def __init__(self, threshold: float = 1.0) -> None:
        self.threshold = threshold

    def generate_signals(
        self,
        df: pd.DataFrame,
        strategies: list,
        params_list: list[dict[str, Any]],
    ) -> pd.Series:
        """Combine signals from multiple strategies via confirmation.

        Parameters
        ----------
        df:
            OHLCV + indicators DataFrame.
        strategies:
            List of strategy instances.
        params_list:
            List of param dicts aligned with strategies.

        Returns
        -------
        Combined signal series {-1, 0, +1}.
        """
        if len(strategies) != len(params_list):
            raise ValueError("strategies and params_list must have the same length.")

        df = add_indicators(df)
        signal_matrix = pd.DataFrame(index=df.index)
        for i, (strat, params) in enumerate(zip(strategies, params_list)):
            col = f"s{i}_{strat.name}"
            signal_matrix[col] = strat.generate_signals(df, params)

        n = len(strategies)
        min_agree = int(np.ceil(self.threshold * n))

        long_count = (signal_matrix == 1).sum(axis=1)
        short_count = (signal_matrix == -1).sum(axis=1)

        combined = pd.Series(0, index=df.index, dtype="int8")
        combined[long_count >= min_agree] = 1
        combined[short_count >= min_agree] = -1
        return combined


import numpy as np
