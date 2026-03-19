"""
Ensemble combiner — averages strategy signals (optionally weighted).

Signal rules:
  - Compute weighted average of individual signals.
  - Round to nearest integer OR threshold-clip to {-1, 0, +1}.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from data.indicators import add_indicators


class EnsembleCombiner:
    """Combine N strategy signals via weighted averaging.

    Parameters
    ----------
    weights:
        Optional list of weights aligned with strategies.
        If None, equal weights are used.
    long_threshold:
        Averaged signal must exceed this to go long (default 0.3).
    short_threshold:
        Averaged signal must fall below -this to go short (default 0.3).
    """

    name = "ensemble"

    def __init__(
        self,
        weights: list[float] | None = None,
        long_threshold: float = 0.3,
        short_threshold: float = 0.3,
    ) -> None:
        self.weights = weights
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold

    def generate_signals(
        self,
        df: pd.DataFrame,
        strategies: list,
        params_list: list[dict[str, Any]],
    ) -> pd.Series:
        """Combine signals via weighted average.

        Returns
        -------
        Combined signal series {-1, 0, +1}.
        """
        if len(strategies) != len(params_list):
            raise ValueError("strategies and params_list must have the same length.")

        df = add_indicators(df)

        weights = self.weights
        if weights is None:
            weights = [1.0 / len(strategies)] * len(strategies)
        else:
            total = sum(weights)
            weights = [w / total for w in weights]

        avg = pd.Series(0.0, index=df.index)
        for strat, params, w in zip(strategies, params_list, weights):
            avg += w * strat.generate_signals(df, params).astype(float)

        combined = pd.Series(0, index=df.index, dtype="int8")
        combined[avg >= self.long_threshold] = 1
        combined[avg <= -self.short_threshold] = -1
        return combined

    def continuous_signal(
        self,
        df: pd.DataFrame,
        strategies: list,
        params_list: list[dict[str, Any]],
    ) -> pd.Series:
        """Return the raw continuous average signal (not thresholded)."""
        df = add_indicators(df)
        weights = self.weights or [1.0 / len(strategies)] * len(strategies)
        total = sum(weights)
        weights = [w / total for w in weights]

        avg = pd.Series(0.0, index=df.index)
        for strat, params, w in zip(strategies, params_list, weights):
            avg += w * strat.generate_signals(df, params).astype(float)
        return avg
