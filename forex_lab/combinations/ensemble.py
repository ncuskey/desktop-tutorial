"""Ensemble combiner — average or weighted-average signals."""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Any

from ..strategies.base import Strategy


class EnsembleCombiner:
    """Combine strategy signals via averaging.

    Supports equal weighting and custom weights.
    The raw average is discretized: > threshold → +1, < -threshold → -1, else 0.
    """

    def __init__(
        self,
        strategies: list[Strategy],
        weights: list[float] | None = None,
        threshold: float = 0.3,
    ):
        self.strategies = strategies
        if weights is None:
            weights = [1.0 / len(strategies)] * len(strategies)
        total = sum(weights)
        self.weights = [w / total for w in weights]
        self.threshold = threshold

    def generate_signals(
        self,
        df: pd.DataFrame,
        params_list: list[dict[str, Any]] | None = None,
    ) -> pd.Series:
        if params_list is None:
            params_list = [s.default_params() for s in self.strategies]

        combined = pd.Series(0.0, index=df.index)
        for strategy, params, weight in zip(self.strategies, params_list, self.weights):
            combined += weight * strategy.generate_signals(df, params)

        signal = pd.Series(0.0, index=df.index)
        signal[combined > self.threshold] = 1.0
        signal[combined < -self.threshold] = -1.0
        return signal

    @property
    def raw_signal(self) -> pd.Series | None:
        """Access last raw combined signal (before discretization)."""
        return getattr(self, "_last_raw", None)
