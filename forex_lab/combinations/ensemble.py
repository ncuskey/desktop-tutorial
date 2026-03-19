"""Ensemble combiner — average or weighted-average of signals."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..strategies.base import Strategy


class EnsembleCombiner:
    """Combine strategy signals via (weighted) averaging."""

    def __init__(
        self,
        strategies: list[Strategy],
        params_list: list[dict[str, Any]],
        weights: list[float] | None = None,
        threshold: float = 0.3,
    ):
        self.strategies = strategies
        self.params_list = params_list
        self.weights = weights or [1.0 / len(strategies)] * len(strategies)
        self.threshold = threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        all_signals = pd.DataFrame(
            {
                s.name: s.generate_signals(df, p).astype(float)
                for s, p in zip(self.strategies, self.params_list)
            }
        )
        weighted = all_signals.mul(self.weights, axis=1).sum(axis=1)

        signal = pd.Series(0, index=df.index, dtype=int)
        signal[weighted > self.threshold] = 1
        signal[weighted < -self.threshold] = -1
        return signal
