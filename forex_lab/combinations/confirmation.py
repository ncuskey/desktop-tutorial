"""Confirmation combiner — only trade when multiple strategies agree."""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Any

from ..strategies.base import Strategy


class ConfirmationCombiner:
    """Requires N strategies to agree before entering a position.

    A position is taken only when at least `min_agree` strategies
    produce the same directional signal.
    """

    def __init__(
        self,
        strategies: list[Strategy],
        min_agree: int | None = None,
    ):
        self.strategies = strategies
        self.min_agree = min_agree or len(strategies)

    def generate_signals(
        self,
        df: pd.DataFrame,
        params_list: list[dict[str, Any]] | None = None,
    ) -> pd.Series:
        """Generate combined signal requiring agreement.

        params_list: one param dict per strategy; if None, use defaults.
        """
        if params_list is None:
            params_list = [s.default_params() for s in self.strategies]

        all_signals = pd.DataFrame(index=df.index)
        for i, (strategy, params) in enumerate(zip(self.strategies, params_list)):
            all_signals[f"s_{i}"] = strategy.generate_signals(df, params)

        long_count = (all_signals > 0).sum(axis=1)
        short_count = (all_signals < 0).sum(axis=1)

        signal = pd.Series(0.0, index=df.index)
        signal[long_count >= self.min_agree] = 1.0
        signal[short_count >= self.min_agree] = -1.0
        return signal
