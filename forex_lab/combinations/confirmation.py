"""Confirmation combiner — only trade when multiple strategies agree."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..strategies.base import Strategy


class ConfirmationCombiner:
    """Require *min_agree* strategies to agree before entering a trade."""

    def __init__(
        self,
        strategies: list[Strategy],
        params_list: list[dict[str, Any]],
        min_agree: int | None = None,
    ):
        self.strategies = strategies
        self.params_list = params_list
        self.min_agree = min_agree or len(strategies)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        all_signals = pd.DataFrame(
            {
                s.name: s.generate_signals(df, p)
                for s, p in zip(self.strategies, self.params_list)
            }
        )
        vote = all_signals.sum(axis=1)
        signal = pd.Series(0, index=df.index, dtype=int)
        signal[vote >= self.min_agree] = 1
        signal[vote <= -self.min_agree] = -1
        return signal
