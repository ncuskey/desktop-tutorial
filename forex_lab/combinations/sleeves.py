"""Specialist sleeves — conditionally activate strategies based on filters."""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from ..strategies.base import Strategy


class SpecialistSleeves:
    """Route to specialist strategies based on market conditions.

    Each sleeve is a (strategy, params, filter_fn) tuple where *filter_fn*
    takes a DataFrame row and returns True when the sleeve should be active.
    """

    def __init__(
        self,
        sleeves: list[tuple[Strategy, dict[str, Any], Callable[[pd.Series], bool]]],
        default_position: int = 0,
    ):
        self.sleeves = sleeves
        self.default_position = default_position

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        all_signals = {}
        for strat, params, _ in self.sleeves:
            all_signals[strat.name] = strat.generate_signals(df, params)

        result = pd.Series(self.default_position, index=df.index, dtype=int)

        for i in range(len(df)):
            row = df.iloc[i]
            for strat, params, filter_fn in self.sleeves:
                if filter_fn(row):
                    result.iloc[i] = all_signals[strat.name].iloc[i]
                    break

        return result
