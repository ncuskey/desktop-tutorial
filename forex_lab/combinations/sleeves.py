"""Specialist sleeves — activate strategies conditionally based on filters."""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Any, Callable

from ..strategies.base import Strategy


class SpecialistSleeves:
    """Activate different strategies based on market conditions.

    Each sleeve is a (strategy, params, filter_fn) tuple.
    When multiple sleeves are active, the first matching sleeve wins
    (priority order).
    """

    def __init__(self):
        self._sleeves: list[tuple[Strategy, dict[str, Any], Callable[[pd.DataFrame], pd.Series]]] = []

    def add_sleeve(
        self,
        strategy: Strategy,
        params: dict[str, Any] | None = None,
        filter_fn: Callable[[pd.DataFrame], pd.Series] | None = None,
    ) -> "SpecialistSleeves":
        """Register a sleeve.

        filter_fn: callable that takes df and returns a boolean Series
                   indicating when this sleeve should be active.
                   If None, sleeve is always active.
        """
        if params is None:
            params = strategy.default_params()
        if filter_fn is None:
            filter_fn = lambda df: pd.Series(True, index=df.index)
        self._sleeves.append((strategy, params, filter_fn))
        return self

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate signals by activating the first matching sleeve per bar."""
        signal = pd.Series(0.0, index=df.index)
        assigned = pd.Series(False, index=df.index)

        for strategy, params, filter_fn in self._sleeves:
            active = filter_fn(df) & ~assigned
            if active.any():
                strat_signals = strategy.generate_signals(df, params)
                signal[active] = strat_signals[active]
                assigned = assigned | active

        return signal


def adx_filter(threshold: float = 25.0, above: bool = True) -> Callable[[pd.DataFrame], pd.Series]:
    """Return a filter function based on ADX level.

    Requires 'adx' column in the DataFrame (from compute_indicators).
    """
    def _filter(df: pd.DataFrame) -> pd.Series:
        if "adx" not in df.columns:
            return pd.Series(True, index=df.index)
        if above:
            return df["adx"] > threshold
        return df["adx"] <= threshold
    return _filter


def volatility_filter(
    atr_col: str = "atr",
    percentile: float = 75.0,
    above: bool = True,
) -> Callable[[pd.DataFrame], pd.Series]:
    """Filter based on ATR percentile rank."""
    def _filter(df: pd.DataFrame) -> pd.Series:
        if atr_col not in df.columns:
            return pd.Series(True, index=df.index)
        rank = df[atr_col].rolling(100, min_periods=20).rank(pct=True)
        if above:
            return rank > percentile / 100
        return rank <= percentile / 100
    return _filter
