"""Base interfaces for strategy implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseStrategy(ABC):
    """Simple contract for research strategies."""

    name: str

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """Return a position series with values in {-1, 0, 1}."""
