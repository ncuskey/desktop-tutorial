"""Base strategy interface."""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseStrategy(ABC):
    """All strategies must implement generate_signals and return position series."""

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        """
        Generate position series from OHLCV + indicators.

        Returns:
            Series of positions: -1 (short), 0 (flat), +1 (long)
            Index must align with df.
        """
        pass

    @property
    @abstractmethod
    def default_params(self) -> dict[str, Any]:
        """Default parameters for the strategy."""
        pass
