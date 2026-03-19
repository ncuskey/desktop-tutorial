"""
Base strategy interface.

Every strategy must implement `generate_signals(df, params)` and return
a pd.Series of positions aligned to df's index with values in {-1, 0, +1}.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

# Type alias for clarity
SignalSeries = pd.Series


class BaseStrategy(ABC):
    """Abstract base class for all strategies."""

    name: str = "base"

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, params: dict[str, Any]) -> SignalSeries:
        """Generate position signals from an OHLCV + indicators DataFrame.

        Parameters
        ----------
        df:
            DataFrame with OHLCV columns plus pre-computed indicators.
            Must NOT contain future data (no lookahead).
        params:
            Strategy-specific hyperparameters.

        Returns
        -------
        pd.Series of int8 with values in {-1, 0, +1}.
        Index aligned to df.index.
        """

    def _clip_signals(self, signals: pd.Series) -> SignalSeries:
        """Clip signal values to {-1, 0, +1} and fill NaN with 0."""
        return signals.fillna(0).clip(-1, 1).astype("int8")

    def default_params(self) -> dict[str, Any]:
        """Return default parameter dictionary for this strategy."""
        return {}
