"""Base interfaces for trading strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseStrategy(ABC):
    """Abstract base class for all strategies."""

    name: str = "base_strategy"

    @abstractmethod
    def generate_signals(self, dataframe: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.Series:
        """Return a position series in the range [-1, 0, 1]."""

    def parameter_grid(self) -> dict[str, list[Any]]:
        """Return a default parameter search space."""
        return {}
