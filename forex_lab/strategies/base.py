"""Abstract base class for all strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class Strategy(ABC):
    """Every strategy must implement ``generate_signals``."""

    name: str = "base"

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        """Return a position series with values in {-1, 0, +1}.

        Signals must be computed using only information available *up to and
        including* the current bar — no look-ahead.
        """

    def __repr__(self) -> str:
        return f"<Strategy: {self.name}>"
