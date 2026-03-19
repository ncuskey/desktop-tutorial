"""Base strategy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd
from typing import Any


class Strategy(ABC):
    """All strategies must implement generate_signals.

    Signals: +1 = long, -1 = short, 0 = flat.
    Signals must not look ahead — they should only depend on data
    available at or before the current bar.
    """

    name: str = "base"

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        """Return a Series of positions aligned to df.index.

        Values: -1 (short), 0 (flat), +1 (long).
        """
        ...

    def default_params(self) -> dict[str, Any]:
        """Return default parameter dict for this strategy."""
        return {}

    def param_grid(self) -> dict[str, list[Any]]:
        """Return parameter grid for sweep/optimization."""
        return {}
