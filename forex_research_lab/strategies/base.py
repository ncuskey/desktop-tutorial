"""Base strategy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class Strategy(ABC):
    """Shared contract for all signal generators."""

    name: str

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        """Return a target position series in {-1, 0, 1}."""

    def _coerce_signal(self, signal: pd.Series) -> pd.Series:
        return signal.fillna(0.0).clip(-1, 1).astype(float)
