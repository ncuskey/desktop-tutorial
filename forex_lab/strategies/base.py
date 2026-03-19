from __future__ import annotations

from typing import Any, Dict, Protocol

import pandas as pd


class Strategy(Protocol):
    """Protocol for all strategies in the lab."""

    def generate_signals(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        ...
