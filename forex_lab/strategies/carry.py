"""Carry trade strategy (simplified proxy)."""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Any

from .base import Strategy

RATE_DIFFERENTIALS = {
    "EURUSD": -0.01,
    "GBPUSD": 0.005,
    "USDJPY": 0.04,
    "AUDUSD": -0.005,
}


class CarryStrategy(Strategy):
    """Simplified carry trade — holds direction aligned with interest rate differential.

    In a real system the rate differential would be fetched from a data feed.
    Here we use a static proxy to demonstrate the concept.
    """

    name = "carry"

    def default_params(self) -> dict[str, Any]:
        return {"min_differential": 0.005}

    def param_grid(self) -> dict[str, list[Any]]:
        return {"min_differential": [0.0, 0.005, 0.01, 0.02]}

    def generate_signals(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        min_diff = params.get("min_differential", 0.005)
        symbol = df["symbol"].iloc[0] if "symbol" in df.columns else "EURUSD"
        diff = RATE_DIFFERENTIALS.get(symbol, 0.0)

        if abs(diff) < min_diff:
            return pd.Series(0.0, index=df.index)

        direction = 1.0 if diff > 0 else -1.0
        return pd.Series(direction, index=df.index)
