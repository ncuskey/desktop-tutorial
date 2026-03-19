"""Carry strategy — simplified interest-rate differential proxy."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .base import Strategy

_RATE_PROXY = {
    "EURUSD": -0.005,
    "GBPUSD": 0.003,
    "USDJPY": 0.015,
    "AUDUSD": 0.002,
}


class CarryStrategy(Strategy):
    """Simplified carry — position based on interest rate differential sign.

    This is a placeholder; a production system would use actual swap rates
    or forward-point data.
    """

    name = "carry"

    def generate_signals(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        symbol = params.get("symbol", "EURUSD")
        rate_diff = params.get("rate_diff", _RATE_PROXY.get(symbol, 0.0))

        if rate_diff > 0:
            pos = 1
        elif rate_diff < 0:
            pos = -1
        else:
            pos = 0

        return pd.Series(pos, index=df.index, dtype=int)
