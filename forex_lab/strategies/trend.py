"""Trend-following strategies."""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Any

from .base import Strategy


class MACrossover(Strategy):
    """Moving average crossover — long when fast MA > slow MA."""

    name = "ma_crossover"

    def default_params(self) -> dict[str, Any]:
        return {"fast_period": 20, "slow_period": 50}

    def param_grid(self) -> dict[str, list[Any]]:
        return {
            "fast_period": [10, 20, 30, 50],
            "slow_period": [50, 100, 150, 200],
        }

    def generate_signals(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        fast = params.get("fast_period", 20)
        slow = params.get("slow_period", 50)
        if fast >= slow:
            return pd.Series(0.0, index=df.index)

        fast_ma = df["close"].rolling(fast).mean()
        slow_ma = df["close"].rolling(slow).mean()

        signal = pd.Series(0.0, index=df.index)
        signal[fast_ma > slow_ma] = 1.0
        signal[fast_ma < slow_ma] = -1.0
        signal.iloc[:slow] = 0.0
        return signal


class DonchianBreakout(Strategy):
    """Donchian channel breakout — long above upper channel, short below lower."""

    name = "donchian_breakout"

    def default_params(self) -> dict[str, Any]:
        return {"period": 20}

    def param_grid(self) -> dict[str, list[Any]]:
        return {"period": [10, 20, 30, 50, 55]}

    def generate_signals(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        period = params.get("period", 20)
        upper = df["high"].rolling(period).max().shift(1)
        lower = df["low"].rolling(period).min().shift(1)

        signal = pd.Series(0.0, index=df.index)
        signal[df["close"] > upper] = 1.0
        signal[df["close"] < lower] = -1.0
        signal.iloc[:period] = 0.0
        return signal
