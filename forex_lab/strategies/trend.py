"""Trend-following strategies."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .base import Strategy


class MACrossover(Strategy):
    """Moving-average crossover — long when fast MA > slow MA, short otherwise."""

    name = "ma_crossover"

    def generate_signals(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        fast = params.get("fast_period", 20)
        slow = params.get("slow_period", 50)
        ma_type = params.get("ma_type", "sma")

        if ma_type == "ema":
            fast_ma = df["close"].ewm(span=fast, adjust=False).mean()
            slow_ma = df["close"].ewm(span=slow, adjust=False).mean()
        else:
            fast_ma = df["close"].rolling(fast, min_periods=fast).mean()
            slow_ma = df["close"].rolling(slow, min_periods=slow).mean()

        signal = pd.Series(0, index=df.index, dtype=int)
        signal[fast_ma > slow_ma] = 1
        signal[fast_ma < slow_ma] = -1
        signal.iloc[: max(fast, slow)] = 0
        return signal


class DonchianBreakout(Strategy):
    """Donchian channel breakout — long on new highs, short on new lows."""

    name = "donchian_breakout"

    def generate_signals(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        period = params.get("period", 20)

        high_channel = df["high"].rolling(period, min_periods=period).max().shift(1)
        low_channel = df["low"].rolling(period, min_periods=period).min().shift(1)

        signal = pd.Series(0, index=df.index, dtype=int)
        signal[df["close"] > high_channel] = 1
        signal[df["close"] < low_channel] = -1
        signal.iloc[:period] = 0
        return signal
