"""Trend-following strategies."""

from __future__ import annotations

import pandas as pd

from .base import Strategy


class MovingAverageCrossoverStrategy(Strategy):
    name = "moving_average_crossover"

    def generate_signals(self, df: pd.DataFrame, params: dict[str, int]) -> pd.Series:
        fast_window = int(params.get("fast_window", 20))
        slow_window = int(params.get("slow_window", 50))
        if fast_window >= slow_window:
            raise ValueError("fast_window must be smaller than slow_window.")

        fast_col = f"ma_{fast_window}"
        slow_col = f"ma_{slow_window}"

        fast = df[fast_col] if fast_col in df.columns else df["close"].rolling(fast_window).mean()
        slow = df[slow_col] if slow_col in df.columns else df["close"].rolling(slow_window).mean()

        signal = pd.Series(0.0, index=df.index)
        signal = signal.mask(fast > slow, 1.0)
        signal = signal.mask(fast < slow, -1.0)
        return self._coerce_signal(signal)


class DonchianBreakoutStrategy(Strategy):
    name = "donchian_breakout"

    def generate_signals(self, df: pd.DataFrame, params: dict[str, int]) -> pd.Series:
        lookback = int(params.get("lookback", 20))
        upper = df["high"].rolling(lookback).max().shift(1)
        lower = df["low"].rolling(lookback).min().shift(1)

        signal = pd.Series(0.0, index=df.index)
        signal = signal.mask(df["close"] > upper, 1.0)
        signal = signal.mask(df["close"] < lower, -1.0)
        return self._coerce_signal(signal.ffill().fillna(0.0))
