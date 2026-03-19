"""Trend-following strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseStrategy


class MovingAverageCrossoverStrategy(BaseStrategy):
    name = "moving_average_crossover"

    def generate_signals(self, df: pd.DataFrame, params: dict) -> pd.Series:
        short_window = int(params["short_window"])
        long_window = int(params["long_window"])
        if short_window >= long_window:
            raise ValueError("short_window must be smaller than long_window")

        short_ma = df["close"].rolling(window=short_window, min_periods=short_window).mean()
        long_ma = df["close"].rolling(window=long_window, min_periods=long_window).mean()
        signal = np.where(short_ma > long_ma, 1, -1)
        signal = pd.Series(signal, index=df.index, dtype=float).where(long_ma.notna(), 0.0)
        return signal.fillna(0.0).astype(int)


class DonchianBreakoutStrategy(BaseStrategy):
    name = "donchian_breakout"

    def generate_signals(self, df: pd.DataFrame, params: dict) -> pd.Series:
        lookback = int(params.get("lookback", 20))
        upper = df["high"].shift(1).rolling(window=lookback, min_periods=lookback).max()
        lower = df["low"].shift(1).rolling(window=lookback, min_periods=lookback).min()

        signal = pd.Series(0, index=df.index, dtype=int)
        signal = signal.mask(df["close"] > upper, 1)
        signal = signal.mask(df["close"] < lower, -1)
        return signal.ffill().fillna(0).astype(int)
