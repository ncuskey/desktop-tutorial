"""Breakout-style strategies."""

from __future__ import annotations

import pandas as pd

from forex_research_lab.data.indicators import compute_atr

from .base import BaseStrategy


class RangeBreakoutStrategy(BaseStrategy):
    name = "range_breakout"

    def generate_signals(self, df: pd.DataFrame, params: dict) -> pd.Series:
        lookback = int(params.get("lookback", 24))
        range_high = df["high"].shift(1).rolling(window=lookback, min_periods=lookback).max()
        range_low = df["low"].shift(1).rolling(window=lookback, min_periods=lookback).min()

        signal = pd.Series(0, index=df.index, dtype=int)
        signal = signal.mask(df["close"] > range_high, 1)
        signal = signal.mask(df["close"] < range_low, -1)
        return signal.ffill().fillna(0).astype(int)


class VolatilityExpansionBreakoutStrategy(BaseStrategy):
    name = "volatility_expansion_breakout"

    def generate_signals(self, df: pd.DataFrame, params: dict) -> pd.Series:
        atr_window = int(params.get("atr_window", 14))
        breakout_window = int(params.get("breakout_window", 20))
        atr_multiple = float(params.get("atr_multiple", 1.2))

        atr = compute_atr(df["high"], df["low"], df["close"], window=atr_window)
        rolling_mean_atr = atr.rolling(window=breakout_window, min_periods=breakout_window).mean()
        high_break = df["high"].shift(1).rolling(window=breakout_window, min_periods=breakout_window).max()
        low_break = df["low"].shift(1).rolling(window=breakout_window, min_periods=breakout_window).min()

        expansion = atr > (rolling_mean_atr * atr_multiple)
        signal = pd.Series(0, index=df.index, dtype=int)
        signal = signal.mask(expansion & (df["close"] > high_break), 1)
        signal = signal.mask(expansion & (df["close"] < low_break), -1)
        return signal.ffill().fillna(0).astype(int)
