"""Breakout strategies."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .base import Strategy


class RangeBreakout(Strategy):
    """Range breakout — enter when price breaks the N-bar range."""

    name = "range_breakout"

    def generate_signals(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        period = params.get("period", 20)

        range_high = df["high"].rolling(period, min_periods=period).max().shift(1)
        range_low = df["low"].rolling(period, min_periods=period).min().shift(1)

        signal = pd.Series(0, index=df.index, dtype=int)
        signal[df["close"] > range_high] = 1
        signal[df["close"] < range_low] = -1
        signal.iloc[:period] = 0
        return signal


class VolatilityExpansion(Strategy):
    """Volatility expansion breakout — enter when ATR expands significantly."""

    name = "volatility_expansion"

    def generate_signals(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        atr_period = params.get("atr_period", 14)
        expansion_mult = params.get("expansion_mult", 1.5)
        lookback = params.get("lookback", 50)

        high, low, close = df["high"], df["low"], df["close"]
        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
        ).max(axis=1)
        atr = tr.rolling(atr_period, min_periods=atr_period).mean()
        atr_avg = atr.rolling(lookback, min_periods=lookback).mean()

        expanding = atr > expansion_mult * atr_avg
        momentum = close - close.shift(1)

        signal = pd.Series(0, index=df.index, dtype=int)
        signal[expanding & (momentum > 0)] = 1
        signal[expanding & (momentum < 0)] = -1
        signal.iloc[: max(atr_period, lookback)] = 0
        return signal
