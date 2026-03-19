"""Breakout strategies."""

from __future__ import annotations

import pandas as pd

from .base import Strategy


class RangeBreakoutStrategy(Strategy):
    name = "range_breakout"

    def generate_signals(self, df: pd.DataFrame, params: dict[str, float]) -> pd.Series:
        lookback = int(params.get("lookback", 24))
        threshold = float(params.get("threshold", 0.0))

        prior_high = df["high"].rolling(lookback).max().shift(1)
        prior_low = df["low"].rolling(lookback).min().shift(1)

        signal = pd.Series(0.0, index=df.index)
        signal = signal.mask(df["close"] > prior_high * (1 + threshold), 1.0)
        signal = signal.mask(df["close"] < prior_low * (1 - threshold), -1.0)
        return self._coerce_signal(signal.ffill().fillna(0.0))


class VolatilityExpansionBreakoutStrategy(Strategy):
    name = "volatility_expansion_breakout"

    def generate_signals(self, df: pd.DataFrame, params: dict[str, float]) -> pd.Series:
        atr_window = int(params.get("atr_window", 14))
        breakout_window = int(params.get("breakout_window", 20))
        expansion_multiple = float(params.get("expansion_multiple", 1.25))

        atr = df["atr"] if "atr" in df.columns else self._fallback_atr(df, atr_window)
        atr_baseline = atr.rolling(breakout_window).mean()
        breakout_level = df["high"].rolling(breakout_window).max().shift(1)
        breakdown_level = df["low"].rolling(breakout_window).min().shift(1)

        expanding = atr > atr_baseline * expansion_multiple
        signal = pd.Series(0.0, index=df.index)
        signal = signal.mask(expanding & (df["close"] > breakout_level), 1.0)
        signal = signal.mask(expanding & (df["close"] < breakdown_level), -1.0)
        return self._coerce_signal(signal.ffill().fillna(0.0))

    @staticmethod
    def _fallback_atr(df: pd.DataFrame, window: int) -> pd.Series:
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
