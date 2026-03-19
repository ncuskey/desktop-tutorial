"""Breakout strategies."""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Any

from .base import Strategy


class RangeBreakout(Strategy):
    """N-bar range breakout — long on break above range high, short on break below."""

    name = "range_breakout"

    def default_params(self) -> dict[str, Any]:
        return {"lookback": 20, "threshold_atr_mult": 0.5}

    def param_grid(self) -> dict[str, list[Any]]:
        return {
            "lookback": [10, 20, 30, 50],
            "threshold_atr_mult": [0.25, 0.5, 1.0],
        }

    def generate_signals(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        lookback = params.get("lookback", 20)
        atr_mult = params.get("threshold_atr_mult", 0.5)

        range_high = df["high"].rolling(lookback).max().shift(1)
        range_low = df["low"].rolling(lookback).min().shift(1)
        atr = self._atr(df, lookback)
        threshold = atr * atr_mult

        signal = pd.Series(0.0, index=df.index)
        signal[df["close"] > (range_high + threshold)] = 1.0
        signal[df["close"] < (range_low - threshold)] = -1.0
        signal.iloc[:lookback] = 0.0
        return signal

    @staticmethod
    def _atr(df: pd.DataFrame, period: int) -> pd.Series:
        high, low = df["high"], df["low"]
        prev_close = df["close"].shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
        ).max(axis=1)
        return tr.ewm(span=period, min_periods=period, adjust=False).mean()


class VolatilityExpansion(Strategy):
    """Volatility expansion breakout — enter when current bar's range
    exceeds a multiple of recent ATR."""

    name = "volatility_expansion"

    def default_params(self) -> dict[str, Any]:
        return {"atr_period": 14, "expansion_mult": 2.0}

    def param_grid(self) -> dict[str, list[Any]]:
        return {
            "atr_period": [10, 14, 20],
            "expansion_mult": [1.5, 2.0, 2.5, 3.0],
        }

    def generate_signals(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        atr_period = params.get("atr_period", 14)
        mult = params.get("expansion_mult", 2.0)

        atr = RangeBreakout._atr(df, atr_period)
        bar_range = df["high"] - df["low"]
        expanded = bar_range > (atr.shift(1) * mult)

        bar_direction = np.sign(df["close"] - df["open"])

        signal = pd.Series(0.0, index=df.index)
        signal[expanded & (bar_direction > 0)] = 1.0
        signal[expanded & (bar_direction < 0)] = -1.0
        signal.iloc[:atr_period] = 0.0
        return signal
