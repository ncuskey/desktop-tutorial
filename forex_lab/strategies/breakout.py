"""Breakout strategies."""

from typing import Any

import numpy as np
import pandas as pd

from .base import BaseStrategy


class RangeBreakoutStrategy(BaseStrategy):
    """
    Range breakout: long when price breaks above N-period high,
    short when price breaks below N-period low.
    """

    @property
    def default_params(self) -> dict[str, Any]:
        return {"period": 20}

    def generate_signals(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        period = params.get("period", self.default_params["period"])

        high_chan = df["high"].shift(1).rolling(period - 1, min_periods=1).max()
        low_chan = df["low"].shift(1).rolling(period - 1, min_periods=1).min()
        close = df["close"]

        long_signal = close > high_chan
        short_signal = close < low_chan

        position = np.zeros(len(df), dtype=int)
        pos = 0
        for i in range(len(df)):
            if long_signal.iloc[i]:
                pos = 1
            elif short_signal.iloc[i]:
                pos = -1
            position[i] = pos

        return pd.Series(position, index=df.index)


class VolatilityExpansionBreakoutStrategy(BaseStrategy):
    """
    Volatility expansion breakout: trade when ATR expands beyond recent average.
    Long on upside breakout, short on downside.
    """

    @property
    def default_params(self) -> dict[str, Any]:
        return {"atr_period": 14, "expansion_mult": 1.5, "lookback": 20}

    def generate_signals(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        atr_period = params.get("atr_period", self.default_params["atr_period"])
        mult = params.get("expansion_mult", self.default_params["expansion_mult"])
        lookback = params.get("lookback", self.default_params["lookback"])

        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(atr_period, min_periods=atr_period).mean()
        atr_avg = atr.rolling(lookback, min_periods=lookback).mean()

        # Expansion: current ATR > mult * avg ATR
        expanded = atr > mult * atr_avg

        high_n = df["high"].shift(1).rolling(lookback - 1, min_periods=1).max()
        low_n = df["low"].shift(1).rolling(lookback - 1, min_periods=1).min()

        long_signal = expanded & (close > high_n)
        short_signal = expanded & (close < low_n)

        position = np.zeros(len(df), dtype=int)
        pos = 0
        for i in range(len(df)):
            if long_signal.iloc[i]:
                pos = 1
            elif short_signal.iloc[i]:
                pos = -1
            position[i] = pos

        return pd.Series(position, index=df.index)
