"""Trend-following strategies."""

from typing import Any

import numpy as np
import pandas as pd

from .base import BaseStrategy


class MACrossoverStrategy(BaseStrategy):
    """
    Moving average crossover: long when fast > slow, short when fast < slow.
    """

    @property
    def default_params(self) -> dict[str, Any]:
        return {"fast": 10, "slow": 30}

    def generate_signals(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        fast = params.get("fast", self.default_params["fast"])
        slow = params.get("slow", self.default_params["slow"])

        ma_fast = df["close"].rolling(fast, min_periods=fast).mean()
        ma_slow = df["close"].rolling(slow, min_periods=slow).mean()

        # Vectorized: no lookahead
        position = np.where(ma_fast > ma_slow, 1, np.where(ma_fast < ma_slow, -1, 0))
        return pd.Series(position, index=df.index).ffill().fillna(0).astype(int)


class DonchianBreakoutStrategy(BaseStrategy):
    """
    Donchian channel breakout: long when price breaks above N-period high,
    short when price breaks below N-period low.
    """

    @property
    def default_params(self) -> dict[str, Any]:
        return {"period": 20}

    def generate_signals(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        period = params.get("period", self.default_params["period"])

        high_n = df["high"].rolling(period, min_periods=period).max()
        low_n = df["low"].rolling(period, min_periods=period).min()
        close = df["close"]

        # Breakout: close > high_n-1 (excluding current bar from high)
        high_prev = df["high"].shift(1).rolling(period - 1, min_periods=period - 1).max()
        low_prev = df["low"].shift(1).rolling(period - 1, min_periods=period - 1).min()

        # Simpler: use rolling max/min of prior bars only (no lookahead)
        high_chan = df["high"].shift(1).rolling(period - 1, min_periods=1).max()
        low_chan = df["low"].shift(1).rolling(period - 1, min_periods=1).min()

        long_signal = close > high_chan
        short_signal = close < low_chan

        position = np.zeros(len(df), dtype=int)
        pos = 0
        for i in range(len(df)):
            if pd.isna(high_chan.iloc[i]) or pd.isna(low_chan.iloc[i]):
                position[i] = pos
                continue
            if long_signal.iloc[i]:
                pos = 1
            elif short_signal.iloc[i]:
                pos = -1
            position[i] = pos

        return pd.Series(position, index=df.index)
