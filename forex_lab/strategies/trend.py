"""
Trend-following strategies.

1. MACrossover  — fast/slow SMA crossover
2. DonchianBreakout — price breaks above/below N-period Donchian channel
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .base import BaseStrategy, SignalSeries
from data.indicators import sma, donchian_channels


class MACrossover(BaseStrategy):
    """Moving average crossover strategy.

    Long  when fast MA > slow MA.
    Short when fast MA < slow MA.

    Params
    ------
    fast_period : int   (default 20)
    slow_period : int   (default 50)
    ma_type     : str   'sma' | 'ema'  (default 'sma')
    """

    name = "ma_crossover"

    def default_params(self) -> dict[str, Any]:
        return {"fast_period": 20, "slow_period": 50, "ma_type": "sma"}

    def generate_signals(self, df: pd.DataFrame, params: dict[str, Any]) -> SignalSeries:
        fast = int(params.get("fast_period", 20))
        slow = int(params.get("slow_period", 50))
        ma_type = params.get("ma_type", "sma")

        close = df["close"]

        if ma_type == "ema":
            fast_ma = close.ewm(span=fast, adjust=False).mean()
            slow_ma = close.ewm(span=slow, adjust=False).mean()
        else:
            fast_ma = close.rolling(fast).mean()
            slow_ma = close.rolling(slow).mean()

        signal = pd.Series(0, index=df.index, dtype="int8")
        signal[fast_ma > slow_ma] = 1
        signal[fast_ma < slow_ma] = -1

        # Require both MAs to be valid (no lookahead warm-up)
        valid = fast_ma.notna() & slow_ma.notna()
        signal[~valid] = 0

        return self._clip_signals(signal)


class DonchianBreakout(BaseStrategy):
    """Donchian channel breakout strategy.

    Long  when price closes above the upper channel of prior N bars.
    Short when price closes below the lower channel of prior N bars.
    Exit  when price crosses the mid channel.

    Params
    ------
    period    : int  (default 20)
    exit_mid  : bool (default True) — exit at midline vs opposite band
    """

    name = "donchian_breakout"

    def default_params(self) -> dict[str, Any]:
        return {"period": 20, "exit_mid": True}

    def generate_signals(self, df: pd.DataFrame, params: dict[str, Any]) -> SignalSeries:
        period = int(params.get("period", 20))
        exit_mid = bool(params.get("exit_mid", True))

        close = df["close"]

        # Shift by 1 to avoid lookahead: signal uses prior-bar channel
        upper = df["high"].rolling(period).max().shift(1)
        lower = df["low"].rolling(period).min().shift(1)
        mid = (upper + lower) / 2

        signal = pd.Series(0, index=df.index, dtype="int8")
        signal[close > upper] = 1
        signal[close < lower] = -1

        if exit_mid:
            # Flatten positions that have reverted to midline
            # (use forward-fill to propagate breakout signal until mid is hit)
            raw = signal.copy()
            position = 0
            positions = []
            for i in range(len(raw)):
                if raw.iloc[i] != 0:
                    position = int(raw.iloc[i])
                elif position == 1 and close.iloc[i] < mid.iloc[i]:
                    position = 0
                elif position == -1 and close.iloc[i] > mid.iloc[i]:
                    position = 0
                positions.append(position)
            signal = pd.Series(positions, index=df.index, dtype="int8")

        valid = upper.notna() & lower.notna()
        signal[~valid] = 0

        return self._clip_signals(signal)
