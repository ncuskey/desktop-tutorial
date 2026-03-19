"""
Breakout strategies.

1. RangeBreakout             — price breaks out of recent N-bar range
2. VolatilityExpansionBreakout — ATR-filtered momentum breakout
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .base import BaseStrategy, SignalSeries
from data.indicators import atr as compute_atr


class RangeBreakout(BaseStrategy):
    """Range breakout strategy.

    Identifies a consolidation range over `lookback` bars and enters
    when price breaks above the range high or below the range low.

    Params
    ------
    lookback    : int   range definition period (default 20)
    hold_bars   : int   maximum bars to hold before flat (default 10)
    """

    name = "range_breakout"

    def default_params(self) -> dict[str, Any]:
        return {"lookback": 20, "hold_bars": 10}

    def generate_signals(self, df: pd.DataFrame, params: dict[str, Any]) -> SignalSeries:
        lookback = int(params.get("lookback", 20))
        hold_bars = int(params.get("hold_bars", 10))

        high = df["high"]
        low = df["low"]
        close = df["close"]

        range_high = high.rolling(lookback).max().shift(1)
        range_low = low.rolling(lookback).min().shift(1)

        position = 0
        bars_held = 0
        positions = []

        for i in range(len(df)):
            rh = range_high.iloc[i]
            rl = range_low.iloc[i]
            c = close.iloc[i]

            if pd.isna(rh) or pd.isna(rl):
                positions.append(0)
                continue

            if bars_held >= hold_bars:
                position = 0
                bars_held = 0

            if position == 0:
                if c > rh:
                    position = 1
                    bars_held = 0
                elif c < rl:
                    position = -1
                    bars_held = 0
            else:
                bars_held += 1

            positions.append(position)

        result = pd.Series(positions, index=df.index, dtype="int8")
        return self._clip_signals(result)


class VolatilityExpansionBreakout(BaseStrategy):
    """Volatility expansion breakout strategy.

    Enters long/short when price moves more than `atr_mult` × ATR from
    the prior bar's close, signalling a volatility expansion event.
    Exits after `hold_bars` bars or when price reverses beyond `atr_mult`.

    Params
    ------
    atr_period  : int   ATR lookback (default 14)
    atr_mult    : float entry threshold multiplier (default 1.5)
    hold_bars   : int   maximum bars to hold (default 8)
    """

    name = "volatility_expansion_breakout"

    def default_params(self) -> dict[str, Any]:
        return {"atr_period": 14, "atr_mult": 1.5, "hold_bars": 8}

    def generate_signals(self, df: pd.DataFrame, params: dict[str, Any]) -> SignalSeries:
        atr_period = int(params.get("atr_period", 14))
        atr_mult = float(params.get("atr_mult", 1.5))
        hold_bars = int(params.get("hold_bars", 8))

        close = df["close"]
        atr_s = compute_atr(df, atr_period).shift(1)
        prev_close = close.shift(1)

        move = close - prev_close

        position = 0
        bars_held = 0
        positions = []

        for i in range(len(df)):
            atr_val = atr_s.iloc[i]
            m = move.iloc[i]

            if pd.isna(atr_val) or pd.isna(m):
                positions.append(0)
                continue

            threshold = atr_mult * atr_val

            if bars_held >= hold_bars:
                position = 0
                bars_held = 0

            if position == 0:
                if m > threshold:
                    position = 1
                    bars_held = 0
                elif m < -threshold:
                    position = -1
                    bars_held = 0
            else:
                bars_held += 1

            positions.append(position)

        result = pd.Series(positions, index=df.index, dtype="int8")
        return self._clip_signals(result)
