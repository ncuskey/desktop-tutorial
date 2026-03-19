"""Breakout strategy implementations."""

from __future__ import annotations

import pandas as pd

from forex_research_lab.data.indicators import atr
from forex_research_lab.strategies.base import BaseStrategy


class RangeBreakoutStrategy(BaseStrategy):
    """Trade breaks of the recent range and hold until a mean reversion exit."""

    name = "range_breakout"

    def parameter_grid(self) -> dict[str, list[int]]:
        return {
            "lookback": [10, 20, 40],
            "exit_window": [5, 10, 20],
        }

    def generate_signals(self, dataframe: pd.DataFrame, params: dict[str, int] | None = None) -> pd.Series:
        settings = {"lookback": 20, "exit_window": 10}
        settings.update(params or {})

        lookback = int(settings["lookback"])
        exit_window = int(settings["exit_window"])

        range_high = dataframe["high"].shift(1).rolling(lookback, min_periods=lookback).max()
        range_low = dataframe["low"].shift(1).rolling(lookback, min_periods=lookback).min()
        exit_mean = dataframe["close"].rolling(exit_window, min_periods=exit_window).mean()

        position = []
        current_position = 0.0
        for index in dataframe.index:
            close = float(dataframe.at[index, "close"])
            upper = range_high.at[index]
            lower = range_low.at[index]
            center = exit_mean.at[index]

            if pd.isna(upper) or pd.isna(lower):
                current_position = 0.0
            elif current_position == 0.0:
                if close > upper:
                    current_position = 1.0
                elif close < lower:
                    current_position = -1.0
            elif current_position > 0.0 and not pd.isna(center) and close < center:
                current_position = 0.0
            elif current_position < 0.0 and not pd.isna(center) and close > center:
                current_position = 0.0

            position.append(current_position)

        return pd.Series(position, index=dataframe.index, dtype=float).clip(-1.0, 1.0)


class VolatilityExpansionBreakoutStrategy(BaseStrategy):
    """Trade channel breaks only when volatility expands materially."""

    name = "volatility_expansion_breakout"

    def parameter_grid(self) -> dict[str, list[float]]:
        return {
            "breakout_window": [10, 20, 40],
            "atr_window": [10, 14, 20],
            "atr_multiplier": [1.0, 1.25, 1.5],
        }

    def generate_signals(self, dataframe: pd.DataFrame, params: dict[str, float] | None = None) -> pd.Series:
        settings = {"breakout_window": 20, "atr_window": 14, "atr_multiplier": 1.25}
        settings.update(params or {})

        breakout_window = int(settings["breakout_window"])
        atr_window = int(settings["atr_window"])
        atr_multiplier = float(settings["atr_multiplier"])

        average_true_range = atr(dataframe, window=atr_window)
        breakout_high = dataframe["high"].shift(1).rolling(breakout_window, min_periods=breakout_window).max()
        breakout_low = dataframe["low"].shift(1).rolling(breakout_window, min_periods=breakout_window).min()
        rolling_range = (dataframe["high"] - dataframe["low"]).rolling(atr_window, min_periods=atr_window).mean()

        position = []
        current_position = 0.0
        for index in dataframe.index:
            close = float(dataframe.at[index, "close"])
            upper = breakout_high.at[index]
            lower = breakout_low.at[index]
            current_range = rolling_range.at[index]
            current_atr = average_true_range.at[index]

            if pd.isna(upper) or pd.isna(lower) or pd.isna(current_range) or pd.isna(current_atr):
                current_position = 0.0
            elif current_range >= current_atr * atr_multiplier:
                if close > upper:
                    current_position = 1.0
                elif close < lower:
                    current_position = -1.0
            else:
                current_position = 0.0
            position.append(current_position)

        return pd.Series(position, index=dataframe.index, dtype=float).clip(-1.0, 1.0)
