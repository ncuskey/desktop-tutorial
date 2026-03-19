"""Trend-following strategy implementations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from forex_research_lab.data.indicators import moving_average
from forex_research_lab.strategies.base import BaseStrategy


class MovingAverageCrossoverStrategy(BaseStrategy):
    """Long when the fast MA is above the slow MA, short otherwise."""

    name = "ma_crossover"

    def parameter_grid(self) -> dict[str, list[int]]:
        return {
            "short_window": [10, 20, 30],
            "long_window": [50, 100, 150],
        }

    def generate_signals(self, dataframe: pd.DataFrame, params: dict[str, int] | None = None) -> pd.Series:
        settings = {"short_window": 20, "long_window": 100}
        settings.update(params or {})

        short_window = int(settings["short_window"])
        long_window = int(settings["long_window"])
        if short_window >= long_window:
            raise ValueError("short_window must be smaller than long_window")

        short_ma = moving_average(dataframe["close"], window=short_window)
        long_ma = moving_average(dataframe["close"], window=long_window)

        signal = pd.Series(0.0, index=dataframe.index, dtype=float)
        valid = short_ma.notna() & long_ma.notna()
        signal.loc[valid] = np.sign(short_ma.loc[valid] - long_ma.loc[valid]).astype(float)
        return signal.clip(-1.0, 1.0)


class DonchianBreakoutStrategy(BaseStrategy):
    """Breakout strategy that holds until a faster channel exit triggers."""

    name = "donchian_breakout"

    def parameter_grid(self) -> dict[str, list[int]]:
        return {
            "lookback": [20, 40, 60],
            "exit_lookback": [10, 20],
        }

    def generate_signals(self, dataframe: pd.DataFrame, params: dict[str, int] | None = None) -> pd.Series:
        settings = {"lookback": 40, "exit_lookback": 20}
        settings.update(params or {})

        lookback = int(settings["lookback"])
        exit_lookback = int(settings["exit_lookback"])
        if exit_lookback >= lookback:
            raise ValueError("exit_lookback should be smaller than lookback")

        breakout_high = dataframe["high"].shift(1).rolling(lookback, min_periods=lookback).max()
        breakout_low = dataframe["low"].shift(1).rolling(lookback, min_periods=lookback).min()
        exit_high = dataframe["high"].shift(1).rolling(exit_lookback, min_periods=exit_lookback).max()
        exit_low = dataframe["low"].shift(1).rolling(exit_lookback, min_periods=exit_lookback).min()

        position = []
        current_position = 0.0
        for index in dataframe.index:
            close = float(dataframe.at[index, "close"])
            upper = breakout_high.at[index]
            lower = breakout_low.at[index]
            upper_exit = exit_high.at[index]
            lower_exit = exit_low.at[index]

            if pd.isna(upper) or pd.isna(lower):
                current_position = 0.0
            elif current_position == 0.0:
                if close > upper:
                    current_position = 1.0
                elif close < lower:
                    current_position = -1.0
            elif current_position > 0.0:
                if close < lower:
                    current_position = -1.0
                elif not pd.isna(lower_exit) and close < lower_exit:
                    current_position = 0.0
            else:
                if close > upper:
                    current_position = 1.0
                elif not pd.isna(upper_exit) and close > upper_exit:
                    current_position = 0.0

            position.append(current_position)

        return pd.Series(position, index=dataframe.index, dtype=float).clip(-1.0, 1.0)
