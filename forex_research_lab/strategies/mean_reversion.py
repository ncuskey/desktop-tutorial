"""Mean-reversion strategy implementations."""

from __future__ import annotations

import pandas as pd

from forex_research_lab.data.indicators import bollinger_bands, rsi
from forex_research_lab.strategies.base import BaseStrategy


class RSIReversalStrategy(BaseStrategy):
    """Fade short-term momentum using RSI extremes."""

    name = "rsi_reversal"

    def parameter_grid(self) -> dict[str, list[int]]:
        return {
            "window": [10, 14, 21],
            "entry_threshold": [25, 30, 35],
        }

    def generate_signals(self, dataframe: pd.DataFrame, params: dict[str, int] | None = None) -> pd.Series:
        settings = {"window": 14, "entry_threshold": 30, "exit_threshold": 50}
        settings.update(params or {})

        window = int(settings["window"])
        entry_threshold = float(settings["entry_threshold"])
        lower_threshold = float(settings.get("lower_threshold", entry_threshold))
        upper_threshold = float(settings.get("upper_threshold", 100.0 - entry_threshold))
        exit_threshold = float(settings["exit_threshold"])

        indicator = rsi(dataframe["close"], window=window)
        position = []
        current_position = 0.0

        for index in dataframe.index:
            value = indicator.at[index]
            if pd.isna(value):
                current_position = 0.0
            elif current_position == 0.0:
                if value < lower_threshold:
                    current_position = 1.0
                elif value > upper_threshold:
                    current_position = -1.0
            elif current_position > 0.0:
                if value >= exit_threshold:
                    current_position = 0.0
            else:
                if value <= exit_threshold:
                    current_position = 0.0
            position.append(current_position)

        return pd.Series(position, index=dataframe.index, dtype=float).clip(-1.0, 1.0)


class BollingerFadeStrategy(BaseStrategy):
    """Fade price excursions outside Bollinger bands."""

    name = "bollinger_fade"

    def parameter_grid(self) -> dict[str, list[float]]:
        return {
            "window": [15, 20, 30],
            "num_std": [1.5, 2.0, 2.5],
        }

    def generate_signals(self, dataframe: pd.DataFrame, params: dict[str, float] | None = None) -> pd.Series:
        settings = {"window": 20, "num_std": 2.0}
        settings.update(params or {})

        window = int(settings["window"])
        num_std = float(settings["num_std"])

        middle, upper, lower = bollinger_bands(dataframe, window=window, num_std=num_std)
        position = []
        current_position = 0.0

        for index in dataframe.index:
            close = float(dataframe.at[index, "close"])
            mid = middle.at[index]
            upper_band = upper.at[index]
            lower_band = lower.at[index]

            if pd.isna(mid) or pd.isna(upper_band) or pd.isna(lower_band):
                current_position = 0.0
            elif current_position == 0.0:
                if close < lower_band:
                    current_position = 1.0
                elif close > upper_band:
                    current_position = -1.0
            elif current_position > 0.0:
                if close >= mid:
                    current_position = 0.0
            else:
                if close <= mid:
                    current_position = 0.0
            position.append(current_position)

        return pd.Series(position, index=dataframe.index, dtype=float).clip(-1.0, 1.0)
