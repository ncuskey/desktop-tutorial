"""Simplified carry strategy placeholder."""

from __future__ import annotations

import numpy as np
import pandas as pd

from forex_research_lab.strategies.base import BaseStrategy


class CarryProxyStrategy(BaseStrategy):
    """Use a carry differential proxy column as a directional signal."""

    name = "carry_proxy"

    def parameter_grid(self) -> dict[str, list[float]]:
        return {
            "window": [5, 10, 20],
            "threshold": [0.0, 0.00005, 0.0001],
        }

    def generate_signals(self, dataframe: pd.DataFrame, params: dict[str, float] | None = None) -> pd.Series:
        settings = {"window": 10, "threshold": 0.0}
        settings.update(params or {})

        if "carry_proxy" not in dataframe.columns:
            raise ValueError("CarryProxyStrategy requires a carry_proxy column")

        window = int(settings["window"])
        threshold = float(settings["threshold"])

        smoothed_proxy = dataframe["carry_proxy"].rolling(window=window, min_periods=window).mean()
        signal = pd.Series(0.0, index=dataframe.index, dtype=float)
        valid = smoothed_proxy.notna()
        signal.loc[valid] = np.where(
            smoothed_proxy.loc[valid] > threshold,
            1.0,
            np.where(smoothed_proxy.loc[valid] < -threshold, -1.0, 0.0),
        )
        return signal.clip(-1.0, 1.0)
