"""Regime classifier: volatility (ATR) + trend strength (ADX) -> switch strategy."""

from typing import Any, Optional

import numpy as np
import pandas as pd

from forex_lab.strategies.base import BaseStrategy


class RegimeClassifierOrchestrator:
    """
    Classify regime using ATR (volatility) and ADX (trend strength).
    Regimes: trending, ranging, volatile.
    """

    def __init__(
        self,
        strategies: dict[str, BaseStrategy],
        atr_column: str = "atr",
        adx_column: str = "adx",
        adx_threshold: float = 25.0,
        atr_pct_threshold: float = 0.5,
    ):
        """

        strategies: {"trending": ..., "ranging": ..., "volatile": ...}
        atr_pct_threshold: ATR percentile above which = volatile
        """
        self.strategies = strategies
        self.atr_column = atr_column
        self.adx_column = adx_column
        self.adx_threshold = adx_threshold
        self.atr_pct_threshold = atr_pct_threshold

    def run(
        self,
        df: pd.DataFrame,
        params_list: dict[str, dict[str, Any]],
    ) -> pd.Series:
        """
        Classify regime: high ADX -> trend, low ADX + low vol -> range, high vol -> volatile.
        """
        if self.atr_column not in df.columns or self.adx_column not in df.columns:
            raise ValueError("DataFrame must have atr and adx columns")

        adx = df[self.adx_column].fillna(0).values
        atr = df[self.atr_column].fillna(0)
        atr_ma = atr.rolling(100, min_periods=20).mean()
        atr_ratio = (atr / atr_ma.replace(0, np.nan)).fillna(1.0).values

        regime = np.where(adx >= self.adx_threshold, "trending", "ranging")
        regime = np.where(atr_ratio > (1 + self.atr_pct_threshold), "volatile", regime)

        signals = {}
        for name, strat in self.strategies.items():
            if name in params_list:
                signals[name] = strat.generate_signals(df, params_list[name])

        position = pd.Series(0, index=df.index)
        for i in range(len(df)):
            r = regime[i]
            if r in signals:
                position.iloc[i] = signals[r].iloc[i]

        return position.fillna(0).astype(int)
