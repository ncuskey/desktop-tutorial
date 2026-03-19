"""Mean reversion strategies."""

from typing import Any

import numpy as np
import pandas as pd

from .base import BaseStrategy


class RSIReversalStrategy(BaseStrategy):
    """
    RSI reversal: long when RSI oversold (< threshold_low), short when overbought (> threshold_high).
    """

    @property
    def default_params(self) -> dict[str, Any]:
        return {"period": 14, "threshold_low": 30, "threshold_high": 70}

    def generate_signals(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        period = params.get("period", self.default_params["period"])
        thresh_low = params.get("threshold_low", self.default_params["threshold_low"])
        thresh_high = params.get("threshold_high", self.default_params["threshold_high"])

        # Compute RSI if not present
        if "rsi" not in df.columns:
            delta = df["close"].diff()
            gain = delta.where(delta > 0, 0.0)
            loss = (-delta).where(delta < 0, 0.0)
            avg_gain = gain.rolling(period, min_periods=period).mean()
            avg_loss = loss.rolling(period, min_periods=period).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            rsi_series = 100 - (100 / (1 + rs))
        else:
            rsi_series = df["rsi"]

        position = np.where(
            rsi_series < thresh_low, 1, np.where(rsi_series > thresh_high, -1, 0)
        )
        out = pd.Series(position, index=df.index).replace(0, np.nan).ffill().fillna(0)
        return out.astype(int)


class BollingerFadeStrategy(BaseStrategy):
    """
    Bollinger band fade: long when price below lower band, short when above upper band.
    """

    @property
    def default_params(self) -> dict[str, Any]:
        return {"period": 20, "num_std": 2.0}

    def generate_signals(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        period = params.get("period", self.default_params["period"])
        num_std = params.get("num_std", self.default_params["num_std"])

        middle = df["close"].rolling(period, min_periods=period).mean()
        std = df["close"].rolling(period, min_periods=period).std()
        upper = middle + num_std * std
        lower = middle - num_std * std
        close = df["close"]

        position = np.where(
            close < lower, 1, np.where(close > upper, -1, 0)
        )
        out = pd.Series(position, index=df.index).replace(0, np.nan).ffill().fillna(0)
        return out.astype(int)
