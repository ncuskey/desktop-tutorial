"""Mean-reversion strategies."""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Any

from .base import Strategy


class RSIReversal(Strategy):
    """RSI mean reversion — long when oversold, short when overbought."""

    name = "rsi_reversal"

    def default_params(self) -> dict[str, Any]:
        return {"rsi_period": 14, "oversold": 30.0, "overbought": 70.0}

    def param_grid(self) -> dict[str, list[Any]]:
        return {
            "rsi_period": [7, 14, 21],
            "oversold": [20, 25, 30],
            "overbought": [70, 75, 80],
        }

    def generate_signals(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        period = params.get("rsi_period", 14)
        oversold = params.get("oversold", 30.0)
        overbought = params.get("overbought", 70.0)

        rsi = self._rsi(df["close"], period)

        signal = pd.Series(0.0, index=df.index)
        signal[rsi < oversold] = 1.0
        signal[rsi > overbought] = -1.0
        signal.iloc[:period] = 0.0
        return signal

    @staticmethod
    def _rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))


class BollingerFade(Strategy):
    """Bollinger Band fade — long below lower band, short above upper band."""

    name = "bollinger_fade"

    def default_params(self) -> dict[str, Any]:
        return {"period": 20, "num_std": 2.0}

    def param_grid(self) -> dict[str, list[Any]]:
        return {
            "period": [15, 20, 25, 30],
            "num_std": [1.5, 2.0, 2.5, 3.0],
        }

    def generate_signals(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        period = params.get("period", 20)
        num_std = params.get("num_std", 2.0)

        mid = df["close"].rolling(period).mean()
        std = df["close"].rolling(period).std()
        upper = mid + num_std * std
        lower = mid - num_std * std

        signal = pd.Series(0.0, index=df.index)
        signal[df["close"] < lower] = 1.0
        signal[df["close"] > upper] = -1.0
        signal.iloc[:period] = 0.0
        return signal
