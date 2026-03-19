"""Mean-reversion strategies."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .base import Strategy


class RSIReversal(Strategy):
    """RSI reversal — long when oversold, short when overbought."""

    name = "rsi_reversal"

    def generate_signals(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        period = params.get("rsi_period", 14)
        oversold = params.get("oversold", 30)
        overbought = params.get("overbought", 70)

        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        signal = pd.Series(0, index=df.index, dtype=int)
        signal[rsi < oversold] = 1
        signal[rsi > overbought] = -1
        signal.iloc[:period] = 0
        return signal


class BollingerFade(Strategy):
    """Bollinger band fade — long at lower band, short at upper band."""

    name = "bollinger_fade"

    def generate_signals(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        period = params.get("bb_period", 20)
        num_std = params.get("bb_std", 2.0)

        mid = df["close"].rolling(period, min_periods=period).mean()
        std = df["close"].rolling(period, min_periods=period).std()
        upper = mid + num_std * std
        lower = mid - num_std * std

        signal = pd.Series(0, index=df.index, dtype=int)
        signal[df["close"] < lower] = 1
        signal[df["close"] > upper] = -1
        signal.iloc[:period] = 0
        return signal
