"""Mean-reversion strategies."""

from __future__ import annotations

import pandas as pd

from forex_research_lab.data.indicators import compute_rsi

from .base import BaseStrategy


class RSIReversalStrategy(BaseStrategy):
    name = "rsi_reversal"

    def generate_signals(self, df: pd.DataFrame, params: dict) -> pd.Series:
        window = int(params.get("window", 14))
        oversold = float(params.get("oversold", 30))
        overbought = float(params.get("overbought", 70))
        neutral_level = float(params.get("neutral_level", 50))
        neutral_band = float(params.get("neutral_band", 5))

        rsi = compute_rsi(df["close"], window=window)
        signal = pd.Series(0, index=df.index, dtype=int)
        signal = signal.mask(rsi < oversold, 1)
        signal = signal.mask(rsi > overbought, -1)

        # Once the market reverts toward neutral, flatten the position.
        neutral = rsi.between(neutral_level - neutral_band, neutral_level + neutral_band)
        signal = signal.where(~neutral, 0)
        return signal.ffill().fillna(0).astype(int)


class BollingerFadeStrategy(BaseStrategy):
    name = "bollinger_fade"

    def generate_signals(self, df: pd.DataFrame, params: dict) -> pd.Series:
        window = int(params.get("window", 20))
        num_std = float(params.get("num_std", 2.0))

        mid = df["close"].rolling(window=window, min_periods=window).mean()
        std = df["close"].rolling(window=window, min_periods=window).std(ddof=0)
        upper = mid + num_std * std
        lower = mid - num_std * std

        signal = pd.Series(0, index=df.index, dtype=int)
        signal = signal.mask(df["close"] < lower, 1)
        signal = signal.mask(df["close"] > upper, -1)
        signal = signal.where(df["close"].notna(), 0)
        return signal.ffill().fillna(0).astype(int)
