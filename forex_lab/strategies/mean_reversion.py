"""
Mean-reversion strategies.

1. RSIReversal      — fade extreme RSI readings
2. BollingerFade    — fade price at Bollinger band extremes
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .base import BaseStrategy, SignalSeries
from data.indicators import rsi as compute_rsi, bollinger_bands


class RSIReversal(BaseStrategy):
    """RSI mean-reversion strategy.

    Long  when RSI falls below oversold threshold (buy dip).
    Short when RSI rises above overbought threshold (sell rally).
    Exit  when RSI crosses back through neutral zone.

    Params
    ------
    period      : int   RSI lookback (default 14)
    oversold    : float RSI level to go long (default 30)
    overbought  : float RSI level to go short (default 70)
    exit_mid    : float RSI mid-level exit (default 50)
    """

    name = "rsi_reversal"

    def default_params(self) -> dict[str, Any]:
        return {"period": 14, "oversold": 30, "overbought": 70, "exit_mid": 50}

    def generate_signals(self, df: pd.DataFrame, params: dict[str, Any]) -> SignalSeries:
        period = int(params.get("period", 14))
        oversold = float(params.get("oversold", 30))
        overbought = float(params.get("overbought", 70))
        exit_mid = float(params.get("exit_mid", 50))

        rsi_s = compute_rsi(df["close"], period)

        signal = pd.Series(0, index=df.index, dtype="int8")

        # Entry signals (using previous bar RSI to avoid lookahead)
        rsi_prev = rsi_s.shift(1)
        signal[rsi_prev < oversold] = 1
        signal[rsi_prev > overbought] = -1

        # Stateful exit: hold until RSI crosses mid
        position = 0
        positions = []
        for i in range(len(signal)):
            entry = int(signal.iloc[i])
            cur_rsi = rsi_s.iloc[i]
            if entry != 0:
                position = entry
            elif position == 1 and cur_rsi >= exit_mid:
                position = 0
            elif position == -1 and cur_rsi <= exit_mid:
                position = 0
            positions.append(position)

        result = pd.Series(positions, index=df.index, dtype="int8")
        result[rsi_s.isna()] = 0
        return self._clip_signals(result)


class BollingerFade(BaseStrategy):
    """Bollinger Band fade strategy.

    Long  when price touches or breaks below lower band.
    Short when price touches or breaks above upper band.
    Exit  when price returns to middle band (SMA).

    Params
    ------
    period  : int   BB lookback (default 20)
    num_std : float standard deviations (default 2.0)
    """

    name = "bollinger_fade"

    def default_params(self) -> dict[str, Any]:
        return {"period": 20, "num_std": 2.0}

    def generate_signals(self, df: pd.DataFrame, params: dict[str, Any]) -> SignalSeries:
        period = int(params.get("period", 20))
        num_std = float(params.get("num_std", 2.0))

        close = df["close"]
        upper, mid, lower = bollinger_bands(close, period, num_std)

        # Shift bands by 1 bar to avoid lookahead
        upper_prev = upper.shift(1)
        lower_prev = lower.shift(1)
        mid_prev = mid.shift(1)

        signal = pd.Series(0, index=df.index, dtype="int8")
        signal[close <= lower_prev] = 1
        signal[close >= upper_prev] = -1

        # Stateful exit at mid
        position = 0
        positions = []
        for i in range(len(signal)):
            entry = int(signal.iloc[i])
            cur_close = close.iloc[i]
            cur_mid = mid.iloc[i]
            if entry != 0:
                position = entry
            elif position == 1 and cur_close >= cur_mid:
                position = 0
            elif position == -1 and cur_close <= cur_mid:
                position = 0
            positions.append(position)

        result = pd.Series(positions, index=df.index, dtype="int8")
        result[upper.isna()] = 0
        return self._clip_signals(result)
