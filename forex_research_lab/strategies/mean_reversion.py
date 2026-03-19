"""Mean-reversion Forex strategies."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from forex_research_lab.data.indicators import bollinger_bands, rsi


def rsi_reversal_generate_signals(df: pd.DataFrame, params: Mapping[str, float]) -> pd.Series:
    """RSI reversal strategy with explicit exit threshold."""

    rsi_window = int(params.get("rsi_window", 14))
    lower = float(params.get("lower", 30))
    upper = float(params.get("upper", 70))
    exit_level = float(params.get("exit_level", 50))

    rsi_series = rsi(df["close"], window=rsi_window)
    rsi_values = rsi_series.to_numpy()
    signals = np.zeros(len(df), dtype=float)

    position = 0.0
    for i, current_rsi in enumerate(rsi_values):
        if np.isnan(current_rsi):
            signals[i] = 0.0
            continue

        if position == 0.0:
            if current_rsi <= lower:
                position = 1.0
            elif current_rsi >= upper:
                position = -1.0
        elif position > 0 and current_rsi >= exit_level:
            position = 0.0
        elif position < 0 and current_rsi <= exit_level:
            position = 0.0

        signals[i] = position

    return pd.Series(signals, index=df.index, name="rsi_reversal_signal")


def bollinger_fade_generate_signals(df: pd.DataFrame, params: Mapping[str, float]) -> pd.Series:
    """Fade Bollinger band extremes and exit at the mid-band."""

    window = int(params.get("window", 20))
    std_factor = float(params.get("std_factor", 2.0))

    bands = bollinger_bands(df["close"], window=window, std_factor=std_factor)
    close = df["close"]
    upper = bands["bb_upper"]
    lower = bands["bb_lower"]
    mid = bands["bb_mid"]

    signals = np.zeros(len(df), dtype=float)
    position = 0.0

    for i in range(len(df)):
        c = close.iat[i]
        up = upper.iat[i]
        lo = lower.iat[i]
        md = mid.iat[i]

        if np.isnan(up) or np.isnan(lo) or np.isnan(md):
            signals[i] = 0.0
            continue

        if position == 0.0:
            if c > up:
                position = -1.0
            elif c < lo:
                position = 1.0
        elif position > 0 and c >= md:
            position = 0.0
        elif position < 0 and c <= md:
            position = 0.0

        signals[i] = position

    return pd.Series(signals, index=df.index, name="bollinger_fade_signal")
