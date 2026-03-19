from __future__ import annotations

import numpy as np
import pandas as pd

from .filters import apply_filter


def rsi_reversal_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    rsi_col = params.get("rsi_col", "rsi_14")
    oversold = float(params.get("oversold", 30))
    overbought = float(params.get("overbought", 70))
    exit_level = float(params.get("exit_level", 50))

    rsi = df[rsi_col]
    signal = pd.Series(0, index=df.index, dtype=float)
    signal[rsi < oversold] = 1
    signal[rsi > overbought] = -1

    # Flat when RSI mean-reverts toward neutral.
    flatten = (signal == 0) & ((rsi >= exit_level - 2) & (rsi <= exit_level + 2))
    signal = signal.replace(0, np.nan).ffill().fillna(0)
    signal[flatten] = 0
    out = signal.astype(int)

    filter_condition = params.get("filter_condition")
    if filter_condition is not None:
        out = apply_filter(out, condition=filter_condition)
    return out


def bollinger_fade_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    upper_col = params.get("upper_col", "bb_upper_20_2")
    lower_col = params.get("lower_col", "bb_lower_20_2")
    mid_col = params.get("mid_col", "bb_mid_20")

    signal = pd.Series(0, index=df.index, dtype=float)
    signal[df["close"] < df[lower_col]] = 1
    signal[df["close"] > df[upper_col]] = -1

    signal = signal.replace(0, np.nan).ffill().fillna(0)
    near_mid = (df["close"] - df[mid_col]).abs() <= (
        0.1 * (df[upper_col] - df[lower_col]).fillna(0)
    )
    signal[near_mid] = 0
    return signal.astype(int)
