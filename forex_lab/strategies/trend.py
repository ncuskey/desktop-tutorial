from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def ma_crossover_signals(df: pd.DataFrame, params: Dict) -> pd.Series:
    short_window = int(params.get("short_window", 20))
    long_window = int(params.get("long_window", 50))
    if short_window >= long_window:
        raise ValueError("short_window must be smaller than long_window")

    fast = df["close"].rolling(short_window).mean()
    slow = df["close"].rolling(long_window).mean()
    signal = pd.Series(np.where(fast > slow, 1, -1), index=df.index, dtype=float)
    signal[(fast.isna()) | (slow.isna())] = 0
    return signal.astype(int)


def donchian_breakout_signals(df: pd.DataFrame, params: Dict) -> pd.Series:
    lookback = int(params.get("lookback", 20))
    upper = df["high"].shift(1).rolling(lookback).max()
    lower = df["low"].shift(1).rolling(lookback).min()

    signal = pd.Series(0, index=df.index, dtype=float)
    signal[df["close"] > upper] = 1
    signal[df["close"] < lower] = -1
    return signal.replace(0, np.nan).ffill().fillna(0).astype(int)
