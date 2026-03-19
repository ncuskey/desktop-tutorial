from __future__ import annotations

import numpy as np
import pandas as pd


def ma_crossover_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    fast = int(params.get("fast", 20))
    slow = int(params.get("slow", 50))
    if fast >= slow:
        raise ValueError("fast MA must be < slow MA")

    fast_ma = df["close"].rolling(fast, min_periods=fast).mean()
    slow_ma = df["close"].rolling(slow, min_periods=slow).mean()
    signal = np.where(fast_ma > slow_ma, 1, -1)
    signal = pd.Series(signal, index=df.index, dtype=float)
    signal[(fast_ma.isna()) | (slow_ma.isna())] = 0
    return signal.astype(int)


def donchian_breakout_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    lookback = int(params.get("lookback", 20))
    upper = df["high"].rolling(lookback, min_periods=lookback).max().shift(1)
    lower = df["low"].rolling(lookback, min_periods=lookback).min().shift(1)

    long_break = df["close"] > upper
    short_break = df["close"] < lower

    signal = pd.Series(0, index=df.index, dtype=float)
    signal[long_break] = 1
    signal[short_break] = -1
    # Carry last position between breakout events.
    return signal.replace(0, np.nan).ffill().fillna(0).astype(int)
