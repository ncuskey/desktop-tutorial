"""Trend-following Forex strategies."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


def ma_crossover_generate_signals(df: pd.DataFrame, params: Mapping[str, float]) -> pd.Series:
    """Moving-average crossover strategy."""

    fast_window = int(params.get("fast_window", 20))
    slow_window = int(params.get("slow_window", 50))
    if fast_window >= slow_window:
        raise ValueError("fast_window must be smaller than slow_window")

    fast_ma = df["close"].rolling(window=fast_window, min_periods=fast_window).mean()
    slow_ma = df["close"].rolling(window=slow_window, min_periods=slow_window).mean()

    signal = np.where(fast_ma > slow_ma, 1.0, np.where(fast_ma < slow_ma, -1.0, 0.0))
    out = pd.Series(signal, index=df.index, name="ma_crossover_signal")
    out = out.where(~(fast_ma.isna() | slow_ma.isna()), 0.0)
    return out


def donchian_breakout_generate_signals(df: pd.DataFrame, params: Mapping[str, float]) -> pd.Series:
    """Donchian channel breakout strategy."""

    lookback = int(params.get("lookback", 20))
    if lookback < 2:
        raise ValueError("lookback must be >= 2")

    upper = df["high"].rolling(window=lookback, min_periods=lookback).max().shift(1)
    lower = df["low"].rolling(window=lookback, min_periods=lookback).min().shift(1)

    raw = np.select(
        condlist=[df["close"] > upper, df["close"] < lower],
        choicelist=[1.0, -1.0],
        default=np.nan,
    )
    out = pd.Series(raw, index=df.index, name="donchian_signal").ffill().fillna(0.0)
    return out
