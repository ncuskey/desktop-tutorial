"""Breakout strategy set."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from forex_research_lab.data.indicators import atr


def range_breakout_generate_signals(df: pd.DataFrame, params: Mapping[str, float]) -> pd.Series:
    """Simple rolling range breakout strategy."""

    lookback = int(params.get("lookback", 24))
    upper = df["high"].rolling(lookback, min_periods=lookback).max().shift(1)
    lower = df["low"].rolling(lookback, min_periods=lookback).min().shift(1)

    raw = np.select(
        condlist=[df["close"] > upper, df["close"] < lower],
        choicelist=[1.0, -1.0],
        default=np.nan,
    )
    return pd.Series(raw, index=df.index, name="range_breakout_signal").ffill().fillna(0.0)


def volatility_expansion_breakout_generate_signals(
    df: pd.DataFrame,
    params: Mapping[str, float],
) -> pd.Series:
    """
    Enter breakouts only when ATR is elevated relative to its own average.
    """

    lookback = int(params.get("lookback", 24))
    atr_window = int(params.get("atr_window", 14))
    vol_multiplier = float(params.get("vol_multiplier", 1.2))

    atr_series = atr(df, window=atr_window)
    atr_mean = atr_series.rolling(window=lookback, min_periods=lookback).mean()

    upper = df["high"].rolling(lookback, min_periods=lookback).max().shift(1)
    lower = df["low"].rolling(lookback, min_periods=lookback).min().shift(1)
    vol_ok = atr_series > (atr_mean * vol_multiplier)

    raw = np.select(
        condlist=[(df["close"] > upper) & vol_ok, (df["close"] < lower) & vol_ok],
        choicelist=[1.0, -1.0],
        default=np.nan,
    )
    return pd.Series(raw, index=df.index, name="vol_expansion_breakout_signal").ffill().fillna(0.0)
