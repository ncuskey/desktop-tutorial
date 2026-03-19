from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def range_breakout_signals(df: pd.DataFrame, params: Dict) -> pd.Series:
    lookback = int(params.get("lookback", 30))
    range_high = df["high"].shift(1).rolling(lookback).max()
    range_low = df["low"].shift(1).rolling(lookback).min()

    raw = pd.Series(0, index=df.index, dtype=float)
    raw[df["close"] > range_high] = 1
    raw[df["close"] < range_low] = -1
    return raw.replace(0, np.nan).ffill().fillna(0).astype(int)


def volatility_expansion_breakout_signals(df: pd.DataFrame, params: Dict) -> pd.Series:
    atr_window = int(params.get("atr_window", 14))
    expansion_multiple = float(params.get("expansion_multiple", 1.2))

    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(atr_window).mean()

    breakout_bar = (df["high"] - df["low"]) > (atr * expansion_multiple)
    up = breakout_bar & (df["close"] > df["close"].shift(1))
    down = breakout_bar & (df["close"] < df["close"].shift(1))

    raw = pd.Series(0, index=df.index, dtype=float)
    raw[up] = 1
    raw[down] = -1
    return raw.replace(0, np.nan).ffill().fillna(0).astype(int)
