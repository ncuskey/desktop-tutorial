from __future__ import annotations

import numpy as np
import pandas as pd


def range_breakout_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    lookback = int(params.get("lookback", 30))
    high_range = df["high"].rolling(lookback, min_periods=lookback).max().shift(1)
    low_range = df["low"].rolling(lookback, min_periods=lookback).min().shift(1)

    signal = pd.Series(0, index=df.index, dtype=float)
    signal[df["close"] > high_range] = 1
    signal[df["close"] < low_range] = -1
    return signal.replace(0, np.nan).ffill().fillna(0).astype(int)


def volatility_expansion_breakout_signals(df: pd.DataFrame, params: dict) -> pd.Series:
    atr_col = params.get("atr_col", "atr_14")
    atr_ma_len = int(params.get("atr_ma_len", 20))
    expansion_threshold = float(params.get("expansion_threshold", 1.2))
    momentum_len = int(params.get("momentum_len", 5))

    atr = df[atr_col]
    atr_ma = atr.rolling(atr_ma_len, min_periods=atr_ma_len).mean()
    expansion = atr > (atr_ma * expansion_threshold)
    momentum = df["close"].pct_change(momentum_len)

    signal = pd.Series(0, index=df.index, dtype=float)
    signal[expansion & (momentum > 0)] = 1
    signal[expansion & (momentum < 0)] = -1
    return signal.replace(0, np.nan).ffill().fillna(0).astype(int)
