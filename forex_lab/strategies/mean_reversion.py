from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def rsi_reversal_signals(df: pd.DataFrame, params: Dict) -> pd.Series:
    period = int(params.get("rsi_period", 14))
    lower = float(params.get("lower", 30))
    upper = float(params.get("upper", 70))
    exit_level = float(params.get("exit_level", 50))

    rsi = df["rsi"] if "rsi" in df.columns else _rsi(df["close"], period=period)
    raw = pd.Series(0, index=df.index, dtype=float)
    raw[rsi < lower] = 1
    raw[rsi > upper] = -1
    pos = raw.replace(0, np.nan).ffill().fillna(0)

    long_exit = (pos > 0) & (rsi >= exit_level)
    short_exit = (pos < 0) & (rsi <= exit_level)
    pos[long_exit | short_exit] = 0
    return pos.replace(0, np.nan).ffill().fillna(0).astype(int)


def bollinger_fade_signals(df: pd.DataFrame, params: Dict) -> pd.Series:
    lookback = int(params.get("lookback", 20))
    stdev = float(params.get("stdev", 2.0))

    mid = df["close"].rolling(lookback).mean()
    sigma = df["close"].rolling(lookback).std()
    upper = mid + sigma * stdev
    lower = mid - sigma * stdev

    raw = pd.Series(0, index=df.index, dtype=float)
    raw[df["close"] < lower] = 1
    raw[df["close"] > upper] = -1
    pos = raw.replace(0, np.nan).ffill().fillna(0)

    exit_mask = ((pos > 0) & (df["close"] >= mid)) | ((pos < 0) & (df["close"] <= mid))
    pos[exit_mask] = 0
    return pos.replace(0, np.nan).ffill().fillna(0).astype(int)
