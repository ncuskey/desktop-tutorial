"""Technical indicator computation — pure pandas/numpy, no look-ahead."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def _bollinger(series: pd.Series, period: int = 20, num_std: float = 2.0):
    mid = _sma(series, period)
    std = series.rolling(period, min_periods=period).std()
    return mid, mid + num_std * std, mid - num_std * std


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)

    mask = plus_dm < minus_dm
    plus_dm[mask] = 0.0
    minus_dm[~mask] = 0.0

    atr = _atr(df, period)
    plus_di = 100 * _ema(plus_dm, period) / atr.replace(0, np.nan)
    minus_di = 100 * _ema(minus_dm, period) / atr.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = _ema(dx, period)
    return adx


def add_indicators(
    df: pd.DataFrame,
    ma_periods: tuple[int, ...] = (20, 50),
    rsi_period: int = 14,
    atr_period: int = 14,
    bb_period: int = 20,
    bb_std: float = 2.0,
    adx_period: int = 14,
) -> pd.DataFrame:
    """Attach standard indicators to an OHLCV DataFrame (in-place and returned)."""
    df = df.copy()
    for p in ma_periods:
        df[f"sma_{p}"] = _sma(df["close"], p)
        df[f"ema_{p}"] = _ema(df["close"], p)

    df["rsi"] = _rsi(df["close"], rsi_period)
    df["atr"] = _atr(df, atr_period)

    bb_mid, bb_upper, bb_lower = _bollinger(df["close"], bb_period, bb_std)
    df["bb_mid"] = bb_mid
    df["bb_upper"] = bb_upper
    df["bb_lower"] = bb_lower

    df["adx"] = _adx(df, adx_period)

    df["donchian_high"] = df["high"].rolling(20, min_periods=20).max()
    df["donchian_low"] = df["low"].rolling(20, min_periods=20).min()

    return df
