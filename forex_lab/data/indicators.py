"""Technical indicators: MA, RSI, ATR, Bollinger, ADX."""

from typing import Optional

import numpy as np
import pandas as pd


def _ensure_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure we have a proper index for vectorized operations."""
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.DatetimeIndex(df.index)
    return df


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=period, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


def bollinger_bands(
    close: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands: middle, upper, lower."""
    middle = sma(close, period)
    std = close.rolling(window=period, min_periods=period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return middle, upper, lower


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average Directional Index (trend strength)."""
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    plus_dm = high - prev_high
    minus_dm = prev_low - low

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = atr(high, low, close, period)

    atr_14 = tr
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr_14.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr_14.replace(0, np.nan))

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_series = dx.rolling(period).mean()

    return adx_series


def compute_indicators(
    df: pd.DataFrame,
    ma_fast: int = 10,
    ma_slow: int = 30,
    rsi_period: int = 14,
    atr_period: int = 14,
    bb_period: int = 20,
    bb_std: float = 2.0,
    adx_period: int = 14,
) -> pd.DataFrame:
    """
    Compute standard indicators and attach to DataFrame.

    No lookahead: all indicators use only past data (rolling).
    """
    df = _ensure_index(df).copy()

    close = df["close"]
    high = df["high"]
    low = df["low"]

    df["ma_fast"] = sma(close, ma_fast)
    df["ma_slow"] = sma(close, ma_slow)
    df["rsi"] = rsi(close, rsi_period)
    df["atr"] = atr(high, low, close, atr_period)

    bb_mid, bb_upper, bb_lower = bollinger_bands(close, bb_period, bb_std)
    df["bb_middle"] = bb_mid
    df["bb_upper"] = bb_upper
    df["bb_lower"] = bb_lower

    df["adx"] = adx(high, low, close, adx_period)

    return df
