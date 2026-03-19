"""Technical indicator computation — all vectorized, no lookahead."""

from __future__ import annotations

import pandas as pd
import numpy as np


def compute_indicators(
    df: pd.DataFrame,
    ma_periods: list[int] | None = None,
    rsi_period: int = 14,
    atr_period: int = 14,
    bb_period: int = 20,
    bb_std: float = 2.0,
    adx_period: int = 14,
) -> pd.DataFrame:
    """Attach all standard indicators to a copy of the DataFrame."""
    df = df.copy()

    if ma_periods is None:
        ma_periods = [20, 50, 200]

    for p in ma_periods:
        df[f"ma_{p}"] = df["close"].rolling(p).mean()

    df["rsi"] = _rsi(df["close"], rsi_period)
    df["atr"] = _atr(df, atr_period)

    df[f"bb_mid"] = df["close"].rolling(bb_period).mean()
    bb_roll_std = df["close"].rolling(bb_period).std()
    df[f"bb_upper"] = df["bb_mid"] + bb_std * bb_roll_std
    df[f"bb_lower"] = df["bb_mid"] - bb_std * bb_roll_std

    df["adx"] = _adx(df, adx_period)

    return df


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(span=period, min_periods=period, adjust=False).mean()


def _adx(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["high"]
    low = df["low"]
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    plus_dm = (high - prev_high).clip(lower=0)
    minus_dm = (prev_low - low).clip(lower=0)

    plus_dm = plus_dm.where(plus_dm > minus_dm, 0.0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0.0)

    atr = _atr(df, period)
    atr_safe = atr.replace(0, np.nan)

    plus_di = 100 * plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr_safe
    minus_di = 100 * minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr_safe

    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    return adx
