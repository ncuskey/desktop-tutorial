"""Technical indicator utilities built on pandas/numpy."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def moving_average(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr_components = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return tr_components.max(axis=1)


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def bollinger_bands(
    series: pd.Series,
    window: int = 20,
    std_factor: float = 2.0,
) -> pd.DataFrame:
    ma = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std(ddof=0)
    upper = ma + std_factor * std
    lower = ma - std_factor * std
    return pd.DataFrame({"bb_mid": ma, "bb_upper": upper, "bb_lower": lower}, index=series.index)


def adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
    )

    tr = true_range(df)
    atr_series = tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / window, min_periods=window, adjust=False).mean() / atr_series
    minus_di = 100 * minus_dm.ewm(alpha=1 / window, min_periods=window, adjust=False).mean() / atr_series

    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = (plus_di - minus_di).abs() / di_sum * 100
    return dx.ewm(alpha=1 / window, min_periods=window, adjust=False).mean().fillna(0.0)


def add_indicators(
    df: pd.DataFrame,
    ma_windows: Iterable[int] = (10, 20, 50),
    rsi_window: int = 14,
    atr_window: int = 14,
    bollinger_window: int = 20,
    bollinger_std: float = 2.0,
    adx_window: int = 14,
) -> pd.DataFrame:
    """Add core indicator columns used by strategies and orchestrators."""

    out = df.copy()
    for window in ma_windows:
        out[f"ma_{window}"] = moving_average(out["close"], window)

    out[f"rsi_{rsi_window}"] = rsi(out["close"], rsi_window)
    out[f"atr_{atr_window}"] = atr(out, atr_window)
    out[f"adx_{adx_window}"] = adx(out, adx_window)

    bb = bollinger_bands(out["close"], window=bollinger_window, std_factor=bollinger_std)
    out = out.join(bb)
    return out
