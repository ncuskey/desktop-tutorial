"""Technical indicators used across strategies and orchestration layers."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def moving_average(series: pd.Series, window: int) -> pd.Series:
    """Compute a simple moving average."""
    return series.rolling(window=window, min_periods=window).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Compute a Wilder-style RSI."""
    delta = series.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)

    average_gain = gains.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    average_loss = losses.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    relative_strength = average_gain / average_loss.replace(0.0, np.nan)
    result = 100.0 - (100.0 / (1.0 + relative_strength))
    result = result.where(average_loss != 0.0, 100.0)
    result = result.where(~((average_gain == 0.0) & (average_loss == 0.0)), 50.0)
    return result


def true_range(dataframe: pd.DataFrame) -> pd.Series:
    """Compute true range."""
    previous_close = dataframe["close"].shift(1)
    ranges = pd.concat(
        [
            dataframe["high"] - dataframe["low"],
            (dataframe["high"] - previous_close).abs(),
            (dataframe["low"] - previous_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(dataframe: pd.DataFrame, window: int = 14) -> pd.Series:
    """Compute average true range."""
    return true_range(dataframe).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def bollinger_bands(dataframe: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute Bollinger band middle, upper, and lower series."""
    middle = moving_average(dataframe["close"], window=window)
    standard_deviation = dataframe["close"].rolling(window=window, min_periods=window).std(ddof=0)
    upper = middle + (num_std * standard_deviation)
    lower = middle - (num_std * standard_deviation)
    return middle, upper, lower


def adx(dataframe: pd.DataFrame, window: int = 14) -> pd.Series:
    """Compute average directional index."""
    up_move = dataframe["high"].diff()
    down_move = -dataframe["low"].diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0.0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0.0), 0.0)

    average_true_range = atr(dataframe, window=window).replace(0.0, np.nan)
    plus_di = 100.0 * plus_dm.ewm(alpha=1 / window, adjust=False, min_periods=window).mean() / average_true_range
    minus_di = 100.0 * minus_dm.ewm(alpha=1 / window, adjust=False, min_periods=window).mean() / average_true_range

    directional_sum = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / directional_sum
    return dx.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def add_basic_indicators(
    dataframe: pd.DataFrame,
    ma_windows: Iterable[int] = (20, 50, 100),
    rsi_window: int = 14,
    atr_window: int = 14,
    bollinger_window: int = 20,
    bollinger_std: float = 2.0,
    adx_window: int = 14,
) -> pd.DataFrame:
    """Attach the standard indicator set used by the research framework."""
    enriched = dataframe.copy()

    for window in ma_windows:
        enriched[f"ma_{window}"] = moving_average(enriched["close"], window)

    enriched[f"rsi_{rsi_window}"] = rsi(enriched["close"], window=rsi_window)
    enriched[f"atr_{atr_window}"] = atr(enriched, window=atr_window)

    middle, upper, lower = bollinger_bands(enriched, window=bollinger_window, num_std=bollinger_std)
    band_suffix = f"{bollinger_window}_{bollinger_std}"
    enriched[f"bb_middle_{band_suffix}"] = middle
    enriched[f"bb_upper_{band_suffix}"] = upper
    enriched[f"bb_lower_{band_suffix}"] = lower
    enriched[f"adx_{adx_window}"] = adx(enriched, window=adx_window)
    return enriched
