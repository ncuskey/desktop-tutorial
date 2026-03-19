"""
Technical indicators computed on OHLCV DataFrames.

All indicators use only past data (no lookahead).
Results are added as new columns to a copy of the input frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


# ---------------------------------------------------------------------------
# Individual indicators
# ---------------------------------------------------------------------------

def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    return _ema(series, period)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    tr = _true_range(df)
    return tr.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder smoothing)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def bollinger_bands(
    series: pd.Series, period: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Upper band, middle (SMA), lower band."""
    mid = sma(series, period)
    std = series.rolling(period).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index (Wilder)."""
    tr = _true_range(df)
    up_move = df["high"] - df["high"].shift(1)
    dn_move = df["low"].shift(1) - df["low"]

    pos_dm = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
    neg_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)

    atr_s = tr.ewm(alpha=1 / period, adjust=False).mean()
    pdi = 100 * pd.Series(pos_dm, index=df.index).ewm(
        alpha=1 / period, adjust=False
    ).mean() / atr_s
    ndi = 100 * pd.Series(neg_dm, index=df.index).ewm(
        alpha=1 / period, adjust=False
    ).mean() / atr_s

    dx = (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan) * 100
    adx_s = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx_s


def donchian_channels(
    df: pd.DataFrame, period: int = 20
) -> tuple[pd.Series, pd.Series]:
    """Upper (rolling max high) and lower (rolling min low) Donchian channels."""
    upper = df["high"].rolling(period).max()
    lower = df["low"].rolling(period).min()
    return upper, lower


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, histogram."""
    fast_ema = _ema(series, fast)
    slow_ema = _ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# ---------------------------------------------------------------------------
# Convenience: add all indicators to a DataFrame
# ---------------------------------------------------------------------------

def add_indicators(
    df: pd.DataFrame,
    sma_periods: list[int] | None = None,
    ema_periods: list[int] | None = None,
    rsi_period: int = 14,
    atr_period: int = 14,
    bb_period: int = 20,
    bb_std: float = 2.0,
    adx_period: int = 14,
    donchian_period: int = 20,
) -> pd.DataFrame:
    """Return a copy of *df* with standard indicators appended as columns."""
    df = df.copy()

    sma_periods = sma_periods or [20, 50, 200]
    ema_periods = ema_periods or [12, 26, 50]

    for p in sma_periods:
        df[f"sma_{p}"] = sma(df["close"], p)
    for p in ema_periods:
        df[f"ema_{p}"] = ema(df["close"], p)

    df["rsi"] = rsi(df["close"], rsi_period)
    df["atr"] = atr(df, atr_period)
    df["atr_pct"] = df["atr"] / df["close"]

    bb_upper, bb_mid, bb_lower = bollinger_bands(df["close"], bb_period, bb_std)
    df["bb_upper"] = bb_upper
    df["bb_mid"] = bb_mid
    df["bb_lower"] = bb_lower
    df["bb_width"] = (bb_upper - bb_lower) / bb_mid
    df["bb_pct"] = (df["close"] - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)

    df["adx"] = adx(df, adx_period)

    dc_upper, dc_lower = donchian_channels(df, donchian_period)
    df["dc_upper"] = dc_upper
    df["dc_lower"] = dc_lower

    macd_line, signal_line, histogram = macd(df["close"])
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = histogram

    return df
