from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_up = up.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    avg_down = down.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    rs = avg_up / avg_down.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()


def _adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr = _atr(df, length=length)
    plus_di = (
        100.0
        * pd.Series(plus_dm, index=df.index)
        .ewm(alpha=1.0 / length, adjust=False, min_periods=length)
        .mean()
        / atr.replace(0.0, np.nan)
    )
    minus_di = (
        100.0
        * pd.Series(minus_dm, index=df.index)
        .ewm(alpha=1.0 / length, adjust=False, min_periods=length)
        .mean()
        / atr.replace(0.0, np.nan)
    )
    dx = (100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)).replace(
        [np.inf, -np.inf], np.nan
    )
    return dx.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()


def add_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]

    out["ma_fast_20"] = close.rolling(window=20, min_periods=20).mean()
    out["ma_slow_50"] = close.rolling(window=50, min_periods=50).mean()
    out["rsi_14"] = _rsi(close, length=14)
    out["atr_14"] = _atr(out, length=14)

    bb_mid = close.rolling(window=20, min_periods=20).mean()
    bb_std = close.rolling(window=20, min_periods=20).std(ddof=0)
    out["bb_mid_20"] = bb_mid
    out["bb_upper_20_2"] = bb_mid + 2.0 * bb_std
    out["bb_lower_20_2"] = bb_mid - 2.0 * bb_std

    out["adx_14"] = _adx(out, length=14)
    return out
