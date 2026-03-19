"""Indicator calculations used across strategies and research modules."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=window, min_periods=window).mean()


def compute_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=high.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=high.index,
    )

    atr = compute_atr(high, low, close, window=window).replace(0.0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1 / window, adjust=False, min_periods=window).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / window, adjust=False, min_periods=window).mean() / atr
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan)
    return dx.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def compute_indicators(
    df: pd.DataFrame,
    ma_windows: tuple[int, ...] = (10, 20, 50, 100),
    rsi_window: int = 14,
    atr_window: int = 14,
    bollinger_window: int = 20,
    bollinger_std: float = 2.0,
    adx_window: int = 14,
) -> pd.DataFrame:
    """Compute causal indicators per symbol without introducing lookahead."""

    frames: list[pd.DataFrame] = []
    for _, group in df.groupby("symbol", sort=True):
        enriched = group.sort_values("timestamp").copy()
        for window in ma_windows:
            enriched[f"ma_{window}"] = enriched["close"].rolling(window=window, min_periods=window).mean()

        enriched[f"rsi_{rsi_window}"] = compute_rsi(enriched["close"], window=rsi_window)
        enriched[f"atr_{atr_window}"] = compute_atr(
            enriched["high"],
            enriched["low"],
            enriched["close"],
            window=atr_window,
        )
        rolling_mean = enriched["close"].rolling(
            window=bollinger_window,
            min_periods=bollinger_window,
        ).mean()
        rolling_std = enriched["close"].rolling(
            window=bollinger_window,
            min_periods=bollinger_window,
        ).std(ddof=0)
        enriched[f"bollinger_mid_{bollinger_window}"] = rolling_mean
        enriched[f"bollinger_upper_{bollinger_window}"] = rolling_mean + bollinger_std * rolling_std
        enriched[f"bollinger_lower_{bollinger_window}"] = rolling_mean - bollinger_std * rolling_std
        enriched[f"adx_{adx_window}"] = compute_adx(
            enriched["high"],
            enriched["low"],
            enriched["close"],
            window=adx_window,
        )
        frames.append(enriched)

    if not frames:
        return df.copy()

    return pd.concat(frames, ignore_index=True).sort_values(["symbol", "timestamp"]).reset_index(drop=True)
