from __future__ import annotations

import numpy as np
import pandas as pd

from .labels import infer_filter_type_from_regime


def rolling_slope(series: pd.Series, window: int = 10) -> pd.Series:
    idx = np.arange(window, dtype=float)
    idx_mean = float(np.mean(idx))
    denom = float(np.sum((idx - idx_mean) ** 2))

    def _slope(values: np.ndarray) -> float:
        y = values.astype(float)
        y_mean = float(np.mean(y))
        num = float(np.sum((idx - idx_mean) * (y - y_mean)))
        return num / denom if denom > 0 else 0.0

    return series.rolling(window=window, min_periods=window).apply(_slope, raw=True)


def build_trade_meta_features(
    df: pd.DataFrame,
    primary_signal: pd.Series,
    momentum_window: int = 12,
    range_window: int = 50,
    trend_window: int = 20,
    breakout_velocity_lookback: int = 6,
    breakout_range_lookback: int = 20,
) -> pd.DataFrame:
    """Build per-bar trade-quality meta features using past-only information."""
    required = {
        "close",
        "adx_14",
        "atr_norm",
        "ma_fast_20",
        "ma_slow_50",
        "rsi_14",
        "bb_upper_20_2",
        "bb_lower_20_2",
        "stable_trend_regime",
        "stable_vol_regime",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for meta features: {sorted(missing)}")

    signal = primary_signal.reindex(df.index).fillna(0.0).astype(float)
    close = df["close"]
    ma_fast = df["ma_fast_20"]
    ma_slow = df["ma_slow_50"]
    bb_width = (df["bb_upper_20_2"] - df["bb_lower_20_2"]).replace(0.0, np.nan)
    atr_norm = df["atr_norm"]
    filter_type = infer_filter_type_from_regime(df).astype(str)

    change_mask = (signal != signal.shift(1)).fillna(True)
    entry_mask = change_mask & (signal != 0.0)

    time_since_last_trade = np.zeros(len(df), dtype=float)
    prev_holding_bars = np.zeros(len(df), dtype=float)
    last_trade_idx: int | None = None
    prev_hold = 0.0
    last_entry_idx: int | None = None

    for i in range(len(df)):
        if last_trade_idx is None:
            time_since_last_trade[i] = np.nan
        else:
            time_since_last_trade[i] = float(i - last_trade_idx)
        prev_holding_bars[i] = float(prev_hold) if prev_hold > 0 else np.nan

        if bool(entry_mask.iloc[i]):
            if last_entry_idx is not None:
                prev_hold = float(i - last_entry_idx)
            last_entry_idx = i
            last_trade_idx = i

    features = pd.DataFrame(index=df.index)
    features["entry_mask"] = entry_mask.astype(int)
    features["signal_side"] = np.sign(signal)
    features["filter_type"] = filter_type

    # Regime features
    features["stable_trend_regime"] = df["stable_trend_regime"].astype(str)
    features["stable_vol_regime"] = df["stable_vol_regime"].astype(str)

    # Volatility features
    features["atr_norm"] = atr_norm
    features["atr_norm_change"] = atr_norm.pct_change(5)
    atr_mean = atr_norm.rolling(trend_window, min_periods=max(5, trend_window // 2)).mean()
    features["atr_expansion"] = atr_norm / atr_mean.replace(0.0, np.nan)
    features["recent_volatility_change"] = atr_norm.diff()
    features["atr_expansion_ratio"] = atr_norm / atr_norm.rolling(
        breakout_range_lookback,
        min_periods=max(10, breakout_range_lookback // 2),
    ).mean().replace(0.0, np.nan)

    # Trend context
    features["adx_14"] = df["adx_14"]
    features["ma_fast_slope_10"] = rolling_slope(ma_fast, window=10) / close.replace(0.0, np.nan)
    features["distance_from_ma_fast"] = (close - ma_fast) / close.replace(0.0, np.nan)
    features["trend_slope"] = rolling_slope(ma_fast, window=trend_window) / close.replace(0.0, np.nan)
    features["price_vs_ma"] = (close - ma_fast) / close.replace(0.0, np.nan)

    # Mean reversion context
    features["rsi_14"] = df["rsi_14"]
    features["dist_to_bb_upper"] = (df["bb_upper_20_2"] - close) / close.replace(0.0, np.nan)
    features["dist_to_bb_lower"] = (close - df["bb_lower_20_2"]) / close.replace(0.0, np.nan)
    features["bb_zscore"] = (close - df["bb_mid_20"]) / (bb_width / 2.0)

    # Market state
    features["momentum_3"] = close.pct_change(3)
    features["momentum_n"] = close.pct_change(momentum_window)
    features["range_compression"] = atr_norm / atr_norm.rolling(20, min_periods=10).mean()

    # Trade-relative context
    ma_strength = (ma_fast - ma_slow).abs() / close.replace(0.0, np.nan)
    rsi_strength = (df["rsi_14"] - 50.0).abs() / 50.0
    features["signal_strength"] = np.where(filter_type == "trend", ma_strength, rsi_strength)

    rolling_min = close.rolling(range_window, min_periods=max(10, range_window // 2)).min()
    rolling_max = close.rolling(range_window, min_periods=max(10, range_window // 2)).max()
    range_span = (rolling_max - rolling_min).replace(0.0, np.nan)
    features["position_in_range"] = (close - rolling_min) / range_span
    features["distance_to_high"] = (rolling_max - close) / close.replace(0.0, np.nan)
    features["distance_to_low"] = (close - rolling_min) / close.replace(0.0, np.nan)

    # Breakout-strength context for stronger trade-quality discrimination.
    atr_abs = pd.to_numeric(
        df.get("atr_14", atr_norm * close.abs()),
        errors="coerce",
    )
    breakout_velocity = (
        (close - close.shift(breakout_velocity_lookback))
        / atr_abs.replace(0.0, np.nan)
    )
    features["breakout_velocity"] = breakout_velocity.abs()
    features["distance_from_range_high"] = (close - rolling_max) / close.replace(0.0, np.nan)

    above_high = close > rolling_max.shift(1)
    below_low = close < rolling_min.shift(1)
    breakout_up_streak = (
        above_high.astype(int)
        .groupby((~above_high).cumsum())
        .cumsum()
        .where(above_high, 0)
    )
    breakout_dn_streak = (
        below_low.astype(int)
        .groupby((~below_low).cumsum())
        .cumsum()
        .where(below_low, 0)
    )
    features["number_of_consecutive_breakout_bars"] = np.where(
        filter_type == "trend",
        np.maximum(breakout_up_streak, breakout_dn_streak),
        0.0,
    )

    acceleration = close.diff().diff()
    features["price_acceleration"] = acceleration / atr_norm.replace(0.0, np.nan)

    # Trade timing context
    bars_since = pd.Series(time_since_last_trade, index=df.index)
    features["time_since_last_trade"] = bars_since
    features["bars_since_last_trade"] = bars_since
    features["prev_holding_bars"] = pd.Series(prev_holding_bars, index=df.index)

    return features
