"""Simple volatility/trend-strength regime classifier."""

from __future__ import annotations

import pandas as pd


def classify_regime(
    df: pd.DataFrame,
    atr_col: str = "atr_14",
    adx_col: str = "adx_14",
    atr_lookback: int = 100,
    adx_threshold: float = 25.0,
) -> pd.Series:
    """
    Regime labels from ATR (volatility) and ADX (trend strength).
    """

    if atr_col not in df.columns or adx_col not in df.columns:
        raise ValueError(f"Dataframe must contain '{atr_col}' and '{adx_col}' columns")

    atr_threshold = df[atr_col].rolling(atr_lookback, min_periods=atr_lookback).median()
    high_vol = df[atr_col] > atr_threshold
    strong_trend = df[adx_col] >= adx_threshold

    labels = pd.Series("unknown", index=df.index, dtype="object")
    labels = labels.mask(strong_trend & high_vol, "trend_high_vol")
    labels = labels.mask(strong_trend & ~high_vol, "trend_low_vol")
    labels = labels.mask(~strong_trend & high_vol, "mean_reversion_high_vol")
    labels = labels.mask(~strong_trend & ~high_vol, "mean_reversion_low_vol")
    return labels


def regime_switch_signal(
    regimes: pd.Series,
    strategy_signals: dict[str, pd.Series],
    default_signal: float = 0.0,
) -> pd.Series:
    """
    Switch active strategy according to regime labels.
    """

    out = pd.Series(default_signal, index=regimes.index, dtype=float, name="regime_orchestrated_signal")
    for regime_label, signal in strategy_signals.items():
        mask = regimes == regime_label
        out.loc[mask] = signal.reindex(regimes.index).fillna(default_signal).loc[mask]
    return out
