"""Simple regime classification and strategy switching."""

from __future__ import annotations

import pandas as pd


def classify_regime(
    df: pd.DataFrame,
    atr_column: str = "atr_14",
    adx_column: str = "adx_14",
    high_adx_threshold: float = 25.0,
    high_vol_quantile: float = 0.7,
) -> pd.Series:
    high_vol_cutoff = df[atr_column].quantile(high_vol_quantile)
    high_trend = df[adx_column] >= high_adx_threshold
    high_vol = df[atr_column] >= high_vol_cutoff

    regime = pd.Series("range_low_vol", index=df.index, dtype="object")
    regime = regime.mask(high_trend & ~high_vol, "trend_low_vol")
    regime = regime.mask(~high_trend & high_vol, "range_high_vol")
    regime = regime.mask(high_trend & high_vol, "trend_high_vol")
    return regime


def regime_switch_signal(
    regimes: pd.Series,
    mapping: dict[str, pd.Series],
    default_signal: pd.Series | None = None,
) -> pd.Series:
    output = default_signal.copy() if default_signal is not None else pd.Series(0, index=regimes.index)
    for regime_name, signal in mapping.items():
        output = output.where(regimes != regime_name, signal)
    return output.fillna(0).astype(int)
