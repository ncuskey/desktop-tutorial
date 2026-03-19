"""Simple ATR/ADX regime classification."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from forex_research_lab.data.indicators import adx, atr


def classify_regime(
    dataframe: pd.DataFrame,
    atr_window: int = 14,
    adx_window: int = 14,
    adx_threshold: float = 25.0,
    volatility_quantile: float = 0.5,
) -> pd.Series:
    """Classify each bar by volatility and trend strength."""
    atr_series = dataframe.get(f"atr_{atr_window}") if f"atr_{atr_window}" in dataframe.columns else atr(dataframe, window=atr_window)
    adx_series = dataframe.get(f"adx_{adx_window}") if f"adx_{adx_window}" in dataframe.columns else adx(dataframe, window=adx_window)

    volatility_cutoff = atr_series.quantile(volatility_quantile)
    high_volatility = atr_series >= volatility_cutoff
    trending = adx_series >= adx_threshold

    regime = pd.Series("range_low_vol", index=dataframe.index, dtype=object)
    regime.loc[high_volatility & trending] = "trend_high_vol"
    regime.loc[~high_volatility & trending] = "trend_low_vol"
    regime.loc[high_volatility & ~trending] = "range_high_vol"
    return regime


def regime_based_signal(regimes: pd.Series, strategy_map: Mapping[str, pd.Series]) -> pd.Series:
    """Switch among pre-computed strategy signals based on a regime label."""
    if not strategy_map:
        raise ValueError("strategy_map must contain at least one regime mapping")

    reference_index = next(iter(strategy_map.values())).index
    output = pd.Series(0.0, index=reference_index, dtype=float)
    aligned_regimes = regimes.reindex(reference_index)

    for regime_name, signal in strategy_map.items():
        mask = aligned_regimes == regime_name
        output.loc[mask.fillna(False)] = signal.reindex(reference_index).fillna(0.0).loc[mask.fillna(False)]
    return output.clip(-1.0, 1.0)
