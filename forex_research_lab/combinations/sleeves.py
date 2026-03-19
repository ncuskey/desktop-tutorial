"""Conditional activation for specialist sleeves."""

from __future__ import annotations

import pandas as pd


def conditional_activate(signal: pd.Series, condition: pd.Series) -> pd.Series:
    """Activate a signal only when a condition is met."""
    aligned_condition = condition.reindex(signal.index).fillna(False).astype(bool)
    return signal.where(aligned_condition, 0.0)


def adx_specialist_sleeves(
    trend_signal: pd.Series,
    mean_reversion_signal: pd.Series,
    adx_series: pd.Series,
    threshold: float = 25.0,
) -> pd.Series:
    """Use trend sleeves in strong-trend environments and mean reversion otherwise."""
    aligned_adx = adx_series.reindex(trend_signal.index)
    aligned_mean_reversion = mean_reversion_signal.reindex(trend_signal.index).fillna(0.0)

    output = pd.Series(0.0, index=trend_signal.index, dtype=float)
    trending = aligned_adx >= threshold
    output.loc[trending.fillna(False)] = trend_signal.loc[trending.fillna(False)]
    output.loc[~trending.fillna(False)] = aligned_mean_reversion.loc[~trending.fillna(False)]
    return output.clip(-1.0, 1.0)
