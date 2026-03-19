"""Rule-based orchestration."""

from __future__ import annotations

import pandas as pd


def adx_rule_based_signal(
    trend_signal: pd.Series,
    mean_reversion_signal: pd.Series,
    adx_series: pd.Series,
    adx_threshold: float = 25.0,
) -> pd.Series:
    """Switch between trend and mean reversion based on ADX."""
    aligned_mean_reversion = mean_reversion_signal.reindex(trend_signal.index).fillna(0.0)
    aligned_adx = adx_series.reindex(trend_signal.index)

    signal = pd.Series(0.0, index=trend_signal.index, dtype=float)
    trending = aligned_adx >= adx_threshold
    signal.loc[trending.fillna(False)] = trend_signal.loc[trending.fillna(False)]
    signal.loc[~trending.fillna(False)] = aligned_mean_reversion.loc[~trending.fillna(False)]
    return signal.clip(-1.0, 1.0)
