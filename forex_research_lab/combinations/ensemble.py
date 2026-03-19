"""Signal ensembles and weighted combinations."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


def average_signal(signals: Mapping[str, pd.Series], threshold: float = 0.0, discrete: bool = True) -> pd.Series:
    """Average aligned strategy signals."""
    if not signals:
        raise ValueError("At least one signal series is required")

    frame = pd.concat(signals, axis=1).fillna(0.0)
    averaged = frame.mean(axis=1).clip(-1.0, 1.0)
    if not discrete:
        return averaged

    result = pd.Series(0.0, index=averaged.index, dtype=float)
    result.loc[averaged >= threshold] = 1.0
    result.loc[averaged <= -threshold] = -1.0
    return result


def weighted_signal(
    signals: Mapping[str, pd.Series],
    weights: Mapping[str, float] | pd.DataFrame,
    threshold: float = 0.0,
    discrete: bool = True,
) -> pd.Series:
    """Combine strategy signals using static or time-varying weights."""
    if not signals:
        raise ValueError("At least one signal series is required")

    frame = pd.concat(signals, axis=1).fillna(0.0)

    if isinstance(weights, pd.DataFrame):
        aligned_weights = weights.reindex(frame.index).fillna(0.0)[list(frame.columns)]
        combined = (frame * aligned_weights).sum(axis=1)
        normalizer = aligned_weights.abs().sum(axis=1).replace(0.0, 1.0)
        combined = combined.div(normalizer)
    else:
        weight_series = pd.Series(weights, dtype=float).reindex(frame.columns).fillna(0.0)
        combined = frame.mul(weight_series, axis=1).sum(axis=1)
        scale = max(weight_series.abs().sum(), 1.0)
        combined = combined / scale

    combined = combined.clip(-1.0, 1.0)
    if not discrete:
        return combined

    result = pd.Series(0.0, index=combined.index, dtype=float)
    result.loc[combined >= threshold] = 1.0
    result.loc[combined <= -threshold] = -1.0
    return result
