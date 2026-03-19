"""Combination logic for multi-strategy research."""

from __future__ import annotations

import pandas as pd


def confirmation_signal(*signals: pd.Series) -> pd.Series:
    """Only trade when all provided strategies agree on the same direction."""

    if not signals:
        raise ValueError("confirmation_signal requires at least one signal series.")

    aligned = pd.concat(signals, axis=1).fillna(0.0)
    sign_sum = aligned.sum(axis=1)
    agreement = aligned.abs().sum(axis=1) == len(signals)

    combined = pd.Series(0.0, index=aligned.index)
    combined = combined.mask(agreement & (sign_sum == len(signals)), 1.0)
    combined = combined.mask(agreement & (sign_sum == -len(signals)), -1.0)
    return combined


def weighted_ensemble_signal(
    signal_map: dict[str, pd.Series],
    weights: dict[str, float] | None = None,
    *,
    threshold: float = 0.05,
) -> pd.Series:
    """Average or weighted-average multiple signal streams."""

    if not signal_map:
        raise ValueError("weighted_ensemble_signal requires at least one signal.")

    weight_map = weights or {name: 1.0 for name in signal_map}
    weighted = []
    total_weight = 0.0
    for name, signal in signal_map.items():
        weight = weight_map.get(name, 0.0)
        total_weight += weight
        weighted.append(signal.fillna(0.0) * weight)

    ensemble = sum(weighted) / total_weight if total_weight else sum(weighted)
    result = pd.Series(0.0, index=ensemble.index)
    result = result.mask(ensemble > threshold, 1.0)
    result = result.mask(ensemble < -threshold, -1.0)
    return result


def specialist_sleeves(
    *,
    primary_signal: pd.Series,
    fallback_signal: pd.Series,
    activation_mask: pd.Series,
) -> pd.Series:
    """Activate a specialist strategy when a condition is met, otherwise fall back."""

    aligned = pd.concat([primary_signal, fallback_signal, activation_mask], axis=1).fillna(0.0)
    aligned.columns = ["primary", "fallback", "active"]
    return aligned["primary"].where(aligned["active"].astype(bool), aligned["fallback"]).astype(float)
