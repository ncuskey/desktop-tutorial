"""Combination methods for multiple strategy signals."""

from __future__ import annotations

import numpy as np
import pandas as pd


def confirmation_signal(signals: dict[str, pd.Series]) -> pd.Series:
    frame = pd.concat(signals, axis=1).fillna(0.0)
    same_direction = frame.nunique(axis=1) == 1
    agreed = frame.iloc[:, 0].where(same_direction, 0.0)
    return agreed.astype(int)


def ensemble_signal(signals: dict[str, pd.Series], weights: dict[str, float] | None = None) -> pd.Series:
    frame = pd.concat(signals, axis=1).fillna(0.0)
    if weights is None:
        averaged = frame.mean(axis=1)
    else:
        weight_series = pd.Series(weights, dtype=float)
        averaged = frame.mul(weight_series, axis=1).sum(axis=1) / weight_series.sum()
    return pd.Series(np.sign(averaged), index=frame.index).astype(int)


def specialist_sleeves_signal(
    signals: dict[str, pd.Series],
    activation_masks: dict[str, pd.Series],
    weights: dict[str, float] | None = None,
) -> pd.Series:
    weighted_components = []
    weights = weights or {name: 1.0 for name in signals}

    for name, signal in signals.items():
        mask = activation_masks.get(name, pd.Series(True, index=signal.index))
        weighted_components.append(signal.where(mask, 0.0) * weights.get(name, 1.0))

    combined = pd.concat(weighted_components, axis=1).sum(axis=1)
    return pd.Series(np.sign(combined), index=combined.index).astype(int)
