"""Signal combination utilities: confirmation, ensemble, specialist sleeves."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


def _stack_signals(signals: Mapping[str, pd.Series]) -> pd.DataFrame:
    if not signals:
        raise ValueError("signals cannot be empty")
    return pd.concat(signals, axis=1).fillna(0.0)


def confirmation_signal(
    signals: Mapping[str, pd.Series],
    min_confirmations: int | None = None,
) -> pd.Series:
    """
    Trade only when enough strategies agree on direction.
    """

    stacked = _stack_signals(signals)
    needed = min_confirmations if min_confirmations is not None else len(stacked.columns)

    long_votes = (stacked > 0).sum(axis=1)
    short_votes = (stacked < 0).sum(axis=1)

    out = pd.Series(0.0, index=stacked.index, name="confirmation_signal")
    out = out.mask((long_votes >= needed) & (short_votes == 0), 1.0)
    out = out.mask((short_votes >= needed) & (long_votes == 0), -1.0)
    return out


def ensemble_signal(
    signals: Mapping[str, pd.Series],
    weights: Mapping[str, float] | None = None,
    threshold: float = 0.0,
) -> pd.Series:
    """
    Weighted ensemble of input strategy signals.
    """

    stacked = _stack_signals(signals)
    if weights is None:
        weight_vec = np.full(len(stacked.columns), 1.0 / len(stacked.columns))
    else:
        raw = np.array([weights.get(name, 0.0) for name in stacked.columns], dtype=float)
        total = raw.sum()
        if total == 0:
            raise ValueError("Weight sum cannot be zero")
        weight_vec = raw / total

    blended = stacked.to_numpy() @ weight_vec
    out = pd.Series(0.0, index=stacked.index, name="ensemble_signal")
    out = out.mask(blended > threshold, 1.0)
    out = out.mask(blended < -threshold, -1.0)
    return out


def specialist_sleeves_signal(
    signals: Mapping[str, pd.Series],
    activation_masks: Mapping[str, pd.Series],
    weights: Mapping[str, float] | None = None,
) -> pd.Series:
    """
    Activate each strategy only under its own condition mask.
    """

    if set(signals) != set(activation_masks):
        raise ValueError("signals and activation_masks must have matching strategy keys")

    active_signals = {}
    for name, signal in signals.items():
        mask = activation_masks[name].reindex(signal.index).fillna(False).astype(bool)
        active_signals[name] = signal.where(mask, 0.0)

    return ensemble_signal(active_signals, weights=weights, threshold=0.0)
