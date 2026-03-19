from __future__ import annotations

import numpy as np
import pandas as pd


def confirmation_signals(signals: dict[str, pd.Series], min_agree: int = 2) -> pd.Series:
    if len(signals) < min_agree:
        raise ValueError("Number of signals must be >= min_agree")
    stacked = pd.concat(signals, axis=1).fillna(0)
    longs = (stacked > 0).sum(axis=1)
    shorts = (stacked < 0).sum(axis=1)
    out = pd.Series(0, index=stacked.index, dtype=int)
    out[longs >= min_agree] = 1
    out[shorts >= min_agree] = -1
    return out


def weighted_ensemble_signals(
    signals: dict[str, pd.Series],
    weights: dict[str, float] | None = None,
    threshold: float = 0.0,
) -> pd.Series:
    if not signals:
        raise ValueError("signals cannot be empty")
    if weights is None:
        equal = 1.0 / len(signals)
        weights = {k: equal for k in signals}

    stacked = pd.concat(signals, axis=1).fillna(0)
    weight_vec = np.array([weights.get(c, 0.0) for c in stacked.columns])
    combined = stacked.to_numpy() @ weight_vec
    out = pd.Series(0, index=stacked.index, dtype=int)
    out[combined > threshold] = 1
    out[combined < -threshold] = -1
    return out


def specialist_sleeve(signal: pd.Series, activation_mask: pd.Series) -> pd.Series:
    aligned_mask = activation_mask.reindex(signal.index).fillna(False).astype(bool)
    out = signal.copy().astype(int)
    out[~aligned_mask] = 0
    return out
