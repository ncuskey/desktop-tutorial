from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def confirmation_signals(signals: Dict[str, pd.Series], min_agreement: int | None = None) -> pd.Series:
    if not signals:
        raise ValueError("signals cannot be empty")
    aligned = pd.concat(signals.values(), axis=1).fillna(0)
    if min_agreement is None:
        min_agreement = len(signals)
    long_votes = (aligned > 0).sum(axis=1)
    short_votes = (aligned < 0).sum(axis=1)
    out = pd.Series(0, index=aligned.index, dtype=float)
    out[long_votes >= min_agreement] = 1
    out[short_votes >= min_agreement] = -1
    return out.astype(int)


def ensemble_average_signals(signals: Dict[str, pd.Series], threshold: float = 0.0) -> pd.Series:
    if not signals:
        raise ValueError("signals cannot be empty")
    aligned = pd.concat(signals.values(), axis=1).fillna(0)
    avg = aligned.mean(axis=1)
    out = pd.Series(0, index=aligned.index, dtype=float)
    out[avg > threshold] = 1
    out[avg < -threshold] = -1
    return out.astype(int)


def ensemble_weighted_signals(signals: Dict[str, pd.Series], weights: Dict[str, float]) -> pd.Series:
    if not signals:
        raise ValueError("signals cannot be empty")
    weighted_sum = None
    total_abs_weight = 0.0
    for name, signal in signals.items():
        w = float(weights.get(name, 0.0))
        total_abs_weight += abs(w)
        component = signal.fillna(0) * w
        weighted_sum = component if weighted_sum is None else weighted_sum.add(component, fill_value=0)
    if weighted_sum is None or total_abs_weight == 0:
        return pd.Series(0, index=next(iter(signals.values())).index, dtype=int)
    normed = weighted_sum / total_abs_weight
    return pd.Series(np.sign(normed), index=normed.index).astype(int)


def specialist_sleeves(base_signals: Dict[str, pd.Series], activation_mask: pd.Series) -> pd.Series:
    if not base_signals:
        raise ValueError("base_signals cannot be empty")
    avg = pd.concat(base_signals.values(), axis=1).fillna(0).mean(axis=1)
    active = activation_mask.reindex(avg.index).fillna(False).astype(bool)
    out = pd.Series(0, index=avg.index, dtype=float)
    out[active] = np.sign(avg[active])
    return out.astype(int)
