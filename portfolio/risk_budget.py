from __future__ import annotations

import numpy as np
import pandas as pd


def apply_weight_constraints(
    weights: pd.Series,
    max_symbol_exposure: float = 0.35,
    gross_exposure_cap: float = 1.0,
) -> pd.Series:
    """
    Enforce per-symbol and gross exposure constraints.
    """
    w = weights.astype(float).copy()
    max_symbol_exposure = float(max(max_symbol_exposure, 0.0))
    gross_exposure_cap = float(max(gross_exposure_cap, 0.0))

    w = w.clip(lower=-max_symbol_exposure, upper=max_symbol_exposure)
    gross = float(w.abs().sum())
    if gross > gross_exposure_cap and gross > 1e-12:
        w = w * (gross_exposure_cap / gross)
    return w


def normalize_positive_scores(scores: pd.Series) -> pd.Series:
    s = scores.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    s = s.clip(lower=0.0)
    total = float(s.sum())
    if total <= 1e-12:
        if len(s) == 0:
            return s
        return pd.Series(1.0 / len(s), index=s.index, dtype=float)
    return s / total
