from __future__ import annotations

import pandas as pd


def apply_filter(signal: pd.Series, condition: pd.Series) -> pd.Series:
    """Keep signal only where condition is true; flat elsewhere."""
    aligned_condition = condition.reindex(signal.index).fillna(False).astype(bool)
    filtered = signal.copy().astype(int)
    filtered[~aligned_condition] = 0
    return filtered
