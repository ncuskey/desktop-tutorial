from __future__ import annotations

import pandas as pd


def build_rebalance_mask(index: pd.DatetimeIndex, rule: str = "1D") -> pd.Series:
    """
    Build boolean mask where True means portfolio should rebalance on that bar.
    """
    if len(index) == 0:
        return pd.Series(dtype=bool)
    idx = pd.DatetimeIndex(index)
    floor = idx.floor(rule)
    change = floor != floor.shift(1)
    change[0] = True
    return pd.Series(change, index=idx, dtype=bool)


def apply_rebalance_schedule(
    target_weights: pd.DataFrame,
    rebalance_mask: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Hold last rebalance weights between rebalance timestamps.
    """
    if target_weights.empty:
        return target_weights.copy()
    w = target_weights.copy().astype(float)
    if rebalance_mask is None:
        return w
    mask = rebalance_mask.reindex(w.index).fillna(False).astype(bool)
    if not mask.any():
        return pd.DataFrame(0.0, index=w.index, columns=w.columns)

    out = pd.DataFrame(0.0, index=w.index, columns=w.columns)
    current = pd.Series(0.0, index=w.columns)
    for ts in w.index:
        if bool(mask.loc[ts]):
            current = w.loc[ts].astype(float)
        out.loc[ts] = current
    return out
