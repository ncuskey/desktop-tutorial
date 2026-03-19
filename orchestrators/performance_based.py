from __future__ import annotations

import numpy as np
import pandas as pd


def _rolling_sharpe(returns: pd.Series, lookback: int) -> pd.Series:
    mean_r = returns.rolling(lookback, min_periods=lookback).mean()
    std_r = returns.rolling(lookback, min_periods=lookback).std(ddof=0)
    sharpe = mean_r / std_r.replace(0.0, np.nan)
    return sharpe.fillna(0.0)


def performance_weighted_signal(
    signal_map: dict[str, pd.Series],
    return_map: dict[str, pd.Series],
    lookback: int = 100,
    threshold: float = 0.0,
) -> pd.Series:
    if not signal_map or not return_map:
        raise ValueError("signal_map and return_map must be non-empty")

    names = list(signal_map.keys())
    index = signal_map[names[0]].index
    sharpe_df = pd.DataFrame(
        {
            name: _rolling_sharpe(return_map[name].reindex(index).fillna(0), lookback)
            for name in names
        },
        index=index,
    )
    positive = sharpe_df.clip(lower=0.0)
    weights = positive.div(positive.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)

    signal_df = pd.DataFrame(
        {name: signal_map[name].reindex(index).fillna(0) for name in names}, index=index
    )
    combined = (signal_df * weights).sum(axis=1)

    out = pd.Series(0, index=index, dtype=int)
    out[combined > threshold] = 1
    out[combined < -threshold] = -1
    return out
