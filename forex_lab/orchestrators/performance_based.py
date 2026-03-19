from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def performance_based_allocation(
    strategy_returns: Dict[str, pd.Series],
    lookback: int = 100,
    min_weight: float = 0.0,
) -> pd.DataFrame:
    """
    Allocate capital by recent rolling Sharpe estimates.
    """
    if not strategy_returns:
        raise ValueError("strategy_returns cannot be empty")

    aligned = pd.concat(strategy_returns, axis=1).fillna(0)
    rolling_mean = aligned.rolling(lookback).mean()
    rolling_std = aligned.rolling(lookback).std(ddof=0).replace(0, np.nan)
    rolling_sharpe = (rolling_mean / rolling_std).clip(lower=0).fillna(0)

    weights = rolling_sharpe.div(rolling_sharpe.sum(axis=1), axis=0).fillna(0)
    if min_weight > 0:
        weights = weights.clip(lower=min_weight)
        weights = weights.div(weights.sum(axis=1), axis=0).fillna(0)
    return weights
