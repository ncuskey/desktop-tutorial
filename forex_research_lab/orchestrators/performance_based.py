"""Performance-based allocation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def performance_weighted_signal(
    signals: dict[str, pd.Series],
    strategy_returns: pd.DataFrame,
    lookback: int = 63,
) -> tuple[pd.Series, pd.DataFrame]:
    rolling_mean = strategy_returns.rolling(window=lookback, min_periods=max(5, lookback // 3)).mean()
    rolling_std = strategy_returns.rolling(window=lookback, min_periods=max(5, lookback // 3)).std(ddof=0)
    rolling_sharpe = (rolling_mean / rolling_std).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    positive_sharpe = rolling_sharpe.clip(lower=0.0)
    weight_sum = positive_sharpe.sum(axis=1).replace(0.0, 1.0)
    weights = positive_sharpe.div(weight_sum, axis=0)

    signal_frame = pd.concat(signals, axis=1).fillna(0.0)
    blended = (signal_frame * weights.reindex(signal_frame.index).fillna(0.0)).sum(axis=1)
    return pd.Series(np.sign(blended), index=signal_frame.index).astype(int), weights
