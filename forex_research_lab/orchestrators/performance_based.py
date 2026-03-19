"""Performance-based strategy allocation orchestration."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_sharpe_allocation(
    returns_by_strategy: dict[str, pd.Series],
    lookback: int = 120,
    rebalance_bars: int = 24,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """
    Allocate more capital to strategies with higher trailing Sharpe.
    Uses shifted Sharpe estimates to avoid lookahead.
    """

    if not returns_by_strategy:
        raise ValueError("returns_by_strategy cannot be empty")

    returns_df = pd.concat(returns_by_strategy, axis=1).fillna(0.0)
    rolling_mean = returns_df.rolling(lookback, min_periods=lookback).mean()
    rolling_std = returns_df.rolling(lookback, min_periods=lookback).std(ddof=0)
    rolling_sharpe = (rolling_mean / rolling_std.replace(0, np.nan)) * np.sqrt(periods_per_year)
    rolling_sharpe = rolling_sharpe.shift(1)

    n_strats = len(returns_df.columns)
    default_weights = np.full(n_strats, 1.0 / n_strats)
    current_weights = default_weights.copy()

    weights_df = pd.DataFrame(index=returns_df.index, columns=returns_df.columns, dtype=float)
    for i in range(len(returns_df)):
        should_rebalance = i >= lookback and ((i - lookback) % rebalance_bars == 0)
        if should_rebalance:
            scores = rolling_sharpe.iloc[i].fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
            if scores.sum() > 0:
                current_weights = scores / scores.sum()
            else:
                current_weights = default_weights.copy()
        weights_df.iloc[i] = current_weights

    return weights_df


def weighted_signal_from_allocations(
    signals_by_strategy: dict[str, pd.Series],
    weights_df: pd.DataFrame,
    threshold: float = 0.0,
) -> pd.Series:
    """
    Build orchestrated signals using dynamic weights.
    """

    if not signals_by_strategy:
        raise ValueError("signals_by_strategy cannot be empty")

    signal_df = pd.concat(signals_by_strategy, axis=1).fillna(0.0)
    aligned_weights = weights_df.reindex(signal_df.index).ffill().fillna(0.0)
    aligned_weights = aligned_weights.reindex(columns=signal_df.columns).fillna(0.0)

    blended = (signal_df * aligned_weights).sum(axis=1)
    out = pd.Series(0.0, index=signal_df.index, name="performance_based_orchestrated_signal")
    out = out.mask(blended > threshold, 1.0)
    out = out.mask(blended < -threshold, -1.0)
    return out
