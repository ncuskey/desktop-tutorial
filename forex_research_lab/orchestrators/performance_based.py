"""Performance-sensitive strategy allocation."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from forex_research_lab.combinations.ensemble import weighted_signal


def rolling_sharpe_weights(
    strategy_returns: Mapping[str, pd.Series],
    lookback: int = 63,
    top_n: int = 1,
) -> pd.DataFrame:
    """Allocate capital to strategies with the strongest recent Sharpe-like score."""
    if not strategy_returns:
        raise ValueError("At least one strategy return series is required")

    frame = pd.concat(strategy_returns, axis=1).fillna(0.0)
    rolling_mean = frame.rolling(lookback, min_periods=max(5, lookback // 3)).mean()
    rolling_std = frame.rolling(lookback, min_periods=max(5, lookback // 3)).std(ddof=0).replace(0.0, pd.NA)
    scores = rolling_mean.div(rolling_std).fillna(0.0)

    weights = pd.DataFrame(0.0, index=frame.index, columns=frame.columns)
    for timestamp, row in scores.iterrows():
        selected = row.sort_values(ascending=False).head(top_n)
        positive = selected[selected > 0.0]
        if positive.empty and not selected.empty:
            weights.loc[timestamp, selected.index] = 1.0 / len(selected)
        elif not positive.empty:
            weights.loc[timestamp, positive.index] = positive / positive.sum()
    return weights


def performance_based_signal(
    strategy_signals: Mapping[str, pd.Series],
    strategy_returns: Mapping[str, pd.Series],
    lookback: int = 63,
    top_n: int = 1,
) -> pd.Series:
    """Blend strategy signals using recent realized performance."""
    weights = rolling_sharpe_weights(strategy_returns=strategy_returns, lookback=lookback, top_n=top_n)
    return weighted_signal(strategy_signals, weights=weights, threshold=0.0, discrete=False).clip(-1.0, 1.0)
