"""Execution engine — convert signals into PnL with realistic costs."""

from __future__ import annotations

import numpy as np
import pandas as pd


def execute_signals(
    df: pd.DataFrame,
    signals: pd.Series,
    initial_capital: float = 100_000.0,
    position_size: float = 1.0,
) -> pd.DataFrame:
    """Simulate execution of a position series against OHLCV data.

    Parameters
    ----------
    df : DataFrame
        Must contain ``close`` and cost-model columns (``half_spread``,
        ``slippage``, ``commission``). Use ``data.attach_cost_model`` first.
    signals : Series
        Position signal in {-1, 0, +1}, same index as *df*.
    initial_capital : float
        Starting equity.
    position_size : float
        Fraction of capital to commit per trade (1.0 = fully invested).

    Returns
    -------
    DataFrame with columns: position, returns, cost, net_returns, equity, drawdown.
    """
    signals = signals.reindex(df.index).fillna(0).astype(int)

    close = df["close"].values
    half_spread = df.get("half_spread", pd.Series(0.0, index=df.index)).values
    slippage = df.get("slippage", pd.Series(0.0, index=df.index)).values
    commission = df.get("commission", pd.Series(0.0, index=df.index)).values

    pos = signals.values.astype(float)
    n = len(close)

    raw_ret = np.zeros(n)
    cost = np.zeros(n)

    for i in range(1, n):
        raw_ret[i] = pos[i - 1] * (close[i] - close[i - 1]) / close[i - 1]

        if pos[i] != pos[i - 1]:
            cost[i] = (half_spread[i] + slippage[i] + commission[i]) / close[i]

    net_ret = raw_ret - cost

    equity = initial_capital * np.cumprod(1 + net_ret * position_size)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak

    result = pd.DataFrame(
        {
            "position": pos,
            "returns": raw_ret,
            "cost": cost,
            "net_returns": net_ret,
            "equity": equity,
            "drawdown": drawdown,
        },
        index=df.index,
    )
    return result
