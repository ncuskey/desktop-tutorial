"""Performance metric calculations."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import pandas as pd


def cagr(equity_curve: pd.Series, periods_per_year: int) -> float:
    if equity_curve.empty:
        return 0.0
    start = float(equity_curve.iloc[0])
    end = float(equity_curve.iloc[-1])
    periods = len(equity_curve)
    if start <= 0 or periods <= 1:
        return 0.0
    years = periods / periods_per_year
    if years <= 0:
        return 0.0
    return (end / start) ** (1 / years) - 1


def sharpe_ratio(returns: pd.Series, periods_per_year: int) -> float:
    std = float(returns.std(ddof=0))
    if std == 0 or np.isnan(std):
        return 0.0
    return float(np.sqrt(periods_per_year) * returns.mean() / std)


def sortino_ratio(returns: pd.Series, periods_per_year: int) -> float:
    downside = returns[returns < 0]
    downside_std = float(downside.std(ddof=0))
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0
    return float(np.sqrt(periods_per_year) * returns.mean() / downside_std)


def max_drawdown(drawdown_curve: pd.Series) -> float:
    if drawdown_curve.empty:
        return 0.0
    return float(drawdown_curve.min())


def profit_factor(
    trades: pd.DataFrame | None = None,
    returns: pd.Series | None = None,
) -> float:
    if trades is not None and not trades.empty and "net_return" in trades.columns:
        wins = trades.loc[trades["net_return"] > 0, "net_return"].sum()
        losses = -trades.loc[trades["net_return"] < 0, "net_return"].sum()
    elif returns is not None:
        wins = returns[returns > 0].sum()
        losses = -returns[returns < 0].sum()
    else:
        return 0.0

    wins = float(wins)
    losses = float(losses)
    if losses == 0:
        return math.inf if wins > 0 else 0.0
    return wins / losses


def win_rate(trades: pd.DataFrame) -> float:
    if trades.empty or "net_return" not in trades.columns:
        return 0.0
    return float((trades["net_return"] > 0).mean())


def expectancy(trades: pd.DataFrame) -> float:
    if trades.empty or "net_return" not in trades.columns:
        return 0.0
    return float(trades["net_return"].mean())


def compute_metrics(
    returns: pd.Series,
    equity_curve: pd.Series,
    drawdown_curve: pd.Series,
    trades: pd.DataFrame,
    periods_per_year: int,
) -> Mapping[str, float]:
    return {
        "cagr": cagr(equity_curve, periods_per_year),
        "sharpe": sharpe_ratio(returns, periods_per_year),
        "sortino": sortino_ratio(returns, periods_per_year),
        "max_drawdown": max_drawdown(drawdown_curve),
        "profit_factor": profit_factor(trades=trades),
        "win_rate": win_rate(trades),
        "expectancy": expectancy(trades),
        "trade_count": float(len(trades)),
    }
