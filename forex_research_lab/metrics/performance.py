"""Backtest metric calculations."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def periods_per_year(timeframe: str) -> int:
    mapping = {
        "H1": 24 * 252,
        "H4": 6 * 252,
        "D1": 252,
    }
    return mapping.get(timeframe.upper(), 252)


def _max_drawdown(equity: pd.Series) -> float:
    drawdown = equity / equity.cummax() - 1
    return float(drawdown.min()) if not drawdown.empty else 0.0


def compute_metrics(
    *,
    returns: pd.Series,
    trades: pd.DataFrame,
    timeframe: str,
    initial_capital: float,
    equity: pd.Series | None = None,
) -> dict[str, float]:
    """Compute a standard metric set for a return stream."""

    clean_returns = returns.fillna(0.0).astype(float)
    if equity is None:
        equity = initial_capital * (1 + clean_returns).cumprod()

    annualization = periods_per_year(timeframe)
    mean_return = clean_returns.mean()
    volatility = clean_returns.std(ddof=0)
    downside = clean_returns.where(clean_returns < 0, 0.0)
    downside_std = downside.std(ddof=0)

    periods = max(len(clean_returns), 1)
    total_return = equity.iloc[-1] / initial_capital - 1 if len(equity) else 0.0
    cagr = (equity.iloc[-1] / initial_capital) ** (annualization / periods) - 1 if len(equity) else 0.0
    sharpe = math.sqrt(annualization) * mean_return / volatility if volatility > 0 else 0.0
    sortino = math.sqrt(annualization) * mean_return / downside_std if downside_std > 0 else 0.0
    max_drawdown = _max_drawdown(equity)

    if trades.empty or "trade_return" not in trades.columns:
        profit_factor = 0.0
        win_rate = 0.0
        expectancy = 0.0
        trade_count = 0.0
    else:
        gains = trades.loc[trades["trade_return"] > 0, "trade_return"].sum()
        losses = trades.loc[trades["trade_return"] < 0, "trade_return"].sum()
        profit_factor = float(gains / abs(losses)) if losses < 0 else float("inf") if gains > 0 else 0.0
        win_rate = float((trades["trade_return"] > 0).mean())
        expectancy = float(trades["trade_return"].mean())
        trade_count = float(len(trades))

    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(max_drawdown),
        "profit_factor": float(profit_factor),
        "win_rate": float(win_rate),
        "expectancy": float(expectancy),
        "trade_count": float(trade_count),
        "volatility": float(volatility * np.sqrt(annualization)),
    }
