from __future__ import annotations

import math
from typing import Dict

import numpy as np
import pandas as pd


def _annual_factor(timeframe: str) -> int:
    mapping = {
        "H1": 24 * 252,
        "H4": 6 * 252,
        "D1": 252,
    }
    return mapping.get(timeframe.upper(), 252)


def compute_metrics(
    equity_curve: pd.Series,
    returns: pd.Series,
    trades: pd.DataFrame,
    timeframe: str = "H1",
) -> Dict[str, float]:
    periods = _annual_factor(timeframe)
    clean_returns = returns.dropna()

    if len(equity_curve) < 2:
        return {
            "CAGR": 0.0,
            "Sharpe": 0.0,
            "Sortino": 0.0,
            "MaxDrawdown": 0.0,
            "ProfitFactor": 0.0,
            "WinRate": 0.0,
            "Expectancy": 0.0,
            "TradeCount": 0.0,
        }

    years = len(clean_returns) / periods if periods > 0 else 0
    total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 and (1 + total_return) > 0 else -1.0

    mean_ret = clean_returns.mean()
    std_ret = clean_returns.std(ddof=0)
    sharpe = math.sqrt(periods) * (mean_ret / std_ret) if std_ret > 0 else 0.0

    downside = clean_returns[clean_returns < 0]
    downside_std = downside.std(ddof=0)
    sortino = math.sqrt(periods) * (mean_ret / downside_std) if downside_std and downside_std > 0 else 0.0

    drawdown = equity_curve / equity_curve.cummax() - 1
    max_dd = float(drawdown.min())

    if trades.empty:
        profit_factor = 0.0
        win_rate = 0.0
        expectancy = 0.0
        trade_count = 0.0
    else:
        trade_returns = trades["trade_return"].astype(float)
        wins = trade_returns[trade_returns > 0]
        losses = trade_returns[trade_returns < 0]
        gross_profit = wins.sum()
        gross_loss = losses.sum()
        profit_factor = float(gross_profit / abs(gross_loss)) if gross_loss < 0 else float(np.inf)
        win_rate = float((trade_returns > 0).mean())
        expectancy = float(trade_returns.mean())
        trade_count = float(len(trades))

    return {
        "CAGR": float(cagr),
        "Sharpe": float(sharpe),
        "Sortino": float(sortino),
        "MaxDrawdown": max_dd,
        "ProfitFactor": profit_factor,
        "WinRate": win_rate,
        "Expectancy": expectancy,
        "TradeCount": trade_count,
    }
