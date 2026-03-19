"""Performance and trade metrics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def bars_per_year(timeframe: str) -> int:
    mapping = {
        "H1": 24 * 252,
        "H4": 6 * 252,
        "D1": 252,
    }
    return mapping.get(timeframe.upper(), 252)


def compute_metrics(
    result_frame: pd.DataFrame,
    trades: pd.DataFrame,
    timeframe: str,
) -> dict[str, float]:
    returns = result_frame["net_return"].fillna(0.0)
    periods = bars_per_year(timeframe)
    avg_return = returns.mean()
    std_return = returns.std(ddof=0)
    downside_std = returns.where(returns < 0, 0.0).std(ddof=0)

    total_bars = max(len(returns), 1)
    total_return = float((1.0 + returns).prod() - 1.0)
    cagr = float((1.0 + total_return) ** (periods / total_bars) - 1.0) if total_bars > 0 and (1.0 + total_return) > 0 else float("nan")
    sharpe = float(np.sqrt(periods) * avg_return / std_return) if std_return > 0 else 0.0
    sortino = float(np.sqrt(periods) * avg_return / downside_std) if downside_std > 0 else 0.0
    max_drawdown = float(result_frame["drawdown"].min()) if not result_frame.empty else 0.0

    gross_profit = trades.loc[trades["pnl"] > 0, "pnl"].sum() if not trades.empty else 0.0
    gross_loss = -trades.loc[trades["pnl"] < 0, "pnl"].sum() if not trades.empty else 0.0
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else math.inf
    win_rate = float((trades["pnl"] > 0).mean()) if not trades.empty else 0.0
    expectancy = float(trades["pnl"].mean()) if not trades.empty else 0.0

    return {
        "total_return": float(total_return),
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "expectancy": expectancy,
        "trade_count": float(len(trades)),
    }
