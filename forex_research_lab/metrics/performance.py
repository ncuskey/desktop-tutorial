"""Performance and trade statistics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def infer_periods_per_year(index: pd.Index | pd.Series) -> float:
    """Infer annualization from a datetime index, falling back to 252 periods."""
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return 252.0

    median_delta = index.to_series().diff().dropna().median()
    if pd.isna(median_delta) or median_delta <= pd.Timedelta(0):
        return 252.0
    return float(pd.Timedelta(days=365.25) / median_delta)


def annualized_sharpe(returns: pd.Series, periods_per_year: float | None = None) -> float:
    """Compute annualized Sharpe ratio."""
    clean_returns = returns.dropna()
    if clean_returns.empty:
        return 0.0

    periods = periods_per_year or infer_periods_per_year(clean_returns.index)
    volatility = clean_returns.std(ddof=0)
    if volatility == 0.0:
        return 0.0
    return float(math.sqrt(periods) * clean_returns.mean() / volatility)


def annualized_sortino(returns: pd.Series, periods_per_year: float | None = None) -> float:
    """Compute annualized Sortino ratio."""
    clean_returns = returns.dropna()
    if clean_returns.empty:
        return 0.0

    periods = periods_per_year or infer_periods_per_year(clean_returns.index)
    downside = clean_returns[clean_returns < 0.0]
    downside_volatility = downside.std(ddof=0)
    if pd.isna(downside_volatility) or downside_volatility == 0.0:
        return 0.0
    return float(math.sqrt(periods) * clean_returns.mean() / downside_volatility)


def max_drawdown(equity_curve: pd.Series) -> float:
    """Return the absolute max drawdown."""
    if equity_curve.empty:
        return 0.0

    running_peak = equity_curve.cummax().replace(0.0, np.nan)
    drawdown = equity_curve.div(running_peak).fillna(1.0) - 1.0
    return float(abs(drawdown.min()))


def compute_performance_metrics(
    returns: pd.Series,
    equity_curve: pd.Series,
    trades: pd.DataFrame | None = None,
    periods_per_year: float | None = None,
) -> dict[str, float]:
    """Compute the platform's core evaluation metrics."""
    clean_returns = returns.dropna()
    periods = periods_per_year or infer_periods_per_year(clean_returns.index if not clean_returns.empty else returns.index)

    if equity_curve.empty:
        return {
            "Total Return": 0.0,
            "CAGR": 0.0,
            "Sharpe": 0.0,
            "Sortino": 0.0,
            "Max Drawdown": 0.0,
            "Profit Factor": 0.0,
            "Win Rate": 0.0,
            "Expectancy": 0.0,
            "Trade Count": 0.0,
        }

    total_return = float((equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1.0) if len(equity_curve) > 1 else 0.0
    if isinstance(equity_curve.index, pd.DatetimeIndex) and len(equity_curve.index) > 1:
        years = max((equity_curve.index[-1] - equity_curve.index[0]).total_seconds() / (365.25 * 24 * 3600), 1e-9)
    else:
        years = max(len(clean_returns) / periods if periods else 0.0, 1e-9)
    cagr = float((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1.0 / years) - 1.0) if len(equity_curve) > 1 else 0.0

    trade_returns = pd.Series(dtype=float)
    if trades is not None and not trades.empty and "net_return" in trades.columns:
        trade_returns = trades["net_return"].astype(float)

    winners = trade_returns[trade_returns > 0.0]
    losers = trade_returns[trade_returns < 0.0]
    if losers.empty:
        profit_factor = float("inf") if not winners.empty else 0.0
    else:
        profit_factor = float(winners.sum() / abs(losers.sum()))

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Sharpe": annualized_sharpe(clean_returns, periods_per_year=periods),
        "Sortino": annualized_sortino(clean_returns, periods_per_year=periods),
        "Max Drawdown": max_drawdown(equity_curve),
        "Profit Factor": profit_factor,
        "Win Rate": float((trade_returns > 0.0).mean()) if not trade_returns.empty else 0.0,
        "Expectancy": float(trade_returns.mean()) if not trade_returns.empty else 0.0,
        "Trade Count": float(len(trade_returns)),
    }
