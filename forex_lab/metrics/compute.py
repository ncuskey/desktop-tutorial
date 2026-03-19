"""Performance metrics computation."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Metrics:
    """Performance metrics container."""

    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float
    profit_factor: float
    win_rate: float
    expectancy: float
    trade_count: int
    total_return: float
    volatility: float


def compute_metrics(
    equity_curve: pd.Series,
    trades: list,
    risk_free_rate: float = 0.0,
    periods_per_year: Optional[float] = None,
) -> Metrics:
    """
    Compute standard performance metrics.

    Args:
        equity_curve: Series of equity values
        trades: List of Trade objects
        risk_free_rate: Annual risk-free rate for Sharpe
        periods_per_year: e.g. 252 for daily, 8760 for hourly
    """
    returns = equity_curve.pct_change().dropna()
    if len(returns) == 0:
        return _empty_metrics()

    # Infer periods per year from index frequency
    if periods_per_year is None:
        freq = pd.infer_freq(equity_curve.index)
        if freq:
            try:
                period = pd.Timedelta(freq)
                periods_per_year = 365.25 * 24 * 60 * 60 / period.total_seconds()
            except Exception:
                periods_per_year = 252.0
        else:
            periods_per_year = 252.0

    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1.0
    n_periods = len(equity_curve)
    years = n_periods / periods_per_year
    if years > 0 and equity_curve.iloc[0] > 0 and equity_curve.iloc[-1] > 0:
        ratio = equity_curve.iloc[-1] / equity_curve.iloc[0]
        cagr = (ratio ** (1 / years)) - 1.0
    else:
        cagr = 0.0

    vol = returns.std()
    volatility = vol * np.sqrt(periods_per_year) if vol and not np.isnan(vol) else 0.0

    excess = returns - risk_free_rate / periods_per_year
    sharpe = (excess.mean() / vol * np.sqrt(periods_per_year)) if vol and vol > 0 else 0.0

    downside = returns[returns < 0]
    downside_vol = downside.std() * np.sqrt(periods_per_year) if len(downside) > 0 else 0.0
    sortino = (returns.mean() / downside.std() * np.sqrt(periods_per_year)) if downside_vol and downside_vol > 0 else 0.0

    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax.replace(0, np.nan)
    max_drawdown = drawdown.min()
    if pd.isna(max_drawdown):
        max_drawdown = 0.0

    # Trade-based metrics
    trade_count = len(trades)
    if trade_count == 0:
        return Metrics(
            cagr=cagr,
            sharpe=sharpe,
            sortino=sortino,
            max_drawdown=max_drawdown,
            profit_factor=0.0,
            win_rate=0.0,
            expectancy=0.0,
            trade_count=0,
            total_return=total_return,
            volatility=volatility,
        )

    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    win_rate = len(wins) / trade_count if trade_count > 0 else 0.0
    expectancy = np.mean(pnls) if pnls else 0.0

    return Metrics(
        cagr=cagr,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_drawdown,
        profit_factor=profit_factor,
        win_rate=win_rate,
        expectancy=expectancy,
        trade_count=trade_count,
        total_return=total_return,
        volatility=volatility,
    )


def _empty_metrics() -> Metrics:
    return Metrics(
        cagr=0.0,
        sharpe=0.0,
        sortino=0.0,
        max_drawdown=0.0,
        profit_factor=0.0,
        win_rate=0.0,
        expectancy=0.0,
        trade_count=0,
        total_return=0.0,
        volatility=0.0,
    )
