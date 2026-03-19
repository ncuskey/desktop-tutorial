"""
Performance metrics for strategy evaluation.

All metrics operate on a series of bar-level returns (log or arithmetic).
Annualisation constants are inferred from inferred bar frequency when possible,
or from an explicit `bars_per_year` parameter.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd


# Common bars-per-year constants
_BARS_PER_YEAR: dict[str, int] = {
    "H": 8760,   # hourly (24×365)
    "H1": 8760,
    "H4": 2190,
    "D": 252,
    "D1": 252,
    "B": 252,
    "W": 52,
    "ME": 12,
    "M": 12,
}


def _infer_bars_per_year(returns: pd.Series) -> int:
    """Infer bars-per-year from the index frequency or median delta."""
    if isinstance(returns.index, pd.DatetimeIndex):
        if returns.index.freq is not None:
            freq_str = returns.index.freqstr
            for k, v in _BARS_PER_YEAR.items():
                if freq_str.startswith(k):
                    return v
        if len(returns) > 1:
            median_delta = pd.Series(returns.index).diff().median()
            hours = median_delta.total_seconds() / 3600
            if hours <= 1.1:
                return 8760
            elif hours <= 4.1:
                return 2190
            elif hours <= 25:
                return 252
            elif hours <= 170:
                return 52
    return 252


@dataclass
class MetricsResult:
    """Structured container for all performance metrics."""

    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float
    profit_factor: float
    win_rate: float
    expectancy: float
    n_trades: int
    avg_trade_pnl: float
    avg_win: float
    avg_loss: float
    calmar: float
    volatility_ann: float
    skewness: float
    kurtosis: float
    bars_per_year: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        lines = [
            f"  CAGR:            {self.cagr:>8.2%}",
            f"  Sharpe:          {self.sharpe:>8.3f}",
            f"  Sortino:         {self.sortino:>8.3f}",
            f"  Calmar:          {self.calmar:>8.3f}",
            f"  Max Drawdown:    {self.max_drawdown:>8.2%}",
            f"  Volatility (ann):{self.volatility_ann:>8.2%}",
            f"  Profit Factor:   {self.profit_factor:>8.3f}",
            f"  Win Rate:        {self.win_rate:>8.2%}",
            f"  Expectancy:      {self.expectancy:>8.5f}",
            f"  Avg Win:         {self.avg_win:>8.5f}",
            f"  Avg Loss:        {self.avg_loss:>8.5f}",
            f"  N Trades:        {self.n_trades:>8d}",
            f"  Skewness:        {self.skewness:>8.3f}",
            f"  Kurtosis:        {self.kurtosis:>8.3f}",
        ]
        return "\n".join(lines)


def compute_metrics(
    returns: pd.Series,
    trades: pd.DataFrame | None = None,
    bars_per_year: int | None = None,
    risk_free_rate: float = 0.0,
) -> MetricsResult:
    """Compute full set of performance metrics.

    Parameters
    ----------
    returns:
        Bar-level returns (log or simple arithmetic).
    trades:
        Optional trade log DataFrame with 'pnl' column.
    bars_per_year:
        Override bars-per-year for annualisation.
    risk_free_rate:
        Annual risk-free rate (default 0).

    Returns
    -------
    MetricsResult with all computed metrics.
    """
    returns = returns.dropna()
    if len(returns) == 0:
        return _empty_metrics(bars_per_year or 252)

    bpy = bars_per_year or _infer_bars_per_year(returns)
    rf_per_bar = (1 + risk_free_rate) ** (1 / bpy) - 1

    # --- CAGR ---
    n_bars = len(returns)
    total_return = np.exp(returns.sum()) - 1  # works for log returns
    years = n_bars / bpy
    cagr = (1 + total_return) ** (1 / max(years, 1e-6)) - 1

    # --- Volatility ---
    vol_ann = returns.std() * np.sqrt(bpy)

    # --- Sharpe ---
    excess = returns - rf_per_bar
    sharpe = (excess.mean() / excess.std() * np.sqrt(bpy)) if excess.std() > 0 else 0.0

    # --- Sortino (downside deviation) ---
    downside = returns[returns < rf_per_bar] - rf_per_bar
    downside_std = np.sqrt((downside**2).mean()) * np.sqrt(bpy) if len(downside) > 0 else 1e-10
    sortino = (returns.mean() - rf_per_bar) * np.sqrt(bpy) / max(downside_std, 1e-10)

    # --- Max Drawdown ---
    equity = (1 + returns).cumprod()
    rolling_max = equity.cummax()
    drawdown_series = (equity - rolling_max) / rolling_max
    max_drawdown = float(drawdown_series.min())

    # --- Calmar ---
    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0.0

    # --- Trade-level metrics ---
    if trades is not None and len(trades) > 0 and "pnl" in trades.columns:
        pnl = trades["pnl"].dropna()
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        n_trades = len(pnl)
        win_rate = len(wins) / n_trades if n_trades > 0 else 0.0
        avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
        avg_trade_pnl = float(pnl.mean())
        gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
        gross_loss = abs(float(losses.sum())) if len(losses) > 0 else 1e-10
        profit_factor = gross_profit / max(gross_loss, 1e-10)
        expectancy = avg_trade_pnl
    else:
        # Estimate from returns
        positive = returns[returns > 0]
        negative = returns[returns < 0]
        n_trades = int((returns.diff().abs() > 1e-8).sum())
        win_rate = len(positive) / len(returns) if len(returns) > 0 else 0.0
        avg_win = float(positive.mean()) if len(positive) > 0 else 0.0
        avg_loss = float(negative.mean()) if len(negative) > 0 else 0.0
        avg_trade_pnl = float(returns.mean())
        gross_profit = float(positive.sum())
        gross_loss = abs(float(negative.sum()))
        profit_factor = gross_profit / max(gross_loss, 1e-10)
        expectancy = avg_trade_pnl

    return MetricsResult(
        cagr=float(cagr),
        sharpe=float(sharpe),
        sortino=float(sortino),
        max_drawdown=float(max_drawdown),
        profit_factor=float(profit_factor),
        win_rate=float(win_rate),
        expectancy=float(expectancy),
        n_trades=int(n_trades),
        avg_trade_pnl=float(avg_trade_pnl),
        avg_win=float(avg_win),
        avg_loss=float(avg_loss),
        calmar=float(calmar),
        volatility_ann=float(vol_ann),
        skewness=float(returns.skew()),
        kurtosis=float(returns.kurtosis()),
        bars_per_year=int(bpy),
    )


def _empty_metrics(bpy: int) -> MetricsResult:
    return MetricsResult(
        cagr=0.0, sharpe=0.0, sortino=0.0, max_drawdown=0.0,
        profit_factor=0.0, win_rate=0.0, expectancy=0.0, n_trades=0,
        avg_trade_pnl=0.0, avg_win=0.0, avg_loss=0.0, calmar=0.0,
        volatility_ann=0.0, skewness=0.0, kurtosis=0.0, bars_per_year=bpy,
    )
