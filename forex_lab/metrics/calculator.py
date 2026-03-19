"""Performance metrics — all computed from return series."""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Any


def compute_metrics(
    returns: pd.Series,
    periods_per_year: int = 252 * 6,
    risk_free_rate: float = 0.0,
    trades: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Compute a full suite of performance metrics.

    Args:
        returns: series of per-period net returns
        periods_per_year: annualization factor (default assumes hourly data, ~252*6)
        risk_free_rate: annual risk-free rate for Sharpe/Sortino
        trades: DataFrame with a 'return' column for trade-level stats
    """
    r = returns.dropna()
    if len(r) == 0:
        return _empty_metrics()

    total_return = (1 + r).prod() - 1
    n_periods = len(r)
    years = n_periods / periods_per_year

    cagr = (1 + total_return) ** (1 / max(years, 1e-6)) - 1 if total_return > -1 else -1.0

    excess = r - risk_free_rate / periods_per_year
    vol = r.std()
    sharpe = (excess.mean() / vol * np.sqrt(periods_per_year)) if vol > 0 else 0.0

    downside = r[r < 0].std()
    sortino = (excess.mean() / downside * np.sqrt(periods_per_year)) if downside > 0 else 0.0

    equity = (1 + r).cumprod()
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_drawdown = drawdown.min()

    gross_profit = r[r > 0].sum()
    gross_loss = abs(r[r < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    if trades is not None and len(trades) > 0:
        wins = trades[trades["return"] > 0]
        win_rate = len(wins) / len(trades)
        avg_win = wins["return"].mean() if len(wins) > 0 else 0.0
        avg_loss = trades[trades["return"] <= 0]["return"].mean()
        avg_loss = avg_loss if not np.isnan(avg_loss) else 0.0
        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
        trade_count = len(trades)
    else:
        positive_periods = (r > 0).sum()
        total_periods = len(r[r != 0])
        win_rate = positive_periods / total_periods if total_periods > 0 else 0.0
        expectancy = r.mean()
        trade_count = total_periods

    return {
        "cagr": cagr,
        "total_return": total_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "expectancy": expectancy,
        "trade_count": trade_count,
        "volatility": vol * np.sqrt(periods_per_year),
    }


def metrics_table(metrics_dict: dict[str, float]) -> pd.DataFrame:
    """Return a nicely formatted metrics table."""
    df = pd.DataFrame.from_dict(metrics_dict, orient="index", columns=["Value"])
    df.index.name = "Metric"

    fmt = {
        "cagr": "{:.2%}",
        "total_return": "{:.2%}",
        "sharpe": "{:.3f}",
        "sortino": "{:.3f}",
        "max_drawdown": "{:.2%}",
        "profit_factor": "{:.3f}",
        "win_rate": "{:.2%}",
        "expectancy": "{:.6f}",
        "trade_count": "{:.0f}",
        "volatility": "{:.2%}",
    }
    df["Formatted"] = df.apply(
        lambda row: fmt.get(row.name, "{:.4f}").format(row["Value"]), axis=1
    )
    return df


def _empty_metrics() -> dict[str, float]:
    return {
        "cagr": 0.0,
        "total_return": 0.0,
        "sharpe": 0.0,
        "sortino": 0.0,
        "max_drawdown": 0.0,
        "profit_factor": 0.0,
        "win_rate": 0.0,
        "expectancy": 0.0,
        "trade_count": 0,
        "volatility": 0.0,
    }
