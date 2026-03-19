"""Performance metrics — computed from execution results."""

from __future__ import annotations

import numpy as np
import pandas as pd

ANNUALIZATION_FACTORS = {
    "h": np.sqrt(252 * 24),
    "1h": np.sqrt(252 * 24),
    "H1": np.sqrt(252 * 24),
    "4h": np.sqrt(252 * 6),
    "H4": np.sqrt(252 * 6),
    "D1": np.sqrt(252),
    "1D": np.sqrt(252),
    "D": np.sqrt(252),
}

PERIODS_PER_YEAR = {
    "h": 252 * 24,
    "1h": 252 * 24,
    "H1": 252 * 24,
    "4h": 252 * 6,
    "H4": 252 * 6,
    "D1": 252,
    "1D": 252,
    "D": 252,
}


def compute_metrics(
    result: pd.DataFrame,
    freq: str = "h",
) -> dict[str, float]:
    """Compute a full suite of performance metrics.

    Parameters
    ----------
    result : DataFrame
        Output of ``execute_signals`` — must contain ``net_returns``,
        ``equity``, ``drawdown``, ``position``.
    freq : str
        Bar frequency for annualization (h, H1, H4, D1, etc.).
    """
    ret = result["net_returns"].values
    equity = result["equity"].values

    ann_factor = ANNUALIZATION_FACTORS.get(freq, np.sqrt(252))
    ppy = PERIODS_PER_YEAR.get(freq, 252)
    n_periods = len(ret)

    total_return = equity[-1] / equity[0] - 1 if equity[0] != 0 else 0.0
    years = n_periods / ppy if ppy > 0 else 1.0
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0

    mean_r = np.mean(ret)
    std_r = np.std(ret, ddof=1) if len(ret) > 1 else 1e-10
    sharpe = (mean_r / std_r * ann_factor) if std_r > 0 else 0.0

    downside = ret[ret < 0]
    down_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-10
    sortino = (mean_r / down_std * ann_factor) if down_std > 0 else 0.0

    max_dd = float(result["drawdown"].min())

    winning = ret[ret > 0]
    losing = ret[ret < 0]
    gross_profit = float(np.sum(winning)) if len(winning) else 0.0
    gross_loss = float(np.abs(np.sum(losing))) if len(losing) else 1e-10
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    pos = result["position"].values
    trades = np.sum(np.abs(np.diff(pos)) > 0)

    active_mask = np.abs(pos) > 0
    trade_returns = ret[active_mask] if np.any(active_mask) else ret
    win_rate = float(np.mean(trade_returns > 0)) if len(trade_returns) > 0 else 0.0

    avg_win = float(np.mean(winning)) if len(winning) > 0 else 0.0
    avg_loss = float(np.mean(np.abs(losing))) if len(losing) > 0 else 1e-10
    expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss

    return {
        "cagr": round(cagr, 6),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "max_drawdown": round(max_dd, 6),
        "profit_factor": round(profit_factor, 4),
        "win_rate": round(win_rate, 4),
        "expectancy": round(expectancy, 8),
        "trade_count": int(trades),
        "total_return": round(total_return, 6),
    }


def metrics_table(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Build a comparison table from ``{name: metrics_dict}``."""
    return pd.DataFrame(results).T
