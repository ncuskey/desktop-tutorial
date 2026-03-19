"""Bootstrap robustness testing."""

from typing import Optional

import numpy as np
import pandas as pd

from forex_lab.metrics.compute import compute_metrics


class BootstrapEngine:
    """
    Resample trades or returns to estimate distribution of outcomes.

    - Sharpe distribution
    - Drawdown distribution
    - Risk of ruin
    """

    def __init__(
        self,
        n_simulations: int = 1000,
        random_state: Optional[int] = None,
    ):
        self.n_simulations = n_simulations
        self.rng = np.random.default_rng(random_state)

    def bootstrap_returns(
        self,
        returns: pd.Series,
        initial_capital: float = 100_000.0,
    ) -> dict:
        """
        Bootstrap resample returns, compute metric distributions.

        Returns:
            Dict with sharpe_dist, sortino_dist, max_dd_dist, cagr_dist
        """
        returns = returns.dropna().values
        n = len(returns)
        if n < 10:
            return {"sharpe_dist": [], "max_dd_dist": [], "cagr_dist": []}

        sharpes = []
        sortinos = []
        max_dds = []
        cagrs = []

        for _ in range(self.n_simulations):
            idx = self.rng.integers(0, n, size=n)
            resampled = returns[idx]
            equity = initial_capital * np.cumprod(1 + resampled)
            equity_series = pd.Series(equity)
            metrics = compute_metrics(equity_series, [])
            sharpes.append(metrics.sharpe)
            sortinos.append(metrics.sortino)
            max_dds.append(metrics.max_drawdown)
            years = n / 252.0  # assume daily
            cagr = (equity[-1] / initial_capital) ** (1 / max(years, 0.001)) - 1.0
            cagrs.append(cagr)

        return {
            "sharpe_dist": np.array(sharpes),
            "sortino_dist": np.array(sortinos),
            "max_dd_dist": np.array(max_dds),
            "cagr_dist": np.array(cagrs),
        }

    def bootstrap_trades(
        self,
        trades: list,
        initial_capital: float = 100_000.0,
    ) -> dict:
        """
        Bootstrap resample trades (with replacement), compute metric distributions.
        """
        n = len(trades)
        if n < 5:
            return {"sharpe_dist": [], "max_dd_dist": [], "risk_of_ruin": 0.0}

        sharpes = []
        max_dds = []
        final_equities = []

        for _ in range(self.n_simulations):
            idx = self.rng.integers(0, n, size=n)
            pnls = [trades[i].pnl for i in idx]
            equity = initial_capital + np.cumsum(pnls)
            equity_series = pd.Series(equity)
            metrics = compute_metrics(equity_series, [])
            sharpes.append(metrics.sharpe)
            max_dds.append(metrics.max_drawdown)
            final_equities.append(equity[-1])

        risk_of_ruin = np.mean(np.array(final_equities) <= 0)

        return {
            "sharpe_dist": np.array(sharpes),
            "max_dd_dist": np.array(max_dds),
            "risk_of_ruin": risk_of_ruin,
        }
