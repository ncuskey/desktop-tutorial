"""Bootstrap robustness engine.

Resamples trades or returns to build distributions of performance metrics
and estimate the probability that observed results are due to chance.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Any

from ..metrics.calculator import compute_metrics


class BootstrapEngine:
    """Bootstrap analysis for strategy robustness.

    Given a series of returns (or trade-level returns), resamples with
    replacement to produce distributions of key metrics.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        sample_size: int | None = None,
        seed: int = 42,
        periods_per_year: int = 252 * 6,
    ):
        self.n_samples = n_samples
        self.sample_size = sample_size
        self.seed = seed
        self.periods_per_year = periods_per_year

    def run(self, returns: pd.Series) -> BootstrapResult:
        """Run bootstrap resampling on a return series.

        Returns distributions of Sharpe, max drawdown, and total return.
        """
        rng = np.random.default_rng(self.seed)
        r = returns.dropna().values
        size = self.sample_size or len(r)

        sharpe_dist = []
        drawdown_dist = []
        return_dist = []
        sortino_dist = []

        for _ in range(self.n_samples):
            sample = rng.choice(r, size=size, replace=True)
            sample_series = pd.Series(sample)
            metrics = compute_metrics(sample_series, periods_per_year=self.periods_per_year)
            sharpe_dist.append(metrics["sharpe"])
            drawdown_dist.append(metrics["max_drawdown"])
            return_dist.append(metrics["total_return"])
            sortino_dist.append(metrics["sortino"])

        return BootstrapResult(
            sharpe=np.array(sharpe_dist),
            max_drawdown=np.array(drawdown_dist),
            total_return=np.array(return_dist),
            sortino=np.array(sortino_dist),
        )

    def risk_of_ruin(
        self,
        returns: pd.Series,
        ruin_threshold: float = -0.5,
    ) -> float:
        """Estimate probability of drawdown exceeding ruin_threshold.

        Uses bootstrap to estimate P(max_drawdown < ruin_threshold).
        """
        result = self.run(returns)
        ruin_count = (result.max_drawdown < ruin_threshold).sum()
        return ruin_count / len(result.max_drawdown)


class BootstrapResult:
    """Container for bootstrap distributions."""

    def __init__(
        self,
        sharpe: np.ndarray,
        max_drawdown: np.ndarray,
        total_return: np.ndarray,
        sortino: np.ndarray,
    ):
        self.sharpe = sharpe
        self.max_drawdown = max_drawdown
        self.total_return = total_return
        self.sortino = sortino

    def summary(self) -> pd.DataFrame:
        """Return summary statistics of all distributions."""
        rows = {}
        for name, arr in [
            ("sharpe", self.sharpe),
            ("sortino", self.sortino),
            ("max_drawdown", self.max_drawdown),
            ("total_return", self.total_return),
        ]:
            rows[name] = {
                "mean": np.mean(arr),
                "std": np.std(arr),
                "p5": np.percentile(arr, 5),
                "p25": np.percentile(arr, 25),
                "median": np.median(arr),
                "p75": np.percentile(arr, 75),
                "p95": np.percentile(arr, 95),
            }
        return pd.DataFrame(rows).T

    def confidence_interval(self, metric: str, alpha: float = 0.05) -> tuple[float, float]:
        """Return (lower, upper) confidence interval for a metric."""
        arr = getattr(self, metric)
        return float(np.percentile(arr, 100 * alpha / 2)), float(
            np.percentile(arr, 100 * (1 - alpha / 2))
        )
