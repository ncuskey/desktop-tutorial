"""
Bootstrap robustness testing.

Two approaches:
1. Trade-level bootstrap: resample the trade P&L sequence with replacement.
2. Return-level block bootstrap: resample blocks of bar returns to preserve
   autocorrelation structure.

For each bootstrap sample we compute metrics, building a distribution
that answers: "Is the observed performance likely due to luck?"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from metrics.performance import compute_metrics, MetricsResult


@dataclass
class BootstrapResult:
    """Distribution of metrics from bootstrap sampling."""

    n_samples: int
    metric_distributions: dict[str, np.ndarray] = field(default_factory=dict)
    original_metrics: MetricsResult | None = None

    def percentile(self, metric: str, pct: float) -> float:
        arr = self.metric_distributions.get(metric, np.array([]))
        return float(np.percentile(arr, pct)) if len(arr) > 0 else 0.0

    def p_value(self, metric: str, threshold: float = 0.0) -> float:
        """Fraction of bootstrap samples where metric <= threshold."""
        arr = self.metric_distributions.get(metric, np.array([]))
        return float((arr <= threshold).mean()) if len(arr) > 0 else 1.0

    def confidence_interval(
        self, metric: str, alpha: float = 0.05
    ) -> tuple[float, float]:
        lo = self.percentile(metric, alpha / 2 * 100)
        hi = self.percentile(metric, (1 - alpha / 2) * 100)
        return lo, hi

    def summary(self) -> pd.DataFrame:
        rows = []
        for metric, dist in self.metric_distributions.items():
            if len(dist) == 0:
                continue
            orig = getattr(self.original_metrics, metric, np.nan) if self.original_metrics else np.nan
            lo, hi = self.confidence_interval(metric)
            rows.append(
                {
                    "metric": metric,
                    "original": orig,
                    "mean": np.mean(dist),
                    "std": np.std(dist),
                    "p5": self.percentile(metric, 5),
                    "p25": self.percentile(metric, 25),
                    "p50": self.percentile(metric, 50),
                    "p75": self.percentile(metric, 75),
                    "p95": self.percentile(metric, 95),
                    "ci_low_95": lo,
                    "ci_high_95": hi,
                    "p_value_zero": self.p_value(metric, 0.0),
                }
            )
        return pd.DataFrame(rows).set_index("metric")


class BootstrapEngine:
    """Monte-Carlo bootstrap engine for strategy robustness testing.

    Parameters
    ----------
    n_samples:
        Number of bootstrap iterations (default 1000).
    method:
        'trades'  — resample trades with replacement (IID assumption).
        'block'   — block bootstrap on bar returns (preserves autocorr).
    block_size:
        Block size for block bootstrap (default 20 bars).
    metrics_to_track:
        List of MetricsResult attribute names to track.
    """

    _DEFAULT_METRICS = [
        "sharpe", "sortino", "cagr", "max_drawdown",
        "profit_factor", "win_rate", "calmar",
    ]

    def __init__(
        self,
        n_samples: int = 1000,
        method: str = "block",
        block_size: int = 20,
        metrics_to_track: list[str] | None = None,
    ) -> None:
        self.n_samples = n_samples
        self.method = method
        self.block_size = block_size
        self.metrics_to_track = metrics_to_track or self._DEFAULT_METRICS

    def run_on_returns(
        self,
        returns: pd.Series,
        trades: pd.DataFrame | None = None,
        bars_per_year: int | None = None,
        verbose: bool = True,
    ) -> BootstrapResult:
        """Bootstrap from bar return series."""
        original_metrics = compute_metrics(returns, trades, bars_per_year)
        distributions: dict[str, list[float]] = {m: [] for m in self.metrics_to_track}

        rng = np.random.default_rng(seed=0)
        ret_arr = returns.values

        iter_range = range(self.n_samples)
        if verbose:
            iter_range = tqdm(iter_range, desc="Bootstrap (returns)")

        for _ in iter_range:
            sampled = self._sample_returns(ret_arr, rng)
            sampled_series = pd.Series(sampled, dtype=float)
            m = compute_metrics(sampled_series, bars_per_year=bars_per_year)
            for metric in self.metrics_to_track:
                distributions[metric].append(getattr(m, metric, 0.0))

        return BootstrapResult(
            n_samples=self.n_samples,
            metric_distributions={k: np.array(v) for k, v in distributions.items()},
            original_metrics=original_metrics,
        )

    def run_on_trades(
        self,
        trades: pd.DataFrame,
        bars_per_year: int = 252,
        verbose: bool = True,
    ) -> BootstrapResult:
        """Bootstrap from trade P&L distribution."""
        if "pnl" not in trades.columns or len(trades) == 0:
            return BootstrapResult(n_samples=self.n_samples)

        pnl = trades["pnl"].dropna().values
        original_metrics = _metrics_from_pnl(pnl, bars_per_year)
        distributions: dict[str, list[float]] = {m: [] for m in self.metrics_to_track}

        rng = np.random.default_rng(seed=0)

        iter_range = range(self.n_samples)
        if verbose:
            iter_range = tqdm(iter_range, desc="Bootstrap (trades)")

        for _ in iter_range:
            sampled_pnl = rng.choice(pnl, size=len(pnl), replace=True)
            m = _metrics_from_pnl(sampled_pnl, bars_per_year)
            for metric in self.metrics_to_track:
                distributions[metric].append(getattr(m, metric, 0.0))

        return BootstrapResult(
            n_samples=self.n_samples,
            metric_distributions={k: np.array(v) for k, v in distributions.items()},
            original_metrics=original_metrics,
        )

    # ------------------------------------------------------------------

    def _sample_returns(self, arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n = len(arr)
        if self.method == "trades" or self.block_size >= n:
            return rng.choice(arr, size=n, replace=True)

        # Block bootstrap
        block_starts = rng.integers(0, max(n - self.block_size, 1), size=n // self.block_size + 1)
        blocks = [arr[s : s + self.block_size] for s in block_starts]
        sampled = np.concatenate(blocks)[:n]
        return sampled


def _metrics_from_pnl(pnl: np.ndarray, bars_per_year: int) -> MetricsResult:
    """Create a MetricsResult from a trade P&L array."""
    returns = pd.Series(pnl)
    trades = pd.DataFrame({"pnl": pnl})
    return compute_metrics(returns, trades, bars_per_year)
