"""Bootstrap engine for robustness testing.

Resamples trade returns with replacement to build distributions of
performance metrics and estimate tail-risk statistics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..metrics.performance import ANNUALIZATION_FACTORS


class BootstrapEngine:
    """Monte-Carlo bootstrap of trade returns."""

    def __init__(
        self,
        n_samples: int = 1000,
        sample_length: int | None = None,
        seed: int = 42,
        freq: str = "h",
    ):
        self.n_samples = n_samples
        self.sample_length = sample_length
        self.seed = seed
        self.freq = freq

    def run(self, returns: np.ndarray | pd.Series) -> dict[str, object]:
        """Run bootstrap resampling on a return series.

        Returns
        -------
        dict with keys:
            sharpe_dist, max_dd_dist, final_equity_dist,
            sharpe_ci, max_dd_ci, risk_of_ruin
        """
        if isinstance(returns, pd.Series):
            returns = returns.values

        returns = returns[np.isfinite(returns)]
        n = self.sample_length or len(returns)
        rng = np.random.default_rng(self.seed)
        ann = ANNUALIZATION_FACTORS.get(self.freq, np.sqrt(252))

        sharpes = np.empty(self.n_samples)
        max_dds = np.empty(self.n_samples)
        final_equities = np.empty(self.n_samples)

        for i in range(self.n_samples):
            sample = rng.choice(returns, size=n, replace=True)
            equity = 100_000 * np.cumprod(1 + sample)
            peak = np.maximum.accumulate(equity)
            dd = (equity - peak) / peak

            mean_r = np.mean(sample)
            std_r = np.std(sample, ddof=1)
            sharpes[i] = (mean_r / std_r * ann) if std_r > 0 else 0.0
            max_dds[i] = dd.min()
            final_equities[i] = equity[-1]

        risk_of_ruin = float(np.mean(final_equities < 100_000 * 0.5))

        return {
            "sharpe_dist": sharpes,
            "max_dd_dist": max_dds,
            "final_equity_dist": final_equities,
            "sharpe_mean": float(np.mean(sharpes)),
            "sharpe_ci": (float(np.percentile(sharpes, 2.5)), float(np.percentile(sharpes, 97.5))),
            "max_dd_mean": float(np.mean(max_dds)),
            "max_dd_ci": (float(np.percentile(max_dds, 2.5)), float(np.percentile(max_dds, 97.5))),
            "risk_of_ruin": risk_of_ruin,
        }
