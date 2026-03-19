from __future__ import annotations

import numpy as np
import pandas as pd


def _max_drawdown_from_returns(returns: np.ndarray) -> float:
    equity = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(equity)
    dd = (equity / peak) - 1.0
    return float(np.min(dd))


def bootstrap_returns(
    returns: pd.Series,
    n_bootstrap: int = 500,
    seed: int = 21,
    ruin_threshold: float = 0.8,
    periods_per_year: int = 24 * 252,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Bootstrap return paths and estimate robustness distributions."""

    clean = returns.dropna().to_numpy()
    if clean.size == 0:
        raise ValueError("Cannot bootstrap empty returns.")

    rng = np.random.default_rng(seed)
    results: list[dict] = []
    for _ in range(n_bootstrap):
        sample = rng.choice(clean, size=clean.size, replace=True)
        mean_r = float(np.mean(sample))
        std_r = float(np.std(sample))
        sharpe = (np.sqrt(periods_per_year) * mean_r / std_r) if std_r > 0 else 0.0
        max_dd = _max_drawdown_from_returns(sample)
        final_equity = float(np.prod(1.0 + sample))
        results.append(
            {
                "Sharpe": sharpe,
                "MaxDrawdown": max_dd,
                "FinalEquityMultiple": final_equity,
                "IsRuin": final_equity < ruin_threshold,
            }
        )

    dist = pd.DataFrame(results)
    summary = {
        "SharpeP05": float(dist["Sharpe"].quantile(0.05)),
        "SharpeMedian": float(dist["Sharpe"].median()),
        "SharpeP95": float(dist["Sharpe"].quantile(0.95)),
        "DrawdownP95": float(dist["MaxDrawdown"].quantile(0.05)),
        "RiskOfRuin": float(dist["IsRuin"].mean()),
    }
    return dist, summary
