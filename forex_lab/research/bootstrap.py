from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _sample_with_replacement(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    idx = rng.integers(0, len(values), len(values))
    return values[idx]


def bootstrap_returns(
    returns: pd.Series,
    n_bootstrap: int = 500,
    ruin_threshold: float = 0.7,
    periods_per_year: int = 252,
    seed: int = 42,
) -> Dict[str, pd.DataFrame | float]:
    """
    Bootstraps strategy returns to stress robustness.
    Returns Sharpe distribution, drawdown distribution, and risk-of-ruin estimate.
    """
    clean = returns.dropna().values
    if clean.size == 0:
        empty = pd.DataFrame(columns=["sharpe", "max_drawdown", "terminal_equity"])
        return {"distribution": empty, "risk_of_ruin": 0.0}

    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_bootstrap):
        sampled = _sample_with_replacement(clean, rng=rng)
        equity = np.cumprod(1 + sampled)
        drawdown = equity / np.maximum.accumulate(equity) - 1
        std = sampled.std(ddof=0)
        sharpe = (sampled.mean() / std) * np.sqrt(periods_per_year) if std > 0 else 0.0
        rows.append(
            {
                "sharpe": sharpe,
                "max_drawdown": float(drawdown.min()) if drawdown.size else 0.0,
                "terminal_equity": float(equity[-1]) if equity.size else 1.0,
            }
        )
    dist = pd.DataFrame(rows)
    risk_of_ruin = float((dist["terminal_equity"] < ruin_threshold).mean())
    return {"distribution": dist, "risk_of_ruin": risk_of_ruin}
