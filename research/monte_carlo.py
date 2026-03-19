from __future__ import annotations

import numpy as np
import pandas as pd


def _max_drawdown_from_returns(returns: np.ndarray) -> float:
    equity = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(equity)
    dd = (equity / peak) - 1.0
    return float(np.min(dd))


def _sample_block_bootstrap_path(
    returns: np.ndarray,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(returns)
    if n == 0:
        return np.array([], dtype=float)
    if block_size <= 1:
        idx = rng.choice(np.arange(n), size=n, replace=True)
        return returns[idx]

    blocks: list[np.ndarray] = []
    while sum(len(b) for b in blocks) < n:
        start = int(rng.integers(0, max(n - block_size + 1, 1)))
        block = returns[start : start + block_size]
        if len(block) == 0:
            continue
        blocks.append(block)
    stitched = np.concatenate(blocks)[:n]
    return stitched


def block_bootstrap_oos_returns(
    returns: pd.Series,
    n_bootstrap: int = 500,
    block_size: int = 24,
    seed: int = 42,
    periods_per_year: int = 24 * 252,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Block bootstrap for stitched OOS returns to preserve short-term serial structure.
    """
    clean = returns.dropna().to_numpy(dtype=float)
    if clean.size == 0:
        raise ValueError("Cannot run block bootstrap on empty returns.")

    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    for _ in range(int(n_bootstrap)):
        path = _sample_block_bootstrap_path(clean, block_size=block_size, rng=rng)
        mean_r = float(np.mean(path))
        std_r = float(np.std(path))
        sharpe = (np.sqrt(periods_per_year) * mean_r / std_r) if std_r > 1e-12 else 0.0
        max_dd = _max_drawdown_from_returns(path)
        final_multiple = float(np.prod(1.0 + path))
        rows.append(
            {
                "Sharpe": sharpe,
                "MaxDrawdown": max_dd,
                "FinalEquityMultiple": final_multiple,
            }
        )

    dist = pd.DataFrame(rows)
    summary = {
        "SharpeP05": float(dist["Sharpe"].quantile(0.05)),
        "SharpeMedian": float(dist["Sharpe"].median()),
        "SharpeP95": float(dist["Sharpe"].quantile(0.95)),
        "MaxDrawdownMedian": float(dist["MaxDrawdown"].median()),
        "MaxDrawdownP95": float(dist["MaxDrawdown"].quantile(0.05)),
        "FinalEquityMedian": float(dist["FinalEquityMultiple"].median()),
        "FinalEquityP05": float(dist["FinalEquityMultiple"].quantile(0.05)),
    }
    return dist, summary
