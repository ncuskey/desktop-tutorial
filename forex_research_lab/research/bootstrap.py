"""Bootstrap robustness analysis for strategy outcomes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from forex_research_lab.metrics import cagr, max_drawdown, sharpe_ratio


@dataclass
class BootstrapResult:
    distribution: pd.DataFrame
    summary: dict[str, float]


def _bootstrap_from_array(
    values: np.ndarray,
    n_samples: int,
    periods_per_year: int,
    seed: int,
    ruin_threshold: float,
) -> BootstrapResult:
    rng = np.random.default_rng(seed)
    sample_len = len(values)
    rows: list[dict[str, float]] = []

    for _ in range(n_samples):
        sampled = rng.choice(values, size=sample_len, replace=True)
        sampled_returns = pd.Series(sampled)
        sampled_equity = (1.0 + sampled_returns).cumprod()
        sampled_dd = sampled_equity / sampled_equity.cummax() - 1.0

        rows.append(
            {
                "sharpe": sharpe_ratio(sampled_returns, periods_per_year),
                "max_drawdown": max_drawdown(sampled_dd),
                "cagr": cagr(sampled_equity * 100_000.0, periods_per_year),
                "ending_equity_multiple": float(sampled_equity.iloc[-1]),
            }
        )

    dist = pd.DataFrame(rows)
    summary = {
        "sharpe_mean": float(dist["sharpe"].mean()),
        "sharpe_p05": float(dist["sharpe"].quantile(0.05)),
        "sharpe_p50": float(dist["sharpe"].quantile(0.50)),
        "sharpe_p95": float(dist["sharpe"].quantile(0.95)),
        "drawdown_mean": float(dist["max_drawdown"].mean()),
        "drawdown_p95_worst": float(dist["max_drawdown"].quantile(0.05)),
        "risk_of_ruin": float((dist["ending_equity_multiple"] <= ruin_threshold).mean()),
    }
    return BootstrapResult(distribution=dist, summary=summary)


def bootstrap_returns(
    returns: pd.Series,
    n_samples: int = 1000,
    periods_per_year: int = 252,
    seed: int = 42,
    ruin_threshold: float = 0.70,
) -> BootstrapResult:
    if returns.empty:
        raise ValueError("Cannot bootstrap an empty returns series")
    values = returns.to_numpy(dtype=float)
    return _bootstrap_from_array(
        values=values,
        n_samples=n_samples,
        periods_per_year=periods_per_year,
        seed=seed,
        ruin_threshold=ruin_threshold,
    )


def bootstrap_trade_returns(
    trades: pd.DataFrame,
    n_samples: int = 1000,
    periods_per_year: int = 252,
    seed: int = 42,
    ruin_threshold: float = 0.70,
) -> BootstrapResult:
    if trades.empty or "net_return" not in trades.columns:
        raise ValueError("Trades dataframe must contain non-empty 'net_return' column")
    values = trades["net_return"].to_numpy(dtype=float)
    return _bootstrap_from_array(
        values=values,
        n_samples=n_samples,
        periods_per_year=periods_per_year,
        seed=seed,
        ruin_threshold=ruin_threshold,
    )
