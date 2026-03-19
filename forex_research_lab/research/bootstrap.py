"""Bootstrap robustness testing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from forex_research_lab.metrics import bars_per_year


@dataclass(slots=True)
class BootstrapSummary:
    samples: pd.DataFrame
    summary: dict[str, float]


def _max_drawdown_from_returns(returns: np.ndarray) -> float:
    equity = np.cumprod(1.0 + returns)
    peaks = np.maximum.accumulate(equity)
    drawdown = equity / peaks - 1.0
    return float(drawdown.min()) if len(drawdown) else 0.0


def bootstrap_returns(
    returns: pd.Series,
    timeframe: str,
    n_bootstrap: int = 500,
    ruin_threshold: float = 0.8,
    seed: int = 7,
) -> BootstrapSummary:
    """Resample return observations to estimate robustness distributions."""

    clean_returns = returns.fillna(0.0).to_numpy()
    rng = np.random.default_rng(seed)
    periods = bars_per_year(timeframe)
    rows: list[dict] = []

    if len(clean_returns) == 0:
        empty = pd.DataFrame(columns=["sharpe", "max_drawdown", "terminal_equity"])
        return BootstrapSummary(
            samples=empty,
            summary={"risk_of_ruin": 0.0},
        )

    for _ in range(n_bootstrap):
        sample = rng.choice(clean_returns, size=len(clean_returns), replace=True)
        mean_return = sample.mean()
        std_return = sample.std(ddof=0)
        sharpe = float(np.sqrt(periods) * mean_return / std_return) if std_return > 0 else 0.0
        terminal_equity = float(np.cumprod(1.0 + sample)[-1])
        rows.append(
            {
                "sharpe": sharpe,
                "max_drawdown": _max_drawdown_from_returns(sample),
                "terminal_equity": terminal_equity,
            }
        )

    samples = pd.DataFrame(rows)
    summary = {
        "sharpe_p05": float(samples["sharpe"].quantile(0.05)),
        "sharpe_p50": float(samples["sharpe"].quantile(0.50)),
        "sharpe_p95": float(samples["sharpe"].quantile(0.95)),
        "drawdown_p95": float(samples["max_drawdown"].quantile(0.05)),
        "risk_of_ruin": float((samples["terminal_equity"] < ruin_threshold).mean()),
    }
    return BootstrapSummary(samples=samples, summary=summary)
