"""Bootstrap robustness testing for returns and trade sequences."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from forex_research_lab.metrics.performance import annualized_sharpe, max_drawdown


@dataclass
class BootstrapResult:
    """Container for bootstrap metrics and summary statistics."""

    samples: pd.DataFrame
    summary: dict[str, float]


def bootstrap_returns(
    returns: pd.Series,
    n_samples: int = 500,
    sample_size: int | None = None,
    ruin_level: float = 0.7,
    seed: int = 42,
) -> BootstrapResult:
    """Bootstrap return bars and estimate outcome distributions."""
    clean_returns = returns.dropna()
    if clean_returns.empty:
        empty = pd.DataFrame(columns=["sample_id", "Sharpe", "Max Drawdown", "Ending Equity", "Ruined"])
        return BootstrapResult(samples=empty, summary={"Risk of Ruin": 0.0})

    sample_length = sample_size or len(clean_returns)
    generator = np.random.default_rng(seed)
    sample_rows: list[dict[str, float]] = []

    for sample_id in range(n_samples):
        sample = pd.Series(generator.choice(clean_returns.to_numpy(), size=sample_length, replace=True))
        equity = (1.0 + sample).cumprod()
        sample_rows.append(
            {
                "sample_id": float(sample_id),
                "Sharpe": annualized_sharpe(sample),
                "Max Drawdown": max_drawdown(equity),
                "Ending Equity": float(equity.iloc[-1]),
                "Ruined": float((equity < ruin_level).any()),
            }
        )

    sample_frame = pd.DataFrame(sample_rows)
    summary = {
        "Sharpe Mean": float(sample_frame["Sharpe"].mean()),
        "Sharpe 5%": float(sample_frame["Sharpe"].quantile(0.05)),
        "Sharpe 95%": float(sample_frame["Sharpe"].quantile(0.95)),
        "Drawdown Mean": float(sample_frame["Max Drawdown"].mean()),
        "Drawdown 95%": float(sample_frame["Max Drawdown"].quantile(0.95)),
        "Risk of Ruin": float(sample_frame["Ruined"].mean()),
    }
    return BootstrapResult(samples=sample_frame, summary=summary)


def bootstrap_trades(
    trades: pd.DataFrame,
    n_samples: int = 500,
    ruin_level: float = 0.7,
    seed: int = 42,
) -> BootstrapResult:
    """Bootstrap trade-level returns if a trade log is available."""
    if trades.empty or "net_return" not in trades.columns:
        empty = pd.DataFrame(columns=["sample_id", "Sharpe", "Max Drawdown", "Ending Equity", "Ruined"])
        return BootstrapResult(samples=empty, summary={"Risk of Ruin": 0.0})

    trade_returns = trades["net_return"].astype(float).dropna()
    return bootstrap_returns(
        returns=trade_returns.reset_index(drop=True),
        n_samples=n_samples,
        sample_size=len(trade_returns),
        ruin_level=ruin_level,
        seed=seed,
    )
