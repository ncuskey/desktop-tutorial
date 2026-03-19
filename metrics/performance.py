from __future__ import annotations

import numpy as np
import pandas as pd


TIMEFRAME_TO_PERIODS = {"H1": 24 * 252, "H4": 6 * 252, "D1": 252}


def compute_metrics(
    returns: pd.Series,
    equity: pd.Series,
    trades: pd.DataFrame,
    timeframe: str = "H1",
) -> dict[str, float]:
    ppy = TIMEFRAME_TO_PERIODS.get(timeframe.upper(), 24 * 252)
    clean_returns = returns.dropna()

    if clean_returns.empty:
        return {
            "CAGR": 0.0,
            "Sharpe": 0.0,
            "Sortino": 0.0,
            "MaxDrawdown": 0.0,
            "ProfitFactor": 0.0,
            "WinRate": 0.0,
            "Expectancy": 0.0,
            "TradeCount": 0.0,
        }

    mean_r = clean_returns.mean()
    std_r = clean_returns.std(ddof=0)
    downside_std = clean_returns[clean_returns < 0].std(ddof=0)

    years = max(len(clean_returns) / ppy, 1e-9)
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if equity.iloc[0] != 0 else 0
    sharpe = np.sqrt(ppy) * (mean_r / std_r) if std_r > 0 else 0.0
    sortino = np.sqrt(ppy) * (mean_r / downside_std) if downside_std > 0 else 0.0
    max_dd = float(((equity / equity.cummax()) - 1.0).min())

    if trades.empty or "trade_return" not in trades:
        profit_factor = 0.0
        win_rate = 0.0
        expectancy = 0.0
        trade_count = 0.0
    else:
        tr = trades["trade_return"]
        gross_profit = tr[tr > 0].sum()
        gross_loss = -tr[tr < 0].sum()
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else float("inf")
        win_rate = float((tr > 0).mean()) if len(tr) > 0 else 0.0
        expectancy = float(tr.mean()) if len(tr) > 0 else 0.0
        trade_count = float(len(tr))

    return {
        "CAGR": float(cagr),
        "Sharpe": float(sharpe),
        "Sortino": float(sortino),
        "MaxDrawdown": max_dd,
        "ProfitFactor": float(profit_factor),
        "WinRate": float(win_rate),
        "Expectancy": float(expectancy),
        "TradeCount": trade_count,
    }


def compute_metrics_by_regime(
    df: pd.DataFrame,
    regime_column: str,
    timeframe: str = "H1",
    returns_column: str = "returns",
    position_column: str = "position",
) -> pd.DataFrame:
    """Compute per-regime metrics for conditional edge analysis."""
    required = {regime_column, returns_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for regime metrics: {sorted(missing)}")

    ppy = TIMEFRAME_TO_PERIODS.get(timeframe.upper(), 24 * 252)
    rows: list[dict] = []

    for regime, g in df.groupby(regime_column, dropna=False, sort=False):
        returns = g[returns_column].dropna()
        if returns.empty:
            rows.append(
                {
                    "Regime": regime,
                    "Sharpe": 0.0,
                    "CAGR": 0.0,
                    "MaxDrawdown": 0.0,
                    "TradeCount": 0.0,
                    "Bars": 0.0,
                    "TimePct": 0.0,
                }
            )
            continue

        equity = (1.0 + returns).cumprod() * 100_000.0
        years = max(len(returns) / ppy, 1e-9)
        cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if equity.iloc[0] != 0 else 0

        std_r = returns.std(ddof=0)
        sharpe = float(np.sqrt(ppy) * (returns.mean() / std_r)) if std_r > 0 else 0.0
        max_dd = float(((equity / equity.cummax()) - 1.0).min())

        trade_count = 0.0
        if position_column in g.columns:
            pos = g[position_column].fillna(0)
            entries = ((pos != 0) & (pos.shift(1).fillna(0) == 0)).sum()
            trade_count = float(entries)

        rows.append(
            {
                "Regime": regime,
                "Sharpe": sharpe,
                "CAGR": float(cagr),
                "MaxDrawdown": max_dd,
                "TradeCount": trade_count,
                "Bars": float(len(returns)),
                "TimePct": float(len(returns) / len(df)) if len(df) > 0 else 0.0,
            }
        )

    return pd.DataFrame(rows).sort_values("Regime").reset_index(drop=True)
