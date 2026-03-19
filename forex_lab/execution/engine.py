from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    drawdown_curve: pd.Series
    returns: pd.Series
    gross_returns: pd.Series
    costs: pd.Series
    effective_position: pd.Series
    trades: pd.DataFrame


def _extract_trades(returns: pd.Series, effective_position: pd.Series) -> pd.DataFrame:
    active = effective_position != 0
    if not active.any():
        return pd.DataFrame(columns=["entry_time", "exit_time", "side", "bars", "trade_return"])

    change_id = (effective_position != effective_position.shift(1)).cumsum()
    rows = []
    for _, segment in returns[active].groupby(change_id[active]):
        idx = segment.index
        pos = int(effective_position.loc[idx[0]])
        trade_ret = (1 + segment).prod() - 1
        rows.append(
            {
                "entry_time": idx[0],
                "exit_time": idx[-1],
                "side": pos,
                "bars": len(segment),
                "trade_return": trade_ret,
            }
        )
    return pd.DataFrame(rows)


def backtest_signals(
    df: pd.DataFrame,
    signal: pd.Series,
    initial_capital: float = 100_000.0,
    default_total_cost_bps: float = 2.2,
) -> BacktestResult:
    """
    Backtest a position signal with realistic transaction costs.

    Lookahead is prevented by applying returns to lagged positions only.
    """
    position = signal.reindex(df.index).fillna(0).clip(-1, 1).astype(float)
    effective_position = position.shift(1).fillna(0)

    close_ret = df["close"].pct_change().fillna(0.0)
    gross_returns = effective_position * close_ret

    turnover = effective_position.diff().abs().fillna(effective_position.abs())
    if "total_cost_bps" in df.columns:
        total_cost_bps = df["total_cost_bps"].reindex(df.index).fillna(default_total_cost_bps)
    else:
        total_cost_bps = pd.Series(default_total_cost_bps, index=df.index, dtype=float)
    costs = turnover * (total_cost_bps / 10_000)

    net_returns = gross_returns - costs
    equity_curve = (1 + net_returns).cumprod() * initial_capital
    drawdown_curve = equity_curve / equity_curve.cummax() - 1

    trades = _extract_trades(net_returns, effective_position)
    return BacktestResult(
        equity_curve=equity_curve,
        drawdown_curve=drawdown_curve,
        returns=net_returns,
        gross_returns=gross_returns,
        costs=costs,
        effective_position=effective_position,
        trades=trades,
    )
