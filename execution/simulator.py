from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data.costs import CostModel


@dataclass
class BacktestResult:
    returns: pd.Series
    gross_returns: pd.Series
    equity: pd.Series
    drawdown: pd.Series
    position: pd.Series
    trades: pd.DataFrame


def _extract_trades(returns: pd.Series, position: pd.Series) -> pd.DataFrame:
    trades: list[dict] = []
    current_side = 0
    start_idx = None
    acc_return = 0.0

    for i in range(len(position)):
        side = int(position.iloc[i])
        r = float(returns.iloc[i])

        if current_side == 0 and side != 0:
            current_side = side
            start_idx = position.index[i]
            acc_return = r
            continue

        if current_side != 0 and side == current_side:
            acc_return += r
            continue

        if current_side != 0 and side != current_side:
            end_idx = position.index[i]
            trades.append(
                {
                    "entry_time": start_idx,
                    "exit_time": end_idx,
                    "side": current_side,
                    "trade_return": acc_return,
                }
            )
            current_side = side
            start_idx = position.index[i] if side != 0 else None
            acc_return = r if side != 0 else 0.0

    if current_side != 0 and start_idx is not None:
        trades.append(
            {
                "entry_time": start_idx,
                "exit_time": position.index[-1],
                "side": current_side,
                "trade_return": acc_return,
            }
        )

    return pd.DataFrame(trades)


def run_backtest(
    df: pd.DataFrame,
    signal: pd.Series,
    cost_model: CostModel,
    initial_capital: float = 100_000.0,
) -> BacktestResult:
    if len(df) != len(signal):
        raise ValueError("Signal length must match df length.")

    close = df["close"]
    raw_position = signal.reindex(df.index).fillna(0).clip(-1, 1)
    # No lookahead: signal at t is executed for return at t+1.
    position = raw_position.shift(1).fillna(0)

    price_returns = close.pct_change().fillna(0.0)
    gross_returns = position * price_returns

    position_change = raw_position.diff().abs().fillna(raw_position.abs())
    one_way_cost = (
        cost_model.spread_bps + cost_model.slippage_bps + cost_model.commission_bps
    ) / 10_000.0
    costs = position_change * one_way_cost
    net_returns = gross_returns - costs

    equity = initial_capital * (1.0 + net_returns).cumprod()
    running_max = equity.cummax()
    drawdown = (equity / running_max) - 1.0

    trades = _extract_trades(net_returns, raw_position)
    return BacktestResult(
        returns=net_returns,
        gross_returns=gross_returns,
        equity=equity,
        drawdown=drawdown,
        position=raw_position,
        trades=trades,
    )
