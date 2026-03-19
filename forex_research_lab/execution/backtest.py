"""Execution model for converting signals into net performance."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class BacktestResult:
    frame: pd.DataFrame
    trades: pd.DataFrame


def _extract_trades(frame: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    active = frame[frame["position"] != 0].copy()
    if active.empty:
        return pd.DataFrame(
            columns=[
                "trade_id",
                "entry_time",
                "exit_time",
                "direction",
                "bars",
                "return_pct",
                "pnl",
            ]
        )

    trade_start = active["position"].ne(active["position"].shift(1))
    active["trade_id"] = trade_start.cumsum()
    trades: list[dict] = []

    for trade_id, group in active.groupby("trade_id", sort=True):
        first_idx = int(group.index[0])
        equity_before = (
            frame.loc[first_idx - 1, "equity"] if first_idx > 0 else initial_capital
        )
        equity_after = float(group["equity"].iloc[-1])
        trades.append(
            {
                "trade_id": int(trade_id),
                "entry_time": group["timestamp"].iloc[0],
                "exit_time": group["timestamp"].iloc[-1],
                "direction": int(group["position"].iloc[0]),
                "bars": int(len(group)),
                "return_pct": float((1.0 + group["net_return"]).prod() - 1.0),
                "pnl": float(equity_after - equity_before),
            }
        )

    return pd.DataFrame(trades)


def run_backtest(
    df: pd.DataFrame,
    signals: pd.Series,
    initial_capital: float = 100_000.0,
) -> BacktestResult:
    """Apply a one-bar execution delay and realistic trading costs."""

    frame = df.copy().reset_index(drop=True)
    frame["signal"] = signals.reindex(df.index).fillna(0).astype(float).reset_index(drop=True)
    frame["position"] = frame["signal"].shift(1).fillna(0.0)
    frame["price_return"] = frame["close"].pct_change().fillna(0.0)
    frame["gross_return"] = frame["position"] * frame["price_return"]
    frame["turnover"] = frame["position"].diff().abs().fillna(frame["position"].abs())

    total_cost_rate = (
        frame.get("spread_bps", 0.0)
        + frame.get("slippage_bps", 0.0)
        + frame.get("commission_bps", 0.0)
    ) / 10_000.0
    frame["transaction_cost"] = frame["turnover"] * total_cost_rate
    frame["net_return"] = frame["gross_return"] - frame["transaction_cost"]
    frame["equity"] = initial_capital * (1.0 + frame["net_return"]).cumprod()
    frame["drawdown"] = frame["equity"] / frame["equity"].cummax() - 1.0

    trades = _extract_trades(frame, initial_capital=initial_capital)
    return BacktestResult(frame=frame, trades=trades)
