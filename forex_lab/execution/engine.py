"""Execution engine — converts signals to trades, applies costs, builds equity curve."""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Any


class ExecutionEngine:
    """Simulates trade execution with realistic costs.

    Costs are only applied when the position changes (a new trade is entered).
    """

    def __init__(
        self,
        spread: float = 0.0002,
        slippage_bps: float = 1.0,
        commission_bps: float = 0.5,
        initial_capital: float = 100_000.0,
    ):
        self.spread = spread
        self.slippage_bps = slippage_bps
        self.commission_bps = commission_bps
        self.initial_capital = initial_capital

    def run(self, df: pd.DataFrame, signals: pd.Series) -> pd.DataFrame:
        """Execute a signal series against OHLCV data.

        Returns a DataFrame with columns:
          position, returns, costs, net_returns, equity, drawdown
        """
        result = pd.DataFrame(index=df.index)
        result["position"] = signals.reindex(df.index).fillna(0.0)

        pct_returns = df["close"].pct_change().fillna(0.0)
        result["gross_returns"] = result["position"].shift(1).fillna(0.0) * pct_returns

        position_changes = result["position"].diff().abs().fillna(0.0)
        has_trade = position_changes > 0

        cost_per_unit = (
            self.spread
            + df["close"] * self.slippage_bps / 10000
            + df["close"] * self.commission_bps / 10000
        )
        result["costs"] = np.where(has_trade, cost_per_unit / df["close"], 0.0)
        result["net_returns"] = result["gross_returns"] - result["costs"]

        result["equity"] = self.initial_capital * (1 + result["net_returns"]).cumprod()

        running_max = result["equity"].cummax()
        result["drawdown"] = (result["equity"] - running_max) / running_max

        result["trade_id"] = has_trade.cumsum()
        result["trade_entry"] = has_trade

        return result

    def extract_trades(self, execution_result: pd.DataFrame) -> pd.DataFrame:
        """Extract individual trade records from execution results."""
        trades = []
        df = execution_result
        position_changes = df["position"].diff().fillna(0.0)
        entries = df.index[position_changes != 0]

        for i, entry_time in enumerate(entries):
            if i + 1 < len(entries):
                exit_time = entries[i + 1]
            else:
                exit_time = df.index[-1]

            mask = (df.index >= entry_time) & (df.index < exit_time)
            if i + 1 >= len(entries):
                mask = (df.index >= entry_time) & (df.index <= exit_time)

            segment = df.loc[mask]
            if len(segment) == 0:
                continue

            trade_return = segment["net_returns"].sum()
            direction = df.loc[entry_time, "position"]

            trades.append(
                {
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "direction": direction,
                    "bars_held": len(segment),
                    "return": trade_return,
                }
            )

        return pd.DataFrame(trades) if trades else pd.DataFrame(
            columns=["entry_time", "exit_time", "direction", "bars_held", "return"]
        )
