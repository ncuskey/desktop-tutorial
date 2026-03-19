"""Convert signals to trades with spread, slippage, commission; compute equity curve."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from forex_lab.data.cost_model import CostModel


@dataclass
class Trade:
    """Single trade record."""

    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int  # 1 long, -1 short
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    costs: float
    bars_held: int


class ExecutionEngine:
    """
    Convert position signals to trades with realistic costs.
    """

    def __init__(
        self,
        cost_model: Optional[CostModel] = None,
        initial_capital: float = 100_000.0,
        lot_size: float = 100_000.0,
    ):
        self.cost_model = cost_model or CostModel()
        self.initial_capital = initial_capital
        self.lot_size = lot_size

    def run(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        size: float = 1.0,
    ) -> tuple[pd.Series, list[Trade], pd.DataFrame]:
        """
        Execute trades from signals.

        Args:
            df: OHLCV with index
            signals: position series (-1, 0, +1)
            size: position size in lots (1.0 = 1 standard lot)

        Returns:
            equity_curve: Series of equity values
            trades: list of Trade objects
            equity_df: DataFrame with equity, drawdown, returns
        """
        signals = signals.reindex(df.index).ffill().fillna(0).astype(int)
        close = df["close"].values
        n = len(df)

        trades: list[Trade] = []
        position = 0
        entry_price = 0.0
        entry_idx = 0

        equity = np.zeros(n)
        equity[0] = self.initial_capital

        for i in range(1, n):
            prev_sig = signals.iloc[i - 1]
            curr_sig = signals.iloc[i]

            # Position change
            if curr_sig != prev_sig:
                if position != 0:
                    # Close existing
                    exit_price = close[i]
                    exit_price_adj = self._apply_exit_cost(exit_price, position, size)
                    pnl = position * (exit_price_adj - entry_price) * self.lot_size * size
                    costs = self._trade_costs(entry_price, exit_price_adj, position, size)
                    pnl_net = pnl - costs

                    trades.append(
                        Trade(
                            entry_time=df.index[entry_idx],
                            exit_time=df.index[i],
                            direction=position,
                            entry_price=entry_price,
                            exit_price=exit_price_adj,
                            size=size,
                            pnl=pnl_net,
                            pnl_pct=100 * pnl_net / (entry_price * self.lot_size * size) if size else 0,
                            costs=costs,
                            bars_held=i - entry_idx,
                        )
                    )
                    equity[i] = equity[i - 1] + pnl_net
                else:
                    equity[i] = equity[i - 1]

                # Open new if any
                if curr_sig != 0:
                    position = curr_sig
                    entry_price = self._apply_entry_cost(close[i], position, size)
                    entry_idx = i
                    equity[i] = equity[i]
                else:
                    position = 0
            else:
                # Mark-to-market open position
                if position != 0:
                    unrealized = position * (close[i] - entry_price) * size
                    equity[i] = equity[i - 1] + position * (close[i] - close[i - 1]) * self.lot_size * size
                else:
                    equity[i] = equity[i - 1]
        # Handle open position at end
        if position != 0:
            exit_price = close[-1]
            exit_price_adj = self._apply_exit_cost(exit_price, position, size)
            pnl = position * (exit_price_adj - entry_price) * self.lot_size * size
            costs = self._trade_costs(entry_price, exit_price_adj, position, size)
            pnl_net = pnl - costs
            trades.append(
                Trade(
                    entry_time=df.index[entry_idx],
                    exit_time=df.index[-1],
                    direction=position,
                    entry_price=entry_price,
                    exit_price=exit_price_adj,
                    size=size,
                    pnl=pnl_net,
                    pnl_pct=100 * pnl_net / (entry_price * self.lot_size * size) if size else 0,
                    costs=costs,
                    bars_held=n - 1 - entry_idx,
                )
            )
            equity[-1] = equity[-2] + pnl_net

        equity_curve = pd.Series(equity, index=df.index)
        equity_curve = equity_curve.ffill().fillna(self.initial_capital)

        returns = equity_curve.pct_change().fillna(0)
        cummax = equity_curve.cummax()
        drawdown = (equity_curve - cummax) / cummax.replace(0, np.nan)
        drawdown = drawdown.fillna(0)

        equity_df = pd.DataFrame(
            {
                "equity": equity_curve,
                "returns": returns,
                "drawdown": drawdown,
            },
            index=df.index,
        )

        return equity_curve, trades, equity_df

    def _apply_entry_cost(self, price: float, direction: int, size: float = 1.0) -> float:
        notional = price * self.lot_size * size
        return self.cost_model.apply_costs(price, direction, notional)

    def _apply_exit_cost(self, price: float, direction: int, size: float = 1.0) -> float:
        notional = price * self.lot_size * size
        return self.cost_model.apply_costs(price, -direction, notional)

    def _trade_costs(
        self,
        entry: float,
        exit: float,
        direction: int,
        size: float,
    ) -> float:
        notional = entry * self.lot_size * size
        bps = self.cost_model.cost_per_trade_bps(notional)
        cost_pct = bps / 1e4
        return notional * cost_pct
