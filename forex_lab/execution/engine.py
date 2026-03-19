"""
Execution engine: converts a signal series into a realistic equity curve.

Design principles:
- Signals are applied at the *next* bar's open (no lookahead execution).
- Spread is paid on every position change (half on open, half on close).
- Slippage is paid on every fill.
- Commission is paid on every round-trip.
- Positions are restricted to {-1, 0, +1} (unit position sizing).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data.costs import CostModel, DEFAULT_COSTS


@dataclass
class BacktestResult:
    """Container for a single backtest run."""

    symbol: str
    strategy: str
    params: dict
    signals: pd.Series           # raw signal series
    positions: pd.Series         # lagged (executed) positions
    gross_returns: pd.Series     # bar returns before costs
    net_returns: pd.Series       # bar returns after costs
    equity_curve: pd.Series      # cumulative equity (starts at 1.0)
    drawdown: pd.Series          # drawdown series (0 to -1)
    trades: pd.DataFrame         # individual trade log
    cost_drag: float             # total cost drag as fraction of gross P&L

    @property
    def n_trades(self) -> int:
        return len(self.trades)


class ExecutionEngine:
    """Converts strategy signals into a realistic backtest result.

    Parameters
    ----------
    cost_model:
        Transaction cost model.  Defaults to symbol-specific standard costs.
    execution_lag:
        Bars of lag between signal generation and execution.
        Default 1 (signal on close of bar t → executed on open of bar t+1,
        approximated as close of bar t+1 for vectorised simplicity).
    """

    def __init__(
        self,
        cost_model: CostModel | None = None,
        execution_lag: int = 1,
    ) -> None:
        self.cost_model = cost_model
        self.execution_lag = execution_lag

    def run(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        symbol: str = "EURUSD",
        strategy_name: str = "unknown",
        params: dict | None = None,
    ) -> BacktestResult:
        """Execute a strategy signal series against price data.

        Parameters
        ----------
        df:
            OHLCV DataFrame (datetime index, sorted ascending).
        signals:
            Position signal series aligned to df.index (values: -1, 0, +1).
        symbol:
            FX pair identifier, used to look up default costs.
        strategy_name:
            Human-readable strategy label.
        params:
            Strategy parameters (stored in result only).

        Returns
        -------
        BacktestResult
        """
        cost_model = self.cost_model or DEFAULT_COSTS.get(
            symbol, CostModel(symbol=symbol)
        )
        params = params or {}

        # --- 1. Lag signals (execution delay) ---
        positions = signals.shift(self.execution_lag).fillna(0).astype("int8")

        # --- 2. Gross bar returns (log returns of close prices) ---
        log_ret = np.log(df["close"] / df["close"].shift(1)).fillna(0)
        gross_returns = (positions.shift(1).fillna(0) * log_ret).fillna(0)

        # --- 3. Transaction costs on position changes ---
        pos_change = positions.diff().abs().fillna(0)
        cost_per_bar = pos_change * cost_model.cost_as_return(df["close"].mean())
        net_returns = gross_returns - cost_per_bar

        # --- 4. Equity curve ---
        equity_curve = (1 + net_returns).cumprod()
        equity_curve = equity_curve / equity_curve.iloc[0]

        # --- 5. Drawdown ---
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max

        # --- 6. Trade log ---
        trades = self._build_trade_log(df, positions, net_returns, cost_model)

        # --- 7. Cost drag ---
        total_gross = gross_returns.sum()
        total_cost = cost_per_bar.sum()
        cost_drag = total_cost / max(abs(total_gross), 1e-10)

        return BacktestResult(
            symbol=symbol,
            strategy=strategy_name,
            params=params,
            signals=signals,
            positions=positions,
            gross_returns=gross_returns,
            net_returns=net_returns,
            equity_curve=equity_curve,
            drawdown=drawdown,
            trades=trades,
            cost_drag=float(cost_drag),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_trade_log(
        self,
        df: pd.DataFrame,
        positions: pd.Series,
        net_returns: pd.Series,
        cost_model: CostModel,
    ) -> pd.DataFrame:
        """Build a trade-level log from the position series."""
        trades = []
        current_pos = 0
        entry_date = None
        entry_price = None
        trade_returns: list[float] = []

        for i, (date, pos) in enumerate(positions.items()):
            pos = int(pos)
            if pos != current_pos:
                if current_pos != 0 and entry_date is not None:
                    # Close existing trade
                    close_price = df["close"].iloc[i - 1] if i > 0 else df["close"].iloc[0]
                    pnl = sum(trade_returns)
                    duration = i - positions.index.get_loc(entry_date)
                    trades.append(
                        {
                            "entry_date": entry_date,
                            "exit_date": date,
                            "direction": current_pos,
                            "entry_price": entry_price,
                            "exit_price": float(close_price),
                            "pnl": pnl,
                            "duration_bars": duration,
                        }
                    )
                    trade_returns = []

                if pos != 0:
                    entry_date = date
                    entry_price = float(df["close"].iloc[i])
                current_pos = pos

            if current_pos != 0:
                trade_returns.append(float(net_returns.iloc[i]))

        if current_pos != 0 and entry_date is not None:
            trades.append(
                {
                    "entry_date": entry_date,
                    "exit_date": positions.index[-1],
                    "direction": current_pos,
                    "entry_price": entry_price,
                    "exit_price": float(df["close"].iloc[-1]),
                    "pnl": sum(trade_returns),
                    "duration_bars": len(positions) - positions.index.get_loc(entry_date),
                }
            )

        if not trades:
            return pd.DataFrame(
                columns=[
                    "entry_date", "exit_date", "direction", "entry_price",
                    "exit_price", "pnl", "duration_bars",
                ]
            )
        return pd.DataFrame(trades)
