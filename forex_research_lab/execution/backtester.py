"""Cost-aware signal execution and equity curve construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    """Container for a single backtest run."""

    position: pd.Series
    executed_position: pd.Series
    asset_returns: pd.Series
    gross_returns: pd.Series
    cost_returns: pd.Series
    net_returns: pd.Series
    equity_curve: pd.Series
    drawdown_curve: pd.Series
    turnover: pd.Series
    trades: pd.DataFrame


def _row_side_cost_return(dataframe: pd.DataFrame, row_position: int, execution_price: float) -> float:
    """Cost per unit of traded position for a row."""
    half_spread = float(dataframe["half_spread"].iloc[row_position]) if "half_spread" in dataframe.columns else 0.0
    slippage_bps = float(dataframe["slippage_bps"].iloc[row_position]) if "slippage_bps" in dataframe.columns else 0.0
    commission_bps = float(dataframe["commission_bps"].iloc[row_position]) if "commission_bps" in dataframe.columns else 0.0
    spread_cost = 0.0 if execution_price == 0.0 else half_spread / execution_price
    return spread_cost + (slippage_bps + commission_bps) / 10_000.0


def extract_trades(
    dataframe: pd.DataFrame,
    executed_position: pd.Series,
    previous_close: float | None = None,
) -> pd.DataFrame:
    """Approximate round-trip trade returns from an executed position series."""
    if dataframe.empty:
        return pd.DataFrame(columns=["entry_time", "exit_time", "direction", "entry_price", "exit_price", "net_return", "holding_period_bars"])

    close = dataframe["close"]
    active_direction = 0.0
    active_trade: dict[str, object] | None = None
    trades: list[dict[str, object]] = []

    for row_number, timestamp in enumerate(dataframe.index):
        execution_price = float(previous_close if row_number == 0 and previous_close is not None else close.iloc[max(row_number - 1, 0)])
        current_direction = float(executed_position.iloc[row_number])
        side_cost = _row_side_cost_return(dataframe, row_number, execution_price)

        if current_direction != active_direction:
            if active_trade is not None:
                entry_price = float(active_trade["entry_price"])
                direction = float(active_trade["direction"])
                exit_price = execution_price
                if direction > 0.0:
                    gross_return = (exit_price / entry_price) - 1.0
                else:
                    gross_return = (entry_price / exit_price) - 1.0
                active_trade["exit_time"] = timestamp
                active_trade["exit_price"] = exit_price
                active_trade["net_return"] = gross_return - float(active_trade["entry_cost"]) - side_cost
                active_trade["holding_period_bars"] = row_number - int(active_trade["entry_bar"])
                trades.append(active_trade)
                active_trade = None

            if current_direction != 0.0:
                active_trade = {
                    "entry_time": timestamp,
                    "entry_price": execution_price,
                    "entry_bar": row_number,
                    "entry_cost": side_cost,
                    "direction": current_direction,
                }

            active_direction = current_direction

    if active_trade is not None:
        final_price = float(close.iloc[-1])
        final_side_cost = _row_side_cost_return(dataframe, len(dataframe) - 1, final_price)
        direction = float(active_trade["direction"])
        entry_price = float(active_trade["entry_price"])
        if direction > 0.0:
            gross_return = (final_price / entry_price) - 1.0
        else:
            gross_return = (entry_price / final_price) - 1.0
        active_trade["exit_time"] = dataframe.index[-1]
        active_trade["exit_price"] = final_price
        active_trade["net_return"] = gross_return - float(active_trade["entry_cost"]) - final_side_cost
        active_trade["holding_period_bars"] = len(dataframe) - int(active_trade["entry_bar"])
        trades.append(active_trade)

    return pd.DataFrame(trades)


def run_backtest(
    dataframe: pd.DataFrame,
    position: pd.Series,
    initial_capital: float = 100_000.0,
    initial_position: float = 0.0,
    previous_close: float | None = None,
) -> BacktestResult:
    """Apply a signal series to market data and produce returns after trading costs."""
    if dataframe.empty:
        empty_series = pd.Series(dtype=float)
        empty_frame = pd.DataFrame()
        return BacktestResult(
            position=empty_series,
            executed_position=empty_series,
            asset_returns=empty_series,
            gross_returns=empty_series,
            cost_returns=empty_series,
            net_returns=empty_series,
            equity_curve=empty_series,
            drawdown_curve=empty_series,
            turnover=empty_series,
            trades=empty_frame,
        )

    aligned_position = position.reindex(dataframe.index).fillna(0.0).clip(-1.0, 1.0).astype(float)
    close = dataframe["close"].astype(float)

    asset_returns = close.pct_change().fillna(0.0)
    if previous_close is not None:
        asset_returns.iloc[0] = (close.iloc[0] / previous_close) - 1.0
    else:
        asset_returns.iloc[0] = 0.0

    executed_position = aligned_position.shift(1).fillna(0.0)
    executed_position.iloc[0] = initial_position
    executed_position = executed_position.clip(-1.0, 1.0)

    prior_executed_position = executed_position.shift(1).fillna(0.0)
    turnover = (executed_position - prior_executed_position).abs()

    half_spread = dataframe["half_spread"] if "half_spread" in dataframe.columns else pd.Series(0.0, index=dataframe.index)
    slippage_bps = dataframe["slippage_bps"] if "slippage_bps" in dataframe.columns else pd.Series(0.0, index=dataframe.index)
    commission_bps = dataframe["commission_bps"] if "commission_bps" in dataframe.columns else pd.Series(0.0, index=dataframe.index)

    spread_cost_return = half_spread.div(close.replace(0.0, np.nan)).fillna(0.0)
    bps_cost_return = (slippage_bps + commission_bps) / 10_000.0
    cost_returns = turnover * (spread_cost_return + bps_cost_return)

    gross_returns = executed_position * asset_returns
    net_returns = gross_returns - cost_returns

    equity_curve = initial_capital * (1.0 + net_returns).cumprod()
    running_peak = equity_curve.cummax().replace(0.0, np.nan)
    drawdown_curve = equity_curve.div(running_peak).fillna(1.0) - 1.0

    trades = extract_trades(dataframe=dataframe, executed_position=executed_position, previous_close=previous_close)

    return BacktestResult(
        position=aligned_position,
        executed_position=executed_position,
        asset_returns=asset_returns,
        gross_returns=gross_returns,
        cost_returns=cost_returns,
        net_returns=net_returns,
        equity_curve=equity_curve,
        drawdown_curve=drawdown_curve,
        turnover=turnover,
        trades=trades,
    )
