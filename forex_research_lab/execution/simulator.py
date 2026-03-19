"""Signal-to-trade execution simulator with transaction costs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from forex_research_lab.data.costs import ExecutionCostModel


@dataclass
class BacktestResult:
    positions: pd.Series
    gross_returns: pd.Series
    net_returns: pd.Series
    cost_returns: pd.Series
    equity_curve: pd.Series
    drawdown_curve: pd.Series
    trades: pd.DataFrame


def _coerce_signal(signal: pd.Series, index: pd.Index) -> pd.Series:
    out = signal.reindex(index).fillna(0.0).astype(float)
    return out.clip(-1.0, 1.0)


def _as_series(value: float | pd.Series, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.Series):
        return value.reindex(index).ffill().bfill()
    return pd.Series(float(value), index=index)


def extract_trades(
    positions: pd.Series,
    gross_returns: pd.Series,
    net_returns: pd.Series,
    cost_returns: pd.Series,
    close: pd.Series,
) -> pd.DataFrame:
    """Build a trade blotter from contiguous non-zero position segments."""

    active = positions != 0
    if not active.any():
        return pd.DataFrame(
            columns=[
                "entry_time",
                "exit_time",
                "direction",
                "bars_held",
                "entry_price",
                "exit_price",
                "gross_return",
                "cost_return",
                "net_return",
            ]
        )

    # A new trade starts whenever executed position changes (including sign flips).
    segment_id = (positions != positions.shift(fill_value=0.0)).cumsum()
    records: list[dict[str, float]] = []

    for _, segment_positions in positions[active].groupby(segment_id[active]):
        entry_time = segment_positions.index[0]
        exit_time = segment_positions.index[-1]
        direction = float(segment_positions.iloc[0])

        seg_gross = gross_returns.loc[entry_time:exit_time]
        seg_net = net_returns.loc[entry_time:exit_time]
        seg_cost = cost_returns.loc[entry_time:exit_time]

        records.append(
            {
                "entry_time": entry_time,
                "exit_time": exit_time,
                "direction": direction,
                "bars_held": float(len(segment_positions)),
                "entry_price": float(close.loc[entry_time]),
                "exit_price": float(close.loc[exit_time]),
                "gross_return": float((1 + seg_gross).prod() - 1),
                "cost_return": float(seg_cost.sum()),
                "net_return": float((1 + seg_net).prod() - 1),
            }
        )

    return pd.DataFrame.from_records(records)


def run_backtest(
    df: pd.DataFrame,
    signal: pd.Series,
    cost_model: ExecutionCostModel,
    initial_capital: float = 100_000.0,
) -> BacktestResult:
    """
    Backtest with strict no-lookahead behavior:
    - signal generated at bar close t
    - position applied starting at bar t+1
    """

    target_signal = _coerce_signal(signal, df.index)
    positions = target_signal.shift(1).fillna(0.0)

    close_returns = df["close"].pct_change().fillna(0.0)
    gross_returns = positions * close_returns

    turnover = positions.diff().abs().fillna(positions.abs())

    spread_pips = _as_series(df["spread_pips"] if "spread_pips" in df.columns else cost_model.spread_pips, df.index)
    pip_size = _as_series(df["pip_size"] if "pip_size" in df.columns else 0.0001, df.index)

    spread_cost = turnover * (spread_pips * pip_size / df["close"]).fillna(0.0)
    slippage_cost = turnover * (cost_model.slippage_bps / 10_000.0)
    commission_cost = turnover * (cost_model.commission_bps / 10_000.0)
    cost_returns = (spread_cost + slippage_cost + commission_cost).fillna(0.0)

    net_returns = (gross_returns - cost_returns).fillna(0.0)
    equity_curve = (1.0 + net_returns).cumprod() * float(initial_capital)
    drawdown_curve = equity_curve / equity_curve.cummax() - 1.0

    trades = extract_trades(
        positions=positions,
        gross_returns=gross_returns,
        net_returns=net_returns,
        cost_returns=cost_returns,
        close=df["close"],
    )

    return BacktestResult(
        positions=positions,
        gross_returns=gross_returns,
        net_returns=net_returns,
        cost_returns=cost_returns,
        equity_curve=equity_curve,
        drawdown_curve=drawdown_curve,
        trades=trades,
    )


def periods_per_year_from_timeframe(timeframe: str) -> int:
    timeframe_u = timeframe.upper()
    if timeframe_u == "H1":
        return 24 * 252
    if timeframe_u == "H4":
        return 6 * 252
    if timeframe_u == "D1":
        return 252
    raise ValueError(f"Unsupported timeframe: {timeframe}")
