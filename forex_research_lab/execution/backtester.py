"""Signal execution and portfolio construction."""

from __future__ import annotations

from typing import Any

import pandas as pd

from forex_research_lab.metrics.performance import compute_metrics
from forex_research_lab.types import BacktestResult


def _extract_trades(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    active_direction = 0.0
    entry_time = None
    entry_price = None
    cumulative_return = 0.0

    for row in frame.itertuples(index=False):
        if active_direction == 0.0 and row.position != 0.0:
            active_direction = row.position
            entry_time = row.timestamp
            entry_price = row.close
            cumulative_return = row.net_return
            continue

        if active_direction != 0.0:
            cumulative_return = (1 + cumulative_return) * (1 + row.net_return) - 1

            if row.position != active_direction:
                records.append(
                    {
                        "symbol": symbol,
                        "entry_time": entry_time,
                        "exit_time": row.timestamp,
                        "direction": active_direction,
                        "entry_price": entry_price,
                        "exit_price": row.close,
                        "trade_return": cumulative_return,
                    }
                )

                if row.position != 0.0:
                    active_direction = row.position
                    entry_time = row.timestamp
                    entry_price = row.close
                    cumulative_return = row.net_return
                else:
                    active_direction = 0.0
                    entry_time = None
                    entry_price = None
                    cumulative_return = 0.0

    if active_direction != 0.0 and entry_time is not None:
        last_row = frame.iloc[-1]
        records.append(
            {
                "symbol": symbol,
                "entry_time": entry_time,
                "exit_time": last_row["timestamp"],
                "direction": active_direction,
                "entry_price": entry_price,
                "exit_price": last_row["close"],
                "trade_return": cumulative_return,
            }
        )

    return pd.DataFrame.from_records(records)


def backtest_signal(
    df: pd.DataFrame,
    signal: pd.Series,
    *,
    timeframe: str,
    initial_capital: float = 100_000.0,
    symbol: str | None = None,
) -> BacktestResult:
    """Backtest a single signal stream without lookahead bias."""

    required = {"timestamp", "close", "spread_bps", "commission_bps", "slippage_bps"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Execution data is missing columns: {sorted(missing)}")

    local = df.sort_values("timestamp").copy()
    local["signal"] = signal.reindex(local.index).fillna(0.0).astype(float)
    local["position"] = local["signal"].shift(1).fillna(0.0)
    local["asset_return"] = local["close"].pct_change().fillna(0.0)
    local["gross_return"] = local["position"] * local["asset_return"]

    turnover = local["position"].diff().abs().fillna(local["position"].abs())
    cost_bps = local["spread_bps"] + local["commission_bps"] + local["slippage_bps"]
    local["cost_return"] = turnover * (cost_bps / 10_000)
    local["net_return"] = local["gross_return"] - local["cost_return"]
    local["equity"] = initial_capital * (1 + local["net_return"]).cumprod()
    local["drawdown"] = local["equity"] / local["equity"].cummax() - 1

    trade_symbol = symbol or str(local.get("symbol", pd.Series(["UNKNOWN"])).iloc[0])
    trades = _extract_trades(local, trade_symbol)
    metrics = compute_metrics(
        returns=local["net_return"],
        trades=trades,
        timeframe=timeframe,
        initial_capital=initial_capital,
        equity=local["equity"],
    )
    return BacktestResult(frame=local, trades=trades, metrics=metrics)


def evaluate_strategy(
    df: pd.DataFrame,
    strategy: Any,
    params: dict[str, Any],
    *,
    timeframe: str,
    initial_capital: float = 100_000.0,
) -> BacktestResult:
    """Evaluate a strategy across one or more symbols with equal-weight aggregation."""

    required = {"timestamp", "symbol", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Strategy evaluation data is missing columns: {sorted(missing)}")

    symbol_results: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []

    for symbol, frame in df.groupby("symbol", sort=False):
        local = frame.sort_values("timestamp").copy().reset_index(drop=True)
        signal = strategy.generate_signals(local, params)
        result = backtest_signal(
            local,
            signal,
            timeframe=timeframe,
            initial_capital=1.0,
            symbol=str(symbol),
        )
        symbol_frame = result.frame[["timestamp", "net_return", "equity", "drawdown"]].copy()
        symbol_frame.rename(
            columns={
                "net_return": f"{symbol}_return",
                "equity": f"{symbol}_equity",
                "drawdown": f"{symbol}_drawdown",
            },
            inplace=True,
        )
        symbol_results.append(symbol_frame)
        trade_frames.append(result.trades)

    portfolio = symbol_results[0]
    for frame in symbol_results[1:]:
        portfolio = portfolio.merge(frame, on="timestamp", how="outer")

    return_columns = [column for column in portfolio.columns if column.endswith("_return")]
    portfolio = portfolio.sort_values("timestamp").reset_index(drop=True)
    portfolio["portfolio_return"] = portfolio[return_columns].mean(axis=1).fillna(0.0)
    portfolio["equity"] = initial_capital * (1 + portfolio["portfolio_return"]).cumprod()
    portfolio["drawdown"] = portfolio["equity"] / portfolio["equity"].cummax() - 1

    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    metrics = compute_metrics(
        returns=portfolio["portfolio_return"],
        trades=trades,
        timeframe=timeframe,
        initial_capital=initial_capital,
        equity=portfolio["equity"],
    )
    return BacktestResult(frame=portfolio, trades=trades, metrics=metrics)
