"""Execution and simulation engine."""

from .simulator import BacktestResult, periods_per_year_from_timeframe, run_backtest

__all__ = ["BacktestResult", "run_backtest", "periods_per_year_from_timeframe"]
