"""Forex Strategy Research Lab package."""

from forex_research_lab.data.costs import CostModel, attach_cost_model, get_default_cost_model
from forex_research_lab.data.indicators import add_basic_indicators
from forex_research_lab.data.loader import load_ohlcv_csv, load_ohlcv_directory, prepare_multi_timeframe, resample_ohlcv
from forex_research_lab.execution.backtester import BacktestResult, run_backtest
from forex_research_lab.metrics.performance import compute_performance_metrics

__all__ = [
    "BacktestResult",
    "CostModel",
    "add_basic_indicators",
    "attach_cost_model",
    "compute_performance_metrics",
    "get_default_cost_model",
    "load_ohlcv_csv",
    "load_ohlcv_directory",
    "prepare_multi_timeframe",
    "resample_ohlcv",
    "run_backtest",
]
