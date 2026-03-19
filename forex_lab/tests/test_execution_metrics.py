"""Tests for execution engine and performance metrics."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from data.loader import generate_synthetic_ohlcv
from data.indicators import add_indicators
from data.costs import CostModel
from strategies.trend import MACrossover
from strategies.mean_reversion import RSIReversal
from execution.engine import ExecutionEngine, BacktestResult
from metrics.performance import compute_metrics, MetricsResult


@pytest.fixture
def df():
    raw = generate_synthetic_ohlcv("EURUSD", start="2020-01-01", end="2022-12-31", freq="h")
    return add_indicators(raw)


@pytest.fixture
def engine():
    return ExecutionEngine()


@pytest.fixture
def ma_result(df, engine):
    strat = MACrossover()
    signals = strat.generate_signals(df, {"fast_period": 20, "slow_period": 50})
    return engine.run(df, signals, "EURUSD", "ma_crossover", {"fast_period": 20, "slow_period": 50})


class TestExecutionEngine:
    def test_result_type(self, ma_result):
        assert isinstance(ma_result, BacktestResult)

    def test_equity_starts_at_one(self, ma_result):
        assert abs(ma_result.equity_curve.iloc[0] - 1.0) < 0.01

    def test_equity_always_positive(self, ma_result):
        assert (ma_result.equity_curve > 0).all()

    def test_drawdown_non_positive(self, ma_result):
        assert (ma_result.drawdown <= 0).all()

    def test_positions_valid_values(self, ma_result):
        assert set(ma_result.positions.unique()).issubset({-1, 0, 1})

    def test_net_returns_less_than_gross(self, ma_result):
        # Net total return should be <= gross (costs reduce returns)
        assert ma_result.net_returns.sum() <= ma_result.gross_returns.sum()

    def test_cost_drag_non_negative(self, ma_result):
        assert ma_result.cost_drag >= 0

    def test_trade_log_has_correct_columns(self, ma_result):
        if len(ma_result.trades) > 0:
            required_cols = {"entry_date", "exit_date", "direction", "pnl"}
            assert required_cols.issubset(set(ma_result.trades.columns))

    def test_execution_lag(self, df, engine):
        strat = MACrossover()
        signals = strat.generate_signals(df, {"fast_period": 20, "slow_period": 50})
        result_lag1 = engine.run(df, signals, "EURUSD")
        engine_lag2 = ExecutionEngine(execution_lag=2)
        result_lag2 = engine_lag2.run(df, signals, "EURUSD")
        # Positions with lag=1 and lag=2 should differ
        assert not result_lag1.positions.equals(result_lag2.positions)

    def test_zero_signals_flat_equity(self, df, engine):
        zero_signals = pd.Series(0, index=df.index, dtype="int8")
        result = engine.run(df, zero_signals, "EURUSD", "flat")
        # With no positions, equity should be flat at 1.0
        assert (result.equity_curve == 1.0).all()
        assert len(result.trades) == 0


class TestMetrics:
    def test_metrics_result_type(self, ma_result):
        m = compute_metrics(ma_result.net_returns, ma_result.trades)
        assert isinstance(m, MetricsResult)

    def test_sharpe_is_finite(self, ma_result):
        m = compute_metrics(ma_result.net_returns)
        assert np.isfinite(m.sharpe)

    def test_max_drawdown_non_positive(self, ma_result):
        m = compute_metrics(ma_result.net_returns)
        assert m.max_drawdown <= 0

    def test_win_rate_bounded(self, ma_result):
        m = compute_metrics(ma_result.net_returns, ma_result.trades)
        assert 0 <= m.win_rate <= 1

    def test_empty_returns_returns_zeros(self):
        m = compute_metrics(pd.Series(dtype=float))
        assert m.sharpe == 0.0
        assert m.cagr == 0.0

    def test_positive_returns_positive_cagr(self):
        # Monotonically increasing returns → positive CAGR
        returns = pd.Series([0.001] * 1000, index=pd.date_range("2020-01-01", periods=1000, freq="h"))
        m = compute_metrics(returns)
        assert m.cagr > 0
        assert m.sharpe > 0

    def test_negative_returns_negative_cagr(self):
        returns = pd.Series([-0.001] * 1000, index=pd.date_range("2020-01-01", periods=1000, freq="h"))
        m = compute_metrics(returns)
        assert m.cagr < 0

    def test_to_dict_contains_all_fields(self, ma_result):
        m = compute_metrics(ma_result.net_returns)
        d = m.to_dict()
        for field in ["cagr", "sharpe", "sortino", "max_drawdown", "profit_factor",
                       "win_rate", "expectancy", "n_trades", "calmar"]:
            assert field in d

    def test_calmar_ratio_consistency(self):
        returns = pd.Series([0.001] * 2000, index=pd.date_range("2020-01-01", periods=2000, freq="h"))
        m = compute_metrics(returns)
        # Calmar = CAGR / |MaxDD|
        if m.max_drawdown != 0:
            expected_calmar = m.cagr / abs(m.max_drawdown)
            assert abs(m.calmar - expected_calmar) < 0.001
