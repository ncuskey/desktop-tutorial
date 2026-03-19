"""Tests for walk-forward, bootstrap, param sweep, and experiment tracker."""

import sys
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from data.loader import generate_synthetic_ohlcv
from data.indicators import add_indicators
from strategies.trend import MACrossover
from strategies.mean_reversion import RSIReversal
from execution.engine import ExecutionEngine
from metrics.performance import compute_metrics
from research.walk_forward import WalkForwardEngine, WalkForwardResult
from research.bootstrap import BootstrapEngine, BootstrapResult
from research.param_sweep import ParameterSweep
from research.experiment_tracker import ExperimentTracker


@pytest.fixture
def df_small():
    """Smaller dataset for faster tests."""
    raw = generate_synthetic_ohlcv("EURUSD", start="2020-01-01", end="2021-12-31", freq="h")
    return add_indicators(raw)


@pytest.fixture
def ma_returns(df_small):
    strat = MACrossover()
    engine = ExecutionEngine()
    signals = strat.generate_signals(df_small, {"fast_period": 20, "slow_period": 50})
    result = engine.run(df_small, signals, "EURUSD")
    return result.net_returns, result.trades


class TestWalkForward:
    def test_returns_walk_forward_result(self, df_small):
        wf = WalkForwardEngine(train_bars=800, test_bars=200, step_bars=200)
        result = wf.run(
            df_small, MACrossover(),
            {"fast_period": [10, 20], "slow_period": [40, 60]},
            "EURUSD", verbose=False,
        )
        assert isinstance(result, WalkForwardResult)

    def test_has_windows(self, df_small):
        wf = WalkForwardEngine(train_bars=800, test_bars=200, step_bars=200)
        result = wf.run(
            df_small, MACrossover(),
            {"fast_period": [10, 20], "slow_period": [40, 60]},
            "EURUSD", verbose=False,
        )
        assert len(result.windows) > 0

    def test_oos_returns_not_empty(self, df_small):
        wf = WalkForwardEngine(train_bars=800, test_bars=200, step_bars=200)
        result = wf.run(
            df_small, MACrossover(),
            {"fast_period": [10, 20], "slow_period": [40, 60]},
            "EURUSD", verbose=False,
        )
        assert len(result.oos_returns) > 0

    def test_no_test_train_overlap(self, df_small):
        wf = WalkForwardEngine(train_bars=800, test_bars=200, step_bars=200)
        result = wf.run(
            df_small, MACrossover(),
            {"fast_period": [10, 20], "slow_period": [40, 60]},
            "EURUSD", verbose=False,
        )
        for w in result.windows:
            assert w.test_start > w.train_end, "Test window overlaps with train window"

    def test_summary_has_correct_columns(self, df_small):
        wf = WalkForwardEngine(train_bars=800, test_bars=200, step_bars=200)
        result = wf.run(
            df_small, MACrossover(),
            {"fast_period": [10, 20], "slow_period": [40, 60]},
            "EURUSD", verbose=False,
        )
        summary = result.summary()
        for col in ["train_sharpe", "test_sharpe", "test_cagr"]:
            assert col in summary.columns

    def test_oos_metrics_finite(self, df_small):
        wf = WalkForwardEngine(train_bars=800, test_bars=200, step_bars=200)
        result = wf.run(
            df_small, MACrossover(),
            {"fast_period": [10, 20], "slow_period": [40, 60]},
            "EURUSD", verbose=False,
        )
        if result.oos_metrics:
            assert np.isfinite(result.oos_metrics.sharpe)


class TestBootstrap:
    def test_returns_bootstrap_result(self, ma_returns):
        returns, trades = ma_returns
        bs = BootstrapEngine(n_samples=50, method="block")
        result = bs.run_on_returns(returns, trades, verbose=False)
        assert isinstance(result, BootstrapResult)

    def test_n_samples_correct(self, ma_returns):
        returns, trades = ma_returns
        bs = BootstrapEngine(n_samples=100)
        result = bs.run_on_returns(returns, verbose=False)
        assert result.n_samples == 100
        assert len(result.metric_distributions["sharpe"]) == 100

    def test_percentile_range(self, ma_returns):
        returns, trades = ma_returns
        bs = BootstrapEngine(n_samples=200)
        result = bs.run_on_returns(returns, verbose=False)
        p5 = result.percentile("sharpe", 5)
        p95 = result.percentile("sharpe", 95)
        assert p5 <= p95

    def test_confidence_interval(self, ma_returns):
        returns, trades = ma_returns
        bs = BootstrapEngine(n_samples=200)
        result = bs.run_on_returns(returns, verbose=False)
        lo, hi = result.confidence_interval("sharpe")
        assert lo <= hi

    def test_summary_has_metrics(self, ma_returns):
        returns, trades = ma_returns
        bs = BootstrapEngine(n_samples=50)
        result = bs.run_on_returns(returns, verbose=False)
        summary = result.summary()
        assert "sharpe" in summary.index

    def test_p_value_between_0_and_1(self, ma_returns):
        returns, trades = ma_returns
        bs = BootstrapEngine(n_samples=100)
        result = bs.run_on_returns(returns, verbose=False)
        pv = result.p_value("sharpe", 0.0)
        assert 0.0 <= pv <= 1.0


class TestParameterSweep:
    def test_returns_dataframe(self, df_small):
        sweeper = ParameterSweep(method="grid")
        results = sweeper.run(
            df_small, MACrossover(),
            {"fast_period": [10, 20], "slow_period": [40, 60]},
            "EURUSD", verbose=False,
        )
        assert isinstance(results, pd.DataFrame)

    def test_result_count_matches_grid(self, df_small):
        sweeper = ParameterSweep(method="grid")
        results = sweeper.run(
            df_small, MACrossover(),
            {"fast_period": [10, 20], "slow_period": [40, 60]},
            "EURUSD", verbose=False,
        )
        assert len(results) == 4  # 2 × 2

    def test_sorted_by_sharpe_descending(self, df_small):
        sweeper = ParameterSweep(method="grid", optimise_metric="sharpe")
        results = sweeper.run(
            df_small, MACrossover(),
            {"fast_period": [10, 20], "slow_period": [40, 60]},
            "EURUSD", verbose=False,
        )
        sharpes = results["sharpe"].values
        assert (sharpes[:-1] >= sharpes[1:]).all()

    def test_random_search_respects_n_samples(self, df_small):
        sweeper = ParameterSweep(method="random", n_samples=5)
        results = sweeper.run(
            df_small, MACrossover(),
            {"fast_period": [10, 20, 30], "slow_period": [40, 60, 80, 100]},
            "EURUSD", verbose=False,
        )
        assert len(results) <= 5

    def test_robustness_ratio_in_range(self, df_small):
        sweeper = ParameterSweep(method="grid")
        results = sweeper.run(
            df_small, MACrossover(),
            {"fast_period": [10, 20], "slow_period": [40, 60]},
            "EURUSD", verbose=False,
        )
        ratio = sweeper.robustness_ratio(results)
        assert 0.0 <= ratio <= 1.0


class TestExperimentTracker:
    def test_log_and_retrieve(self):
        tracker = ExperimentTracker()
        m = compute_metrics(pd.Series([0.001] * 100, dtype=float))
        run_id = tracker.log("test_strat", {"p": 10}, "EURUSD", "H4", m)
        assert isinstance(run_id, str)
        df = tracker.results
        assert len(df) == 1
        assert df.iloc[0]["strategy"] == "test_strat"

    def test_best_runs(self):
        tracker = ExperimentTracker()
        for sharpe_val in [0.5, 1.0, -0.3, 0.8]:
            returns = pd.Series([sharpe_val * 0.001] * 500, dtype=float)
            m = compute_metrics(returns)
            tracker.log("strat", {}, "EURUSD", "H4", m)
        best = tracker.best_runs(metric="sharpe", n=2)
        assert len(best) == 2
        assert best.iloc[0]["sharpe"] >= best.iloc[1]["sharpe"]

    def test_sqlite_persistence(self, tmp_path):
        db_path = tmp_path / "test.db"
        tracker = ExperimentTracker(db_path=db_path)
        m = compute_metrics(pd.Series([0.001] * 100, dtype=float))
        tracker.log("strat", {}, "EURUSD", "H4", m)
        assert db_path.exists()
        loaded = tracker.load_from_db()
        assert len(loaded) == 1

    def test_multiple_symbols_filter(self):
        tracker = ExperimentTracker()
        m = compute_metrics(pd.Series([0.001] * 100, dtype=float))
        tracker.log("strat", {}, "EURUSD", "H4", m)
        tracker.log("strat", {}, "GBPUSD", "H4", m)
        eu_best = tracker.best_runs(symbol="EURUSD")
        assert all(eu_best["symbol"] == "EURUSD")
