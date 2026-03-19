from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from forex_research_lab.data import (
    add_basic_indicators,
    attach_cost_model,
    ensure_sample_data,
    load_ohlcv_directory,
    prepare_multi_timeframe,
)
from forex_research_lab.execution import run_backtest
from forex_research_lab.metrics import compute_performance_metrics
from forex_research_lab.research import run_walk_forward
from forex_research_lab.strategies import MovingAverageCrossoverStrategy, RSIReversalStrategy


class ForexResearchLabSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        sample_dir = Path(self.temp_dir.name) / "sample_data"
        ensure_sample_data(sample_dir, symbols=("EURUSD",), periods=24 * 180)

        raw = load_ohlcv_directory(sample_dir)
        resampled = prepare_multi_timeframe(raw, timeframes=("H1", "H4", "D1"))
        self.dataframe = attach_cost_model(add_basic_indicators(resampled["EURUSD"]["H4"]), symbol="EURUSD")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_ma_crossover_backtest_smoke(self) -> None:
        strategy = MovingAverageCrossoverStrategy()
        signals = strategy.generate_signals(self.dataframe, params={"short_window": 10, "long_window": 50})
        result = run_backtest(self.dataframe, signals)
        metrics = compute_performance_metrics(result.net_returns, result.equity_curve, result.trades)

        self.assertFalse(result.equity_curve.empty)
        self.assertIn("Sharpe", metrics)
        self.assertIn("Trade Count", metrics)

    def test_walk_forward_smoke(self) -> None:
        strategy = RSIReversalStrategy()
        result = run_walk_forward(
            dataframe=self.dataframe,
            strategy=strategy,
            parameter_space={"window": [10, 14], "entry_threshold": [25, 30]},
            train_size=180,
            test_size=60,
            step_size=60,
            symbol="EURUSD",
            timeframe="H4",
        )

        self.assertFalse(result.split_summary.empty)
        self.assertFalse(result.equity_curve.empty)
        self.assertIn("Sharpe", result.aggregate_metrics)


if __name__ == "__main__":
    unittest.main()
