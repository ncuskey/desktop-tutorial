"""Smoke tests for the Forex Strategy Research Lab prototype."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from forex_research_lab.data import (  # noqa: E402
    attach_cost_model,
    compute_basic_indicators,
    generate_mock_ohlcv,
    resample_ohlcv,
)
from forex_research_lab.execution import evaluate_strategy  # noqa: E402
from forex_research_lab.metrics import compute_metrics  # noqa: E402
from forex_research_lab.research import WalkForwardEngine  # noqa: E402
from forex_research_lab.strategies import (  # noqa: E402
    MovingAverageCrossoverStrategy,
    RSIReversalStrategy,
)


class ForexResearchLabTests(unittest.TestCase):
    def setUp(self) -> None:
        raw = generate_mock_ohlcv(periods=24 * 120)
        self.h1 = compute_basic_indicators(attach_cost_model(raw))

    def test_resampling_preserves_symbols(self) -> None:
        h4 = resample_ohlcv(self.h1, "4h")
        self.assertGreater(len(h4), 0)
        self.assertEqual(set(h4["symbol"].unique()), {"EURUSD", "GBPUSD", "USDJPY", "AUDUSD"})

    def test_strategy_backtest_generates_metrics(self) -> None:
        result = evaluate_strategy(
            self.h1,
            MovingAverageCrossoverStrategy(),
            {"fast_window": 10, "slow_window": 50},
            timeframe="H1",
        )
        self.assertIn("sharpe", result.metrics)
        self.assertIn("equity", result.frame.columns)
        self.assertGreaterEqual(result.metrics["trade_count"], 0.0)

    def test_walk_forward_runs_without_lookahead_errors(self) -> None:
        engine = WalkForwardEngine(train_bars=24 * 60, test_bars=24 * 20, step_bars=24 * 20)
        result = engine.run(
            self.h1,
            RSIReversalStrategy(),
            {
                "window": [10, 14],
                "oversold": [25, 30],
                "overbought": [70, 75],
                "exit_level": [45, 50],
            },
            timeframe="H1",
        )
        metrics = compute_metrics(
            returns=result.aggregated_frame["portfolio_return"],
            trades=result.aggregated_trades,
            timeframe="H1",
            initial_capital=100_000.0,
            equity=result.aggregated_frame["equity"],
        )
        self.assertGreater(len(result.split_metrics), 0)
        self.assertIn("sharpe", metrics)


if __name__ == "__main__":
    unittest.main()
