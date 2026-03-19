"""
Walk-forward analysis engine.

Methodology:
  1. Divide the full history into overlapping train/test windows.
  2. On each train window: optimise strategy parameters via grid search.
  3. On each test window: evaluate the best parameters out-of-sample.
  4. Aggregate out-of-sample results across all windows.

This avoids look-ahead bias and overfitting to a single in-sample period.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

from data.indicators import add_indicators
from execution.engine import ExecutionEngine, BacktestResult
from metrics.performance import compute_metrics, MetricsResult


@dataclass
class WindowResult:
    """Results for a single walk-forward window."""

    window_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_params: dict
    train_metrics: MetricsResult
    test_metrics: MetricsResult
    test_equity: pd.Series
    test_returns: pd.Series


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward results."""

    windows: list[WindowResult] = field(default_factory=list)
    oos_returns: pd.Series = field(default_factory=pd.Series)
    oos_equity: pd.Series = field(default_factory=pd.Series)
    oos_metrics: MetricsResult | None = None
    param_stability: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self) -> pd.DataFrame:
        rows = []
        for w in self.windows:
            row = {
                "window": w.window_id,
                "train_start": w.train_start,
                "train_end": w.train_end,
                "test_start": w.test_start,
                "test_end": w.test_end,
                "train_sharpe": w.train_metrics.sharpe,
                "test_sharpe": w.test_metrics.sharpe,
                "test_cagr": w.test_metrics.cagr,
                "test_max_dd": w.test_metrics.max_drawdown,
            }
            row.update({f"param_{k}": v for k, v in w.best_params.items()})
            rows.append(row)
        return pd.DataFrame(rows)


class WalkForwardEngine:
    """Rolling walk-forward optimisation and testing engine.

    Parameters
    ----------
    train_bars:
        Number of bars in each training window.
    test_bars:
        Number of bars in each out-of-sample test window.
    step_bars:
        Step size between consecutive windows (default = test_bars).
    optimise_metric:
        Metric to maximise during in-sample optimisation ('sharpe', 'cagr',
        'sortino', 'calmar').
    """

    def __init__(
        self,
        train_bars: int = 2000,
        test_bars: int = 500,
        step_bars: int | None = None,
        optimise_metric: str = "sharpe",
    ) -> None:
        self.train_bars = train_bars
        self.test_bars = test_bars
        self.step_bars = step_bars or test_bars
        self.optimise_metric = optimise_metric
        self._engine = ExecutionEngine()

    def run(
        self,
        df: pd.DataFrame,
        strategy,
        param_grid: dict[str, list],
        symbol: str = "EURUSD",
        verbose: bool = True,
    ) -> WalkForwardResult:
        """Run walk-forward analysis.

        Parameters
        ----------
        df:
            Full OHLCV DataFrame (with indicators already added, or raw).
        strategy:
            Strategy instance implementing generate_signals().
        param_grid:
            Dict mapping parameter names to lists of candidate values.
        symbol:
            FX pair symbol.
        verbose:
            Show progress bar.
        """
        df = add_indicators(df)
        param_combinations = list(
            dict(zip(param_grid.keys(), combo))
            for combo in itertools.product(*param_grid.values())
        )

        n = len(df)
        windows: list[WindowResult] = []
        window_id = 0

        positions_iter = range(
            0,
            n - self.train_bars - self.test_bars + 1,
            self.step_bars,
        )
        iter_list = list(positions_iter)

        if verbose:
            iter_list = tqdm(iter_list, desc=f"Walk-forward [{strategy.name}]")

        for start_idx in iter_list:
            train_slice = df.iloc[start_idx : start_idx + self.train_bars]
            test_slice = df.iloc[
                start_idx + self.train_bars : start_idx + self.train_bars + self.test_bars
            ]

            if len(train_slice) < self.train_bars // 2 or len(test_slice) < 10:
                continue

            # --- Optimise on train ---
            best_params, train_metrics, _ = self._optimise(
                train_slice, strategy, param_combinations, symbol
            )

            # --- Evaluate on test ---
            test_signals = strategy.generate_signals(test_slice, best_params)
            test_result = self._engine.run(
                test_slice, test_signals, symbol, strategy.name, best_params
            )
            test_metrics = compute_metrics(test_result.net_returns, test_result.trades)

            windows.append(
                WindowResult(
                    window_id=window_id,
                    train_start=train_slice.index[0],
                    train_end=train_slice.index[-1],
                    test_start=test_slice.index[0],
                    test_end=test_slice.index[-1],
                    best_params=best_params,
                    train_metrics=train_metrics,
                    test_metrics=test_metrics,
                    test_equity=test_result.equity_curve,
                    test_returns=test_result.net_returns,
                )
            )
            window_id += 1

        return self._aggregate(windows)

    # ------------------------------------------------------------------

    def _optimise(
        self,
        df: pd.DataFrame,
        strategy,
        param_combinations: list[dict],
        symbol: str,
    ) -> tuple[dict, MetricsResult, list[tuple[dict, float]]]:
        """Grid-search over param_combinations on df; return best params."""
        best_score = -np.inf
        best_params = param_combinations[0]
        best_metrics = None
        scores = []

        for params in param_combinations:
            signals = strategy.generate_signals(df, params)
            result = self._engine.run(df, signals, symbol, strategy.name, params)
            metrics = compute_metrics(result.net_returns, result.trades)
            score = getattr(metrics, self.optimise_metric, metrics.sharpe)
            scores.append((params, score))
            if score > best_score:
                best_score = score
                best_params = params
                best_metrics = metrics

        return best_params, best_metrics or compute_metrics(pd.Series(dtype=float)), scores

    def _aggregate(self, windows: list[WindowResult]) -> WalkForwardResult:
        if not windows:
            return WalkForwardResult()

        oos_parts = [w.test_returns for w in windows]
        oos_returns = pd.concat(oos_parts).sort_index()
        # Remove duplicate index entries (overlapping windows shouldn't exist, but be safe)
        oos_returns = oos_returns[~oos_returns.index.duplicated(keep="first")]

        oos_equity = (1 + oos_returns).cumprod()
        oos_metrics = compute_metrics(oos_returns)

        param_rows = []
        for w in windows:
            row = {"window": w.window_id, "test_start": w.test_start}
            row.update(w.best_params)
            param_rows.append(row)
        param_stability = pd.DataFrame(param_rows)

        return WalkForwardResult(
            windows=windows,
            oos_returns=oos_returns,
            oos_equity=oos_equity,
            oos_metrics=oos_metrics,
            param_stability=param_stability,
        )
