"""Parameter search helpers for systematic strategy studies."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import pandas as pd

from forex_research_lab.execution import run_backtest
from forex_research_lab.metrics import compute_metrics


@dataclass(slots=True)
class SweepResult:
    results: pd.DataFrame
    best_params: dict
    best_metrics: dict[str, float]


class ParameterSweep:
    def run(
        self,
        strategy,
        df: pd.DataFrame,
        param_grid: dict[str, list],
        timeframe: str,
        objective: str = "sharpe",
    ) -> SweepResult:
        keys = list(param_grid.keys())
        rows: list[dict] = []
        best_params: dict = {}
        best_metrics: dict[str, float] = {}
        best_score = float("-inf")

        for values in product(*(param_grid[key] for key in keys)):
            params = dict(zip(keys, values, strict=True))
            try:
                signals = strategy.generate_signals(df, params)
                result = run_backtest(df, signals)
                metrics = compute_metrics(result.frame, result.trades, timeframe=timeframe)
            except Exception as exc:
                rows.append({**params, "objective": float("-inf"), "error": str(exc)})
                continue

            score = float(metrics.get(objective, float("-inf")))
            rows.append({**params, **metrics, "objective": score, "error": None})
            if score > best_score:
                best_score = score
                best_params = params
                best_metrics = metrics

        return SweepResult(
            results=pd.DataFrame(rows).sort_values("objective", ascending=False).reset_index(drop=True),
            best_params=best_params,
            best_metrics=best_metrics,
        )
