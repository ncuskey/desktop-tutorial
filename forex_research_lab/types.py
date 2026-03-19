"""Shared data structures for the research lab."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(slots=True)
class CostModel:
    """Execution cost assumptions expressed in basis points."""

    spread_bps: float = 1.0
    slippage_bps: float = 0.5
    commission_bps: float = 0.1

    @property
    def round_turn_bps(self) -> float:
        return self.spread_bps + self.slippage_bps + self.commission_bps


@dataclass(slots=True)
class StrategyDefinition:
    """Minimal strategy contract used by research engines."""

    name: str
    generate_signals: Any


@dataclass(slots=True)
class BacktestResult:
    """Normalized backtest output."""

    frame: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, float]


@dataclass(slots=True)
class WalkForwardSplit:
    """Metadata for a single train/test split."""

    split_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_params: dict[str, Any]
    train_metric: float


@dataclass(slots=True)
class WalkForwardResult:
    """Aggregated walk-forward output."""

    aggregated_frame: pd.DataFrame
    aggregated_trades: pd.DataFrame
    split_metrics: pd.DataFrame
    splits: list[WalkForwardSplit]
    parameter_results: pd.DataFrame


@dataclass(slots=True)
class BacktestArtifactPaths:
    """Filesystem outputs produced by the prototype runner."""

    base_dir: Path
    equity_curve_csv: Path
    drawdown_curve_csv: Path
    metrics_csv: Path
    heatmap_csv: Path
    experiment_log_db: Path
    equity_curve_png: Path | None = None
    drawdown_curve_png: Path | None = None
    heatmap_png: Path | None = None
    extra_files: dict[str, Path] = field(default_factory=dict)
