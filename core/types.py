from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping

import pandas as pd


MarketFrame = pd.DataFrame
SignalSeries = pd.Series
FeatureFrame = pd.DataFrame
LabelSeries = pd.Series
PositionSeries = pd.Series

Params = Mapping[str, Any]
MutableParams = MutableMapping[str, Any]
SleeveSignals = Mapping[str, SignalSeries]


@dataclass
class EvaluationResult:
    """Minimal evaluator output contract for registrable evaluators."""

    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)


@dataclass
class AllocationResult:
    """Portfolio allocation output container."""

    weights: pd.DataFrame
    diagnostics: dict[str, Any] = field(default_factory=dict)
