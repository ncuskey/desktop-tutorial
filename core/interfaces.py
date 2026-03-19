from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

import pandas as pd

from .types import AllocationResult, EvaluationResult, FeatureFrame, LabelSeries, MarketFrame, SignalSeries, SleeveSignals


@runtime_checkable
class Strategy(Protocol):
    """Strategy interface: produce position signals in {-1, 0, +1} (or scaled)."""

    name: str

    def generate_signals(self, df: MarketFrame, params: dict[str, Any] | None = None) -> SignalSeries:
        ...


@runtime_checkable
class Orchestrator(Protocol):
    """Orchestrator interface: route/combine sleeve signals into a final signal."""

    name: str

    def orchestrate(
        self,
        df: MarketFrame,
        sleeve_signals: SleeveSignals,
        params: dict[str, Any] | None = None,
    ) -> SignalSeries:
        ...


@runtime_checkable
class MetaFilter(Protocol):
    """Meta-filter interface: fit on train events and apply/transform on test events."""

    name: str

    def fit(
        self,
        X: FeatureFrame,
        y: LabelSeries,
        **kwargs: Any,
    ) -> "MetaFilter":
        ...

    def transform(
        self,
        X: FeatureFrame,
        **kwargs: Any,
    ) -> pd.DataFrame:
        ...


@runtime_checkable
class Evaluator(Protocol):
    """Evaluator interface for backtest / walk-forward / robustness engines."""

    name: str

    def evaluate(self, df: MarketFrame, **kwargs: Any) -> EvaluationResult:
        ...


@runtime_checkable
class PortfolioAllocator(Protocol):
    """Portfolio allocation interface operating on per-symbol sleeves/signals."""

    name: str

    def allocate(
        self,
        signal_frame: pd.DataFrame,
        **kwargs: Any,
    ) -> AllocationResult:
        ...


@dataclass
class FunctionStrategyAdapter:
    """
    Adapter for existing function-style strategies:
    fn(df, params) -> signal
    """

    name: str
    fn: Callable[[MarketFrame, dict[str, Any]], SignalSeries]
    default_params: dict[str, Any] = field(default_factory=dict)

    def generate_signals(self, df: MarketFrame, params: dict[str, Any] | None = None) -> SignalSeries:
        effective = dict(self.default_params)
        if params:
            effective.update(params)
        return self.fn(df, effective)


@dataclass
class FunctionOrchestratorAdapter:
    """
    Adapter for function-style orchestrators:
    fn(df, sleeve_signals, params) -> signal
    """

    name: str
    fn: Callable[[MarketFrame, SleeveSignals, dict[str, Any]], SignalSeries]
    default_params: dict[str, Any] = field(default_factory=dict)

    def orchestrate(
        self,
        df: MarketFrame,
        sleeve_signals: SleeveSignals,
        params: dict[str, Any] | None = None,
    ) -> SignalSeries:
        effective = dict(self.default_params)
        if params:
            effective.update(params)
        return self.fn(df, sleeve_signals, effective)
