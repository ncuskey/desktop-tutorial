"""Core contracts, shared types, and component registry for V2."""

from .interfaces import (
    Evaluator,
    FunctionOrchestratorAdapter,
    FunctionStrategyAdapter,
    MetaFilter,
    Orchestrator,
    PortfolioAllocator,
    Strategy,
)
from .registry import ComponentRegistry, GLOBAL_REGISTRY, register_default_components
from .types import (
    AllocationResult,
    EvaluationResult,
    FeatureFrame,
    LabelSeries,
    MarketFrame,
    PositionSeries,
    SignalSeries,
    SleeveSignals,
)

__all__ = [
    "Strategy",
    "Orchestrator",
    "MetaFilter",
    "Evaluator",
    "PortfolioAllocator",
    "FunctionStrategyAdapter",
    "FunctionOrchestratorAdapter",
    "ComponentRegistry",
    "GLOBAL_REGISTRY",
    "register_default_components",
    "MarketFrame",
    "SignalSeries",
    "FeatureFrame",
    "LabelSeries",
    "PositionSeries",
    "SleeveSignals",
    "EvaluationResult",
    "AllocationResult",
]
