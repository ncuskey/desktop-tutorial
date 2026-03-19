"""Core contracts, shared types, and component registry for V2."""

from typing import Any

from .interfaces import (
    Evaluator,
    FunctionOrchestratorAdapter,
    FunctionStrategyAdapter,
    MetaFilter,
    Orchestrator,
    PortfolioAllocator,
    Strategy,
)
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


def __getattr__(name: str) -> Any:
    if name in {"ComponentRegistry", "GLOBAL_REGISTRY", "register_default_components"}:
        from .registry import ComponentRegistry, GLOBAL_REGISTRY, register_default_components

        mapping = {
            "ComponentRegistry": ComponentRegistry,
            "GLOBAL_REGISTRY": GLOBAL_REGISTRY,
            "register_default_components": register_default_components,
        }
        return mapping[name]
    raise AttributeError(f"module 'core' has no attribute '{name}'")
