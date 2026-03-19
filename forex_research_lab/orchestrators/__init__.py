"""Orchestration layers across multiple strategies."""

from .performance_based import rolling_sharpe_allocation, weighted_signal_from_allocations
from .regime_classifier import classify_regime, regime_switch_signal
from .rule_based import adx_rule_switch

__all__ = [
    "adx_rule_switch",
    "rolling_sharpe_allocation",
    "weighted_signal_from_allocations",
    "classify_regime",
    "regime_switch_signal",
]
