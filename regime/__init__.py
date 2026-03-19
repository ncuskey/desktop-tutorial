"""Market regime detection utilities."""

from .regime_detection import attach_regime_labels
from .state_machine import (
    attach_stable_regime_state,
    build_stable_regime_series,
    regime_duration_distribution,
)

__all__ = [
    "attach_regime_labels",
    "attach_stable_regime_state",
    "build_stable_regime_series",
    "regime_duration_distribution",
]
