"""
Backward-compatible facade for legacy meta-label imports.

V2 implementation lives in:
- metalabel.labels
- metalabel.features_trade_quality
- metalabel.filter_rule_based
- metalabel.ablation
"""

from __future__ import annotations

from .ablation import run_feature_ablation
from .features_trade_quality import build_trade_meta_features, rolling_slope
from .filter_rule_based import RuleBasedMetaFilter, apply_meta_trade_filter
from .labels import (
    compute_forward_trade_returns,
    create_trade_success_labels,
    entry_mask_from_signal,
    infer_filter_type_from_regime,
    resolve_signal,
)

__all__ = [
    "build_trade_meta_features",
    "rolling_slope",
    "create_trade_success_labels",
    "compute_forward_trade_returns",
    "entry_mask_from_signal",
    "resolve_signal",
    "infer_filter_type_from_regime",
    "RuleBasedMetaFilter",
    "apply_meta_trade_filter",
    "run_feature_ablation",
]
