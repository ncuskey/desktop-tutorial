"""Meta-labeling utilities for trade quality filtering."""

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
    "RuleBasedMetaFilter",
    "apply_meta_trade_filter",
    "build_trade_meta_features",
    "rolling_slope",
    "create_trade_success_labels",
    "compute_forward_trade_returns",
    "entry_mask_from_signal",
    "resolve_signal",
    "infer_filter_type_from_regime",
    "run_feature_ablation",
]
