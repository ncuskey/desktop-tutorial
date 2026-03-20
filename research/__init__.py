"""Research engines: walk-forward, bootstrap, sweep, experiment tracking."""

from .bootstrap import bootstrap_returns
from .candidate_validation import run_candidate_validation, select_top_candidates
from .feature_pruning import build_feature_pruning_tables
from .monte_carlo import block_bootstrap_oos_returns
from .multi_symbol_runner import run_multi_symbol_evaluation
from .parameter_sweep import grid_parameter_sweep, random_parameter_sweep
from .purged_walk_forward import PurgedWalkForwardResult, run_purged_walk_forward
from .sleeve_ranking import build_sleeve_symbol_ranking, classify_component_decisions
from .stability import feature_stability_report, threshold_stability_report
from .switch_diagnostics import (
    compute_regime_duration_stats,
    compute_switch_diagnostics,
    compute_switches_per_1000_bars,
)
from .tracking import ExperimentTracker
from .v2_runner import run_v2_evaluation
from .v21_runner import run_v21_refinement
from .walk_forward import WalkForwardResult, run_walk_forward

__all__ = [
    "bootstrap_returns",
    "block_bootstrap_oos_returns",
    "run_multi_symbol_evaluation",
    "grid_parameter_sweep",
    "random_parameter_sweep",
    "run_purged_walk_forward",
    "PurgedWalkForwardResult",
    "threshold_stability_report",
    "feature_stability_report",
    "build_sleeve_symbol_ranking",
    "classify_component_decisions",
    "build_feature_pruning_tables",
    "select_top_candidates",
    "run_candidate_validation",
    "run_v2_evaluation",
    "run_v21_refinement",
    "compute_switch_diagnostics",
    "compute_regime_duration_stats",
    "compute_switches_per_1000_bars",
    "ExperimentTracker",
    "WalkForwardResult",
    "run_walk_forward",
]
