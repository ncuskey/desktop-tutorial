"""Research engines: walk-forward, bootstrap, sweep, experiment tracking."""

from .bootstrap import bootstrap_returns
from .parameter_sweep import grid_parameter_sweep, random_parameter_sweep
from .switch_diagnostics import (
    compute_regime_duration_stats,
    compute_switch_diagnostics,
    compute_switches_per_1000_bars,
)
from .regime_gated_runner import run_regime_gated_evaluation
from .r13_trend_gating import run_r13_trend_gating
from .r14_meta_labeling import run_r14_meta_labeling
from .tracking import ExperimentTracker
from .v22_runner import V22RunArtifacts, run_v22_candidate_hardening
from .walk_forward import WalkForwardResult, run_walk_forward

__all__ = [
    "bootstrap_returns",
    "grid_parameter_sweep",
    "random_parameter_sweep",
    "compute_switch_diagnostics",
    "compute_regime_duration_stats",
    "compute_switches_per_1000_bars",
    "run_regime_gated_evaluation",
    "run_r13_trend_gating",
    "run_r14_meta_labeling",
    "ExperimentTracker",
    "WalkForwardResult",
    "run_walk_forward",
    "V22RunArtifacts",
    "run_v22_candidate_hardening",
]
