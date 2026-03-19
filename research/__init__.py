"""Research engines: walk-forward, bootstrap, sweep, experiment tracking."""

from .bootstrap import bootstrap_returns
from .parameter_sweep import grid_parameter_sweep, random_parameter_sweep
from .switch_diagnostics import compute_switch_diagnostics
from .tracking import ExperimentTracker
from .walk_forward import WalkForwardResult, run_walk_forward

__all__ = [
    "bootstrap_returns",
    "grid_parameter_sweep",
    "random_parameter_sweep",
    "compute_switch_diagnostics",
    "ExperimentTracker",
    "WalkForwardResult",
    "run_walk_forward",
]
