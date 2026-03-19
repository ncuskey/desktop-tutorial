"""Research workflows: sweeps, walk-forward, bootstrap, tracking, reporting."""

from forex_research_lab.research.bootstrap import BootstrapResult, bootstrap_returns, bootstrap_trades
from forex_research_lab.research.parameter_sweep import expand_parameter_grid, run_parameter_sweep
from forex_research_lab.research.reporting import save_experiment_outputs
from forex_research_lab.research.tracking import ExperimentTracker
from forex_research_lab.research.walk_forward import WalkForwardResult, run_walk_forward

__all__ = [
    "BootstrapResult",
    "ExperimentTracker",
    "WalkForwardResult",
    "bootstrap_returns",
    "bootstrap_trades",
    "expand_parameter_grid",
    "run_parameter_sweep",
    "run_walk_forward",
    "save_experiment_outputs",
]
