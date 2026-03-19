"""Research engines: walk-forward, bootstrap, parameter sweep, tracking."""

from .bootstrap import BootstrapResult, bootstrap_returns, bootstrap_trade_returns
from .parameter_sweep import (
    grid_search,
    iter_param_grid,
    random_search,
    select_best_params,
    to_heatmap_matrix,
)
from .tracking import ExperimentTracker
from .walk_forward import WalkForwardConfig, WalkForwardResult, run_walk_forward

__all__ = [
    "BootstrapResult",
    "bootstrap_returns",
    "bootstrap_trade_returns",
    "iter_param_grid",
    "grid_search",
    "random_search",
    "select_best_params",
    "to_heatmap_matrix",
    "ExperimentTracker",
    "WalkForwardConfig",
    "WalkForwardResult",
    "run_walk_forward",
]
