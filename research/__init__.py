"""Research engines: walk-forward, bootstrap, sweep, experiment tracking."""

from .bootstrap import bootstrap_returns
from .candidate_validation import run_candidate_validation, select_top_candidates
from .feature_pruning import build_feature_pruning_tables
from .monte_carlo import block_bootstrap_oos_returns
from .multi_symbol_runner import run_multi_symbol_evaluation
from .parameter_sweep import grid_parameter_sweep, random_parameter_sweep
from .parameter_robustness import RobustnessArtifacts, analyze_parameter_robustness
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
from .v22_runner import V22RunArtifacts, run_v22_candidate_hardening
from .v23_runner import run_v23_edge_amplification, summarize_tail_metrics
from .walk_forward import WalkForwardResult, run_walk_forward
from .strategy_runner import run_strategy_research
from .regime_diagnostics import run_regime_diagnostics
from .regime_gated_runner import run_regime_gated_evaluation
from .r14_execution_diagnostics import run_r14_execution_diagnostics
from .r14_tail_selection import run_r14_tail_selection
from .r15_failure_decomposition import run_r15_failure_decomposition
from .promotion_framework import (
    PromotionArtifacts,
    PromotionThresholds,
    SymbolPromotionResult,
    run_strategy_promotion_framework,
)
from .strategy_spec import (
    StrategySpec,
    build_strategy_spec,
    generate_strategy_spec,
    render_strategy_spec_markdown,
    write_strategy_spec_outputs,
)
from .r14_execution_layer import R14ExecutionArtifacts, run_r14_execution_layer

__all__ = [
    "bootstrap_returns",
    "block_bootstrap_oos_returns",
    "run_multi_symbol_evaluation",
    "grid_parameter_sweep",
    "random_parameter_sweep",
    "RobustnessArtifacts",
    "analyze_parameter_robustness",
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
    "V22RunArtifacts",
    "run_v22_candidate_hardening",
    "run_v23_edge_amplification",
    "summarize_tail_metrics",
    "run_strategy_research",
    "run_regime_diagnostics",
    "run_regime_gated_evaluation",
    "run_r14_execution_diagnostics",
    "run_r14_tail_selection",
    "run_r15_failure_decomposition",
    "PromotionThresholds",
    "SymbolPromotionResult",
    "PromotionArtifacts",
    "run_strategy_promotion_framework",
    "StrategySpec",
    "build_strategy_spec",
    "generate_strategy_spec",
    "render_strategy_spec_markdown",
    "write_strategy_spec_outputs",
    "R14ExecutionArtifacts",
    "run_r14_execution_layer",
]
