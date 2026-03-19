#!/usr/bin/env python3
"""Forex Strategy Research Lab — minimal working prototype.

Demonstrates:
  1. Generating synthetic OHLC data for EURUSD
  2. Running MA-crossover (trend) and RSI-reversal (mean reversion) strategies
  3. Executing trades with realistic costs
  4. Walk-forward evaluation
  5. Bootstrap robustness testing
  6. Combination (ensemble) of both strategies
  7. Rule-based orchestration (ADX switch)
  8. Experiment tracking
  9. Output of metrics + equity curves
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the parent directory is on sys.path so forex_lab is importable.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from forex_lab.data import generate_sample_data, add_indicators, attach_cost_model
from forex_lab.strategies import MACrossover, RSIReversal
from forex_lab.execution import execute_signals
from forex_lab.metrics import compute_metrics, metrics_table
from forex_lab.research import WalkForwardEngine, BootstrapEngine, ExperimentTracker
from forex_lab.combinations import EnsembleCombiner
from forex_lab.orchestrators import RuleBasedOrchestrator

SEPARATOR = "=" * 72


def section(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def run() -> None:
    # ------------------------------------------------------------------
    # 1. Generate sample data
    # ------------------------------------------------------------------
    section("1. Data Generation")
    df_raw = generate_sample_data("EURUSD", periods=5000, freq="h", seed=42)
    df = add_indicators(df_raw)
    df = attach_cost_model(df)
    print(f"  Bars: {len(df)}  |  Range: {df.index[0]} → {df.index[-1]}")
    print(f"  Columns: {list(df.columns)}")

    # ------------------------------------------------------------------
    # 2a. MA Crossover
    # ------------------------------------------------------------------
    section("2a. MA Crossover Strategy")
    ma_strat = MACrossover()
    ma_params = {"fast_period": 20, "slow_period": 50, "ma_type": "sma"}
    ma_signals = ma_strat.generate_signals(df, ma_params)
    ma_result = execute_signals(df, ma_signals)
    ma_metrics = compute_metrics(ma_result, freq="h")
    print(f"  Metrics: {ma_metrics}")

    # ------------------------------------------------------------------
    # 2b. RSI Reversal
    # ------------------------------------------------------------------
    section("2b. RSI Reversal Strategy")
    rsi_strat = RSIReversal()
    rsi_params = {"rsi_period": 14, "oversold": 30, "overbought": 70}
    rsi_signals = rsi_strat.generate_signals(df, rsi_params)
    rsi_result = execute_signals(df, rsi_signals)
    rsi_metrics = compute_metrics(rsi_result, freq="h")
    print(f"  Metrics: {rsi_metrics}")

    # ------------------------------------------------------------------
    # 3. Comparison Table
    # ------------------------------------------------------------------
    section("3. Strategy Comparison")
    table = metrics_table({"MA Crossover": ma_metrics, "RSI Reversal": rsi_metrics})
    print(table.to_string())

    # ------------------------------------------------------------------
    # 4. Walk-Forward Evaluation (MA Crossover)
    # ------------------------------------------------------------------
    section("4. Walk-Forward Analysis — MA Crossover")
    wf = WalkForwardEngine(
        strategy=ma_strat,
        param_grid={
            "fast_period": [10, 20, 30],
            "slow_period": [50, 100],
            "ma_type": ["sma"],
        },
        train_size=2000,
        test_size=500,
        freq="h",
    )
    wf_result = wf.run(df)
    print(f"  Folds: {len(wf_result['folds'])}")
    for i, fold in enumerate(wf_result["folds"]):
        print(f"    Fold {i+1}: params={fold['best_params']}  OOS Sharpe={fold['oos_metrics']['sharpe']:.4f}")
    print(f"  Aggregate OOS metrics: {wf_result['aggregate_metrics']}")

    # ------------------------------------------------------------------
    # 5. Bootstrap Robustness
    # ------------------------------------------------------------------
    section("5. Bootstrap Robustness — MA Crossover")
    bootstrap = BootstrapEngine(n_samples=500, freq="h", seed=42)
    bs_result = bootstrap.run(ma_result["net_returns"])
    print(f"  Sharpe mean: {bs_result['sharpe_mean']:.4f}")
    print(f"  Sharpe 95% CI: ({bs_result['sharpe_ci'][0]:.4f}, {bs_result['sharpe_ci'][1]:.4f})")
    print(f"  Max DD mean: {bs_result['max_dd_mean']:.4f}")
    print(f"  Max DD 95% CI: ({bs_result['max_dd_ci'][0]:.4f}, {bs_result['max_dd_ci'][1]:.4f})")
    print(f"  Risk of ruin (50% loss): {bs_result['risk_of_ruin']:.4f}")

    # ------------------------------------------------------------------
    # 6. Ensemble Combination
    # ------------------------------------------------------------------
    section("6. Ensemble (MA + RSI)")
    ensemble = EnsembleCombiner(
        strategies=[ma_strat, rsi_strat],
        params_list=[ma_params, rsi_params],
        weights=[0.6, 0.4],
        threshold=0.3,
    )
    ens_signals = ensemble.generate_signals(df)
    ens_result = execute_signals(df, ens_signals)
    ens_metrics = compute_metrics(ens_result, freq="h")
    print(f"  Metrics: {ens_metrics}")

    # ------------------------------------------------------------------
    # 7. Rule-Based Orchestration
    # ------------------------------------------------------------------
    section("7. Rule-Based Orchestrator (ADX Switch)")
    orchestrator = RuleBasedOrchestrator(
        trend_strategy=ma_strat,
        trend_params=ma_params,
        mr_strategy=rsi_strat,
        mr_params=rsi_params,
        adx_threshold=25.0,
    )
    orch_signals = orchestrator.generate_signals(df)
    orch_result = execute_signals(df, orch_signals)
    orch_metrics = compute_metrics(orch_result, freq="h")
    print(f"  Metrics: {orch_metrics}")

    # ------------------------------------------------------------------
    # 8. Full Comparison
    # ------------------------------------------------------------------
    section("8. Full Comparison Table")
    full_table = metrics_table(
        {
            "MA Crossover": ma_metrics,
            "RSI Reversal": rsi_metrics,
            "Ensemble": ens_metrics,
            "Orchestrator": orch_metrics,
            "WF Aggregate": wf_result["aggregate_metrics"],
        }
    )
    print(full_table.to_string())

    # ------------------------------------------------------------------
    # 9. Experiment Tracking
    # ------------------------------------------------------------------
    section("9. Experiment Tracking")
    tracker = ExperimentTracker(db_path=":memory:")
    tracker.log("ma_crossover", "EURUSD", "H1", ma_params, ma_metrics, "prototype run")
    tracker.log("rsi_reversal", "EURUSD", "H1", rsi_params, rsi_metrics, "prototype run")
    tracker.log("ensemble", "EURUSD", "H1", {"ma": ma_params, "rsi": rsi_params}, ens_metrics, "prototype run")
    tracker.log("orchestrator", "EURUSD", "H1", {"adx_threshold": 25.0}, orch_metrics, "prototype run")

    exp_df = tracker.to_dataframe()
    print(f"  Logged {len(exp_df)} experiments")
    print(exp_df[["strategy", "symbol", "timeframe", "notes"]].to_string(index=False))
    tracker.close()

    # ------------------------------------------------------------------
    # 10. Equity Curve Summary
    # ------------------------------------------------------------------
    section("10. Equity Curve Summary")
    print(f"  MA Crossover — Start: {ma_result['equity'].iloc[0]:,.2f}  End: {ma_result['equity'].iloc[-1]:,.2f}  Max DD: {ma_result['drawdown'].min():.4f}")
    print(f"  RSI Reversal — Start: {rsi_result['equity'].iloc[0]:,.2f}  End: {rsi_result['equity'].iloc[-1]:,.2f}  Max DD: {rsi_result['drawdown'].min():.4f}")
    print(f"  Ensemble     — Start: {ens_result['equity'].iloc[0]:,.2f}  End: {ens_result['equity'].iloc[-1]:,.2f}  Max DD: {ens_result['drawdown'].min():.4f}")
    print(f"  Orchestrator — Start: {orch_result['equity'].iloc[0]:,.2f}  End: {orch_result['equity'].iloc[-1]:,.2f}  Max DD: {orch_result['drawdown'].min():.4f}")

    print(f"\n{SEPARATOR}")
    print("  Prototype complete.")
    print(SEPARATOR)


if __name__ == "__main__":
    run()
