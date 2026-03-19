#!/usr/bin/env python3
"""Forex Strategy Research Lab — Working Prototype Demo.

This script demonstrates the full pipeline:
  1. Generate sample OHLC data
  2. Compute indicators
  3. Run MA Crossover and RSI Mean Reversion strategies
  4. Execute trades with realistic costs
  5. Run walk-forward evaluation
  6. Run bootstrap robustness analysis
  7. Test combination and orchestration layers
  8. Output metrics + equity curve summary
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from forex_lab.data.loader import generate_sample_data, generate_multi_symbol_data
from forex_lab.data.indicators import compute_indicators
from forex_lab.data.costs import attach_costs

from forex_lab.strategies.trend import MACrossover, DonchianBreakout
from forex_lab.strategies.mean_reversion import RSIReversal, BollingerFade
from forex_lab.strategies.breakout import RangeBreakout
from forex_lab.strategies.carry import CarryStrategy

from forex_lab.execution.engine import ExecutionEngine
from forex_lab.metrics.calculator import compute_metrics, metrics_table

from forex_lab.research.walk_forward import WalkForwardEngine
from forex_lab.research.bootstrap import BootstrapEngine
from forex_lab.research.param_sweep import ParameterSweep
from forex_lab.research.experiment_tracker import ExperimentTracker

from forex_lab.combinations.confirmation import ConfirmationCombiner
from forex_lab.combinations.ensemble import EnsembleCombiner
from forex_lab.combinations.sleeves import SpecialistSleeves, adx_filter

from forex_lab.orchestrators.rule_based import RuleBasedOrchestrator
from forex_lab.orchestrators.regime import RegimeOrchestrator, Regime


def separator(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def run_single_strategy(
    name: str,
    strategy,
    df: pd.DataFrame,
    engine: ExecutionEngine,
    tracker: ExperimentTracker,
    symbol: str = "EURUSD",
    timeframe: str = "H1",
) -> dict:
    params = strategy.default_params()
    signals = strategy.generate_signals(df, params)
    result = engine.run(df, signals)
    trades = engine.extract_trades(result)
    metrics = compute_metrics(result["net_returns"], trades=trades)

    tracker.log_run(
        strategy=name,
        params=params,
        symbol=symbol,
        timeframe=timeframe,
        metrics=metrics,
    )

    print(f"Strategy: {name}")
    print(f"  Params: {params}")
    print(metrics_table(metrics).to_string())
    print(f"  Final equity: ${result['equity'].iloc[-1]:,.2f}")
    print(f"  Trades: {len(trades)}")
    print()

    return {"result": result, "metrics": metrics, "trades": trades}


def main():
    separator("FOREX STRATEGY RESEARCH LAB — DEMO")

    # ── 1. Generate Data ──────────────────────────────────────────────
    separator("1. DATA GENERATION")
    df = generate_sample_data(symbol="EURUSD", periods=5000, freq="1h")
    df = compute_indicators(df)
    df = attach_costs(df)
    print(f"Generated {len(df)} bars of EURUSD H1 data")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Columns: {list(df.columns)}")
    print(f"Price range: {df['close'].min():.5f} - {df['close'].max():.5f}")

    engine = ExecutionEngine(
        spread=0.00010,
        slippage_bps=1.0,
        commission_bps=0.5,
        initial_capital=100_000,
    )
    tracker = ExperimentTracker()

    # ── 2. Individual Strategies ──────────────────────────────────────
    separator("2. INDIVIDUAL STRATEGY EVALUATION")

    strategies = {
        "MA Crossover": MACrossover(),
        "RSI Reversal": RSIReversal(),
        "Bollinger Fade": BollingerFade(),
        "Donchian Breakout": DonchianBreakout(),
        "Range Breakout": RangeBreakout(),
        "Carry": CarryStrategy(),
    }

    results = {}
    for name, strat in strategies.items():
        results[name] = run_single_strategy(name, strat, df, engine, tracker)

    # ── 3. Walk-Forward Analysis ──────────────────────────────────────
    separator("3. WALK-FORWARD ANALYSIS (MA Crossover)")
    wf_engine = WalkForwardEngine(
        strategy=MACrossover(),
        execution_engine=engine,
        train_bars=1000,
        test_bars=250,
        optimize_metric="sharpe",
    )
    wf_result = wf_engine.run(
        df,
        param_grid={"fast_period": [10, 20, 30], "slow_period": [50, 100, 150]},
    )
    print("Walk-Forward OOS Aggregate Metrics:")
    print(wf_result.summary().to_string())
    print(f"\nNumber of windows: {len(wf_result.windows)}")
    print("\nPer-window best params:")
    for _, row in wf_result.windows.iterrows():
        print(f"  {row['test_start'].strftime('%Y-%m-%d')} - {row['test_end'].strftime('%Y-%m-%d')}: "
              f"params={row['best_params']}, OOS Sharpe={row['oos_sharpe']:.3f}")

    separator("3b. WALK-FORWARD ANALYSIS (RSI Reversal)")
    wf_rsi = WalkForwardEngine(
        strategy=RSIReversal(),
        execution_engine=engine,
        train_bars=1000,
        test_bars=250,
        optimize_metric="sharpe",
    )
    wf_rsi_result = wf_rsi.run(
        df,
        param_grid={
            "rsi_period": [7, 14, 21],
            "oversold": [25, 30],
            "overbought": [70, 75],
        },
    )
    print("Walk-Forward OOS Aggregate Metrics (RSI Reversal):")
    print(wf_rsi_result.summary().to_string())

    # ── 4. Bootstrap Robustness ──────────────────────────────────────
    separator("4. BOOTSTRAP ROBUSTNESS ANALYSIS")
    bootstrap = BootstrapEngine(n_samples=500, seed=42)

    for name in ["MA Crossover", "RSI Reversal"]:
        returns = results[name]["result"]["net_returns"]
        boot_result = bootstrap.run(returns)
        risk = bootstrap.risk_of_ruin(returns, ruin_threshold=-0.3)
        ci_sharpe = boot_result.confidence_interval("sharpe")

        print(f"\n{name} Bootstrap (500 samples):")
        print(boot_result.summary().to_string())
        print(f"  Risk of Ruin (30% DD): {risk:.2%}")
        print(f"  Sharpe 95% CI: [{ci_sharpe[0]:.3f}, {ci_sharpe[1]:.3f}]")

    # ── 5. Parameter Sweep ───────────────────────────────────────────
    separator("5. PARAMETER SWEEP (MA Crossover)")
    sweep = ParameterSweep(strategy=MACrossover(), execution_engine=engine)
    sweep_results = sweep.grid_search(
        df,
        param_grid={"fast_period": [10, 20, 30, 50], "slow_period": [50, 100, 150, 200]},
    )
    valid = sweep_results[sweep_results["fast_period"] < sweep_results["slow_period"]]
    print("Top 5 parameter combinations by Sharpe:")
    top5 = valid.sort_values("sharpe", ascending=False).head(5)
    print(top5[["fast_period", "slow_period", "sharpe", "cagr", "max_drawdown", "trade_count"]].to_string(index=False))

    # ── 6. Combination Strategies ────────────────────────────────────
    separator("6. COMBINATION STRATEGIES")

    print("--- Confirmation (MA + RSI agree) ---")
    confirm = ConfirmationCombiner(
        strategies=[MACrossover(), RSIReversal()],
        min_agree=2,
    )
    conf_signals = confirm.generate_signals(df)
    conf_result = engine.run(df, conf_signals)
    conf_metrics = compute_metrics(conf_result["net_returns"])
    tracker.log_run("Confirmation(MA+RSI)", {}, "EURUSD", "H1", conf_metrics)
    print(metrics_table(conf_metrics).to_string())
    print(f"  Final equity: ${conf_result['equity'].iloc[-1]:,.2f}")

    print("\n--- Ensemble (weighted average) ---")
    ensemble = EnsembleCombiner(
        strategies=[MACrossover(), RSIReversal(), BollingerFade()],
        weights=[0.5, 0.3, 0.2],
        threshold=0.3,
    )
    ens_signals = ensemble.generate_signals(df)
    ens_result = engine.run(df, ens_signals)
    ens_metrics = compute_metrics(ens_result["net_returns"])
    tracker.log_run("Ensemble(MA+RSI+BB)", {}, "EURUSD", "H1", ens_metrics)
    print(metrics_table(ens_metrics).to_string())
    print(f"  Final equity: ${ens_result['equity'].iloc[-1]:,.2f}")

    print("\n--- Specialist Sleeves (ADX filter) ---")
    sleeves = SpecialistSleeves()
    sleeves.add_sleeve(MACrossover(), filter_fn=adx_filter(25.0, above=True))
    sleeves.add_sleeve(RSIReversal(), filter_fn=adx_filter(25.0, above=False))
    sleeve_signals = sleeves.generate_signals(df)
    sleeve_result = engine.run(df, sleeve_signals)
    sleeve_metrics = compute_metrics(sleeve_result["net_returns"])
    tracker.log_run("Sleeves(Trend|MR)", {}, "EURUSD", "H1", sleeve_metrics)
    print(metrics_table(sleeve_metrics).to_string())
    print(f"  Final equity: ${sleeve_result['equity'].iloc[-1]:,.2f}")

    # ── 7. Orchestration ─────────────────────────────────────────────
    separator("7. ORCHESTRATION")

    print("--- Rule-Based Orchestrator ---")
    rule_orch = RuleBasedOrchestrator(
        trend_strategy=MACrossover(),
        mean_reversion_strategy=RSIReversal(),
        high_adx=30.0,
        low_adx=20.0,
    )
    rule_signals = rule_orch.generate_signals(df)
    rule_result = engine.run(df, rule_signals)
    rule_metrics = compute_metrics(rule_result["net_returns"])
    tracker.log_run("RuleBased(ADX)", {}, "EURUSD", "H1", rule_metrics)
    print(metrics_table(rule_metrics).to_string())
    print(f"  Final equity: ${rule_result['equity'].iloc[-1]:,.2f}")

    print("\n--- Regime Orchestrator ---")
    regime_orch = RegimeOrchestrator(adx_threshold=25.0, vol_threshold=0.5)
    regime_orch.set_strategy(Regime.TRENDING_HIGH_VOL, MACrossover())
    regime_orch.set_strategy(Regime.TRENDING_LOW_VOL, DonchianBreakout())
    regime_orch.set_strategy(Regime.RANGING_HIGH_VOL, RSIReversal())
    regime_orch.set_strategy(Regime.RANGING_LOW_VOL, BollingerFade())

    regime_signals = regime_orch.generate_signals(df)
    regime_result = engine.run(df, regime_signals)
    regime_metrics = compute_metrics(regime_result["net_returns"])
    tracker.log_run("Regime(4-state)", {}, "EURUSD", "H1", regime_metrics)
    print(metrics_table(regime_metrics).to_string())
    print(f"  Final equity: ${regime_result['equity'].iloc[-1]:,.2f}")

    regimes = regime_orch.classify_regime(df)
    print("\nRegime distribution:")
    for r in Regime:
        count = (regimes == r).sum()
        print(f"  {r.value}: {count} bars ({count / len(df):.1%})")

    # ── 8. Multi-Symbol Run ──────────────────────────────────────────
    separator("8. MULTI-SYMBOL EVALUATION (MA Crossover)")
    multi_data = generate_multi_symbol_data(periods=3000, freq="1h")
    for sym, sym_df in multi_data.items():
        sym_df = compute_indicators(sym_df)
        sym_df = attach_costs(sym_df)
        strat = MACrossover()
        signals = strat.generate_signals(sym_df, strat.default_params())
        exec_result = engine.run(sym_df, signals)
        m = compute_metrics(exec_result["net_returns"])
        tracker.log_run("MA Crossover", strat.default_params(), sym, "H1", m)
        print(f"  {sym}: Sharpe={m['sharpe']:.3f}, CAGR={m['cagr']:.2%}, "
              f"MaxDD={m['max_drawdown']:.2%}, Equity=${exec_result['equity'].iloc[-1]:,.2f}")

    # ── 9. Experiment Tracker Summary ────────────────────────────────
    separator("9. EXPERIMENT TRACKER — ALL RUNS")
    all_runs = tracker.to_dataframe()
    display_cols = ["strategy", "symbol", "timeframe", "sharpe", "cagr", "max_drawdown", "win_rate"]
    available_cols = [c for c in display_cols if c in all_runs.columns]
    print(all_runs[available_cols].to_string(index=False))

    print("\n--- Best runs by Sharpe ---")
    best = tracker.best_runs(metric="sharpe", n=5)
    print(best[available_cols].to_string(index=False))

    separator("DEMO COMPLETE")
    print("All modules functional. The system is ready for research.")


if __name__ == "__main__":
    main()
