"""
Forex Strategy Research Lab — Minimal Working Prototype

Demonstrates the full pipeline:
  1. Generate synthetic OHLCV data for 4 FX pairs
  2. Compute indicators
  3. Run individual strategies (MA crossover, RSI reversal)
  4. Execute trades with realistic costs
  5. Walk-forward evaluation
  6. Bootstrap robustness testing
  7. Parameter sweep
  8. Combination strategies
  9. Regime orchestration
 10. Output metrics, equity curves, and heatmaps

Usage:
    cd forex_lab
    python main.py

Outputs are saved to ./output/
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# Make sure we can import sibling packages
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np

from data.loader import generate_synthetic_ohlcv
from data.resampler import resample_ohlcv
from data.indicators import add_indicators
from data.costs import DEFAULT_COSTS

from strategies.trend import MACrossover, DonchianBreakout
from strategies.mean_reversion import RSIReversal, BollingerFade
from strategies.breakout import RangeBreakout, VolatilityExpansionBreakout
from strategies.carry import CarryProxy

from execution.engine import ExecutionEngine

from metrics.performance import compute_metrics

from research.walk_forward import WalkForwardEngine
from research.bootstrap import BootstrapEngine
from research.param_sweep import ParameterSweep
from research.experiment_tracker import ExperimentTracker

from combinations.confirmation import ConfirmationCombiner
from combinations.ensemble import EnsembleCombiner
from combinations.sleeves import (
    SpecialistSleeves, Sleeve, adx_trend_filter, adx_range_filter,
)

from orchestrators.rule_based import RuleBasedOrchestrator, Rule
from orchestrators.regime import RegimeClassifier, RegimeOrchestrator

import visualization as viz


OUTPUT_DIR = Path(__file__).parent / "output"
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
TIMEFRAMES = ["H1", "H4", "D1"]


# ==========================================================================
# Helpers
# ==========================================================================

def banner(msg: str) -> None:
    width = 60
    print("\n" + "=" * width)
    print(f"  {msg}")
    print("=" * width)


def print_metrics(label: str, metrics) -> None:
    print(f"\n{'─'*40}")
    print(f"  {label}")
    print(f"{'─'*40}")
    print(metrics)


# ==========================================================================
# Data generation
# ==========================================================================

def generate_data() -> dict[str, dict[str, pd.DataFrame]]:
    """Generate synthetic OHLCV for each symbol × timeframe."""
    banner("Generating synthetic OHLCV data")
    data: dict[str, dict[str, pd.DataFrame]] = {}

    for symbol in SYMBOLS:
        data[symbol] = {}
        base_h1 = generate_synthetic_ohlcv(
            symbol=symbol,
            start="2018-01-01",
            end="2023-12-31",
            freq="h",
            seed=hash(symbol) % 10000,
        )
        data[symbol]["H1"] = base_h1
        data[symbol]["H4"] = resample_ohlcv(base_h1, "H4")
        data[symbol]["D1"] = resample_ohlcv(base_h1, "D1")
        print(f"  {symbol}: H1={len(base_h1)} bars | "
              f"H4={len(data[symbol]['H4'])} bars | "
              f"D1={len(data[symbol]['D1'])} bars")

    return data


# ==========================================================================
# Individual strategy runs
# ==========================================================================

def run_individual_strategies(
    data: dict,
    tracker: ExperimentTracker,
    engine: ExecutionEngine,
) -> dict[str, dict]:
    """Run all individual strategies on EURUSD H4."""
    banner("Running individual strategies — EURUSD H4")

    symbol = "EURUSD"
    tf = "H4"
    df = add_indicators(data[symbol][tf])

    strategies = [
        (MACrossover(),       {"fast_period": 20, "slow_period": 50}),
        (MACrossover(),       {"fast_period": 10, "slow_period": 30, "ma_type": "ema"}),
        (DonchianBreakout(),  {"period": 20}),
        (RSIReversal(),       {"period": 14, "oversold": 30, "overbought": 70}),
        (BollingerFade(),     {"period": 20, "num_std": 2.0}),
        (RangeBreakout(),     {"lookback": 20, "hold_bars": 10}),
        (VolatilityExpansionBreakout(), {"atr_period": 14, "atr_mult": 1.5}),
        (CarryProxy(),        {"symbol": symbol, "min_diff": 0.5}),
    ]

    results = {}
    for strat, params in strategies:
        signals = strat.generate_signals(df, params)
        result = engine.run(df, signals, symbol, strat.name, params)
        metrics = compute_metrics(result.net_returns, result.trades)
        run_id = tracker.log(strat.name, params, symbol, tf, metrics)

        label = f"{strat.name} ({symbol}/{tf})"
        print_metrics(label, metrics)
        results[strat.name] = {"result": result, "metrics": metrics, "signals": signals}

    return results


# ==========================================================================
# Walk-forward evaluation
# ==========================================================================

def run_walk_forward(
    data: dict,
    tracker: ExperimentTracker,
) -> None:
    """Walk-forward on MA crossover and RSI reversal."""
    banner("Walk-Forward Analysis")

    symbol = "EURUSD"
    tf = "H4"
    df = data[symbol][tf]

    wf_engine = WalkForwardEngine(
        train_bars=1500,
        test_bars=400,
        step_bars=400,
        optimise_metric="sharpe",
    )

    strategies_and_grids = [
        (
            MACrossover(),
            {
                "fast_period": [10, 20, 30],
                "slow_period": [40, 60, 80, 100],
                "ma_type": ["sma", "ema"],
            },
        ),
        (
            RSIReversal(),
            {
                "period": [10, 14, 21],
                "oversold": [25, 30, 35],
                "overbought": [65, 70, 75],
            },
        ),
    ]

    for strat, param_grid in strategies_and_grids:
        print(f"\n  Walk-forward: {strat.name} on {symbol}/{tf}")
        wf_result = wf_engine.run(df, strat, param_grid, symbol, verbose=True)

        if wf_result.oos_metrics:
            print(f"\n  OOS Aggregate metrics ({strat.name}):")
            print(wf_result.oos_metrics)
            tracker.log(
                strategy=f"{strat.name}_wf",
                params={"method": "walk_forward"},
                symbol=symbol,
                timeframe=tf,
                metrics=wf_result.oos_metrics,
            )

        print("\n  Window-by-window summary:")
        summary = wf_result.summary()
        print(summary[["window", "train_sharpe", "test_sharpe", "test_cagr", "test_max_dd"]].to_string(index=False))

        print("\n  Parameter stability:")
        print(wf_result.param_stability.to_string(index=False))

        # Save walk-forward chart
        wf_path = OUTPUT_DIR / f"wf_{strat.name}_{symbol}_{tf}.png"
        viz.plot_walk_forward(wf_result, title=f"Walk-Forward: {strat.name}", save_path=wf_path)
        print(f"  Chart saved: {wf_path}")


# ==========================================================================
# Bootstrap robustness
# ==========================================================================

def run_bootstrap(
    data: dict,
    individual_results: dict,
) -> None:
    """Bootstrap test on MA crossover result."""
    banner("Bootstrap Robustness Testing")

    symbol = "EURUSD"
    tf = "H4"

    for strat_name in ["ma_crossover", "rsi_reversal"]:
        if strat_name not in individual_results:
            continue

        result = individual_results[strat_name]["result"]
        bootstrap = BootstrapEngine(n_samples=500, method="block", block_size=20)
        bs_result = bootstrap.run_on_returns(
            result.net_returns, result.trades, verbose=True
        )

        print(f"\n  Bootstrap results for {strat_name}:")
        print(bs_result.summary().to_string())

        bs_path = OUTPUT_DIR / f"bootstrap_{strat_name}.png"
        viz.plot_bootstrap_distribution(
            bs_result,
            metric="sharpe",
            title=f"Bootstrap Sharpe Distribution: {strat_name}",
            save_path=bs_path,
        )
        print(f"  Chart saved: {bs_path}")


# ==========================================================================
# Parameter sweep
# ==========================================================================

def run_param_sweep(
    data: dict,
    tracker: ExperimentTracker,
) -> pd.DataFrame | None:
    """Grid search over MA crossover parameters."""
    banner("Parameter Sweep — MA Crossover")

    symbol = "EURUSD"
    tf = "H4"
    df = data[symbol][tf]

    sweeper = ParameterSweep(method="grid", optimise_metric="sharpe")
    sweep_results = sweeper.run(
        df,
        MACrossover(),
        param_grid={
            "fast_period": [5, 10, 15, 20, 30],
            "slow_period": [40, 50, 60, 80, 100],
            "ma_type": ["sma", "ema"],
        },
        symbol=symbol,
        verbose=True,
    )

    if len(sweep_results) > 0:
        print(f"\n  Top 10 parameter combinations (by Sharpe):")
        cols = ["fast_period", "slow_period", "ma_type", "sharpe", "cagr", "max_drawdown"]
        print(sweep_results[cols].head(10).to_string(index=False))

        robustness = sweeper.robustness_ratio(sweep_results)
        print(f"\n  Robustness ratio (% combos with Sharpe > 0): {robustness:.1%}")

        heatmap_path = OUTPUT_DIR / f"heatmap_ma_crossover_{symbol}.png"
        viz.plot_param_heatmap(
            sweep_results,
            param_x="fast_period",
            param_y="slow_period",
            metric="sharpe",
            title=f"Sharpe Heatmap: MA Crossover ({symbol}/{tf})",
            save_path=heatmap_path,
        )
        print(f"  Heatmap saved: {heatmap_path}")

        # Log top result to tracker
        top = sweep_results.iloc[0]
        params = {k: top[k] for k in ["fast_period", "slow_period", "ma_type"]}
        metrics = compute_metrics(
            pd.Series(dtype=float)  # placeholder — metrics are in the row
        )
        return sweep_results

    return None


# ==========================================================================
# Combination strategies
# ==========================================================================

def run_combinations(
    data: dict,
    engine: ExecutionEngine,
    tracker: ExperimentTracker,
) -> None:
    """Run confirmation, ensemble, and sleeve combinations."""
    banner("Combination Strategies")

    symbol = "EURUSD"
    tf = "H4"
    df = add_indicators(data[symbol][tf])

    ma = MACrossover()
    rsi = RSIReversal()
    donch = DonchianBreakout()
    boll = BollingerFade()

    ma_params = {"fast_period": 20, "slow_period": 50}
    rsi_params = {"period": 14, "oversold": 30, "overbought": 70}
    donch_params = {"period": 20}
    boll_params = {"period": 20, "num_std": 2.0}

    # 1. Confirmation (MA + Donchian must agree)
    print("\n  [1] Confirmation: MA crossover + Donchian breakout")
    conf = ConfirmationCombiner(threshold=1.0)
    conf_signals = conf.generate_signals(
        df, [ma, donch], [ma_params, donch_params]
    )
    conf_result = engine.run(df, conf_signals, symbol, "confirmation_ma_donch", {})
    conf_metrics = compute_metrics(conf_result.net_returns, conf_result.trades)
    tracker.log("confirmation_ma_donch", {}, symbol, tf, conf_metrics)
    print_metrics("Confirmation (MA + Donchian)", conf_metrics)

    # 2. Ensemble (average of all 4 strategies)
    print("\n  [2] Ensemble: MA + RSI + Donchian + Bollinger")
    ens = EnsembleCombiner(long_threshold=0.25, short_threshold=0.25)
    ens_signals = ens.generate_signals(
        df,
        [ma, rsi, donch, boll],
        [ma_params, rsi_params, donch_params, boll_params],
    )
    ens_result = engine.run(df, ens_signals, symbol, "ensemble_4strat", {})
    ens_metrics = compute_metrics(ens_result.net_returns, ens_result.trades)
    tracker.log("ensemble_4strat", {}, symbol, tf, ens_metrics)
    print_metrics("Ensemble (4 strategies)", ens_metrics)

    # 3. Specialist sleeves
    print("\n  [3] Specialist Sleeves: MA when trending, RSI when ranging")
    sleeves = SpecialistSleeves([
        Sleeve(ma, ma_params, adx_trend_filter(threshold=22.0), weight=1.0, name="ma_trend"),
        Sleeve(rsi, rsi_params, adx_range_filter(threshold=22.0), weight=1.0, name="rsi_range"),
    ])
    sleeve_signals = sleeves.generate_signals(df)
    sleeve_result = engine.run(df, sleeve_signals, symbol, "sleeves_ma_rsi", {})
    sleeve_metrics = compute_metrics(sleeve_result.net_returns, sleeve_result.trades)
    tracker.log("sleeves_ma_rsi", {}, symbol, tf, sleeve_metrics)
    print_metrics("Sleeves (MA trend + RSI range)", sleeve_metrics)

    # Save equity charts
    for label, result in [
        ("Confirmation", conf_result),
        ("Ensemble", ens_result),
        ("Sleeves", sleeve_result),
    ]:
        eq_path = OUTPUT_DIR / f"equity_{label.lower()}_{symbol}.png"
        viz.plot_equity_drawdown(
            result.equity_curve,
            result.drawdown,
            title=f"{label} Strategy — {symbol}/{tf}",
            save_path=eq_path,
        )
        print(f"  Chart saved: {eq_path}")


# ==========================================================================
# Orchestration
# ==========================================================================

def run_orchestrators(
    data: dict,
    engine: ExecutionEngine,
    tracker: ExperimentTracker,
) -> None:
    """Run rule-based and regime orchestrators."""
    banner("Orchestration Layer")

    symbol = "EURUSD"
    tf = "H4"
    df = add_indicators(data[symbol][tf])

    ma = MACrossover()
    rsi = RSIReversal()
    donch = DonchianBreakout()
    rng_bo = RangeBreakout()

    ma_params = {"fast_period": 20, "slow_period": 50}
    rsi_params = {"period": 14, "oversold": 30, "overbought": 70}
    donch_params = {"period": 20}
    rng_params = {"lookback": 20, "hold_bars": 10}

    # 1. Rule-based: ADX high → trend (MA), ADX low → mean reversion (RSI)
    print("\n  [1] Rule-based orchestrator (ADX dispatch)")
    orch_rules = RuleBasedOrchestrator(
        rules=[
            Rule("adx_high", ma, ma_params, priority=2),
            Rule("adx_low", rsi, rsi_params, priority=1),
            Rule("always", donch, donch_params, priority=0),
        ],
        adx_high_threshold=25.0,
        adx_low_threshold=20.0,
    )
    rb_signals = orch_rules.generate_signals(df)
    rb_result = engine.run(df, rb_signals, symbol, "rule_based_orch", {})
    rb_metrics = compute_metrics(rb_result.net_returns, rb_result.trades)
    tracker.log("rule_based_orch", {"adx_high": 25, "adx_low": 20}, symbol, tf, rb_metrics)
    print_metrics("Rule-based orchestrator", rb_metrics)

    eq_path = OUTPUT_DIR / f"equity_rule_based_orch_{symbol}.png"
    viz.plot_equity_drawdown(
        rb_result.equity_curve, rb_result.drawdown,
        title=f"Rule-Based Orchestrator — {symbol}/{tf}", save_path=eq_path
    )
    print(f"  Chart saved: {eq_path}")

    # 2. Regime orchestrator
    print("\n  [2] Regime orchestrator (ATR + ADX regime classifier)")
    classifier = RegimeClassifier(
        adx_trend_threshold=25.0,
        adx_range_threshold=20.0,
        atr_vol_multiplier=1.8,
    )
    regime_stats = classifier.regime_stats(df)
    print(f"  Regime distribution:\n{regime_stats.to_string()}")

    regime_orch = RegimeOrchestrator(
        regime_strategy_map={
            "trending": (ma, ma_params),
            "ranging":  (rsi, rsi_params),
            "volatile": (rng_bo, rng_params),
            "neutral":  (donch, donch_params),
        },
        classifier=classifier,
    )
    reg_signals = regime_orch.generate_signals(df)
    reg_result = engine.run(df, reg_signals, symbol, "regime_orch", {})
    reg_metrics = compute_metrics(reg_result.net_returns, reg_result.trades)
    tracker.log("regime_orch", {}, symbol, tf, reg_metrics)
    print_metrics("Regime orchestrator", reg_metrics)

    eq_path = OUTPUT_DIR / f"equity_regime_orch_{symbol}.png"
    viz.plot_equity_drawdown(
        reg_result.equity_curve, reg_result.drawdown,
        title=f"Regime Orchestrator — {symbol}/{tf}", save_path=eq_path
    )
    print(f"  Chart saved: {eq_path}")


# ==========================================================================
# Multi-symbol / multi-timeframe sweep
# ==========================================================================

def run_multi_symbol(
    data: dict,
    engine: ExecutionEngine,
    tracker: ExperimentTracker,
) -> None:
    """Run MA crossover across all 4 symbols and 3 timeframes."""
    banner("Multi-Symbol / Multi-Timeframe Run")

    ma = MACrossover()
    params = {"fast_period": 20, "slow_period": 50}

    rows = []
    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            df = add_indicators(data[symbol][tf])
            if len(df) < 200:
                continue
            signals = ma.generate_signals(df, params)
            result = engine.run(df, signals, symbol, ma.name, params)
            metrics = compute_metrics(result.net_returns, result.trades)
            tracker.log(ma.name, params, symbol, tf, metrics)
            rows.append({
                "symbol": symbol,
                "timeframe": tf,
                "sharpe": round(metrics.sharpe, 3),
                "cagr": f"{metrics.cagr:.2%}",
                "max_dd": f"{metrics.max_drawdown:.2%}",
                "n_trades": metrics.n_trades,
            })

    df_results = pd.DataFrame(rows)
    print(f"\n  MA Crossover results (fast=20, slow=50):")
    print(df_results.to_string(index=False))


# ==========================================================================
# Experiment summary
# ==========================================================================

def print_experiment_summary(tracker: ExperimentTracker) -> None:
    banner("Experiment Summary (Top 10 by Sharpe)")
    best = tracker.best_runs(metric="sharpe", n=10)
    if best.empty:
        print("  No experiments logged.")
        return
    cols = ["strategy", "symbol", "timeframe", "sharpe", "cagr", "max_drawdown", "n_trades"]
    available = [c for c in cols if c in best.columns]
    print(best[available].to_string(index=False))

    db_path = OUTPUT_DIR / "experiments.db"
    tracker_with_db = ExperimentTracker(db_path=db_path)
    for _, row in tracker.results.iterrows():
        from metrics.performance import MetricsResult
        m = MetricsResult(**{k: row.get(k, 0.0) for k in MetricsResult.__dataclass_fields__})
        params = row.get("params", {})
        if isinstance(params, str):
            import json
            params = json.loads(params)
        tracker_with_db.log(
            strategy=row.get("strategy", ""),
            params=params,
            symbol=row.get("symbol", ""),
            timeframe=row.get("timeframe", ""),
            metrics=m,
        )
    print(f"\n  Experiments saved to: {db_path}")


# ==========================================================================
# Main
# ==========================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tracker = ExperimentTracker()
    engine = ExecutionEngine()

    # 1. Data
    data = generate_data()

    # 2. Individual strategies
    individual_results = run_individual_strategies(data, tracker, engine)

    # 3. Walk-forward
    run_walk_forward(data, tracker)

    # 4. Bootstrap
    run_bootstrap(data, individual_results)

    # 5. Parameter sweep
    run_param_sweep(data, tracker)

    # 6. Combinations
    run_combinations(data, engine, tracker)

    # 7. Orchestration
    run_orchestrators(data, engine, tracker)

    # 8. Multi-symbol/timeframe
    run_multi_symbol(data, engine, tracker)

    # 9. Summary
    print_experiment_summary(tracker)

    # Save a final equity chart for MA crossover
    if "ma_crossover" in individual_results:
        r = individual_results["ma_crossover"]["result"]
        viz.plot_equity_drawdown(
            r.equity_curve, r.drawdown,
            title="MA Crossover — EURUSD H4",
            save_path=OUTPUT_DIR / "equity_ma_crossover_EURUSD_H4.png",
        )

    banner("Done — all outputs saved to ./output/")


if __name__ == "__main__":
    main()
