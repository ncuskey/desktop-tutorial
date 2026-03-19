from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from combinations import weighted_ensemble_signals
from data import (
    CostModel,
    add_basic_indicators,
    attach_costs,
    ensure_mock_ohlcv_csv,
    load_ohlcv_csv,
    load_symbol_data,
)
from execution import (
    apply_no_trade_filter_high_vol,
    apply_volatility_targeting,
    run_backtest,
)
from metrics import compute_metrics, compute_metrics_by_regime
from orchestrators import RegimeSpecialistOrchestrator
from regime import attach_regime_labels
from research import (
    ExperimentTracker,
    bootstrap_returns,
    compute_switch_diagnostics,
    grid_parameter_sweep,
    run_walk_forward,
)
from strategies.filters import apply_filter
from strategies.mean_reversion import rsi_reversal_signals
from strategies.trend import ma_crossover_signals


OUTPUT_DIR = Path("outputs")
SAMPLE_DATA_PATH = Path("data/sample_ohlcv.csv")


def _save_equity_and_drawdown(timestamps: pd.Series, curve_map: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for name, curve in curve_map.items():
        axes[0].plot(timestamps, curve["equity"].values, label=name)
    axes[0].set_title("Equity Curve")
    axes[0].set_ylabel("Equity")
    axes[0].legend()

    for name, curve in curve_map.items():
        axes[1].plot(timestamps, curve["drawdown"].values, label=f"{name} DD")
    axes[1].set_title("Drawdown Curve")
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Time")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "equity_drawdown_curves.png", dpi=150)
    plt.close(fig)


def _save_heatmap(sweep_df: pd.DataFrame) -> None:
    rows = []
    for _, row in sweep_df.iterrows():
        params = row["params"]
        rows.append({"fast": params["fast"], "slow": params["slow"], "Sharpe": row["Sharpe"]})
    heatmap_df = pd.DataFrame(rows).pivot(index="fast", columns="slow", values="Sharpe")

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(heatmap_df.values, aspect="auto")
    ax.set_xticks(range(len(heatmap_df.columns)))
    ax.set_xticklabels(heatmap_df.columns.tolist())
    ax.set_yticks(range(len(heatmap_df.index)))
    ax.set_yticklabels(heatmap_df.index.tolist())
    ax.set_xlabel("Slow MA")
    ax.set_ylabel("Fast MA")
    ax.set_title("Parameter Robustness Heatmap (MA Crossover Sharpe)")

    for i in range(heatmap_df.shape[0]):
        for j in range(heatmap_df.shape[1]):
            ax.text(j, i, f"{heatmap_df.iloc[i, j]:.2f}", ha="center", va="center", color="w")

    fig.colorbar(im, ax=ax, label="Sharpe")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "ma_parameter_heatmap.png", dpi=150)
    plt.close(fig)


def _save_sharpe_by_regime_chart(metrics_by_regime: pd.DataFrame) -> None:
    trend_rows = metrics_by_regime[metrics_by_regime["RegimeColumn"] == "trend_regime"].copy()
    if trend_rows.empty:
        return

    pivot = trend_rows.pivot(index="Regime", columns="Strategy", values="Sharpe").fillna(0.0)
    x = np.arange(len(pivot.index))
    width = 0.8 / max(len(pivot.columns), 1)

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, col in enumerate(pivot.columns):
        ax.bar(x + (i - (len(pivot.columns) - 1) / 2) * width, pivot[col], width=width, label=col)

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index.tolist())
    ax.set_ylabel("Sharpe")
    ax.set_title("Sharpe by Regime per Strategy")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "sharpe_by_regime_strategy.png", dpi=150)
    plt.close(fig)


def _analysis_frame(
    df: pd.DataFrame,
    returns: pd.Series,
    position: pd.Series,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": df["timestamp"].values,
            "returns": returns.values,
            "position": position.values,
            "trend_regime": df["trend_regime"].values,
            "vol_regime": df["vol_regime"].values,
            "regime_label": df["regime_label"].values,
        }
    )


def _strategy_metrics_row(
    strategy_name: str,
    metrics: dict[str, float],
    evaluation: str = "InSample",
) -> dict:
    return {"Strategy": strategy_name, "Evaluation": evaluation, **metrics}


def _build_strategy_comparison_table(rows: list[dict]) -> pd.DataFrame:
    table = pd.DataFrame(rows)
    desired_cols = [
        "Strategy",
        "Evaluation",
        "CAGR",
        "Sharpe",
        "Sortino",
        "MaxDrawdown",
        "ProfitFactor",
        "WinRate",
        "Expectancy",
        "TradeCount",
        "ExposurePct",
        "AvgHoldingBars",
    ]
    cols = [c for c in desired_cols if c in table.columns]
    return table[cols].sort_values(["Evaluation", "Strategy"]).reset_index(drop=True)


def _orchestrator_regime_map() -> dict[str, str]:
    # Keep HIGH_VOL regimes intentionally unmapped to stay flat.
    return {
        "TRENDING_LOW_VOL": "trend_sleeve",
        "TRENDING_MID_VOL": "trend_sleeve",
        "RANGING_LOW_VOL": "mean_reversion_sleeve",
        "RANGING_MID_VOL": "mean_reversion_sleeve",
    }


def _save_orchestrated_equity_curve(
    timestamps: pd.Series,
    orchestrated_equity: pd.Series,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(timestamps, orchestrated_equity.values, label="Orchestrated Specialist")
    ax.set_title("Orchestrated Specialist Equity Curve")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "orchestrated_equity_curve.png", dpi=150)
    plt.close(fig)


def _save_pnl_by_regime_chart(regime_attribution: pd.DataFrame) -> None:
    label_rows = regime_attribution[regime_attribution["RegimeColumn"] == "regime_label"].copy()
    if label_rows.empty:
        return
    pivot = label_rows.pivot(index="Regime", columns="Strategy", values="PnL").fillna(0.0)

    x = np.arange(len(pivot.index))
    width = 0.8 / max(len(pivot.columns), 1)
    fig, ax = plt.subplots(figsize=(13, 7))
    for i, col in enumerate(pivot.columns):
        offset = (i - (len(pivot.columns) - 1) / 2.0) * width
        ax.bar(x + offset, pivot[col].values, width=width, label=col)

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index.tolist(), rotation=30, ha="right")
    ax.set_ylabel("PnL (sum of net returns)")
    ax.set_title("PnL by Regime")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pnl_by_regime.png", dpi=150)
    plt.close(fig)


def _collect_regime_metrics(
    strategy_name: str,
    base_df: pd.DataFrame,
    returns: pd.Series,
    position: pd.Series,
    timeframe: str,
) -> pd.DataFrame:
    analysis_df = _analysis_frame(base_df, returns, position)
    frames: list[pd.DataFrame] = []
    for regime_column in ("trend_regime", "vol_regime", "regime_label"):
        m = compute_metrics_by_regime(
            analysis_df,
            regime_column=regime_column,
            timeframe=timeframe,
        )
        m["Strategy"] = strategy_name
        m["RegimeColumn"] = regime_column
        frames.append(m)
    return pd.concat(frames, ignore_index=True)


def _build_specialist_signals(
    df: pd.DataFrame,
    ma_signal: pd.Series,
    rsi_signal: pd.Series,
    allow_high_vol_entries: bool = False,
) -> tuple[pd.Series, pd.Series]:
    ma_filtered = apply_filter(ma_signal, condition=df["trend_regime"] == "TRENDING")
    rsi_filtered = apply_filter(rsi_signal, condition=df["trend_regime"] == "RANGING")
    ma_filtered = apply_no_trade_filter_high_vol(
        ma_filtered, vol_regime=df["vol_regime"], allow_high_vol=allow_high_vol_entries
    )
    rsi_filtered = apply_no_trade_filter_high_vol(
        rsi_filtered, vol_regime=df["vol_regime"], allow_high_vol=allow_high_vol_entries
    )
    return ma_filtered.astype(float), rsi_filtered.astype(float)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ensure_mock_ohlcv_csv(SAMPLE_DATA_PATH)
    raw = load_ohlcv_csv(SAMPLE_DATA_PATH)
    symbol = "EURUSD"
    timeframe = "H1"

    df = load_symbol_data(raw, symbol=symbol, timeframe=timeframe)
    df = add_basic_indicators(df)
    df = attach_regime_labels(df, adx_threshold=25.0)
    cost_model = CostModel(spread_bps=0.8, slippage_bps=0.5, commission_bps=0.3)
    df = attach_costs(df, cost_model)
    df = df.dropna(
        subset=[
            "ma_fast_20",
            "ma_slow_50",
            "rsi_14",
            "atr_14",
            "adx_14",
            "bb_upper_20_2",
            "bb_lower_20_2",
        ]
    ).reset_index(drop=True)

    ma_params = {"fast": 20, "slow": 80}
    rsi_params = {"rsi_col": "rsi_14", "oversold": 35, "overbought": 65, "exit_level": 50}
    allow_high_vol_entries = False

    ma_signal = ma_crossover_signals(df, ma_params).astype(float)
    rsi_signal = rsi_reversal_signals(df, rsi_params).astype(float)
    ma_filtered_signal, rsi_filtered_signal = _build_specialist_signals(
        df, ma_signal, rsi_signal, allow_high_vol_entries=allow_high_vol_entries
    )

    filtered_ensemble_signal = weighted_ensemble_signals(
        {"ma_filtered": ma_filtered_signal, "rsi_filtered": rsi_filtered_signal},
        weights={"ma_filtered": 0.5, "rsi_filtered": 0.5},
        threshold=0.0,
    ).astype(float)

    specialist_orchestrator = RegimeSpecialistOrchestrator(
        regime_column="regime_label",
        vol_regime_column="vol_regime",
        regime_to_sleeve=_orchestrator_regime_map(),
        fallback="flat",
        sleeve_weights={"trend_sleeve": 1.0, "mean_reversion_sleeve": 1.0},
        allow_high_vol_entries=allow_high_vol_entries,
        use_vol_targeting=True,
        target_atr_norm=0.001,
        max_leverage=1.0,
    )
    orchestrated_signal = specialist_orchestrator.orchestrate(
        df,
        sleeve_signals={
            "trend_sleeve": ma_filtered_signal,
            "mean_reversion_sleeve": rsi_filtered_signal,
        },
    )

    ma_bt = run_backtest(df, ma_signal, cost_model=cost_model)
    rsi_bt = run_backtest(df, rsi_signal, cost_model=cost_model)
    ma_filtered_bt = run_backtest(df, ma_filtered_signal, cost_model=cost_model)
    rsi_filtered_bt = run_backtest(df, rsi_filtered_signal, cost_model=cost_model)
    ensemble_bt = run_backtest(df, filtered_ensemble_signal, cost_model=cost_model)
    orchestrated_bt = run_backtest(df, orchestrated_signal, cost_model=cost_model)

    strategy_results = {
        "MA Baseline": ma_bt,
        "RSI Baseline": rsi_bt,
        "MA Specialist (TRENDING)": ma_filtered_bt,
        "RSI Specialist (RANGING)": rsi_filtered_bt,
        "Filtered Ensemble (Equal Weight)": ensemble_bt,
        "Regime Specialist Orchestrated": orchestrated_bt,
    }
    strategy_metrics = {
        name: compute_metrics(
            bt.returns,
            bt.equity,
            bt.trades,
            timeframe=timeframe,
            position=bt.position,
        )
        for name, bt in strategy_results.items()
    }

    train_bars = 800
    test_bars = 200
    ma_grid = {"fast": [10, 20, 30], "slow": [50, 80, 120]}
    rsi_grid = {"oversold": [25, 30, 35], "overbought": [65, 70, 75], "exit_level": [50]}
    orchestrated_grid = {
        "ma_fast": [10, 20],
        "ma_slow": [50, 80],
        "rsi_oversold": [30, 35],
        "rsi_overbought": [65, 70],
        "rsi_exit": [50],
    }

    def _ma_filtered_strategy(frame: pd.DataFrame, params: dict) -> pd.Series:
        base = ma_crossover_signals(frame, params).astype(float)
        filtered, _ = _build_specialist_signals(
            frame, base, rsi_signal=pd.Series(0.0, index=frame.index), allow_high_vol_entries=False
        )
        return filtered

    def _rsi_filtered_strategy(frame: pd.DataFrame, params: dict) -> pd.Series:
        base = rsi_reversal_signals(frame, params).astype(float)
        _, filtered = _build_specialist_signals(
            frame, ma_signal=pd.Series(0.0, index=frame.index), rsi_signal=base, allow_high_vol_entries=False
        )
        return filtered

    def _orchestrated_strategy(frame: pd.DataFrame, params: dict) -> pd.Series:
        ma_local = ma_crossover_signals(
            frame,
            {"fast": int(params["ma_fast"]), "slow": int(params["ma_slow"])},
        ).astype(float)
        rsi_local = rsi_reversal_signals(
            frame,
            {
                "rsi_col": "rsi_14",
                "oversold": float(params["rsi_oversold"]),
                "overbought": float(params["rsi_overbought"]),
                "exit_level": float(params["rsi_exit"]),
            },
        ).astype(float)
        ma_spec, rsi_spec = _build_specialist_signals(
            frame, ma_local, rsi_local, allow_high_vol_entries=False
        )
        local_orchestrator = RegimeSpecialistOrchestrator(
            regime_column="regime_label",
            vol_regime_column="vol_regime",
            regime_to_sleeve=_orchestrator_regime_map(),
            fallback="flat",
            sleeve_weights={"trend_sleeve": 1.0, "mean_reversion_sleeve": 1.0},
            allow_high_vol_entries=False,
            use_vol_targeting=True,
            target_atr_norm=0.001,
            max_leverage=1.0,
        )
        return local_orchestrator.orchestrate(
            frame,
            sleeve_signals={
                "trend_sleeve": ma_spec,
                "mean_reversion_sleeve": rsi_spec,
            },
        )

    wf_ma = run_walk_forward(
        df=df,
        strategy_fn=ma_crossover_signals,
        param_grid=ma_grid,
        train_bars=train_bars,
        test_bars=test_bars,
        cost_model=cost_model,
        timeframe=timeframe,
        regime_column="regime_label",
    )
    wf_rsi = run_walk_forward(
        df=df,
        strategy_fn=rsi_reversal_signals,
        param_grid=rsi_grid,
        train_bars=train_bars,
        test_bars=test_bars,
        cost_model=cost_model,
        timeframe=timeframe,
        regime_column="regime_label",
    )
    wf_ma_filtered = run_walk_forward(
        df=df,
        strategy_fn=_ma_filtered_strategy,
        param_grid=ma_grid,
        train_bars=train_bars,
        test_bars=test_bars,
        cost_model=cost_model,
        timeframe=timeframe,
        regime_column="regime_label",
    )
    wf_rsi_filtered = run_walk_forward(
        df=df,
        strategy_fn=_rsi_filtered_strategy,
        param_grid=rsi_grid,
        train_bars=train_bars,
        test_bars=test_bars,
        cost_model=cost_model,
        timeframe=timeframe,
        regime_column="regime_label",
    )
    wf_orchestrated = run_walk_forward(
        df=df,
        strategy_fn=_orchestrated_strategy,
        param_grid=orchestrated_grid,
        train_bars=train_bars,
        test_bars=test_bars,
        cost_model=cost_model,
        timeframe=timeframe,
        regime_column="regime_label",
    )

    ma_bootstrap_dist, ma_bootstrap_summary = bootstrap_returns(
        wf_ma.combined_returns, n_bootstrap=300
    )
    rsi_bootstrap_dist, rsi_bootstrap_summary = bootstrap_returns(
        wf_rsi.combined_returns, n_bootstrap=300
    )

    tracker = ExperimentTracker(OUTPUT_DIR / "experiments.sqlite")
    tracker.log_run("ma_crossover", ma_params, symbol, timeframe, strategy_metrics["MA Baseline"])
    tracker.log_run("rsi_reversal", rsi_params, symbol, timeframe, strategy_metrics["RSI Baseline"])
    tracker.log_run(
        "ma_specialist_trending",
        {"base_params": ma_params, "filter": "trend_regime == TRENDING"},
        symbol,
        timeframe,
        strategy_metrics["MA Specialist (TRENDING)"],
    )
    tracker.log_run(
        "rsi_specialist_ranging",
        {"base_params": rsi_params, "filter": "trend_regime == RANGING"},
        symbol,
        timeframe,
        strategy_metrics["RSI Specialist (RANGING)"],
    )
    tracker.log_run(
        "regime_specialist_orchestrated",
        {"regime_to_sleeve": _orchestrator_regime_map(), "vol_targeting": True},
        symbol,
        timeframe,
        strategy_metrics["Regime Specialist Orchestrated"],
    )
    tracker.log_run("ma_filtered_walk_forward", {}, symbol, timeframe, wf_ma_filtered.aggregate_metrics)
    tracker.log_run(
        "rsi_filtered_walk_forward", {}, symbol, timeframe, wf_rsi_filtered.aggregate_metrics
    )
    tracker.log_run(
        "orchestrated_walk_forward", {}, symbol, timeframe, wf_orchestrated.aggregate_metrics
    )

    regime_attribution = pd.concat(
        [
            _collect_regime_metrics("MA Baseline", df, ma_bt.returns, ma_bt.position, timeframe),
            _collect_regime_metrics("RSI Baseline", df, rsi_bt.returns, rsi_bt.position, timeframe),
            _collect_regime_metrics(
                "MA Specialist (TRENDING)", df, ma_filtered_bt.returns, ma_filtered_bt.position, timeframe
            ),
            _collect_regime_metrics(
                "RSI Specialist (RANGING)",
                df,
                rsi_filtered_bt.returns,
                rsi_filtered_bt.position,
                timeframe,
            ),
            _collect_regime_metrics(
                "Filtered Ensemble (Equal Weight)",
                df,
                ensemble_bt.returns,
                ensemble_bt.position,
                timeframe,
            ),
            _collect_regime_metrics(
                "Regime Specialist Orchestrated",
                df,
                orchestrated_bt.returns,
                orchestrated_bt.position,
                timeframe,
            ),
        ],
        ignore_index=True,
    )

    switch_diagnostics = compute_switch_diagnostics(df["regime_label"], orchestrated_bt.returns, n_bars=24)

    comparison_rows = [
        _strategy_metrics_row("MA Baseline", strategy_metrics["MA Baseline"], "InSample"),
        _strategy_metrics_row("RSI Baseline", strategy_metrics["RSI Baseline"], "InSample"),
        _strategy_metrics_row(
            "MA Specialist (TRENDING)", strategy_metrics["MA Specialist (TRENDING)"], "InSample"
        ),
        _strategy_metrics_row(
            "RSI Specialist (RANGING)", strategy_metrics["RSI Specialist (RANGING)"], "InSample"
        ),
        _strategy_metrics_row(
            "Filtered Ensemble (Equal Weight)",
            strategy_metrics["Filtered Ensemble (Equal Weight)"],
            "InSample",
        ),
        _strategy_metrics_row(
            "Regime Specialist Orchestrated",
            strategy_metrics["Regime Specialist Orchestrated"],
            "InSample",
        ),
        _strategy_metrics_row("MA Baseline", wf_ma.aggregate_metrics, "WalkForward"),
        _strategy_metrics_row("RSI Baseline", wf_rsi.aggregate_metrics, "WalkForward"),
        _strategy_metrics_row(
            "MA Specialist (TRENDING)", wf_ma_filtered.aggregate_metrics, "WalkForward"
        ),
        _strategy_metrics_row(
            "RSI Specialist (RANGING)", wf_rsi_filtered.aggregate_metrics, "WalkForward"
        ),
        _strategy_metrics_row(
            "Regime Specialist Orchestrated", wf_orchestrated.aggregate_metrics, "WalkForward"
        ),
    ]
    specialist_vs_orchestrated = _build_strategy_comparison_table(comparison_rows)

    metrics_table = pd.DataFrame(
        [
            {"Strategy": "MA Crossover (In-Sample)", **strategy_metrics["MA Baseline"]},
            {"Strategy": "RSI Reversal (In-Sample)", **strategy_metrics["RSI Baseline"]},
            {"Strategy": "MA Crossover Filtered (In-Sample)", **strategy_metrics["MA Specialist (TRENDING)"]},
            {"Strategy": "RSI Reversal Filtered (In-Sample)", **strategy_metrics["RSI Specialist (RANGING)"]},
            {"Strategy": "Orchestrated Specialist (In-Sample)", **strategy_metrics["Regime Specialist Orchestrated"]},
            {"Strategy": "MA Crossover (Walk-Forward)", **wf_ma.aggregate_metrics},
            {"Strategy": "RSI Reversal (Walk-Forward)", **wf_rsi.aggregate_metrics},
            {"Strategy": "MA Filtered (Walk-Forward)", **wf_ma_filtered.aggregate_metrics},
            {"Strategy": "RSI Filtered (Walk-Forward)", **wf_rsi_filtered.aggregate_metrics},
            {"Strategy": "Orchestrated Specialist (Walk-Forward)", **wf_orchestrated.aggregate_metrics},
        ]
    )

    metrics_table.to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False)
    regime_attribution.to_csv(OUTPUT_DIR / "regime_attribution.csv", index=False)
    regime_attribution.to_csv(OUTPUT_DIR / "metrics_by_regime.csv", index=False)
    specialist_vs_orchestrated.to_csv(
        OUTPUT_DIR / "specialist_vs_orchestrated_comparison.csv", index=False
    )
    switch_diagnostics.to_csv(OUTPUT_DIR / "switch_diagnostics.csv", index=False)
    wf_ma.fold_results.to_csv(OUTPUT_DIR / "walk_forward_ma_folds.csv", index=False)
    wf_rsi.fold_results.to_csv(OUTPUT_DIR / "walk_forward_rsi_folds.csv", index=False)
    wf_ma_filtered.fold_results.to_csv(OUTPUT_DIR / "walk_forward_ma_filtered_folds.csv", index=False)
    wf_rsi_filtered.fold_results.to_csv(OUTPUT_DIR / "walk_forward_rsi_filtered_folds.csv", index=False)
    wf_orchestrated.fold_results.to_csv(OUTPUT_DIR / "walk_forward_orchestrated_folds.csv", index=False)
    ma_bootstrap_dist.to_csv(OUTPUT_DIR / "bootstrap_ma_distribution.csv", index=False)
    rsi_bootstrap_dist.to_csv(OUTPUT_DIR / "bootstrap_rsi_distribution.csv", index=False)

    ma_sweep = grid_parameter_sweep(
        df=df,
        strategy_fn=ma_crossover_signals,
        param_grid=ma_grid,
        cost_model=cost_model,
        timeframe=timeframe,
    )
    ma_sweep.to_csv(OUTPUT_DIR / "ma_parameter_sweep.csv", index=False)

    _save_equity_and_drawdown(
        timestamps=df["timestamp"],
        curve_map={
            "MA Crossover": pd.DataFrame({"equity": ma_bt.equity, "drawdown": ma_bt.drawdown}),
            "RSI Reversal": pd.DataFrame({"equity": rsi_bt.equity, "drawdown": rsi_bt.drawdown}),
            "MA Filtered": pd.DataFrame(
                {"equity": ma_filtered_bt.equity, "drawdown": ma_filtered_bt.drawdown}
            ),
            "RSI Filtered": pd.DataFrame(
                {"equity": rsi_filtered_bt.equity, "drawdown": rsi_filtered_bt.drawdown}
            ),
            "Orchestrated": pd.DataFrame(
                {"equity": orchestrated_bt.equity, "drawdown": orchestrated_bt.drawdown}
            ),
        },
    )
    _save_orchestrated_equity_curve(df["timestamp"], orchestrated_bt.equity)
    _save_heatmap(ma_sweep)
    _save_sharpe_by_regime_chart(regime_attribution)
    _save_pnl_by_regime_chart(regime_attribution)

    with open(OUTPUT_DIR / "bootstrap_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "ma": ma_bootstrap_summary,
                "rsi": rsi_bootstrap_summary,
            },
            f,
            indent=2,
        )

    print("Prototype run completed.")
    print(metrics_table.round(4).to_string(index=False))
    print("\nSpecialist vs orchestrated comparison:")
    print(specialist_vs_orchestrated.round(4).to_string(index=False))
    print(f"Outputs written to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
