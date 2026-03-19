from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import (
    CostModel,
    add_basic_indicators,
    attach_costs,
    ensure_mock_ohlcv_csv,
    load_ohlcv_csv,
    load_symbol_data,
)
from execution import run_backtest
from metrics import compute_metrics, compute_metrics_by_regime
from regime import attach_regime_labels
from research import (
    ExperimentTracker,
    bootstrap_returns,
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


def _comparison_table(
    baseline: dict[str, dict[str, float]],
    filtered: dict[str, dict[str, float]],
) -> pd.DataFrame:
    rows: list[dict] = []
    tracked_metrics = ("Sharpe", "CAGR", "MaxDrawdown", "TradeCount")
    for strategy_name in ("MA Crossover", "RSI Reversal"):
        for metric in tracked_metrics:
            base_val = float(baseline[strategy_name][metric])
            filt_val = float(filtered[strategy_name][metric])
            rows.append(
                {
                    "Strategy": strategy_name,
                    "Metric": metric,
                    "Baseline": base_val,
                    "Filtered": filt_val,
                    "Delta_Filtered_minus_Baseline": filt_val - base_val,
                }
            )
    return pd.DataFrame(rows)


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

    ma_signal = ma_crossover_signals(df, ma_params)
    rsi_signal = rsi_reversal_signals(df, rsi_params)
    ma_filtered_signal = apply_filter(ma_signal, condition=df["trend_regime"] == "TRENDING")
    rsi_filtered_signal = apply_filter(rsi_signal, condition=df["trend_regime"] == "RANGING")

    ma_bt = run_backtest(df, ma_signal, cost_model=cost_model)
    rsi_bt = run_backtest(df, rsi_signal, cost_model=cost_model)
    ma_filtered_bt = run_backtest(df, ma_filtered_signal, cost_model=cost_model)
    rsi_filtered_bt = run_backtest(df, rsi_filtered_signal, cost_model=cost_model)

    ma_metrics = compute_metrics(ma_bt.returns, ma_bt.equity, ma_bt.trades, timeframe=timeframe)
    rsi_metrics = compute_metrics(
        rsi_bt.returns, rsi_bt.equity, rsi_bt.trades, timeframe=timeframe
    )
    ma_filtered_metrics = compute_metrics(
        ma_filtered_bt.returns, ma_filtered_bt.equity, ma_filtered_bt.trades, timeframe=timeframe
    )
    rsi_filtered_metrics = compute_metrics(
        rsi_filtered_bt.returns, rsi_filtered_bt.equity, rsi_filtered_bt.trades, timeframe=timeframe
    )

    train_bars = 800
    test_bars = 200
    ma_grid = {"fast": [10, 20, 30], "slow": [50, 80, 120]}
    rsi_grid = {"oversold": [25, 30, 35], "overbought": [65, 70, 75], "exit_level": [50]}

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

    ma_bootstrap_dist, ma_bootstrap_summary = bootstrap_returns(
        wf_ma.combined_returns, n_bootstrap=300
    )
    rsi_bootstrap_dist, rsi_bootstrap_summary = bootstrap_returns(
        wf_rsi.combined_returns, n_bootstrap=300
    )

    tracker = ExperimentTracker(OUTPUT_DIR / "experiments.sqlite")
    tracker.log_run("ma_crossover", ma_params, symbol, timeframe, ma_metrics)
    tracker.log_run("rsi_reversal", rsi_params, symbol, timeframe, rsi_metrics)
    tracker.log_run(
        "ma_crossover_filtered_trending_only",
        {"base_params": ma_params, "filter": "trend_regime == TRENDING"},
        symbol,
        timeframe,
        ma_filtered_metrics,
    )
    tracker.log_run(
        "rsi_reversal_filtered_ranging_only",
        {"base_params": rsi_params, "filter": "trend_regime == RANGING"},
        symbol,
        timeframe,
        rsi_filtered_metrics,
    )
    tracker.log_run("ma_crossover_walk_forward", {}, symbol, timeframe, wf_ma.aggregate_metrics)
    tracker.log_run("rsi_reversal_walk_forward", {}, symbol, timeframe, wf_rsi.aggregate_metrics)

    baseline_metrics_map = {"MA Crossover": ma_metrics, "RSI Reversal": rsi_metrics}
    filtered_metrics_map = {
        "MA Crossover": ma_filtered_metrics,
        "RSI Reversal": rsi_filtered_metrics,
    }
    comparison_df = _comparison_table(baseline_metrics_map, filtered_metrics_map)

    ma_regime = _collect_regime_metrics(
        "MA Crossover Baseline", df, ma_bt.returns, ma_bt.position, timeframe
    )
    rsi_regime = _collect_regime_metrics(
        "RSI Reversal Baseline", df, rsi_bt.returns, rsi_bt.position, timeframe
    )
    ma_filtered_regime = _collect_regime_metrics(
        "MA Crossover Filtered (TRENDING)", df, ma_filtered_bt.returns, ma_filtered_bt.position, timeframe
    )
    rsi_filtered_regime = _collect_regime_metrics(
        "RSI Reversal Filtered (RANGING)",
        df,
        rsi_filtered_bt.returns,
        rsi_filtered_bt.position,
        timeframe,
    )

    regime_metrics_all = pd.concat(
        [ma_regime, rsi_regime, ma_filtered_regime, rsi_filtered_regime], ignore_index=True
    )

    metrics_table = pd.DataFrame(
        [
            {"Strategy": "MA Crossover (In-Sample)", **ma_metrics},
            {"Strategy": "RSI Reversal (In-Sample)", **rsi_metrics},
            {"Strategy": "MA Crossover Filtered (In-Sample)", **ma_filtered_metrics},
            {"Strategy": "RSI Reversal Filtered (In-Sample)", **rsi_filtered_metrics},
            {"Strategy": "MA Crossover (Walk-Forward)", **wf_ma.aggregate_metrics},
            {"Strategy": "RSI Reversal (Walk-Forward)", **wf_rsi.aggregate_metrics},
        ]
    )
    metrics_table.to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False)
    wf_ma.fold_results.to_csv(OUTPUT_DIR / "walk_forward_ma_folds.csv", index=False)
    wf_rsi.fold_results.to_csv(OUTPUT_DIR / "walk_forward_rsi_folds.csv", index=False)
    ma_bootstrap_dist.to_csv(OUTPUT_DIR / "bootstrap_ma_distribution.csv", index=False)
    rsi_bootstrap_dist.to_csv(OUTPUT_DIR / "bootstrap_rsi_distribution.csv", index=False)
    regime_metrics_all.to_csv(OUTPUT_DIR / "metrics_by_regime.csv", index=False)
    comparison_df.to_csv(OUTPUT_DIR / "baseline_vs_filtered_comparison.csv", index=False)

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
        },
    )
    _save_heatmap(ma_sweep)
    _save_sharpe_by_regime_chart(regime_metrics_all)

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
    print("\nBaseline vs filtered comparison:")
    print(comparison_df.round(4).to_string(index=False))
    print(f"Outputs written to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
