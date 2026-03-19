from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
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
from metrics import compute_metrics
from research import (
    ExperimentTracker,
    bootstrap_returns,
    grid_parameter_sweep,
    run_walk_forward,
)
from strategies.mean_reversion import rsi_reversal_signals
from strategies.trend import ma_crossover_signals


OUTPUT_DIR = Path("outputs")
SAMPLE_DATA_PATH = Path("data/sample_ohlcv.csv")


def _save_equity_and_drawdown(
    timestamps: pd.Series,
    ma_equity: pd.Series,
    rsi_equity: pd.Series,
    ma_dd: pd.Series,
    rsi_dd: pd.Series,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(timestamps, ma_equity.values, label="MA Crossover")
    axes[0].plot(timestamps, rsi_equity.values, label="RSI Reversal")
    axes[0].set_title("Equity Curve")
    axes[0].set_ylabel("Equity")
    axes[0].legend()

    axes[1].plot(timestamps, ma_dd.values, label="MA Crossover DD")
    axes[1].plot(timestamps, rsi_dd.values, label="RSI Reversal DD")
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


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ensure_mock_ohlcv_csv(SAMPLE_DATA_PATH)
    raw = load_ohlcv_csv(SAMPLE_DATA_PATH)
    symbol = "EURUSD"
    timeframe = "H1"

    df = load_symbol_data(raw, symbol=symbol, timeframe=timeframe)
    df = add_basic_indicators(df)
    cost_model = CostModel(spread_bps=0.8, slippage_bps=0.5, commission_bps=0.3)
    df = attach_costs(df, cost_model)
    df = df.dropna().reset_index(drop=True)

    ma_params = {"fast": 20, "slow": 80}
    rsi_params = {"rsi_col": "rsi_14", "oversold": 30, "overbought": 70, "exit_level": 50}

    ma_signal = ma_crossover_signals(df, ma_params)
    rsi_signal = rsi_reversal_signals(df, rsi_params)

    ma_bt = run_backtest(df, ma_signal, cost_model=cost_model)
    rsi_bt = run_backtest(df, rsi_signal, cost_model=cost_model)

    ma_metrics = compute_metrics(ma_bt.returns, ma_bt.equity, ma_bt.trades, timeframe=timeframe)
    rsi_metrics = compute_metrics(
        rsi_bt.returns, rsi_bt.equity, rsi_bt.trades, timeframe=timeframe
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
    )
    wf_rsi = run_walk_forward(
        df=df,
        strategy_fn=rsi_reversal_signals,
        param_grid=rsi_grid,
        train_bars=train_bars,
        test_bars=test_bars,
        cost_model=cost_model,
        timeframe=timeframe,
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
    tracker.log_run("ma_crossover_walk_forward", {}, symbol, timeframe, wf_ma.aggregate_metrics)
    tracker.log_run("rsi_reversal_walk_forward", {}, symbol, timeframe, wf_rsi.aggregate_metrics)

    metrics_table = pd.DataFrame(
        [
            {"Strategy": "MA Crossover (In-Sample)", **ma_metrics},
            {"Strategy": "RSI Reversal (In-Sample)", **rsi_metrics},
            {"Strategy": "MA Crossover (Walk-Forward)", **wf_ma.aggregate_metrics},
            {"Strategy": "RSI Reversal (Walk-Forward)", **wf_rsi.aggregate_metrics},
        ]
    )
    metrics_table.to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False)
    wf_ma.fold_results.to_csv(OUTPUT_DIR / "walk_forward_ma_folds.csv", index=False)
    wf_rsi.fold_results.to_csv(OUTPUT_DIR / "walk_forward_rsi_folds.csv", index=False)
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
        ma_equity=ma_bt.equity,
        rsi_equity=rsi_bt.equity,
        ma_dd=ma_bt.drawdown,
        rsi_dd=rsi_bt.drawdown,
    )
    _save_heatmap(ma_sweep)

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
    print(f"Outputs written to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
