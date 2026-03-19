from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forex_lab.data import (
    add_basic_indicators,
    attach_cost_model,
    ensure_sample_data,
    load_symbol_data,
    resample_ohlcv,
)
from forex_lab.execution import backtest_signals
from forex_lab.metrics import compute_metrics
from forex_lab.research import ExperimentTracker, bootstrap_returns, parameter_grid, run_walk_forward
from forex_lab.strategies import ma_crossover_signals, rsi_reversal_signals


def _timeframe_to_pandas_rule(timeframe: str) -> str:
    mapping = {"H1": "1h", "H4": "4h", "D1": "1d"}
    if timeframe not in mapping:
        raise ValueError(f"Unsupported timeframe {timeframe}. Use one of {list(mapping)}")
    return mapping[timeframe]


def _periods_per_year(timeframe: str) -> int:
    mapping = {"H1": 24 * 252, "H4": 6 * 252, "D1": 252}
    return mapping[timeframe]


def prepare_dataset(symbol: str, timeframe: str) -> pd.DataFrame:
    data_root = Path("forex_lab/sample_data")
    ensure_sample_data(data_root)
    data = load_symbol_data(data_root, symbols=["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"])
    df = data[symbol]
    if timeframe != "H1":
        df = resample_ohlcv(df, _timeframe_to_pandas_rule(timeframe))
    df = add_basic_indicators(df)
    df = attach_cost_model(df)
    return df.dropna()


def build_ma_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    grid = parameter_grid({"short_window": [10, 20, 30, 40], "long_window": [60, 90, 120, 150]})
    rows = []
    for params in grid:
        if params["short_window"] >= params["long_window"]:
            continue
        signal = ma_crossover_signals(df, params)
        result = backtest_signals(df, signal)
        metrics = compute_metrics(result.equity_curve, result.returns, result.trades, timeframe="H1")
        rows.append({"short_window": params["short_window"], "long_window": params["long_window"], "Sharpe": metrics["Sharpe"]})
    table = pd.DataFrame(rows)
    return table.pivot(index="short_window", columns="long_window", values="Sharpe")


def run_experiment(
    df: pd.DataFrame,
    strategy_name: str,
    signal_fn,
    param_candidates: list[dict],
    timeframe: str,
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    wf = run_walk_forward(
        df=df,
        signal_fn=signal_fn,
        param_candidates=param_candidates,
        train_bars=1500,
        test_bars=500,
        timeframe=timeframe,
    )

    equity = wf.aggregated_result.equity_curve.rename("equity").to_frame()
    drawdown = wf.aggregated_result.drawdown_curve.rename("drawdown").to_frame()
    folds = wf.fold_results.copy()
    metrics = wf.aggregate_metrics
    metrics["strategy"] = strategy_name
    return metrics, equity, drawdown, folds, wf.aggregated_result.returns


def main() -> None:
    symbol = "EURUSD"
    timeframe = "H1"
    output_root = Path("forex_lab/outputs") / f"prototype_{pd.Timestamp.now('UTC').strftime('%Y%m%d_%H%M%S')}"
    output_root.mkdir(parents=True, exist_ok=True)

    df = prepare_dataset(symbol=symbol, timeframe=timeframe)

    tracker = ExperimentTracker(db_path=str(output_root / "experiments.sqlite"))

    ma_params = parameter_grid({"short_window": [10, 20, 30], "long_window": [60, 90, 120]})
    rsi_params = parameter_grid(
        {
            "rsi_period": [10, 14],
            "lower": [25, 30],
            "upper": [70, 75],
            "exit_level": [50],
        }
    )

    metrics_rows = []
    for strategy_name, fn, params in [
        ("ma_crossover", ma_crossover_signals, ma_params),
        ("rsi_reversal", rsi_reversal_signals, rsi_params),
    ]:
        metrics, equity, drawdown, folds, wf_returns = run_experiment(
            df=df,
            strategy_name=strategy_name,
            signal_fn=fn,
            param_candidates=params,
            timeframe=timeframe,
        )
        tracker.log(
            strategy=strategy_name,
            params={"walk_forward_param_candidates": params},
            symbol=symbol,
            timeframe=timeframe,
            metrics=metrics,
        )
        metrics_rows.append(metrics)
        equity.to_csv(output_root / f"{strategy_name}_equity_curve.csv")
        drawdown.to_csv(output_root / f"{strategy_name}_drawdown_curve.csv")
        folds.to_csv(output_root / f"{strategy_name}_walk_forward_folds.csv", index=False)

        # Robustness check via bootstrap on OOS walk-forward returns.
        boot = bootstrap_returns(
            wf_returns,
            n_bootstrap=300,
            ruin_threshold=0.8,
            periods_per_year=_periods_per_year(timeframe),
        )
        boot["distribution"].to_csv(output_root / f"{strategy_name}_bootstrap_distribution.csv", index=False)
        pd.DataFrame([{"risk_of_ruin": boot["risk_of_ruin"]}]).to_csv(
            output_root / f"{strategy_name}_bootstrap_summary.csv", index=False
        )

    metrics_df = pd.DataFrame(metrics_rows).set_index("strategy")
    metrics_df.to_csv(output_root / "metrics_table.csv")

    heatmap = build_ma_heatmap(df.iloc[:1500])
    heatmap.to_csv(output_root / "ma_parameter_robustness_heatmap.csv")
    tracker.to_dataframe().to_csv(output_root / "experiment_tracking.csv", index=False)

    print(f"Prototype complete for {symbol} {timeframe}.")
    print(f"Outputs written to: {output_root}")
    print(metrics_df.round(4))


if __name__ == "__main__":
    main()
