from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from data import CostModel, load_ohlcv_csv
from execution.simulator import run_backtest
from research.r14_execution_layer import (
    _apply_execution_layer_to_fold,
    _extract_trade_segments,
    _fit_fold_meta_model,
    _load_hardened_folds,
    _load_hardened_params,
    _prepare_symbol_frame,
    _resolve_price_data_path,
    _segment_return,
)
from strategies import trend_breakout_v2_signals


@dataclass
class R14ExecutionDiagnosticsArtifacts:
    threshold_sweep: pd.DataFrame
    percentile_sweep: pd.DataFrame
    score_bins: pd.DataFrame
    separation_tests: pd.DataFrame
    output_dir: Path


def run_threshold_sweep(df: pd.DataFrame, thresholds: np.ndarray) -> pd.DataFrame:
    results: list[dict[str, Any]] = []
    base = df.copy()
    base = base[pd.to_numeric(base["meta_score"], errors="coerce").notna()].copy()
    if base.empty:
        return pd.DataFrame(
            columns=["threshold", "pass_rate", "n_pass", "n_fail", "E_pass", "E_fail", "separation"]
        )

    base["meta_score"] = pd.to_numeric(base["meta_score"], errors="coerce")
    base["realized_return"] = pd.to_numeric(base["realized_return"], errors="coerce")
    base = base.dropna(subset=["meta_score", "realized_return"])

    for t in thresholds:
        pass_mask = base["meta_score"] >= float(t)
        pass_returns = base.loc[pass_mask, "realized_return"]
        fail_returns = base.loc[~pass_mask, "realized_return"]
        e_pass = float(pass_returns.mean()) if not pass_returns.empty else np.nan
        e_fail = float(fail_returns.mean()) if not fail_returns.empty else np.nan
        results.append(
            {
                "threshold": float(t),
                "pass_rate": float(pass_mask.mean()),
                "n_pass": int(pass_returns.count()),
                "n_fail": int(fail_returns.count()),
                "E_pass": e_pass,
                "E_fail": e_fail,
                "separation": float(e_pass - e_fail) if np.isfinite(e_pass) and np.isfinite(e_fail) else np.nan,
            }
        )
    return pd.DataFrame(results)


def run_percentile_sweep(df: pd.DataFrame, percentiles: list[float]) -> pd.DataFrame:
    results: list[dict[str, Any]] = []
    base = df.copy()
    base["meta_score"] = pd.to_numeric(base["meta_score"], errors="coerce")
    base["realized_return"] = pd.to_numeric(base["realized_return"], errors="coerce")
    base = base.dropna(subset=["meta_score", "realized_return"])
    if base.empty:
        return pd.DataFrame(
            columns=["percentile", "threshold", "pass_rate", "n_pass", "n_fail", "E_pass", "E_fail", "separation"]
        )

    for p in percentiles:
        q = float(np.clip(p, 0.0, 1.0))
        threshold = float(base["meta_score"].quantile(q))
        pass_mask = base["meta_score"] >= threshold
        pass_returns = base.loc[pass_mask, "realized_return"]
        fail_returns = base.loc[~pass_mask, "realized_return"]
        e_pass = float(pass_returns.mean()) if not pass_returns.empty else np.nan
        e_fail = float(fail_returns.mean()) if not fail_returns.empty else np.nan
        results.append(
            {
                "percentile": q,
                "threshold": threshold,
                "pass_rate": float(pass_mask.mean()),
                "n_pass": int(pass_returns.count()),
                "n_fail": int(fail_returns.count()),
                "E_pass": e_pass,
                "E_fail": e_fail,
                "separation": float(e_pass - e_fail) if np.isfinite(e_pass) and np.isfinite(e_fail) else np.nan,
            }
        )
    return pd.DataFrame(results)


def score_buckets(df: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    base = df.copy()
    base["meta_score"] = pd.to_numeric(base["meta_score"], errors="coerce")
    base["realized_return"] = pd.to_numeric(base["realized_return"], errors="coerce")
    base = base.dropna(subset=["meta_score", "realized_return"])
    if base.empty:
        return pd.DataFrame(columns=["score_bin", "mean", "count", "std", "bin_mid"])

    bins = max(int(n_bins), 2)
    base["score_bin"] = pd.qcut(base["meta_score"], bins, duplicates="drop")
    grouped = (
        base.groupby("score_bin", observed=False)["realized_return"]
        .agg(["mean", "count", "std"])
        .reset_index()
    )
    grouped["bin_mid"] = grouped["score_bin"].apply(lambda x: float(x.mid) if pd.notna(x) else np.nan)
    return grouped


def separation_test(df: pd.DataFrame, threshold: float) -> dict[str, Any]:
    base = df.copy()
    base["meta_score"] = pd.to_numeric(base["meta_score"], errors="coerce")
    base["realized_return"] = pd.to_numeric(base["realized_return"], errors="coerce")
    base = base.dropna(subset=["meta_score", "realized_return"])
    if base.empty:
        return {
            "threshold": float(threshold),
            "p_value": np.nan,
            "E_pass": np.nan,
            "E_fail": np.nan,
            "n_pass": 0,
            "n_fail": 0,
        }

    pass_mask = base["meta_score"] >= float(threshold)
    pass_returns = base.loc[pass_mask, "realized_return"]
    fail_returns = base.loc[~pass_mask, "realized_return"]
    if pass_returns.empty or fail_returns.empty:
        p_value = np.nan
    else:
        _stat, p_value = mannwhitneyu(pass_returns, fail_returns, alternative="greater")
    return {
        "threshold": float(threshold),
        "p_value": float(p_value) if p_value is not None and np.isfinite(p_value) else np.nan,
        "E_pass": float(pass_returns.mean()) if not pass_returns.empty else np.nan,
        "E_fail": float(fail_returns.mean()) if not fail_returns.empty else np.nan,
        "n_pass": int(pass_returns.count()),
        "n_fail": int(fail_returns.count()),
    }


def _build_trade_level_dataset_from_execution(
    input_df: pd.DataFrame,
    artifacts_root: Path,
    source_csv: str | Path | None,
    timeframe: str,
    output_dir: Path,
    early_window: int,
    meta_threshold: float,
    scale_threshold: float,
    allow_fallback_scorer: bool,
    meta_min_train_samples: int,
) -> pd.DataFrame:
    symbols = (
        input_df["symbol"].dropna().astype(str).unique().tolist()
        if "symbol" in input_df.columns
        else ["EURUSD", "GBPUSD", "AUDUSD"]
    )
    price_path = _resolve_price_data_path(
        artifacts_root=artifacts_root,
        explicit_source_csv=source_csv,
        symbols=symbols,
    )
    raw_prices = load_ohlcv_csv(price_path)
    cost_model = CostModel(spread_bps=0.8, slippage_bps=0.5, commission_bps=0.2)
    rows: list[dict[str, Any]] = []

    for symbol in symbols:
        strategy_params = _load_hardened_params(artifacts_root, symbol=symbol)
        fold_windows = _load_hardened_folds(
            artifacts_root=artifacts_root,
            symbol=symbol,
            hardened_params=strategy_params,
        )
        frame = _prepare_symbol_frame(
            raw_prices=raw_prices,
            symbol=symbol,
            timeframe=timeframe,
            cost_model=cost_model,
        )
        if frame.empty:
            continue

        for _, fold in fold_windows.iterrows():
            fold_id = int(fold["fold_id"])
            train_start = pd.Timestamp(fold["fold_start"])
            train_end = pd.Timestamp(fold["fold_train_end"])
            test_start = pd.Timestamp(fold["fold_test_start"])
            test_end = pd.Timestamp(fold["fold_test_end"])
            train_df = frame[(frame["timestamp"] >= train_start) & (frame["timestamp"] <= train_end)].copy()
            test_df = frame[(frame["timestamp"] >= test_start) & (frame["timestamp"] <= test_end)].copy()
            if train_df.empty or test_df.empty:
                continue

            model, _train_meta, fallback_state = _fit_fold_meta_model(
                train_df=train_df,
                strategy_params=strategy_params,
                early_window=early_window,
                forward_horizon=24,
                label_method="top_quantile",
                label_quantile=0.30,
                meta_min_train_samples=meta_min_train_samples,
                allow_fallback_scorer=allow_fallback_scorer,
            )
            raw_signal = trend_breakout_v2_signals(test_df, strategy_params).astype(float)
            _exec_signal, decisions = _apply_execution_layer_to_fold(
                test_df=test_df,
                strategy_params=strategy_params,
                model=model,
                fallback_state=fallback_state,
                early_window=early_window,
                meta_threshold=meta_threshold,
                scale_threshold=scale_threshold,
                scale_factor=1.5,
                enable_scaling=True,
                min_hold_bars=early_window,
            )
            bt_base = run_backtest(test_df, raw_signal, cost_model=cost_model)
            segments = _extract_trade_segments(raw_signal)

            for seg_id, (entry_i, end_i, side) in enumerate(segments):
                realized_return = _segment_return(bt_base.returns, entry_i, end_i)
                decision_row = decisions[
                    (pd.to_numeric(decisions["entry_i"], errors="coerce") == float(entry_i))
                    & (pd.to_numeric(decisions["end_i"], errors="coerce") == float(end_i))
                ]
                if decision_row.empty:
                    decision_row = decisions[
                        pd.to_numeric(decisions["entry_i"], errors="coerce") == float(entry_i)
                    ]
                if decision_row.empty:
                    meta_score = np.nan
                    decision = "continue_missing"
                else:
                    meta_score = float(pd.to_numeric(decision_row.iloc[0].get("meta_score"), errors="coerce"))
                    decision = str(decision_row.iloc[0].get("decision", "continue_missing"))

                rows.append(
                    {
                        "trade_id": f"{symbol}_{fold_id}_{seg_id}",
                        "symbol": symbol,
                        "fold_id": fold_id,
                        "trade_side": int(np.sign(side)),
                        "meta_score": meta_score,
                        "realized_return": float(realized_return),
                        "early_exit_flag": int(decision == "early_fail_exit"),
                        "decision": decision,
                    }
                )

    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "r14_execution_trade_scores.csv", index=False)
    return out


def _plot_line(
    x: pd.Series,
    y: pd.Series,
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, marker="o")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_r14_execution_diagnostics(
    input_path: str | Path = "outputs/r14_execution_fold_results.csv",
    artifacts_root: str | Path = "outputs/TrendBreakout_V2",
    source_csv: str | Path | None = None,
    timeframe: str = "H1",
    output_dir: str | Path = "outputs",
    early_window: int = 3,
    meta_threshold: float = 0.6,
    scale_threshold: float = 0.75,
    n_bins: int = 5,
    allow_fallback_scorer: bool = True,
    meta_min_train_samples: int = 30,
) -> R14ExecutionDiagnosticsArtifacts:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = out_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    input_df = pd.read_csv(input_path)
    required_trade_cols = {"trade_id", "fold_id", "meta_score", "realized_return", "early_exit_flag", "symbol"}

    if required_trade_cols.issubset(input_df.columns):
        trade_df = input_df.copy()
    else:
        trade_df = _build_trade_level_dataset_from_execution(
            input_df=input_df,
            artifacts_root=Path(artifacts_root),
            source_csv=source_csv,
            timeframe=timeframe,
            output_dir=out_dir,
            early_window=early_window,
            meta_threshold=meta_threshold,
            scale_threshold=scale_threshold,
            allow_fallback_scorer=allow_fallback_scorer,
            meta_min_train_samples=meta_min_train_samples,
        )

    thresholds = np.linspace(0.5, 0.9, 9)
    percentiles = [0.5, 0.6, 0.7, 0.8, 0.9]

    threshold_sweep = run_threshold_sweep(trade_df, thresholds)
    percentile_sweep = run_percentile_sweep(trade_df, percentiles)
    score_bins = score_buckets(trade_df, n_bins=n_bins)
    separation_tests = pd.DataFrame([separation_test(trade_df, t) for t in thresholds])

    threshold_sweep.to_csv(out_dir / "r14_execution_threshold_sweep.csv", index=False)
    percentile_sweep.to_csv(out_dir / "r14_execution_percentile_sweep.csv", index=False)
    score_bins.to_csv(out_dir / "r14_execution_score_bins.csv", index=False)
    separation_tests.to_csv(out_dir / "r14_execution_separation_tests.csv", index=False)

    if not threshold_sweep.empty:
        _plot_line(
            x=threshold_sweep["threshold"],
            y=threshold_sweep["separation"],
            output_path=charts_dir / "r14_threshold_vs_separation.png",
            title="Threshold vs Separation",
            xlabel="Threshold",
            ylabel="Separation (E_pass - E_fail)",
        )
    if not percentile_sweep.empty:
        _plot_line(
            x=percentile_sweep["percentile"],
            y=percentile_sweep["separation"],
            output_path=charts_dir / "r14_percentile_vs_separation.png",
            title="Percentile vs Separation",
            xlabel="Percentile",
            ylabel="Separation (E_pass - E_fail)",
        )
    if not score_bins.empty:
        _plot_line(
            x=score_bins["bin_mid"],
            y=score_bins["mean"],
            output_path=charts_dir / "r14_score_monotonicity.png",
            title="Score vs Return (Monotonicity Check)",
            xlabel="Score Bin Midpoint",
            ylabel="Avg Return",
        )

    return R14ExecutionDiagnosticsArtifacts(
        threshold_sweep=threshold_sweep,
        percentile_sweep=percentile_sweep,
        score_bins=score_bins,
        separation_tests=separation_tests,
        output_dir=out_dir,
    )


__all__ = [
    "R14ExecutionDiagnosticsArtifacts",
    "run_threshold_sweep",
    "run_percentile_sweep",
    "score_buckets",
    "separation_test",
    "run_r14_execution_diagnostics",
]

