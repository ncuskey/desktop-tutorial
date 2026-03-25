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
from strategies import trend_breakout_v2_signals

from .r14_execution_layer import (
    _compute_early_features_for_trade,
    _extract_trade_segments,
    _load_hardened_folds,
    _load_hardened_params,
    _prepare_symbol_frame,
    _resolve_price_data_path,
    _segment_return,
)


REQUIRED_FEATURE_COLUMNS = [
    "early_return_1",
    "early_return_3",
    "early_mfe",
    "early_mae",
    "early_slope",
    "early_volatility",
    "early_range_expansion",
    "volatility_spike_flag",
]


@dataclass
class R15FailureDecompositionArtifacts:
    feature_separation: pd.DataFrame
    rule_candidates: pd.DataFrame
    rule_evaluation: pd.DataFrame
    output_dir: Path
    dataset_path: Path


def _build_meta_feature_dataset_from_baseline(
    input_path: Path,
    symbols: list[str],
    timeframe: str,
    early_window: int,
    artifacts_root: Path,
    source_csv: str | Path | None,
) -> pd.DataFrame:
    price_path = _resolve_price_data_path(
        artifacts_root=artifacts_root,
        explicit_source_csv=source_csv,
        symbols=symbols,
    )
    raw_prices = load_ohlcv_csv(price_path)
    cost_model = CostModel(spread_bps=0.8, slippage_bps=0.5, commission_bps=0.2)
    rows: list[dict[str, Any]] = []

    for symbol in symbols:
        params = _load_hardened_params(artifacts_root, symbol=symbol)
        folds = _load_hardened_folds(artifacts_root=artifacts_root, symbol=symbol, hardened_params=params)
        frame = _prepare_symbol_frame(
            raw_prices=raw_prices,
            symbol=symbol,
            timeframe=timeframe,
            cost_model=cost_model,
        )
        if frame.empty:
            continue

        for _, fold in folds.iterrows():
            fold_id = int(fold["fold_id"])
            test_start = pd.Timestamp(fold["fold_test_start"])
            test_end = pd.Timestamp(fold["fold_test_end"])
            test_df = frame[
                (frame["timestamp"] >= test_start) & (frame["timestamp"] <= test_end)
            ].copy()
            if test_df.empty:
                continue

            signal = trend_breakout_v2_signals(test_df, params).astype(float)
            bt = run_backtest(test_df, signal, cost_model=cost_model, max_abs_position=1.0)
            segments = _extract_trade_segments(signal)

            for seg_id, (entry_i, end_i, side) in enumerate(segments):
                realized_return = _segment_return(bt.returns, entry_i, end_i)
                early = _compute_early_features_for_trade(
                    df=test_df,
                    entry_i=entry_i,
                    side=side,
                    early_window=early_window,
                )
                record = {
                    "trade_id": f"{symbol}_{fold_id}_{seg_id}",
                    "symbol": symbol,
                    "fold_id": fold_id,
                    "realized_return": float(realized_return),
                }
                for col in REQUIRED_FEATURE_COLUMNS:
                    record[col] = np.nan if early is None else float(early.get(col, np.nan))
                rows.append(record)

    dataset = pd.DataFrame(rows)
    input_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(input_path, index=False)
    return dataset


def _load_dataset(
    input_path: Path,
    symbols: list[str],
    timeframe: str,
    early_window: int,
    artifacts_root: Path,
    source_csv: str | Path | None,
    rebuild_if_missing: bool,
) -> pd.DataFrame:
    if input_path.exists():
        return pd.read_csv(input_path)
    if not rebuild_if_missing:
        raise FileNotFoundError(f"Input dataset not found: {input_path}")
    return _build_meta_feature_dataset_from_baseline(
        input_path=input_path,
        symbols=symbols,
        timeframe=timeframe,
        early_window=early_window,
        artifacts_root=artifacts_root,
        source_csv=source_csv,
    )


def compute_feature_stats(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out: list[dict[str, Any]] = []
    labeled = df.copy()
    labeled["label"] = (pd.to_numeric(labeled["realized_return"], errors="coerce") > 0).astype(int)

    for feature in features:
        if feature not in labeled.columns:
            continue
        values = pd.to_numeric(labeled[feature], errors="coerce")
        working = labeled.assign(_x=values).dropna(subset=["_x", "label"])
        winners = working.loc[working["label"] == 1, "_x"]
        losers = working.loc[working["label"] == 0, "_x"]
        if winners.empty or losers.empty:
            p_value = np.nan
        else:
            _stat, p_value = mannwhitneyu(winners, losers, alternative="two-sided")
        sep = float(winners.mean() - losers.mean()) if not winners.empty and not losers.empty else np.nan
        out.append(
            {
                "feature": feature,
                "mean_winner": float(winners.mean()) if not winners.empty else np.nan,
                "mean_loser": float(losers.mean()) if not losers.empty else np.nan,
                "median_winner": float(winners.median()) if not winners.empty else np.nan,
                "median_loser": float(losers.median()) if not losers.empty else np.nan,
                "std_winner": float(winners.std(ddof=0)) if len(winners) > 1 else 0.0,
                "std_loser": float(losers.std(ddof=0)) if len(losers) > 1 else 0.0,
                "separation": sep,
                "separation_abs": abs(sep) if np.isfinite(sep) else np.nan,
                "p_value": float(p_value) if p_value is not None and np.isfinite(p_value) else np.nan,
                "direction": (
                    "winner_gt_loser"
                    if np.isfinite(sep) and sep >= 0
                    else "winner_lt_loser"
                ),
                "winner_count": int(winners.count()),
                "loser_count": int(losers.count()),
            }
        )

    if not out:
        return pd.DataFrame(
            columns=[
                "feature",
                "mean_winner",
                "mean_loser",
                "median_winner",
                "median_loser",
                "std_winner",
                "std_loser",
                "separation",
                "separation_abs",
                "p_value",
                "direction",
                "winner_count",
                "loser_count",
            ]
        )
    return pd.DataFrame(out).sort_values("separation_abs", ascending=False).reset_index(drop=True)


def evaluate_rule(df: pd.DataFrame, feature: str, threshold: float, direction: str) -> dict[str, Any]:
    data = df.copy()
    x = pd.to_numeric(data[feature], errors="coerce")
    data = data.assign(_x=x).dropna(subset=["_x"])
    if data.empty:
        return {
            "feature": feature,
            "threshold": float(threshold),
            "direction": direction,
            "reject_rate": np.nan,
            "kept_trades": 0,
            "E_all": np.nan,
            "E_kept": np.nan,
            "lift": np.nan,
            "loser_removal_rate": np.nan,
            "winner_loss_rate": np.nan,
        }

    if direction == "lt":
        reject = data["_x"] < float(threshold)
    elif direction == "gt":
        reject = data["_x"] > float(threshold)
    else:
        raise ValueError("direction must be one of ['lt', 'gt']")

    kept = data.loc[~reject].copy()
    e_all = float(pd.to_numeric(data["realized_return"], errors="coerce").mean())
    e_kept = float(pd.to_numeric(kept["realized_return"], errors="coerce").mean()) if not kept.empty else np.nan

    losers = data["label"] == 0
    winners = data["label"] == 1
    loser_denom = int(losers.sum())
    winner_denom = int(winners.sum())
    loser_removal = float((losers & reject).sum() / loser_denom) if loser_denom > 0 else np.nan
    winner_loss = float((winners & reject).sum() / winner_denom) if winner_denom > 0 else np.nan

    return {
        "feature": feature,
        "threshold": float(threshold),
        "direction": direction,
        "reject_rate": float(reject.mean()),
        "kept_trades": int(len(kept)),
        "E_all": e_all,
        "E_kept": e_kept,
        "lift": float(e_kept - e_all) if np.isfinite(e_kept) and np.isfinite(e_all) else np.nan,
        "loser_removal_rate": loser_removal,
        "winner_loss_rate": winner_loss,
    }


def _build_rule_candidates(
    df: pd.DataFrame,
    top_features: list[str],
    threshold_percentiles: list[int],
) -> pd.DataFrame:
    candidates: list[dict[str, Any]] = []
    for feature in top_features:
        series = pd.to_numeric(df[feature], errors="coerce").dropna()
        if series.empty:
            continue
        thresholds = np.unique(np.percentile(series, threshold_percentiles)).tolist()
        for percentile, threshold in zip(threshold_percentiles, thresholds, strict=False):
            candidates.append(
                {
                    "feature": feature,
                    "threshold": float(threshold),
                    "threshold_percentile": int(percentile),
                    "direction": "lt",
                }
            )
            candidates.append(
                {
                    "feature": feature,
                    "threshold": float(threshold),
                    "threshold_percentile": int(percentile),
                    "direction": "gt",
                }
            )
    if not candidates:
        return pd.DataFrame(columns=["feature", "threshold", "threshold_percentile", "direction"])
    return pd.DataFrame(candidates)


def _plot_feature_distributions(
    df: pd.DataFrame,
    top_features: list[str],
    output_path: Path,
) -> None:
    if not top_features:
        return
    winners = df[df["label"] == 1]
    losers = df[df["label"] == 0]
    n = len(top_features)
    fig, axes = plt.subplots(n, 1, figsize=(10, max(3.2 * n, 4)))
    if n == 1:
        axes = [axes]
    for ax, feature in zip(axes, top_features, strict=False):
        w = pd.to_numeric(winners[feature], errors="coerce").dropna()
        l = pd.to_numeric(losers[feature], errors="coerce").dropna()
        if w.empty and l.empty:
            continue
        bins = 25
        ax.hist(l, bins=bins, alpha=0.5, label="losers", density=True)
        ax.hist(w, bins=bins, alpha=0.5, label="winners", density=True)
        ax.set_title(f"{feature}: winners vs losers")
        ax.grid(alpha=0.2)
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_rule_lift_vs_reject(rule_eval: pd.DataFrame, output_path: Path) -> None:
    if rule_eval.empty:
        return
    plot_df = rule_eval.copy()
    plot_df = plot_df.dropna(subset=["reject_rate", "lift"])
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for feature, grp in plot_df.groupby("feature", observed=True):
        ax.scatter(
            grp["reject_rate"],
            grp["lift"],
            alpha=0.7,
            label=str(feature),
        )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("Reject Rate")
    ax.set_ylabel("Lift (E_kept - E_all)")
    ax.set_title("R1.5 Rule Lift vs Reject Rate")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_r15_failure_decomposition(
    input_path: str | Path = "outputs/meta_feature_dataset.csv",
    output_dir: str | Path = "outputs",
    top_n_features: int = 5,
    threshold_percentiles: list[int] | None = None,
    symbols: list[str] | None = None,
    timeframe: str = "H1",
    early_window: int = 3,
    artifacts_root: str | Path = "outputs/TrendBreakout_V2",
    source_csv: str | Path | None = None,
    rebuild_if_missing: bool = True,
) -> R15FailureDecompositionArtifacts:
    in_path = Path(input_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = out_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    threshold_percentiles = threshold_percentiles or [20, 30, 40, 50, 60, 70, 80]
    run_symbols = symbols or ["EURUSD", "GBPUSD", "AUDUSD"]

    dataset = _load_dataset(
        input_path=in_path,
        symbols=run_symbols,
        timeframe=timeframe,
        early_window=early_window,
        artifacts_root=Path(artifacts_root),
        source_csv=source_csv,
        rebuild_if_missing=rebuild_if_missing,
    )
    if dataset.empty:
        raise ValueError("Input meta-feature dataset is empty.")
    if "realized_return" not in dataset.columns:
        raise ValueError("Dataset must include 'realized_return' column.")

    dataset = dataset.copy()
    dataset["realized_return"] = pd.to_numeric(dataset["realized_return"], errors="coerce")
    dataset = dataset.dropna(subset=["realized_return"])
    dataset["label"] = (dataset["realized_return"] > 0).astype(int)

    available_features = [f for f in REQUIRED_FEATURE_COLUMNS if f in dataset.columns]
    if not available_features:
        raise ValueError("No required R1.5 features found in dataset.")

    feature_separation = compute_feature_stats(dataset, available_features)
    top_features = feature_separation.head(max(int(top_n_features), 1))["feature"].astype(str).tolist()

    rule_candidates = _build_rule_candidates(
        df=dataset,
        top_features=top_features,
        threshold_percentiles=threshold_percentiles,
    )

    rule_eval_rows: list[dict[str, Any]] = []
    for _, row in rule_candidates.iterrows():
        eval_row = evaluate_rule(
            df=dataset,
            feature=str(row["feature"]),
            threshold=float(row["threshold"]),
            direction=str(row["direction"]),
        )
        rule_eval_rows.append(eval_row)
    rule_evaluation = pd.DataFrame(rule_eval_rows)
    if not rule_evaluation.empty:
        rule_evaluation["rule_good"] = (
            (pd.to_numeric(rule_evaluation["lift"], errors="coerce") > 0)
            & (pd.to_numeric(rule_evaluation["loser_removal_rate"], errors="coerce") >= 0.30)
            & (pd.to_numeric(rule_evaluation["winner_loss_rate"], errors="coerce") <= 0.30)
            & (pd.to_numeric(rule_evaluation["kept_trades"], errors="coerce") >= 2)
        )
        rule_evaluation = rule_evaluation.sort_values(
            ["rule_good", "lift", "loser_removal_rate", "winner_loss_rate"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)

    # Persist requested outputs
    feature_separation.to_csv(out_dir / "r15_feature_separation.csv", index=False)
    rule_candidates.to_csv(out_dir / "r15_rule_candidates.csv", index=False)
    rule_evaluation.to_csv(out_dir / "r15_rule_evaluation.csv", index=False)

    # Requested plots
    _plot_feature_distributions(
        df=dataset,
        top_features=top_features,
        output_path=charts_dir / "r15_feature_distributions.png",
    )
    _plot_rule_lift_vs_reject(
        rule_eval=rule_evaluation,
        output_path=charts_dir / "r15_rule_lift_vs_reject.png",
    )

    return R15FailureDecompositionArtifacts(
        feature_separation=feature_separation,
        rule_candidates=rule_candidates,
        rule_evaluation=rule_evaluation,
        output_dir=out_dir,
        dataset_path=in_path,
    )


__all__ = [
    "R15FailureDecompositionArtifacts",
    "compute_feature_stats",
    "evaluate_rule",
    "run_r15_failure_decomposition",
]

