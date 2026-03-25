from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import CostModel, load_ohlcv_csv
from execution.simulator import run_backtest
from metalabel.features_trade_quality import build_trade_meta_features
from metrics.performance import compute_metrics
from strategies import trend_breakout_v2_signals

from .r14_execution_layer import (
    _compute_early_features_for_trade,
    _extract_trade_segments,
    _fit_fold_meta_model,
    _fallback_meta_score,
    _load_hardened_folds,
    _load_hardened_params,
    _prepare_symbol_frame,
    _resolve_price_data_path,
    _segment_return,
    _safe_float,
)


@dataclass
class R14TailSelectionArtifacts:
    comparison: pd.DataFrame
    fold_results: pd.DataFrame
    stability: pd.DataFrame
    coverage: pd.DataFrame
    output_dir: Path


def _build_tail_mask(
    scores: pd.Series,
    tail_percentile: float,
    min_pass_per_fold: int,
) -> tuple[pd.Series, float]:
    if scores.empty:
        return pd.Series(dtype=bool), float("nan")

    q = float(np.clip(tail_percentile, 0.0, 1.0))
    threshold = float(scores.quantile(q))
    passed = (scores >= threshold).astype(bool)

    if int(passed.sum()) < int(min_pass_per_fold):
        k = min(int(min_pass_per_fold), len(scores))
        top_idx = scores.nlargest(k).index
        passed = scores.index.isin(top_idx)
        passed = pd.Series(passed, index=scores.index, dtype=bool)

    return passed, threshold


def _score_trade(
    row: pd.DataFrame,
    model: Any,
    fallback_state: dict[str, Any] | None,
) -> float:
    if model is not None:
        return _safe_float(model.predict_proba(row).iloc[0], np.nan)
    return _fallback_meta_score(row, fallback_state=fallback_state)


def _plot_tail_fold_returns(fold_df: pd.DataFrame, out_path: Path) -> None:
    if fold_df.empty:
        return
    symbols = sorted(fold_df["symbol"].astype(str).unique().tolist())
    fig, axes = plt.subplots(len(symbols), 1, figsize=(11, 3.5 * len(symbols)), sharex=False)
    if len(symbols) == 1:
        axes = [axes]

    for ax, symbol in zip(axes, symbols, strict=False):
        g = fold_df[fold_df["symbol"] == symbol].sort_values("fold_id")
        x = pd.to_numeric(g["fold_id"], errors="coerce")
        y = pd.to_numeric(g["E_tail"], errors="coerce")
        ax.plot(x, y, marker="o", linewidth=1.5, label=f"{symbol} tail return/fold")
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.set_title(f"{symbol} — Tail Return per Fold")
        ax.set_xlabel("Fold")
        ax.set_ylabel("E_tail")
        ax.grid(alpha=0.2)
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_tail_vs_baseline(stability_df: pd.DataFrame, out_path: Path) -> None:
    if stability_df.empty:
        return
    plot_df = stability_df[stability_df["symbol"] != "ALL"].copy()
    if plot_df.empty:
        plot_df = stability_df.copy()
    labels = plot_df["symbol"].astype(str).tolist()
    baseline_vals = pd.to_numeric(plot_df["baseline_return"], errors="coerce").fillna(0.0).to_numpy()
    tail_vals = pd.to_numeric(plot_df["mean_tail_return"], errors="coerce").fillna(0.0).to_numpy()
    x = np.arange(len(labels), dtype=float)
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, baseline_vals, width=width, label="baseline")
    ax.bar(x + width / 2, tail_vals, width=width, label="tail")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Average Return")
    ax.set_title("R1.4.2 Tail vs Baseline")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_r14_tail_selection(
    symbols: list[str],
    timeframe: str = "H1",
    early_window: int = 3,
    tail_percentile: float = 0.9,
    min_pass_per_fold: int = 2,
    strategy: str = "TrendBreakout_V2",
    artifacts_root: str | Path = "outputs/TrendBreakout_V2",
    source_csv: str | Path | None = None,
    output_dir: str | Path = "outputs",
    forward_horizon: int = 24,
    label_method: str = "top_quantile",
    label_quantile: float = 0.30,
    meta_min_train_samples: int = 30,
    allow_fallback_scorer: bool = True,
) -> R14TailSelectionArtifacts:
    if strategy != "TrendBreakout_V2":
        raise ValueError("R1.4.2 currently supports strategy='TrendBreakout_V2' only.")
    if not symbols:
        raise ValueError("At least one symbol is required.")
    if early_window < 1:
        raise ValueError("early_window must be >= 1")
    if min_pass_per_fold < 1:
        raise ValueError("min_pass_per_fold must be >= 1")

    artifacts_root_path = Path(artifacts_root)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = out_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    price_path = _resolve_price_data_path(
        artifacts_root=artifacts_root_path,
        explicit_source_csv=source_csv,
        symbols=symbols,
    )
    raw_prices = load_ohlcv_csv(price_path)
    cost_model = CostModel(spread_bps=0.8, slippage_bps=0.5, commission_bps=0.2)

    fold_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []

    for symbol in symbols:
        strategy_params = _load_hardened_params(artifacts_root_path, symbol=symbol)
        fold_windows = _load_hardened_folds(
            artifacts_root=artifacts_root_path,
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

        baseline_returns_all: list[pd.Series] = []
        tail_returns_all: list[pd.Series] = []
        e_tail_per_fold: list[float] = []
        e_base_per_fold: list[float] = []
        fold_tail_pass_rates: list[float] = []
        fold_total_trades: list[int] = []
        total_early_exits = 0
        total_tail_trades = 0
        total_trades = 0
        total_eval_trades = 0

        for _, fold in fold_windows.iterrows():
            fold_id = int(fold["fold_id"])
            train_start = pd.Timestamp(fold["fold_start"])
            train_end = pd.Timestamp(fold["fold_train_end"])
            test_start = pd.Timestamp(fold["fold_test_start"])
            test_end = pd.Timestamp(fold["fold_test_end"])

            train_df = frame[
                (frame["timestamp"] >= train_start) & (frame["timestamp"] <= train_end)
            ].copy()
            test_df = frame[
                (frame["timestamp"] >= test_start) & (frame["timestamp"] <= test_end)
            ].copy()
            if train_df.empty or test_df.empty:
                continue

            model, train_meta, fallback_state = _fit_fold_meta_model(
                train_df=train_df,
                strategy_params=strategy_params,
                early_window=early_window,
                forward_horizon=forward_horizon,
                label_method=label_method,
                label_quantile=label_quantile,
                meta_min_train_samples=meta_min_train_samples,
                allow_fallback_scorer=allow_fallback_scorer,
            )

            raw_signal = trend_breakout_v2_signals(test_df, strategy_params).astype(float)
            base_bt = run_backtest(test_df, raw_signal, cost_model=cost_model, max_abs_position=1.0)
            baseline_returns_all.append(base_bt.returns)
            base_features = build_trade_meta_features(test_df, raw_signal).drop(
                columns=["entry_mask"],
                errors="ignore",
            )

            segments = _extract_trade_segments(raw_signal)
            score_records: list[dict[str, Any]] = []
            for seg_id, (entry_i, end_i, side) in enumerate(segments):
                baseline_ret = _segment_return(base_bt.returns, entry_i, end_i)
                if (entry_i + early_window) > end_i:
                    score_records.append(
                        {
                            "trade_id": f"{symbol}_{fold_id}_{seg_id}",
                            "fold_id": fold_id,
                            "symbol": symbol,
                            "entry_i": entry_i,
                            "end_i": end_i,
                            "realized_return": baseline_ret,
                            "meta_score": np.nan,
                            "evaluated": 0,
                        }
                    )
                    continue

                early = _compute_early_features_for_trade(
                    df=test_df,
                    entry_i=entry_i,
                    side=side,
                    early_window=early_window,
                )
                if early is None:
                    score_records.append(
                        {
                            "trade_id": f"{symbol}_{fold_id}_{seg_id}",
                            "fold_id": fold_id,
                            "symbol": symbol,
                            "entry_i": entry_i,
                            "end_i": end_i,
                            "realized_return": baseline_ret,
                            "meta_score": np.nan,
                            "evaluated": 0,
                        }
                    )
                    continue

                entry_idx = test_df.index[entry_i]
                eval_i = entry_i + early_window
                eval_idx = test_df.index[eval_i]
                row = base_features.loc[entry_idx].to_dict() if entry_idx in base_features.index else {}
                row.update(early)
                features = pd.DataFrame([row], index=[eval_idx])
                score = _score_trade(features, model=model, fallback_state=fallback_state)
                score_records.append(
                    {
                        "trade_id": f"{symbol}_{fold_id}_{seg_id}",
                        "fold_id": fold_id,
                        "symbol": symbol,
                        "entry_i": entry_i,
                        "end_i": end_i,
                        "eval_i": eval_i,
                        "realized_return": baseline_ret,
                        "meta_score": score,
                        "evaluated": int(np.isfinite(score)),
                    }
                )

            score_df = pd.DataFrame(score_records)
            if "meta_score" not in score_df.columns:
                score_df["meta_score"] = np.nan
            if "realized_return" not in score_df.columns:
                score_df["realized_return"] = np.nan
            evaluated_df = score_df[pd.to_numeric(score_df["meta_score"], errors="coerce").notna()].copy()
            pass_mask = pd.Series(False, index=evaluated_df.index, dtype=bool)
            threshold = float("nan")
            if not evaluated_df.empty:
                pass_mask, threshold = _build_tail_mask(
                    scores=pd.to_numeric(evaluated_df["meta_score"], errors="coerce"),
                    tail_percentile=tail_percentile,
                    min_pass_per_fold=min_pass_per_fold,
                )
                evaluated_df["pass"] = pass_mask.astype(bool)
            else:
                evaluated_df["pass"] = False

            tail_signal = raw_signal.copy()
            early_exits_fold = 0
            for _, row in evaluated_df.iterrows():
                if bool(row["pass"]):
                    continue
                eval_i = int(row["eval_i"])
                end_i = int(row["end_i"])
                tail_signal.iloc[eval_i : end_i + 1] = 0.0
                early_exits_fold += 1

            tail_bt = run_backtest(test_df, tail_signal, cost_model=cost_model, max_abs_position=1.0)
            tail_returns_all.append(tail_bt.returns)

            trades_total_fold = int(len(score_df))
            trades_eval_fold = int(len(evaluated_df))
            tail_pass_count = int(evaluated_df["pass"].sum()) if trades_eval_fold > 0 else 0
            tail_pass_rate = float(tail_pass_count / trades_eval_fold) if trades_eval_fold > 0 else 0.0
            e_tail = float(
                pd.to_numeric(
                    evaluated_df.loc[evaluated_df["pass"], "realized_return"],
                    errors="coerce",
                ).mean()
            ) if tail_pass_count > 0 else np.nan
            e_base = float(pd.to_numeric(score_df["realized_return"], errors="coerce").mean()) if trades_total_fold > 0 else np.nan

            e_tail_per_fold.append(e_tail)
            e_base_per_fold.append(e_base)
            fold_tail_pass_rates.append(tail_pass_rate)
            fold_total_trades.append(trades_total_fold)
            total_early_exits += early_exits_fold
            total_tail_trades += tail_pass_count
            total_trades += trades_total_fold
            total_eval_trades += trades_eval_fold

            met_base = compute_metrics(
                base_bt.returns,
                base_bt.equity,
                base_bt.trades,
                timeframe=timeframe,
                position=base_bt.position,
            )
            met_tail = compute_metrics(
                tail_bt.returns,
                tail_bt.equity,
                tail_bt.trades,
                timeframe=timeframe,
                position=tail_bt.position,
            )

            fold_rows.append(
                {
                    "symbol": symbol,
                    "fold_id": fold_id,
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "threshold": threshold,
                    "tail_percentile": float(tail_percentile),
                    "trades_total": trades_total_fold,
                    "trades_evaluated": trades_eval_fold,
                    "tail_pass_count": tail_pass_count,
                    "tail_pass_rate": tail_pass_rate,
                    "early_exit_count": early_exits_fold,
                    "baseline_sharpe": _safe_float(met_base.get("Sharpe")),
                    "tail_sharpe": _safe_float(met_tail.get("Sharpe")),
                    "baseline_expectancy": e_base,
                    "tail_expectancy": _safe_float(
                        pd.to_numeric(
                            evaluated_df.loc[evaluated_df["pass"], "realized_return"],
                            errors="coerce",
                        ).mean(),
                        np.nan,
                    ),
                    "baseline_max_dd": _safe_float(met_base.get("MaxDrawdown")),
                    "tail_max_dd": _safe_float(met_tail.get("MaxDrawdown")),
                    "E_tail": e_tail,
                    "E_baseline_all_trades": e_base,
                    "tail_minus_baseline": e_tail - e_base if np.isfinite(e_tail) and np.isfinite(e_base) else np.nan,
                    "model_fitted": bool(model is not None),
                    "used_fallback_scorer": bool(model is None and fallback_state is not None),
                    "train_meta_samples": int(len(train_meta)),
                }
            )

        if not baseline_returns_all or not tail_returns_all:
            continue

        stitched_base = pd.concat(baseline_returns_all).sort_index()
        stitched_tail = pd.concat(tail_returns_all).sort_index()
        eq_base = 100_000.0 * (1.0 + stitched_base).cumprod()
        eq_tail = 100_000.0 * (1.0 + stitched_tail).cumprod()
        met_base_all = compute_metrics(stitched_base, eq_base, pd.DataFrame(), timeframe=timeframe)
        met_tail_all = compute_metrics(stitched_tail, eq_tail, pd.DataFrame(), timeframe=timeframe)

        comparison_rows.extend(
            [
                {
                    "symbol": symbol,
                    "variant": "baseline",
                    "Sharpe": _safe_float(met_base_all.get("Sharpe")),
                    "Expectancy": float(np.nanmean(e_base_per_fold)) if e_base_per_fold else np.nan,
                    "MaxDD": _safe_float(met_base_all.get("MaxDrawdown")),
                    "TradeCount": float(total_trades),
                },
                {
                    "symbol": symbol,
                    "variant": "tail_selection",
                    "Sharpe": _safe_float(met_tail_all.get("Sharpe")),
                    "Expectancy": float(np.nanmean(e_tail_per_fold)) if e_tail_per_fold else np.nan,
                    "MaxDD": _safe_float(met_tail_all.get("MaxDrawdown")),
                    "TradeCount": float(total_tail_trades),
                },
            ]
        )

        e_tail_clean = pd.to_numeric(pd.Series(e_tail_per_fold), errors="coerce").dropna()
        baseline_return = float(np.nanmean(e_base_per_fold)) if e_base_per_fold else np.nan
        stability_rows.append(
            {
                "symbol": symbol,
                "mean_tail_return": float(e_tail_clean.mean()) if not e_tail_clean.empty else np.nan,
                "std_tail_return": float(e_tail_clean.std(ddof=0)) if len(e_tail_clean) > 1 else 0.0,
                "positive_fold_pct": float((e_tail_clean > 0).mean()) if not e_tail_clean.empty else 0.0,
                "min_tail_return": float(e_tail_clean.min()) if not e_tail_clean.empty else np.nan,
                "max_tail_return": float(e_tail_clean.max()) if not e_tail_clean.empty else np.nan,
                "baseline_return": baseline_return,
                "avg_trades_per_fold": float(np.mean(fold_total_trades)) if fold_total_trades else 0.0,
            }
        )
        coverage_rows.append(
            {
                "symbol": symbol,
                "tail_pass_rate": float(total_tail_trades / max(total_eval_trades, 1)),
                "avg_trades_per_fold": float(np.mean(fold_total_trades)) if fold_total_trades else 0.0,
                "total_trades": int(total_trades),
                "trades_evaluated": int(total_eval_trades),
                "avg_tail_trades_per_fold": float(total_tail_trades / max(len(fold_total_trades), 1)),
                "early_exit_count": int(total_early_exits),
            }
        )

    comparison = pd.DataFrame(comparison_rows)
    fold_results = pd.DataFrame(fold_rows).sort_values(["symbol", "fold_id"]).reset_index(drop=True)
    stability = pd.DataFrame(stability_rows)
    coverage = pd.DataFrame(coverage_rows)

    if not comparison.empty:
        all_rows = []
        for variant, grp in comparison.groupby("variant"):
            all_rows.append(
                {
                    "symbol": "ALL",
                    "variant": variant,
                    "Sharpe": float(pd.to_numeric(grp["Sharpe"], errors="coerce").mean()),
                    "Expectancy": float(pd.to_numeric(grp["Expectancy"], errors="coerce").mean()),
                    "MaxDD": float(pd.to_numeric(grp["MaxDD"], errors="coerce").mean()),
                    "TradeCount": float(pd.to_numeric(grp["TradeCount"], errors="coerce").sum()),
                }
            )
        comparison = pd.concat([comparison, pd.DataFrame(all_rows)], ignore_index=True)

    if not stability.empty:
        e_tail = pd.to_numeric(stability["mean_tail_return"], errors="coerce")
        baseline = pd.to_numeric(stability["baseline_return"], errors="coerce")
        stability_all = {
            "symbol": "ALL",
            "mean_tail_return": float(e_tail.mean()),
            "std_tail_return": float(pd.to_numeric(stability["std_tail_return"], errors="coerce").mean()),
            "positive_fold_pct": float(pd.to_numeric(stability["positive_fold_pct"], errors="coerce").mean()),
            "min_tail_return": float(e_tail.min()),
            "max_tail_return": float(e_tail.max()),
            "baseline_return": float(baseline.mean()),
            "avg_trades_per_fold": float(pd.to_numeric(stability["avg_trades_per_fold"], errors="coerce").mean()),
        }
        stability = pd.concat([stability, pd.DataFrame([stability_all])], ignore_index=True)

    if not coverage.empty:
        cov_all = {
            "symbol": "ALL",
            "tail_pass_rate": float(pd.to_numeric(coverage["tail_pass_rate"], errors="coerce").mean()),
            "avg_trades_per_fold": float(pd.to_numeric(coverage["avg_trades_per_fold"], errors="coerce").mean()),
            "total_trades": int(pd.to_numeric(coverage["total_trades"], errors="coerce").sum()),
            "trades_evaluated": int(pd.to_numeric(coverage["trades_evaluated"], errors="coerce").sum()),
            "avg_tail_trades_per_fold": float(
                pd.to_numeric(coverage["avg_tail_trades_per_fold"], errors="coerce").mean()
            ),
            "early_exit_count": int(pd.to_numeric(coverage["early_exit_count"], errors="coerce").sum()),
        }
        coverage = pd.concat([coverage, pd.DataFrame([cov_all])], ignore_index=True)

    # Requested plots
    _plot_tail_fold_returns(fold_results, charts_dir / "r14_tail_fold_returns.png")
    _plot_tail_vs_baseline(stability, charts_dir / "r14_tail_vs_baseline.png")

    comparison.to_csv(out_dir / "r14_tail_comparison.csv", index=False)
    fold_results.to_csv(out_dir / "r14_tail_fold_results.csv", index=False)
    stability.to_csv(out_dir / "r14_tail_stability.csv", index=False)
    coverage.to_csv(out_dir / "r14_tail_coverage.csv", index=False)

    return R14TailSelectionArtifacts(
        comparison=comparison,
        fold_results=fold_results,
        stability=stability,
        coverage=coverage,
        output_dir=out_dir,
    )


__all__ = ["R14TailSelectionArtifacts", "run_r14_tail_selection"]
