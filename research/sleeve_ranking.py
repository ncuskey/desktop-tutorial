from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _zscore(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").fillna(0.0)
    std = float(x.std(ddof=0))
    if std <= 1e-12:
        return pd.Series(0.0, index=x.index)
    return (x - float(x.mean())) / std


def build_sleeve_symbol_ranking(
    metrics_df: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Rank sleeve-symbol combinations with expectancy-first composite scoring.
    """
    if metrics_df.empty:
        return pd.DataFrame()

    w = {
        "filtered_expectancy": 0.35,
        "expectancy_delta": 0.30,
        "filtered_sharpe": 0.10,
        "sharpe_delta": 0.10,
        "maxdd_delta": 0.10,
        "threshold_stability": -0.08,
        "filter_rate_stability": -0.07,
        "expectancy_fold_win_rate": 0.08,
        "drawdown_fold_win_rate": 0.07,
        "avg_filter_rate_error_abs": -0.12,
        "unstable_fold_penalty": -0.10,
    }
    if weights:
        w.update(weights)

    df = metrics_df.copy()
    if "avg_filter_rate_error_abs" not in df.columns:
        df["avg_filter_rate_error_abs"] = pd.to_numeric(
            df.get("avg_filter_rate_error", 0.0), errors="coerce"
        ).abs()
    if "unstable_fold_penalty" not in df.columns:
        exp_win = pd.to_numeric(df.get("expectancy_fold_win_rate", 0.0), errors="coerce").fillna(0.0)
        dd_win = pd.to_numeric(df.get("drawdown_fold_win_rate", 0.0), errors="coerce").fillna(0.0)
        df["unstable_fold_penalty"] = 1.0 - (0.5 * exp_win + 0.5 * dd_win)

    score = pd.Series(0.0, index=df.index, dtype=float)
    for col, weight in w.items():
        if col not in df.columns:
            continue
        score += float(weight) * _zscore(df[col])

    df["composite_score"] = score
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def classify_component_decisions(ranking_df: pd.DataFrame) -> pd.DataFrame:
    """
    Bucket ranked components into PROMOTE / WATCH / PRUNE.
    """
    if ranking_df.empty:
        return pd.DataFrame(columns=["symbol", "sleeve", "decision", "reason", "composite_score"])

    rows: list[dict[str, Any]] = []
    for _, row in ranking_df.iterrows():
        exp_f = float(row.get("filtered_expectancy", 0.0))
        dd_delta = float(row.get("maxdd_delta", 0.0))
        th_stab = float(row.get("threshold_stability", 999.0))
        fr_stab = float(row.get("filter_rate_stability", 999.0))
        exp_win = float(row.get("expectancy_fold_win_rate", 0.0))
        fr_err_abs = abs(float(row.get("avg_filter_rate_error", 0.0)))

        if (
            exp_f >= -5e-5
            and dd_delta >= 0.0
            and th_stab <= 0.15
            and fr_stab <= 0.20
            and exp_win >= 0.5
            and fr_err_abs <= 0.10
        ):
            decision = "PROMOTE"
            reason = "Expectancy resilient, DD improved, and stable calibration."
        elif dd_delta > 0.0 or exp_win >= 0.45:
            decision = "WATCH"
            reason = "Mixed expectancy profile but useful risk-control behavior."
        else:
            decision = "PRUNE"
            reason = "Weak expectancy profile without enough drawdown/stability benefit."

        rows.append(
            {
                "symbol": row.get("symbol"),
                "sleeve": row.get("sleeve"),
                "decision": decision,
                "reason": reason,
                "composite_score": float(row.get("composite_score", 0.0)),
                "rank": int(row.get("rank", 0)),
            }
        )
    return pd.DataFrame(rows).sort_values(["decision", "rank"]).reset_index(drop=True)
