from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd


def threshold_stability_report(fold_results: pd.DataFrame) -> pd.DataFrame:
    """Compute threshold stability diagnostics from fold-level meta outputs."""
    if "meta_threshold" not in fold_results.columns:
        return pd.DataFrame(
            [{"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "cv": 0.0}]
        )
    th = pd.to_numeric(fold_results["meta_threshold"], errors="coerce").dropna()
    if th.empty:
        return pd.DataFrame(
            [{"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "cv": 0.0}]
        )
    mean = float(th.mean())
    std = float(th.std(ddof=0))
    return pd.DataFrame(
        [
            {
                "count": int(len(th)),
                "mean": mean,
                "std": std,
                "min": float(th.min()),
                "max": float(th.max()),
                "cv": float(std / mean) if abs(mean) > 1e-12 else 0.0,
                "p25": float(th.quantile(0.25)),
                "p50": float(th.quantile(0.50)),
                "p75": float(th.quantile(0.75)),
            }
        ]
    )


def _extract_numeric_weights(meta_state: str | dict[str, Any]) -> dict[str, float]:
    if isinstance(meta_state, str):
        try:
            parsed = json.loads(meta_state) if meta_state else {}
        except json.JSONDecodeError:
            return {}
    elif isinstance(meta_state, dict):
        parsed = meta_state
    else:
        return {}

    weights: dict[str, float] = {}
    base = parsed.get("numeric_weights", {}) or {}
    for k, v in base.items():
        try:
            weights[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    # Optional split models (trend / mean_reversion) may expose lightweight fields only.
    return weights


def feature_stability_report(fold_results: pd.DataFrame) -> pd.DataFrame:
    """
    Compute feature-weight stability across folds from serialized meta_state.
    """
    if "meta_state" not in fold_results.columns:
        return pd.DataFrame(columns=["feature", "mean_weight", "std_weight", "abs_mean", "sign_consistency", "count"])

    rows: list[dict[str, Any]] = []
    for _, row in fold_results.iterrows():
        weights = _extract_numeric_weights(row.get("meta_state", ""))
        fold_id = int(row.get("fold_id", row.name))
        for feature, weight in weights.items():
            rows.append({"fold": fold_id, "feature": feature, "weight": float(weight)})

    if not rows:
        return pd.DataFrame(columns=["feature", "mean_weight", "std_weight", "abs_mean", "sign_consistency", "count"])

    weights_df = pd.DataFrame(rows)

    def _sign_consistency(series: pd.Series) -> float:
        s = np.sign(series.to_numpy(dtype=float))
        s = s[s != 0]
        if len(s) == 0:
            return 0.0
        return float(max((s > 0).mean(), (s < 0).mean()))

    out = (
        weights_df.groupby("feature")
        .agg(
            mean_weight=("weight", "mean"),
            std_weight=("weight", "std"),
            abs_mean=("weight", lambda x: float(np.mean(np.abs(x)))),
            sign_consistency=("weight", _sign_consistency),
            count=("weight", "count"),
        )
        .reset_index()
        .sort_values(["abs_mean", "sign_consistency"], ascending=[False, False])
    )
    out["std_weight"] = out["std_weight"].fillna(0.0)
    return out
