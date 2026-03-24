from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RobustnessArtifacts:
    candidate_robustness: pd.DataFrame
    sensitivity: pd.DataFrame
    false_peaks: pd.DataFrame


def _encode_param_col(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(int).astype(float)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").astype(float)
    codes, _ = pd.factorize(series.astype(str), sort=True)
    return pd.Series(codes, index=series.index, dtype=float)


def _normalized_param_matrix(df: pd.DataFrame, param_cols: list[str]) -> pd.DataFrame:
    enc = pd.DataFrame(index=df.index)
    for c in param_cols:
        s = _encode_param_col(df[c])
        lo = float(s.min()) if s.notna().any() else 0.0
        hi = float(s.max()) if s.notna().any() else 1.0
        span = hi - lo
        if span <= 1e-12:
            enc[c] = 0.0
        else:
            enc[c] = (s - lo) / span
    return enc.fillna(0.0)


def analyze_parameter_robustness(
    results: pd.DataFrame,
    param_cols: list[str],
    objective_col: str = "oos_expectancy",
    pre_score_col: str = "pre_robust_score",
    n_neighbors: int = 6,
) -> RobustnessArtifacts:
    if results.empty:
        empty = pd.DataFrame()
        return RobustnessArtifacts(
            candidate_robustness=empty,
            sensitivity=empty,
            false_peaks=empty,
        )

    out = results.copy()
    X = _normalized_param_matrix(out, param_cols)
    arr = X.to_numpy(dtype=float)
    obj = pd.to_numeric(out[objective_col], errors="coerce").fillna(0.0).to_numpy()
    base = pd.to_numeric(out[pre_score_col], errors="coerce").fillna(0.0).to_numpy()

    n = len(out)
    neigh_obj_mean = np.zeros(n, dtype=float)
    neigh_obj_std = np.zeros(n, dtype=float)
    neigh_base_mean = np.zeros(n, dtype=float)
    isolation_penalty = np.zeros(n, dtype=float)
    plateau_bonus = np.zeros(n, dtype=float)
    neigh_count = np.zeros(n, dtype=int)

    for i in range(n):
        d = np.sqrt(((arr - arr[i]) ** 2).sum(axis=1))
        d[i] = np.inf
        k = min(max(1, n_neighbors), n - 1) if n > 1 else 0
        if k <= 0:
            neigh_obj_mean[i] = obj[i]
            neigh_obj_std[i] = 0.0
            neigh_base_mean[i] = base[i]
            neigh_count[i] = 0
            continue
        idx = np.argpartition(d, k)[:k]
        neigh_obj = obj[idx]
        neigh_base = base[idx]
        neigh_obj_mean[i] = float(np.mean(neigh_obj))
        neigh_obj_std[i] = float(np.std(neigh_obj))
        neigh_base_mean[i] = float(np.mean(neigh_base))
        neigh_count[i] = int(k)
        # Penalize isolated spikes where local neighborhood is much weaker.
        isolation_penalty[i] = max(0.0, (obj[i] - neigh_obj_mean[i]) * 10_000.0)
        # Reward broad plateaus with strong local mean and low variance.
        plateau_bonus[i] = (neigh_obj_mean[i] * 10_000.0) - (neigh_obj_std[i] * 7_500.0)

    out["neighborhood_expectancy_mean"] = neigh_obj_mean
    out["neighborhood_expectancy_std"] = neigh_obj_std
    out["neighborhood_base_score_mean"] = neigh_base_mean
    out["parameter_isolation_penalty"] = isolation_penalty
    out["plateau_bonus"] = plateau_bonus
    out["neighbor_count"] = neigh_count
    out["robustness_score"] = (
        out[pre_score_col]
        + 0.35 * out["plateau_bonus"]
        + 0.15 * out["neighborhood_base_score_mean"]
        - 0.60 * out["parameter_isolation_penalty"]
    )
    out["robustness_rank"] = out["robustness_score"].rank(ascending=False, method="dense")
    out["expectancy_rank"] = out[objective_col].rank(ascending=False, method="dense")

    sensitivity_rows: list[dict] = []
    y = pd.to_numeric(out[objective_col], errors="coerce").fillna(0.0)
    for c in param_cols:
        x = _encode_param_col(out[c]).fillna(0.0)
        # Use rank-based Pearson correlation to avoid scipy dependency for spearman.
        if len(out) > 2:
            xr = x.rank(method="average")
            yr = y.rank(method="average")
            corr = float(xr.corr(yr))
            if np.isnan(corr):
                corr = 0.0
        else:
            corr = 0.0
        grouped = (
            pd.DataFrame({"x": x, "y": y})
            .groupby("x", as_index=False)["y"]
            .mean()
            .sort_values("x")
        )
        if len(grouped) >= 2:
            top = float(grouped["y"].max())
            bot = float(grouped["y"].min())
            spread = top - bot
        else:
            spread = 0.0
        sensitivity_rows.append(
            {
                "section": "sensitivity",
                "parameter": c,
                "spearman_corr_with_expectancy": corr,
                "abs_spearman_corr": abs(corr),
                "grouped_expectancy_spread": float(spread),
            }
        )
    sensitivity = pd.DataFrame(sensitivity_rows).sort_values(
        "abs_spearman_corr", ascending=False
    )

    false_peaks = out.sort_values(objective_col, ascending=False).head(min(10, len(out))).copy()
    false_peaks = false_peaks[
        false_peaks["robustness_score"] < false_peaks["robustness_score"].median()
    ].copy()
    false_peaks["section"] = "false_peak"

    candidate_robustness = out.copy()
    candidate_robustness["section"] = "candidate"
    return RobustnessArtifacts(
        candidate_robustness=candidate_robustness,
        sensitivity=sensitivity,
        false_peaks=false_peaks,
    )

