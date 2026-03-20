from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .labels import infer_filter_type_from_regime


def _sigmoid(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass
class RuleBasedMetaFilter:
    """Rule-based trade-quality meta model with sleeve-specific splits."""

    target_filter_rate: float = 0.4
    min_filter_rate: float = 0.2
    max_filter_rate: float = 0.6
    min_split_samples: int = 25
    split_feature_col: str = "filter_type"
    numeric_weights: dict[str, float] = field(default_factory=dict)
    categorical_effects: dict[str, dict[str, float]] = field(default_factory=dict)
    numeric_medians: dict[str, float] = field(default_factory=dict)
    numeric_means: dict[str, float] = field(default_factory=dict)
    numeric_stds: dict[str, float] = field(default_factory=dict)
    threshold: float = 0.5
    bias: float = 0.0
    feature_columns: list[str] = field(default_factory=list)
    type_models: dict[str, dict[str, Any]] = field(default_factory=dict)
    latest_filter_type_used: pd.Series | None = None
    calibration: dict[str, Any] = field(default_factory=dict)
    fitted_: bool = False

    def _fit_single_state(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        forward_returns: pd.Series | None = None,
    ) -> dict[str, Any]:
        if len(X) == 0:
            raise ValueError("Cannot fit meta filter with empty feature matrix.")
        y = y.astype(float).reindex(X.index)
        valid = y.notna()
        X = X.loc[valid].copy()
        y = y.loc[valid].fillna(0.0)
        if len(X) == 0:
            raise ValueError("Cannot fit meta filter: no valid labeled samples.")

        state: dict[str, Any] = {
            "feature_columns": X.columns.tolist(),
            "numeric_weights": {},
            "categorical_effects": {},
            "numeric_medians": {},
            "numeric_means": {},
            "numeric_stds": {},
            "threshold": 0.5,
            "bias": 0.0,
            "fitted": True,
        }

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in X.columns if c not in numeric_cols]

        score = np.zeros(len(X), dtype=float)
        target_for_weights = y.astype(float)

        for col in numeric_cols:
            series = X[col].astype(float)
            med = float(series.median()) if not series.dropna().empty else 0.0
            filled = series.fillna(med)
            mean = float(filled.mean())
            std = float(filled.std(ddof=0))
            if std <= 1e-12:
                std = 1.0
            z = (filled - mean) / std
            corr = float(np.corrcoef(z, target_for_weights)[0, 1]) if len(z) > 1 else 0.0
            if np.isnan(corr):
                corr = 0.0

            state["numeric_medians"][col] = med
            state["numeric_means"][col] = mean
            state["numeric_stds"][col] = std
            state["numeric_weights"][col] = corr
            score += corr * z.to_numpy()

        base_rate = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
        bias = float(np.log(base_rate / (1.0 - base_rate)))
        base_level = base_rate
        state["bias"] = bias

        for col in cat_cols:
            as_str = X[col].astype(str).fillna("UNKNOWN")
            group_mean = target_for_weights.groupby(as_str).mean().to_dict()
            effect = {k: float(v - base_level) for k, v in group_mean.items()}
            state["categorical_effects"][col] = effect
            score += as_str.map(effect).fillna(0.0).to_numpy()

        raw = score + bias
        proba = _sigmoid(raw)
        proba_s = pd.Series(proba, index=X.index, dtype=float)
        if forward_returns is not None:
            returns_s = forward_returns.reindex(X.index).astype(float)
        else:
            returns_s = y.reindex(X.index).astype(float)

        calibration = self._optimize_threshold(
            proba=proba_s,
            forward_returns=returns_s,
        )
        state["threshold"] = float(calibration["threshold"])
        state["calibration"] = calibration
        return state

    def _optimize_threshold(
        self,
        proba: pd.Series,
        forward_returns: pd.Series,
    ) -> dict[str, Any]:
        p = proba.astype(float).dropna()
        r = forward_returns.reindex(p.index).astype(float).fillna(0.0)
        if p.empty:
            return {
                "threshold": 0.5,
                "realized_filter_rate": 0.0,
                "target_filter_rate_band": [float(self.min_filter_rate), float(self.max_filter_rate)],
                "target_filter_rate_midpoint": float(
                    (self.min_filter_rate + self.max_filter_rate) / 2.0
                ),
                "threshold_clipped": True,
                "score_distribution_summary": {
                    "min": 0.0,
                    "p10": 0.0,
                    "p50": 0.0,
                    "p90": 0.0,
                    "max": 0.0,
                    "mean": 0.0,
                    "std": 0.0,
                },
            }

        q_grid = np.linspace(0.05, 0.95, 37)
        candidates = sorted({float(p.quantile(q)) for q in q_grid} | {0.5})
        target_mid = float((self.min_filter_rate + self.max_filter_rate) / 2.0)
        score_summary = {
            "min": float(p.min()),
            "p10": float(p.quantile(0.10)),
            "p50": float(p.quantile(0.50)),
            "p90": float(p.quantile(0.90)),
            "max": float(p.max()),
            "mean": float(p.mean()),
            "std": float(p.std(ddof=0)),
        }

        records: list[dict[str, float | bool]] = []

        for thr in candidates:
            take = p >= thr
            kept_n = int(take.sum())
            if kept_n == 0:
                continue
            filter_rate = 1.0 - float(take.mean())
            if not (self.min_filter_rate <= filter_rate <= self.max_filter_rate):
                continue
            min_kept = max(8, int(0.15 * len(p)))
            if kept_n < min_kept:
                continue

            kept = r.loc[take]
            expectancy = float(kept.mean()) if not kept.empty else -np.inf
            std = float(kept.std(ddof=0)) if len(kept) > 1 else 0.0
            sharpe = float((kept.mean() / std) if std > 1e-12 else (1e6 if expectancy > 0 else -1e6))
            in_band = self.min_filter_rate <= filter_rate <= self.max_filter_rate
            dist_mid = abs(filter_rate - target_mid)
            records.append(
                {
                    "threshold": float(thr),
                    "filter_rate": float(filter_rate),
                    "expectancy": float(expectancy),
                    "sharpe": float(sharpe),
                    "in_band": bool(in_band),
                    "dist_mid": float(dist_mid),
                }
            )

        if not records:
            fallback_q = float(np.clip(1.0 - target_mid, 0.01, 0.99))
            fallback_thr = float(p.quantile(fallback_q))
            fallback_rate = float(1.0 - (p >= fallback_thr).mean())
            return {
                "threshold": fallback_thr,
                "realized_filter_rate": fallback_rate,
                "target_filter_rate_band": [float(self.min_filter_rate), float(self.max_filter_rate)],
                "target_filter_rate_midpoint": float(target_mid),
                "threshold_clipped": True,
                "score_distribution_summary": score_summary,
            }

        in_band_records = [r for r in records if bool(r["in_band"])]
        if in_band_records:
            chosen = sorted(
                in_band_records,
                key=lambda x: (
                    float(x["expectancy"]),
                    float(x["sharpe"]),
                    -float(x["dist_mid"]),
                ),
                reverse=True,
            )[0]
            threshold_clipped = False
        else:
            # No threshold can satisfy the strict bounds with available candidate set.
            # Choose the closest threshold to the target midpoint, then maximize expectancy.
            chosen = sorted(
                records,
                key=lambda x: (
                    float(x["dist_mid"]),
                    -float(x["expectancy"]),
                    -float(x["sharpe"]),
                ),
            )[0]
            threshold_clipped = True

        return {
            "threshold": float(chosen["threshold"]),
            "realized_filter_rate": float(chosen["filter_rate"]),
            "target_filter_rate_band": [float(self.min_filter_rate), float(self.max_filter_rate)],
            "target_filter_rate_midpoint": float(target_mid),
            "threshold_clipped": bool(threshold_clipped),
            "score_distribution_summary": score_summary,
        }

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        forward_returns: pd.Series | None = None,
        filter_type: pd.Series | None = None,
    ) -> "RuleBasedMetaFilter":
        X = X.copy()
        if filter_type is None and self.split_feature_col in X.columns:
            filter_type = X[self.split_feature_col].astype(str)
        if filter_type is not None:
            filter_type = filter_type.reindex(X.index).fillna("global").astype(str)
        X_model = X.drop(columns=[self.split_feature_col], errors="ignore")

        global_state = self._fit_single_state(X_model, y, forward_returns=forward_returns)
        self.feature_columns = list(global_state["feature_columns"])
        self.numeric_weights = dict(global_state["numeric_weights"])
        self.categorical_effects = {
            k: dict(v) for k, v in global_state["categorical_effects"].items()
        }
        self.numeric_medians = dict(global_state["numeric_medians"])
        self.numeric_means = dict(global_state["numeric_means"])
        self.numeric_stds = dict(global_state["numeric_stds"])
        self.bias = float(global_state["bias"])
        self.threshold = float(global_state["threshold"])
        self.calibration = dict(global_state.get("calibration", {}))
        self.type_models = {}

        if filter_type is not None:
            ft = filter_type.reindex(X_model.index).fillna("global").astype(str)
            for label in ("trend", "mean_reversion"):
                idx = ft[ft == label].index
                if len(idx) < self.min_split_samples:
                    continue
                y_sub = y.reindex(idx).dropna()
                if y_sub.nunique() < 2:
                    continue
                X_sub = X_model.reindex(y_sub.index)
                fr_sub = forward_returns.reindex(y_sub.index) if forward_returns is not None else None
                try:
                    state = self._fit_single_state(X_sub, y_sub, forward_returns=fr_sub)
                except ValueError:
                    continue
                self.type_models[label] = state

        self.fitted_ = True
        return self

    def _score_from_state(self, X: pd.DataFrame, state: dict[str, Any]) -> np.ndarray:
        X = X.copy()
        for col in state["feature_columns"]:
            if col not in X.columns:
                X[col] = np.nan
        X = X[state["feature_columns"]]

        score = np.zeros(len(X), dtype=float)
        numeric_weights = state["numeric_weights"]
        numeric_cols = [c for c in X.columns if c in numeric_weights]
        cat_cols = [c for c in X.columns if c not in numeric_cols]

        for col in numeric_cols:
            series = X[col].astype(float).fillna(state["numeric_medians"].get(col, 0.0))
            mean = state["numeric_means"].get(col, 0.0)
            std = state["numeric_stds"].get(col, 1.0)
            z = (series - mean) / std
            score += numeric_weights.get(col, 0.0) * z.to_numpy()

        for col in cat_cols:
            effect = state["categorical_effects"].get(col, {})
            score += X[col].astype(str).fillna("UNKNOWN").map(effect).fillna(0.0).to_numpy()

        return score + float(state["bias"])

    def _choose_state_for_type(self, t: str | None) -> tuple[dict[str, Any], str]:
        if t is not None and t in self.type_models:
            return self.type_models[t], t
        global_state = {
            "feature_columns": self.feature_columns,
            "numeric_weights": self.numeric_weights,
            "categorical_effects": self.categorical_effects,
            "numeric_medians": self.numeric_medians,
            "numeric_means": self.numeric_means,
            "numeric_stds": self.numeric_stds,
            "threshold": self.threshold,
            "bias": self.bias,
        }
        return global_state, "global"

    def predict_proba(
        self,
        X: pd.DataFrame,
        filter_type: pd.Series | None = None,
    ) -> pd.Series:
        if not self.fitted_:
            raise ValueError("RuleBasedMetaFilter must be fit before predict_proba.")

        X = X.copy()
        if filter_type is None:
            if self.split_feature_col in X.columns:
                filter_type = X[self.split_feature_col].astype(str)
            else:
                filter_type = pd.Series("global", index=X.index, dtype=str)
        else:
            filter_type = filter_type.reindex(X.index).fillna("global").astype(str)

        X_model = X.drop(columns=[self.split_feature_col], errors="ignore")
        proba = pd.Series(np.nan, index=X.index, dtype=float, name="meta_take_proba")
        used = pd.Series("global", index=X.index, dtype=str, name="filter_type_used")

        for t, idx in filter_type.groupby(filter_type).groups.items():
            state, used_label = self._choose_state_for_type(str(t))
            raw = self._score_from_state(X_model.loc[idx], state)
            proba.loc[idx] = _sigmoid(raw)
            used.loc[idx] = used_label

        self.latest_filter_type_used = used.copy()
        return proba

    def predict(
        self,
        X: pd.DataFrame,
        filter_type: pd.Series | None = None,
    ) -> pd.Series:
        proba = self.predict_proba(X, filter_type=filter_type)
        used = self.latest_filter_type_used if self.latest_filter_type_used is not None else pd.Series(
            "global",
            index=X.index,
            dtype=str,
        )
        take = pd.Series(0, index=X.index, dtype=int, name="meta_take")
        for t, idx in used.groupby(used).groups.items():
            state, _ = self._choose_state_for_type(str(t))
            thr = float(state.get("threshold", self.threshold))
            take.loc[idx] = (proba.loc[idx] >= thr).astype(int)
        return take

    def transform(
        self,
        X: pd.DataFrame,
        filter_type: pd.Series | None = None,
    ) -> pd.DataFrame:
        proba = self.predict_proba(X, filter_type=filter_type)
        take = self.predict(X, filter_type=filter_type)
        used = self.latest_filter_type_used if self.latest_filter_type_used is not None else pd.Series(
            "global",
            index=X.index,
            dtype=str,
        )
        return pd.DataFrame(
            {
                "meta_take_proba": proba,
                "meta_take": take,
                "filter_type_used": used,
            },
            index=X.index,
        )

    def apply(
        self,
        primary_signal: pd.Series,
        entry_mask: pd.Series,
        X_events: pd.DataFrame,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        transformed = self.transform(X_events)
        take_series = pd.Series(1, index=primary_signal.index, dtype=int)
        proba_series = pd.Series(np.nan, index=primary_signal.index, dtype=float)
        filter_type_series = pd.Series("global", index=primary_signal.index, dtype=str)
        take_series.loc[transformed.index] = transformed["meta_take"].astype(int)
        proba_series.loc[transformed.index] = transformed["meta_take_proba"].astype(float)
        filter_type_series.loc[transformed.index] = transformed["filter_type_used"].astype(str)
        self.latest_filter_type_used = filter_type_series
        filtered_signal = apply_meta_trade_filter(
            primary_signal=primary_signal,
            entry_mask=entry_mask,
            meta_take_decision=take_series,
            filter_type_series=filter_type_series,
        )
        return filtered_signal, take_series, proba_series

    def to_dict(self) -> dict:
        return {
            "target_filter_rate": float(self.target_filter_rate),
            "target_filter_rate_band": [float(self.min_filter_rate), float(self.max_filter_rate)],
            "threshold": float(self.threshold),
            "bias": float(self.bias),
            "calibration": dict(self.calibration),
            "feature_columns": list(self.feature_columns),
            "numeric_weights": dict(self.numeric_weights),
            "numeric_medians": dict(self.numeric_medians),
            "numeric_means": dict(self.numeric_means),
            "numeric_stds": dict(self.numeric_stds),
            "categorical_effects": {
                k: {kk: float(vv) for kk, vv in v.items()}
                for k, v in self.categorical_effects.items()
            },
            "type_models": {
                t: {
                    "threshold": float(m.get("threshold", 0.5)),
                    "feature_columns": list(m.get("feature_columns", [])),
                    "calibration": dict(m.get("calibration", {})),
                }
                for t, m in self.type_models.items()
            },
            "fitted": bool(self.fitted_),
        }


def apply_meta_trade_filter(
    primary_signal: pd.Series,
    entry_mask: pd.Series,
    meta_take_decision: pd.Series,
    filter_type_series: pd.Series | None = None,
    context_df: pd.DataFrame | None = None,
) -> pd.Series:
    """Apply meta decisions on top of the primary signal."""
    signal = primary_signal.fillna(0.0).astype(float)
    entries = entry_mask.reindex(signal.index).fillna(False).astype(bool)
    take = meta_take_decision.reindex(signal.index).fillna(1).astype(int)
    if filter_type_series is None and context_df is not None:
        filter_type_series = infer_filter_type_from_regime(context_df).reindex(signal.index)
    if filter_type_series is None:
        filter_type_series = pd.Series("global", index=signal.index, dtype=str)
    else:
        filter_type_series = filter_type_series.reindex(signal.index).fillna("global").astype(str)

    out = pd.Series(0.0, index=signal.index, dtype=float)
    current = 0.0

    for i, idx in enumerate(signal.index):
        desired = float(signal.iloc[i])
        if np.isclose(desired, current):
            out.iloc[i] = current
            continue

        if np.isclose(desired, 0.0):
            current = 0.0
            out.iloc[i] = current
            continue

        if bool(entries.loc[idx]) and int(take.loc[idx]) == 0:
            current = 0.0
            out.iloc[i] = current
            continue

        current = desired
        out.iloc[i] = current

    return out
