from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


def _sigmoid(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _rolling_slope(series: pd.Series, window: int = 10) -> pd.Series:
    idx = np.arange(window, dtype=float)
    idx_mean = float(np.mean(idx))
    denom = float(np.sum((idx - idx_mean) ** 2))

    def _slope(values: np.ndarray) -> float:
        y = values.astype(float)
        y_mean = float(np.mean(y))
        num = float(np.sum((idx - idx_mean) * (y - y_mean)))
        return num / denom if denom > 0 else 0.0

    return series.rolling(window=window, min_periods=window).apply(_slope, raw=True)


def _entry_mask_from_signal(signal: pd.Series) -> pd.Series:
    s = signal.fillna(0.0).astype(float)
    change = (s != s.shift(1)).fillna(True)
    return (change & (s != 0.0)).astype(bool)


def _resolve_signal(df: pd.DataFrame, signal_col: str | pd.Series) -> pd.Series:
    if isinstance(signal_col, pd.Series):
        return signal_col.reindex(df.index).fillna(0.0).astype(float)
    if signal_col not in df.columns:
        raise ValueError(f"Signal column not found: {signal_col}")
    return pd.to_numeric(df[signal_col], errors="coerce").fillna(0.0).astype(float)


def _infer_filter_type_from_regime(df: pd.DataFrame) -> pd.Series:
    if "stable_trend_regime" in df.columns:
        trend_reg = df["stable_trend_regime"].astype(str).str.upper()
        return np.where(trend_reg.str.contains("TREND"), "trend", "mean_reversion")
    if "stable_regime_label" in df.columns:
        label = df["stable_regime_label"].astype(str).str.upper()
        return np.where(label.str.contains("TREND"), "trend", "mean_reversion")
    return pd.Series("global", index=df.index)


def _compute_forward_trade_returns(
    df: pd.DataFrame,
    signal: pd.Series,
    forward_horizon: int,
) -> pd.Series:
    close = pd.to_numeric(df["close"], errors="coerce")
    side = np.sign(signal.reindex(df.index).fillna(0.0).astype(float))
    forward_price_return = close.shift(-forward_horizon) / close - 1.0
    return side * forward_price_return


def build_trade_meta_features(
    df: pd.DataFrame,
    primary_signal: pd.Series,
    momentum_window: int = 12,
    range_window: int = 50,
    trend_window: int = 20,
) -> pd.DataFrame:
    """Build per-bar meta-features used to evaluate trade quality."""
    required = {
        "close",
        "adx_14",
        "atr_norm",
        "ma_fast_20",
        "ma_slow_50",
        "rsi_14",
        "bb_upper_20_2",
        "bb_lower_20_2",
        "stable_trend_regime",
        "stable_vol_regime",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for meta features: {sorted(missing)}")

    signal = primary_signal.reindex(df.index).fillna(0.0).astype(float)
    close = df["close"]
    ma_fast = df["ma_fast_20"]
    ma_slow = df["ma_slow_50"]
    bb_width = (df["bb_upper_20_2"] - df["bb_lower_20_2"]).replace(0.0, np.nan)
    atr_norm = df["atr_norm"]
    filter_type = pd.Series(_infer_filter_type_from_regime(df), index=df.index).astype(str)

    change_mask = (signal != signal.shift(1)).fillna(True)
    entry_mask = change_mask & (signal != 0.0)

    time_since_last_trade = np.zeros(len(df), dtype=float)
    prev_holding_bars = np.zeros(len(df), dtype=float)
    last_trade_idx: int | None = None
    prev_hold = 0.0
    last_entry_idx: int | None = None

    for i in range(len(df)):
        if last_trade_idx is None:
            time_since_last_trade[i] = np.nan
        else:
            time_since_last_trade[i] = float(i - last_trade_idx)
        prev_holding_bars[i] = float(prev_hold) if prev_hold > 0 else np.nan

        if bool(entry_mask.iloc[i]):
            if last_entry_idx is not None:
                prev_hold = float(i - last_entry_idx)
            last_entry_idx = i
            last_trade_idx = i

    features = pd.DataFrame(index=df.index)
    features["entry_mask"] = entry_mask.astype(int)
    features["signal_side"] = np.sign(signal)
    features["filter_type"] = filter_type

    # Regime features
    features["stable_trend_regime"] = df["stable_trend_regime"].astype(str)
    features["stable_vol_regime"] = df["stable_vol_regime"].astype(str)

    # Volatility features (past-looking only).
    features["atr_norm"] = atr_norm
    features["atr_norm_change"] = atr_norm.pct_change(5)
    atr_mean = atr_norm.rolling(trend_window, min_periods=max(5, trend_window // 2)).mean()
    features["atr_expansion"] = atr_norm / atr_mean.replace(0.0, np.nan)
    features["recent_volatility_change"] = atr_norm.diff()

    # Trend context.
    features["adx_14"] = df["adx_14"]
    features["ma_fast_slope_10"] = _rolling_slope(ma_fast, window=10) / close.replace(0.0, np.nan)
    features["distance_from_ma_fast"] = (close - ma_fast) / close.replace(0.0, np.nan)
    features["trend_slope"] = _rolling_slope(ma_fast, window=trend_window) / close.replace(0.0, np.nan)
    features["price_vs_ma"] = (close - ma_fast) / close.replace(0.0, np.nan)

    # Mean reversion context.
    features["rsi_14"] = df["rsi_14"]
    features["dist_to_bb_upper"] = (df["bb_upper_20_2"] - close) / close.replace(0.0, np.nan)
    features["dist_to_bb_lower"] = (close - df["bb_lower_20_2"]) / close.replace(0.0, np.nan)
    features["bb_zscore"] = (close - df["bb_mid_20"]) / (bb_width / 2.0)

    # Market state
    features["momentum_3"] = close.pct_change(3)
    features["momentum_n"] = close.pct_change(momentum_window)
    features["range_compression"] = atr_norm / atr_norm.rolling(20, min_periods=10).mean()

    # Trade-relative context.
    ma_strength = (ma_fast - ma_slow).abs() / close.replace(0.0, np.nan)
    rsi_strength = (df["rsi_14"] - 50.0).abs() / 50.0
    features["signal_strength"] = np.where(
        filter_type == "trend",
        ma_strength,
        rsi_strength,
    )

    rolling_min = close.rolling(range_window, min_periods=max(10, range_window // 2)).min()
    rolling_max = close.rolling(range_window, min_periods=max(10, range_window // 2)).max()
    range_span = (rolling_max - rolling_min).replace(0.0, np.nan)
    features["position_in_range"] = (close - rolling_min) / range_span
    features["distance_to_high"] = (rolling_max - close) / close.replace(0.0, np.nan)
    features["distance_to_low"] = (close - rolling_min) / close.replace(0.0, np.nan)

    # Trade context
    bars_since = pd.Series(time_since_last_trade, index=df.index)
    features["time_since_last_trade"] = bars_since
    features["bars_since_last_trade"] = bars_since
    features["prev_holding_bars"] = pd.Series(prev_holding_bars, index=df.index)

    return features


def create_trade_success_labels(
    df: pd.DataFrame,
    signal_col: str | pd.Series,
    entry_mask: pd.Series | None = None,
    forward_horizon: int = 24,
    method: str = "top_quantile",
    quantile: float = 0.3,
    cost_threshold: float | None = None,
    horizon_bars: int | None = None,  # backward-compatible alias
    success_threshold: float | None = None,  # backward-compatible alias for cost threshold
) -> pd.Series:
    """
    Create trade-success labels aligned to df.index.

    Modes:
      - directional_accuracy
      - cost_adjusted_return
      - top_quantile (default)
    """
    if horizon_bars is not None:
        forward_horizon = int(horizon_bars)
    if success_threshold is not None and cost_threshold is None:
        cost_threshold = float(success_threshold)
    if forward_horizon <= 0:
        raise ValueError("forward_horizon must be > 0")

    signal = _resolve_signal(df, signal_col)
    side = np.sign(signal)
    if entry_mask is None:
        entry_mask = _entry_mask_from_signal(signal)
    entry_mask = entry_mask.reindex(df.index).fillna(False).astype(bool)

    close = pd.to_numeric(df["close"], errors="coerce")
    forward_price_return = close.shift(-forward_horizon) / close - 1.0
    forward_trade_return = side * forward_price_return
    valid = entry_mask & (side != 0.0) & forward_trade_return.notna()

    labels = pd.Series(np.nan, index=df.index, dtype=float)
    method_n = str(method).lower()

    if method_n == "directional_accuracy":
        direction_ok = np.sign(forward_price_return) == np.sign(side)
        labels.loc[valid] = direction_ok.loc[valid].astype(int).astype(float)
    elif method_n == "cost_adjusted_return":
        if cost_threshold is None:
            spread_bps = pd.to_numeric(df.get("spread_bps", 0.0), errors="coerce").fillna(0.0)
            slippage_bps = pd.to_numeric(df.get("slippage_bps", 0.0), errors="coerce").fillna(0.0)
            commission_bps = pd.to_numeric(df.get("commission_bps", 0.0), errors="coerce").fillna(0.0)
            threshold_series = 2.0 * (spread_bps + slippage_bps + commission_bps) / 10_000.0
        else:
            threshold_series = pd.Series(float(cost_threshold), index=df.index)
        labels.loc[valid] = (
            forward_trade_return.loc[valid] > threshold_series.loc[valid]
        ).astype(int).astype(float)
    elif method_n == "top_quantile":
        q = float(np.clip(quantile, 0.05, 0.95))
        returns = forward_trade_return.loc[valid]
        if returns.empty:
            return labels
        filter_type = pd.Series(_infer_filter_type_from_regime(df), index=df.index).astype(str)
        grouped_idx = filter_type.loc[valid].groupby(filter_type.loc[valid]).groups
        for t, idx in grouped_idx.items():
            grp_ret = returns.loc[idx]
            if len(grp_ret) < 8:
                continue
            cutoff = float(grp_ret.quantile(1.0 - q))
            labels.loc[idx] = (grp_ret >= cutoff).astype(int).astype(float)
        remaining = valid & labels.isna()
        if remaining.any():
            rem_ret = returns.loc[remaining]
            cutoff = float(rem_ret.quantile(1.0 - q))
            labels.loc[remaining] = (rem_ret >= cutoff).astype(int).astype(float)
    else:
        raise ValueError(
            f"Unsupported label method: {method}. Use directional_accuracy, "
            "cost_adjusted_return, or top_quantile."
        )

    return labels


@dataclass
class RuleBasedMetaFilter:
    """Simple meta model that scores trade-quality probabilities."""

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

        threshold = self._optimize_threshold(
            proba=proba_s,
            forward_returns=returns_s,
        )
        state["threshold"] = float(threshold)
        return state

    def _optimize_threshold(
        self,
        proba: pd.Series,
        forward_returns: pd.Series,
    ) -> float:
        p = proba.astype(float).dropna()
        r = forward_returns.reindex(p.index).astype(float).fillna(0.0)
        if p.empty:
            return 0.5

        q_grid = np.linspace(0.05, 0.95, 37)
        candidates = sorted({float(p.quantile(q)) for q in q_grid} | {0.5})
        best: tuple[float, float, float, float] | None = None
        best_thr = 0.5
        target = float(np.clip(self.target_filter_rate, self.min_filter_rate, self.max_filter_rate))

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
            tie = -abs(filter_rate - target)
            key = (expectancy, sharpe, tie, -filter_rate)
            if best is None or key > best:
                best = key
                best_thr = float(thr)

        if best is not None:
            return best_thr

        # Fallback if no threshold satisfies the filter-rate constraint.
        fallback_q = float(np.clip(1.0 - target, 0.01, 0.99))
        return float(p.quantile(fallback_q))

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

    def _score(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted_:
            raise ValueError("RuleBasedMetaFilter must be fit before scoring.")
        state = {
            "feature_columns": self.feature_columns,
            "numeric_weights": self.numeric_weights,
            "categorical_effects": self.categorical_effects,
            "numeric_medians": self.numeric_medians,
            "numeric_means": self.numeric_means,
            "numeric_stds": self.numeric_stds,
            "bias": self.bias,
        }
        return self._score_from_state(X, state)

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
        """Apply learned calibration to new data without refitting."""
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
        """Filter entry events using learned meta decisions."""
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
        """Serialize learned calibration for fold diagnostics."""
        return {
            "target_filter_rate": float(self.target_filter_rate),
            "threshold": float(self.threshold),
            "bias": float(self.bias),
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
        # Detect sleeve/source from regime when available.
        filter_type_series = pd.Series(_infer_filter_type_from_regime(context_df), index=signal.index)
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

        # Entry or flip into a new non-zero direction.
        if bool(entries.loc[idx]) and int(take.loc[idx]) == 0:
            # Skip new desired direction.
            current = 0.0 if np.isclose(current, 0.0) else 0.0
            out.iloc[i] = current
            continue

        current = desired
        out.iloc[i] = current

    return out
