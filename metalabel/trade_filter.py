from __future__ import annotations

from dataclasses import dataclass, field

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


def build_trade_meta_features(
    df: pd.DataFrame,
    primary_signal: pd.Series,
    momentum_window: int = 12,
) -> pd.DataFrame:
    """Build per-bar meta-features used to evaluate trade quality."""
    required = {
        "close",
        "adx_14",
        "atr_norm",
        "ma_fast_20",
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
    bb_width = (df["bb_upper_20_2"] - df["bb_lower_20_2"]).replace(0.0, np.nan)
    atr_norm = df["atr_norm"]

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

    # Regime features
    features["stable_trend_regime"] = df["stable_trend_regime"].astype(str)
    features["stable_vol_regime"] = df["stable_vol_regime"].astype(str)

    # Volatility features
    features["atr_norm"] = atr_norm
    features["atr_norm_change"] = atr_norm.pct_change(5)

    # Trend context
    features["adx_14"] = df["adx_14"]
    features["ma_fast_slope_10"] = _rolling_slope(ma_fast, window=10) / close.replace(0.0, np.nan)
    features["distance_from_ma_fast"] = (close - ma_fast) / close.replace(0.0, np.nan)

    # Mean reversion context
    features["rsi_14"] = df["rsi_14"]
    features["dist_to_bb_upper"] = (df["bb_upper_20_2"] - close) / close.replace(0.0, np.nan)
    features["dist_to_bb_lower"] = (close - df["bb_lower_20_2"]) / close.replace(0.0, np.nan)
    features["bb_zscore"] = (close - df["bb_mid_20"]) / (bb_width / 2.0)

    # Market state
    features["momentum_3"] = close.pct_change(3)
    features["momentum_n"] = close.pct_change(momentum_window)
    features["range_compression"] = atr_norm / atr_norm.rolling(20, min_periods=10).mean()

    # Trade context
    features["time_since_last_trade"] = pd.Series(time_since_last_trade, index=df.index)
    features["prev_holding_bars"] = pd.Series(prev_holding_bars, index=df.index)

    return features


def create_trade_success_labels(
    df: pd.DataFrame,
    signal: pd.Series,
    entry_mask: pd.Series,
    horizon_bars: int = 24,
    success_threshold: float = 0.0002,
) -> pd.DataFrame:
    """Create forward-return trade success labels for entry signals."""
    close = df["close"]
    side = np.sign(signal.reindex(df.index).fillna(0.0).astype(float))
    forward_price_return = close.shift(-horizon_bars) / close - 1.0
    forward_trade_return = side * forward_price_return
    labels = (forward_trade_return > success_threshold).astype(int)

    out = pd.DataFrame(
        {
            "forward_return": forward_trade_return,
            "label_success": labels,
            "entry_mask": entry_mask.reindex(df.index).fillna(False).astype(bool),
        },
        index=df.index,
    )
    return out[out["entry_mask"]].drop(columns=["entry_mask"]).dropna()


@dataclass
class RuleBasedMetaFilter:
    """Simple meta model that scores trade-quality probabilities."""

    target_filter_rate: float = 0.4
    numeric_weights: dict[str, float] = field(default_factory=dict)
    categorical_effects: dict[str, dict[str, float]] = field(default_factory=dict)
    numeric_medians: dict[str, float] = field(default_factory=dict)
    numeric_means: dict[str, float] = field(default_factory=dict)
    numeric_stds: dict[str, float] = field(default_factory=dict)
    threshold: float = 0.5
    bias: float = 0.0
    feature_columns: list[str] = field(default_factory=list)
    fitted_: bool = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RuleBasedMetaFilter":
        if len(X) == 0:
            raise ValueError("Cannot fit meta filter with empty feature matrix.")
        y = y.astype(float).reindex(X.index).fillna(0.0)
        self.feature_columns = X.columns.tolist()

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in X.columns if c not in numeric_cols]

        score = np.zeros(len(X), dtype=float)

        for col in numeric_cols:
            series = X[col].astype(float)
            med = float(series.median()) if not series.dropna().empty else 0.0
            filled = series.fillna(med)
            mean = float(filled.mean())
            std = float(filled.std(ddof=0))
            if std <= 1e-12:
                std = 1.0
            z = (filled - mean) / std
            corr = float(np.corrcoef(z, y)[0, 1]) if len(z) > 1 else 0.0
            if np.isnan(corr):
                corr = 0.0

            self.numeric_medians[col] = med
            self.numeric_means[col] = mean
            self.numeric_stds[col] = std
            self.numeric_weights[col] = corr
            score += corr * z.to_numpy()

        base_rate = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
        self.bias = float(np.log(base_rate / (1.0 - base_rate)))

        for col in cat_cols:
            as_str = X[col].astype(str).fillna("UNKNOWN")
            group_mean = y.groupby(as_str).mean().to_dict()
            effect = {k: float(v - base_rate) for k, v in group_mean.items()}
            self.categorical_effects[col] = effect
            score += as_str.map(effect).fillna(0.0).to_numpy()

        raw = score + self.bias
        proba = _sigmoid(raw)

        filter_rate = float(np.clip(self.target_filter_rate, 0.2, 0.6))
        quantile = float(np.clip(filter_rate, 0.01, 0.99))
        self.threshold = float(np.quantile(proba, quantile))
        self.fitted_ = True
        return self

    def _score(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted_:
            raise ValueError("RuleBasedMetaFilter must be fit before scoring.")
        X = X.copy()
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = np.nan
        X = X[self.feature_columns]

        score = np.zeros(len(X), dtype=float)
        numeric_cols = [c for c in X.columns if c in self.numeric_weights]
        cat_cols = [c for c in X.columns if c not in numeric_cols]

        for col in numeric_cols:
            series = X[col].astype(float).fillna(self.numeric_medians.get(col, 0.0))
            mean = self.numeric_means.get(col, 0.0)
            std = self.numeric_stds.get(col, 1.0)
            z = (series - mean) / std
            score += self.numeric_weights.get(col, 0.0) * z.to_numpy()

        for col in cat_cols:
            effect = self.categorical_effects.get(col, {})
            score += X[col].astype(str).fillna("UNKNOWN").map(effect).fillna(0.0).to_numpy()

        return score + self.bias

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        raw_score = self._score(X)
        return pd.Series(_sigmoid(raw_score), index=X.index, name="meta_take_proba")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        proba = self.predict_proba(X)
        return (proba >= self.threshold).astype(int).rename("meta_take")


def apply_meta_trade_filter(
    primary_signal: pd.Series,
    entry_mask: pd.Series,
    meta_take_decision: pd.Series,
) -> pd.Series:
    """Apply meta decisions on top of the primary signal."""
    signal = primary_signal.fillna(0.0).astype(float)
    entries = entry_mask.reindex(signal.index).fillna(False).astype(bool)
    take = meta_take_decision.reindex(signal.index).fillna(1).astype(int)

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
