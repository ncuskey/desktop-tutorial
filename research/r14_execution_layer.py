from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data import CostModel, add_basic_indicators, attach_costs, ensure_mock_ohlcv_csv, load_ohlcv_csv, load_symbol_data
from execution.simulator import run_backtest
from metalabel.features_trade_quality import build_trade_meta_features
from metalabel.filter_rule_based import RuleBasedMetaFilter
from metalabel.labels import create_trade_success_labels, entry_mask_from_signal
from metrics.performance import compute_metrics
from regime import attach_regime_labels, attach_stable_regime_state
from strategies import trend_breakout_v2_signals
from research.r153_composite_scoring import FAILURE_SCORE_CUTOFF, compute_failure_score
from research.r16_position_sizing import compute_position_size


EARLY_WINDOW = 3
META_THRESHOLD = 0.60
SCALE_THRESHOLD = 0.75
MIN_HOLD_BARS = EARLY_WINDOW
USE_R15_RULES = True
EARLY_MFE_THRESHOLD = 0.00065
EARLY_RETURN3_THRESHOLD = -0.00024
ENABLE_RULE_LOGGING = True
USE_R153_COMPOSITE = True
USE_R16_POSITION_SIZING = True
MAX_POSITION_SIZE = 2.0


@dataclass
class R14ExecutionArtifacts:
    comparison: pd.DataFrame
    fold_results: pd.DataFrame
    coverage: pd.DataFrame
    conditional_stats: pd.DataFrame
    rule_effectiveness: pd.DataFrame
    output_dir: Path


@dataclass
class _FallbackScorer:
    feature_weights: dict[str, float]
    threshold: float

    def predict_proba(self, x_eval: pd.DataFrame) -> pd.Series:
        if x_eval.empty:
            return pd.Series(dtype=float)
        row = x_eval.iloc[0]
        score = 0.0
        for feat, weight in self.feature_weights.items():
            value = _safe_float(row.get(feat), 0.0)
            score += weight * value
        # squash to [0,1] deterministically
        proba = 1.0 / (1.0 + float(np.exp(-np.clip(score, -20.0, 20.0))))
        return pd.Series([proba], index=x_eval.index, dtype=float)

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if np.isfinite(out):
            return out
        return default
    except Exception:
        return default


def _to_native(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_to_native(v) for v in value]
    return value


def _clean_param_string(raw: Any) -> str:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return "{}"
    text = str(raw).strip()
    if not text:
        return "{}"
    replacements = {
        "np.True_": "True",
        "np.False_": "False",
        "np.nan": "None",
        "nan": "None",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"np\.(?:int|float|bool)\d*\(([^()]*)\)", r"\1", text)
    text = re.sub(r"np\.(?:int|float|bool)_\(([^()]*)\)", r"\1", text)
    text = text.replace("true", "True").replace("false", "False").replace("null", "None")
    return text


def _parse_params(raw: Any) -> dict[str, Any]:
    cleaned = _clean_param_string(raw)
    try:
        parsed = ast.literal_eval(cleaned)
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(k): _to_native(v) for k, v in parsed.items()}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _params_equal(left: dict[str, Any], right: dict[str, Any], tol: float = 1e-12) -> bool:
    if set(left.keys()) != set(right.keys()):
        return False
    for key in left:
        lv = _to_native(left[key])
        rv = _to_native(right[key])
        if _is_number(lv) and _is_number(rv):
            if abs(float(lv) - float(rv)) > tol:
                return False
        elif lv != rv:
            return False
    return True


def _resolve_symbol_file(artifacts_root: Path, symbol: str, filename: str) -> Path:
    candidates = [
        artifacts_root / symbol / filename,
        artifacts_root / symbol.upper() / filename,
        artifacts_root / symbol.lower() / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Missing artifact for symbol '{symbol}': {filename} under {artifacts_root}"
    )


def _resolve_price_data_path(
    artifacts_root: Path,
    explicit_source_csv: str | Path | None,
    symbols: list[str],
) -> Path:
    if explicit_source_csv is not None:
        path = Path(explicit_source_csv)
        if not path.exists():
            raise FileNotFoundError(f"source-csv not found: {path}")
        return path

    candidates = [
        artifacts_root / "r1_shared_ohlcv.csv",
        Path("outputs/TrendBreakout_V2/r1_shared_ohlcv.csv"),
        Path("outputs/strategy_research_mock_ohlcv.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p

    fallback = Path("outputs/strategy_research_mock_ohlcv.csv")
    ensure_mock_ohlcv_csv(fallback, symbols=symbols, periods=10_000, freq="1h", seed=73)
    return fallback


def _load_hardened_params(artifacts_root: Path, symbol: str) -> dict[str, Any]:
    rec_path = _resolve_symbol_file(artifacts_root, symbol, "strategy_research_recommendation.csv")
    rec = pd.read_csv(rec_path)
    hardened = rec[rec["candidate_type"].astype(str).str.upper() == "HARDENED_DEFAULT"]
    if hardened.empty:
        raise ValueError(f"No HARDENED_DEFAULT found in {rec_path}")
    return _parse_params(hardened.iloc[0].get("params"))


def _load_hardened_folds(
    artifacts_root: Path,
    symbol: str,
    hardened_params: dict[str, Any],
) -> pd.DataFrame:
    fold_path = _resolve_symbol_file(artifacts_root, symbol, "strategy_research_fold_results.csv")
    folds = pd.read_csv(fold_path)
    parsed = folds["params"].apply(_parse_params)
    matched = folds[parsed.apply(lambda p: _params_equal(p, hardened_params))].copy()
    if matched.empty:
        raise ValueError(
            f"Could not match HARDENED_DEFAULT params to fold rows for {symbol} ({fold_path})"
        )
    for col in ("fold_start", "fold_train_end", "fold_test_start", "fold_test_end"):
        matched[col] = pd.to_datetime(matched[col], utc=True, errors="coerce")
    matched = matched.dropna(subset=["fold_start", "fold_train_end", "fold_test_start", "fold_test_end"])
    matched = matched.sort_values("fold_test_start").reset_index(drop=True)
    matched["fold_id"] = np.arange(len(matched), dtype=int)
    return matched


def _prepare_symbol_frame(
    raw_prices: pd.DataFrame,
    symbol: str,
    timeframe: str,
    cost_model: CostModel,
) -> pd.DataFrame:
    df = load_symbol_data(raw_prices, symbol=symbol, timeframe=timeframe).copy()
    df = add_basic_indicators(df)
    df = attach_regime_labels(df, adx_threshold=25.0)
    df = attach_stable_regime_state(
        df,
        raw_regime_col="regime_label",
        adx_col="adx_14",
        atr_norm_col="atr_norm",
        atr_norm_pct_col="atr_norm_pct_rank",
        enter_trending=28.0,
        exit_trending=22.0,
        min_regime_bars=12,
        confirm_bars=6,
    )
    df = attach_costs(df, cost_model=cost_model)
    required = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "atr_14",
        "atr_norm",
        "atr_norm_pct_rank",
        "adx_14",
        "ma_fast_20",
        "ma_slow_50",
        "rsi_14",
        "bb_mid_20",
        "bb_upper_20_2",
        "bb_lower_20_2",
        "stable_trend_regime",
        "stable_vol_regime",
    ]
    return df.dropna(subset=required).reset_index(drop=True)


def _extract_trade_segments(signal: pd.Series) -> list[tuple[int, int, int]]:
    s = pd.to_numeric(signal, errors="coerce").fillna(0.0).astype(float).to_numpy()
    side = np.sign(s)
    segments: list[tuple[int, int, int]] = []
    current_side = 0
    entry_i: int | None = None
    for i, side_i in enumerate(side):
        this_side = int(np.sign(side_i))
        if current_side == 0 and this_side != 0:
            current_side = this_side
            entry_i = i
            continue
        if current_side != 0 and this_side != current_side:
            if entry_i is not None:
                segments.append((entry_i, i - 1, current_side))
            if this_side != 0:
                current_side = this_side
                entry_i = i
            else:
                current_side = 0
                entry_i = None
    if current_side != 0 and entry_i is not None:
        segments.append((entry_i, len(signal) - 1, current_side))
    return segments


def _compute_early_features_for_trade(
    df: pd.DataFrame,
    entry_i: int,
    side: int,
    early_window: int,
) -> dict[str, float] | None:
    if early_window < 1:
        return None
    end_i = entry_i + early_window
    if end_i >= len(df):
        return None

    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    atr_norm = pd.to_numeric(df["atr_norm"], errors="coerce")
    atr_abs = pd.to_numeric(df["atr_14"], errors="coerce")

    entry_px = _safe_float(close.iloc[entry_i], default=np.nan)
    if not np.isfinite(entry_px) or entry_px == 0.0:
        return None

    window_close = close.iloc[entry_i + 1 : end_i + 1].astype(float)
    window_high = high.iloc[entry_i + 1 : end_i + 1].astype(float)
    window_low = low.iloc[entry_i + 1 : end_i + 1].astype(float)
    if window_close.empty:
        return None

    idx1 = entry_i + 1
    idx3 = min(entry_i + 3, end_i)

    side_f = float(np.sign(side))
    early_return_1 = side_f * (_safe_float(close.iloc[idx1], np.nan) / entry_px - 1.0)
    early_return_3 = side_f * (_safe_float(close.iloc[idx3], np.nan) / entry_px - 1.0)

    ret_series = window_close.pct_change().dropna()
    early_volatility = float(ret_series.std(ddof=0)) if len(ret_series) > 1 else 0.0

    max_high = _safe_float(window_high.max(), np.nan)
    min_low = _safe_float(window_low.min(), np.nan)
    atr_entry = _safe_float(atr_abs.iloc[entry_i], np.nan)
    if not np.isfinite(atr_entry) or atr_entry <= 0:
        atr_entry = abs(entry_px) * max(_safe_float(atr_norm.iloc[entry_i], 0.0), 1e-6)
    atr_entry = max(atr_entry, 1e-9)

    if side_f > 0:
        early_mfe = (max_high - entry_px) / entry_px
        early_mae = (entry_px - min_low) / entry_px
    else:
        early_mfe = (entry_px - min_low) / entry_px
        early_mae = (max_high - entry_px) / entry_px

    early_range_expansion = (max_high - min_low) / atr_entry
    early_slope = side_f * ((_safe_float(window_close.iloc[-1], np.nan) - _safe_float(window_close.iloc[0], np.nan)) / (entry_px * max(len(window_close) - 1, 1)))

    atr_window = atr_norm.iloc[entry_i : end_i + 1].astype(float)
    atr_entry_norm = _safe_float(atr_norm.iloc[entry_i], np.nan)
    if not np.isfinite(atr_entry_norm) or atr_entry_norm <= 0:
        volatility_spike_flag = 0.0
    else:
        volatility_spike_flag = float((atr_window.max() >= 1.2 * atr_entry_norm))

    return {
        "early_return_1": float(early_return_1),
        "early_return_3": float(early_return_3),
        "early_volatility": float(early_volatility),
        "early_range_expansion": float(early_range_expansion),
        "early_mfe": float(early_mfe),
        "early_mae": float(early_mae),
        "early_slope": float(early_slope),
        "volatility_spike_flag": float(volatility_spike_flag),
    }


def _fit_fold_meta_model(
    train_df: pd.DataFrame,
    strategy_params: dict[str, Any],
    early_window: int,
    forward_horizon: int,
    label_method: str,
    label_quantile: float,
    meta_min_train_samples: int,
    allow_fallback_scorer: bool = True,
) -> tuple[RuleBasedMetaFilter | None, pd.DataFrame, dict[str, Any] | None]:
    train_signal = trend_breakout_v2_signals(train_df, strategy_params).astype(float)
    base_features = build_trade_meta_features(train_df, train_signal)
    entry_mask = entry_mask_from_signal(train_signal)
    labels = create_trade_success_labels(
        train_df,
        train_signal,
        entry_mask=entry_mask,
        forward_horizon=forward_horizon,
        method=label_method,
        quantile=label_quantile,
    )
    entry_rows = base_features.loc[entry_mask].drop(columns=["entry_mask"], errors="ignore").copy()
    if entry_rows.empty:
        return None, pd.DataFrame(), None

    train_records: list[dict[str, Any]] = []
    for idx in entry_rows.index:
        entry_i = int(train_df.index.get_loc(idx))
        side = int(np.sign(_safe_float(train_signal.loc[idx], 0.0)))
        if side == 0:
            continue
        early = _compute_early_features_for_trade(
            df=train_df,
            entry_i=entry_i,
            side=side,
            early_window=early_window,
        )
        # R1.4.1 target: follow-through label (TP before SL within horizon), not top-quantile.
        y = _compute_follow_through_label(
            train_df,
            entry_i=entry_i,
            side=side,
            tp_atr_mult=1.0,
            sl_atr_mult=0.5,
            max_horizon=forward_horizon,
        )
        if early is None or pd.isna(y):
            continue
        row = entry_rows.loc[idx].to_dict()
        row.update(early)
        row["label"] = int(y)
        train_records.append(row)

    if not train_records:
        return None, pd.DataFrame(), None

    train_meta = pd.DataFrame(train_records)
    y_train = train_meta["label"].astype(int)
    x_train = train_meta.drop(columns=["label"], errors="ignore")
    if len(x_train) < meta_min_train_samples or y_train.nunique() < 2:
        fallback_state = _build_fallback_state(train_meta) if allow_fallback_scorer else None
        return None, train_meta, fallback_state

    model = RuleBasedMetaFilter(target_filter_rate=0.4, min_filter_rate=0.2, max_filter_rate=0.6)
    model.fit(x_train, y_train)
    return model, train_meta, None


def _build_fallback_state(train_meta: pd.DataFrame) -> dict[str, Any] | None:
    if train_meta.empty or "label" not in train_meta.columns:
        return None
    labels = pd.to_numeric(train_meta["label"], errors="coerce")
    if labels.dropna().nunique() < 2:
        return None

    candidate_cols = [
        "early_return_1",
        "early_return_3",
        "early_slope",
        "early_mfe",
        "early_mae",
        "early_volatility",
        "early_range_expansion",
        "volatility_spike_flag",
    ]
    feature_cols: list[str] = []
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    weights: dict[str, float] = {}

    y = labels.astype(float)
    for col in candidate_cols:
        if col not in train_meta.columns:
            continue
        x = pd.to_numeric(train_meta[col], errors="coerce")
        valid = x.notna() & y.notna()
        if valid.sum() < 3:
            continue
        xv = x.loc[valid].astype(float)
        yv = y.loc[valid].astype(float)
        std = float(xv.std(ddof=0))
        if std <= 1e-12:
            continue
        mean = float(xv.mean())
        z = (xv - mean) / std
        corr = float(np.corrcoef(z, yv)[0, 1]) if len(z) > 1 else 0.0
        if not np.isfinite(corr):
            corr = 0.0
        feature_cols.append(col)
        means[col] = mean
        stds[col] = std
        weights[col] = corr

    if not feature_cols:
        return None

    base_rate = float(np.clip(y.mean(), 1e-6, 1.0 - 1e-6))
    bias = float(np.log(base_rate / (1.0 - base_rate)))
    return {
        "feature_cols": feature_cols,
        "means": means,
        "stds": stds,
        "weights": weights,
        "bias": bias,
    }


def _fallback_meta_score(x_eval: pd.DataFrame, fallback_state: dict[str, Any] | None) -> float:
    if fallback_state is None or x_eval.empty:
        return float("nan")
    row = x_eval.iloc[0]
    score = float(fallback_state.get("bias", 0.0))
    for col in fallback_state.get("feature_cols", []):
        mean = float(fallback_state["means"].get(col, 0.0))
        std = float(max(fallback_state["stds"].get(col, 1.0), 1e-12))
        w = float(fallback_state["weights"].get(col, 0.0))
        value = _safe_float(row.get(col), np.nan)
        if not np.isfinite(value):
            value = mean
        score += w * ((value - mean) / std)
    clipped = np.clip(score, -30.0, 30.0)
    return float(1.0 / (1.0 + np.exp(-clipped)))


def _compute_follow_through_label(
    df: pd.DataFrame,
    entry_i: int,
    side: int,
    tp_atr_mult: float,
    sl_atr_mult: float,
    max_horizon: int,
) -> float:
    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    atr = pd.to_numeric(df["atr_14"], errors="coerce")
    if entry_i >= len(df) - 1:
        return np.nan

    entry_px = _safe_float(close.iloc[entry_i], np.nan)
    atr_i = _safe_float(atr.iloc[entry_i], np.nan)
    if not np.isfinite(entry_px):
        return np.nan
    if not np.isfinite(atr_i) or atr_i <= 0:
        atr_i = abs(entry_px) * max(_safe_float(df["atr_norm"].iloc[entry_i], 0.0), 1e-6)
    atr_i = max(atr_i, 1e-9)

    horizon_end = min(len(df) - 1, entry_i + max_horizon)
    if horizon_end <= entry_i:
        return np.nan

    side_sign = int(np.sign(side))
    if side_sign == 0:
        return np.nan

    if side_sign > 0:
        tp_level = entry_px + tp_atr_mult * atr_i
        sl_level = entry_px - sl_atr_mult * atr_i
        for j in range(entry_i + 1, horizon_end + 1):
            hi = _safe_float(high.iloc[j], np.nan)
            lo = _safe_float(low.iloc[j], np.nan)
            if np.isfinite(lo) and lo <= sl_level:
                return 0.0
            if np.isfinite(hi) and hi >= tp_level:
                return 1.0
    else:
        tp_level = entry_px - tp_atr_mult * atr_i
        sl_level = entry_px + sl_atr_mult * atr_i
        for j in range(entry_i + 1, horizon_end + 1):
            hi = _safe_float(high.iloc[j], np.nan)
            lo = _safe_float(low.iloc[j], np.nan)
            if np.isfinite(hi) and hi >= sl_level:
                return 0.0
            if np.isfinite(lo) and lo <= tp_level:
                return 1.0

    return 0.0


def _apply_execution_layer_to_fold(
    test_df: pd.DataFrame,
    strategy_params: dict[str, Any],
    model: RuleBasedMetaFilter | None,
    fallback_state: dict[str, Any] | None,
    early_window: int,
    meta_threshold: float,
    scale_threshold: float,
    scale_factor: float,
    enable_scaling: bool,
    min_hold_bars: int,
    enable_r15_rules: bool,
    enable_r153_composite: bool,
    enable_r16_position_sizing: bool,
    max_position_size: float,
) -> tuple[pd.Series, pd.DataFrame]:
    raw_signal = trend_breakout_v2_signals(test_df, strategy_params).astype(float)
    exec_signal = raw_signal.copy()

    base_features = build_trade_meta_features(test_df, raw_signal).drop(columns=["entry_mask"], errors="ignore")
    segments = _extract_trade_segments(raw_signal)
    diagnostics: list[dict[str, Any]] = []

    for entry_i, end_i, side in segments:
        if (entry_i + min_hold_bars) > end_i:
            diagnostics.append(
                {
                    "entry_i": entry_i,
                    "end_i": end_i,
                    "side": side,
                    "evaluated": 0,
                    "meta_score": np.nan,
                    "decision": "continue_short_trade",
                    "rule_exit": 0,
                    "rule_reason": None,
                    "size_multiplier": 1.0,
                }
            )
            continue

        eval_i = entry_i + early_window
        if eval_i > end_i:
            diagnostics.append(
                {
                    "entry_i": entry_i,
                    "end_i": end_i,
                    "side": side,
                    "evaluated": 0,
                    "meta_score": np.nan,
                    "decision": "continue_no_window",
                    "rule_exit": 0,
                    "rule_reason": None,
                    "size_multiplier": 1.0,
                }
            )
            continue

        entry_idx = test_df.index[entry_i]
        eval_idx = test_df.index[eval_i]
        early = _compute_early_features_for_trade(
            df=test_df,
            entry_i=entry_i,
            side=side,
            early_window=early_window,
        )
        if early is None:
            diagnostics.append(
                {
                    "entry_i": entry_i,
                    "end_i": end_i,
                    "side": side,
                    "evaluated": 0,
                    "meta_score": np.nan,
                    "decision": "continue_invalid_early",
                    "rule_exit": 0,
                    "rule_reason": None,
                    "size_multiplier": 1.0,
                }
            )
            continue

        row = base_features.loc[entry_idx].to_dict()
        row.update(early)
        x_eval = pd.DataFrame([row], index=[eval_idx])

        rule_exit_reason: str | None = None
        composite_exit = 0
        failure_score = np.nan

        if enable_r16_position_sizing:
            side_sign = float(np.sign(side))
            size_multiplier = float(
                np.clip(
                    compute_position_size(early),
                    0.0,
                    max(max_position_size, 0.0),
                )
            )
            exec_signal.iloc[eval_i : end_i + 1] = side_sign * size_multiplier
            diagnostics.append(
                {
                    "entry_i": entry_i,
                    "end_i": end_i,
                    "side": side,
                    "evaluated": 1,
                    "meta_score": np.nan,
                    "decision": "r16_position_resize",
                    "used_fallback_scorer": 0,
                    "rule_exit": 0,
                    "rule_reason": None,
                    "composite_exit": 0,
                    "failure_score": np.nan,
                    "size_multiplier": size_multiplier,
                    **early,
                }
            )
            continue

        # R1.5.3: when composite mode is enabled, this is the primary
        # failure-removal decision path and meta logic is bypassed.
        if enable_r153_composite:
            failure_score = float(compute_failure_score(early))
            if failure_score <= float(FAILURE_SCORE_CUTOFF):
                exec_signal.iloc[eval_i : end_i + 1] = 0.0
                diagnostics.append(
                    {
                        "entry_i": entry_i,
                        "end_i": end_i,
                        "side": side,
                        "evaluated": 1,
                        "meta_score": np.nan,
                        "decision": "composite_failure_exit",
                        "used_fallback_scorer": 0,
                        "rule_exit": 0,
                        "rule_reason": None,
                        "composite_exit": 1,
                        "failure_score": failure_score,
                        "size_multiplier": 1.0,
                        **early,
                    }
                )
                continue

            diagnostics.append(
                {
                    "entry_i": entry_i,
                    "end_i": end_i,
                    "side": side,
                    "evaluated": 1,
                    "meta_score": np.nan,
                    "decision": "continue_composite",
                    "used_fallback_scorer": 0,
                    "rule_exit": 0,
                    "rule_reason": None,
                    "composite_exit": 0,
                    "failure_score": failure_score,
                    "size_multiplier": 1.0,
                    **early,
                }
            )
            continue

        if enable_r15_rules:
            if _safe_float(early.get("early_mfe"), np.nan) < EARLY_MFE_THRESHOLD:
                rule_exit_reason = "low_mfe"
            elif _safe_float(early.get("early_return_3"), np.nan) < EARLY_RETURN3_THRESHOLD:
                rule_exit_reason = "negative_momentum"

        if rule_exit_reason is not None:
            exec_signal.iloc[eval_i : end_i + 1] = 0.0
            diagnostics.append(
                {
                    "entry_i": entry_i,
                    "end_i": end_i,
                    "side": side,
                    "evaluated": 1,
                    "meta_score": np.nan,
                    "decision": f"rule_exit_{rule_exit_reason}",
                    "used_fallback_scorer": 0,
                    "rule_exit": 1,
                    "rule_reason": rule_exit_reason,
                    "composite_exit": composite_exit,
                    "failure_score": failure_score,
                    "size_multiplier": 1.0,
                    **early,
                }
            )
            continue

        used_fallback = False
        if model is None:
            score = _fallback_meta_score(x_eval, fallback_state=fallback_state)
            used_fallback = np.isfinite(score)
            if not np.isfinite(score):
                decision = "continue_unfitted"
            elif score < meta_threshold:
                exec_signal.iloc[eval_i : end_i + 1] = 0.0
                decision = "early_fail_exit"
            elif enable_scaling and score >= scale_threshold:
                side_sign = float(np.sign(side))
                exec_signal.iloc[eval_i : end_i + 1] = side_sign * scale_factor
                decision = "early_confirm_scale"
            else:
                decision = "continue"
        else:
            score = _safe_float(model.predict_proba(x_eval).iloc[0], np.nan)
            if not np.isfinite(score):
                decision = "continue_nan_score"
            elif score < meta_threshold:
                exec_signal.iloc[eval_i : end_i + 1] = 0.0
                decision = "early_fail_exit"
            elif enable_scaling and score >= scale_threshold:
                side_sign = float(np.sign(side))
                exec_signal.iloc[eval_i : end_i + 1] = side_sign * scale_factor
                decision = "early_confirm_scale"
            else:
                decision = "continue"

        diagnostics.append(
            {
                "entry_i": entry_i,
                "end_i": end_i,
                "side": side,
                "evaluated": 1,
                "meta_score": score,
                "decision": decision,
                "used_fallback_scorer": int(used_fallback),
                "rule_exit": 0,
                "rule_reason": None,
                "composite_exit": 0,
                "failure_score": failure_score,
                "size_multiplier": 1.0,
                **early,
            }
        )

    return exec_signal, pd.DataFrame(diagnostics)


def _segment_return(returns: pd.Series, entry_i: int, end_i: int) -> float:
    if len(returns) == 0 or entry_i >= len(returns):
        return 0.0
    end_i = min(end_i, len(returns) - 1)
    if end_i < entry_i:
        return 0.0
    return float(pd.to_numeric(returns.iloc[entry_i : end_i + 1], errors="coerce").fillna(0.0).sum())


def run_r14_execution_layer(
    symbols: list[str],
    timeframe: str = "H1",
    early_window: int = EARLY_WINDOW,
    meta_threshold: float = META_THRESHOLD,
    scale_threshold: float = SCALE_THRESHOLD,
    min_hold_bars: int = MIN_HOLD_BARS,
    disable_scaling: bool = False,
    fixed_size_only: bool = False,
    scale_factor: float = 1.5,
    strategy: str = "TrendBreakout_V2",
    artifacts_root: str | Path = "outputs/TrendBreakout_V2",
    source_csv: str | Path | None = None,
    output_dir: str | Path = "outputs",
    forward_horizon: int = 24,
    label_method: str = "top_quantile",
    label_quantile: float = 0.30,
    meta_min_train_samples: int = 30,
    allow_fallback_scorer: bool = True,
    enable_r15_rules: bool = USE_R15_RULES,
    enable_r153_composite: bool = USE_R153_COMPOSITE,
    enable_r16_position_sizing: bool = USE_R16_POSITION_SIZING,
    max_position_size: float = MAX_POSITION_SIZE,
) -> R14ExecutionArtifacts:
    if strategy != "TrendBreakout_V2":
        raise ValueError("R1.4.1 currently supports strategy='TrendBreakout_V2' only.")
    if not symbols:
        raise ValueError("At least one symbol is required.")
    if early_window < 1:
        raise ValueError("early_window must be >= 1")

    use_scaling = (not enable_r16_position_sizing) and (not disable_scaling) and (not fixed_size_only)
    artifacts_root_path = Path(artifacts_root)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    price_path = _resolve_price_data_path(
        artifacts_root=artifacts_root_path,
        explicit_source_csv=source_csv,
        symbols=symbols,
    )
    raw_prices = load_ohlcv_csv(price_path)
    cost_model = CostModel(spread_bps=0.8, slippage_bps=0.5, commission_bps=0.2)

    fold_rows: list[dict[str, Any]] = []
    cond_rows: list[dict[str, Any]] = []
    symbol_variant_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []

    rule_effectiveness_rows: list[dict[str, Any]] = []
    all_rule_removed_returns: list[float] = []
    all_rule_kept_returns: list[float] = []
    score_distribution_rows: list[dict[str, Any]] = []
    score_vs_return_rows: list[dict[str, Any]] = []
    size_distribution_rows: list[dict[str, Any]] = []
    size_vs_return_rows: list[dict[str, Any]] = []

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
        execution_returns_all: list[pd.Series] = []
        total_trades = 0
        total_evaluated = 0
        total_early_exits = 0
        total_scaled = 0
        total_rule_exits = 0
        total_composite_exits = 0
        total_low_mfe_exits = 0
        total_negative_momentum_exits = 0
        total_removed_trades = 0
        total_position_size = 0.0
        total_position_size_count = 0
        survivor_returns: list[float] = []
        rejected_returns: list[float] = []
        baseline_fail_returns: list[float] = []
        baseline_pass_returns: list[float] = []
        baseline_trade_returns: list[float] = []
        execution_trade_returns: list[float] = []
        loss_avoided_list: list[float] = []
        missed_profit_list: list[float] = []
        symbol_rule_removed_returns: list[float] = []
        symbol_rule_kept_returns: list[float] = []

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

            model, train_meta, fallback_model_state = _fit_fold_meta_model(
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
            exec_signal, decisions = _apply_execution_layer_to_fold(
                test_df=test_df,
                strategy_params=strategy_params,
                model=model,
                fallback_state=fallback_model_state,
                early_window=early_window,
                meta_threshold=meta_threshold,
                scale_threshold=scale_threshold,
                scale_factor=scale_factor,
                enable_scaling=use_scaling,
                min_hold_bars=min_hold_bars,
                enable_r15_rules=enable_r15_rules,
                enable_r153_composite=enable_r153_composite,
                enable_r16_position_sizing=enable_r16_position_sizing,
                max_position_size=max_position_size,
            )

            bt_base = run_backtest(test_df, raw_signal, cost_model=cost_model)
            bt_exec = run_backtest(
                test_df,
                exec_signal,
                cost_model=cost_model,
                max_abs_position=(
                    max_position_size
                    if enable_r16_position_sizing
                    else (max(1.0, scale_factor) if use_scaling else 1.0)
                ),
            )
            met_base = compute_metrics(
                bt_base.returns,
                bt_base.equity,
                bt_base.trades,
                timeframe=timeframe,
                position=bt_base.position,
            )
            met_exec = compute_metrics(
                bt_exec.returns,
                bt_exec.equity,
                bt_exec.trades,
                timeframe=timeframe,
                position=bt_exec.position,
            )

            baseline_returns_all.append(bt_base.returns)
            execution_returns_all.append(bt_exec.returns)

            segments = _extract_trade_segments(raw_signal)
            segment_records: list[dict[str, Any]] = []
            for seg_id, (entry_i, end_i, _) in enumerate(segments):
                baseline_ret = _segment_return(bt_base.returns, entry_i, end_i)
                exec_ret = _segment_return(bt_exec.returns, entry_i, end_i)
                decision_row = decisions[
                    (pd.to_numeric(decisions["entry_i"], errors="coerce") == float(entry_i))
                    & (pd.to_numeric(decisions["end_i"], errors="coerce") == float(end_i))
                ]
                if decision_row.empty:
                    decision_row = decisions[
                        pd.to_numeric(decisions["entry_i"], errors="coerce") == float(entry_i)
                    ]
                if decision_row.empty:
                    decision = "continue_missing"
                    score = np.nan
                    rule_exit = 0
                    rule_reason = None
                    composite_exit = 0
                    failure_score = np.nan
                    size_multiplier = 1.0
                else:
                    decision = str(decision_row.iloc[0].get("decision", "continue_missing"))
                    score = _safe_float(decision_row.iloc[0].get("meta_score"), np.nan)
                    rule_exit = int(_safe_float(decision_row.iloc[0].get("rule_exit"), 0.0))
                    raw_reason = decision_row.iloc[0].get("rule_reason")
                    rule_reason = None if pd.isna(raw_reason) else str(raw_reason)
                    composite_exit = int(_safe_float(decision_row.iloc[0].get("composite_exit"), 0.0))
                    failure_score = _safe_float(decision_row.iloc[0].get("failure_score"), np.nan)
                    size_multiplier = _safe_float(decision_row.iloc[0].get("size_multiplier"), 1.0)

                segment_records.append(
                    {
                        "segment_id": seg_id,
                        "entry_i": entry_i,
                        "end_i": end_i,
                        "baseline_return": baseline_ret,
                        "execution_return": exec_ret,
                        "meta_score": score,
                        "decision": decision,
                        "rule_exit": rule_exit,
                        "rule_reason": rule_reason,
                        "composite_exit": composite_exit,
                        "failure_score": failure_score,
                        "size_multiplier": size_multiplier,
                    }
                )

            seg_df = pd.DataFrame(segment_records)
            trades_fold = int(len(seg_df))
            evaluated_fold = int(pd.to_numeric(seg_df.get("meta_score"), errors="coerce").notna().sum()) if not seg_df.empty else 0
            early_exits_fold = int((seg_df["decision"] == "early_fail_exit").sum()) if not seg_df.empty else 0
            scaled_fold = int((seg_df["decision"] == "early_confirm_scale").sum()) if not seg_df.empty else 0
            rule_exits_fold = int(pd.to_numeric(seg_df.get("rule_exit"), errors="coerce").fillna(0).astype(int).sum()) if not seg_df.empty else 0
            low_mfe_exits_fold = int((seg_df["rule_reason"] == "low_mfe").sum()) if not seg_df.empty else 0
            negative_momentum_exits_fold = int((seg_df["rule_reason"] == "negative_momentum").sum()) if not seg_df.empty else 0
            composite_exits_fold = int(pd.to_numeric(seg_df.get("composite_exit"), errors="coerce").fillna(0).astype(int).sum()) if not seg_df.empty else 0
            if not seg_df.empty:
                if enable_r16_position_sizing:
                    rejected_mask = pd.Series(False, index=seg_df.index, dtype=bool)
                elif enable_r153_composite:
                    rejected_mask = pd.to_numeric(
                        seg_df.get("composite_exit"),
                        errors="coerce",
                    ).fillna(0).astype(int).astype(bool)
                else:
                    rejected_mask = (
                        (seg_df["decision"] == "early_fail_exit")
                        | pd.to_numeric(seg_df.get("rule_exit"), errors="coerce").fillna(0).astype(int).astype(bool)
                    )
            else:
                rejected_mask = pd.Series([], dtype=bool)
            removed_fold = int(rejected_mask.sum()) if not seg_df.empty else 0
            survivors_fold = int((~rejected_mask).sum()) if not seg_df.empty else 0
            survival_pct_fold = float(survivors_fold / max(trades_fold, 1))
            avg_position_size_fold = (
                float(pd.to_numeric(seg_df["size_multiplier"], errors="coerce").fillna(1.0).mean())
                if not seg_df.empty
                else 1.0
            )

            avg_survivor_ret = float(seg_df.loc[~rejected_mask, "execution_return"].mean()) if not seg_df.empty else 0.0
            avg_rejected_ret = float(seg_df.loc[rejected_mask, "baseline_return"].mean()) if not seg_df.empty else 0.0
            baseline_expectancy_fold = float(seg_df["baseline_return"].mean()) if not seg_df.empty else 0.0
            execution_expectancy_fold = float(seg_df["execution_return"].mean()) if not seg_df.empty else 0.0
            pct_rule_exits_fold = float(rule_exits_fold / max(trades_fold, 1))

            fold_rows.append(
                {
                    "symbol": symbol,
                    "fold_id": fold_id,
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "baseline_sharpe": _safe_float(met_base.get("Sharpe")),
                    "execution_sharpe": _safe_float(met_exec.get("Sharpe")),
                    "baseline_expectancy": baseline_expectancy_fold,
                    "execution_expectancy": execution_expectancy_fold,
                    "baseline_max_dd": _safe_float(met_base.get("MaxDrawdown")),
                    "execution_max_dd": _safe_float(met_exec.get("MaxDrawdown")),
                    "baseline_trade_count": _safe_float(met_base.get("TradeCount")),
                    "execution_trade_count": _safe_float(met_exec.get("TradeCount")),
                    "early_exits_count": early_exits_fold,
                    "rule_exit_count": rule_exits_fold,
                    "pct_rule_exits": pct_rule_exits_fold,
                    "low_mfe_exits": low_mfe_exits_fold,
                    "negative_momentum_exits": negative_momentum_exits_fold,
                    "composite_exit_count": composite_exits_fold,
                    "pct_composite_exits": float(composite_exits_fold / max(trades_fold, 1)),
                    "scaled_count": scaled_fold,
                    "trades_total": trades_fold,
                    "trades_evaluated": evaluated_fold,
                    "survival_pct": survival_pct_fold,
                    "avg_return_survivors": avg_survivor_ret,
                    "avg_return_rejected": avg_rejected_ret,
                    "model_fitted": bool(model is not None),
                    "used_fallback_scorer": bool(model is None and fallback_model_state is not None),
                    "train_meta_samples": int(len(train_meta)),
                    "avg_position_size": avg_position_size_fold,
                    "weighted_expectancy": execution_expectancy_fold,
                    "weighted_return": float(pd.to_numeric(bt_exec.returns, errors="coerce").mean()),
                }
            )

            if not seg_df.empty:
                pass_mask = seg_df["meta_score"] >= meta_threshold
                fail_mask = seg_df["meta_score"] < meta_threshold
                pass_ret = seg_df.loc[pass_mask, "baseline_return"]
                fail_ret = seg_df.loc[fail_mask, "baseline_return"]
                cond_rows.append(
                    {
                        "symbol": symbol,
                        "fold_id": fold_id,
                        "E_return_pass": float(pass_ret.mean()) if not pass_ret.empty else np.nan,
                        "E_return_fail": float(fail_ret.mean()) if not fail_ret.empty else np.nan,
                        "pass_count": int(pass_ret.count()),
                        "fail_count": int(fail_ret.count()),
                    }
                )

                if enable_r153_composite:
                    rejected = seg_df[
                        pd.to_numeric(seg_df.get("composite_exit"), errors="coerce")
                        .fillna(0)
                        .astype(int)
                        .astype(bool)
                    ].copy()
                else:
                    rejected = seg_df[
                        (seg_df["decision"] == "early_fail_exit")
                        | pd.to_numeric(seg_df.get("rule_exit"), errors="coerce").fillna(0).astype(int).astype(bool)
                    ].copy()
                if not rejected.empty:
                    improvement = rejected["execution_return"] - rejected["baseline_return"]
                    avoid = improvement.where(rejected["baseline_return"] < 0, np.nan)
                    missed = (rejected["baseline_return"] - rejected["execution_return"]).where(
                        rejected["baseline_return"] > 0,
                        np.nan,
                    )
                    loss_avoided_list.extend(pd.to_numeric(avoid, errors="coerce").dropna().tolist())
                    missed_profit_list.extend(pd.to_numeric(missed, errors="coerce").dropna().tolist())
                baseline_pass_returns.extend(pass_ret.dropna().tolist())
                baseline_fail_returns.extend(fail_ret.dropna().tolist())
                baseline_trade_returns.extend(
                    pd.to_numeric(seg_df["baseline_return"], errors="coerce").dropna().tolist()
                )
                execution_trade_returns.extend(
                    pd.to_numeric(seg_df["execution_return"], errors="coerce").dropna().tolist()
                )
                survivor_returns.extend(
                    seg_df.loc[~rejected_mask, "execution_return"].dropna().tolist()
                )
                rejected_returns.extend(
                    seg_df.loc[rejected_mask, "baseline_return"].dropna().tolist()
                )
                total_position_size += float(
                    pd.to_numeric(seg_df["size_multiplier"], errors="coerce").fillna(1.0).sum()
                )
                total_position_size_count += int(len(seg_df))
                for _, seg_row in seg_df.iterrows():
                    size_val = _safe_float(seg_row.get("size_multiplier"), 1.0)
                    size_distribution_rows.append(
                        {
                            "symbol": symbol,
                            "size_multiplier": size_val,
                        }
                    )
                    size_vs_return_rows.append(
                        {
                            "symbol": symbol,
                            "size_multiplier": size_val,
                            "realized_return": _safe_float(seg_row.get("baseline_return"), np.nan),
                            "weighted_return": _safe_float(seg_row.get("execution_return"), np.nan),
                        }
                    )

                if enable_r153_composite:
                    composite_removed = seg_df[
                        pd.to_numeric(seg_df.get("composite_exit"), errors="coerce")
                        .fillna(0)
                        .astype(int)
                        .astype(bool)
                    ]["baseline_return"]
                    composite_kept = seg_df[
                        ~pd.to_numeric(seg_df.get("composite_exit"), errors="coerce")
                        .fillna(0)
                        .astype(int)
                        .astype(bool)
                    ]["baseline_return"]
                    symbol_rule_removed_returns.extend(pd.to_numeric(composite_removed, errors="coerce").dropna().tolist())
                    symbol_rule_kept_returns.extend(pd.to_numeric(composite_kept, errors="coerce").dropna().tolist())
                elif enable_r15_rules:
                    rule_removed = seg_df[
                        pd.to_numeric(seg_df.get("rule_exit"), errors="coerce").fillna(0).astype(int).astype(bool)
                    ]["baseline_return"]
                    rule_kept = seg_df[
                        ~pd.to_numeric(seg_df.get("rule_exit"), errors="coerce").fillna(0).astype(int).astype(bool)
                    ]["baseline_return"]
                    symbol_rule_removed_returns.extend(pd.to_numeric(rule_removed, errors="coerce").dropna().tolist())
                    symbol_rule_kept_returns.extend(pd.to_numeric(rule_kept, errors="coerce").dropna().tolist())
                if enable_r153_composite:
                    fs = pd.to_numeric(seg_df.get("failure_score"), errors="coerce")
                    if fs.notna().any():
                        for score_val, grp in seg_df.loc[fs.notna()].groupby(fs.loc[fs.notna()]):
                            score_distribution_rows.append(
                                {
                                    "symbol": symbol,
                                    "failure_score": int(score_val),
                                    "trade_count": int(len(grp)),
                                }
                            )
                            score_vs_return_rows.append(
                                {
                                    "symbol": symbol,
                                    "failure_score": int(score_val),
                                    "avg_realized_return": float(pd.to_numeric(grp["baseline_return"], errors="coerce").mean()),
                                    "trade_count": int(len(grp)),
                                }
                            )

            total_trades += trades_fold
            total_evaluated += evaluated_fold
            total_early_exits += early_exits_fold
            total_scaled += scaled_fold
            total_rule_exits += rule_exits_fold
            total_composite_exits += composite_exits_fold
            total_low_mfe_exits += low_mfe_exits_fold
            total_negative_momentum_exits += negative_momentum_exits_fold
            total_removed_trades += removed_fold

        if not baseline_returns_all or not execution_returns_all:
            continue

        stitched_base = pd.concat(baseline_returns_all).sort_index()
        stitched_exec = pd.concat(execution_returns_all).sort_index()
        eq_base = 100_000.0 * (1.0 + stitched_base).cumprod()
        eq_exec = 100_000.0 * (1.0 + stitched_exec).cumprod()
        mt_base = compute_metrics(stitched_base, eq_base, pd.DataFrame(), timeframe=timeframe)
        mt_exec = compute_metrics(stitched_exec, eq_exec, pd.DataFrame(), timeframe=timeframe)
        baseline_expectancy_symbol = (
            float(np.mean(baseline_trade_returns)) if baseline_trade_returns else np.nan
        )
        execution_expectancy_symbol = (
            float(np.mean(execution_trade_returns)) if execution_trade_returns else np.nan
        )

        symbol_variant_rows.extend(
            [
                {
                    "symbol": symbol,
                    "variant": "baseline",
                    "Sharpe": _safe_float(mt_base.get("Sharpe")),
                    "Expectancy": baseline_expectancy_symbol,
                    "MaxDD": _safe_float(mt_base.get("MaxDrawdown")),
                    "TradeCount": float(total_trades),
                    "EarlyExitsCount": 0,
                    "rule_exit_count": 0,
                    "pct_rule_exits": 0.0,
                    "low_mfe_exits": 0,
                    "negative_momentum_exits": 0,
                    "composite_exit_count": 0,
                    "pct_composite_exits": 0.0,
                    "TradesSurvivingPct": 1.0,
                    "AvgReturnSurvivors": float(np.mean(baseline_pass_returns)) if baseline_pass_returns else np.nan,
                    "AvgReturnRejected": np.nan,
                    "avg_position_size": 1.0,
                    "weighted_expectancy": baseline_expectancy_symbol,
                    "weighted_return": float(pd.to_numeric(stitched_base, errors="coerce").mean()),
                },
                {
                    "symbol": symbol,
                    "variant": "execution_layer",
                    "Sharpe": _safe_float(mt_exec.get("Sharpe")),
                    "Expectancy": execution_expectancy_symbol,
                    "MaxDD": _safe_float(mt_exec.get("MaxDrawdown")),
                    "TradeCount": float(total_trades - total_removed_trades),
                    "EarlyExitsCount": int(total_early_exits),
                    "rule_exit_count": int(total_rule_exits),
                    "pct_rule_exits": float(total_rule_exits / max(total_trades, 1)),
                    "low_mfe_exits": int(total_low_mfe_exits),
                    "negative_momentum_exits": int(total_negative_momentum_exits),
                    "composite_exit_count": int(total_composite_exits if enable_r153_composite else 0),
                    "pct_composite_exits": float((total_composite_exits / max(total_trades, 1)) if enable_r153_composite else 0.0),
                    "TradesSurvivingPct": float((total_trades - total_removed_trades) / max(total_trades, 1)),
                    "AvgReturnSurvivors": float(np.mean(survivor_returns)) if survivor_returns else np.nan,
                    "AvgReturnRejected": float(np.mean(rejected_returns)) if rejected_returns else np.nan,
                    "avg_position_size": float(total_position_size / max(total_position_size_count, 1)),
                    "weighted_expectancy": execution_expectancy_symbol,
                    "weighted_return": float(pd.to_numeric(stitched_exec, errors="coerce").mean()),
                },
            ]
        )

        coverage_rows.append(
            {
                "symbol": symbol,
                "trades_total": int(total_trades),
                "trades_evaluated": int(total_evaluated),
                "early_exits_count": int(total_early_exits),
                "rule_exit_count": int(total_rule_exits),
                "pct_rule_exits": float(total_rule_exits / max(total_trades, 1)),
                "low_mfe_exits": int(total_low_mfe_exits),
                "negative_momentum_exits": int(total_negative_momentum_exits),
                "composite_exit_count": int(total_composite_exits if enable_r153_composite else 0),
                "pct_composite_exits": float((total_composite_exits / max(total_trades, 1)) if enable_r153_composite else 0.0),
                "scaled_count": int(total_scaled),
                "trades_surviving_pct": float((total_trades - total_removed_trades) / max(total_trades, 1)),
                "avg_loss_avoided_by_early_exit": float(np.mean(loss_avoided_list)) if loss_avoided_list else 0.0,
                "avg_missed_profit_false_negatives": float(np.mean(missed_profit_list)) if missed_profit_list else 0.0,
                "E_return_pass": float(np.mean(baseline_pass_returns)) if baseline_pass_returns else np.nan,
                "E_return_fail": float(np.mean(baseline_fail_returns)) if baseline_fail_returns else np.nan,
                "conditional_gap_pass_minus_fail": (
                    float(np.mean(baseline_pass_returns) - np.mean(baseline_fail_returns))
                    if baseline_pass_returns and baseline_fail_returns
                    else np.nan
                ),
                "avg_position_size": float(total_position_size / max(total_position_size_count, 1)),
            }
        )

        if enable_r153_composite:
            e_all = float(np.mean(symbol_rule_kept_returns + symbol_rule_removed_returns)) if (symbol_rule_kept_returns or symbol_rule_removed_returns) else np.nan
            e_after_composite = float(np.mean(symbol_rule_kept_returns)) if symbol_rule_kept_returns else np.nan
            e_removed = float(np.mean(symbol_rule_removed_returns)) if symbol_rule_removed_returns else np.nan
            removed_trade_count = int(len(symbol_rule_removed_returns))
            kept_trade_count = int(len(symbol_rule_kept_returns))
            pct_removed = float(removed_trade_count / max(removed_trade_count + kept_trade_count, 1))
            rule_effectiveness_rows.append(
                {
                    "symbol": symbol,
                    "E_all": e_all,
                    "E_after_composite": e_after_composite,
                    "E_removed": e_removed,
                    "removed_trade_count": removed_trade_count,
                    "kept_trade_count": kept_trade_count,
                    "pct_removed": pct_removed,
                    "rule_low_mfe_count": int(total_low_mfe_exits),
                    "rule_neg_momentum_count": int(total_negative_momentum_exits),
                }
            )
            all_rule_removed_returns.extend(symbol_rule_removed_returns)
            all_rule_kept_returns.extend(symbol_rule_kept_returns)
        elif enable_r15_rules:
            e_all = float(np.mean(symbol_rule_kept_returns + symbol_rule_removed_returns)) if (symbol_rule_kept_returns or symbol_rule_removed_returns) else np.nan
            e_after_rules = float(np.mean(symbol_rule_kept_returns)) if symbol_rule_kept_returns else np.nan
            e_removed = float(np.mean(symbol_rule_removed_returns)) if symbol_rule_removed_returns else np.nan
            removed_trade_count = int(len(symbol_rule_removed_returns))
            kept_trade_count = int(len(symbol_rule_kept_returns))
            pct_removed = float(removed_trade_count / max(removed_trade_count + kept_trade_count, 1))
            rule_effectiveness_rows.append(
                {
                    "symbol": symbol,
                    "E_all": e_all,
                    "E_after_rules": e_after_rules,
                    "E_removed": e_removed,
                    "removed_trade_count": removed_trade_count,
                    "kept_trade_count": kept_trade_count,
                    "pct_removed": pct_removed,
                    "rule_low_mfe_count": int(total_low_mfe_exits),
                    "rule_neg_momentum_count": int(total_negative_momentum_exits),
                }
            )
            all_rule_removed_returns.extend(symbol_rule_removed_returns)
            all_rule_kept_returns.extend(symbol_rule_kept_returns)

    comparison = pd.DataFrame(symbol_variant_rows)
    fold_results = pd.DataFrame(fold_rows)
    coverage = pd.DataFrame(coverage_rows)
    conditional_stats = pd.DataFrame(cond_rows)

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
                    "EarlyExitsCount": int(pd.to_numeric(grp["EarlyExitsCount"], errors="coerce").sum()),
                    "rule_exit_count": int(pd.to_numeric(grp["rule_exit_count"], errors="coerce").sum()),
                    "pct_rule_exits": float(pd.to_numeric(grp["pct_rule_exits"], errors="coerce").mean()),
                    "low_mfe_exits": int(pd.to_numeric(grp["low_mfe_exits"], errors="coerce").sum()),
                    "negative_momentum_exits": int(
                        pd.to_numeric(grp["negative_momentum_exits"], errors="coerce").sum()
                    ),
                    "composite_exit_count": int(pd.to_numeric(grp.get("composite_exit_count"), errors="coerce").sum()) if "composite_exit_count" in grp else 0,
                    "pct_composite_exits": float(pd.to_numeric(grp.get("pct_composite_exits"), errors="coerce").mean()) if "pct_composite_exits" in grp else 0.0,
                    "TradesSurvivingPct": float(pd.to_numeric(grp["TradesSurvivingPct"], errors="coerce").mean()),
                    "AvgReturnSurvivors": float(pd.to_numeric(grp["AvgReturnSurvivors"], errors="coerce").mean()),
                    "AvgReturnRejected": float(pd.to_numeric(grp["AvgReturnRejected"], errors="coerce").mean()),
                    "avg_position_size": float(pd.to_numeric(grp["avg_position_size"], errors="coerce").mean()),
                    "weighted_expectancy": float(pd.to_numeric(grp["weighted_expectancy"], errors="coerce").mean()),
                    "weighted_return": float(pd.to_numeric(grp["weighted_return"], errors="coerce").mean()),
                }
            )
        comparison = pd.concat([comparison, pd.DataFrame(all_rows)], ignore_index=True)

    if not coverage.empty:
        all_cov = {
            "symbol": "ALL",
            "trades_total": int(pd.to_numeric(coverage["trades_total"], errors="coerce").sum()),
            "trades_evaluated": int(pd.to_numeric(coverage["trades_evaluated"], errors="coerce").sum()),
            "early_exits_count": int(pd.to_numeric(coverage["early_exits_count"], errors="coerce").sum()),
            "rule_exit_count": int(pd.to_numeric(coverage["rule_exit_count"], errors="coerce").sum()),
            "pct_rule_exits": float(pd.to_numeric(coverage["pct_rule_exits"], errors="coerce").mean()),
            "low_mfe_exits": int(pd.to_numeric(coverage["low_mfe_exits"], errors="coerce").sum()),
            "negative_momentum_exits": int(
                pd.to_numeric(coverage["negative_momentum_exits"], errors="coerce").sum()
            ),
            "composite_exit_count": int(pd.to_numeric(coverage.get("composite_exit_count"), errors="coerce").sum()) if "composite_exit_count" in coverage else 0,
            "pct_composite_exits": float(pd.to_numeric(coverage.get("pct_composite_exits"), errors="coerce").mean()) if "pct_composite_exits" in coverage else 0.0,
            "scaled_count": int(pd.to_numeric(coverage["scaled_count"], errors="coerce").sum()),
            "trades_surviving_pct": float(pd.to_numeric(coverage["trades_surviving_pct"], errors="coerce").mean()),
            "avg_loss_avoided_by_early_exit": float(
                pd.to_numeric(coverage["avg_loss_avoided_by_early_exit"], errors="coerce").mean()
            ),
            "avg_missed_profit_false_negatives": float(
                pd.to_numeric(coverage["avg_missed_profit_false_negatives"], errors="coerce").mean()
            ),
            "E_return_pass": float(pd.to_numeric(coverage["E_return_pass"], errors="coerce").mean()),
            "E_return_fail": float(pd.to_numeric(coverage["E_return_fail"], errors="coerce").mean()),
            "conditional_gap_pass_minus_fail": float(
                pd.to_numeric(coverage["conditional_gap_pass_minus_fail"], errors="coerce").mean()
            ),
            "avg_position_size": float(pd.to_numeric(coverage["avg_position_size"], errors="coerce").mean()),
        }
        coverage = pd.concat([coverage, pd.DataFrame([all_cov])], ignore_index=True)

    rule_effectiveness = pd.DataFrame(rule_effectiveness_rows)
    if enable_r153_composite:
        removed_count_all = int(len(all_rule_removed_returns))
        kept_count_all = int(len(all_rule_kept_returns))
        if not rule_effectiveness.empty:
            all_row = {
                "symbol": "ALL",
                "E_all": float(np.mean(all_rule_removed_returns + all_rule_kept_returns)) if (all_rule_removed_returns or all_rule_kept_returns) else np.nan,
                "E_after_composite": float(np.mean(all_rule_kept_returns)) if all_rule_kept_returns else np.nan,
                "E_removed": float(np.mean(all_rule_removed_returns)) if all_rule_removed_returns else np.nan,
                "removed_trade_count": removed_count_all,
                "kept_trade_count": kept_count_all,
                "pct_removed": float(removed_count_all / max(removed_count_all + kept_count_all, 1)),
                "rule_low_mfe_count": 0,
                "rule_neg_momentum_count": 0,
            }
            rule_effectiveness = pd.concat([rule_effectiveness, pd.DataFrame([all_row])], ignore_index=True)
    elif enable_r15_rules:
        removed_count_all = int(len(all_rule_removed_returns))
        kept_count_all = int(len(all_rule_kept_returns))
        if not rule_effectiveness.empty:
            low_mfe_total = int(pd.to_numeric(rule_effectiveness["rule_low_mfe_count"], errors="coerce").sum())
            neg_mom_total = int(pd.to_numeric(rule_effectiveness["rule_neg_momentum_count"], errors="coerce").sum())
            all_row = {
                "symbol": "ALL",
                "E_all": float(np.mean(all_rule_removed_returns + all_rule_kept_returns)) if (all_rule_removed_returns or all_rule_kept_returns) else np.nan,
                "E_after_rules": float(np.mean(all_rule_kept_returns)) if all_rule_kept_returns else np.nan,
                "E_removed": float(np.mean(all_rule_removed_returns)) if all_rule_removed_returns else np.nan,
                "removed_trade_count": removed_count_all,
                "kept_trade_count": kept_count_all,
                "pct_removed": float(removed_count_all / max(removed_count_all + kept_count_all, 1)),
                "rule_low_mfe_count": low_mfe_total,
                "rule_neg_momentum_count": neg_mom_total,
            }
            rule_effectiveness = pd.concat([rule_effectiveness, pd.DataFrame([all_row])], ignore_index=True)

    comparison.to_csv(out_dir / "r14_execution_comparison.csv", index=False)
    fold_results.to_csv(out_dir / "r14_execution_fold_results.csv", index=False)
    coverage.to_csv(out_dir / "r14_execution_coverage.csv", index=False)
    conditional_stats.to_csv(out_dir / "r14_execution_conditional_stats.csv", index=False)
    rule_effectiveness.to_csv(out_dir / "r15_rule_effectiveness.csv", index=False)
    if size_distribution_rows:
        size_distribution_df = (
            pd.DataFrame(size_distribution_rows)
            .assign(size_multiplier=lambda d: pd.to_numeric(d["size_multiplier"], errors="coerce").round(4))
            .groupby(["symbol", "size_multiplier"], as_index=False, observed=False)
            .size()
            .rename(columns={"size": "trade_count"})
            .sort_values(["symbol", "size_multiplier"])
            .reset_index(drop=True)
        )
        size_distribution_df.to_csv(out_dir / "r16_size_distribution.csv", index=False)
    else:
        pd.DataFrame(columns=["symbol", "size_multiplier", "trade_count"]).to_csv(
            out_dir / "r16_size_distribution.csv", index=False
        )

    if size_vs_return_rows:
        size_vs_return_df = (
            pd.DataFrame(size_vs_return_rows)
            .assign(size_multiplier=lambda d: pd.to_numeric(d["size_multiplier"], errors="coerce").round(4))
            .groupby(["symbol", "size_multiplier"], as_index=False, observed=False)
            .agg(
                avg_realized_return=("realized_return", "mean"),
                avg_weighted_return=("weighted_return", "mean"),
                trade_count=("size_multiplier", "size"),
            )
            .sort_values(["symbol", "size_multiplier"])
            .reset_index(drop=True)
        )
        size_vs_return_df.to_csv(out_dir / "r16_size_vs_return.csv", index=False)
    else:
        pd.DataFrame(
            columns=["symbol", "size_multiplier", "avg_realized_return", "avg_weighted_return", "trade_count"]
        ).to_csv(out_dir / "r16_size_vs_return.csv", index=False)
    if score_distribution_rows:
        score_distribution_df = (
            pd.DataFrame(score_distribution_rows)
            .groupby(["symbol", "failure_score"], as_index=False, observed=False)["trade_count"]
            .sum()
            .sort_values(["symbol", "failure_score"])
            .reset_index(drop=True)
        )
        score_distribution_df.to_csv(out_dir / "r153_score_distribution.csv", index=False)
    else:
        pd.DataFrame(columns=["symbol", "failure_score", "trade_count"]).to_csv(
            out_dir / "r153_score_distribution.csv", index=False
        )

    if score_vs_return_rows:
        score_vs_return_df = (
            pd.DataFrame(score_vs_return_rows)
            .groupby(["symbol", "failure_score"], as_index=False, observed=False)
            .agg(
                avg_realized_return=("avg_realized_return", "mean"),
                trade_count=("trade_count", "sum"),
            )
            .sort_values(["symbol", "failure_score"])
            .reset_index(drop=True)
        )
        score_vs_return_df.to_csv(out_dir / "r153_score_vs_return.csv", index=False)
    else:
        pd.DataFrame(columns=["symbol", "failure_score", "avg_realized_return", "trade_count"]).to_csv(
            out_dir / "r153_score_vs_return.csv", index=False
        )

    return R14ExecutionArtifacts(
        comparison=comparison,
        fold_results=fold_results,
        coverage=coverage,
        conditional_stats=conditional_stats,
        rule_effectiveness=rule_effectiveness,
        output_dir=out_dir,
    )


__all__ = ["R14ExecutionArtifacts", "run_r14_execution_layer"]

