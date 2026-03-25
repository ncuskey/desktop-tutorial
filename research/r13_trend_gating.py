from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text

from data import CostModel, add_basic_indicators, load_ohlcv_csv, load_symbol_data
from execution.simulator import run_backtest
from metrics.performance import compute_metrics
from regime import attach_regime_labels
from strategies import trend_breakout_v2_signals


@dataclass
class R13Artifacts:
    trade_feature_dataset: pd.DataFrame
    trade_feature_bins: pd.DataFrame
    gate_model_scores: pd.DataFrame
    gate_comparison_by_fold: pd.DataFrame
    gate_coverage_metrics: pd.DataFrame
    gate_rule_export: dict[str, Any]
    output_dir: Path


FEATURE_COLUMNS = [
    "tsmom_24",
    "tsmom_48",
    "tsmom_72",
    "tsmom_avg",
    "vr_6",
    "vr_12",
    "vr_24",
    "ema_fast_12",
    "ema_slow_48",
    "ma_spread",
    "ma_slope",
    "adx_14",
    "adx_slope",
    "trend_variance_ratio",
    "range_ratio",
    "atr_percentile",
    "atr_ratio",
    "atr_expansion_recent",
    "breakout_strength_atr_mult",
    "breakout_velocity",
    "price_acceleration",
    "distance_from_range_high",
]


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
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, bool
    )


def _params_equal(left: dict[str, Any], right: dict[str, Any], tol: float = 1e-12) -> bool:
    if set(left.keys()) != set(right.keys()):
        return False
    for key in left:
        lv = _to_native(left[key])
        rv = _to_native(right[key])
        if _is_number(lv) and _is_number(rv):
            if abs(float(lv) - float(rv)) > tol:
                return False
        else:
            if lv != rv:
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
) -> Path:
    if explicit_source_csv is not None:
        p = Path(explicit_source_csv)
        if not p.exists():
            raise FileNotFoundError(f"source-csv not found: {p}")
        return p
    candidates = [
        artifacts_root / "r1_shared_ohlcv.csv",
        Path("/tmp/r122_artifacts/r1_shared_ohlcv.csv"),
        Path("outputs/TrendBreakout_V2/r1_shared_ohlcv.csv"),
        Path("outputs/strategy_research_mock_ohlcv.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not locate price data for R1.3 evaluation. Pass --source-csv explicitly."
    )


def _resolve_artifacts_root(path_like: str | Path) -> Path:
    explicit = Path(path_like)
    if explicit.exists():
        return explicit
    fallback = Path("/tmp/r122_artifacts")
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Artifacts root not found: {explicit}")


def _load_hardened_params(artifacts_root: Path, symbol: str) -> dict[str, Any]:
    rec_path = _resolve_symbol_file(artifacts_root, symbol, "strategy_research_recommendation.csv")
    rec = pd.read_csv(rec_path)
    if "candidate_type" not in rec.columns:
        raise ValueError(f"{rec_path} missing candidate_type column")
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
    if "params" not in folds.columns:
        raise ValueError(f"{fold_path} missing params column")
    parsed = folds["params"].apply(_parse_params)
    matched = folds[parsed.apply(lambda p: _params_equal(p, hardened_params))].copy()
    if matched.empty:
        raise ValueError(
            f"Could not match HARDENED_DEFAULT params to fold rows for {symbol} ({fold_path})"
        )
    matched["fold_start"] = pd.to_datetime(matched["fold_start"], utc=True, errors="coerce")
    matched["fold_train_end"] = pd.to_datetime(
        matched["fold_train_end"], utc=True, errors="coerce"
    )
    matched["fold_test_start"] = pd.to_datetime(
        matched["fold_test_start"], utc=True, errors="coerce"
    )
    matched["fold_test_end"] = pd.to_datetime(matched["fold_test_end"], utc=True, errors="coerce")
    matched = matched.dropna(
        subset=["fold_start", "fold_train_end", "fold_test_start", "fold_test_end"]
    )
    matched = matched.sort_values("fold_test_start").reset_index(drop=True)
    matched["fold_id"] = np.arange(len(matched), dtype=int)
    return matched


def _symbol_cost_model(symbol: str) -> CostModel:
    per_symbol = {
        "EURUSD": CostModel(spread_bps=0.7, slippage_bps=0.4, commission_bps=0.2),
        "GBPUSD": CostModel(spread_bps=0.9, slippage_bps=0.5, commission_bps=0.2),
        "AUDUSD": CostModel(spread_bps=0.8, slippage_bps=0.5, commission_bps=0.2),
    }
    return per_symbol.get(symbol, CostModel())


def _prepare_symbol_frame(raw_prices: pd.DataFrame, symbol: str, timeframe: str = "H1") -> pd.DataFrame:
    df = load_symbol_data(raw_prices, symbol=symbol, timeframe=timeframe).copy()
    df = add_basic_indicators(df)
    df = attach_regime_labels(df, adx_threshold=25.0)
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
        "trend_variance_ratio",
    ]
    out = df.dropna(subset=required).copy()
    out = out.sort_values("timestamp").reset_index(drop=True)
    return out


def _entry_mask_from_signal(signal: pd.Series) -> pd.Series:
    s = pd.to_numeric(signal, errors="coerce").fillna(0.0).astype(float)
    prev = s.shift(1).fillna(0.0)
    return ((s != 0.0) & (s != prev)).astype(bool)


def _apply_entry_gate_to_signal(
    raw_signal: pd.Series,
    entry_allow: pd.Series,
    min_trades_per_fold: int = 5,
) -> pd.Series:
    raw = pd.to_numeric(raw_signal, errors="coerce").fillna(0.0).astype(float)
    allow = entry_allow.fillna(False).astype(bool)
    out = pd.Series(0.0, index=raw.index, dtype=float)
    eps = 1e-12

    for i in range(len(raw)):
        raw_i = float(raw.iloc[i])
        raw_prev = float(raw.iloc[i - 1]) if i > 0 else 0.0
        out_prev = float(out.iloc[i - 1]) if i > 0 else 0.0
        raw_nonzero = abs(raw_i) > eps
        raw_prev_nonzero = abs(raw_prev) > eps
        out_prev_nonzero = abs(out_prev) > eps
        is_entry = (not raw_prev_nonzero and raw_nonzero) or (
            raw_prev_nonzero and raw_nonzero and (np.sign(raw_prev) != np.sign(raw_i))
        )

        if is_entry:
            out.iloc[i] = raw_i if bool(allow.iloc[i]) else 0.0
            continue
        if not raw_nonzero:
            out.iloc[i] = 0.0
            continue
        if out_prev_nonzero and np.sign(out_prev) == np.sign(raw_i):
            out.iloc[i] = raw_i
        else:
            out.iloc[i] = 0.0
    # Density stabilization: if no or too-few entries would remain, progressively
    # relax by adding earliest blocked entries until the fold reaches minimum trade count.
    gated_entries = _entry_mask_from_signal(out).sum()
    if gated_entries < max(1, int(min_trades_per_fold)):
        needed = max(1, int(min_trades_per_fold)) - int(gated_entries)
        if needed > 0:
            raw_entries = _entry_mask_from_signal(raw)
            blocked = raw_entries & (~_entry_mask_from_signal(out))
            blocked_idx = blocked[blocked].index.tolist()
            if blocked_idx:
                promote_idx = blocked_idx[:needed]
                allow_relaxed = entry_allow.copy()
                allow_relaxed.loc[promote_idx] = True
                out = _apply_entry_gate_to_signal(raw, allow_relaxed, min_trades_per_fold=0)
    return out


def _compute_bar_features(df: pd.DataFrame, strategy_params: dict[str, Any]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    atr = pd.to_numeric(df["atr_14"], errors="coerce")
    atr_norm = pd.to_numeric(df["atr_norm"], errors="coerce")
    atr_pct = pd.to_numeric(df["atr_norm_pct_rank"], errors="coerce")
    adx = pd.to_numeric(df["adx_14"], errors="coerce")
    trend_variance_ratio = pd.to_numeric(df["trend_variance_ratio"], errors="coerce")
    eps = 1e-12

    log_close = np.log(close.clip(lower=eps))
    log_ret = log_close.diff()

    for h in (24, 48, 72):
        ret_h = log_close - log_close.shift(h)
        vol_h = log_ret.rolling(h, min_periods=max(10, h // 3)).std() * np.sqrt(h)
        out[f"tsmom_{h}"] = ret_h / (vol_h + eps)
    out["tsmom_avg"] = out[["tsmom_24", "tsmom_48", "tsmom_72"]].mean(axis=1)

    for q in (6, 12, 24):
        agg = log_ret.rolling(q, min_periods=q).sum()
        vr_window = max(60, q * 10)
        var_agg = agg.rolling(vr_window, min_periods=max(20, vr_window // 3)).var(ddof=0)
        var_single = log_ret.rolling(vr_window, min_periods=max(20, vr_window // 3)).var(ddof=0)
        out[f"vr_{q}"] = var_agg / (q * var_single.replace(0.0, np.nan))

    ema_fast = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_slow = close.ewm(span=48, adjust=False, min_periods=48).mean()
    out["ema_fast_12"] = ema_fast
    out["ema_slow_48"] = ema_slow
    out["ma_spread"] = (ema_fast - ema_slow) / atr.replace(0.0, np.nan)
    out["ma_slope"] = (ema_slow - ema_slow.shift(12)) / atr.replace(0.0, np.nan)

    out["adx_14"] = adx
    out["adx_slope"] = adx - adx.shift(5)
    out["trend_variance_ratio"] = trend_variance_ratio

    range_short = (
        high.rolling(12, min_periods=12).max() - low.rolling(12, min_periods=12).min()
    ) / close.clip(lower=eps)
    range_long = (
        high.rolling(48, min_periods=48).max() - low.rolling(48, min_periods=48).min()
    ) / close.clip(lower=eps)
    out["range_ratio"] = range_short / range_long.replace(0.0, np.nan)

    out["atr_percentile"] = atr_pct
    atr_roll_mean = atr_norm.rolling(20, min_periods=8).mean()
    out["atr_ratio"] = atr_norm / atr_roll_mean.replace(0.0, np.nan)
    out["atr_expansion_recent"] = out["atr_ratio"].rolling(12, min_periods=4).max()

    lookback = max(2, int(strategy_params.get("lookback", 20)))
    velocity_lookback = max(1, int(strategy_params.get("velocity_lookback", 6)))
    range_high = high.rolling(lookback, min_periods=lookback).max().shift(1)
    range_low = low.rolling(lookback, min_periods=lookback).min().shift(1)
    out["breakout_velocity"] = (close - close.shift(velocity_lookback)).abs() / atr.replace(
        0.0, np.nan
    )
    upper_strength = (close - range_high) / atr.replace(0.0, np.nan)
    lower_strength = (range_low - close) / atr.replace(0.0, np.nan)
    out["breakout_strength_atr_mult"] = pd.concat([upper_strength, lower_strength], axis=1).max(
        axis=1
    )
    out["price_acceleration"] = close.diff().diff() / atr.replace(0.0, np.nan)
    out["distance_from_range_high"] = (range_high - close) / atr.replace(0.0, np.nan)

    return out


def _compute_trade_excursions(
    segment_df: pd.DataFrame,
    entry_idx: int,
    exit_idx: int,
    side: int,
) -> tuple[float, float]:
    if entry_idx not in segment_df.index or exit_idx not in segment_df.index:
        return np.nan, np.nan
    entry_px = float(segment_df.loc[entry_idx, "close"])
    if not np.isfinite(entry_px) or entry_px == 0:
        return np.nan, np.nan
    path = segment_df.loc[entry_idx:exit_idx]
    if path.empty:
        return np.nan, np.nan
    high = pd.to_numeric(path["high"], errors="coerce")
    low = pd.to_numeric(path["low"], errors="coerce")

    if side >= 0:
        mfe = float((high.max() / entry_px) - 1.0) if len(high) > 0 else np.nan
        mae = float((low.min() / entry_px) - 1.0) if len(low) > 0 else np.nan
        return mfe, mae
    # Short trade: favorable move is down, adverse is up.
    mfe = float((entry_px / low.min()) - 1.0) if len(low) > 0 else np.nan
    mae = float((entry_px / high.max()) - 1.0) if len(high) > 0 else np.nan
    return mfe, mae


def _build_trade_feature_rows(
    segment_df: pd.DataFrame,
    bar_features: pd.DataFrame,
    signal: pd.Series,
    symbol: str,
    fold_id: int,
    split: str,
    cost_model: CostModel,
) -> list[dict[str, Any]]:
    bt = run_backtest(segment_df, signal, cost_model=cost_model)
    if bt.trades.empty:
        return []

    rows: list[dict[str, Any]] = []
    for _, trade in bt.trades.iterrows():
        entry_idx = int(trade["entry_time"])
        exit_idx = int(trade["exit_time"])
        if entry_idx not in segment_df.index:
            continue
        entry_time = pd.to_datetime(segment_df.loc[entry_idx, "timestamp"], utc=True, errors="coerce")
        if pd.isna(entry_time):
            continue

        feat_row = bar_features.loc[entry_idx] if entry_idx in bar_features.index else pd.Series()
        row: dict[str, Any] = {
            "symbol": symbol,
            "fold_id": int(fold_id),
            "split": split,
            "entry_time": entry_time,
            "exit_time": pd.to_datetime(
                segment_df.loc[exit_idx, "timestamp"], utc=True, errors="coerce"
            )
            if exit_idx in segment_df.index
            else pd.NaT,
            "side": int(trade.get("side", 0)),
            "return": _safe_float(trade.get("trade_return"), default=np.nan),
            "win": int(_safe_float(trade.get("trade_return"), default=0.0) > 0.0),
            "holding_bars": _safe_float(trade.get("holding_bars"), default=np.nan),
        }
        mfe, mae = _compute_trade_excursions(
            segment_df=segment_df,
            entry_idx=entry_idx,
            exit_idx=exit_idx,
            side=int(trade.get("side", 0)),
        )
        row["mfe"] = mfe
        row["mae"] = mae

        for col in FEATURE_COLUMNS:
            row[col] = _safe_float(feat_row.get(col), default=np.nan)
        rows.append(row)
    return rows


def _prepare_model_inputs(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    x = df[feature_cols].copy()
    medians = x.median(axis=0, numeric_only=True)
    x = x.fillna(medians)
    return x, medians


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _fit_logistic_and_tree(
    train_trade_df: pd.DataFrame,
    feature_cols: list[str],
    score_quantile: float,
) -> dict[str, Any]:
    x_train_raw, medians = _prepare_model_inputs(train_trade_df, feature_cols)
    y_train = train_trade_df["win"].astype(int)
    if len(train_trade_df) < 12 or y_train.nunique() < 2:
        # Deterministic fallback rule from train-only quantiles (no outcome fit possible).
        fallback_rule = {
            "conditions": [
                {"feature": "tsmom_avg", "op": ">", "threshold": _safe_float(train_trade_df["tsmom_avg"].median(), 0.0)},
                {"feature": "vr_12", "op": ">", "threshold": _safe_float(train_trade_df["vr_12"].median(), 1.0)},
                {"feature": "ma_spread", "op": ">", "threshold": _safe_float(train_trade_df["ma_spread"].median(), 0.0)},
            ],
            "min_hits": 2,
        }
        return {
            "fitted": False,
            "reason": "insufficient_train_samples_or_single_class",
            "medians": medians.to_dict(),
            "threshold": -np.inf,
            "feature_cols": list(feature_cols),
            "guard_rule": fallback_rule,
        }

    x_train = x_train_raw.to_numpy(dtype=float)
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std <= 1e-12, 1.0, std)
    x_scaled = (x_train - mean) / std

    logistic = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
        class_weight="balanced",
    )
    logistic.fit(x_scaled, y_train.to_numpy())
    train_score = logistic.predict_proba(x_scaled)[:, 1]
    score_threshold = float(np.nanquantile(train_score, score_quantile))

    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=42)
    tree.fit(x_train, y_train.to_numpy())
    tree_text = export_text(tree, feature_names=feature_cols)

    coef_series = pd.Series(logistic.coef_[0], index=feature_cols)
    top_features = coef_series.abs().sort_values(ascending=False).head(4).index.tolist()
    conditions: list[dict[str, Any]] = []
    for feat in top_features:
        sign = float(np.sign(coef_series.loc[feat]))
        cond_threshold = _safe_float(train_trade_df[feat].median(), default=0.0)
        op = ">" if sign >= 0 else "<"
        conditions.append({"feature": feat, "op": op, "threshold": cond_threshold})
    guard_rule = {"conditions": conditions, "min_hits": 2}

    return {
        "fitted": True,
        "feature_cols": list(feature_cols),
        "medians": medians.to_dict(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "coef": logistic.coef_[0].tolist(),
        "intercept": float(logistic.intercept_[0]),
        "threshold": score_threshold,
        "tree_text": tree_text,
        "guard_rule": guard_rule,
    }


def _score_with_model(model: dict[str, Any], feature_df: pd.DataFrame) -> pd.Series:
    if feature_df.empty:
        return pd.Series(dtype=float)
    if not model.get("fitted", False):
        return pd.Series(np.nan, index=feature_df.index, dtype=float)

    feature_cols = model["feature_cols"]
    medians = pd.Series(model["medians"])
    x = feature_df[feature_cols].copy().fillna(medians)
    mean = np.asarray(model["mean"], dtype=float)
    std = np.asarray(model["std"], dtype=float)
    coef = np.asarray(model["coef"], dtype=float)
    intercept = float(model["intercept"])
    x_scaled = (x.to_numpy(dtype=float) - mean) / std
    logits = np.clip(x_scaled @ coef + intercept, -20.0, 20.0)
    score = _sigmoid(logits)
    return pd.Series(score, index=feature_df.index, dtype=float)


def _apply_guard_rule(model: dict[str, Any], feature_df: pd.DataFrame) -> pd.Series:
    if feature_df.empty:
        return pd.Series(dtype=bool)
    rule = model.get("guard_rule") or {}
    conditions = rule.get("conditions") or []
    min_hits = int(rule.get("min_hits", 1))
    if not conditions:
        return pd.Series(True, index=feature_df.index, dtype=bool)

    hits = pd.Series(0, index=feature_df.index, dtype=int)
    for cond in conditions:
        feat = str(cond.get("feature", ""))
        if feat not in feature_df.columns:
            continue
        val = pd.to_numeric(feature_df[feat], errors="coerce")
        thr = float(cond.get("threshold", 0.0))
        op = str(cond.get("op", ">"))
        passed = (val > thr) if op == ">" else (val < thr)
        hits = hits + passed.fillna(False).astype(int)
    return (hits >= min_hits).fillna(False)


def _percentile_threshold(scores: pd.Series, quantile: float) -> float:
    s = pd.to_numeric(scores, errors="coerce").dropna()
    if s.empty:
        return np.nan
    return float(np.nanquantile(s.to_numpy(dtype=float), quantile))


def _select_entries_by_rank(
    scores: pd.Series,
    feature_df: pd.DataFrame,
    *,
    min_trades_per_fold: int,
    top_k_percent: float,
    top_n_count: int,
    target_pass_rate_min: float,
    target_pass_rate_max: float,
    hybrid_tsmom_enabled: bool,
    train_tsmom48_median: float,
) -> tuple[pd.Series, dict[str, Any]]:
    idx = scores.index
    allow = pd.Series(False, index=idx, dtype=bool)
    n = int(len(scores))
    if n <= 0:
        return allow, {
            "selected_count": 0,
            "effective_trade_coverage": 0.0,
            "selection_expansion_events": 0,
            "fallback_to_baseline": False,
            "fold_trade_viability": False,
        }

    s = pd.to_numeric(scores, errors="coerce").fillna(-np.inf)
    ranked_idx = s.sort_values(ascending=False, kind="stable").index.tolist()

    min_required = min(max(1, int(min_trades_per_fold)), n)
    k_pct = max(1, int(np.ceil(n * float(top_k_percent))))
    k_topn = min(max(1, int(top_n_count)), n)
    k_target_min = max(1, int(np.ceil(n * float(target_pass_rate_min))))
    k_target_max = max(1, int(np.floor(n * float(target_pass_rate_max))))
    k_base = max(k_pct, k_topn)
    k = min(max(k_base, 1), n)

    expansion_events = 0
    selected = ranked_idx[:k]
    hybrid_relaxed = False

    if hybrid_tsmom_enabled and "tsmom_48" in feature_df.columns and np.isfinite(train_tsmom48_median):
        cond = pd.to_numeric(feature_df["tsmom_48"], errors="coerce") > float(train_tsmom48_median)
        selected = [i for i in selected if bool(cond.reindex([i]).fillna(False).iloc[0])]
        while len(selected) < min_required and k < n:
            k += 1
            expansion_events += 1
            candidate = ranked_idx[:k]
            selected = [i for i in candidate if bool(cond.reindex([i]).fillna(False).iloc[0])]
        if len(selected) < min_required:
            # Relax hybrid constraint to satisfy minimum fold density.
            selected = ranked_idx[:max(min_required, k_target_min, k)]
            hybrid_relaxed = True
            expansion_events += 1
    else:
        while len(selected) < min_required and k < n:
            k += 1
            expansion_events += 1
            selected = ranked_idx[:k]

    # Enforce target minimum pass-rate (deterministic expansion).
    min_target_count = min(max(1, k_target_min), n)
    while len(selected) < min_target_count and k < n:
        k += 1
        expansion_events += 1
        if hybrid_tsmom_enabled and "tsmom_48" in feature_df.columns and np.isfinite(train_tsmom48_median):
            cond = pd.to_numeric(feature_df["tsmom_48"], errors="coerce") > float(train_tsmom48_median)
            selected = [i for i in ranked_idx[:k] if bool(cond.reindex([i]).fillna(False).iloc[0])]
        else:
            selected = ranked_idx[:k]

    # Respect target maximum pass-rate when feasible and still viable.
    max_target_count = min(max(1, k_target_max), n)
    if len(selected) > max_target_count and max_target_count >= min_required:
        selected = ranked_idx[:max_target_count]

    if len(selected) <= 0:
        # Mandatory safety fallback: preserve baseline behavior for the fold.
        allow[:] = True
        selected_count = n
        fallback = True
    else:
        allow.loc[selected] = True
        selected_count = int(len(selected))
        fallback = False

    return allow, {
        "selected_count": selected_count,
        "effective_trade_coverage": float(selected_count / max(n, 1)),
        "selection_expansion_events": int(expansion_events),
        "fallback_to_baseline": bool(fallback),
        "fold_trade_viability": bool(selected_count >= min_required),
        "rank_top_k_percent": float(top_k_percent),
        "rank_top_n_count": int(k_topn),
        "rank_initial_k": int(k_base),
        "rank_final_k": int(k),
        "rank_min_required": int(min_required),
        "hybrid_relaxed": bool(hybrid_relaxed),
    }


def _summarize_feature_bins(
    trade_df: pd.DataFrame,
    output_charts_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in FEATURE_COLUMNS:
        series = pd.to_numeric(trade_df[feature], errors="coerce")
        valid = trade_df[series.notna()].copy()
        if valid.empty or valid[feature].nunique() < 6:
            continue
        try:
            valid["bin"] = pd.qcut(valid[feature], q=10, labels=False, duplicates="drop")
        except Exception:
            continue
        grouped = valid.groupby("bin", dropna=True)
        stats = grouped["return"].agg(["count", "mean", "median"]).reset_index()
        win_rate = grouped["win"].mean().reset_index(name="win_rate")
        merged = stats.merge(win_rate, on="bin", how="left").sort_values("bin")
        for _, r in merged.iterrows():
            rows.append(
                {
                    "feature": feature,
                    "bin": int(r["bin"]),
                    "count": int(r["count"]),
                    "mean_return": float(r["mean"]),
                    "median_return": float(r["median"]),
                    "win_rate": float(r["win_rate"]),
                }
            )

        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1.bar(merged["bin"].astype(str), merged["mean"], color="#4C78A8", alpha=0.75)
        ax1.set_ylabel("Mean Return")
        ax1.set_xlabel("Decile Bin")
        ax1.set_title(f"{feature} decile outcomes")
        ax2 = ax1.twinx()
        ax2.plot(merged["bin"].astype(str), merged["win_rate"], color="#F58518", marker="o")
        ax2.set_ylabel("Win Rate")
        fig.tight_layout()
        fig.savefig(output_charts_dir / f"feature_bins_{feature}.png", dpi=140)
        plt.close(fig)
    return pd.DataFrame(rows)


def _plot_interaction_heatmap(
    trade_df: pd.DataFrame,
    feature_a: str,
    feature_b: str,
    output_path: Path,
) -> None:
    local = trade_df[[feature_a, feature_b, "return"]].copy()
    local = local.replace([np.inf, -np.inf], np.nan).dropna()
    if len(local) < 30:
        return
    try:
        local["a_bin"] = pd.qcut(local[feature_a], q=6, labels=False, duplicates="drop")
        local["b_bin"] = pd.qcut(local[feature_b], q=6, labels=False, duplicates="drop")
    except Exception:
        return
    heat = (
        local.groupby(["a_bin", "b_bin"])["return"]
        .mean()
        .unstack(fill_value=np.nan)
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    if heat.empty:
        return

    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    im = ax.imshow(heat.to_numpy(), aspect="auto", origin="lower", cmap="RdYlGn")
    ax.set_title(f"{feature_a} x {feature_b} (mean return)")
    ax.set_xlabel(feature_b)
    ax.set_ylabel(feature_a)
    ax.set_xticks(np.arange(heat.shape[1]))
    ax.set_yticks(np.arange(heat.shape[0]))
    ax.set_xticklabels([str(int(x)) for x in heat.columns], rotation=0)
    ax.set_yticklabels([str(int(x)) for x in heat.index], rotation=0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_score_decile_uplift(gate_model_scores: pd.DataFrame, output_path: Path) -> None:
    local = gate_model_scores[
        (gate_model_scores["split"] == "test") & (gate_model_scores["model"] == "logistic_primary")
    ].copy()
    if local.empty:
        return
    local["predicted_score"] = pd.to_numeric(local["predicted_score"], errors="coerce")
    local = local.dropna(subset=["predicted_score", "actual_return", "actual_win"])
    if local.empty:
        return
    try:
        local["score_decile"] = pd.qcut(
            local["predicted_score"], q=10, labels=False, duplicates="drop"
        )
    except Exception:
        return
    grouped = local.groupby("score_decile")
    uplift = grouped["actual_return"].mean().reset_index(name="avg_return")
    win = grouped["actual_win"].mean().reset_index(name="win_rate")
    merged = uplift.merge(win, on="score_decile", how="left")

    fig, ax1 = plt.subplots(figsize=(8, 4.2))
    ax1.bar(merged["score_decile"].astype(str), merged["avg_return"], color="#4C78A8", alpha=0.8)
    ax1.set_ylabel("Average Return")
    ax1.set_xlabel("Predicted Score Decile")
    ax1.set_title("Predicted score decile uplift (test entries)")
    ax2 = ax1.twinx()
    ax2.plot(merged["score_decile"].astype(str), merged["win_rate"], color="#F58518", marker="o")
    ax2.set_ylabel("Win Rate")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_trade_rank_vs_return(gate_model_scores: pd.DataFrame, output_path: Path) -> None:
    local = gate_model_scores[
        (gate_model_scores["split"] == "test") & (gate_model_scores["model"] == "logistic_primary")
    ].copy()
    if local.empty:
        return
    local["predicted_score"] = pd.to_numeric(local["predicted_score"], errors="coerce")
    local["actual_return"] = pd.to_numeric(local["actual_return"], errors="coerce")
    local = local.dropna(subset=["predicted_score", "actual_return"])
    if len(local) < 20:
        return

    local["score_rank_pct"] = local["predicted_score"].rank(pct=True, method="average") * 100.0
    # 20 bins of 5 percentile points each.
    local["rank_bin"] = (np.floor(local["score_rank_pct"] / 5.0) * 5.0).clip(0.0, 95.0)
    grouped = local.groupby("rank_bin", dropna=True)["actual_return"].mean().reset_index()
    if grouped.empty:
        return

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.plot(grouped["rank_bin"], grouped["actual_return"], marker="o", color="#4C78A8")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_title("Trade rank percentile vs average return")
    ax.set_xlabel("Score percentile bin")
    ax.set_ylabel("Average return")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_fold_uplift_boxplot(gate_comparison: pd.DataFrame, output_path: Path) -> None:
    cols = ["delta_sharpe", "delta_expectancy", "delta_max_dd", "delta_trade_count"]
    local = gate_comparison[cols].replace([np.inf, -np.inf], np.nan).dropna(how="all")
    if local.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.boxplot([local[c].dropna() for c in cols], labels=cols, patch_artist=True)
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_title("Fold-level uplift distribution (gated - baseline)")
    ax.set_ylabel("Delta")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _compute_ks_drift(
    trade_feature_df: pd.DataFrame,
    output_chart_path: Path,
    output_csv_path: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in FEATURE_COLUMNS:
        feature_vals = []
        for (symbol, fold_id), g in trade_feature_df.groupby(["symbol", "fold_id"], dropna=False):
            train_vals = pd.to_numeric(
                g.loc[g["split"] == "train", feature],
                errors="coerce",
            ).dropna()
            test_vals = pd.to_numeric(
                g.loc[g["split"] == "test", feature],
                errors="coerce",
            ).dropna()
            if len(train_vals) < 5 or len(test_vals) < 5:
                continue
            stat, p = ks_2samp(train_vals.to_numpy(), test_vals.to_numpy(), alternative="two-sided")
            feature_vals.append((float(stat), float(p), str(symbol), int(fold_id)))

        if not feature_vals:
            continue
        ks_stats = [x[0] for x in feature_vals]
        p_vals = [x[1] for x in feature_vals]
        rows.append(
            {
                "feature": feature,
                "ks_stat_mean": float(np.mean(ks_stats)),
                "ks_stat_median": float(np.median(ks_stats)),
                "p_value_median": float(np.median(p_vals)),
                "fold_pairs": int(len(feature_vals)),
            }
        )
    drift = pd.DataFrame(rows).sort_values("ks_stat_mean", ascending=False).reset_index(drop=True)
    drift.to_csv(output_csv_path, index=False)
    if drift.empty:
        return drift

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(drift["feature"], drift["ks_stat_mean"], color="#54A24B")
    ax.set_title("Feature drift KS statistic (train vs test entry distributions)")
    ax.set_ylabel("Mean KS statistic")
    ax.set_xlabel("Feature")
    ax.tick_params(axis="x", rotation=60, labelsize=8)
    fig.tight_layout()
    fig.savefig(output_chart_path, dpi=150)
    plt.close(fig)
    return drift


def _bootstrap_delta_stats(values: np.ndarray, n_bootstrap: int = 4000) -> tuple[float, float]:
    if values.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=values.size, replace=True)
        boots.append(float(np.median(sample)))
    return float(np.quantile(boots, 0.05)), float(np.quantile(boots, 0.95))


def _run_stat_tests(gate_comparison: pd.DataFrame, output_csv: Path, spa_stub_path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    tests = {
        "delta_sharpe": gate_comparison["delta_sharpe"].to_numpy(dtype=float),
        "delta_expectancy": gate_comparison["delta_expectancy"].to_numpy(dtype=float),
        "delta_max_dd": gate_comparison["delta_max_dd"].to_numpy(dtype=float),
        "delta_trade_count": gate_comparison["delta_trade_count"].to_numpy(dtype=float),
    }
    for metric, values in tests.items():
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        try:
            # Wilcoxon signed-rank: paired fold deltas against zero.
            w = wilcoxon(finite, zero_method="wilcox", alternative="two-sided")
            w_stat = float(w.statistic)
            w_p = float(w.pvalue)
        except Exception:
            w_stat = np.nan
            w_p = np.nan
        ci_low, ci_high = _bootstrap_delta_stats(finite)
        rows.append(
            {
                "metric": metric,
                "n_folds": int(finite.size),
                "median_delta": float(np.median(finite)),
                "mean_delta": float(np.mean(finite)),
                "wilcoxon_stat": w_stat,
                "wilcoxon_p_value": w_p,
                "bootstrap_median_ci_05": ci_low,
                "bootstrap_median_ci_95": ci_high,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(output_csv, index=False)

    spa_stub = {
        "status": "stub",
        "description": "Placeholder schema for White Reality Check / SPA integration.",
        "candidate_models": ["baseline_no_gate", "r13_logistic_gate_q70"],
        "required_inputs": [
            "fold_paired_returns_baseline",
            "fold_paired_returns_gated",
            "loss_differential_series",
            "block_bootstrap_params",
        ],
        "next_step": "Implement block-bootstrap SPA p-values over fold/segment loss differentials.",
    }
    spa_stub_path.write_text(json.dumps(spa_stub, indent=2), encoding="utf-8")
    return out


def _render_rule_export_markdown(rule_export: dict[str, Any]) -> str:
    lines = [
        "# Gate Rule Export (R1.3)",
        "",
        f"- Strategy: `{rule_export.get('strategy')}`",
        f"- Gate mode: `{rule_export.get('gate_mode')}`",
        f"- Deterministic decision: `{rule_export.get('deterministic_rule')}`",
        "",
        "## Threshold Stability",
        f"- Quantile target: `{rule_export.get('score_quantile')}`",
        f"- Threshold mean: `{rule_export.get('threshold_summary', {}).get('mean')}`",
        f"- Threshold std: `{rule_export.get('threshold_summary', {}).get('std')}`",
        f"- Threshold min/max: `{rule_export.get('threshold_summary', {}).get('min')} / {rule_export.get('threshold_summary', {}).get('max')}`",
        "",
        "## Coefficient Stability (median coefficient, sign consistency)",
        "",
        "| feature | median_coef | sign_consistency |",
        "|---|---:|---:|",
    ]
    for row in rule_export.get("coefficient_stability", []):
        lines.append(
            f"| {row['feature']} | {row['median_coef']:.6f} | {row['sign_consistency']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Tree Rule Snapshots",
            "",
        ]
    )
    for i, txt in enumerate(rule_export.get("tree_rule_snapshots", [])[:5], start=1):
        lines.append(f"### Fold Tree {i}")
        lines.append("```")
        lines.append(str(txt))
        lines.append("```")
    return "\n".join(lines) + "\n"


def _export_gate_rules(
    fold_models: list[dict[str, Any]],
    strategy: str,
    score_quantile: float,
    top_k_percent: float,
    min_trades_per_fold: int,
    target_pass_rate_min: float,
    target_pass_rate_max: float,
    hybrid_tsmom_enabled: bool,
    output_json: Path,
    output_md: Path,
) -> dict[str, Any]:
    fitted = [m for m in fold_models if m.get("fitted")]
    thresholds = [float(m["threshold"]) for m in fitted if np.isfinite(m.get("threshold", np.nan))]
    coef_rows: list[dict[str, Any]] = []
    if fitted:
        coef_df = pd.DataFrame(
            [dict(zip(m["feature_cols"], m["coef"], strict=False)) for m in fitted]
        )
        for col in coef_df.columns:
            vals = coef_df[col].dropna().to_numpy(dtype=float)
            if vals.size == 0:
                continue
            signs = np.sign(vals)
            sign_consistency = float(max((signs > 0).mean(), (signs < 0).mean()))
            coef_rows.append(
                {
                    "feature": col,
                    "median_coef": float(np.median(vals)),
                    "sign_consistency": sign_consistency,
                }
            )
    coef_rows = sorted(coef_rows, key=lambda x: abs(x["median_coef"]), reverse=True)

    rule_export = {
        "strategy": strategy,
        "gate_mode": "rank_based_selection",
        "deterministic_rule": (
            "ALLOW entry if selected by fold-local rank gate: "
            "top-K-percent union top-N minimum, with deterministic expansion to satisfy minimum density"
        ),
        "rank_selection_config": {
            "top_k_percent": float(top_k_percent),
            "min_trades_per_fold": int(min_trades_per_fold),
            "target_pass_rate_min": float(target_pass_rate_min),
            "target_pass_rate_max": float(target_pass_rate_max),
            "hybrid_tsmom_enabled": bool(hybrid_tsmom_enabled),
        },
        "score_quantile": score_quantile,
        "threshold_summary": {
            "count": int(len(thresholds)),
            "mean": _safe_float(np.mean(thresholds), default=np.nan) if thresholds else np.nan,
            "std": _safe_float(np.std(thresholds), default=np.nan) if thresholds else np.nan,
            "min": _safe_float(np.min(thresholds), default=np.nan) if thresholds else np.nan,
            "max": _safe_float(np.max(thresholds), default=np.nan) if thresholds else np.nan,
        },
        "thresholds_by_fold": thresholds,
        "coefficient_stability": coef_rows,
        "tree_rule_snapshots": [m.get("tree_text", "") for m in fitted if m.get("tree_text")],
    }
    output_json.write_text(json.dumps(rule_export, indent=2), encoding="utf-8")
    output_md.write_text(_render_rule_export_markdown(rule_export), encoding="utf-8")
    return rule_export


def run_r13_trend_gating(
    strategy: str = "TrendBreakout_V2",
    symbols: list[str] | None = None,
    artifacts_root: str | Path = "outputs/TrendBreakout_V2",
    output_dir: str | Path = "outputs",
    timeframe: str = "H1",
    source_csv: str | Path | None = None,
    score_quantile: float = 0.70,
    top_k_percent: float = 0.30,
    min_trades_per_fold: int = 5,
    target_pass_rate_min: float = 0.25,
    target_pass_rate_max: float = 0.40,
    hybrid_tsmom_enabled: bool = False,
) -> R13Artifacts:
    if strategy != "TrendBreakout_V2":
        raise ValueError("R1.3 currently supports strategy='TrendBreakout_V2' only.")
    target_symbols = symbols or ["EURUSD", "GBPUSD", "AUDUSD"]
    artifacts_root_path = _resolve_artifacts_root(artifacts_root)
    out_dir = Path(output_dir)
    charts_dir = out_dir / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    price_path = _resolve_price_data_path(artifacts_root_path, source_csv)
    raw_prices = load_ohlcv_csv(price_path)

    trade_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    fold_models: list[dict[str, Any]] = []

    for symbol in target_symbols:
        cost_model = _symbol_cost_model(symbol)
        hardened_params = _load_hardened_params(artifacts_root_path, symbol)
        fold_windows = _load_hardened_folds(artifacts_root_path, symbol, hardened_params)
        frame = _prepare_symbol_frame(raw_prices=raw_prices, symbol=symbol, timeframe=timeframe)
        if frame.empty:
            continue
        bar_features = _compute_bar_features(frame, hardened_params)

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
            train_features = bar_features.loc[train_df.index].copy()
            test_features = bar_features.loc[test_df.index].copy()

            train_signal = trend_breakout_v2_signals(train_df, hardened_params).astype(float)
            test_signal = trend_breakout_v2_signals(test_df, hardened_params).astype(float)

            train_trade_rows = _build_trade_feature_rows(
                segment_df=train_df,
                bar_features=train_features,
                signal=train_signal,
                symbol=symbol,
                fold_id=fold_id,
                split="train",
                cost_model=cost_model,
            )
            test_trade_rows = _build_trade_feature_rows(
                segment_df=test_df,
                bar_features=test_features,
                signal=test_signal,
                symbol=symbol,
                fold_id=fold_id,
                split="test",
                cost_model=cost_model,
            )
            trade_rows.extend(train_trade_rows)
            trade_rows.extend(test_trade_rows)

            train_trade_df = pd.DataFrame(train_trade_rows)
            test_trade_df = pd.DataFrame(test_trade_rows)

            model = _fit_logistic_and_tree(
                train_trade_df=train_trade_df if not train_trade_df.empty else pd.DataFrame(columns=FEATURE_COLUMNS + ["win"]),
                feature_cols=FEATURE_COLUMNS,
                score_quantile=score_quantile,
            )
            model["symbol"] = symbol
            model["fold_id"] = fold_id
            fold_models.append(model)

            test_entry_mask = _entry_mask_from_signal(test_signal)
            test_entry_idx = test_signal.index[test_entry_mask]

            if not test_trade_df.empty and model.get("fitted", False):
                train_scores = _score_with_model(model, train_trade_df[FEATURE_COLUMNS])
                for i, row in train_trade_df.iterrows():
                    score_rows.append(
                        {
                            "symbol": symbol,
                            "fold_id": fold_id,
                            "split": "train",
                            "model": "logistic_primary",
                            "entry_time": row["entry_time"],
                            "predicted_score": _safe_float(train_scores.iloc[i], default=np.nan),
                            "actual_return": _safe_float(row["return"], default=np.nan),
                            "actual_win": int(row["win"]),
                            "allow_trade": 0,
                            "threshold": _safe_float(model["threshold"], default=np.nan),
                        }
                    )

            if test_entry_idx.size > 0:
                test_entry_features = test_features.loc[test_entry_idx, FEATURE_COLUMNS]
                train_tsmom_median = _safe_float(
                    pd.to_numeric(train_trade_df.get("tsmom_48"), errors="coerce").median(),
                    default=np.nan,
                )
                if model.get("fitted", False):
                    test_entry_scores = _score_with_model(model, test_entry_features)
                    train_entry_scores = (
                        _score_with_model(model, train_trade_df[FEATURE_COLUMNS])
                        if not train_trade_df.empty
                        else pd.Series(dtype=float)
                    )
                else:
                    # Non-fitted fold: deterministic fallback rank by a trend proxy.
                    test_entry_scores = pd.to_numeric(
                        test_entry_features.get("tsmom_avg"), errors="coerce"
                    ).fillna(0.0)
                    train_entry_scores = pd.to_numeric(
                        train_trade_df.get("tsmom_avg"), errors="coerce"
                    ).fillna(0.0)

                allow_entries, selection_diag = _select_entries_by_rank(
                    scores=test_entry_scores,
                    feature_df=test_entry_features,
                    min_trades_per_fold=min_trades_per_fold,
                    top_k_percent=top_k_percent,
                    top_n_count=min_trades_per_fold,
                    target_pass_rate_min=target_pass_rate_min,
                    target_pass_rate_max=target_pass_rate_max,
                    hybrid_tsmom_enabled=hybrid_tsmom_enabled,
                    train_tsmom48_median=train_tsmom_median,
                )
                train_allow, _ = _select_entries_by_rank(
                    scores=train_entry_scores,
                    feature_df=train_trade_df[FEATURE_COLUMNS] if not train_trade_df.empty else pd.DataFrame(columns=FEATURE_COLUMNS),
                    min_trades_per_fold=min_trades_per_fold,
                    top_k_percent=top_k_percent,
                    top_n_count=min_trades_per_fold,
                    target_pass_rate_min=target_pass_rate_min,
                    target_pass_rate_max=target_pass_rate_max,
                    hybrid_tsmom_enabled=hybrid_tsmom_enabled,
                    train_tsmom48_median=train_tsmom_median,
                )
                train_threshold = _percentile_threshold(train_entry_scores, score_quantile)
                test_threshold = _percentile_threshold(test_entry_scores, score_quantile)
                threshold_drift = (
                    float(test_threshold - train_threshold)
                    if np.isfinite(train_threshold) and np.isfinite(test_threshold)
                    else np.nan
                )

                if bool(selection_diag.get("fallback_to_baseline", False)):
                    allow_entries = pd.Series(True, index=test_entry_idx, dtype=bool)
                    selection_diag["selected_count"] = int(len(test_entry_idx))
                    selection_diag["effective_trade_coverage"] = 1.0
            else:
                test_entry_scores = pd.Series(dtype=float)
                allow_entries = pd.Series(dtype=bool)
                train_allow = pd.Series(dtype=bool)
                selection_diag = {
                    "selected_count": 0,
                    "effective_trade_coverage": 0.0,
                    "selection_expansion_events": 0,
                    "fallback_to_baseline": False,
                    "fold_trade_viability": False,
                }
                train_threshold = np.nan
                test_threshold = np.nan
                threshold_drift = np.nan

            if not train_trade_df.empty:
                train_trade_df = train_trade_df.reset_index(drop=True)
                train_scores_for_rows = (
                    _score_with_model(model, train_trade_df[FEATURE_COLUMNS])
                    if model.get("fitted", False)
                    else pd.to_numeric(train_trade_df.get("tsmom_avg"), errors="coerce").fillna(0.0)
                )
                for i, row in train_trade_df.iterrows():
                    allow_train = int(bool(train_allow.iloc[i])) if i < len(train_allow) else 0
                    score_rows.append(
                        {
                            "symbol": symbol,
                            "fold_id": fold_id,
                            "split": "train",
                            "model": "logistic_primary",
                            "entry_time": row["entry_time"],
                            "predicted_score": _safe_float(train_scores_for_rows.iloc[i], default=np.nan),
                            "actual_return": _safe_float(row["return"], default=np.nan),
                            "actual_win": int(row["win"]),
                            "allow_trade": allow_train,
                            "threshold": _safe_float(train_threshold, default=np.nan),
                        }
                    )

            for _, row in test_trade_df.iterrows():
                entry_time = pd.to_datetime(row["entry_time"], utc=True, errors="coerce")
                idx_match = test_df.index[test_df["timestamp"] == entry_time]
                score_val = np.nan
                allow_val = 1
                if len(idx_match) > 0:
                    idx0 = idx_match[0]
                    if idx0 in test_entry_scores.index:
                        score_val = _safe_float(test_entry_scores.loc[idx0], default=np.nan)
                    if idx0 in allow_entries.index:
                        allow_val = int(bool(allow_entries.loc[idx0]))
                score_rows.append(
                    {
                        "symbol": symbol,
                        "fold_id": fold_id,
                        "split": "test",
                        "model": "logistic_primary",
                        "entry_time": entry_time,
                        "predicted_score": score_val,
                        "actual_return": _safe_float(row["return"], default=np.nan),
                        "actual_win": int(row["win"]),
                        "allow_trade": allow_val,
                        "threshold": _safe_float(test_threshold, default=np.nan),
                    }
                )

            entry_allow_mask = pd.Series(False, index=test_signal.index, dtype=bool)
            if len(allow_entries) > 0:
                entry_allow_mask.loc[allow_entries.index] = allow_entries
            gated_signal = _apply_entry_gate_to_signal(test_signal, entry_allow_mask)

            baseline_bt = run_backtest(test_df, test_signal, cost_model=cost_model)
            gated_bt = run_backtest(test_df, gated_signal, cost_model=cost_model)
            baseline_metrics = compute_metrics(
                baseline_bt.returns,
                baseline_bt.equity,
                baseline_bt.trades,
                timeframe=timeframe,
                position=baseline_bt.position,
            )
            gated_metrics = compute_metrics(
                gated_bt.returns,
                gated_bt.equity,
                gated_bt.trades,
                timeframe=timeframe,
                position=gated_bt.position,
            )

            if model.get("fitted", False):
                test_bar_scores = _score_with_model(model, test_features[FEATURE_COLUMNS])
                bar_allow = test_bar_scores >= test_threshold if np.isfinite(test_threshold) else pd.Series(
                    True, index=test_bar_scores.index, dtype=bool
                )
            else:
                test_bar_scores = pd.to_numeric(test_features.get("tsmom_avg"), errors="coerce").fillna(0.0)
                bar_threshold = _percentile_threshold(test_bar_scores, score_quantile)
                bar_allow = test_bar_scores >= bar_threshold if np.isfinite(bar_threshold) else pd.Series(
                    True, index=test_bar_scores.index, dtype=bool
                )

            entry_pass_rate = float(allow_entries.mean()) if len(allow_entries) > 0 else 0.0
            trades_per_fold = float(len(gated_bt.trades))
            coverage_rows.append(
                {
                    "symbol": symbol,
                    "fold_id": fold_id,
                    "allowed_bar_pct": float(bar_allow.mean()),
                    "entry_pass_rate": entry_pass_rate,
                    "trades_per_fold": trades_per_fold,
                    "effective_trade_coverage": float(selection_diag.get("effective_trade_coverage", 0.0)),
                    "fold_trade_viability": bool(selection_diag.get("fold_trade_viability", False)),
                    "selection_expansion_events": int(selection_diag.get("selection_expansion_events", 0)),
                    "train_percentile_threshold": _safe_float(train_threshold, default=np.nan),
                    "test_percentile_threshold": _safe_float(test_threshold, default=np.nan),
                    "threshold_drift": _safe_float(threshold_drift, default=np.nan),
                    "zero_trade_fold": float(trades_per_fold <= 0),
                }
            )

            fold_rows.append(
                {
                    "symbol": symbol,
                    "fold_id": fold_id,
                    "fold_test_start": test_start,
                    "fold_test_end": test_end,
                    "baseline_sharpe": float(baseline_metrics["Sharpe"]),
                    "gated_sharpe": float(gated_metrics["Sharpe"]),
                    "delta_sharpe": float(gated_metrics["Sharpe"] - baseline_metrics["Sharpe"]),
                    "baseline_expectancy": float(baseline_metrics["Expectancy"]),
                    "gated_expectancy": float(gated_metrics["Expectancy"]),
                    "delta_expectancy": float(
                        gated_metrics["Expectancy"] - baseline_metrics["Expectancy"]
                    ),
                    "baseline_max_dd": float(baseline_metrics["MaxDrawdown"]),
                    "gated_max_dd": float(gated_metrics["MaxDrawdown"]),
                    "delta_max_dd": float(gated_metrics["MaxDrawdown"] - baseline_metrics["MaxDrawdown"]),
                    "baseline_trade_count": float(baseline_metrics["TradeCount"]),
                    "gated_trade_count": float(gated_metrics["TradeCount"]),
                    "gated_trades_per_fold": float(gated_metrics["TradeCount"]),
                    "delta_trade_count": float(
                        gated_metrics["TradeCount"] - baseline_metrics["TradeCount"]
                    ),
                    "baseline_win_rate": float(baseline_metrics["WinRate"]),
                    "gated_win_rate": float(gated_metrics["WinRate"]),
                }
            )

    trade_feature_dataset = pd.DataFrame(trade_rows).sort_values(
        ["symbol", "fold_id", "split", "entry_time"]
    )
    gate_model_scores = pd.DataFrame(score_rows).sort_values(
        ["symbol", "fold_id", "split", "entry_time"]
    )
    gate_comparison_by_fold = pd.DataFrame(fold_rows).sort_values(["symbol", "fold_id"])
    gate_coverage = pd.DataFrame(coverage_rows).sort_values(["symbol", "fold_id"])

    if not gate_coverage.empty:
        summary_rows = []
        for symbol, g in gate_coverage.groupby("symbol"):
            summary_rows.append(
                {
                    "symbol": symbol,
                    "fold_id": "SUMMARY",
                    "allowed_bar_pct": float(g["allowed_bar_pct"].mean()),
                    "entry_pass_rate": float(g["entry_pass_rate"].mean()),
                    "trades_per_fold": float(g["trades_per_fold"].mean()),
                    "effective_trade_coverage": float(g["effective_trade_coverage"].mean()),
                    "fold_trade_viability": float(g["fold_trade_viability"].mean()),
                    "selection_expansion_events": float(g["selection_expansion_events"].sum()),
                    "train_percentile_threshold": float(g["train_percentile_threshold"].mean()),
                    "test_percentile_threshold": float(g["test_percentile_threshold"].mean()),
                    "threshold_drift": float(g["threshold_drift"].mean()),
                    "zero_trade_fold": float(g["zero_trade_fold"].mean()),
                }
            )
        summary_rows.append(
            {
                "symbol": "ALL",
                "fold_id": "SUMMARY",
                "allowed_bar_pct": float(gate_coverage["allowed_bar_pct"].mean()),
                "entry_pass_rate": float(gate_coverage["entry_pass_rate"].mean()),
                "trades_per_fold": float(gate_coverage["trades_per_fold"].mean()),
                "effective_trade_coverage": float(gate_coverage["effective_trade_coverage"].mean()),
                "fold_trade_viability": float(gate_coverage["fold_trade_viability"].mean()),
                "selection_expansion_events": float(gate_coverage["selection_expansion_events"].sum()),
                "train_percentile_threshold": float(gate_coverage["train_percentile_threshold"].mean()),
                "test_percentile_threshold": float(gate_coverage["test_percentile_threshold"].mean()),
                "threshold_drift": float(gate_coverage["threshold_drift"].mean()),
                "zero_trade_fold": float(gate_coverage["zero_trade_fold"].mean()),
            }
        )
        gate_coverage = pd.concat([gate_coverage, pd.DataFrame(summary_rows)], ignore_index=True)
        gate_coverage = gate_coverage.rename(columns={"zero_trade_fold": "zero_trade_folds_pct"})

    # Univariate diagnostics on out-of-sample trades.
    test_trade_dataset = trade_feature_dataset[trade_feature_dataset["split"] == "test"].copy()
    trade_feature_bins = _summarize_feature_bins(test_trade_dataset, charts_dir)

    # Required interaction plots.
    _plot_interaction_heatmap(
        test_trade_dataset,
        "tsmom_avg",
        "atr_percentile",
        charts_dir / "interaction_heatmap_tsmom_avg_x_atr_percentile.png",
    )
    _plot_interaction_heatmap(
        test_trade_dataset,
        "vr_12",
        "tsmom_avg",
        charts_dir / "interaction_heatmap_vr_12_x_tsmom_avg.png",
    )
    _plot_interaction_heatmap(
        test_trade_dataset,
        "adx_slope",
        "tsmom_avg",
        charts_dir / "interaction_heatmap_adx_slope_x_tsmom_avg.png",
    )
    _plot_interaction_heatmap(
        test_trade_dataset,
        "trend_variance_ratio",
        "tsmom_avg",
        charts_dir / "interaction_heatmap_trend_variance_ratio_x_tsmom_avg.png",
    )

    _plot_score_decile_uplift(gate_model_scores, charts_dir / "score_deciles_uplift.png")
    _plot_trade_rank_vs_return(gate_model_scores, charts_dir / "trade_rank_vs_return.png")
    _plot_fold_uplift_boxplot(gate_comparison_by_fold, charts_dir / "fold_uplift_boxplot.png")
    _compute_ks_drift(
        trade_feature_df=trade_feature_dataset,
        output_chart_path=charts_dir / "feature_drift_ks.png",
        output_csv_path=out_dir / "feature_drift_ks.csv",
    )
    _run_stat_tests(
        gate_comparison_by_fold,
        output_csv=out_dir / "gate_statistical_tests.csv",
        spa_stub_path=out_dir / "reality_check_spa_stub.json",
    )
    gate_rule_export = _export_gate_rules(
        fold_models=fold_models,
        strategy=strategy,
        score_quantile=score_quantile,
        top_k_percent=top_k_percent,
        min_trades_per_fold=min_trades_per_fold,
        target_pass_rate_min=target_pass_rate_min,
        target_pass_rate_max=target_pass_rate_max,
        hybrid_tsmom_enabled=hybrid_tsmom_enabled,
        output_json=out_dir / "gate_rule_export.json",
        output_md=out_dir / "gate_rule_export.md",
    )

    trade_feature_dataset.to_csv(out_dir / "trade_feature_dataset.csv", index=False)
    trade_feature_bins.to_csv(out_dir / "trade_feature_bins.csv", index=False)
    gate_model_scores.to_csv(out_dir / "gate_model_scores.csv", index=False)
    gate_comparison_by_fold.to_csv(out_dir / "gate_comparison_by_fold.csv", index=False)
    gate_coverage.to_csv(out_dir / "gate_coverage_metrics.csv", index=False)

    return R13Artifacts(
        trade_feature_dataset=trade_feature_dataset,
        trade_feature_bins=trade_feature_bins,
        gate_model_scores=gate_model_scores,
        gate_comparison_by_fold=gate_comparison_by_fold,
        gate_coverage_metrics=gate_coverage,
        gate_rule_export=gate_rule_export,
        output_dir=out_dir,
    )


__all__ = ["R13Artifacts", "run_r13_trend_gating", "FEATURE_COLUMNS"]
