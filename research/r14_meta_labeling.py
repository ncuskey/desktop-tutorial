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
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text

from data import CostModel, add_basic_indicators, load_ohlcv_csv, load_symbol_data
from execution.simulator import run_backtest
from metrics.performance import compute_metrics
from regime import attach_regime_labels
from strategies import trend_breakout_v2_signals


@dataclass
class R14Artifacts:
    meta_feature_dataset: pd.DataFrame
    meta_model_scores: pd.DataFrame
    meta_gate_comparison: pd.DataFrame
    meta_coverage_metrics: pd.DataFrame
    meta_feature_importance: pd.DataFrame
    output_dir: Path


ENTRY_FEATURES = [
    "tsmom_24",
    "tsmom_48",
    "tsmom_72",
    "tsmom_avg",
    "vr_6",
    "vr_12",
    "vr_24",
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
]

EARLY_FEATURES = [
    "early_return_1",
    "early_return_3",
    "early_volatility",
    "early_range_expansion",
    "early_mfe",
    "early_mae",
    "early_slope",
    "volatility_spike_flag",
]

META_FEATURE_COLUMNS = ENTRY_FEATURES + EARLY_FEATURES


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


def _resolve_artifacts_root(path_like: str | Path) -> Path:
    explicit = Path(path_like)
    if explicit.exists():
        return explicit
    fallback = Path("/tmp/r122_artifacts")
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Artifacts root not found: {explicit}")


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
        "Could not locate price data for R1.4 evaluation. Pass --source-csv explicitly."
    )


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


def _apply_entry_gate_to_signal(raw_signal: pd.Series, entry_allow: pd.Series) -> pd.Series:
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
    return out


def _compute_meta_label_and_early_features(
    segment_df: pd.DataFrame,
    bar_features: pd.DataFrame,
    entry_idx: int,
    side: int,
    *,
    tp_atr_mult: float,
    sl_atr_mult: float,
    max_horizon: int,
    early_window: int,
) -> dict[str, Any]:
    row = {}
    if entry_idx not in segment_df.index:
        return row
    entry_pos = segment_df.index.get_loc(entry_idx)
    if isinstance(entry_pos, slice):
        entry_pos = entry_pos.start
    entry_px = _safe_float(segment_df.loc[entry_idx, "close"], default=np.nan)
    atr_entry = _safe_float(segment_df.loc[entry_idx, "atr_14"], default=np.nan)
    if not np.isfinite(entry_px) or not np.isfinite(atr_entry) or atr_entry <= 0:
        return row

    start = entry_pos + 1
    end = min(entry_pos + max_horizon, len(segment_df) - 1)
    if start > end:
        return row
    horizon = segment_df.iloc[start : end + 1]
    if horizon.empty:
        return row

    tp_level_long = entry_px + tp_atr_mult * atr_entry
    sl_level_long = entry_px - sl_atr_mult * atr_entry
    tp_level_short = entry_px - tp_atr_mult * atr_entry
    sl_level_short = entry_px + sl_atr_mult * atr_entry

    label = 0
    for _, bar in horizon.iterrows():
        hi = _safe_float(bar.get("high"), default=np.nan)
        lo = _safe_float(bar.get("low"), default=np.nan)
        if not np.isfinite(hi) or not np.isfinite(lo):
            continue
        if side >= 0:
            tp_hit = hi >= tp_level_long
            sl_hit = lo <= sl_level_long
        else:
            tp_hit = lo <= tp_level_short
            sl_hit = hi >= sl_level_short
        if tp_hit and not sl_hit:
            label = 1
            break
        if sl_hit:
            label = 0
            break

    e_end = min(entry_pos + early_window, len(segment_df) - 1)
    e_slice = segment_df.iloc[entry_pos + 1 : e_end + 1]
    if e_slice.empty:
        early_return_1 = np.nan
        early_return_3 = np.nan
        early_volatility = np.nan
        early_range_expansion = np.nan
        early_mfe = np.nan
        early_mae = np.nan
        early_slope = np.nan
        volatility_spike_flag = np.nan
    else:
        close = pd.to_numeric(e_slice["close"], errors="coerce")
        high = pd.to_numeric(e_slice["high"], errors="coerce")
        low = pd.to_numeric(e_slice["low"], errors="coerce")
        first_close = _safe_float(close.iloc[0], np.nan)
        third_close = _safe_float(close.iloc[min(2, len(close) - 1)], np.nan)
        early_return_1 = np.log(first_close / entry_px) if np.isfinite(first_close) else np.nan
        early_return_3 = np.log(third_close / entry_px) if np.isfinite(third_close) else np.nan
        early_rets = np.log(close / close.shift(1)).dropna()
        early_volatility = _safe_float(early_rets.std(ddof=0), np.nan)
        local_range = (_safe_float(high.max(), np.nan) - _safe_float(low.min(), np.nan)) / entry_px
        early_range_expansion = local_range
        if side >= 0:
            early_mfe = (_safe_float(high.max(), np.nan) - entry_px) / entry_px
            early_mae = (entry_px - _safe_float(low.min(), np.nan)) / entry_px
        else:
            early_mfe = (entry_px - _safe_float(low.min(), np.nan)) / entry_px
            early_mae = (_safe_float(high.max(), np.nan) - entry_px) / entry_px
        n = len(close)
        if n >= 2:
            x = np.arange(n, dtype=float)
            y = close.to_numpy(dtype=float)
            y = y / entry_px
            slope = np.polyfit(x, y, 1)[0]
            early_slope = float(slope if side >= 0 else -slope)
        else:
            early_slope = np.nan
        atr_local = pd.to_numeric(e_slice.get("atr_14"), errors="coerce")
        atr_mean = _safe_float(atr_local.mean(), np.nan)
        volatility_spike_flag = float(atr_mean > (1.15 * atr_entry)) if np.isfinite(atr_mean) else np.nan

    row.update(
        {
            "follow_through_label": int(label),
            "early_return_1": early_return_1,
            "early_return_3": early_return_3,
            "early_volatility": early_volatility,
            "early_range_expansion": early_range_expansion,
            "early_mfe": early_mfe,
            "early_mae": early_mae,
            "early_slope": early_slope,
            "volatility_spike_flag": volatility_spike_flag,
        }
    )
    return row


def _build_meta_trade_rows(
    segment_df: pd.DataFrame,
    bar_features: pd.DataFrame,
    signal: pd.Series,
    symbol: str,
    fold_id: int,
    split: str,
    cost_model: CostModel,
    *,
    tp_atr_mult: float,
    sl_atr_mult: float,
    max_horizon: int,
    early_window: int,
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
        base = {
            "symbol": symbol,
            "fold_id": int(fold_id),
            "split": split,
            "entry_idx": int(entry_idx),
            "entry_time": entry_time,
            "exit_time": pd.to_datetime(segment_df.loc[exit_idx, "timestamp"], utc=True, errors="coerce")
            if exit_idx in segment_df.index
            else pd.NaT,
            "side": int(trade.get("side", 0)),
            "trade_return": _safe_float(trade.get("trade_return"), default=np.nan),
            "holding_bars": _safe_float(trade.get("holding_bars"), default=np.nan),
        }
        feat = bar_features.loc[entry_idx] if entry_idx in bar_features.index else pd.Series()
        for c in ENTRY_FEATURES:
            base[c] = _safe_float(feat.get(c), default=np.nan)
        extra = _compute_meta_label_and_early_features(
            segment_df=segment_df,
            bar_features=bar_features,
            entry_idx=entry_idx,
            side=int(trade.get("side", 0)),
            tp_atr_mult=tp_atr_mult,
            sl_atr_mult=sl_atr_mult,
            max_horizon=max_horizon,
            early_window=early_window,
        )
        if not extra:
            continue
        base.update(extra)
        rows.append(base)
    return rows


def _prepare_model_inputs(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    x = df[feature_cols].copy()
    medians = x.median(axis=0, numeric_only=True)
    x = x.fillna(medians)
    return x, medians


def _score_with_model(model: dict[str, Any], feature_df: pd.DataFrame) -> pd.Series:
    if feature_df.empty:
        return pd.Series(dtype=float)
    if not model.get("fitted", False):
        return pd.Series(np.nan, index=feature_df.index, dtype=float)
    x = feature_df[model["feature_cols"]].copy()
    medians = pd.Series(model["medians"])
    x = x.fillna(medians)
    mean = np.asarray(model["mean"], dtype=float)
    std = np.asarray(model["std"], dtype=float)
    coef = np.asarray(model["coef"], dtype=float)
    intercept = float(model["intercept"])
    x_scaled = (x.to_numpy(dtype=float) - mean) / std
    logits = np.clip(x_scaled @ coef + intercept, -20.0, 20.0)
    probs = 1.0 / (1.0 + np.exp(-logits))
    return pd.Series(probs, index=feature_df.index, dtype=float)


def _fit_meta_models(train_df: pd.DataFrame, feature_cols: list[str]) -> dict[str, Any]:
    x_raw, medians = _prepare_model_inputs(train_df, feature_cols)
    y = pd.to_numeric(train_df["follow_through_label"], errors="coerce").fillna(0).astype(int)
    if len(train_df) < 20 or y.nunique() < 2:
        return {
            "fitted": False,
            "reason": "insufficient_train_samples_or_single_class",
            "feature_cols": list(feature_cols),
            "medians": medians.to_dict(),
        }
    x = x_raw.to_numpy(dtype=float)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std <= 1e-12, 1.0, std)
    x_scaled = (x - mean) / std

    logit = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, class_weight="balanced", random_state=42)
    logit.fit(x_scaled, y.to_numpy())

    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=42)
    tree.fit(x, y.to_numpy())
    tree_text = export_text(tree, feature_names=feature_cols)
    return {
        "fitted": True,
        "feature_cols": list(feature_cols),
        "medians": medians.to_dict(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "coef": logit.coef_[0].tolist(),
        "intercept": float(logit.intercept_[0]),
        "tree_text": tree_text,
    }


def _select_by_rank(scores: pd.Series, min_trades_per_fold: int, top_k_percent: float) -> pd.Series:
    if scores.empty:
        return pd.Series(dtype=bool)
    s = pd.to_numeric(scores, errors="coerce").fillna(-np.inf)
    ranked = s.sort_values(ascending=False, kind="stable").index.tolist()
    n = len(ranked)
    k_pct = max(1, int(np.ceil(n * float(top_k_percent))))
    k_min = min(max(1, int(min_trades_per_fold)), n)
    k = max(k_pct, k_min)
    selected = ranked[:k]
    allow = pd.Series(False, index=scores.index, dtype=bool)
    allow.loc[selected] = True
    return allow


def _bootstrap_ci(values: np.ndarray, n_bootstrap: int = 4000) -> tuple[float, float]:
    if values.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(42)
    samples = []
    for _ in range(n_bootstrap):
        s = rng.choice(values, size=values.size, replace=True)
        samples.append(float(np.median(s)))
    return float(np.quantile(samples, 0.05)), float(np.quantile(samples, 0.95))


def _plot_meta_score_vs_return(meta_scores: pd.DataFrame, output_path: Path) -> None:
    local = meta_scores[meta_scores["split"] == "test"].copy()
    if local.empty:
        return
    local["follow_through_probability"] = pd.to_numeric(
        local["follow_through_probability"], errors="coerce"
    )
    local["actual_return"] = pd.to_numeric(local["actual_return"], errors="coerce")
    local = local.dropna(subset=["follow_through_probability", "actual_return"])
    if len(local) < 20:
        return
    local["rank_pct"] = local["follow_through_probability"].rank(pct=True, method="average") * 100.0
    local["rank_bin"] = (np.floor(local["rank_pct"] / 5.0) * 5.0).clip(0.0, 95.0)
    grouped = local.groupby("rank_bin", dropna=True)["actual_return"].mean().reset_index()
    if grouped.empty:
        return
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.plot(grouped["rank_bin"], grouped["actual_return"], marker="o", color="#4C78A8")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_title("Meta score percentile vs average return")
    ax.set_xlabel("Score percentile bin")
    ax.set_ylabel("Average return")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_meta_calibration(meta_scores: pd.DataFrame, output_path: Path) -> None:
    local = meta_scores[meta_scores["split"] == "test"].copy()
    if local.empty:
        return
    local["follow_through_probability"] = pd.to_numeric(
        local["follow_through_probability"], errors="coerce"
    )
    local["actual_label"] = pd.to_numeric(local["actual_label"], errors="coerce")
    local = local.dropna(subset=["follow_through_probability", "actual_label"])
    if len(local) < 20:
        return
    try:
        local["bin"] = pd.qcut(local["follow_through_probability"], q=10, labels=False, duplicates="drop")
    except Exception:
        return
    grouped = local.groupby("bin", dropna=True).agg(
        predicted=("follow_through_probability", "mean"),
        observed=("actual_label", "mean"),
    ).reset_index()
    if grouped.empty:
        return
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1.0)
    ax.plot(grouped["predicted"], grouped["observed"], marker="o", color="#F58518")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Meta calibration curve")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed follow-through rate")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_r14_meta_labeling(
    strategy: str = "TrendBreakout_V2",
    symbols: list[str] | None = None,
    artifacts_root: str | Path = "outputs/TrendBreakout_V2",
    output_dir: str | Path = "outputs",
    timeframe: str = "H1",
    source_csv: str | Path | None = None,
    tp_atr_mult: float = 1.0,
    sl_atr_mult: float = 0.5,
    max_horizon: int = 24,
    early_window: int = 3,
    top_k_percent: float = 0.30,
    min_trades_per_fold: int = 5,
) -> R14Artifacts:
    if strategy != "TrendBreakout_V2":
        raise ValueError("R1.4 currently supports strategy='TrendBreakout_V2' only.")
    target_symbols = symbols or ["EURUSD", "GBPUSD", "AUDUSD"]
    artifacts_root_path = _resolve_artifacts_root(artifacts_root)
    out_dir = Path(output_dir)
    charts_dir = out_dir / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    price_path = _resolve_price_data_path(artifacts_root_path, source_csv)
    raw_prices = load_ohlcv_csv(price_path)

    meta_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []

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

            train_meta_rows = _build_meta_trade_rows(
                segment_df=train_df,
                bar_features=train_features,
                signal=train_signal,
                symbol=symbol,
                fold_id=fold_id,
                split="train",
                cost_model=cost_model,
                tp_atr_mult=tp_atr_mult,
                sl_atr_mult=sl_atr_mult,
                max_horizon=max_horizon,
                early_window=early_window,
            )
            test_meta_rows = _build_meta_trade_rows(
                segment_df=test_df,
                bar_features=test_features,
                signal=test_signal,
                symbol=symbol,
                fold_id=fold_id,
                split="test",
                cost_model=cost_model,
                tp_atr_mult=tp_atr_mult,
                sl_atr_mult=sl_atr_mult,
                max_horizon=max_horizon,
                early_window=early_window,
            )
            meta_rows.extend(train_meta_rows)
            meta_rows.extend(test_meta_rows)

            train_meta_df = pd.DataFrame(train_meta_rows)
            test_meta_df = pd.DataFrame(test_meta_rows)
            model = _fit_meta_models(train_meta_df, META_FEATURE_COLUMNS)

            if model.get("fitted", False):
                coef_series = pd.Series(model["coef"], index=model["feature_cols"])
                for feature, coef in coef_series.items():
                    importance_rows.append(
                        {
                            "symbol": symbol,
                            "fold_id": fold_id,
                            "model": "logistic_primary",
                            "feature": feature,
                            "importance": float(coef),
                            "abs_importance": float(abs(coef)),
                        }
                    )
                importance_rows.append(
                    {
                        "symbol": symbol,
                        "fold_id": fold_id,
                        "model": "tree_snapshot",
                        "feature": "__tree_text__",
                        "importance": np.nan,
                        "abs_importance": np.nan,
                        "tree_text": model.get("tree_text", ""),
                    }
                )

            # Build entry-aligned feature matrices from meta rows (includes early post-entry features).
            train_entry_features_df = (
                train_meta_df[["entry_idx"] + META_FEATURE_COLUMNS].dropna(subset=["entry_idx"]).copy()
                if not train_meta_df.empty
                else pd.DataFrame(columns=["entry_idx"] + META_FEATURE_COLUMNS)
            )
            if not train_entry_features_df.empty:
                train_entry_features_df["entry_idx"] = train_entry_features_df["entry_idx"].astype(int)
                train_entry_features = train_entry_features_df.set_index("entry_idx")[META_FEATURE_COLUMNS]
            else:
                train_entry_features = pd.DataFrame(columns=META_FEATURE_COLUMNS)

            test_entry_features_df = (
                test_meta_df[["entry_idx"] + META_FEATURE_COLUMNS].dropna(subset=["entry_idx"]).copy()
                if not test_meta_df.empty
                else pd.DataFrame(columns=["entry_idx"] + META_FEATURE_COLUMNS)
            )
            if not test_entry_features_df.empty:
                test_entry_features_df["entry_idx"] = test_entry_features_df["entry_idx"].astype(int)
                test_entry_features = test_entry_features_df.set_index("entry_idx")[META_FEATURE_COLUMNS]
            else:
                test_entry_features = pd.DataFrame(columns=META_FEATURE_COLUMNS)

            test_entry_mask = _entry_mask_from_signal(test_signal)
            test_entry_idx = test_signal.index[test_entry_mask]
            if len(test_entry_idx) > 0:
                # Restrict to entries that have full meta-feature rows.
                test_entry_idx = pd.Index([i for i in test_entry_idx if i in test_entry_features.index])
                if model.get("fitted", False):
                    probs = _score_with_model(model, test_entry_features.loc[test_entry_idx])
                else:
                    probs = pd.Series(0.5, index=test_entry_idx, dtype=float)
                allow_entries = _select_by_rank(
                    scores=probs,
                    min_trades_per_fold=min_trades_per_fold,
                    top_k_percent=top_k_percent,
                )
            else:
                probs = pd.Series(dtype=float)
                allow_entries = pd.Series(dtype=bool)

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

            for _, row in test_meta_df.iterrows():
                entry_time = pd.to_datetime(row["entry_time"], utc=True, errors="coerce")
                idx_match = test_df.index[test_df["timestamp"] == entry_time]
                prob_val = np.nan
                allow_val = 1
                if len(idx_match) > 0:
                    idx0 = idx_match[0]
                    if idx0 in probs.index:
                        prob_val = _safe_float(probs.loc[idx0], default=np.nan)
                    if idx0 in allow_entries.index:
                        allow_val = int(bool(allow_entries.loc[idx0]))
                score_rows.append(
                    {
                        "symbol": symbol,
                        "fold_id": fold_id,
                        "split": "test",
                        "entry_time": entry_time,
                        "follow_through_probability": prob_val,
                        "actual_label": int(row.get("follow_through_label", 0)),
                        "actual_return": _safe_float(row.get("trade_return"), default=np.nan),
                        "allow_trade": allow_val,
                    }
                )

            coverage_rows.append(
                {
                    "symbol": symbol,
                    "fold_id": fold_id,
                    "trades_per_fold": float(gated_metrics["TradeCount"]),
                    "effective_trade_coverage": float(allow_entries.mean()) if len(allow_entries) else 0.0,
                    "zero_trade_fold": float(gated_metrics["TradeCount"] <= 0),
                }
            )
            fold_rows.append(
                {
                    "symbol": symbol,
                    "fold_id": fold_id,
                    "baseline_sharpe": float(baseline_metrics["Sharpe"]),
                    "meta_sharpe": float(gated_metrics["Sharpe"]),
                    "delta_sharpe": float(gated_metrics["Sharpe"] - baseline_metrics["Sharpe"]),
                    "baseline_expectancy": float(baseline_metrics["Expectancy"]),
                    "meta_expectancy": float(gated_metrics["Expectancy"]),
                    "delta_expectancy": float(
                        gated_metrics["Expectancy"] - baseline_metrics["Expectancy"]
                    ),
                    "baseline_max_dd": float(baseline_metrics["MaxDrawdown"]),
                    "meta_max_dd": float(gated_metrics["MaxDrawdown"]),
                    "delta_max_dd": float(
                        gated_metrics["MaxDrawdown"] - baseline_metrics["MaxDrawdown"]
                    ),
                    "baseline_trade_count": float(baseline_metrics["TradeCount"]),
                    "meta_trade_count": float(gated_metrics["TradeCount"]),
                    "delta_trade_count": float(
                        gated_metrics["TradeCount"] - baseline_metrics["TradeCount"]
                    ),
                }
            )

    meta_feature_dataset = pd.DataFrame(meta_rows).sort_values(
        ["symbol", "fold_id", "split", "entry_time"]
    )
    meta_model_scores = pd.DataFrame(score_rows).sort_values(
        ["symbol", "fold_id", "entry_time"]
    )
    meta_gate_comparison = pd.DataFrame(fold_rows).sort_values(["symbol", "fold_id"])
    meta_coverage = pd.DataFrame(coverage_rows).sort_values(["symbol", "fold_id"])
    meta_feature_importance = pd.DataFrame(importance_rows)

    if not meta_coverage.empty:
        summary_rows = []
        for symbol, g in meta_coverage.groupby("symbol"):
            summary_rows.append(
                {
                    "symbol": symbol,
                    "fold_id": "SUMMARY",
                    "trades_per_fold": float(g["trades_per_fold"].mean()),
                    "effective_trade_coverage": float(g["effective_trade_coverage"].mean()),
                    "zero_trade_fold": float(g["zero_trade_fold"].mean()),
                }
            )
        summary_rows.append(
            {
                "symbol": "ALL",
                "fold_id": "SUMMARY",
                "trades_per_fold": float(meta_coverage["trades_per_fold"].mean()),
                "effective_trade_coverage": float(meta_coverage["effective_trade_coverage"].mean()),
                "zero_trade_fold": float(meta_coverage["zero_trade_fold"].mean()),
            }
        )
        meta_coverage = pd.concat([meta_coverage, pd.DataFrame(summary_rows)], ignore_index=True)
        meta_coverage = meta_coverage.rename(columns={"zero_trade_fold": "zero_trade_folds_pct"})

    _plot_meta_score_vs_return(meta_model_scores, charts_dir / "meta_score_vs_return.png")
    _plot_meta_calibration(meta_model_scores, charts_dir / "meta_calibration_curve.png")

    # Statistical tests.
    stat_rows: list[dict[str, Any]] = []
    for metric in ["delta_sharpe", "delta_expectancy", "delta_max_dd", "delta_trade_count"]:
        vals = pd.to_numeric(meta_gate_comparison.get(metric), errors="coerce").dropna().to_numpy(
            dtype=float
        )
        if vals.size == 0:
            continue
        try:
            w = wilcoxon(vals, zero_method="wilcox", alternative="two-sided")
            w_stat = float(w.statistic)
            w_p = float(w.pvalue)
        except Exception:
            w_stat = np.nan
            w_p = np.nan
        ci_low, ci_high = _bootstrap_ci(vals)
        stat_rows.append(
            {
                "metric": metric,
                "n_folds": int(vals.size),
                "median_delta": float(np.median(vals)),
                "mean_delta": float(np.mean(vals)),
                "wilcoxon_stat": w_stat,
                "wilcoxon_p_value": w_p,
                "bootstrap_median_ci_05": ci_low,
                "bootstrap_median_ci_95": ci_high,
            }
        )
    meta_stat_tests = pd.DataFrame(stat_rows)

    # Write outputs.
    meta_feature_dataset.to_csv(out_dir / "meta_feature_dataset.csv", index=False)
    meta_model_scores.to_csv(out_dir / "meta_model_scores.csv", index=False)
    meta_gate_comparison.to_csv(out_dir / "meta_gate_comparison.csv", index=False)
    meta_coverage.to_csv(out_dir / "meta_coverage_metrics.csv", index=False)
    meta_stat_tests.to_csv(out_dir / "meta_stat_tests.csv", index=False)
    meta_feature_importance.to_csv(out_dir / "meta_feature_importance.csv", index=False)

    return R14Artifacts(
        meta_feature_dataset=meta_feature_dataset,
        meta_model_scores=meta_model_scores,
        meta_gate_comparison=meta_gate_comparison,
        meta_coverage_metrics=meta_coverage,
        meta_feature_importance=meta_feature_importance,
        output_dir=out_dir,
    )


__all__ = ["R14Artifacts", "run_r14_meta_labeling", "META_FEATURE_COLUMNS"]

