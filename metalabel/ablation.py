from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from .features_trade_quality import build_trade_meta_features
from .filter_rule_based import RuleBasedMetaFilter
from .labels import compute_forward_trade_returns, create_trade_success_labels


def _max_drawdown_from_returns(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    dd = (equity / equity.cummax()) - 1.0
    return float(dd.min())


def _sharpe_from_returns(returns: pd.Series, periods_per_year: int = 24 * 252) -> float:
    r = returns.dropna().astype(float)
    if r.empty:
        return 0.0
    std = float(r.std(ddof=0))
    if std <= 1e-12:
        return 0.0
    return float((r.mean() / std) * (periods_per_year**0.5))


def _default_feature_groups(columns: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {
        "regime_features": [],
        "volatility_features": [],
        "trend_structure_features": [],
        "mean_reversion_features": [],
        "trade_context_features": [],
        "timing_features": [],
    }
    for c in columns:
        lc = c.lower()
        if c in {"stable_trend_regime", "stable_vol_regime", "filter_type"}:
            groups["regime_features"].append(c)
        elif "atr" in lc or "vol" in lc:
            groups["volatility_features"].append(c)
        elif "adx" in lc or "ma" in lc or "trend" in lc:
            groups["trend_structure_features"].append(c)
        elif "rsi" in lc or "bb_" in lc or "dist_to_bb" in lc:
            groups["mean_reversion_features"].append(c)
        elif c in {"signal_strength", "position_in_range", "distance_to_high", "distance_to_low"}:
            groups["trade_context_features"].append(c)
        elif "bars_since" in lc or "time_since" in lc or "holding" in lc or "momentum" in lc or "range_compression" in lc:
            groups["timing_features"].append(c)
    return {k: v for k, v in groups.items() if v}


def run_feature_ablation(
    df: pd.DataFrame,
    primary_signal: pd.Series,
    *,
    feature_builder: Callable[[pd.DataFrame, pd.Series], pd.DataFrame] = build_trade_meta_features,
    label_builder: Callable[..., pd.Series] = create_trade_success_labels,
    meta_filter_class: type = RuleBasedMetaFilter,
    label_kwargs: dict[str, Any] | None = None,
    meta_filter_kwargs: dict[str, Any] | None = None,
    split_ratio: float = 0.7,
    min_train_samples: int = 30,
    feature_groups: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """
    Run feature-group ablation for trade-quality meta filter.

    Returns rows with baseline and without_<group> containing expectancy deltas.
    """
    label_kwargs = label_kwargs or {}
    meta_filter_kwargs = meta_filter_kwargs or {}
    horizon = int(label_kwargs.get("forward_horizon", label_kwargs.get("horizon_bars", 24)))

    features = feature_builder(df, primary_signal)
    entry_mask = (
        features["entry_mask"].astype(bool)
        if "entry_mask" in features.columns
        else ((primary_signal != primary_signal.shift(1)).fillna(True) & (primary_signal != 0))
    )
    labels = label_builder(
        df,
        primary_signal,
        entry_mask=entry_mask,
        **label_kwargs,
    )
    forward_returns = compute_forward_trade_returns(df, primary_signal, forward_horizon=horizon)
    event_idx = entry_mask[entry_mask].index
    valid = labels.reindex(event_idx).notna() & forward_returns.reindex(event_idx).notna()
    event_idx = event_idx[valid]

    if len(event_idx) < max(min_train_samples + 10, 40):
        return pd.DataFrame(
            [
                {
                    "variant": "baseline",
                    "expectancy_unfiltered": 0.0,
                    "expectancy_filtered": 0.0,
                    "expectancy_delta": 0.0,
                    "sharpe_unfiltered": 0.0,
                    "sharpe_filtered": 0.0,
                    "sharpe_delta": 0.0,
                    "maxdd_unfiltered": 0.0,
                    "maxdd_filtered": 0.0,
                    "maxdd_delta": 0.0,
                    "filter_rate": 0.0,
                    "samples": int(len(event_idx)),
                    "status": "insufficient_samples",
                }
            ]
        )

    split_idx = max(int(len(event_idx) * split_ratio), min_train_samples)
    split_idx = min(split_idx, len(event_idx) - 1)
    train_idx = event_idx[:split_idx]
    test_idx = event_idx[split_idx:]

    X_all = features.drop(columns=["entry_mask"], errors="ignore")
    y_all = labels.astype(float)
    r_all = forward_returns.astype(float)
    groups = feature_groups or _default_feature_groups(X_all.columns.tolist())

    variants: list[tuple[str, list[str]]] = [("baseline", [])]
    for g, cols in groups.items():
        variants.append((f"without_{g}", cols))

    rows: list[dict[str, Any]] = []
    for variant_name, drop_cols in variants:
        keep_cols = [c for c in X_all.columns if c not in set(drop_cols)]
        X_train = X_all.loc[train_idx, keep_cols]
        X_test = X_all.loc[test_idx, keep_cols]
        y_train = y_all.loc[train_idx].astype(int)
        r_train = r_all.loc[train_idx].astype(float)
        r_test = r_all.loc[test_idx].astype(float)

        can_fit = len(X_train) >= min_train_samples and y_train.nunique() > 1
        if not can_fit:
            rows.append(
                {
                    "variant": variant_name,
                    "expectancy_unfiltered": float(r_test.mean()) if not r_test.empty else 0.0,
                    "expectancy_filtered": float(r_test.mean()) if not r_test.empty else 0.0,
                    "expectancy_delta": 0.0,
                    "sharpe_unfiltered": _sharpe_from_returns(r_test),
                    "sharpe_filtered": _sharpe_from_returns(r_test),
                    "sharpe_delta": 0.0,
                    "maxdd_unfiltered": _max_drawdown_from_returns(r_test),
                    "maxdd_filtered": _max_drawdown_from_returns(r_test),
                    "maxdd_delta": 0.0,
                    "filter_rate": 0.0,
                    "samples": int(len(X_test)),
                    "status": "insufficient_train",
                }
            )
            continue

        fit_type = X_train["filter_type"].astype(str) if "filter_type" in X_train.columns else None
        test_type = X_test["filter_type"].astype(str) if "filter_type" in X_test.columns else None
        model = meta_filter_class(**meta_filter_kwargs)
        model.fit(X_train, y_train, forward_returns=r_train, filter_type=fit_type)
        take = model.predict(X_test, filter_type=test_type)

        kept = r_test.loc[take == 1]
        exp_unf = float(r_test.mean()) if not r_test.empty else 0.0
        exp_f = float(kept.mean()) if not kept.empty else 0.0
        sharpe_unf = _sharpe_from_returns(r_test)
        sharpe_f = _sharpe_from_returns(kept)
        maxdd_unf = _max_drawdown_from_returns(r_test)
        maxdd_f = _max_drawdown_from_returns(kept)
        filter_rate = float((take == 0).mean()) if len(take) > 0 else 0.0
        rows.append(
            {
                "variant": variant_name,
                "expectancy_unfiltered": exp_unf,
                "expectancy_filtered": exp_f,
                "expectancy_delta": exp_f - exp_unf,
                "sharpe_unfiltered": sharpe_unf,
                "sharpe_filtered": sharpe_f,
                "sharpe_delta": sharpe_f - sharpe_unf,
                "maxdd_unfiltered": maxdd_unf,
                "maxdd_filtered": maxdd_f,
                "maxdd_delta": maxdd_f - maxdd_unf,
                "filter_rate": filter_rate,
                "samples": int(len(X_test)),
                "status": "ok",
            }
        )

    return pd.DataFrame(rows)
