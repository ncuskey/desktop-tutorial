from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from data.costs import CostModel
from execution.simulator import run_backtest
from metrics.performance import compute_metrics
from research.parameter_sweep import grid_parameter_sweep


@dataclass
class WalkForwardResult:
    fold_results: pd.DataFrame
    combined_returns: pd.Series
    combined_equity: pd.Series
    combined_drawdown: pd.Series
    aggregate_metrics: dict[str, float]
    filtered_combined_returns: pd.Series | None = None
    filtered_combined_equity: pd.Series | None = None
    filtered_combined_drawdown: pd.Series | None = None
    filtered_aggregate_metrics: dict[str, float] | None = None
    meta_filter_diagnostics: dict[str, float] | None = None


def _entry_mask_from_signal(signal: pd.Series) -> pd.Series:
    s = signal.fillna(0.0).astype(float)
    change = (s != s.shift(1)).fillna(True)
    return (change & (s != 0.0)).astype(bool)


def run_walk_forward(
    df: pd.DataFrame,
    strategy_fn: Callable[[pd.DataFrame, dict], pd.Series],
    param_grid: dict[str, list],
    train_bars: int,
    test_bars: int,
    cost_model: CostModel,
    timeframe: str = "H1",
    objective_metric: str = "Sharpe",
    regime_column: str | None = None,
    meta_filter_class: type | None = None,
    meta_filter_kwargs: dict[str, Any] | None = None,
    meta_feature_builder: Callable[[pd.DataFrame, pd.Series], pd.DataFrame] | None = None,
    meta_label_builder: Callable[..., pd.Series] | None = None,
    meta_label_kwargs: dict[str, Any] | None = None,
    meta_apply_fn: Callable[[pd.Series, pd.Series, pd.Series], pd.Series] | None = None,
    meta_min_train_samples: int = 30,
) -> WalkForwardResult:
    use_meta = meta_filter_class is not None
    if use_meta and (meta_feature_builder is None or meta_label_builder is None):
        raise ValueError(
            "meta_feature_builder and meta_label_builder are required when meta_filter_class is provided."
        )

    meta_filter_kwargs = meta_filter_kwargs or {}
    meta_label_kwargs = meta_label_kwargs or {}
    meta_label_method = str(meta_label_kwargs.get("method", "top_quantile"))
    forward_horizon = int(
        meta_label_kwargs.get(
            "forward_horizon",
            meta_label_kwargs.get("horizon_bars", 24),
        )
    )

    folds: list[dict] = []
    stitched_returns: list[np.ndarray] = []
    stitched_positions: list[np.ndarray] = []
    stitched_index: list = []
    aggregated_trades: list[pd.DataFrame] = []

    filtered_stitched_returns: list[np.ndarray] = []
    filtered_stitched_positions: list[np.ndarray] = []
    filtered_aggregated_trades: list[pd.DataFrame] = []
    meta_filter_rates: list[float] = []
    meta_thresholds: list[float] = []
    sharpe_improved: list[bool] = []
    expectancy_improved: list[bool] = []
    drawdown_improved: list[bool] = []
    avg_signal_strength_filtered: list[float] = []
    avg_signal_strength_unfiltered: list[float] = []
    filter_rate_errors: list[float] = []
    threshold_clipped_flags: list[bool] = []

    i = 0
    while i + train_bars + test_bars <= len(df):
        train_df = df.iloc[i : i + train_bars].copy()
        test_df = df.iloc[i + train_bars : i + train_bars + test_bars].copy()

        sweep = grid_parameter_sweep(
            train_df,
            strategy_fn=strategy_fn,
            param_grid=param_grid,
            cost_model=cost_model,
            timeframe=timeframe,
        )
        best = sweep.iloc[0]
        best_params = best["params"]

        train_signal = strategy_fn(train_df, best_params).astype(float)
        test_signal = strategy_fn(test_df, best_params).astype(float)

        unfiltered_bt = run_backtest(test_df, test_signal, cost_model=cost_model)
        unfiltered_metrics = compute_metrics(
            unfiltered_bt.returns,
            unfiltered_bt.equity,
            unfiltered_bt.trades,
            timeframe=timeframe,
            position=unfiltered_bt.position,
        )

        filtered_bt = None
        filtered_metrics = None
        fold_filter_rate = 0.0
        fold_threshold = np.nan
        meta_state_json = "{}"
        target_filter_rate_band = [float(meta_filter_kwargs.get("min_filter_rate", 0.2)), float(meta_filter_kwargs.get("max_filter_rate", 0.6))]
        target_filter_rate_midpoint = float(np.mean(target_filter_rate_band))
        train_realized_filter_rate = np.nan
        threshold_clipped = False
        score_distribution_summary: dict[str, float] = {}

        if use_meta and meta_feature_builder is not None and meta_label_builder is not None:
            train_features = meta_feature_builder(train_df, train_signal)
            train_entry_mask = (
                train_features["entry_mask"].astype(bool)
                if "entry_mask" in train_features.columns
                else _entry_mask_from_signal(train_signal)
            )
            close_train = pd.to_numeric(train_df["close"], errors="coerce")
            train_forward_price = close_train.shift(-forward_horizon) / close_train - 1.0
            train_forward_trade_returns = np.sign(train_signal) * train_forward_price
            train_labels = meta_label_builder(
                train_df,
                train_signal,
                entry_mask=train_entry_mask,
                **meta_label_kwargs,
            )
            train_event_idx = train_entry_mask[train_entry_mask].index
            valid_train_idx = train_event_idx[
                train_labels.reindex(train_event_idx).notna()
                & train_forward_trade_returns.reindex(train_event_idx).notna()
            ]

            X_train = train_features.loc[valid_train_idx].drop(
                columns=["entry_mask"], errors="ignore"
            )
            y_train = train_labels.loc[valid_train_idx].astype(int)
            r_train = train_forward_trade_returns.loc[valid_train_idx].astype(float)

            test_features = meta_feature_builder(test_df, test_signal)
            test_entry_mask = (
                test_features["entry_mask"].astype(bool)
                if "entry_mask" in test_features.columns
                else _entry_mask_from_signal(test_signal)
            )
            X_test_events = test_features.loc[test_entry_mask].drop(
                columns=["entry_mask"], errors="ignore"
            )

            meta_take = pd.Series(1, index=test_df.index, dtype=int)
            meta_proba = pd.Series(np.nan, index=test_df.index, dtype=float)
            filter_type_used = pd.Series("global", index=test_df.index, dtype=str)
            can_fit = len(X_train) >= meta_min_train_samples and y_train.nunique() > 1

            if can_fit:
                meta_model = meta_filter_class(**meta_filter_kwargs)
                fit_filter_type = (
                    X_train["filter_type"].astype(str) if "filter_type" in X_train.columns else None
                )
                meta_model.fit(
                    X_train,
                    y_train,
                    forward_returns=r_train,
                    filter_type=fit_filter_type,
                )
                fold_threshold = float(getattr(meta_model, "threshold", np.nan))
                if hasattr(meta_model, "to_dict"):
                    model_dict = meta_model.to_dict()
                    meta_state_json = json.dumps(model_dict, sort_keys=True)
                    calibration = model_dict.get("calibration", {}) or {}
                    band = calibration.get("target_filter_rate_band", model_dict.get("target_filter_rate_band"))
                    if isinstance(band, list) and len(band) == 2:
                        target_filter_rate_band = [float(band[0]), float(band[1])]
                    target_filter_rate_midpoint = float(
                        calibration.get(
                            "target_filter_rate_midpoint",
                            np.mean(target_filter_rate_band),
                        )
                    )
                    train_realized_filter_rate = float(
                        calibration.get("realized_filter_rate", np.nan)
                    )
                    threshold_clipped = bool(calibration.get("threshold_clipped", False))
                    raw_summary = calibration.get("score_distribution_summary", {}) or {}
                    score_distribution_summary = {
                        str(k): float(v) for k, v in raw_summary.items() if v is not None
                    }
                if len(X_test_events) > 0:
                    if hasattr(meta_model, "apply"):
                        filtered_signal, meta_take, meta_proba = meta_model.apply(
                            primary_signal=test_signal,
                            entry_mask=test_entry_mask,
                            X_events=X_test_events,
                        )
                        latest_type = getattr(meta_model, "latest_filter_type_used", None)
                        if latest_type is not None:
                            filter_type_used.loc[latest_type.index] = latest_type.astype(str)
                    else:
                        transformed = meta_model.transform(X_test_events)
                        meta_take.loc[transformed.index] = transformed["meta_take"].astype(int)
                        if "meta_take_proba" in transformed.columns:
                            meta_proba.loc[transformed.index] = transformed["meta_take_proba"].astype(float)
                        if "filter_type_used" in transformed.columns:
                            filter_type_used.loc[transformed.index] = transformed["filter_type_used"].astype(str)
                        if meta_apply_fn is None:
                            raise ValueError(
                                "meta_apply_fn is required when meta model has no apply method."
                            )
                        filtered_signal = meta_apply_fn(test_signal, test_entry_mask, meta_take)
                else:
                    filtered_signal = test_signal.copy()
            else:
                filtered_signal = test_signal.copy()
                meta_state_json = json.dumps(
                    {"fitted": False, "reason": "insufficient_train_samples"},
                    sort_keys=True,
                )

            if len(X_test_events) > 0:
                fold_filter_rate = float((meta_take.loc[X_test_events.index] == 0).mean())
                if "signal_strength" in test_features.columns:
                    strength = pd.to_numeric(
                        test_features.loc[X_test_events.index, "signal_strength"],
                        errors="coerce",
                    )
                    avg_signal_strength_unfiltered.append(float(strength.mean()))
                    kept = strength.loc[meta_take.loc[X_test_events.index] == 1]
                    avg_signal_strength_filtered.append(float(kept.mean()) if not kept.empty else 0.0)
                else:
                    avg_signal_strength_unfiltered.append(0.0)
                    avg_signal_strength_filtered.append(0.0)
            if np.isfinite(target_filter_rate_midpoint):
                filter_rate_errors.append(float(fold_filter_rate - target_filter_rate_midpoint))
            threshold_clipped_flags.append(bool(threshold_clipped))

            filtered_bt = run_backtest(test_df, filtered_signal, cost_model=cost_model)
            filtered_metrics = compute_metrics(
                filtered_bt.returns,
                filtered_bt.equity,
                filtered_bt.trades,
                timeframe=timeframe,
                position=filtered_bt.position,
            )
            meta_filter_rates.append(fold_filter_rate)
            if not np.isnan(fold_threshold):
                meta_thresholds.append(float(fold_threshold))
            sharpe_improved.append(filtered_metrics["Sharpe"] > unfiltered_metrics["Sharpe"])
            expectancy_improved.append(
                filtered_metrics["Expectancy"] > unfiltered_metrics["Expectancy"]
            )
            drawdown_improved.append(
                filtered_metrics["MaxDrawdown"] > unfiltered_metrics["MaxDrawdown"]
            )
            test_event_types = filter_type_used.loc[X_test_events.index].astype(str)
            dominant_filter_type = (
                test_event_types.mode().iloc[0] if not test_event_types.empty else "global"
            )
            filter_type_breakdown = (
                test_event_types.value_counts(normalize=True).astype(float).to_dict()
                if not test_event_types.empty
                else {"global": 1.0}
            )
        else:
            dominant_filter_type = "global"
            filter_type_breakdown = {"global": 1.0}

        regime_return_breakdown: dict[str, float] = {}
        regime_time_pct: dict[str, float] = {}
        if regime_column and regime_column in test_df.columns:
            regime_values = test_df[regime_column].fillna("UNKNOWN").astype(str)
            regime_return_breakdown = (
                unfiltered_bt.returns.groupby(regime_values).sum().astype(float).to_dict()
            )
            regime_time_pct = regime_values.value_counts(normalize=True).astype(float).to_dict()

        fold_row = {
            "fold_start": train_df["timestamp"].iloc[0],
            "fold_train_end": train_df["timestamp"].iloc[-1],
            "fold_test_start": test_df["timestamp"].iloc[0],
            "fold_test_end": test_df["timestamp"].iloc[-1],
            "best_params": best_params,
            "train_objective": float(best[objective_metric]),
            "test_regime_return_breakdown": json.dumps(regime_return_breakdown, sort_keys=True),
            "test_regime_time_pct": json.dumps(regime_time_pct, sort_keys=True),
            "meta_filter_rate": float(fold_filter_rate),
            "meta_realized_filter_rate": float(fold_filter_rate),
            "meta_threshold": None if np.isnan(fold_threshold) else float(fold_threshold),
            "meta_threshold_selected": None if np.isnan(fold_threshold) else float(fold_threshold),
            "meta_threshold_clipped": bool(threshold_clipped),
            "meta_target_filter_rate_band": json.dumps(target_filter_rate_band),
            "meta_target_filter_rate_midpoint": float(target_filter_rate_midpoint),
            "meta_train_realized_filter_rate": None
            if np.isnan(train_realized_filter_rate)
            else float(train_realized_filter_rate),
            "meta_score_distribution_summary": json.dumps(score_distribution_summary, sort_keys=True),
            "meta_filter_rate_error": float(fold_filter_rate - target_filter_rate_midpoint),
            "meta_state": meta_state_json,
            "meta_label_method": meta_label_method,
            "meta_filter_type": dominant_filter_type,
            "meta_filter_type_breakdown": json.dumps(filter_type_breakdown, sort_keys=True),
            **{f"test_{k}": v for k, v in unfiltered_metrics.items()},
        }
        if filtered_metrics is not None:
            fold_row.update(
                {
                    "test_Sharpe_unfiltered": float(unfiltered_metrics["Sharpe"]),
                    "test_Sharpe_filtered": float(filtered_metrics["Sharpe"]),
                    "test_CAGR_unfiltered": float(unfiltered_metrics["CAGR"]),
                    "test_CAGR_filtered": float(filtered_metrics["CAGR"]),
                    "test_MaxDrawdown_unfiltered": float(unfiltered_metrics["MaxDrawdown"]),
                    "test_MaxDrawdown_filtered": float(filtered_metrics["MaxDrawdown"]),
                    "test_Expectancy_unfiltered": float(unfiltered_metrics["Expectancy"]),
                    "test_Expectancy_filtered": float(filtered_metrics["Expectancy"]),
                }
            )
        folds.append(fold_row)

        stitched_returns.append(unfiltered_bt.returns.to_numpy())
        stitched_positions.append(unfiltered_bt.position.to_numpy())
        stitched_index.extend(test_df["timestamp"].values.tolist())
        if not unfiltered_bt.trades.empty:
            aggregated_trades.append(unfiltered_bt.trades.copy())

        if filtered_bt is not None:
            filtered_stitched_returns.append(filtered_bt.returns.to_numpy())
            filtered_stitched_positions.append(filtered_bt.position.to_numpy())
            if not filtered_bt.trades.empty:
                filtered_aggregated_trades.append(filtered_bt.trades.copy())
        i += test_bars

    if not stitched_returns:
        raise ValueError("Not enough data for one train/test fold.")

    combined_returns = pd.Series(
        data=np.concatenate(stitched_returns),
        index=pd.to_datetime(stitched_index, utc=True),
        name="returns",
    ).sort_index()
    combined_position = pd.Series(
        data=np.concatenate(stitched_positions),
        index=pd.to_datetime(stitched_index, utc=True),
        name="position",
    ).sort_index()
    combined_equity = (1.0 + combined_returns).cumprod() * 100_000.0
    combined_drawdown = (combined_equity / combined_equity.cummax()) - 1.0

    combined_trades = (
        pd.concat(aggregated_trades, ignore_index=True)
        if aggregated_trades
        else pd.DataFrame()
    )
    aggregate_metrics = compute_metrics(
        combined_returns,
        combined_equity,
        combined_trades,
        timeframe=timeframe,
        position=combined_position,
    )

    filtered_combined_returns = None
    filtered_combined_equity = None
    filtered_combined_drawdown = None
    filtered_aggregate_metrics = None
    meta_filter_diagnostics = None

    if use_meta and filtered_stitched_returns:
        filtered_combined_returns = pd.Series(
            data=np.concatenate(filtered_stitched_returns),
            index=pd.to_datetime(stitched_index, utc=True),
            name="filtered_returns",
        ).sort_index()
        filtered_combined_position = pd.Series(
            data=np.concatenate(filtered_stitched_positions),
            index=pd.to_datetime(stitched_index, utc=True),
            name="filtered_position",
        ).sort_index()
        filtered_combined_equity = (1.0 + filtered_combined_returns).cumprod() * 100_000.0
        filtered_combined_drawdown = (
            filtered_combined_equity / filtered_combined_equity.cummax()
        ) - 1.0
        filtered_trades = (
            pd.concat(filtered_aggregated_trades, ignore_index=True)
            if filtered_aggregated_trades
            else pd.DataFrame()
        )
        filtered_aggregate_metrics = compute_metrics(
            filtered_combined_returns,
            filtered_combined_equity,
            filtered_trades,
            timeframe=timeframe,
            position=filtered_combined_position,
        )
        meta_filter_diagnostics = {
            "AvgFilterRateByFold": float(np.mean(meta_filter_rates)) if meta_filter_rates else 0.0,
            "StdFilterRateByFold": float(np.std(meta_filter_rates)) if meta_filter_rates else 0.0,
            "AvgMetaThresholdByFold": float(np.mean(meta_thresholds)) if meta_thresholds else 0.0,
            "StdMetaThresholdByFold": float(np.std(meta_thresholds)) if meta_thresholds else 0.0,
            "PctFoldsFilteredSharpeImproved": float(np.mean(sharpe_improved)) if sharpe_improved else 0.0,
            "PctFoldsFilteredExpectancyImproved": float(np.mean(expectancy_improved))
            if expectancy_improved
            else 0.0,
            "PctFoldsFilteredDrawdownImproved": float(np.mean(drawdown_improved))
            if drawdown_improved
            else 0.0,
            "FoldCount": float(len(meta_filter_rates)),
            "ExpectancyDelta": float(
                filtered_aggregate_metrics["Expectancy"] - aggregate_metrics["Expectancy"]
            ),
            "AvgSignalStrengthFiltered": float(np.mean(avg_signal_strength_filtered))
            if avg_signal_strength_filtered
            else 0.0,
            "AvgSignalStrengthUnfiltered": float(np.mean(avg_signal_strength_unfiltered))
            if avg_signal_strength_unfiltered
            else 0.0,
            "LabelMethod": meta_label_method,
            "AvgFilterRateError": float(np.mean(filter_rate_errors)) if filter_rate_errors else 0.0,
            "StdFilterRateError": float(np.std(filter_rate_errors)) if filter_rate_errors else 0.0,
            "PctThresholdClipped": float(np.mean(threshold_clipped_flags))
            if threshold_clipped_flags
            else 0.0,
        }

    return WalkForwardResult(
        fold_results=pd.DataFrame(folds),
        combined_returns=combined_returns,
        combined_equity=combined_equity,
        combined_drawdown=combined_drawdown,
        aggregate_metrics=aggregate_metrics,
        filtered_combined_returns=filtered_combined_returns,
        filtered_combined_equity=filtered_combined_equity,
        filtered_combined_drawdown=filtered_combined_drawdown,
        filtered_aggregate_metrics=filtered_aggregate_metrics,
        meta_filter_diagnostics=meta_filter_diagnostics,
    )
