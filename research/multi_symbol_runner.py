from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from data import (
    CostModel,
    add_basic_indicators,
    attach_costs,
    load_real_fx_csv,
    normalize_fx_dataframe,
    resolve_symbol_cost_model,
)
from data.real_loader import build_data_quality_flags, estimate_missing_bars, infer_timeframe_from_series
from metalabel import (
    RuleBasedMetaFilter,
    apply_meta_trade_filter,
    build_trade_meta_features,
    create_trade_success_labels,
)
from orchestrators import RegimeSpecialistOrchestrator
from regime import attach_regime_labels, attach_stable_regime_state
from research.walk_forward import WalkForwardResult, run_walk_forward
from strategies.mean_reversion import rsi_reversal_signals
from strategies.trend import ma_crossover_signals


@dataclass
class SymbolRunArtifacts:
    symbol: str
    timeframe: str
    wf_result: WalkForwardResult
    bar_count: int


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def _orchestrator_regime_map() -> dict[str, str]:
    return {
        "TRENDING_LOW_VOL": "trend_sleeve",
        "TRENDING_MID_VOL": "trend_sleeve",
        "RANGING_LOW_VOL": "mean_reversion_sleeve",
        "RANGING_MID_VOL": "mean_reversion_sleeve",
    }


def _orchestrated_strategy(frame: pd.DataFrame, params: dict) -> pd.Series:
    ma_local = ma_crossover_signals(
        frame,
        {"fast": int(params["ma_fast"]), "slow": int(params["ma_slow"])},
    ).astype(float)
    rsi_local = rsi_reversal_signals(
        frame,
        {
            "rsi_col": "rsi_14",
            "oversold": float(params["rsi_oversold"]),
            "overbought": float(params["rsi_overbought"]),
            "exit_level": float(params["rsi_exit"]),
        },
    ).astype(float)
    orchestrator = RegimeSpecialistOrchestrator(
        regime_column="regime_label",
        stable_regime_column="stable_regime_label",
        use_stable_regime=True,
        vol_regime_column="vol_regime",
        regime_to_sleeve=_orchestrator_regime_map(),
        fallback="flat",
        sleeve_weights={"trend_sleeve": 1.0, "mean_reversion_sleeve": 1.0},
        switch_cooldown_bars=12,
        switch_penalty_bps=0.0,
        allow_high_vol_entries=False,
        use_vol_targeting=True,
        target_atr_norm=0.001,
        max_leverage=1.0,
    )
    return orchestrator.orchestrate(
        frame,
        sleeve_signals={"trend_sleeve": ma_local, "mean_reversion_sleeve": rsi_local},
    )


def _prepare_symbol_dataframe(
    filepath: str | Path,
    symbol: str,
    timeframe: str,
    timezone: str | None,
    column_map: dict[str, str] | None,
    cost_model: CostModel,
    regime_params: dict[str, Any],
    stable_params: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    raw = load_real_fx_csv(
        filepath=filepath,
        symbol=symbol,
        column_map=column_map,
        timezone=timezone,
    )
    audit = dict(raw.attrs.get("ingestion_audit", {}))
    normalized = normalize_fx_dataframe(raw, symbol=symbol, timeframe=timeframe)
    detected_tf = infer_timeframe_from_series(normalized["timestamp"])
    spread_source = "csv" if normalized["spread_bps"].notna().any() else "static_default"

    df = add_basic_indicators(normalized)
    df = attach_regime_labels(df, **regime_params)
    df = attach_stable_regime_state(df, **stable_params)
    df = attach_costs(df, cost_model)
    df = df.dropna(
        subset=[
            "ma_fast_20",
            "ma_slow_50",
            "rsi_14",
            "atr_14",
            "adx_14",
            "bb_upper_20_2",
            "bb_lower_20_2",
            "stable_regime_label",
        ]
    ).reset_index(drop=True)

    quality = build_data_quality_flags(normalized, timeframe=detected_tf)
    quality["symbol"] = symbol

    audit_row = {
        "symbol": symbol,
        "source_file": str(filepath),
        "row_count": int(audit.get("row_count_clean", len(normalized))),
        "start_timestamp": str(normalized["timestamp"].min()) if not normalized.empty else "",
        "end_timestamp": str(normalized["timestamp"].max()) if not normalized.empty else "",
        "detected_timeframe": detected_tf,
        "duplicate_rows_removed": int(audit.get("duplicate_rows_removed", 0)),
        "missing_bars_estimate": int(estimate_missing_bars(normalized, timeframe=detected_tf)),
        "columns_found": ",".join(audit.get("columns_found", [])),
        "spread_source_used": spread_source,
    }
    return df, audit_row, quality


def _plot_multi_symbol_equity(
    artifacts: list[SymbolRunArtifacts],
    output_path: Path,
) -> None:
    if not artifacts:
        return
    n = len(artifacts)
    fig, axes = plt.subplots(n, 1, figsize=(12, max(4, 3.2 * n)), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, art in zip(axes, artifacts, strict=False):
        base = art.wf_result.combined_equity
        filt = art.wf_result.filtered_combined_equity
        if base is None or filt is None:
            continue
        b = base / float(base.iloc[0]) if float(base.iloc[0]) != 0 else base
        f = filt / float(filt.iloc[0]) if float(filt.iloc[0]) != 0 else filt
        ax.plot(base.index, b.values, label=f"{art.symbol} Unfiltered")
        ax.plot(filt.index, f.values, label=f"{art.symbol} Filtered")
        ax.set_ylabel("Normalized Equity")
        ax.set_title(f"{art.symbol} ({art.timeframe})")
        ax.legend(loc="best")

    axes[-1].set_xlabel("Timestamp")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_multi_symbol_evaluation(
    data_sources_config: str | Path = "configs/data_sources.yaml",
    symbols_config: str | Path = "configs/symbols.yaml",
    output_dir: str | Path = "outputs",
) -> dict[str, pd.DataFrame]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    sources_cfg = _load_yaml(data_sources_config)
    symbols_cfg = _load_yaml(symbols_config)

    defaults = sources_cfg.get("defaults", {})
    providers = sources_cfg.get("providers", {})
    symbols_list = symbols_cfg.get("symbols", [])
    if not symbols_list:
        raise ValueError("No symbols configured in symbols.yaml")

    default_timeframe = str(defaults.get("timeframe", "H1"))
    train_bars = int(defaults.get("train_bars", 800))
    test_bars = int(defaults.get("test_bars", 200))
    timezone = defaults.get("timezone", "UTC")

    default_cost = CostModel(
        spread_bps=float(defaults.get("cost_model", {}).get("spread_bps", 0.8)),
        slippage_bps=float(defaults.get("cost_model", {}).get("slippage_bps", 0.5)),
        commission_bps=float(defaults.get("cost_model", {}).get("commission_bps", 0.3)),
    )
    cost_overrides = defaults.get("cost_overrides", {})
    regime_params = {"adx_threshold": 25.0}
    stable_params = {
        "raw_regime_col": "regime_label",
        "adx_col": "adx_14",
        "atr_norm_col": "atr_norm",
        "atr_norm_pct_col": "atr_norm_pct_rank",
        "enter_trending": 28.0,
        "exit_trending": 22.0,
        "min_regime_bars": 12,
        "confirm_bars": 6,
    }
    meta_cfg = defaults.get("meta_filter", {})
    meta_target_filter_rate = float(meta_cfg.get("target_filter_rate", 0.4))
    meta_horizon = int(meta_cfg.get("horizon_bars", 24))
    meta_threshold = float(meta_cfg.get("success_threshold", 0.0002))
    meta_min_train_samples = int(meta_cfg.get("min_train_samples", 30))

    orchestrated_grid = {
        "ma_fast": [10, 20],
        "ma_slow": [50, 80],
        "rsi_oversold": [30, 35],
        "rsi_overbought": [65, 70],
        "rsi_exit": [50],
    }

    summary_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    meta_diag_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    artifacts: list[SymbolRunArtifacts] = []

    for entry in symbols_list:
        symbol = str(entry["symbol"])
        filepath = Path(entry["filepath"])
        provider = str(entry.get("provider", "generic_ohlc"))
        provider_map = providers.get(provider, {}).get("column_map", {})
        entry_column_map = entry.get("column_map", {})
        merged_column_map = {**provider_map, **entry_column_map}
        timeframe = str(entry.get("timeframe", default_timeframe))

        symbol_cost = resolve_symbol_cost_model(default_cost, symbol, overrides=cost_overrides)

        try:
            df, audit, quality = _prepare_symbol_dataframe(
                filepath=filepath,
                symbol=symbol,
                timeframe=timeframe,
                timezone=timezone,
                column_map=merged_column_map if merged_column_map else None,
                cost_model=symbol_cost,
                regime_params=regime_params,
                stable_params=stable_params,
            )
            audit_rows.append(audit)
            quality_rows.append(quality)

            if len(df) < (train_bars + test_bars + 10):
                summary_rows.append(
                    {
                        "symbol": symbol,
                        "bar_count": int(len(df)),
                        "timeframe": timeframe,
                        "status": "insufficient_data",
                    }
                )
                continue

            wf = run_walk_forward(
                df=df,
                strategy_fn=_orchestrated_strategy,
                param_grid=orchestrated_grid,
                train_bars=train_bars,
                test_bars=test_bars,
                cost_model=symbol_cost,
                timeframe=timeframe,
                regime_column="stable_regime_label",
                meta_filter_class=RuleBasedMetaFilter,
                meta_filter_kwargs={"target_filter_rate": meta_target_filter_rate},
                meta_feature_builder=build_trade_meta_features,
                meta_label_builder=create_trade_success_labels,
                meta_label_kwargs={
                    "horizon_bars": meta_horizon,
                    "success_threshold": meta_threshold,
                },
                meta_apply_fn=apply_meta_trade_filter,
                meta_min_train_samples=meta_min_train_samples,
            )
            if wf.filtered_aggregate_metrics is None:
                raise RuntimeError(f"Meta filter did not produce filtered metrics for {symbol}")

            diag = wf.meta_filter_diagnostics or {}
            summary_rows.append(
                {
                    "symbol": symbol,
                    "bar_count": int(len(df)),
                    "timeframe": timeframe,
                    "unfiltered_oos_sharpe": float(wf.aggregate_metrics["Sharpe"]),
                    "filtered_oos_sharpe": float(wf.filtered_aggregate_metrics["Sharpe"]),
                    "unfiltered_oos_cagr": float(wf.aggregate_metrics["CAGR"]),
                    "filtered_oos_cagr": float(wf.filtered_aggregate_metrics["CAGR"]),
                    "unfiltered_oos_maxdd": float(wf.aggregate_metrics["MaxDrawdown"]),
                    "filtered_oos_maxdd": float(wf.filtered_aggregate_metrics["MaxDrawdown"]),
                    "unfiltered_oos_expectancy": float(wf.aggregate_metrics["Expectancy"]),
                    "filtered_oos_expectancy": float(wf.filtered_aggregate_metrics["Expectancy"]),
                    "avg_filter_rate": float(diag.get("AvgFilterRateByFold", 0.0)),
                    "pct_folds_filtered_sharpe_gt_unfiltered": float(
                        diag.get("PctFoldsFilteredSharpeImproved", 0.0)
                    ),
                    "pct_folds_filtered_expectancy_gt_unfiltered": float(
                        diag.get("PctFoldsFilteredExpectancyImproved", 0.0)
                    ),
                    "pct_folds_filtered_drawdown_improved": float(
                        diag.get("PctFoldsFilteredDrawdownImproved", 0.0)
                    ),
                }
            )
            meta_diag_rows.append(
                {
                    "symbol": symbol,
                    "average_filter_rate": float(diag.get("AvgFilterRateByFold", 0.0)),
                    "filter_rate_std": float(diag.get("StdFilterRateByFold", 0.0)),
                    "average_meta_threshold": float(diag.get("AvgMetaThresholdByFold", 0.0)),
                    "threshold_stability": float(diag.get("StdMetaThresholdByFold", 0.0)),
                    "fold_count": float(diag.get("FoldCount", 0.0)),
                }
            )

            fold_df = wf.fold_results.copy()
            fold_df = fold_df.reset_index(drop=True)
            for fold_idx, row in fold_df.iterrows():
                fold_rows.append(
                    {
                        "symbol": symbol,
                        "fold": int(fold_idx),
                        "train_start": row.get("fold_start"),
                        "train_end": row.get("fold_train_end"),
                        "test_start": row.get("fold_test_start"),
                        "test_end": row.get("fold_test_end"),
                        "unfiltered_sharpe": row.get("test_Sharpe_unfiltered"),
                        "filtered_sharpe": row.get("test_Sharpe_filtered"),
                        "unfiltered_expectancy": row.get("test_Expectancy_unfiltered"),
                        "filtered_expectancy": row.get("test_Expectancy_filtered"),
                        "unfiltered_maxdd": row.get("test_MaxDrawdown_unfiltered"),
                        "filtered_maxdd": row.get("test_MaxDrawdown_filtered"),
                        "filter_rate": row.get("meta_filter_rate", 0.0),
                    }
                )

            artifacts.append(SymbolRunArtifacts(symbol=symbol, timeframe=timeframe, wf_result=wf, bar_count=len(df)))
        except Exception as exc:  # noqa: BLE001
            summary_rows.append(
                {
                    "symbol": symbol,
                    "bar_count": 0,
                    "timeframe": timeframe,
                    "status": f"error: {exc}",
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    folds_df = pd.DataFrame(fold_rows)
    meta_diag_df = pd.DataFrame(meta_diag_rows)
    audit_df = pd.DataFrame(audit_rows)
    quality_df = pd.DataFrame(quality_rows)

    summary_df.to_csv(output_path / "multi_symbol_summary.csv", index=False)
    folds_df.to_csv(output_path / "multi_symbol_fold_summary.csv", index=False)
    meta_diag_df.to_csv(output_path / "multi_symbol_meta_diagnostics.csv", index=False)
    audit_df.to_csv(output_path / "data_ingestion_audit.csv", index=False)
    quality_df.to_csv(output_path / "data_quality_flags.csv", index=False)
    _plot_multi_symbol_equity(artifacts, output_path / "multi_symbol_equity_comparison.png")

    return {
        "multi_symbol_summary": summary_df,
        "multi_symbol_fold_summary": folds_df,
        "multi_symbol_meta_diagnostics": meta_diag_df,
        "data_ingestion_audit": audit_df,
        "data_quality_flags": quality_df,
    }
