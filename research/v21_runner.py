from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from data import CostModel, resolve_symbol_cost_model
from metalabel import (
    RuleBasedMetaFilter,
    apply_meta_trade_filter,
    build_trade_meta_features,
    create_trade_success_labels,
    run_feature_ablation,
)
from research.candidate_validation import run_candidate_validation, select_top_candidates
from research.feature_pruning import build_feature_pruning_tables
from research.sleeve_ranking import build_sleeve_symbol_ranking, classify_component_decisions
from research.v2_runner import _build_v2_orchestrated_signal, _load_yaml, _prepare_symbol_dataframe
from research.walk_forward import run_walk_forward
from strategies.mean_reversion import rsi_reversal_signals
from strategies.mean_reversion_confirmed import mean_reversion_confirmed_signals
from strategies.trend import ma_crossover_signals
from strategies.trend_breakout import trend_breakout_signals


def _strategy_catalog() -> dict[str, dict[str, Any]]:
    return {
        "MA_Baseline": {
            "fn": lambda frame, p: ma_crossover_signals(
                frame,
                {"fast": int(p.get("fast", 20)), "slow": int(p.get("slow", 50))},
            ).astype(float),
            "grid": {"fast": [10, 20], "slow": [50, 80]},
            "ablation_params": {"fast": 20, "slow": 50},
        },
        "RSI_Baseline": {
            "fn": lambda frame, p: rsi_reversal_signals(
                frame,
                {
                    "oversold": float(p.get("oversold", 30)),
                    "overbought": float(p.get("overbought", 70)),
                    "exit_level": float(p.get("exit_level", 50)),
                },
            ).astype(float),
            "grid": {"oversold": [28, 30, 32], "overbought": [68, 70, 72], "exit_level": [50]},
            "ablation_params": {"oversold": 30, "overbought": 70, "exit_level": 50},
        },
        "TrendBreakout_V2": {
            "fn": lambda frame, p: trend_breakout_signals(
                frame,
                {
                    "donchian_lookback": int(p.get("donchian_lookback", 20)),
                    "atr_breakout_mult": float(p.get("atr_breakout_mult", 1.2)),
                    "trailing_stop_atr_mult": float(p.get("trailing_stop_atr_mult", 2.0)),
                },
            ).astype(float),
            "grid": {
                "donchian_lookback": [20, 30],
                "atr_breakout_mult": [1.0, 1.2],
                "trailing_stop_atr_mult": [1.5, 2.0],
            },
            "ablation_params": {
                "donchian_lookback": 20,
                "atr_breakout_mult": 1.2,
                "trailing_stop_atr_mult": 2.0,
            },
        },
        "MeanRevConfirmed_V2": {
            "fn": lambda frame, p: mean_reversion_confirmed_signals(
                frame,
                {
                    "long_entry_rsi": float(p.get("long_entry_rsi", 30)),
                    "short_entry_rsi": float(p.get("short_entry_rsi", 70)),
                    "require_bb_confirmation": True,
                    "exit_mode": p.get("exit_mode", "mean_touch"),
                },
            ).astype(float),
            "grid": {"long_entry_rsi": [28, 30, 32], "short_entry_rsi": [68, 70, 72], "exit_mode": ["mean_touch"]},
            "ablation_params": {
                "long_entry_rsi": 30,
                "short_entry_rsi": 70,
                "require_bb_confirmation": True,
                "exit_mode": "mean_touch",
            },
        },
        "Orchestrated_V2": {
            "fn": lambda frame, p: _build_v2_orchestrated_signal(frame, p).astype(float),
            "grid": {
                "donchian_lookback": [20, 30],
                "atr_breakout_mult": [1.0, 1.2],
                "long_entry_rsi": [28, 32],
                "short_entry_rsi": [68, 72],
                "trailing_stop_atr_mult": [2.0],
                "exit_mode": ["mean_touch"],
            },
            "ablation_params": {
                "donchian_lookback": 20,
                "atr_breakout_mult": 1.2,
                "long_entry_rsi": 30,
                "short_entry_rsi": 70,
                "trailing_stop_atr_mult": 2.0,
                "exit_mode": "mean_touch",
            },
        },
    }


def _parse_band(raw: Any) -> tuple[float, float]:
    if isinstance(raw, str):
        try:
            val = json.loads(raw)
        except json.JSONDecodeError:
            val = None
    else:
        val = raw
    if isinstance(val, list) and len(val) == 2:
        return float(val[0]), float(val[1])
    return 0.2, 0.6


def run_v21_refinement(
    data_sources_config: str | Path = "configs/data_sources.yaml",
    symbols_config: str | Path = "configs/symbols.yaml",
    output_dir: str | Path = "outputs",
    longer_start: str | None = None,
    longer_end: str | None = None,
    use_purged_candidates: bool | None = None,
) -> dict[str, pd.DataFrame]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sources_cfg = _load_yaml(data_sources_config)
    symbols_cfg = _load_yaml(symbols_config)
    defaults = sources_cfg.get("defaults", {})
    providers = sources_cfg.get("providers", {})
    symbols_list = symbols_cfg.get("symbols", [])
    if not symbols_list:
        raise ValueError("No symbols configured in symbols.yaml")

    v21_cfg = sources_cfg.get("v21", {}) or {}
    candidate_top_n = int(v21_cfg.get("candidate_top_n", 3))
    if longer_start is None:
        longer_start = v21_cfg.get("longer_history_start")
    if longer_end is None:
        longer_end = v21_cfg.get("longer_history_end")
    if use_purged_candidates is None:
        use_purged_candidates = bool(v21_cfg.get("use_purged_candidates", False))
    purge_bars = int(v21_cfg.get("purge_bars", 0))
    embargo_bars = int(v21_cfg.get("embargo_bars", 0))

    timeframe = str(defaults.get("timeframe", "H1"))
    timezone = defaults.get("timezone", "UTC")
    train_bars = int(defaults.get("train_bars", 800))
    test_bars = int(defaults.get("test_bars", 200))
    cost_defaults = defaults.get("cost_model", {})
    default_cost = CostModel(
        spread_bps=float(cost_defaults.get("spread_bps", 0.8)),
        slippage_bps=float(cost_defaults.get("slippage_bps", 0.5)),
        commission_bps=float(cost_defaults.get("commission_bps", 0.3)),
    )
    cost_overrides = defaults.get("cost_overrides", {})

    meta_cfg = defaults.get("meta_filter", {})
    meta_method = str(meta_cfg.get("method", "top_quantile"))
    meta_quantile = float(meta_cfg.get("quantile", 0.3))
    meta_horizon = int(meta_cfg.get("forward_horizon", 24))
    meta_cost_threshold = meta_cfg.get("cost_threshold", None)
    meta_target_rate = float(meta_cfg.get("target_filter_rate", 0.4))
    meta_min_rate = float(meta_cfg.get("min_filter_rate", 0.2))
    meta_max_rate = float(meta_cfg.get("max_filter_rate", 0.6))
    meta_min_train_samples = int(meta_cfg.get("min_train_samples", 30))

    strategy_catalog = _strategy_catalog()
    ranking_rows: list[dict[str, Any]] = []
    filter_diag_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []

    for entry in symbols_list:
        symbol = str(entry["symbol"])
        provider = str(entry.get("provider", "generic_ohlc"))
        provider_map = providers.get(provider, {}).get("column_map", {}) or {}
        entry_map = entry.get("column_map", {}) or {}
        column_map = {**provider_map, **entry_map}
        symbol_cost = resolve_symbol_cost_model(default_cost, symbol, overrides=cost_overrides)
        df = _prepare_symbol_dataframe(
            filepath=entry["filepath"],
            symbol=symbol,
            timeframe=str(entry.get("timeframe", timeframe)),
            timezone=timezone,
            column_map=column_map if column_map else None,
            cost_model=symbol_cost,
        )
        if len(df) < (train_bars + test_bars + 20):
            continue

        for sleeve, spec in strategy_catalog.items():
            strategy_fn: Callable[[pd.DataFrame, dict], pd.Series] = spec["fn"]
            param_grid = spec["grid"]

            wf = run_walk_forward(
                df=df,
                strategy_fn=strategy_fn,
                param_grid=param_grid,
                train_bars=train_bars,
                test_bars=test_bars,
                cost_model=symbol_cost,
                timeframe=timeframe,
                regime_column="stable_regime_label",
                meta_filter_class=RuleBasedMetaFilter,
                meta_filter_kwargs={
                    "target_filter_rate": meta_target_rate,
                    "min_filter_rate": meta_min_rate,
                    "max_filter_rate": meta_max_rate,
                },
                meta_feature_builder=build_trade_meta_features,
                meta_label_builder=create_trade_success_labels,
                meta_label_kwargs={
                    "method": meta_method,
                    "quantile": meta_quantile,
                    "forward_horizon": meta_horizon,
                    "cost_threshold": meta_cost_threshold,
                },
                meta_apply_fn=apply_meta_trade_filter,
                meta_min_train_samples=meta_min_train_samples,
            )
            if wf.filtered_aggregate_metrics is None:
                continue

            folds = wf.fold_results.copy()
            threshold_stability = float(pd.to_numeric(folds["meta_threshold"], errors="coerce").std(ddof=0))
            if pd.isna(threshold_stability):
                threshold_stability = 0.0
            filter_rate_stability = float(pd.to_numeric(folds["meta_filter_rate"], errors="coerce").std(ddof=0))
            if pd.isna(filter_rate_stability):
                filter_rate_stability = 0.0
            exp_win = float(
                (pd.to_numeric(folds["test_Expectancy_filtered"], errors="coerce")
                 > pd.to_numeric(folds["test_Expectancy_unfiltered"], errors="coerce")).mean()
            )
            dd_win = float(
                (pd.to_numeric(folds["test_MaxDrawdown_filtered"], errors="coerce")
                 > pd.to_numeric(folds["test_MaxDrawdown_unfiltered"], errors="coerce")).mean()
            )
            sharpe_win = float(
                (pd.to_numeric(folds["test_Sharpe_filtered"], errors="coerce")
                 > pd.to_numeric(folds["test_Sharpe_unfiltered"], errors="coerce")).mean()
            )
            avg_fr_err = float(pd.to_numeric(folds["meta_filter_rate_error"], errors="coerce").mean())
            avg_fr = float(pd.to_numeric(folds["meta_filter_rate"], errors="coerce").mean())
            out_of_band = 0.0
            if "meta_target_filter_rate_band" in folds.columns:
                rates = pd.to_numeric(folds["meta_filter_rate"], errors="coerce").fillna(0.0)
                lows = []
                highs = []
                for raw in folds["meta_target_filter_rate_band"].tolist():
                    lo, hi = _parse_band(raw)
                    lows.append(lo)
                    highs.append(hi)
                lo_s = pd.Series(lows, index=folds.index, dtype=float)
                hi_s = pd.Series(highs, index=folds.index, dtype=float)
                out_of_band = float(((rates < lo_s) | (rates > hi_s)).mean())
            threshold_clip_pct = float(pd.to_numeric(folds["meta_threshold_clipped"], errors="coerce").fillna(0).mean())

            ranking_rows.append(
                {
                    "symbol": symbol,
                    "sleeve": sleeve,
                    "filtered_expectancy": float(wf.filtered_aggregate_metrics["Expectancy"]),
                    "expectancy_delta": float(
                        wf.filtered_aggregate_metrics["Expectancy"] - wf.aggregate_metrics["Expectancy"]
                    ),
                    "filtered_sharpe": float(wf.filtered_aggregate_metrics["Sharpe"]),
                    "sharpe_delta": float(
                        wf.filtered_aggregate_metrics["Sharpe"] - wf.aggregate_metrics["Sharpe"]
                    ),
                    "filtered_maxdd": float(wf.filtered_aggregate_metrics["MaxDrawdown"]),
                    "maxdd_delta": float(
                        wf.filtered_aggregate_metrics["MaxDrawdown"] - wf.aggregate_metrics["MaxDrawdown"]
                    ),
                    "threshold_stability": threshold_stability,
                    "filter_rate_stability": filter_rate_stability,
                    "expectancy_fold_win_rate": exp_win,
                    "drawdown_fold_win_rate": dd_win,
                    "sharpe_fold_win_rate": sharpe_win,
                    "avg_filter_rate": avg_fr,
                    "avg_filter_rate_error": avg_fr_err,
                    "filter_rate_out_of_band_pct": out_of_band,
                    "threshold_clipped_pct": threshold_clip_pct,
                }
            )

            for fold_idx, fr in folds.reset_index(drop=True).iterrows():
                band = fr.get("meta_target_filter_rate_band")
                lo, hi = _parse_band(band)
                score_summary = fr.get("meta_score_distribution_summary", "{}")
                filter_diag_rows.append(
                    {
                        "symbol": symbol,
                        "sleeve": sleeve,
                        "fold": int(fold_idx),
                        "target_filter_rate_band": json.dumps([lo, hi]),
                        "target_midpoint": float(fr.get("meta_target_filter_rate_midpoint", (lo + hi) / 2.0)),
                        "realized_filter_rate": float(fr.get("meta_filter_rate", 0.0)),
                        "train_realized_filter_rate": fr.get("meta_train_realized_filter_rate"),
                        "threshold_selected": fr.get("meta_threshold_selected"),
                        "threshold_clipped": bool(fr.get("meta_threshold_clipped", False)),
                        "filter_rate_error": float(fr.get("meta_filter_rate_error", 0.0)),
                        "score_distribution_summary": score_summary,
                        "label_method": fr.get("meta_label_method", meta_method),
                        "filter_type": fr.get("meta_filter_type", "global"),
                    }
                )

            # Phase 3: ablation-driven pruning data per symbol+sleeve.
            sig = strategy_fn(df, dict(spec.get("ablation_params", {}))).astype(float)
            abl = run_feature_ablation(
                df=df,
                primary_signal=sig,
                label_kwargs={
                    "method": meta_method,
                    "quantile": meta_quantile,
                    "forward_horizon": meta_horizon,
                    "cost_threshold": meta_cost_threshold,
                },
                meta_filter_kwargs={
                    "target_filter_rate": meta_target_rate,
                    "min_filter_rate": meta_min_rate,
                    "max_filter_rate": meta_max_rate,
                },
                min_train_samples=meta_min_train_samples,
            )
            if not abl.empty:
                abl.insert(0, "sleeve", sleeve)
                abl.insert(0, "symbol", symbol)
                ablation_rows.extend(abl.to_dict(orient="records"))

    filter_df = pd.DataFrame(filter_diag_rows)
    filter_df.to_csv(out_dir / "v21_filter_rate_diagnostics.csv", index=False)

    ranking_input = pd.DataFrame(ranking_rows)
    # Reuse existing V2 sleeve table if present for context columns.
    v2_sleeve_path = out_dir / "v2_sleeve_comparison.csv"
    if v2_sleeve_path.exists() and not ranking_input.empty:
        v2s = pd.read_csv(v2_sleeve_path)
        v2s = v2s.rename(
            columns={
                "CAGR": "raw_cagr",
                "Sharpe": "raw_sharpe",
                "MaxDrawdown": "raw_maxdd",
                "Expectancy": "raw_expectancy",
            }
        )
        keep_cols = [c for c in ["symbol", "sleeve", "raw_cagr", "raw_sharpe", "raw_maxdd", "raw_expectancy"] if c in v2s.columns]
        ranking_input = ranking_input.merge(v2s[keep_cols], on=["symbol", "sleeve"], how="left")

    ranking_df = build_sleeve_symbol_ranking(ranking_input)
    ranking_df.to_csv(out_dir / "v21_sleeve_symbol_ranking.csv", index=False)
    decisions_df = classify_component_decisions(ranking_df)
    decisions_df.to_csv(out_dir / "v21_component_decisions.csv", index=False)

    ablation_df = pd.DataFrame(ablation_rows)
    feature_local_df, feature_global_df = build_feature_pruning_tables(ablation_df)
    feature_local_df.to_csv(out_dir / "v21_feature_pruning.csv", index=False)
    feature_global_df.to_csv(out_dir / "v21_feature_group_summary.csv", index=False)

    selected_candidates = select_top_candidates(ranking_df, top_n=candidate_top_n, include_meanrev_if_high=True)
    candidate_df = run_candidate_validation(
        selected_candidates,
        data_sources_config=data_sources_config,
        symbols_config=symbols_config,
        output_dir=output_dir,
        longer_start=longer_start,
        longer_end=longer_end,
        use_purged=bool(use_purged_candidates),
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
    )

    return {
        "v21_filter_rate_diagnostics": filter_df,
        "v21_sleeve_symbol_ranking": ranking_df,
        "v21_component_decisions": decisions_df,
        "v21_feature_pruning": feature_local_df,
        "v21_feature_group_summary": feature_global_df,
        "v21_candidate_validation": candidate_df,
    }
