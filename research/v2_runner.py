from __future__ import annotations

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
from execution.simulator import run_backtest
from metalabel import (
    RuleBasedMetaFilter,
    apply_meta_trade_filter,
    build_trade_meta_features,
    create_trade_success_labels,
    run_feature_ablation,
)
from metrics.performance import compute_metrics
from orchestrators import RegimeSpecialistOrchestrator
from portfolio import V2PortfolioAllocator
from regime import attach_regime_labels, attach_stable_regime_state
from research.stability import feature_stability_report, threshold_stability_report
from research.walk_forward import run_walk_forward
from strategies.mean_reversion import rsi_reversal_signals
from strategies.mean_reversion_confirmed import mean_reversion_confirmed_signals
from strategies.trend import ma_crossover_signals
from strategies.trend_breakout import trend_breakout_signals


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be mapping: {path}")
    return data


def _orchestrator_regime_map() -> dict[str, str]:
    return {
        "TRENDING_LOW_VOL": "trend_sleeve",
        "TRENDING_MID_VOL": "trend_sleeve",
        "RANGING_LOW_VOL": "mean_reversion_sleeve",
        "RANGING_MID_VOL": "mean_reversion_sleeve",
    }


def _prepare_symbol_dataframe(
    filepath: str | Path,
    symbol: str,
    timeframe: str,
    timezone: str | None,
    column_map: dict[str, str] | None,
    cost_model: CostModel,
) -> pd.DataFrame:
    raw = load_real_fx_csv(
        filepath=filepath,
        symbol=symbol,
        column_map=column_map,
        timezone=timezone,
    )
    normalized = normalize_fx_dataframe(raw, symbol=symbol, timeframe=timeframe)
    df = add_basic_indicators(normalized)
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
    return df


def _build_v2_orchestrated_signal(df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    trend_sig = trend_breakout_signals(
        df,
        {
            "donchian_lookback": int(params.get("donchian_lookback", 20)),
            "atr_breakout_mult": float(params.get("atr_breakout_mult", 1.2)),
            "combine_mode": params.get("combine_mode", "either"),
            "trailing_stop_atr_mult": params.get("trailing_stop_atr_mult", 2.0),
        },
    ).astype(float)
    mr_sig = mean_reversion_confirmed_signals(
        df,
        {
            "long_entry_rsi": float(params.get("long_entry_rsi", 30)),
            "short_entry_rsi": float(params.get("short_entry_rsi", 70)),
            "require_bb_confirmation": bool(params.get("require_bb_confirmation", True)),
            "exit_mode": params.get("exit_mode", "mean_touch"),
            "fixed_horizon_bars": int(params.get("fixed_horizon_bars", 12)),
            "time_stop_bars": int(params.get("time_stop_bars", 24)),
        },
    ).astype(float)
    orch = RegimeSpecialistOrchestrator(
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
    return orch.orchestrate(
        df,
        sleeve_signals={"trend_sleeve": trend_sig, "mean_reversion_sleeve": mr_sig},
    ).astype(float)


def _save_portfolio_equity(curves: dict[str, pd.Series], output_path: Path) -> None:
    if not curves:
        return
    plt.figure(figsize=(12, 6))
    for name, eq in curves.items():
        if eq.empty:
            continue
        norm = eq / float(eq.iloc[0]) if float(eq.iloc[0]) != 0 else eq
        plt.plot(eq.index, norm.values, label=name)
    plt.title("V2 Portfolio Equity by Allocation Mode")
    plt.xlabel("Timestamp")
    plt.ylabel("Normalized Equity")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_v2_evaluation(
    data_sources_config: str | Path = "configs/data_sources.yaml",
    symbols_config: str | Path = "configs/symbols.yaml",
    output_dir: str | Path = "outputs",
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
    target_filter_rate = float(meta_cfg.get("target_filter_rate", 0.4))
    min_train_samples = int(meta_cfg.get("min_train_samples", 30))

    v2_grid = {
        "donchian_lookback": [20, 30],
        "atr_breakout_mult": [1.0, 1.2],
        "long_entry_rsi": [28, 32],
        "short_entry_rsi": [68, 72],
        "trailing_stop_atr_mult": [2.0],
        "exit_mode": ["mean_touch"],
    }

    sleeve_rows: list[dict[str, Any]] = []
    trade_quality_rows: list[dict[str, Any]] = []
    ablation_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    portfolio_rows: list[dict[str, Any]] = []

    signal_matrix: list[pd.Series] = []
    return_matrix: list[pd.Series] = []
    expectancy_scores: dict[str, float] = {}

    for entry in symbols_list:
        symbol = str(entry["symbol"])
        provider = str(entry.get("provider", "generic_ohlc"))
        filepath = entry["filepath"]
        provider_map = providers.get(provider, {}).get("column_map", {}) or {}
        entry_map = entry.get("column_map", {}) or {}
        column_map = {**provider_map, **entry_map}
        symbol_cost = resolve_symbol_cost_model(default_cost, symbol, overrides=cost_overrides)

        df = _prepare_symbol_dataframe(
            filepath=filepath,
            symbol=symbol,
            timeframe=timeframe,
            timezone=timezone,
            column_map=column_map if column_map else None,
            cost_model=symbol_cost,
        )
        if len(df) < (train_bars + test_bars + 20):
            continue

        # Sleeve comparison (full-period deterministic backtest).
        ma_sig = ma_crossover_signals(df, {"fast": 20, "slow": 50})
        rsi_sig = rsi_reversal_signals(df, {"oversold": 30, "overbought": 70, "exit_level": 50})
        trend_v2_sig = trend_breakout_signals(df, {"donchian_lookback": 20, "atr_breakout_mult": 1.2, "trailing_stop_atr_mult": 2.0})
        meanrev_v2_sig = mean_reversion_confirmed_signals(
            df,
            {"long_entry_rsi": 30, "short_entry_rsi": 70, "require_bb_confirmation": True, "exit_mode": "mean_touch"},
        )
        orch_v2_sig = _build_v2_orchestrated_signal(df, {"donchian_lookback": 20, "atr_breakout_mult": 1.2, "long_entry_rsi": 30, "short_entry_rsi": 70})

        sleeve_map = {
            "MA_Baseline": ma_sig,
            "RSI_Baseline": rsi_sig,
            "TrendBreakout_V2": trend_v2_sig,
            "MeanRevConfirmed_V2": meanrev_v2_sig,
            "Orchestrated_V2": orch_v2_sig,
        }
        for sleeve_name, sig in sleeve_map.items():
            bt = run_backtest(df, sig.astype(float), symbol_cost)
            metrics = compute_metrics(
                bt.returns,
                bt.equity,
                bt.trades,
                timeframe=timeframe,
                position=bt.position,
            )
            sleeve_rows.append(
                {
                    "symbol": symbol,
                    "sleeve": sleeve_name,
                    **metrics,
                }
            )

        # Strict WF trade-quality comparison on orchestrated V2 sleeves.
        wf = run_walk_forward(
            df=df,
            strategy_fn=lambda frame, p: _build_v2_orchestrated_signal(frame, p),
            param_grid=v2_grid,
            train_bars=train_bars,
            test_bars=test_bars,
            cost_model=symbol_cost,
            timeframe=timeframe,
            regime_column="stable_regime_label",
            meta_filter_class=RuleBasedMetaFilter,
            meta_filter_kwargs={"target_filter_rate": target_filter_rate},
            meta_feature_builder=build_trade_meta_features,
            meta_label_builder=create_trade_success_labels,
            meta_label_kwargs={
                "method": meta_method,
                "quantile": meta_quantile,
                "forward_horizon": meta_horizon,
                "cost_threshold": meta_cost_threshold,
            },
            meta_apply_fn=apply_meta_trade_filter,
            meta_min_train_samples=min_train_samples,
        )
        if wf.filtered_aggregate_metrics is None:
            continue

        diag = wf.meta_filter_diagnostics or {}
        trade_quality_rows.append(
            {
                "symbol": symbol,
                "label_method": meta_method,
                "unfiltered_sharpe": float(wf.aggregate_metrics["Sharpe"]),
                "filtered_sharpe": float(wf.filtered_aggregate_metrics["Sharpe"]),
                "unfiltered_cagr": float(wf.aggregate_metrics["CAGR"]),
                "filtered_cagr": float(wf.filtered_aggregate_metrics["CAGR"]),
                "unfiltered_maxdd": float(wf.aggregate_metrics["MaxDrawdown"]),
                "filtered_maxdd": float(wf.filtered_aggregate_metrics["MaxDrawdown"]),
                "unfiltered_expectancy": float(wf.aggregate_metrics["Expectancy"]),
                "filtered_expectancy": float(wf.filtered_aggregate_metrics["Expectancy"]),
                "expectancy_delta": float(
                    wf.filtered_aggregate_metrics["Expectancy"] - wf.aggregate_metrics["Expectancy"]
                ),
                "avg_filter_rate": float(diag.get("AvgFilterRateByFold", 0.0)),
            }
        )
        expectancy_scores[symbol] = float(wf.filtered_aggregate_metrics["Expectancy"])

        th = threshold_stability_report(wf.fold_results)
        feat = feature_stability_report(wf.fold_results)
        top_feat = feat.iloc[0]["feature"] if not feat.empty else ""
        top_feat_abs = float(feat.iloc[0]["abs_mean"]) if not feat.empty else 0.0
        threshold_rows.append(
            {
                "symbol": symbol,
                "count": int(th.iloc[0]["count"]),
                "mean": float(th.iloc[0]["mean"]),
                "std": float(th.iloc[0]["std"]),
                "cv": float(th.iloc[0]["cv"]),
                "min": float(th.iloc[0]["min"]),
                "max": float(th.iloc[0]["max"]),
                "top_feature": top_feat,
                "top_feature_abs_mean": top_feat_abs,
            }
        )

        # Feature ablation on full-data orchestrator signal.
        abl = run_feature_ablation(
            df=df,
            primary_signal=orch_v2_sig.astype(float),
            label_kwargs={
                "method": meta_method,
                "quantile": meta_quantile,
                "forward_horizon": meta_horizon,
                "cost_threshold": meta_cost_threshold,
            },
            meta_filter_kwargs={"target_filter_rate": target_filter_rate},
            min_train_samples=min_train_samples,
        )
        abl.insert(0, "symbol", symbol)
        ablation_rows.extend(abl.to_dict(orient="records"))

        # Data for portfolio layer.
        ts = pd.to_datetime(df["timestamp"], utc=True)
        signal_matrix.append(pd.Series(orch_v2_sig.values, index=ts, name=symbol))
        bt_port = run_backtest(df, orch_v2_sig.astype(float), symbol_cost)
        return_matrix.append(pd.Series(bt_port.returns.values, index=ts, name=symbol))

    sleeve_df = pd.DataFrame(sleeve_rows)
    tq_df = pd.DataFrame(trade_quality_rows)
    abl_df = pd.DataFrame(ablation_rows)
    th_df = pd.DataFrame(threshold_rows)

    signal_df = pd.concat(signal_matrix, axis=1).sort_index() if signal_matrix else pd.DataFrame()
    returns_df = pd.concat(return_matrix, axis=1).sort_index() if return_matrix else pd.DataFrame()

    curves: dict[str, pd.Series] = {}
    if not signal_df.empty and not returns_df.empty:
        scores = pd.Series(expectancy_scores, dtype=float)
        for mode in ["equal_weight", "inverse_volatility", "expectancy_score"]:
            allocator = V2PortfolioAllocator(
                mode=mode,  # type: ignore[arg-type]
                max_symbol_exposure=0.35,
                gross_exposure_cap=1.0,
                rebalance_rule="1D",
                flat_on_missing_data=True,
            )
            alloc = allocator.allocate(
                signal_frame=signal_df,
                returns_frame=returns_df,
                expectancy_scores=scores if mode == "expectancy_score" else None,
            )
            weights = alloc.weights.reindex(returns_df.index).fillna(0.0)
            port_ret = (weights.shift(1).fillna(0.0) * returns_df.fillna(0.0)).sum(axis=1)
            equity = (1.0 + port_ret).cumprod() * 100_000.0
            curves[mode] = equity
            max_dd = float(((equity / equity.cummax()) - 1.0).min()) if not equity.empty else 0.0
            years = max(len(port_ret.dropna()) / (24 * 252), 1e-9)
            cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0 if len(equity) > 1 else 0.0
            sharpe = 0.0
            if port_ret.std(ddof=0) > 1e-12:
                sharpe = float((port_ret.mean() / port_ret.std(ddof=0)) * (24 * 252) ** 0.5)
            portfolio_rows.append(
                {
                    "mode": mode,
                    "CAGR": float(cagr),
                    "Sharpe": float(sharpe),
                    "MaxDrawdown": float(max_dd),
                    "MeanBarReturn": float(port_ret.mean()),
                    "GrossExposureAvg": float(weights.abs().sum(axis=1).mean()),
                }
            )

    portfolio_df = pd.DataFrame(portfolio_rows)
    _save_portfolio_equity(curves, out_dir / "v2_portfolio_equity.png")

    sleeve_df.to_csv(out_dir / "v2_sleeve_comparison.csv", index=False)
    tq_df.to_csv(out_dir / "v2_trade_quality_comparison.csv", index=False)
    abl_df.to_csv(out_dir / "v2_feature_ablation.csv", index=False)
    th_df.to_csv(out_dir / "v2_threshold_stability.csv", index=False)
    portfolio_df.to_csv(out_dir / "v2_portfolio_summary.csv", index=False)

    return {
        "v2_sleeve_comparison": sleeve_df,
        "v2_trade_quality_comparison": tq_df,
        "v2_feature_ablation": abl_df,
        "v2_threshold_stability": th_df,
        "v2_portfolio_summary": portfolio_df,
    }
