from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import pandas as pd

from data import CostModel, resolve_symbol_cost_model
from metalabel import (
    RuleBasedMetaFilter,
    apply_meta_trade_filter,
    build_trade_meta_features,
    create_trade_success_labels,
)
from research.purged_walk_forward import run_purged_walk_forward
from research.v2_runner import _build_v2_orchestrated_signal, _load_yaml, _prepare_symbol_dataframe
from research.walk_forward import run_walk_forward
from strategies.mean_reversion import rsi_reversal_signals
from strategies.mean_reversion_confirmed import mean_reversion_confirmed_signals
from strategies.trend import ma_crossover_signals
from strategies.trend_breakout import trend_breakout_signals


@dataclass
class CandidateEvalResult:
    symbol: str
    sleeve: str
    combined_equity: pd.Series
    filtered_equity: pd.Series
    summary: dict[str, Any]


def select_top_candidates(
    ranking_df: pd.DataFrame,
    top_n: int = 3,
    include_meanrev_if_high: bool = True,
) -> pd.DataFrame:
    if ranking_df.empty:
        return ranking_df.copy()
    ranked = ranking_df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    selected = ranked.head(top_n).copy()
    if include_meanrev_if_high:
        meanrev_top = ranked[ranked["sleeve"] == "MeanRevConfirmed_V2"].head(1)
        if not meanrev_top.empty:
            meanrev_row = meanrev_top.iloc[0]
            already = (
                (selected["symbol"] == meanrev_row["symbol"])
                & (selected["sleeve"] == meanrev_row["sleeve"])
            ).any()
            if not already and int(meanrev_top.index[0]) <= 4 and len(selected) > 0:
                selected = pd.concat([selected.iloc[:-1], meanrev_top], ignore_index=True)
                selected = selected.sort_values("composite_score", ascending=False).reset_index(drop=True)
    selected["candidate_rank"] = selected.index + 1
    return selected


def _strategy_fn_and_grid(sleeve: str) -> tuple[Callable[[pd.DataFrame, dict], pd.Series], dict[str, list]]:
    sleeve = str(sleeve)
    if sleeve == "MA_Baseline":
        return (
            lambda frame, p: ma_crossover_signals(
                frame,
                {"fast": int(p.get("fast", 20)), "slow": int(p.get("slow", 50))},
            ).astype(float),
            {"fast": [10, 20], "slow": [50, 80]},
        )
    if sleeve == "RSI_Baseline":
        return (
            lambda frame, p: rsi_reversal_signals(
                frame,
                {
                    "oversold": float(p.get("oversold", 30)),
                    "overbought": float(p.get("overbought", 70)),
                    "exit_level": float(p.get("exit_level", 50)),
                },
            ).astype(float),
            {"oversold": [28, 30, 32], "overbought": [68, 70, 72], "exit_level": [50]},
        )
    if sleeve == "TrendBreakout_V2":
        return (
            lambda frame, p: trend_breakout_signals(
                frame,
                {
                    "donchian_lookback": int(p.get("donchian_lookback", 20)),
                    "atr_breakout_mult": float(p.get("atr_breakout_mult", 1.2)),
                    "trailing_stop_atr_mult": float(p.get("trailing_stop_atr_mult", 2.0)),
                },
            ).astype(float),
            {
                "donchian_lookback": [20, 30],
                "atr_breakout_mult": [1.0, 1.2],
                "trailing_stop_atr_mult": [1.5, 2.0],
            },
        )
    if sleeve == "MeanRevConfirmed_V2":
        return (
            lambda frame, p: mean_reversion_confirmed_signals(
                frame,
                {
                    "long_entry_rsi": float(p.get("long_entry_rsi", 30)),
                    "short_entry_rsi": float(p.get("short_entry_rsi", 70)),
                    "require_bb_confirmation": True,
                    "exit_mode": p.get("exit_mode", "mean_touch"),
                },
            ).astype(float),
            {"long_entry_rsi": [28, 30, 32], "short_entry_rsi": [68, 70, 72], "exit_mode": ["mean_touch"]},
        )
    # Default to orchestrated V2.
    return (
        lambda frame, p: _build_v2_orchestrated_signal(frame, p).astype(float),
        {
            "donchian_lookback": [20, 30],
            "atr_breakout_mult": [1.0, 1.2],
            "long_entry_rsi": [28, 32],
            "short_entry_rsi": [68, 72],
            "trailing_stop_atr_mult": [2.0],
            "exit_mode": ["mean_touch"],
        },
    )


def _plot_candidate_equity(results: list[CandidateEvalResult], output_path: Path) -> None:
    if not results:
        return
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(12, max(4, n * 3.2)), sharex=False)
    if n == 1:
        axes = [axes]
    for ax, res in zip(axes, results, strict=False):
        base = res.combined_equity
        filt = res.filtered_equity
        b = base / float(base.iloc[0]) if float(base.iloc[0]) != 0 else base
        f = filt / float(filt.iloc[0]) if float(filt.iloc[0]) != 0 else filt
        ax.plot(base.index, b.values, label=f"{res.symbol}-{res.sleeve} Unfiltered")
        ax.plot(filt.index, f.values, label=f"{res.symbol}-{res.sleeve} Filtered")
        ax.legend(loc="best")
        ax.set_ylabel("Normalized Equity")
    axes[-1].set_xlabel("Timestamp")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_candidate_validation(
    candidates_df: pd.DataFrame,
    data_sources_config: str | Path = "configs/data_sources.yaml",
    symbols_config: str | Path = "configs/symbols.yaml",
    output_dir: str | Path = "outputs",
    longer_start: str | None = None,
    longer_end: str | None = None,
    use_purged: bool = False,
    purge_bars: int = 0,
    embargo_bars: int = 0,
) -> pd.DataFrame:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if candidates_df.empty:
        empty = pd.DataFrame()
        empty.to_csv(out_dir / "v21_candidate_validation.csv", index=False)
        return empty

    sources_cfg = _load_yaml(data_sources_config)
    symbols_cfg = _load_yaml(symbols_config)
    defaults = sources_cfg.get("defaults", {})
    providers = sources_cfg.get("providers", {})
    symbols_list = symbols_cfg.get("symbols", [])
    symbol_cfg = {str(s["symbol"]): s for s in symbols_list}

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

    label_method = str(meta_cfg.get("method", "top_quantile"))
    quantile = float(meta_cfg.get("quantile", 0.3))
    horizon = int(meta_cfg.get("forward_horizon", 24))
    cost_threshold = meta_cfg.get("cost_threshold", None)
    min_train_samples = int(meta_cfg.get("min_train_samples", 30))
    target_filter_rate = float(meta_cfg.get("target_filter_rate", 0.4))

    rows: list[dict[str, Any]] = []
    equity_results: list[CandidateEvalResult] = []

    for _, cand in candidates_df.iterrows():
        symbol = str(cand["symbol"])
        sleeve = str(cand["sleeve"])
        cfg = symbol_cfg.get(symbol)
        if cfg is None:
            continue
        provider = str(cfg.get("provider", "generic_ohlc"))
        filepath = cfg["filepath"]
        provider_map = providers.get(provider, {}).get("column_map", {}) or {}
        entry_map = cfg.get("column_map", {}) or {}
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
        if longer_start:
            df = df[pd.to_datetime(df["timestamp"], utc=True) >= pd.Timestamp(longer_start, tz="UTC")]
        if longer_end:
            df = df[pd.to_datetime(df["timestamp"], utc=True) <= pd.Timestamp(longer_end, tz="UTC")]
        df = df.reset_index(drop=True)
        if len(df) < (train_bars + test_bars + 20):
            continue

        strategy_fn, param_grid = _strategy_fn_and_grid(sleeve)
        if use_purged:
            wf = run_purged_walk_forward(
                df=df,
                strategy_fn=strategy_fn,
                param_grid=param_grid,
                train_bars=train_bars,
                test_bars=test_bars,
                cost_model=symbol_cost,
                purge_bars=purge_bars,
                embargo_bars=embargo_bars,
                timeframe=timeframe,
                regime_column="stable_regime_label",
                meta_filter_class=RuleBasedMetaFilter,
                meta_filter_kwargs={"target_filter_rate": target_filter_rate},
                meta_feature_builder=build_trade_meta_features,
                meta_label_builder=create_trade_success_labels,
                meta_label_kwargs={
                    "method": label_method,
                    "quantile": quantile,
                    "forward_horizon": horizon,
                    "cost_threshold": cost_threshold,
                },
                meta_apply_fn=apply_meta_trade_filter,
                meta_min_train_samples=min_train_samples,
            )
        else:
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
                meta_filter_kwargs={"target_filter_rate": target_filter_rate},
                meta_feature_builder=build_trade_meta_features,
                meta_label_builder=create_trade_success_labels,
                meta_label_kwargs={
                    "method": label_method,
                    "quantile": quantile,
                    "forward_horizon": horizon,
                    "cost_threshold": cost_threshold,
                },
                meta_apply_fn=apply_meta_trade_filter,
                meta_min_train_samples=min_train_samples,
            )
        if wf.filtered_aggregate_metrics is None or wf.filtered_combined_equity is None:
            continue
        diag = wf.meta_filter_diagnostics or {}
        rows.append(
            {
                "symbol": symbol,
                "sleeve": sleeve,
                "candidate_rank": int(cand.get("candidate_rank", 0)),
                "label_method": label_method,
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
                "fold_count": float(diag.get("FoldCount", 0.0)),
                "use_purged": bool(use_purged),
                "purge_bars": int(purge_bars),
                "embargo_bars": int(embargo_bars),
                "longer_start": longer_start or "",
                "longer_end": longer_end or "",
            }
        )
        equity_results.append(
            CandidateEvalResult(
                symbol=symbol,
                sleeve=sleeve,
                combined_equity=wf.combined_equity,
                filtered_equity=wf.filtered_combined_equity,
                summary=rows[-1],
            )
        )

    out = pd.DataFrame(rows).sort_values(["candidate_rank", "symbol", "sleeve"]).reset_index(drop=True)
    out.to_csv(out_dir / "v21_candidate_validation.csv", index=False)
    _plot_candidate_equity(equity_results, out_dir / "v21_candidate_equity.png")
    return out
