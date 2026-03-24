from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import (
    CostModel,
    add_basic_indicators,
    attach_costs,
    ensure_mock_ohlcv_csv,
    load_ohlcv_csv,
    load_symbol_data,
)
from execution.simulator import run_backtest
from metrics.performance import compute_metrics
from research.parameter_robustness import (
    analyze_parameter_robustness,
)
from research.purged_walk_forward import run_purged_walk_forward
from research.walk_forward import run_walk_forward
from regime import attach_regime_labels, attach_stable_regime_state
from strategies import trend_breakout_v2_signals


@dataclass
class StrategyResearchArtifacts:
    summary: pd.DataFrame
    top_params: pd.DataFrame
    fold_results: pd.DataFrame
    robustness: pd.DataFrame
    component_ablation: pd.DataFrame
    recommendation: pd.DataFrame
    output_dir: Path


def _trend_breakout_param_space() -> dict[str, list]:
    return {
        # Entry
        "lookback": [18, 20, 24, 30],
        "velocity_lookback": [4, 6, 8],
        "velocity_threshold": [0.8, 1.0, 1.2, 1.4],
        "confirmation_bars": [1, 2, 3],
        "vol_compression_max_pct": [0.30, 0.40, 0.50],
        "breakout_strength_atr_mult": [0.15, 0.22, 0.30, 0.40],
        "retest_entry_mode": [False, True],
        # Exit
        "trailing_stop_atr_mult": [1.4, 1.8, 2.2],
        "max_holding_bars": [48, 72, 96],
        "vol_exit_pct_rank_threshold": [0.15, 0.22, 0.30],
        "partial_take_profit_rr": [0.0, 1.0, 1.3, 1.6],
        "partial_take_profit_size": [0.25, 0.35, 0.50],
        "winner_extension_enabled": [False, True],
        "extension_trigger_atr_multiple": [1.8, 2.0, 2.5],
        "extension_stop_multiplier": [1.8, 2.4, 3.0],
        "extension_max_holding_bars": [120, 160, 240],
        # Trade management
        "min_bars_between_trades": [20, 24, 30],
        "dynamic_cooldown_by_vol": [False, True],
        "high_vol_cooldown_mult": [1.2, 1.5, 2.0],
        # Keep these stable but explicit
        "expansion_lookback": [12],
        "expansion_threshold": [1.02, 1.10],
        "vol_contraction_exit_mult": [0.80],
        "vol_contraction_window": [20],
        "winner_extension_enabled": [False, True],
    }


def _sample_param_candidates(
    param_space: dict[str, list],
    search_method: str = "random",
    n_samples: int = 120,
    seed: int = 42,
) -> list[dict]:
    keys = list(param_space.keys())
    if search_method.lower() == "grid":
        combos = [dict(zip(keys, vals, strict=False)) for vals in __import__("itertools").product(*[param_space[k] for k in keys])]
        return combos

    rng = np.random.default_rng(seed)
    seen: set[tuple] = set()
    out: list[dict] = []
    max_attempts = n_samples * 20
    attempts = 0
    while len(out) < n_samples and attempts < max_attempts:
        attempts += 1
        p = {k: rng.choice(param_space[k]) for k in keys}
        sig = tuple((k, p[k]) for k in keys)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(p)
    return out


def _to_builtin(obj):
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def _jsonify_params(params_obj) -> str:
    if isinstance(params_obj, str):
        return params_obj
    if isinstance(params_obj, dict):
        return json.dumps(_to_builtin(params_obj), sort_keys=True)
    return json.dumps(_to_builtin(params_obj), sort_keys=True)


def _flatten_params(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    params_df = pd.json_normalize(df["params"])
    params_df.columns = [str(c) for c in params_df.columns]
    return pd.concat([df.reset_index(drop=True), params_df.reset_index(drop=True)], axis=1)


def _prepare_frame(
    symbol: str,
    timeframe: str,
    source_csv: str | Path,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    ensure_mock_ohlcv_csv(source_csv, symbols=[symbol], periods=10_000, freq="1h", seed=73)
    raw = load_ohlcv_csv(source_csv)
    df = load_symbol_data(raw, symbol=symbol, timeframe=timeframe)
    if start_date is not None:
        df = df[df["timestamp"] >= pd.Timestamp(start_date, tz="UTC")]
    if end_date is not None:
        df = df[df["timestamp"] <= pd.Timestamp(end_date, tz="UTC")]
    df = df.reset_index(drop=True)
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
    df = attach_costs(df, CostModel(spread_bps=0.8, slippage_bps=0.5, commission_bps=0.2))
    req = [
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
    ]
    return df.dropna(subset=req).reset_index(drop=True)


def _evaluate_param_candidate(
    df: pd.DataFrame,
    strategy_fn: Callable[[pd.DataFrame, dict], pd.Series],
    params: dict,
    timeframe: str,
    cost_model: CostModel,
    train_bars: int,
    test_bars: int,
    use_purge: bool,
    purge_bars: int,
    embargo_bars: int,
) -> dict:
    def _fixed_strategy(frame: pd.DataFrame, _: dict) -> pd.Series:
        return strategy_fn(frame, params).astype(float)

    param_grid = {"_single": [0]}
    if use_purge:
        wf = run_purged_walk_forward(
            df=df,
            strategy_fn=_fixed_strategy,
            param_grid=param_grid,
            train_bars=train_bars,
            test_bars=test_bars,
            cost_model=cost_model,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
            timeframe=timeframe,
        )
        fold_df = wf.fold_results.copy()
        agg = wf.aggregate_metrics
        combined_equity = wf.combined_equity
        combined_returns = wf.combined_returns
        combined_drawdown = wf.combined_drawdown
    else:
        wf = run_walk_forward(
            df=df,
            strategy_fn=_fixed_strategy,
            param_grid=param_grid,
            train_bars=train_bars,
            test_bars=test_bars,
            cost_model=cost_model,
            timeframe=timeframe,
        )
        fold_df = wf.fold_results.copy()
        agg = wf.aggregate_metrics
        combined_equity = wf.combined_equity
        combined_returns = wf.combined_returns
        combined_drawdown = wf.combined_drawdown

    fold_expectancy = pd.to_numeric(fold_df.get("test_Expectancy", pd.Series(dtype=float)), errors="coerce")
    fold_sharpe = pd.to_numeric(fold_df.get("test_Sharpe", pd.Series(dtype=float)), errors="coerce")
    pos_fold_rate = float((fold_expectancy > 0).mean()) if not fold_expectancy.empty else 0.0
    exp_std = float(fold_expectancy.std(ddof=0)) if len(fold_expectancy) > 1 else 0.0
    sharpe_std = float(fold_sharpe.std(ddof=0)) if len(fold_sharpe) > 1 else 0.0
    instability_penalty = float(exp_std + 0.25 * sharpe_std)
    dd_penalty = abs(float(agg.get("MaxDrawdown", 0.0)))

    row = {
        "params": params,
        "OOS_Expectancy": float(agg.get("Expectancy", 0.0)),
        "OOS_Sharpe": float(agg.get("Sharpe", 0.0)),
        "OOS_MaxDrawdown": float(agg.get("MaxDrawdown", 0.0)),
        "OOS_TradeCount": float(agg.get("TradeCount", 0.0)),
        "OOS_CAGR": float(agg.get("CAGR", 0.0)),
        "FoldCount": int(len(fold_df)),
        "PctPositiveFolds": pos_fold_rate,
        "ExpectancyStdByFold": exp_std,
        "SharpeStdByFold": sharpe_std,
        "InstabilityPenalty": instability_penalty,
        "DrawdownPenaltyAbs": dd_penalty,
        "robust_score": float(
            (1_000.0 * float(agg.get("Expectancy", 0.0)))
            + (0.35 * float(agg.get("Sharpe", 0.0)))
            + (0.30 * pos_fold_rate)
            - (2.0 * dd_penalty)
            - (0.60 * instability_penalty)
        ),
        "combined_equity": combined_equity,
        "combined_returns": combined_returns,
        "combined_drawdown": combined_drawdown,
        "fold_results": fold_df,
    }
    return row


def _component_ablation(
    base_params: dict,
    df: pd.DataFrame,
    strategy_fn: Callable[[pd.DataFrame, dict], pd.Series],
    timeframe: str,
    cost_model: CostModel,
    train_bars: int,
    test_bars: int,
    use_purge: bool,
    purge_bars: int,
    embargo_bars: int,
) -> pd.DataFrame:
    tests: list[tuple[str, dict]] = [
        ("base", base_params),
        ("retest_entry_off", {**base_params, "retest_entry_mode": False}),
        ("retest_entry_on", {**base_params, "retest_entry_mode": True}),
        ("winner_extension_off", {**base_params, "winner_extension_enabled": False}),
        ("winner_extension_on", {**base_params, "winner_extension_enabled": True}),
        ("partial_tp_off", {**base_params, "partial_take_profit_rr": 0.0}),
        ("partial_tp_on", {**base_params, "partial_take_profit_rr": max(1.0, float(base_params.get("partial_take_profit_rr", 1.2)))}),
        ("contraction_exit_loose", {**base_params, "vol_exit_pct_rank_threshold": 0.10}),
        ("contraction_exit_tight", {**base_params, "vol_exit_pct_rank_threshold": 0.30}),
        ("dynamic_cooldown_off", {**base_params, "dynamic_cooldown_by_vol": False}),
        ("dynamic_cooldown_on", {**base_params, "dynamic_cooldown_by_vol": True}),
    ]
    rows: list[dict] = []
    for name, p in tests:
        r = _evaluate_param_candidate(
            df=df,
            strategy_fn=strategy_fn,
            params=p,
            timeframe=timeframe,
            cost_model=cost_model,
            train_bars=train_bars,
            test_bars=test_bars,
            use_purge=use_purge,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
        )
        rows.append(
            {
                "component_test": name,
                "OOS_Expectancy": r["OOS_Expectancy"],
                "OOS_Sharpe": r["OOS_Sharpe"],
                "OOS_MaxDrawdown": r["OOS_MaxDrawdown"],
                "OOS_TradeCount": r["OOS_TradeCount"],
                "PctPositiveFolds": r["PctPositiveFolds"],
                "robust_score": r["robust_score"],
            }
        )
    out = pd.DataFrame(rows)
    base_row = out[out["component_test"] == "base"].iloc[0]
    for col in ["OOS_Expectancy", "OOS_Sharpe", "OOS_MaxDrawdown", "robust_score"]:
        out[f"delta_{col}"] = out[col] - float(base_row[col])
    return out.sort_values("robust_score", ascending=False).reset_index(drop=True)


def _plot_research_equity(equity: pd.Series, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(equity.index, equity.values, label="Best Candidate OOS Equity")
    ax.set_title("Strategy Research OOS Equity")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Equity")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_research_heatmaps(results: pd.DataFrame, output_path: Path) -> None:
    # Keep the chart robust even when parameter names differ.
    if results.empty:
        return
    params_df = pd.json_normalize(results["params"])
    full = pd.concat([results.reset_index(drop=True), params_df], axis=1)
    # Avoid duplicated column names between score table and normalized params.
    full = full.loc[:, ~full.columns.duplicated(keep="last")]
    candidates = [c for c in ["lookback", "velocity_threshold", "breakout_strength_atr_mult"] if c in full.columns]
    if len(candidates) < 2:
        return
    x_col, y_col = candidates[0], candidates[1]
    pivot = full.pivot_table(index=y_col, columns=x_col, values="OOS_Expectancy", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(v) for v in pivot.columns.tolist()])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(v) for v in pivot.index.tolist()])
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title("OOS Expectancy Heatmap")
    fig.colorbar(im, ax=ax, label="OOS Expectancy")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _build_recommendation(
    ranked: pd.DataFrame,
    robust: pd.DataFrame,
) -> pd.DataFrame:
    if ranked.empty:
        return pd.DataFrame()
    best_peak = ranked.sort_values("OOS_Expectancy", ascending=False).iloc[0]
    best_robust = robust.sort_values("robustness_score", ascending=False).iloc[0]
    hardened = ranked.sort_values("robust_score", ascending=False).iloc[0]
    return pd.DataFrame(
        [
            {
                "candidate_type": "BEST_PEAK",
                "params": best_peak["params"],
                "OOS_Expectancy": best_peak["OOS_Expectancy"],
                "OOS_Sharpe": best_peak["OOS_Sharpe"],
                "OOS_MaxDrawdown": best_peak["OOS_MaxDrawdown"],
                "robust_score": best_peak["robust_score"],
            },
            {
                "candidate_type": "BEST_ROBUST",
                "params": best_robust["params"],
                "OOS_Expectancy": best_robust["OOS_Expectancy"],
                "OOS_Sharpe": best_robust["OOS_Sharpe"],
                "OOS_MaxDrawdown": best_robust["OOS_MaxDrawdown"],
                "robust_score": best_robust["robust_score"],
            },
            {
                "candidate_type": "HARDENED_DEFAULT",
                "params": hardened["params"],
                "OOS_Expectancy": hardened["OOS_Expectancy"],
                "OOS_Sharpe": hardened["OOS_Sharpe"],
                "OOS_MaxDrawdown": hardened["OOS_MaxDrawdown"],
                "robust_score": hardened["robust_score"],
            },
        ]
    )


def run_strategy_research(
    strategy_family: str = "TrendBreakout_V2",
    symbol: str = "EURUSD",
    timeframe: str = "H1",
    start: str | None = None,
    end: str | None = None,
    use_purge: bool | None = None,
    purge_bars: int = 0,
    embargo_bars: int = 0,
    search_mode: str = "random",
    n_random_samples: int = 120,
    train_bars: int = 2200,
    test_bars: int = 500,
    output_dir: str | Path = "outputs",
    source_csv: str | Path = "outputs/strategy_research_mock_ohlcv.csv",
    seed: int = 42,
) -> StrategyResearchArtifacts:
    if strategy_family != "TrendBreakout_V2":
        raise ValueError("Phase R1 currently supports strategy_family='TrendBreakout_V2' only.")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if use_purge is None:
        use_purge = (purge_bars > 0) or (embargo_bars > 0)

    df = _prepare_frame(
        symbol=symbol,
        timeframe=timeframe,
        source_csv=source_csv,
        start_date=start,
        end_date=end,
    )
    strategy_fn = trend_breakout_v2_signals
    cost_model = CostModel(spread_bps=0.8, slippage_bps=0.5, commission_bps=0.2)
    param_space = _trend_breakout_param_space()
    candidates = _sample_param_candidates(
        param_space=param_space,
        search_method=search_mode,
        n_samples=n_random_samples,
        seed=seed,
    )

    rows: list[dict] = []
    fold_frames: list[pd.DataFrame] = []
    for idx, params in enumerate(candidates):
        res = _evaluate_param_candidate(
            df=df,
            strategy_fn=strategy_fn,
            params=params,
            timeframe=timeframe,
            cost_model=cost_model,
            train_bars=train_bars,
            test_bars=test_bars,
            use_purge=use_purge,
            purge_bars=purge_bars,
            embargo_bars=embargo_bars,
        )
        fold_df = res.pop("fold_results")
        fold_df = fold_df.copy()
        fold_df["candidate_id"] = idx
        fold_df["params"] = str(params)
        fold_frames.append(fold_df)
        rows.append({"candidate_id": idx, **res})

    result_table = pd.DataFrame(rows)
    model_table = _flatten_params(result_table)
    param_cols = [c for c in model_table.columns if c in param_space]
    model_table["oos_expectancy"] = model_table["OOS_Expectancy"]
    model_table["pre_robust_score"] = model_table["robust_score"]
    robustness_art = analyze_parameter_robustness(
        results=model_table,
        param_cols=param_cols,
        objective_col="oos_expectancy",
        pre_score_col="pre_robust_score",
        n_neighbors=min(10, max(3, len(model_table) // 8)),
    )
    robust_candidates = robustness_art.candidate_robustness.copy()
    result_table = robust_candidates.sort_values(
        ["robust_score", "OOS_Expectancy", "OOS_Sharpe"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    result_table["rank"] = np.arange(1, len(result_table) + 1)
    fold_results = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()

    best = result_table.iloc[0]
    best_candidate_id = int(best["candidate_id"])
    best_equity = rows[best_candidate_id]["combined_equity"]

    ablation = _component_ablation(
        base_params=dict(best["params"]),
        df=df,
        strategy_fn=strategy_fn,
        timeframe=timeframe,
        cost_model=cost_model,
        train_bars=train_bars,
        test_bars=test_bars,
        use_purge=use_purge,
        purge_bars=purge_bars,
        embargo_bars=embargo_bars,
    )
    sensitivity = robustness_art.sensitivity.copy()
    false_peaks = robustness_art.false_peaks.copy()
    recommendation = _build_recommendation(result_table, robust_candidates)

    summary = pd.DataFrame(
        [
            {
                "strategy_family": strategy_family,
                "symbol": symbol,
                "timeframe": timeframe,
                "bar_count": int(len(df)),
                "candidate_count": int(len(result_table)),
                "search_method": search_mode,
                "use_purge": bool(use_purge),
                "purge_bars": int(purge_bars),
                "embargo_bars": int(embargo_bars),
                "best_peak_expectancy": float(result_table["OOS_Expectancy"].max()),
                "best_robust_score": float(result_table["robust_score"].max()),
                "best_hardened_expectancy": float(best["OOS_Expectancy"]),
                "best_hardened_sharpe": float(best["OOS_Sharpe"]),
                "best_hardened_max_drawdown": float(best["OOS_MaxDrawdown"]),
            }
        ]
    )

    # Artifacts
    summary.to_csv(out_dir / "strategy_research_summary.csv", index=False)
    result_table.head(25).to_csv(out_dir / "strategy_research_top_params.csv", index=False)
    fold_results.to_csv(out_dir / "strategy_research_fold_results.csv", index=False)
    robustness_export = pd.concat(
        [
            robust_candidates.copy(),
            sensitivity.copy(),
            false_peaks.copy(),
        ],
        ignore_index=True,
        sort=False,
    )
    robustness_export.to_csv(out_dir / "strategy_research_robustness.csv", index=False)
    ablation.to_csv(out_dir / "strategy_research_component_ablation.csv", index=False)
    recommendation.to_csv(out_dir / "strategy_research_recommendation.csv", index=False)
    sensitivity.to_csv(out_dir / "strategy_research_param_sensitivity.csv", index=False)
    false_peaks.to_csv(out_dir / "strategy_research_false_peaks.csv", index=False)

    _plot_research_equity(best_equity, out_dir / "strategy_research_equity.png")
    _plot_research_heatmaps(result_table, out_dir / "strategy_research_heatmaps.png")

    return StrategyResearchArtifacts(
        summary=summary,
        top_params=result_table.head(25).copy(),
        fold_results=fold_results,
        robustness=robustness_export,
        component_ablation=ablation,
        recommendation=recommendation,
        output_dir=out_dir,
    )
