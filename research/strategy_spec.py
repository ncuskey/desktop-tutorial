from __future__ import annotations

import ast
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class StrategySpec:
    name: str
    version: str
    symbols: list[str]
    timeframe: str
    data_range: dict[str, str | None]
    metadata: dict[str, Any]
    overview: dict[str, Any]
    core_thesis: str
    entry_logic: dict[str, Any]
    exit_logic: dict[str, Any]
    trade_management: dict[str, Any]
    parameter_set: dict[str, Any]
    performance_summary: dict[str, Any]
    edge_characterization: dict[str, Any]
    robustness_assessment: dict[str, Any]
    failure_modes: dict[str, Any]
    component_insights: dict[str, Any]
    promotion_status: dict[str, Any]
    next_steps: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    if isinstance(value, (np.ndarray, list, tuple)):
        return [_to_native(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_native(v) for k, v in value.items()}
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
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
        else:
            if lv != rv:
                return False
    return True


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    return pd.read_csv(path)


def _load_inputs(output_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "summary": _read_required_csv(output_dir / "strategy_research_summary.csv"),
        "recommendation": _read_required_csv(output_dir / "strategy_research_recommendation.csv"),
        "robustness": _read_required_csv(output_dir / "strategy_research_robustness.csv"),
        "ablation": _read_required_csv(output_dir / "strategy_research_component_ablation.csv"),
        "folds": _read_required_csv(output_dir / "strategy_research_fold_results.csv"),
    }


def _select_summary_row(summary: pd.DataFrame, strategy: str, symbol: str | None) -> pd.Series:
    out = summary.copy()
    if "strategy_family" in out.columns:
        out = out[out["strategy_family"].astype(str) == strategy]
    if symbol and "symbol" in out.columns:
        narrowed = out[out["symbol"].astype(str) == symbol]
        if not narrowed.empty:
            out = narrowed
    if out.empty:
        raise ValueError(
            f"No summary row found for strategy='{strategy}'"
            + (f" symbol='{symbol}'" if symbol else "")
        )
    return out.iloc[0]


def _select_hardened_row(recommendation: pd.DataFrame) -> pd.Series:
    out = recommendation.copy()
    if "candidate_type" not in out.columns:
        raise ValueError("Recommendation artifact missing 'candidate_type' column.")
    hardened = out[out["candidate_type"].astype(str).str.upper() == "HARDENED_DEFAULT"]
    if hardened.empty:
        raise ValueError("No HARDENED_DEFAULT row found in recommendation artifact.")
    return hardened.iloc[0]


def _candidate_rows_from_robustness(robustness: pd.DataFrame) -> pd.DataFrame:
    out = robustness.copy()
    if "section" in out.columns:
        out = out[out["section"].astype(str) == "candidate"]
    if "candidate_id" in out.columns:
        mask = pd.to_numeric(out["candidate_id"], errors="coerce").notna()
        out = out[mask]
    return out


def _infer_hardened_candidate_id(
    robustness: pd.DataFrame,
    hardened_params: dict[str, Any],
) -> int | None:
    candidates = _candidate_rows_from_robustness(robustness)
    if candidates.empty or "params" not in candidates.columns:
        return None
    parsed = candidates["params"].apply(_parse_params)
    matches = parsed.apply(lambda p: _params_equal(p, hardened_params))
    matched_rows = candidates[matches]
    if matched_rows.empty:
        return None
    return int(float(matched_rows.iloc[0]["candidate_id"]))


def _select_fold_rows(
    folds: pd.DataFrame,
    hardened_params: dict[str, Any],
    hardened_candidate_id: int | None,
) -> pd.DataFrame:
    out = folds.copy()
    if "candidate_id" in out.columns and hardened_candidate_id is not None:
        by_id = out[pd.to_numeric(out["candidate_id"], errors="coerce") == hardened_candidate_id]
        if not by_id.empty:
            return by_id
    if "params" in out.columns:
        parsed = out["params"].apply(_parse_params)
        mask = parsed.apply(lambda p: _params_equal(p, hardened_params))
        matched = out[mask]
        if not matched.empty:
            return matched
    return out.iloc[0:0]


def _select_hardened_robustness_row(
    robustness: pd.DataFrame,
    hardened_params: dict[str, Any],
    hardened_candidate_id: int | None,
) -> pd.Series | None:
    candidates = _candidate_rows_from_robustness(robustness)
    if candidates.empty:
        return None
    if hardened_candidate_id is not None and "candidate_id" in candidates.columns:
        by_id = candidates[
            pd.to_numeric(candidates["candidate_id"], errors="coerce") == hardened_candidate_id
        ]
        if not by_id.empty:
            return by_id.iloc[0]
    if "params" in candidates.columns:
        parsed = candidates["params"].apply(_parse_params)
        matches = parsed.apply(lambda p: _params_equal(p, hardened_params))
        by_params = candidates[matches]
        if not by_params.empty:
            return by_params.iloc[0]
    return None


def _format_param_groups(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    entry_keywords = (
        "lookback",
        "velocity",
        "confirmation",
        "compression",
        "breakout",
        "retest",
        "expansion",
    )
    exit_keywords = (
        "stop",
        "holding",
        "take_profit",
        "extension",
        "vol_exit",
        "contraction",
    )
    trade_keywords = ("cooldown", "min_bars_between_trades")

    entry: dict[str, Any] = {}
    exit_: dict[str, Any] = {}
    trade: dict[str, Any] = {}
    leftovers: dict[str, Any] = {}

    for key, value in params.items():
        key_l = key.lower()
        if any(word in key_l for word in trade_keywords):
            trade[key] = value
        elif any(word in key_l for word in exit_keywords):
            exit_[key] = value
        elif any(word in key_l for word in entry_keywords):
            entry[key] = value
        else:
            leftovers[key] = value

    # Keep uncategorized controls visible in trade-management bucket.
    trade.update(leftovers)
    return entry, exit_, trade


def _parse_timestamp_series(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    return parsed.dropna()


def _fold_stability_stats(fold_rows: pd.DataFrame) -> dict[str, Any]:
    if fold_rows.empty:
        return {
            "fold_count": 0,
            "positive_expectancy_pct": 0.0,
            "positive_sharpe_pct": 0.0,
            "drawdown_improved_folds_pct": 0.0,
            "expectancy_std": 0.0,
            "sharpe_std": 0.0,
            "zero_trade_fold_pct": 0.0,
            "avg_trade_count_per_fold": 0.0,
            "worst_folds": [],
        }

    expectancy = pd.to_numeric(fold_rows.get("test_Expectancy"), errors="coerce").fillna(0.0)
    sharpe = pd.to_numeric(fold_rows.get("test_Sharpe"), errors="coerce").fillna(0.0)
    drawdown = pd.to_numeric(fold_rows.get("test_MaxDrawdown"), errors="coerce").fillna(0.0)
    trade_count = pd.to_numeric(fold_rows.get("test_TradeCount"), errors="coerce").fillna(0.0)

    worst = fold_rows.copy()
    worst["test_Expectancy"] = expectancy
    worst["test_Sharpe"] = sharpe
    worst["test_MaxDrawdown"] = drawdown
    worst = worst.sort_values(["test_Expectancy", "test_Sharpe"], ascending=[True, True]).head(3)
    worst_folds: list[dict[str, Any]] = []
    for _, row in worst.iterrows():
        worst_folds.append(
            {
                "test_start": str(row.get("fold_test_start") or row.get("test_start") or ""),
                "test_end": str(row.get("fold_test_end") or row.get("test_end") or ""),
                "expectancy": _safe_float(row.get("test_Expectancy")),
                "sharpe": _safe_float(row.get("test_Sharpe")),
                "max_drawdown": _safe_float(row.get("test_MaxDrawdown")),
            }
        )

    return {
        "fold_count": int(len(fold_rows)),
        "positive_expectancy_pct": float((expectancy > 0).mean()),
        "positive_sharpe_pct": float((sharpe > 0).mean()),
        "drawdown_improved_folds_pct": float((drawdown > -0.005).mean()),
        "expectancy_std": float(expectancy.std(ddof=0)),
        "sharpe_std": float(sharpe.std(ddof=0)),
        "zero_trade_fold_pct": float((trade_count <= 0).mean()),
        "avg_trade_count_per_fold": float(trade_count.mean()),
        "worst_folds": worst_folds,
    }


def _extract_data_range(
    fold_rows: pd.DataFrame,
    summary_row: pd.Series,
) -> dict[str, str | None]:
    train_start_col = "fold_start" if "fold_start" in fold_rows.columns else "train_start"
    test_end_col = "fold_test_end" if "fold_test_end" in fold_rows.columns else "test_end"

    if fold_rows.empty or train_start_col not in fold_rows.columns or test_end_col not in fold_rows.columns:
        return {"start": None, "end": None, "bar_count": int(_safe_float(summary_row.get("bar_count")))}

    start_ts = _parse_timestamp_series(fold_rows[train_start_col])
    end_ts = _parse_timestamp_series(fold_rows[test_end_col])
    start = start_ts.min().isoformat() if not start_ts.empty else None
    end = end_ts.max().isoformat() if not end_ts.empty else None
    return {"start": start, "end": end, "bar_count": int(_safe_float(summary_row.get("bar_count")))}


def _top_sensitivity_rows(robustness: pd.DataFrame, top_n: int = 5) -> list[dict[str, Any]]:
    if "section" not in robustness.columns:
        return []
    sens = robustness[robustness["section"].astype(str) == "sensitivity"].copy()
    if sens.empty:
        return []
    if "abs_spearman_corr" in sens.columns:
        sens["abs_spearman_corr"] = pd.to_numeric(sens["abs_spearman_corr"], errors="coerce").fillna(0.0)
        sens = sens.sort_values("abs_spearman_corr", ascending=False)
    rows: list[dict[str, Any]] = []
    for _, row in sens.head(top_n).iterrows():
        rows.append(
            {
                "parameter": str(row.get("parameter")),
                "abs_spearman_corr": _safe_float(row.get("abs_spearman_corr")),
                "spearman_corr_with_expectancy": _safe_float(row.get("spearman_corr_with_expectancy")),
                "grouped_expectancy_spread": _safe_float(row.get("grouped_expectancy_spread")),
            }
        )
    return rows


def _false_peak_summary(robustness: pd.DataFrame) -> dict[str, Any]:
    if "section" not in robustness.columns:
        return {"count": 0, "top_examples": []}
    peaks = robustness[robustness["section"].astype(str) == "false_peak"].copy()
    if peaks.empty:
        return {"count": 0, "top_examples": []}
    peaks["oos_expectancy"] = pd.to_numeric(peaks.get("oos_expectancy"), errors="coerce").fillna(0.0)
    peaks = peaks.sort_values("oos_expectancy", ascending=False)
    examples: list[dict[str, Any]] = []
    for _, row in peaks.head(3).iterrows():
        examples.append(
            {
                "candidate_id": int(_safe_float(row.get("candidate_id"), default=-1)),
                "oos_expectancy": _safe_float(row.get("oos_expectancy")),
                "robustness_score": _safe_float(row.get("robustness_score")),
            }
        )
    return {"count": int(len(peaks)), "top_examples": examples}


def _component_insights(ablation: pd.DataFrame) -> dict[str, Any]:
    out = ablation.copy()
    if out.empty:
        return {"positive_components": [], "negative_components": [], "neutral_components": []}

    for col in ("delta_OOS_Expectancy", "delta_robust_score", "delta_OOS_Sharpe", "delta_OOS_MaxDrawdown"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
        else:
            out[col] = 0.0

    if "component_test" not in out.columns:
        out["component_test"] = ""
    out["component_test"] = out["component_test"].astype(str)
    non_base = out[~out["component_test"].str.lower().eq("base")].copy()
    positive = non_base.sort_values(
        ["delta_robust_score", "delta_OOS_Expectancy"], ascending=[False, False]
    ).head(3)
    negative = non_base.sort_values(
        ["delta_robust_score", "delta_OOS_Expectancy"], ascending=[True, True]
    ).head(3)
    neutral = non_base[
        (non_base["delta_robust_score"].abs() < 1e-12)
        & (non_base["delta_OOS_Expectancy"].abs() < 1e-12)
    ].head(5)

    def rows(df: pd.DataFrame) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for _, r in df.iterrows():
            payload.append(
                {
                    "component_test": str(r.get("component_test")),
                    "delta_expectancy": _safe_float(r.get("delta_OOS_Expectancy")),
                    "delta_sharpe": _safe_float(r.get("delta_OOS_Sharpe")),
                    "delta_max_drawdown": _safe_float(r.get("delta_OOS_MaxDrawdown")),
                    "delta_robust_score": _safe_float(r.get("delta_robust_score")),
                }
            )
        return payload

    return {
        "positive_components": rows(positive),
        "negative_components": rows(negative),
        "neutral_components": rows(neutral),
    }


def _promotion_decision(
    perf: dict[str, Any],
    stability: dict[str, Any],
    robustness: dict[str, Any],
) -> dict[str, Any]:
    expectancy = _safe_float(perf.get("oos_expectancy"))
    sharpe = _safe_float(perf.get("oos_sharpe"))
    drawdown = abs(_safe_float(perf.get("oos_max_drawdown")))
    robust_score = _safe_float(robustness.get("robustness_score"))
    pos_expectancy = _safe_float(stability.get("positive_expectancy_pct"))

    rationale: list[str] = []
    if expectancy > 0:
        rationale.append("Positive stitched OOS expectancy.")
    else:
        rationale.append("Non-positive stitched OOS expectancy.")
    if sharpe > 0:
        rationale.append("Positive stitched OOS Sharpe.")
    else:
        rationale.append("Non-positive stitched OOS Sharpe.")
    if pos_expectancy >= 0.5:
        rationale.append("Majority of folds show positive expectancy.")
    else:
        rationale.append("Less than half of folds show positive expectancy.")
    if drawdown <= 0.01:
        rationale.append("Contained OOS max drawdown under 1%.")
    else:
        rationale.append("OOS max drawdown above preferred 1% threshold.")
    if robust_score > 0:
        rationale.append("Robustness score is positive.")
    else:
        rationale.append("Robustness score is not positive.")

    if expectancy > 0 and sharpe > 0 and pos_expectancy >= 0.5 and robust_score > 0:
        status = "PROMOTE"
    elif expectancy > 0 and robust_score > 0:
        status = "WATCH"
    else:
        status = "HOLD"

    return {"status": status, "rationale": rationale}


def _next_steps(
    stability: dict[str, Any],
    robustness: dict[str, Any],
    component_insights: dict[str, Any],
) -> list[str]:
    steps: list[str] = []
    if _safe_float(stability.get("positive_expectancy_pct")) < 0.5:
        steps.append("Increase fold count or extend history to improve confidence in fold-level expectancy stability.")
    if _safe_float(stability.get("zero_trade_fold_pct")) > 0.2:
        steps.append("Review entry strictness to reduce no-trade folds while preserving expectancy.")
    if int(_safe_float(robustness.get("false_peak_count"))) > 0:
        steps.append("Densify local parameter search around robust candidates to avoid isolated false peaks.")
    negative = component_insights.get("negative_components", [])
    if negative:
        worst = negative[0].get("component_test", "top_negative_component")
        steps.append(f"Stress-test and potentially disable '{worst}' given negative ablation impact.")
    if not steps:
        steps.append("Proceed to longer-history and multi-symbol confirmation using this hardened default.")
    return steps


def _git_commit_hash(repo_root: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return proc.stdout.strip() or None
    except Exception:
        return None


def build_strategy_spec(
    strategy: str,
    symbol: str | None = None,
    output_dir: str | Path = "outputs",
    version_suffix: str = "R1",
) -> StrategySpec:
    output_path = Path(output_dir)
    inputs = _load_inputs(output_path)
    summary = inputs["summary"]
    recommendation = inputs["recommendation"]
    robustness = inputs["robustness"]
    ablation = inputs["ablation"]
    folds = inputs["folds"]

    summary_row = _select_summary_row(summary, strategy=strategy, symbol=symbol)
    hardened = _select_hardened_row(recommendation)
    hardened_params = _parse_params(hardened.get("params"))

    candidate_id = _infer_hardened_candidate_id(robustness, hardened_params)
    fold_rows = _select_fold_rows(folds, hardened_params, candidate_id)
    robust_row = _select_hardened_robustness_row(robustness, hardened_params, candidate_id)

    entry_params, exit_params, trade_params = _format_param_groups(hardened_params)
    stability = _fold_stability_stats(fold_rows)
    data_range = _extract_data_range(fold_rows, summary_row)
    sensitivity = _top_sensitivity_rows(robustness)
    false_peaks = _false_peak_summary(robustness)
    components = _component_insights(ablation)

    perf = {
        "oos_expectancy": _safe_float(hardened.get("OOS_Expectancy")),
        "oos_sharpe": _safe_float(hardened.get("OOS_Sharpe")),
        "oos_max_drawdown": _safe_float(hardened.get("OOS_MaxDrawdown")),
        "best_peak_expectancy": _safe_float(summary_row.get("best_peak_expectancy")),
        "best_robust_score": _safe_float(summary_row.get("best_robust_score")),
        "best_hardened_expectancy": _safe_float(summary_row.get("best_hardened_expectancy")),
        "best_hardened_sharpe": _safe_float(summary_row.get("best_hardened_sharpe")),
        "best_hardened_max_drawdown": _safe_float(summary_row.get("best_hardened_max_drawdown")),
        "candidate_count": int(_safe_float(summary_row.get("candidate_count"))),
        "search_method": str(summary_row.get("search_method")),
    }

    robust_payload = {
        "robustness_score": _safe_float(robust_row.get("robustness_score") if robust_row is not None else 0.0),
        "robustness_rank": int(
            _safe_float(robust_row.get("robustness_rank") if robust_row is not None else 0.0, default=0)
        ),
        "expectancy_rank": int(
            _safe_float(robust_row.get("expectancy_rank") if robust_row is not None else 0.0, default=0)
        ),
        "parameter_isolation_penalty": _safe_float(
            robust_row.get("parameter_isolation_penalty") if robust_row is not None else 0.0
        ),
        "plateau_bonus": _safe_float(robust_row.get("plateau_bonus") if robust_row is not None else 0.0),
        "neighbor_count": int(
            _safe_float(robust_row.get("neighbor_count") if robust_row is not None else 0.0, default=0)
        ),
        "top_parameter_sensitivity": sensitivity,
        "false_peak_count": false_peaks["count"],
        "false_peak_examples": false_peaks["top_examples"],
    }

    symbols = [str(symbol or summary_row.get("symbol"))]
    timeframe = str(summary_row.get("timeframe"))
    version_id = f"{strategy}_{version_suffix}"
    repo_root = Path(__file__).resolve().parents[1]
    metadata = {
        "strategy_name": strategy,
        "version_id": version_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "commit_hash": _git_commit_hash(repo_root),
        "source_artifacts": {
            "summary": str(output_path / "strategy_research_summary.csv"),
            "recommendation": str(output_path / "strategy_research_recommendation.csv"),
            "robustness": str(output_path / "strategy_research_robustness.csv"),
            "component_ablation": str(output_path / "strategy_research_component_ablation.csv"),
            "fold_results": str(output_path / "strategy_research_fold_results.csv"),
        },
    }

    overview = {
        "strategy": strategy,
        "symbols": symbols,
        "timeframe": timeframe,
        "data_range": data_range,
        "search_method": perf["search_method"],
        "candidate_count": perf["candidate_count"],
        "hardened_candidate_id": candidate_id,
    }

    core_thesis = (
        f"HARDENED_DEFAULT for {strategy} on {', '.join(symbols)} ({timeframe}) targets a reproducible "
        f"positive OOS expectancy of {perf['oos_expectancy']:.6f} with OOS Sharpe {perf['oos_sharpe']:.3f} "
        f"and max drawdown {perf['oos_max_drawdown']:.6f}. Fold-level positive expectancy appears in "
        f"{stability['positive_expectancy_pct']:.1%} of folds, with robustness score "
        f"{robust_payload['robustness_score']:.3f}."
    )

    edge_profile = {
        "expectancy_per_trade": perf["oos_expectancy"],
        "positive_fold_expectancy_pct": stability["positive_expectancy_pct"],
        "positive_fold_sharpe_pct": stability["positive_sharpe_pct"],
        "expectancy_std_by_fold": stability["expectancy_std"],
        "sharpe_std_by_fold": stability["sharpe_std"],
        "avg_trades_per_fold": stability["avg_trade_count_per_fold"],
    }

    failure_modes = {
        "zero_trade_fold_pct": stability["zero_trade_fold_pct"],
        "worst_folds": stability["worst_folds"],
        "drawdown_risk_note": (
            "Fold-level drawdown clusters are visible in the worst-fold sample. "
            "Validate stress periods before promotion."
        ),
    }

    promotion = _promotion_decision(perf, stability, robust_payload)
    next_steps = _next_steps(stability, robust_payload, components)

    return StrategySpec(
        name=strategy,
        version=version_id,
        symbols=symbols,
        timeframe=timeframe,
        data_range=data_range,
        metadata=metadata,
        overview=overview,
        core_thesis=core_thesis,
        entry_logic={"description": "Derived from calibrated entry parameters.", "parameters": entry_params},
        exit_logic={"description": "Derived from calibrated exit parameters.", "parameters": exit_params},
        trade_management={
            "description": "Derived from calibrated trade-management controls.",
            "parameters": trade_params,
        },
        parameter_set={
            "candidate_type": "HARDENED_DEFAULT",
            "params": hardened_params,
            "oos_expectancy": perf["oos_expectancy"],
            "oos_sharpe": perf["oos_sharpe"],
            "oos_max_drawdown": perf["oos_max_drawdown"],
        },
        performance_summary={**perf, **stability},
        edge_characterization=edge_profile,
        robustness_assessment=robust_payload,
        failure_modes=failure_modes,
        component_insights=components,
        promotion_status=promotion,
        next_steps=next_steps,
    )


def _dict_markdown_table(payload: dict[str, Any]) -> str:
    rows = ["| Field | Value |", "|---|---|"]
    for key, value in payload.items():
        if isinstance(value, (dict, list)):
            rendered = json.dumps(value, ensure_ascii=True)
        else:
            rendered = str(value)
        rows.append(f"| {key} | {rendered} |")
    return "\n".join(rows)


def render_strategy_spec_markdown(spec: StrategySpec) -> str:
    lines: list[str] = []
    lines.append(f"# Strategy Specification: {spec.name}")
    lines.append("")
    lines.append(f"- Version: `{spec.version}`")
    lines.append(f"- Symbols: `{', '.join(spec.symbols)}`")
    lines.append(f"- Timeframe: `{spec.timeframe}`")
    lines.append(f"- Data Range: `{spec.data_range.get('start')}` to `{spec.data_range.get('end')}`")
    lines.append(f"- Timestamp (UTC): `{spec.metadata.get('timestamp_utc')}`")
    lines.append(f"- Commit: `{spec.metadata.get('commit_hash')}`")
    lines.append("")
    lines.append("## Overview")
    lines.append(_dict_markdown_table(spec.overview))
    lines.append("")
    lines.append("## Core Thesis")
    lines.append(spec.core_thesis)
    lines.append("")
    lines.append("## Mechanics")
    lines.append("### Entry Logic")
    lines.append(_dict_markdown_table(spec.entry_logic["parameters"]))
    lines.append("")
    lines.append("### Exit Logic")
    lines.append(_dict_markdown_table(spec.exit_logic["parameters"]))
    lines.append("")
    lines.append("### Trade Management")
    lines.append(_dict_markdown_table(spec.trade_management["parameters"]))
    lines.append("")
    lines.append("## Parameters")
    lines.append(_dict_markdown_table(spec.parameter_set["params"]))
    lines.append("")
    lines.append("## Performance")
    lines.append(_dict_markdown_table(spec.performance_summary))
    lines.append("")
    lines.append("## Edge Profile")
    lines.append(_dict_markdown_table(spec.edge_characterization))
    lines.append("")
    lines.append("## Robustness")
    lines.append(_dict_markdown_table(spec.robustness_assessment))
    lines.append("")
    lines.append("## Risks")
    lines.append(_dict_markdown_table(spec.failure_modes))
    lines.append("")
    lines.append("## Component Insights")
    lines.append(_dict_markdown_table(spec.component_insights))
    lines.append("")
    lines.append("## Promotion Status")
    lines.append(f"**Status:** `{spec.promotion_status.get('status')}`")
    lines.append("")
    for reason in spec.promotion_status.get("rationale", []):
        lines.append(f"- {reason}")
    lines.append("")
    lines.append("## Next Steps")
    for step in spec.next_steps:
        lines.append(f"- {step}")
    lines.append("")
    return "\n".join(lines)


def _strategy_slug(strategy: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", strategy).strip("_")


def write_strategy_spec_outputs(
    spec: StrategySpec,
    output_dir: str | Path = "outputs",
) -> tuple[Path, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = _strategy_slug(spec.name)
    md_path = out_dir / f"strategy_spec_{slug}.md"
    json_path = out_dir / f"strategy_spec_{slug}.json"

    md_path.write_text(render_strategy_spec_markdown(spec), encoding="utf-8")
    json_path.write_text(json.dumps(spec.to_dict(), indent=2, ensure_ascii=True), encoding="utf-8")
    return md_path, json_path


def generate_strategy_spec(
    strategy: str,
    symbol: str | None = None,
    output_dir: str | Path = "outputs",
    version_suffix: str = "R1",
) -> tuple[StrategySpec, Path, Path]:
    spec = build_strategy_spec(
        strategy=strategy,
        symbol=symbol,
        output_dir=output_dir,
        version_suffix=version_suffix,
    )
    md_path, json_path = write_strategy_spec_outputs(spec=spec, output_dir=output_dir)
    return spec, md_path, json_path


__all__ = [
    "StrategySpec",
    "build_strategy_spec",
    "render_strategy_spec_markdown",
    "write_strategy_spec_outputs",
    "generate_strategy_spec",
]

