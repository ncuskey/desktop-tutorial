from __future__ import annotations

import ast
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_ARTIFACTS = (
    "strategy_research_summary.csv",
    "strategy_research_fold_results.csv",
    "strategy_research_robustness.csv",
    "strategy_research_recommendation.csv",
)


@dataclass
class PromotionThresholds:
    min_sharpe_promote: float = 0.5
    min_positive_fold_pct_promote: float = 0.6
    max_robustness_gap_ratio_promote: float = 0.15
    min_positive_fold_pct_stability: float = 0.35
    max_sharpe_std_stability: float = 2.5
    max_expectancy_std_ratio: float = 5.0


@dataclass
class SymbolPromotionResult:
    strategy_name: str
    symbol: str
    expectancy: float
    sharpe: float
    max_dd: float
    trade_count: float
    positive_fold_pct: float
    robustness_gap: float
    robust_score: float | None
    classification: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PromotionArtifacts:
    summary: pd.DataFrame
    overview: pd.DataFrame
    parameter_alignment: pd.DataFrame


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


def _resolve_artifact_path(
    artifacts_root: Path,
    symbol: str,
    filename: str,
    allow_root_fallback: bool,
) -> Path:
    stem, suffix = filename.rsplit(".", 1)
    candidates = [
        artifacts_root / symbol / filename,
        artifacts_root / symbol.upper() / filename,
        artifacts_root / symbol.lower() / filename,
        artifacts_root / f"{stem}_{symbol}.{suffix}",
        artifacts_root / f"{stem}_{symbol.upper()}.{suffix}",
        artifacts_root / f"{stem}_{symbol.lower()}.{suffix}",
        artifacts_root / f"{symbol}_{filename}",
    ]
    if allow_root_fallback:
        candidates.append(artifacts_root / filename)
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not resolve artifact '{filename}' for symbol '{symbol}' under '{artifacts_root}'. "
        "Supported layouts: outputs/<SYMBOL>/<file> or outputs/<file>_<SYMBOL>.csv. "
        "For multi-symbol runs, symbol-specific files are required."
    )


def _load_symbol_artifacts(
    artifacts_root: Path,
    symbol: str,
    allow_root_fallback: bool,
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for filename in REQUIRED_ARTIFACTS:
        path = _resolve_artifact_path(
            artifacts_root,
            symbol=symbol,
            filename=filename,
            allow_root_fallback=allow_root_fallback,
        )
        out[filename] = pd.read_csv(path)
    return out


def _select_summary_row(summary: pd.DataFrame, strategy: str, symbol: str) -> pd.Series:
    out = summary.copy()
    if "strategy_family" in out.columns:
        out = out[out["strategy_family"].astype(str) == strategy]
    if "symbol" in out.columns:
        narrowed = out[out["symbol"].astype(str).str.upper() == symbol.upper()]
        if not narrowed.empty:
            out = narrowed
    if out.empty:
        raise ValueError(f"No summary row found for strategy='{strategy}' symbol='{symbol}'")
    return out.iloc[0]


def _select_hardened_row(recommendation: pd.DataFrame) -> pd.Series:
    out = recommendation.copy()
    if "candidate_type" not in out.columns:
        raise ValueError("Recommendation artifact missing 'candidate_type' column.")
    hardened = out[out["candidate_type"].astype(str).str.upper() == "HARDENED_DEFAULT"]
    if hardened.empty:
        raise ValueError("No HARDENED_DEFAULT candidate found in recommendation artifact.")
    return hardened.iloc[0]


def _candidate_rows_from_robustness(robustness: pd.DataFrame) -> pd.DataFrame:
    out = robustness.copy()
    if "section" in out.columns:
        out = out[out["section"].astype(str) == "candidate"]
    if "candidate_id" in out.columns:
        out = out[pd.to_numeric(out["candidate_id"], errors="coerce").notna()]
    return out


def _infer_hardened_candidate_id(robustness: pd.DataFrame, hardened_params: dict[str, Any]) -> int | None:
    candidates = _candidate_rows_from_robustness(robustness)
    if candidates.empty or "params" not in candidates.columns:
        return None
    parsed = candidates["params"].apply(_parse_params)
    match = parsed.apply(lambda p: _params_equal(p, hardened_params))
    rows = candidates[match]
    if rows.empty:
        return None
    return int(float(rows.iloc[0]["candidate_id"]))


def _select_hardened_candidate_row(
    robustness: pd.DataFrame,
    candidate_id: int | None,
    hardened_params: dict[str, Any],
) -> pd.Series | None:
    candidates = _candidate_rows_from_robustness(robustness)
    if candidates.empty:
        return None
    if candidate_id is not None and "candidate_id" in candidates.columns:
        by_id = candidates[pd.to_numeric(candidates["candidate_id"], errors="coerce") == candidate_id]
        if not by_id.empty:
            return by_id.iloc[0]
    if "params" in candidates.columns:
        parsed = candidates["params"].apply(_parse_params)
        match = parsed.apply(lambda p: _params_equal(p, hardened_params))
        by_params = candidates[match]
        if not by_params.empty:
            return by_params.iloc[0]
    return None


def _select_hardened_fold_rows(
    folds: pd.DataFrame,
    candidate_id: int | None,
    hardened_params: dict[str, Any],
) -> pd.DataFrame:
    out = folds.copy()
    if candidate_id is not None and "candidate_id" in out.columns:
        by_id = out[pd.to_numeric(out["candidate_id"], errors="coerce") == candidate_id]
        if not by_id.empty:
            return by_id
    if "params" in out.columns:
        parsed = out["params"].apply(_parse_params)
        match = parsed.apply(lambda p: _params_equal(p, hardened_params))
        by_params = out[match]
        if not by_params.empty:
            return by_params
    return out.iloc[0:0]


def _robustness_gap(summary_row: pd.Series) -> float:
    peak = _safe_float(summary_row.get("best_peak_expectancy"))
    hardened = _safe_float(summary_row.get("best_hardened_expectancy"))
    return peak - hardened


def _is_highly_unstable(
    expectancy: float,
    positive_fold_pct: float,
    sharpe_std: float,
    expectancy_std: float,
    thresholds: PromotionThresholds,
) -> bool:
    if positive_fold_pct < thresholds.min_positive_fold_pct_stability:
        return True
    if sharpe_std > thresholds.max_sharpe_std_stability:
        return True
    scale = max(abs(expectancy), 1e-9)
    if expectancy_std > thresholds.max_expectancy_std_ratio * scale:
        return True
    return False


def _classify_symbol(
    expectancy: float,
    sharpe: float,
    positive_fold_pct: float,
    robustness_gap: float,
    sharpe_std: float,
    expectancy_std: float,
    thresholds: PromotionThresholds,
) -> str:
    if expectancy <= 0:
        return "REJECT"

    unstable = _is_highly_unstable(
        expectancy=expectancy,
        positive_fold_pct=positive_fold_pct,
        sharpe_std=sharpe_std,
        expectancy_std=expectancy_std,
        thresholds=thresholds,
    )
    if unstable:
        return "REJECT"

    gap_ratio = robustness_gap / max(abs(expectancy), 1e-9)
    is_promote = (
        sharpe >= thresholds.min_sharpe_promote
        and positive_fold_pct >= thresholds.min_positive_fold_pct_promote
        and gap_ratio <= thresholds.max_robustness_gap_ratio_promote
    )
    if is_promote:
        return "PROMOTE"
    return "CONDITIONAL"


def _build_symbol_result(
    strategy: str,
    symbol: str,
    summary_row: pd.Series,
    hardened_row: pd.Series,
    hardened_candidate_row: pd.Series | None,
    fold_rows: pd.DataFrame,
    thresholds: PromotionThresholds,
) -> tuple[SymbolPromotionResult, dict[str, Any]]:
    expectancy = _safe_float(hardened_row.get("OOS_Expectancy"))
    sharpe = _safe_float(hardened_row.get("OOS_Sharpe"))
    max_dd = _safe_float(hardened_row.get("OOS_MaxDrawdown"))
    robust_score = (
        _safe_float(hardened_candidate_row.get("robustness_score"))
        if hardened_candidate_row is not None and "robustness_score" in hardened_candidate_row.index
        else _safe_float(hardened_row.get("robust_score"), default=np.nan)
    )
    if math.isnan(robust_score):
        robust_score_value: float | None = None
    else:
        robust_score_value = robust_score

    if hardened_candidate_row is not None and "OOS_TradeCount" in hardened_candidate_row.index:
        trade_count = _safe_float(hardened_candidate_row.get("OOS_TradeCount"))
    else:
        fold_trade_series = (
            pd.to_numeric(fold_rows["test_TradeCount"], errors="coerce")
            if "test_TradeCount" in fold_rows.columns
            else pd.Series(dtype=float)
        )
        trade_count = _safe_float(fold_trade_series.sum())

    fold_sharpe = pd.to_numeric(fold_rows.get("test_Sharpe"), errors="coerce").fillna(0.0)
    fold_expectancy = pd.to_numeric(fold_rows.get("test_Expectancy"), errors="coerce").fillna(0.0)
    positive_fold_pct = float((fold_sharpe > 0).mean()) if len(fold_sharpe) else 0.0
    sharpe_std = float(fold_sharpe.std(ddof=0)) if len(fold_sharpe) else 0.0
    expectancy_std = float(fold_expectancy.std(ddof=0)) if len(fold_expectancy) else 0.0

    gap = _robustness_gap(summary_row)
    classification = _classify_symbol(
        expectancy=expectancy,
        sharpe=sharpe,
        positive_fold_pct=positive_fold_pct,
        robustness_gap=gap,
        sharpe_std=sharpe_std,
        expectancy_std=expectancy_std,
        thresholds=thresholds,
    )

    result = SymbolPromotionResult(
        strategy_name=strategy,
        symbol=symbol,
        expectancy=expectancy,
        sharpe=sharpe,
        max_dd=max_dd,
        trade_count=trade_count,
        positive_fold_pct=positive_fold_pct,
        robustness_gap=gap,
        robust_score=robust_score_value,
        classification=classification,
    )
    return result, {"hardened_params": _parse_params(hardened_row.get("params"))}


def _overall_classification(n_promote: int, n_symbols: int) -> str:
    if n_symbols <= 0:
        return "REJECT"
    if n_symbols == 3:
        if n_promote == 3:
            return "UNIVERSAL"
        if n_promote == 2:
            return "STRONG"
        if n_promote == 1:
            return "NICHE"
        return "REJECT"

    ratio = n_promote / n_symbols
    if ratio >= 0.999:
        return "UNIVERSAL"
    if ratio >= 0.667:
        return "STRONG"
    if ratio > 0:
        return "NICHE"
    return "REJECT"


def _render_param_value(value: Any) -> Any:
    native = _to_native(value)
    if isinstance(native, (dict, list, tuple)):
        return str(native)
    return native


def _alignment_score(values: list[Any]) -> float:
    non_null = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if len(non_null) <= 1:
        return 1.0
    if all(_is_number(v) for v in non_null):
        vals = np.asarray([float(v) for v in non_null], dtype=float)
        scale = max(float(np.max(np.abs(vals))), abs(float(np.mean(vals))), 1.0)
        distance = (float(vals.max()) - float(vals.min())) / scale
        return float(max(0.0, 1.0 - distance))
    as_text = [str(v) for v in non_null]
    mode_freq = max(as_text.count(v) for v in set(as_text))
    return float(mode_freq / len(as_text))


def _build_parameter_alignment(
    strategy: str,
    symbols: list[str],
    symbol_params: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    union_params: set[str] = set()
    for params in symbol_params.values():
        union_params.update(params.keys())

    rows: list[dict[str, Any]] = []
    for param in sorted(union_params):
        row: dict[str, Any] = {"parameter_name": param}
        vals: list[Any] = []
        for symbol in symbols:
            value = symbol_params.get(symbol, {}).get(param)
            row[f"{symbol}_value"] = _render_param_value(value) if value is not None else np.nan
            vals.append(value)
        row["alignment_score"] = _alignment_score(vals)
        rows.append(row)
    return pd.DataFrame(rows)


def run_strategy_promotion_framework(
    strategy: str,
    symbols: list[str],
    artifacts_root: str | Path = "outputs",
    output_dir: str | Path = "outputs",
    thresholds: PromotionThresholds | None = None,
) -> PromotionArtifacts:
    if not symbols:
        raise ValueError("At least one symbol is required.")

    thresholds = thresholds or PromotionThresholds()
    artifacts_root_path = Path(artifacts_root)

    summary_rows: list[dict[str, Any]] = []
    symbol_params: dict[str, dict[str, Any]] = {}

    allow_root_fallback = len(symbols) == 1
    for symbol in symbols:
        try:
            artifacts = _load_symbol_artifacts(
                artifacts_root=artifacts_root_path,
                symbol=symbol,
                allow_root_fallback=allow_root_fallback,
            )
            summary_row = _select_summary_row(
                artifacts["strategy_research_summary.csv"], strategy=strategy, symbol=symbol
            )
            hardened_row = _select_hardened_row(artifacts["strategy_research_recommendation.csv"])
            hardened_params = _parse_params(hardened_row.get("params"))

            candidate_id = _infer_hardened_candidate_id(
                artifacts["strategy_research_robustness.csv"], hardened_params=hardened_params
            )
            fold_rows = _select_hardened_fold_rows(
                artifacts["strategy_research_fold_results.csv"],
                candidate_id=candidate_id,
                hardened_params=hardened_params,
            )
            hardened_candidate_row = _select_hardened_candidate_row(
                artifacts["strategy_research_robustness.csv"],
                candidate_id=candidate_id,
                hardened_params=hardened_params,
            )

            result, meta = _build_symbol_result(
                strategy=strategy,
                symbol=symbol,
                summary_row=summary_row,
                hardened_row=hardened_row,
                hardened_candidate_row=hardened_candidate_row,
                fold_rows=fold_rows,
                thresholds=thresholds,
            )
            summary_rows.append(result.to_dict())
            symbol_params[symbol] = meta["hardened_params"]
        except (FileNotFoundError, ValueError):
            # Deterministic placeholder for symbols without resolved artifacts.
            summary_rows.append(
                {
                    "strategy_name": strategy,
                    "symbol": symbol,
                    "expectancy": np.nan,
                    "sharpe": np.nan,
                    "max_dd": np.nan,
                    "trade_count": 0.0,
                    "positive_fold_pct": 0.0,
                    "robustness_gap": np.nan,
                    "classification": "REJECT",
                }
            )
            symbol_params[symbol] = {}

    summary_df = pd.DataFrame(summary_rows)
    summary_columns = [
        "strategy_name",
        "symbol",
        "expectancy",
        "sharpe",
        "max_dd",
        "trade_count",
        "positive_fold_pct",
        "robustness_gap",
        "classification",
    ]
    summary_df = summary_df.reindex(columns=summary_columns)
    n_promote = int((summary_df["classification"] == "PROMOTE").sum())
    n_conditional = int((summary_df["classification"] == "CONDITIONAL").sum())
    n_reject = int((summary_df["classification"] == "REJECT").sum())

    overview_df = pd.DataFrame(
        [
            {
                "strategy_name": strategy,
                "n_promote": n_promote,
                "n_conditional": n_conditional,
                "n_reject": n_reject,
                "overall_classification": _overall_classification(n_promote=n_promote, n_symbols=len(symbols)),
            }
        ]
    )

    alignment_df = _build_parameter_alignment(
        strategy=strategy, symbols=symbols, symbol_params=symbol_params
    )
    alignment_columns = ["parameter_name", *[f"{s}_value" for s in symbols], "alignment_score"]
    alignment_df = alignment_df.reindex(columns=alignment_columns)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_dir / "strategy_promotion_summary.csv", index=False)
    overview_df.to_csv(out_dir / "strategy_promotion_overview.csv", index=False)
    alignment_df.to_csv(out_dir / "strategy_parameter_alignment.csv", index=False)

    return PromotionArtifacts(summary=summary_df, overview=overview_df, parameter_alignment=alignment_df)


__all__ = [
    "PromotionThresholds",
    "SymbolPromotionResult",
    "PromotionArtifacts",
    "run_strategy_promotion_framework",
]

