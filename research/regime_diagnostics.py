from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import add_basic_indicators, load_ohlcv_csv, load_symbol_data
from regime import attach_regime_labels


@dataclass
class RegimeDiagnosticsArtifacts:
    fold_diagnostics: pd.DataFrame
    feature_correlations: pd.DataFrame
    bin_summary: pd.DataFrame
    regime_vs_sharpe_path: Path
    regime_bin_performance_path: Path


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if np.isfinite(out):
            return out
        return default
    except Exception:
        return default


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
    text = _clean_param_string(raw)
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}
    out: dict[str, Any] = {}
    for k, v in parsed.items():
        if isinstance(v, np.generic):
            out[str(k)] = v.item()
        else:
            out[str(k)] = v
    return out


def _resolve_artifact_path(artifacts_root: Path, symbol: str, filename: str) -> Path:
    candidates = [
        artifacts_root / symbol / filename,
        artifacts_root / symbol.upper() / filename,
        artifacts_root / symbol.lower() / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Missing '{filename}' for symbol '{symbol}' under '{artifacts_root}'. "
        "Expected path: outputs/<strategy>/<symbol>/<file>."
    )


def _load_symbol_fold_results(artifacts_root: Path, symbol: str) -> pd.DataFrame:
    fold_path = _resolve_artifact_path(
        artifacts_root=artifacts_root,
        symbol=symbol,
        filename="strategy_research_fold_results.csv",
    )
    return pd.read_csv(fold_path)


def _resolve_price_data_path(
    artifacts_root: Path,
    explicit_price_csv: str | Path | None,
) -> Path:
    if explicit_price_csv is not None:
        p = Path(explicit_price_csv)
        if p.exists():
            return p
        raise FileNotFoundError(f"price_csv not found: {p}")

    # Prefer strategy-scoped R1 shared source.
    candidates = [
        artifacts_root / "r1_shared_ohlcv.csv",
        Path("outputs/TrendBreakout_V2/r1_shared_ohlcv.csv"),
        Path("outputs/strategy_research_mock_ohlcv.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not locate underlying OHLC source CSV. "
        "Pass --price-csv explicitly."
    )


def _prepare_symbol_frame(price_df: pd.DataFrame, symbol: str, timeframe: str = "H1") -> pd.DataFrame:
    frame = load_symbol_data(price_df, symbol=symbol, timeframe=timeframe).copy()
    frame = add_basic_indicators(frame)
    frame = attach_regime_labels(frame, adx_threshold=25.0)
    frame["bar_range"] = (frame["high"] - frame["low"]).abs()
    frame["return"] = frame["close"].pct_change().fillna(0.0)

    # Rolling structure proxies for compression vs expansion.
    rolling_high = frame["high"].rolling(20, min_periods=20).max()
    rolling_low = frame["low"].rolling(20, min_periods=20).min()
    frame["rolling_range_20"] = (rolling_high - rolling_low).abs()
    long_avg_range = frame["rolling_range_20"].rolling(100, min_periods=30).mean()
    frame["range_ratio"] = frame["rolling_range_20"] / long_avg_range.replace(0.0, np.nan)
    return frame


def _trend_variance_ratio_window(test_window: pd.DataFrame, slope_window: int = 20) -> float:
    if len(test_window) < slope_window + 2:
        return 0.0
    close = pd.to_numeric(test_window["close"], errors="coerce")
    slope = close.pct_change(slope_window).abs()
    vol = close.pct_change().rolling(slope_window, min_periods=slope_window).std()
    ratio = slope / vol.replace(0.0, np.nan)
    return float(pd.to_numeric(ratio, errors="coerce").mean(skipna=True) or 0.0)


def _return_autocorr_window(test_window: pd.DataFrame) -> float:
    ret = pd.to_numeric(test_window["return"], errors="coerce")
    if len(ret.dropna()) < 8:
        return 0.0
    out = ret.autocorr(lag=1)
    return _safe_float(out, default=0.0)


def _fold_features(
    test_window: pd.DataFrame,
    fold_row: pd.Series,
    symbol: str,
    fold_id: int,
) -> dict[str, Any]:
    adx = pd.to_numeric(test_window["adx_14"], errors="coerce")
    atr_norm = pd.to_numeric(test_window["atr_norm"], errors="coerce")
    atr_pct = pd.to_numeric(test_window["atr_norm_pct_rank"], errors="coerce")
    range_ratio = pd.to_numeric(test_window["range_ratio"], errors="coerce")
    bar_range = pd.to_numeric(test_window["bar_range"], errors="coerce")

    test_sharpe = _safe_float(fold_row.get("test_Sharpe"))
    test_expectancy = _safe_float(fold_row.get("test_Expectancy"))
    test_trade_count = _safe_float(fold_row.get("test_TradeCount"))

    return {
        "symbol": symbol,
        "fold_id": fold_id,
        "fold_test_start": str(fold_row.get("fold_test_start", "")),
        "fold_test_end": str(fold_row.get("fold_test_end", "")),
        "test_sharpe": test_sharpe,
        "test_expectancy": test_expectancy,
        "test_trade_count": test_trade_count,
        "fold_is_positive": int(test_sharpe > 0.0),
        "fold_is_strong": int(test_sharpe > 0.5),
        "fold_is_negative": int(test_sharpe <= 0.0),
        "avg_adx": float(adx.mean(skipna=True) or 0.0),
        "pct_bars_adx_gt_25": float((adx > 25.0).mean()),
        "avg_atr_norm": float(atr_norm.mean(skipna=True) or 0.0),
        "atr_percentile_mean": float(atr_pct.mean(skipna=True) or 0.0),
        "atr_percentile_std": float(atr_pct.std(ddof=0, skipna=True) or 0.0),
        "pct_bars_low_vol": float((atr_pct < 0.3).mean()),
        "pct_bars_high_vol": float((atr_pct > 0.7).mean()),
        "range_ratio": float(range_ratio.mean(skipna=True) or 0.0),
        "trend_variance_ratio": _trend_variance_ratio_window(test_window),
        "avg_return_autocorrelation": _return_autocorr_window(test_window),
        "avg_bar_range": float(bar_range.mean(skipna=True) or 0.0),
    }


def _hardened_candidate_id(folds_df: pd.DataFrame) -> int:
    # Folds are grouped by candidate_id; use candidate with best mean test Sharpe.
    tmp = folds_df.copy()
    tmp["test_Sharpe"] = pd.to_numeric(tmp["test_Sharpe"], errors="coerce")
    grouped = (
        tmp.groupby("candidate_id", as_index=False)["test_Sharpe"]
        .mean()
        .sort_values("test_Sharpe", ascending=False)
    )
    if grouped.empty:
        return 0
    return int(_safe_float(grouped.iloc[0]["candidate_id"], default=0))


def _build_fold_diagnostics_for_symbol(
    symbol: str,
    artifacts_root: Path,
    price_df: pd.DataFrame,
    timeframe: str,
) -> pd.DataFrame:
    folds = _load_symbol_fold_results(artifacts_root=artifacts_root, symbol=symbol)
    if folds.empty:
        return pd.DataFrame()

    # Keep only hardened candidate folds by selecting the candidate with best mean test Sharpe.
    hardened_id = _hardened_candidate_id(folds)
    folds = folds[pd.to_numeric(folds["candidate_id"], errors="coerce") == hardened_id].copy()
    if folds.empty:
        return pd.DataFrame()

    symbol_frame = _prepare_symbol_frame(price_df=price_df, symbol=symbol, timeframe=timeframe)
    rows: list[dict[str, Any]] = []

    for fold_id, (_, fold_row) in enumerate(folds.iterrows()):
        test_start = pd.to_datetime(fold_row["fold_test_start"], utc=True, errors="coerce")
        test_end = pd.to_datetime(fold_row["fold_test_end"], utc=True, errors="coerce")
        if pd.isna(test_start) or pd.isna(test_end):
            continue
        test_window = symbol_frame[
            (symbol_frame["timestamp"] >= test_start) & (symbol_frame["timestamp"] <= test_end)
        ].copy()
        if test_window.empty:
            continue
        rows.append(_fold_features(test_window=test_window, fold_row=fold_row, symbol=symbol, fold_id=fold_id))

    return pd.DataFrame(rows)


def _feature_correlation_table(fold_diag: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "avg_adx",
        "pct_bars_adx_gt_25",
        "avg_atr_norm",
        "atr_percentile_mean",
        "atr_percentile_std",
        "pct_bars_low_vol",
        "pct_bars_high_vol",
        "range_ratio",
        "trend_variance_ratio",
        "avg_return_autocorrelation",
        "avg_bar_range",
    ]
    rows: list[dict[str, Any]] = []
    sharpe = pd.to_numeric(fold_diag["test_sharpe"], errors="coerce")
    pos_mask = fold_diag["fold_is_positive"].astype(int) == 1
    neg_mask = fold_diag["fold_is_negative"].astype(int) == 1

    for f in feature_cols:
        x = pd.to_numeric(fold_diag[f], errors="coerce")
        corr = x.corr(sharpe) if len(fold_diag) > 1 else np.nan
        mean_pos = float(x[pos_mask].mean(skipna=True) or 0.0) if pos_mask.any() else np.nan
        mean_neg = float(x[neg_mask].mean(skipna=True) or 0.0) if neg_mask.any() else np.nan
        rows.append(
            {
                "feature": f,
                "correlation_with_sharpe": _safe_float(corr, default=np.nan),
                "mean_positive_folds": mean_pos,
                "mean_negative_folds": mean_neg,
                "difference": (
                    _safe_float(mean_pos, default=np.nan) - _safe_float(mean_neg, default=np.nan)
                    if not (pd.isna(mean_pos) or pd.isna(mean_neg))
                    else np.nan
                ),
            }
        )
    out = pd.DataFrame(rows).sort_values("correlation_with_sharpe", ascending=False, na_position="last")
    return out.reset_index(drop=True)


def _regime_bin_summary(fold_diag: pd.DataFrame) -> pd.DataFrame:
    out_rows: list[dict[str, Any]] = []

    adx_bins = pd.cut(
        pd.to_numeric(fold_diag["avg_adx"], errors="coerce"),
        bins=[-np.inf, 20.0, 25.0, np.inf],
        labels=["ADX_LOW", "ADX_MEDIUM", "ADX_HIGH"],
    )
    atr_bins = pd.cut(
        pd.to_numeric(fold_diag["atr_percentile_mean"], errors="coerce"),
        bins=[-np.inf, 0.3, 0.7, np.inf],
        labels=["ATRP_LOW", "ATRP_MEDIUM", "ATRP_HIGH"],
    )

    def _summarize(label_series: pd.Series, prefix: str) -> None:
        tmp = fold_diag.copy()
        tmp["_bin"] = label_series.astype(str)
        for b, g in tmp.groupby("_bin"):
            if b == "nan":
                continue
            out_rows.append(
                {
                    "feature_bin": f"{prefix}:{b}",
                    "avg_sharpe": float(pd.to_numeric(g["test_sharpe"], errors="coerce").mean(skipna=True) or 0.0),
                    "avg_expectancy": float(
                        pd.to_numeric(g["test_expectancy"], errors="coerce").mean(skipna=True) or 0.0
                    ),
                    "positive_fold_pct": float((g["fold_is_positive"].astype(int) == 1).mean()),
                }
            )

    _summarize(adx_bins, "ADX")
    _summarize(atr_bins, "ATR_PERCENTILE")
    return pd.DataFrame(out_rows).sort_values("feature_bin").reset_index(drop=True)


def _plot_regime_vs_sharpe(fold_diag: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x1 = pd.to_numeric(fold_diag["avg_adx"], errors="coerce")
    y = pd.to_numeric(fold_diag["test_sharpe"], errors="coerce")
    x2 = pd.to_numeric(fold_diag["atr_percentile_mean"], errors="coerce")

    axes[0].scatter(x1, y, alpha=0.75)
    axes[0].set_xlabel("avg_adx")
    axes[0].set_ylabel("test_sharpe")
    axes[0].set_title("ADX vs Fold Sharpe")
    axes[0].grid(alpha=0.2)

    axes[1].scatter(x2, y, alpha=0.75, color="tab:orange")
    axes[1].set_xlabel("atr_percentile_mean")
    axes[1].set_ylabel("test_sharpe")
    axes[1].set_title("ATR Percentile Mean vs Fold Sharpe")
    axes[1].grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_regime_bin_performance(bin_summary: pd.DataFrame, output_path: Path) -> None:
    if bin_summary.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No bin summary data", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(bin_summary))
    labels = bin_summary["feature_bin"].astype(str).tolist()

    axes[0].bar(x, pd.to_numeric(bin_summary["avg_sharpe"], errors="coerce").fillna(0.0).to_numpy())
    axes[0].set_title("Average Sharpe by Regime Bin")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha="right")
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(
        x,
        pd.to_numeric(bin_summary["positive_fold_pct"], errors="coerce").fillna(0.0).to_numpy(),
        color="tab:green",
    )
    axes[1].set_title("Positive Fold % by Regime Bin")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_regime_diagnostics(
    strategy: str,
    symbols: list[str],
    artifacts_root: str | Path = "outputs/TrendBreakout_V2",
    output_dir: str | Path = "outputs",
    timeframe: str = "H1",
    price_csv: str | Path | None = None,
) -> RegimeDiagnosticsArtifacts:
    if not symbols:
        raise ValueError("At least one symbol is required.")
    if strategy != "TrendBreakout_V2":
        raise ValueError("R1.1 regime diagnostics currently targets strategy='TrendBreakout_V2'.")

    artifacts_root_path = Path(artifacts_root)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    price_path = _resolve_price_data_path(artifacts_root=artifacts_root_path, explicit_price_csv=price_csv)
    price_df = load_ohlcv_csv(price_path)

    all_fold_rows: list[pd.DataFrame] = []
    for symbol in symbols:
        fold_df = _build_fold_diagnostics_for_symbol(
            symbol=symbol,
            artifacts_root=artifacts_root_path,
            price_df=price_df,
            timeframe=timeframe,
        )
        if not fold_df.empty:
            all_fold_rows.append(fold_df)

    if not all_fold_rows:
        raise ValueError("No fold diagnostics could be built. Check artifacts_root and symbols.")

    fold_diag = pd.concat(all_fold_rows, ignore_index=True)
    corr_df = _feature_correlation_table(fold_diag)
    bin_df = _regime_bin_summary(fold_diag)

    fold_csv = out_dir / "regime_fold_diagnostics.csv"
    corr_csv = out_dir / "regime_feature_correlations.csv"
    bin_csv = out_dir / "regime_bin_summary.csv"
    fold_diag.to_csv(fold_csv, index=False)
    corr_df.to_csv(corr_csv, index=False)
    bin_df.to_csv(bin_csv, index=False)

    regime_vs_sharpe_path = out_dir / "regime_vs_sharpe.png"
    regime_bin_performance_path = out_dir / "regime_bin_performance.png"
    _plot_regime_vs_sharpe(fold_diag, regime_vs_sharpe_path)
    _plot_regime_bin_performance(bin_df, regime_bin_performance_path)

    return RegimeDiagnosticsArtifacts(
        fold_diagnostics=fold_diag,
        feature_correlations=corr_df,
        bin_summary=bin_df,
        regime_vs_sharpe_path=regime_vs_sharpe_path,
        regime_bin_performance_path=regime_bin_performance_path,
    )


__all__ = [
    "RegimeDiagnosticsArtifacts",
    "run_regime_diagnostics",
]

