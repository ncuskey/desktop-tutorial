from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import CostModel, add_basic_indicators, load_ohlcv_csv, load_symbol_data
from execution.simulator import run_backtest
from regime import attach_regime_labels
from strategies import trend_breakout_v2_signals


TIMEFRAME_TO_PERIODS = {"H1": 24 * 252, "H4": 6 * 252, "D1": 252}


@dataclass
class RegimeGatedArtifacts:
    comparison: pd.DataFrame
    fold_results: pd.DataFrame
    equity_plot_path: Path


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
    if isinstance(value, dict):
        return {str(k): _to_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_to_native(v) for v in value]
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
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, bool
    )


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


def _resolve_symbol_file(
    artifacts_root: Path,
    symbol: str,
    filename: str,
) -> Path:
    candidates = [
        artifacts_root / symbol / filename,
        artifacts_root / symbol.upper() / filename,
        artifacts_root / symbol.lower() / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Missing artifact for symbol '{symbol}': {filename} under {artifacts_root}"
    )


def _resolve_price_data_path(
    artifacts_root: Path,
    explicit_source_csv: str | Path | None,
) -> Path:
    if explicit_source_csv is not None:
        p = Path(explicit_source_csv)
        if not p.exists():
            raise FileNotFoundError(f"source-csv not found: {p}")
        return p
    candidates = [
        artifacts_root / "r1_shared_ohlcv.csv",
        Path("outputs/TrendBreakout_V2/r1_shared_ohlcv.csv"),
        Path("outputs/strategy_research_mock_ohlcv.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not locate price data for regime-gated evaluation. "
        "Pass --source-csv explicitly."
    )


def _load_hardened_params(artifacts_root: Path, symbol: str) -> dict[str, Any]:
    rec_path = _resolve_symbol_file(artifacts_root, symbol, "strategy_research_recommendation.csv")
    rec = pd.read_csv(rec_path)
    if "candidate_type" not in rec.columns:
        raise ValueError(f"{rec_path} missing candidate_type column")
    hardened = rec[rec["candidate_type"].astype(str).str.upper() == "HARDENED_DEFAULT"]
    if hardened.empty:
        raise ValueError(f"No HARDENED_DEFAULT found in {rec_path}")
    return _parse_params(hardened.iloc[0].get("params"))


def _load_hardened_folds(
    artifacts_root: Path,
    symbol: str,
    hardened_params: dict[str, Any],
) -> pd.DataFrame:
    fold_path = _resolve_symbol_file(artifacts_root, symbol, "strategy_research_fold_results.csv")
    folds = pd.read_csv(fold_path)
    if "params" not in folds.columns:
        raise ValueError(f"{fold_path} missing params column")
    parsed = folds["params"].apply(_parse_params)
    matched = folds[parsed.apply(lambda p: _params_equal(p, hardened_params))].copy()
    if matched.empty:
        raise ValueError(
            f"Could not match HARDENED_DEFAULT params to fold rows for {symbol} ({fold_path})"
        )
    matched["fold_test_start"] = pd.to_datetime(matched["fold_test_start"], utc=True, errors="coerce")
    matched["fold_test_end"] = pd.to_datetime(matched["fold_test_end"], utc=True, errors="coerce")
    matched = matched.dropna(subset=["fold_test_start", "fold_test_end"])
    matched = matched.sort_values("fold_test_start").reset_index(drop=True)
    matched["fold_id"] = np.arange(len(matched), dtype=int)
    return matched


def _prepare_symbol_frame(
    raw_prices: pd.DataFrame,
    symbol: str,
    timeframe: str = "H1",
) -> pd.DataFrame:
    df = load_symbol_data(raw_prices, symbol=symbol, timeframe=timeframe).copy()
    df = add_basic_indicators(df)
    df = attach_regime_labels(df, adx_threshold=25.0)
    # Needed by strategy and gating layers.
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
        "trend_variance_ratio",
    ]
    return df.dropna(subset=req).reset_index(drop=True)


def apply_regime_filter(df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """
    Bar-level no-lookahead regime gate.

    ALLOW when:
      - rolling avg ADX > threshold_adx OR
      - rolling pct(ADX > adx_gt_level) > threshold_pct
    and optionally:
      - trend_variance_ratio > threshold (constant or expanding median split).
    """
    adx = pd.to_numeric(df.get("adx_14"), errors="coerce")
    tvr = pd.to_numeric(df.get("trend_variance_ratio"), errors="coerce")

    adx_window = int(params.get("regime_adx_window", 48))
    adx_gt_level = float(params.get("regime_adx_gt_level", 25.0))
    threshold_adx = float(params.get("regime_threshold_adx", 25.0))
    threshold_pct = float(params.get("regime_threshold_pct", 0.30))
    use_tvr = bool(params.get("regime_use_trend_variance_filter", True))
    tvr_min_periods = int(params.get("regime_tvr_min_periods", 80))
    tvr_threshold_param = params.get("regime_trend_variance_threshold")

    rolling_avg_adx = adx.rolling(adx_window, min_periods=max(10, adx_window // 3)).mean()
    rolling_pct_adx_gt = (adx > adx_gt_level).astype(float).rolling(
        adx_window, min_periods=max(10, adx_window // 3)
    ).mean()

    allow = (rolling_avg_adx > threshold_adx) | (rolling_pct_adx_gt > threshold_pct)
    if use_tvr:
        if tvr_threshold_param is None:
            # Median split with expanding history only (shifted => no lookahead).
            tvr_threshold = tvr.expanding(min_periods=tvr_min_periods).median().shift(1)
        else:
            tvr_threshold = pd.Series(float(tvr_threshold_param), index=df.index, dtype=float)
        allow = allow & (tvr > tvr_threshold)

    return allow.fillna(False).astype(bool)


def _annualized_sharpe(returns: pd.Series, timeframe: str) -> float:
    ppy = TIMEFRAME_TO_PERIODS.get(timeframe.upper(), 24 * 252)
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if r.empty:
        return 0.0
    std = float(r.std(ddof=0))
    if std <= 0:
        return 0.0
    return float(np.sqrt(ppy) * (float(r.mean()) / std))


def _max_drawdown_from_returns(returns: pd.Series, initial_equity: float = 100_000.0) -> float:
    r = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    if r.empty:
        return 0.0
    equity = initial_equity * (1.0 + r).cumprod()
    dd = (equity / equity.cummax()) - 1.0
    return float(dd.min())


def _aggregate_variant_metrics(
    fold_variant_df: pd.DataFrame,
    stitched_returns: pd.Series,
    timeframe: str,
) -> dict[str, float]:
    if fold_variant_df.empty:
        return {
            "sharpe": 0.0,
            "expectancy": 0.0,
            "max_dd": 0.0,
            "trade_count": 0.0,
            "positive_fold_pct": 0.0,
        }
    tc = pd.to_numeric(fold_variant_df["test_trade_count"], errors="coerce").fillna(0.0)
    exp = pd.to_numeric(fold_variant_df["test_expectancy"], errors="coerce").fillna(0.0)
    total_trades = float(tc.sum())
    if total_trades > 0:
        expectancy = float((exp * tc).sum() / total_trades)
    else:
        expectancy = 0.0
    sharpe = _annualized_sharpe(stitched_returns, timeframe=timeframe)
    max_dd = _max_drawdown_from_returns(stitched_returns)
    positive_fold_pct = float((pd.to_numeric(fold_variant_df["test_sharpe"], errors="coerce") > 0).mean())
    return {
        "sharpe": sharpe,
        "expectancy": expectancy,
        "max_dd": max_dd,
        "trade_count": total_trades,
        "positive_fold_pct": positive_fold_pct,
    }


def _plot_regime_gated_equity(
    symbol_equity: dict[str, dict[str, pd.Series]],
    output_path: Path,
) -> None:
    symbols = list(symbol_equity.keys())
    if not symbols:
        return
    fig, axes = plt.subplots(len(symbols), 1, figsize=(12, 4 * len(symbols)), sharex=False)
    if len(symbols) == 1:
        axes = [axes]
    for ax, symbol in zip(axes, symbols, strict=False):
        unfiltered = symbol_equity[symbol].get("unfiltered", pd.Series(dtype=float))
        gated = symbol_equity[symbol].get("gated", pd.Series(dtype=float))
        if not unfiltered.empty:
            ax.plot(unfiltered.index, unfiltered.values, label="unfiltered", linewidth=1.5)
        if not gated.empty:
            ax.plot(gated.index, gated.values, label="gated", linewidth=1.5)
        ax.set_title(f"{symbol}: Regime-Gated vs Unfiltered Equity")
        ax.set_ylabel("Equity")
        ax.grid(alpha=0.2)
        ax.legend(loc="best")
    axes[-1].set_xlabel("Timestamp")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_regime_gated_evaluation(
    strategy: str,
    symbols: list[str],
    artifacts_root: str | Path = "outputs/TrendBreakout_V2",
    output_dir: str | Path = "outputs",
    use_hardened_default: bool = True,
    timeframe: str = "H1",
    source_csv: str | Path | None = None,
    gating_params: dict[str, Any] | None = None,
) -> RegimeGatedArtifacts:
    if strategy != "TrendBreakout_V2":
        raise ValueError("R1.2 currently supports strategy='TrendBreakout_V2' only.")
    if not use_hardened_default:
        raise ValueError("R1.2 requires --use-hardened-default (no re-optimization).")
    if not symbols:
        raise ValueError("At least one symbol is required.")

    gating_params = gating_params or {}
    artifacts_root_path = Path(artifacts_root)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    price_path = _resolve_price_data_path(
        artifacts_root=artifacts_root_path,
        explicit_source_csv=source_csv,
    )
    raw_prices = load_ohlcv_csv(price_path)
    cost_model = CostModel(spread_bps=0.8, slippage_bps=0.5, commission_bps=0.2)

    all_fold_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    symbol_equity: dict[str, dict[str, pd.Series]] = {}

    for symbol in symbols:
        hardened_params = _load_hardened_params(artifacts_root_path, symbol=symbol)
        fold_windows = _load_hardened_folds(
            artifacts_root=artifacts_root_path,
            symbol=symbol,
            hardened_params=hardened_params,
        )
        frame = _prepare_symbol_frame(raw_prices=raw_prices, symbol=symbol, timeframe=timeframe)
        if frame.empty:
            raise ValueError(f"No prepared frame rows for {symbol}")

        per_variant_returns: dict[str, list[pd.Series]] = {"unfiltered": [], "gated": []}

        for _, fold in fold_windows.iterrows():
            fold_id = int(fold["fold_id"])
            test_start = pd.Timestamp(fold["fold_test_start"])
            test_end = pd.Timestamp(fold["fold_test_end"])
            test_df = frame[
                (frame["timestamp"] >= test_start) & (frame["timestamp"] <= test_end)
            ].copy()
            if test_df.empty:
                continue

            raw_signal = trend_breakout_v2_signals(test_df, hardened_params).astype(float)
            gate_mask = apply_regime_filter(test_df, gating_params)
            gated_signal = raw_signal.where(gate_mask, 0.0)

            variant_signals = {"unfiltered": raw_signal, "gated": gated_signal}
            for variant, signal in variant_signals.items():
                bt = run_backtest(test_df, signal, cost_model=cost_model)
                # Reindex stitched series by true timestamp.
                idx = pd.to_datetime(test_df["timestamp"], utc=True, errors="coerce")
                ret = bt.returns.copy()
                ret.index = idx
                per_variant_returns[variant].append(ret)

                # Fold metrics (per requested diagnostics).
                test_trade_count = float(len(bt.trades))
                test_expectancy = (
                    float(pd.to_numeric(bt.trades["trade_return"], errors="coerce").mean())
                    if test_trade_count > 0
                    else 0.0
                )
                fold_sharpe = _annualized_sharpe(ret, timeframe=timeframe)
                fold_max_dd = _max_drawdown_from_returns(ret)
                all_fold_rows.append(
                    {
                        "symbol": symbol,
                        "fold_id": fold_id,
                        "variant": variant,
                        "fold_test_start": test_start,
                        "fold_test_end": test_end,
                        "test_sharpe": fold_sharpe,
                        "test_expectancy": test_expectancy,
                        "test_max_dd": fold_max_dd,
                        "test_trade_count": test_trade_count,
                        "regime_allowed_pct": float(gate_mask.mean()) if variant == "gated" else 1.0,
                    }
                )

        symbol_equity[symbol] = {}
        for variant in ("unfiltered", "gated"):
            stitched = (
                pd.concat(per_variant_returns[variant]).sort_index()
                if per_variant_returns[variant]
                else pd.Series(dtype=float)
            )
            if stitched.empty:
                equity = pd.Series(dtype=float)
            else:
                equity = 100_000.0 * (1.0 + stitched).cumprod()
            symbol_equity[symbol][variant] = equity

    fold_results = pd.DataFrame(all_fold_rows).sort_values(
        ["symbol", "variant", "fold_id"]
    ).reset_index(drop=True)

    for symbol in symbols:
        for variant in ("unfiltered", "gated"):
            fv = fold_results[
                (fold_results["symbol"] == symbol) & (fold_results["variant"] == variant)
            ].copy()
            if fv.empty:
                metrics = {
                    "sharpe": 0.0,
                    "expectancy": 0.0,
                    "max_dd": 0.0,
                    "trade_count": 0.0,
                    "positive_fold_pct": 0.0,
                }
            else:
                stitched_returns = (
                    (symbol_equity[symbol][variant].pct_change().fillna(0.0))
                    if not symbol_equity[symbol][variant].empty
                    else pd.Series(dtype=float)
                )
                metrics = _aggregate_variant_metrics(
                    fv,
                    stitched_returns=stitched_returns,
                    timeframe=timeframe,
                )
            comparison_rows.append(
                {
                    "symbol": symbol,
                    "variant": variant,
                    "sharpe": metrics["sharpe"],
                    "expectancy": metrics["expectancy"],
                    "max_dd": metrics["max_dd"],
                    "trade_count": metrics["trade_count"],
                    "positive_fold_pct": metrics["positive_fold_pct"],
                }
            )

    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["symbol", "variant"]
    ).reset_index(drop=True)

    comparison_csv = out_dir / "regime_gated_comparison.csv"
    fold_csv = out_dir / "regime_gated_fold_results.csv"
    plot_path = out_dir / "regime_gated_equity.png"
    comparison.to_csv(comparison_csv, index=False)
    fold_results.to_csv(fold_csv, index=False)
    _plot_regime_gated_equity(symbol_equity, plot_path)

    return RegimeGatedArtifacts(
        comparison=comparison,
        fold_results=fold_results,
        equity_plot_path=plot_path,
    )


__all__ = [
    "RegimeGatedArtifacts",
    "apply_regime_filter",
    "run_regime_gated_evaluation",
]

