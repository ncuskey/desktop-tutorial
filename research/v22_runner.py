from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from data import (
    CostModel,
    add_basic_indicators,
    attach_costs,
    ensure_mock_ohlcv_csv,
    load_ohlcv_csv,
    load_symbol_data,
)
from metalabel import (
    RuleBasedMetaFilter,
    apply_meta_trade_filter,
    build_trade_meta_features,
    create_trade_success_labels,
)
from regime import attach_regime_labels, attach_stable_regime_state
from research.walk_forward import WalkForwardResult, run_walk_forward
from strategies import trend_breakout_v2_signals


DEFAULT_SYMBOLS = ["EURUSD", "GBPUSD", "AUDUSD"]
DEFAULT_TIMEFRAME = "H1"


@dataclass
class V22RunArtifacts:
    summary: pd.DataFrame
    trade_distribution: pd.DataFrame
    output_dir: Path


def _symbol_cost_model(symbol: str) -> CostModel:
    per_symbol = {
        "EURUSD": CostModel(spread_bps=0.7, slippage_bps=0.4, commission_bps=0.2),
        "GBPUSD": CostModel(spread_bps=0.9, slippage_bps=0.5, commission_bps=0.2),
        "AUDUSD": CostModel(spread_bps=0.8, slippage_bps=0.5, commission_bps=0.2),
    }
    return per_symbol.get(symbol, CostModel())


def _prepare_symbol_frame(raw: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
    df = load_symbol_data(raw, symbol=symbol, timeframe=timeframe)
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
    df = attach_costs(df, _symbol_cost_model(symbol))

    required = [
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
        "rsi_14",
        "bb_mid_20",
        "bb_upper_20_2",
        "bb_lower_20_2",
        "stable_trend_regime",
        "stable_vol_regime",
    ]
    return df.dropna(subset=required).reset_index(drop=True)


def _trend_breakout_v22_grid() -> dict[str, list]:
    # Keep the grid compact to maintain strict WF runtime.
    return {
        "lookback": [20],
        "vol_compression_max_pct": [0.40],
        "breakout_strength_atr_mult": [0.20, 0.35],
        "retest_entry_mode": [False, True],
        "retest_expiry_bars": [8],
        "retest_tolerance_atr_mult": [0.15],
        "retest_confirm_buffer_atr_mult": [0.05],
        "trailing_stop_atr_mult": [1.8],
        "max_holding_bars": [72],
        "vol_contraction_exit_mult": [0.80],
        "vol_contraction_window": [20],
        "partial_take_profit_rr": [0.0, 1.0],
        "partial_take_profit_size": [0.5],
        "min_bars_between_trades": [6, 12],
    }


def _trade_distribution_rows(
    symbol: str,
    variant: str,
    trades: pd.DataFrame | None,
) -> dict[str, float | str]:
    if trades is None or trades.empty or "trade_return" not in trades.columns:
        return {
            "symbol": symbol,
            "variant": variant,
            "trade_count": 0.0,
            "win_rate": 0.0,
            "expectancy": 0.0,
            "median_trade_return": 0.0,
            "p10_trade_return": 0.0,
            "p90_trade_return": 0.0,
            "avg_holding_bars": 0.0,
        }
    tr = trades["trade_return"].astype(float)
    return {
        "symbol": symbol,
        "variant": variant,
        "trade_count": float(len(tr)),
        "win_rate": float((tr > 0).mean()),
        "expectancy": float(tr.mean()),
        "median_trade_return": float(tr.median()),
        "p10_trade_return": float(tr.quantile(0.10)),
        "p90_trade_return": float(tr.quantile(0.90)),
        "avg_holding_bars": float(trades["holding_bars"].mean())
        if "holding_bars" in trades.columns
        else 0.0,
    }


def _plot_candidate_equity(
    result_by_symbol: dict[str, WalkForwardResult],
    output_path: Path,
) -> None:
    n = len(result_by_symbol)
    fig, axes = plt.subplots(n, 1, figsize=(13, 4.2 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, (symbol, wf_res) in zip(axes, result_by_symbol.items(), strict=False):
        ax.plot(
            wf_res.combined_equity.index,
            wf_res.combined_equity.values,
            label=f"{symbol} Unfiltered",
            alpha=0.9,
        )
        if wf_res.filtered_combined_equity is not None:
            ax.plot(
                wf_res.filtered_combined_equity.index,
                wf_res.filtered_combined_equity.values,
                label=f"{symbol} Meta-Filtered",
                alpha=0.9,
            )
        ax.set_title(f"{symbol} - V2.2 OOS Equity")
        ax.set_ylabel("Equity")
        ax.legend(loc="best")
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Timestamp")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_v22_candidate_hardening(
    output_dir: str | Path = "outputs",
    symbols: list[str] | None = None,
    timeframe: str = DEFAULT_TIMEFRAME,
    source_csv: str | Path = "outputs/v22_mock_ohlcv.csv",
) -> V22RunArtifacts:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_symbols = symbols or list(DEFAULT_SYMBOLS)
    ensure_mock_ohlcv_csv(source_csv, symbols=target_symbols, periods=9_000, freq="1h", seed=22)
    raw = load_ohlcv_csv(source_csv)

    summary_rows: list[dict] = []
    trade_rows: list[dict] = []
    results_by_symbol: dict[str, WalkForwardResult] = {}

    param_grid = _trend_breakout_v22_grid()
    for symbol in target_symbols:
        symbol_df = _prepare_symbol_frame(raw, symbol=symbol, timeframe=timeframe)
        cost_model = _symbol_cost_model(symbol)

        wf = run_walk_forward(
            df=symbol_df,
            strategy_fn=trend_breakout_v2_signals,
            param_grid=param_grid,
            train_bars=2_200,
            test_bars=500,
            cost_model=cost_model,
            timeframe=timeframe,
            regime_column="stable_regime_label",
            meta_filter_class=RuleBasedMetaFilter,
            meta_filter_kwargs={"target_filter_rate": 0.4},
            meta_feature_builder=build_trade_meta_features,
            meta_label_builder=create_trade_success_labels,
            meta_label_kwargs={"horizon_bars": 24, "success_threshold": 0.0001},
            meta_apply_fn=apply_meta_trade_filter,
            meta_min_train_samples=20,
        )
        results_by_symbol[symbol] = wf

        unfiltered = wf.aggregate_metrics
        filtered = wf.filtered_aggregate_metrics or {}
        diag = wf.meta_filter_diagnostics or {}

        summary_rows.append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "bar_count": int(len(symbol_df)),
                "fold_count": int(len(wf.fold_results)),
                "unfiltered_sharpe": float(unfiltered.get("Sharpe", 0.0)),
                "filtered_sharpe": float(filtered.get("Sharpe", 0.0)),
                "unfiltered_cagr": float(unfiltered.get("CAGR", 0.0)),
                "filtered_cagr": float(filtered.get("CAGR", 0.0)),
                "unfiltered_max_drawdown": float(unfiltered.get("MaxDrawdown", 0.0)),
                "filtered_max_drawdown": float(filtered.get("MaxDrawdown", 0.0)),
                "unfiltered_expectancy": float(unfiltered.get("Expectancy", 0.0)),
                "filtered_expectancy": float(filtered.get("Expectancy", 0.0)),
                "unfiltered_trade_count": float(unfiltered.get("TradeCount", 0.0)),
                "filtered_trade_count": float(filtered.get("TradeCount", 0.0)),
                "avg_filter_rate": float(diag.get("AvgFilterRateByFold", 0.0)),
                "pct_folds_sharpe_improved": float(
                    diag.get("PctFoldsFilteredSharpeImproved", 0.0)
                ),
                "pct_folds_expectancy_improved": float(
                    diag.get("PctFoldsFilteredExpectancyImproved", 0.0)
                ),
                "pct_folds_drawdown_improved": float(
                    diag.get("PctFoldsFilteredDrawdownImproved", 0.0)
                ),
            }
        )

        trade_rows.append(_trade_distribution_rows(symbol, "unfiltered", wf.combined_trades))
        trade_rows.append(
            _trade_distribution_rows(symbol, "meta_filtered", wf.filtered_combined_trades)
        )

    summary = pd.DataFrame(summary_rows).sort_values("symbol").reset_index(drop=True)
    trade_distribution = pd.DataFrame(trade_rows).sort_values(["symbol", "variant"]).reset_index(
        drop=True
    )

    summary.to_csv(out_dir / "v22_candidate_summary.csv", index=False)
    trade_distribution.to_csv(out_dir / "v22_trade_distribution.csv", index=False)
    _plot_candidate_equity(results_by_symbol, out_dir / "v22_candidate_equity.png")

    return V22RunArtifacts(
        summary=summary,
        trade_distribution=trade_distribution,
        output_dir=out_dir,
    )
