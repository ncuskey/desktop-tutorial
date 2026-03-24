from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.v22_runner import V22RunArtifacts, run_v22_candidate_hardening


def run_v23_edge_amplification(
    output_dir: str | Path = "outputs",
    symbols: list[str] | None = None,
    timeframe: str = "H1",
    source_csv: str | Path = "outputs/v23_mock_ohlcv.csv",
) -> V22RunArtifacts:
    # V2.3 reuses the V2.2 runner pipeline with hardened defaults already baked into
    # trend_breakout_v2 and tighter meta-filter settings.
    return run_v22_candidate_hardening(
        output_dir=output_dir,
        symbols=symbols,
        timeframe=timeframe,
        source_csv=source_csv,
    )


def summarize_tail_metrics(
    trade_distribution: pd.DataFrame,
    summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []
    filtered = trade_distribution[trade_distribution["variant"] == "meta_filtered"].copy()
    for _, row in filtered.iterrows():
        symbol = str(row["symbol"])
        summary_row = summary[summary["symbol"] == symbol]
        trade_count = float(row.get("trade_count", 0.0))
        if summary_row.empty:
            sharpe = 0.0
            expectancy = 0.0
        else:
            sharpe = float(summary_row.iloc[0]["filtered_sharpe"])
            expectancy = float(summary_row.iloc[0]["filtered_expectancy"])

        p10 = float(row.get("p10_trade_return", 0.0))
        p90 = float(row.get("p90_trade_return", 0.0))
        p95 = float(row.get("p95_trade_return", p90))
        p99 = float(row.get("p99_trade_return", p95))
        right_tail_ratio = float(p90 / abs(p10)) if abs(p10) > 1e-12 else 0.0
        rows.append(
            {
                "symbol": symbol,
                "variant": "meta_filtered",
                "trade_count": trade_count,
                "filtered_sharpe": sharpe,
                "filtered_expectancy": expectancy,
                "p90_trade_return": p90,
                "p95_trade_return": p95,
                "p99_trade_return": p99,
                "right_tail_ratio_p90_over_abs_p10": right_tail_ratio,
                "pct_trades_contributing_50pct_pnl": float(
                    row.get("pct_trades_contributing_50pct_pnl", 0.0)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)
