from __future__ import annotations

import argparse

from research import run_r13_trend_gating


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run R1.3 Trend Gating (WF-safe, evidence-backed)."
    )
    parser.add_argument("--strategy", default="TrendBreakout_V2")
    parser.add_argument("--symbols", nargs="+", default=["EURUSD", "GBPUSD", "AUDUSD"])
    parser.add_argument("--artifacts-root", default="outputs/TrendBreakout_V2")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--source-csv", default=None)
    parser.add_argument("--timeframe", default="H1")
    parser.add_argument("--score-quantile", type=float, default=0.70)
    parser.add_argument("--top-k-percent", type=float, default=0.30)
    parser.add_argument("--min-trades-per-fold", type=int, default=5)
    parser.add_argument("--target-pass-rate-min", type=float, default=0.25)
    parser.add_argument("--target-pass-rate-max", type=float, default=0.40)
    parser.add_argument("--hybrid-tsmom-enabled", action="store_true")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    artifacts = run_r13_trend_gating(
        strategy=args.strategy,
        symbols=args.symbols,
        artifacts_root=args.artifacts_root,
        output_dir=args.output_dir,
        source_csv=args.source_csv,
        timeframe=args.timeframe,
        score_quantile=args.score_quantile,
        top_k_percent=args.top_k_percent,
        min_trades_per_fold=args.min_trades_per_fold,
        target_pass_rate_min=args.target_pass_rate_min,
        target_pass_rate_max=args.target_pass_rate_max,
        hybrid_tsmom_enabled=args.hybrid_tsmom_enabled,
    )
    print("R1.3 trend gating run completed.")
    print(artifacts.gate_comparison_by_fold.head(10).to_string(index=False))
    print(f"Trade feature rows: {len(artifacts.trade_feature_dataset)}")
    print("Outputs written under:", args.output_dir)


if __name__ == "__main__":
    main()
