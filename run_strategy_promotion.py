from __future__ import annotations

import argparse

from research import PromotionThresholds, run_strategy_promotion_framework


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify strategy promotion readiness from strict OOS R1 artifacts."
    )
    parser.add_argument("--strategy", default="TrendBreakout_V2")
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--artifacts-root", default="outputs")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--min-sharpe-promote", type=float, default=0.5)
    parser.add_argument("--min-positive-fold-pct-promote", type=float, default=0.6)
    parser.add_argument("--max-robustness-gap-ratio-promote", type=float, default=0.15)
    parser.add_argument("--min-positive-fold-pct-stability", type=float, default=0.35)
    parser.add_argument("--max-sharpe-std-stability", type=float, default=2.5)
    parser.add_argument("--max-expectancy-std-ratio", type=float, default=5.0)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    thresholds = PromotionThresholds(
        min_sharpe_promote=args.min_sharpe_promote,
        min_positive_fold_pct_promote=args.min_positive_fold_pct_promote,
        max_robustness_gap_ratio_promote=args.max_robustness_gap_ratio_promote,
        min_positive_fold_pct_stability=args.min_positive_fold_pct_stability,
        max_sharpe_std_stability=args.max_sharpe_std_stability,
        max_expectancy_std_ratio=args.max_expectancy_std_ratio,
    )

    artifacts = run_strategy_promotion_framework(
        strategy=args.strategy,
        symbols=args.symbols,
        artifacts_root=args.artifacts_root,
        output_dir=args.output_dir,
        thresholds=thresholds,
    )

    print("Strategy promotion framework completed.")
    print(artifacts.overview.to_string(index=False))
    print(f"Summary rows: {len(artifacts.summary)}")
    print(
        "Outputs: "
        f"{args.output_dir}/strategy_promotion_summary.csv, "
        f"{args.output_dir}/strategy_promotion_overview.csv, "
        f"{args.output_dir}/strategy_parameter_alignment.csv"
    )


if __name__ == "__main__":
    main()
