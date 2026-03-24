from __future__ import annotations

import argparse

from research import run_regime_gated_evaluation


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run R1.2 regime-gated evaluation using existing R1 artifacts."
    )
    parser.add_argument("--strategy", default="TrendBreakout_V2")
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--artifacts-root", default="outputs/TrendBreakout_V2")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--use-hardened-default", action="store_true")
    parser.add_argument("--source-csv", default=None)

    parser.add_argument("--threshold-adx", type=float, default=25.0)
    parser.add_argument("--threshold-pct", type=float, default=0.3)
    parser.add_argument("--use-trend-variance", action="store_true")
    parser.add_argument("--trend-variance-threshold", type=float, default=None)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if not args.use_hardened_default:
        raise ValueError("R1.2 requires --use-hardened-default.")

    artifacts = run_regime_gated_evaluation(
        strategy=args.strategy,
        symbols=args.symbols,
        artifacts_root=args.artifacts_root,
        output_dir=args.output_dir,
        use_hardened_default=args.use_hardened_default,
        source_csv=args.source_csv,
        gating_params={
            "threshold_adx": args.threshold_adx,
            "threshold_pct": args.threshold_pct,
            "use_trend_variance": args.use_trend_variance,
            "trend_variance_threshold": args.trend_variance_threshold,
        },
    )

    print("Regime-gated evaluation completed.")
    print(artifacts.comparison.to_string(index=False))
    print(f"Fold rows: {len(artifacts.fold_results)}")
    print(
        "Outputs: "
        f"{args.output_dir}/regime_gated_comparison.csv, "
        f"{args.output_dir}/regime_gated_fold_results.csv, "
        f"{args.output_dir}/regime_gated_equity.png"
    )


if __name__ == "__main__":
    main()
