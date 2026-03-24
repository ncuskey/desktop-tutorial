from __future__ import annotations

import argparse

from research import run_regime_diagnostics


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run R1.1 regime diagnostics from existing fold artifacts."
    )
    parser.add_argument("--strategy", default="TrendBreakout_V2")
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--artifacts-root", default="outputs/TrendBreakout_V2")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--source-csv", default=None)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    artifacts = run_regime_diagnostics(
        strategy=args.strategy,
        symbols=args.symbols,
        artifacts_root=args.artifacts_root,
        output_dir=args.output_dir,
        price_csv=args.source_csv,
    )

    print("Regime diagnostics completed.")
    print(f"Fold diagnostics rows: {len(artifacts.fold_diagnostics)}")
    print(artifacts.bin_summary.head(10).to_string(index=False))
    print(
        "Outputs: "
        f"{args.output_dir}/regime_fold_diagnostics.csv, "
        f"{args.output_dir}/regime_feature_correlations.csv, "
        f"{args.output_dir}/regime_bin_summary.csv, "
        f"{args.output_dir}/regime_vs_sharpe.png, "
        f"{args.output_dir}/regime_bin_performance.png"
    )


if __name__ == "__main__":
    main()
