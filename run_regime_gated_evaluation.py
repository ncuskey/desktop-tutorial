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

    parser.add_argument("--adx-slope-window", type=int, default=5)
    parser.add_argument("--adx-slope-threshold", type=float, default=0.0)
    parser.add_argument("--atr-expansion-threshold", type=float, default=1.05)
    parser.add_argument("--compression-threshold", type=float, default=0.2)
    parser.add_argument("--atr-expansion-window", type=int, default=20)
    parser.add_argument("--compression-window", type=int, default=20)
    parser.add_argument("--low-vol-percentile", type=float, default=0.3)
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
            "adx_slope_window": args.adx_slope_window,
            "adx_slope_threshold": args.adx_slope_threshold,
            "atr_expansion_threshold": args.atr_expansion_threshold,
            "compression_threshold": args.compression_threshold,
            "atr_expansion_window": args.atr_expansion_window,
            "compression_window": args.compression_window,
            "low_vol_percentile": args.low_vol_percentile,
        },
    )

    print("Regime-gated evaluation completed.")
    print(artifacts.comparison.to_string(index=False))
    print(f"Fold rows: {len(artifacts.fold_results)}")
    print(
        "Outputs: "
        f"{args.output_dir}/regime_gated_comparison.csv, "
        f"{args.output_dir}/regime_gated_fold_results.csv, "
        f"{args.output_dir}/regime_gated_equity.png, "
        f"{args.output_dir}/state_filter_diagnostics.csv"
    )


if __name__ == "__main__":
    main()
