from __future__ import annotations

import argparse

from research import run_r15_failure_decomposition


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run R1.5 failure decomposition (winner vs loser diagnostics)."
    )
    parser.add_argument(
        "--input",
        default="outputs/meta_feature_dataset.csv",
        help="Input CSV with trade-level features and realized_return.",
    )
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument(
        "--top-n-features",
        type=int,
        default=5,
        help="Top separated features to use for rule generation.",
    )
    parser.add_argument("--timeframe", default="H1")
    parser.add_argument("--early-window", type=int, default=3)
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["EURUSD", "GBPUSD", "AUDUSD"],
        help="Symbols to use when rebuilding dataset if input is missing.",
    )
    parser.add_argument("--artifacts-root", default="outputs/TrendBreakout_V2")
    parser.add_argument("--source-csv", default=None)
    parser.add_argument(
        "--no-rebuild-if-missing",
        action="store_true",
        help="Fail if --input is missing instead of rebuilding from baseline artifacts.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    artifacts = run_r15_failure_decomposition(
        input_path=args.input,
        output_dir=args.output_dir,
        top_n_features=args.top_n_features,
        symbols=args.symbols,
        timeframe=args.timeframe,
        early_window=args.early_window,
        artifacts_root=args.artifacts_root,
        source_csv=args.source_csv,
        rebuild_if_missing=not args.no_rebuild_if_missing,
    )
    print("R1.5 failure decomposition completed.")
    print(artifacts.feature_separation.head(10).round(6).to_string(index=False))
    print(f"Outputs written to: {artifacts.output_dir.resolve()}")


if __name__ == "__main__":
    main()
