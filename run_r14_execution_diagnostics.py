from __future__ import annotations

import argparse

from research import run_r14_execution_diagnostics


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run R1.4.1a execution-layer diagnostics (threshold/calibration sweeps)."
    )
    parser.add_argument(
        "--input",
        default="outputs/r14_execution_trade_scores.csv",
        help=(
            "Trade-level input CSV with columns: symbol, fold_id, meta_score, "
            "realized_return, early_exit_flag (trade_id optional)."
        ),
    )
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--n-bins", type=int, default=5)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    artifacts = run_r14_execution_diagnostics(
        input_path=args.input,
        output_dir=args.output_dir,
        n_bins=args.n_bins,
    )

    print("R1.4.1a execution diagnostics completed.")
    print(artifacts.threshold_sweep.round(6).to_string(index=False))
    print(artifacts.separation_tests.round(6).to_string(index=False))
    print(f"Outputs written to: {artifacts.output_dir.resolve()}")


if __name__ == "__main__":
    main()
