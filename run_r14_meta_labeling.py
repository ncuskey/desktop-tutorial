from __future__ import annotations

import argparse

from research import run_r14_meta_labeling


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run R1.4 meta-labeling for breakout follow-through."
    )
    parser.add_argument("--strategy", default="TrendBreakout_V2")
    parser.add_argument("--symbols", nargs="+", default=["EURUSD", "GBPUSD", "AUDUSD"])
    parser.add_argument("--artifacts-root", default="outputs/TrendBreakout_V2")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--source-csv", default=None)
    parser.add_argument("--timeframe", default="H1")
    parser.add_argument("--tp-atr-mult", type=float, default=1.0)
    parser.add_argument("--sl-atr-mult", type=float, default=0.5)
    parser.add_argument("--max-horizon", type=int, default=24)
    parser.add_argument("--early-window", type=int, default=3)
    parser.add_argument("--top-k-percent", type=float, default=0.30)
    parser.add_argument("--min-trades-per-fold", type=int, default=5)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    artifacts = run_r14_meta_labeling(
        strategy=args.strategy,
        symbols=args.symbols,
        artifacts_root=args.artifacts_root,
        output_dir=args.output_dir,
        source_csv=args.source_csv,
        timeframe=args.timeframe,
        tp_atr_mult=args.tp_atr_mult,
        sl_atr_mult=args.sl_atr_mult,
        max_horizon=args.max_horizon,
        early_window=args.early_window,
        top_k_percent=args.top_k_percent,
        min_trades_per_fold=args.min_trades_per_fold,
    )
    print("R1.4 meta-labeling run completed.")
    print(artifacts.meta_gate_comparison.head(10).to_string(index=False))
    print(f"Meta feature rows: {len(artifacts.meta_feature_dataset)}")
    print("Outputs written under:", args.output_dir)


if __name__ == "__main__":
    main()

