from __future__ import annotations

import argparse

from research import run_r14_execution_layer


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run R1.4.1 early-confirmation execution layer evaluation."
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["EURUSD", "GBPUSD", "AUDUSD"],
        help="Symbols to evaluate independently.",
    )
    parser.add_argument("--timeframe", default="H1")
    parser.add_argument("--early-window", type=int, default=3)
    parser.add_argument("--meta-threshold", type=float, default=0.60)
    parser.add_argument("--scale-threshold", type=float, default=0.75)
    parser.add_argument(
        "--min-hold-bars",
        type=int,
        default=None,
        help="Minimum hold bars before early-eval (defaults to early-window).",
    )
    parser.add_argument(
        "--disable-scaling",
        action="store_true",
        help="Disable early-confirm scaling even when score >= scale-threshold.",
    )
    parser.add_argument(
        "--fixed-size-only",
        action="store_true",
        help="Alias for disable-scaling to keep position sizing fixed.",
    )
    parser.add_argument("--scale-factor", type=float, default=1.5)
    parser.add_argument("--strategy", default="TrendBreakout_V2")
    parser.add_argument("--artifacts-root", default="outputs/TrendBreakout_V2")
    parser.add_argument("--source-csv", default=None)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--forward-horizon", type=int, default=24)
    parser.add_argument(
        "--label-method",
        default="top_quantile",
        choices=["top_quantile", "directional_accuracy", "cost_adjusted_return"],
    )
    parser.add_argument("--label-quantile", type=float, default=0.30)
    parser.add_argument("--meta-min-train-samples", type=int, default=30)
    parser.add_argument(
        "--allow-fallback-scorer",
        action="store_true",
        default=True,
        help="Enable deterministic fallback scorer when fold-local model cannot be fit (default: enabled).",
    )
    parser.add_argument(
        "--disable-fallback-scorer",
        action="store_true",
        help="Disable deterministic fallback scorer.",
    )
    parser.add_argument(
        "--enable-r15-rules",
        action="store_true",
        default=True,
        help="Enable fixed R1.5 failure-removal rules (default: enabled).",
    )
    parser.add_argument(
        "--disable-r15-rules",
        action="store_true",
        help="Disable fixed R1.5 failure-removal rules.",
    )
    parser.add_argument(
        "--enable-r153-composite",
        action="store_true",
        default=True,
        help="Enable R1.5.3 composite failure scoring exits (default: enabled).",
    )
    parser.add_argument(
        "--disable-r153-composite",
        action="store_true",
        help="Disable R1.5.3 composite failure scoring exits.",
    )
    parser.add_argument(
        "--enable-r16-position-sizing",
        action="store_true",
        default=True,
        help="Enable R1.6 confidence-based position resizing (default: enabled).",
    )
    parser.add_argument(
        "--disable-r16-position-sizing",
        action="store_true",
        help="Disable R1.6 confidence-based position resizing.",
    )
    parser.add_argument(
        "--max-position-size",
        type=float,
        default=2.0,
        help="Maximum absolute position size when R1.6 sizing is enabled.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    artifacts = run_r14_execution_layer(
        symbols=args.symbols,
        timeframe=args.timeframe,
        early_window=args.early_window,
        meta_threshold=args.meta_threshold,
        scale_threshold=args.scale_threshold,
        min_hold_bars=args.min_hold_bars or args.early_window,
        disable_scaling=args.disable_scaling,
        fixed_size_only=args.fixed_size_only,
        scale_factor=args.scale_factor,
        strategy=args.strategy,
        artifacts_root=args.artifacts_root,
        source_csv=args.source_csv,
        output_dir=args.output_dir,
        forward_horizon=args.forward_horizon,
        label_method=args.label_method,
        label_quantile=args.label_quantile,
        meta_min_train_samples=args.meta_min_train_samples,
        allow_fallback_scorer=(args.allow_fallback_scorer and not args.disable_fallback_scorer),
        enable_r15_rules=(args.enable_r15_rules and not args.disable_r15_rules),
        enable_r153_composite=(args.enable_r153_composite and not args.disable_r153_composite),
        enable_r16_position_sizing=(
            args.enable_r16_position_sizing and not args.disable_r16_position_sizing
        ),
        max_position_size=args.max_position_size,
    )

    print("R1.4.1 execution-layer run completed.")
    print(artifacts.comparison.round(6).to_string(index=False))
    print(f"Outputs written to: {artifacts.output_dir.resolve()}")


if __name__ == "__main__":
    main()
