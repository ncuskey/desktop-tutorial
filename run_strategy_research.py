from __future__ import annotations

import argparse

from research import run_strategy_research


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run single-strategy research workflow (R1)."
    )
    parser.add_argument("--strategy-family", default="TrendBreakout_V2")
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--timeframe", default="H1")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument(
        "--search-mode",
        choices=("grid", "random"),
        default="grid",
    )
    parser.add_argument("--n-random-samples", type=int, default=50)
    parser.add_argument("--train-bars", type=int, default=2200)
    parser.add_argument("--test-bars", type=int, default=500)
    parser.add_argument("--purge-bars", type=int, default=0)
    parser.add_argument("--embargo-bars", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    artifacts = run_strategy_research(
        strategy_family=args.strategy_family,
        symbol=args.symbol,
        timeframe=args.timeframe,
        start=args.start,
        end=args.end,
        search_mode=args.search_mode,
        n_random_samples=args.n_random_samples,
        train_bars=args.train_bars,
        test_bars=args.test_bars,
        use_purge=(args.purge_bars > 0 or args.embargo_bars > 0),
        purge_bars=args.purge_bars,
        embargo_bars=args.embargo_bars,
        output_dir=args.output_dir,
    )

    print("Strategy research run completed.")
    print(artifacts.summary.round(6).to_string(index=False))
    print(artifacts.recommendation)
    print(f"Outputs written to: {artifacts.output_dir.resolve()}")


if __name__ == "__main__":
    main()
