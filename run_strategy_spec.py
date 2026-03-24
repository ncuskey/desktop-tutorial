from __future__ import annotations

import argparse

from research import generate_strategy_spec


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a formal strategy specification from R1 artifacts."
    )
    parser.add_argument("--strategy", default="TrendBreakout_V2")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--version-suffix", default="R1")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    spec, md_path, json_path = generate_strategy_spec(
        strategy=args.strategy,
        symbol=args.symbol,
        output_dir=args.output_dir,
        version_suffix=args.version_suffix,
    )

    print("Strategy spec generation completed.")
    print(f"Strategy: {spec.name}")
    print(f"Version: {spec.version}")
    print(f"Markdown: {md_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
