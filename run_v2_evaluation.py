from __future__ import annotations

import argparse
from pathlib import Path

from research.v2_runner import run_v2_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Forex Research Lab V2 evaluation.")
    parser.add_argument("--data-sources", default="configs/data_sources.yaml")
    parser.add_argument("--symbols", default="configs/symbols.yaml")
    parser.add_argument("--outputs", default="outputs")
    args = parser.parse_args()

    results = run_v2_evaluation(
        data_sources_config=args.data_sources,
        symbols_config=args.symbols,
        output_dir=args.outputs,
    )

    print("\nV2 outputs generated:")
    print(f"- {Path(args.outputs) / 'v2_sleeve_comparison.csv'}")
    print(f"- {Path(args.outputs) / 'v2_trade_quality_comparison.csv'}")
    print(f"- {Path(args.outputs) / 'v2_feature_ablation.csv'}")
    print(f"- {Path(args.outputs) / 'v2_threshold_stability.csv'}")
    print(f"- {Path(args.outputs) / 'v2_portfolio_summary.csv'}")
    print(f"- {Path(args.outputs) / 'v2_portfolio_equity.png'}")
    print("\nPreview:")
    for key, df in results.items():
        print(f"\n[{key}] rows={len(df)}")
        if not df.empty:
            print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
