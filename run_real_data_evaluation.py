from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from data import ensure_mock_ohlcv_csv, load_ohlcv_csv
from research import run_multi_symbol_evaluation


def _load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def _ensure_demo_real_csvs(symbols_config: str | Path) -> None:
    cfg = _load_yaml(symbols_config)
    symbols = cfg.get("symbols", [])
    if not symbols:
        return

    sample_path = ensure_mock_ohlcv_csv("data/sample_ohlcv.csv")
    sample = load_ohlcv_csv(sample_path)
    for entry in symbols:
        symbol = str(entry["symbol"])
        filepath = Path(entry["filepath"])
        if filepath.exists():
            continue
        filepath.parent.mkdir(parents=True, exist_ok=True)
        sym_df = sample[sample["symbol"] == symbol].copy()
        if sym_df.empty:
            continue
        sym_df.to_csv(filepath, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run real-data multi-symbol strict walk-forward evaluation."
    )
    parser.add_argument(
        "--data-sources",
        default="configs/data_sources.yaml",
        help="Path to data sources YAML config.",
    )
    parser.add_argument(
        "--symbols",
        default="configs/symbols.yaml",
        help="Path to symbols YAML config.",
    )
    parser.add_argument(
        "--outputs",
        default="outputs",
        help="Directory to write output artifacts.",
    )
    parser.add_argument(
        "--create-demo-if-missing",
        action="store_true",
        help="Create demo CSV files from synthetic sample if configured files are missing.",
    )
    args = parser.parse_args()

    if args.create_demo_if_missing:
        _ensure_demo_real_csvs(args.symbols)

    results = run_multi_symbol_evaluation(
        data_sources_config=args.data_sources,
        symbols_config=args.symbols,
        output_dir=args.outputs,
    )

    summary = results["multi_symbol_summary"]
    if summary.empty:
        print("No symbol results produced.")
        return

    print("\nMulti-symbol strict WF summary:")
    print(summary.to_string(index=False))
    print("\nSaved artifacts:")
    print(f"- {Path(args.outputs) / 'multi_symbol_summary.csv'}")
    print(f"- {Path(args.outputs) / 'multi_symbol_fold_summary.csv'}")
    print(f"- {Path(args.outputs) / 'multi_symbol_equity_comparison.png'}")
    print(f"- {Path(args.outputs) / 'multi_symbol_meta_diagnostics.csv'}")
    print(f"- {Path(args.outputs) / 'data_ingestion_audit.csv'}")
    print(f"- {Path(args.outputs) / 'data_quality_flags.csv'}")


if __name__ == "__main__":
    main()
