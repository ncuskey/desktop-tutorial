from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

from data import ensure_mock_ohlcv_csv, load_ohlcv_csv
from data.apilayer_loader import (
    append_snapshot_rows_to_symbol_csv,
    build_canonical_snapshot,
    fetch_apilayer_live_quotes,
)
from data.dukascopy_loader import pull_dukascopy_history_to_canonical
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


def _configured_symbols(symbols_config: str | Path) -> list[str]:
    cfg = _load_yaml(symbols_config)
    rows = cfg.get("symbols", [])
    out: list[str] = []
    for row in rows:
        symbol = str(row.get("symbol", "")).upper()
        if len(symbol) == 6:
            out.append(symbol)
    return out


def _required_currencies_from_symbols(symbols: list[str], source: str) -> list[str]:
    source = source.upper()
    cur: set[str] = set()
    for sym in symbols:
        base = sym[:3]
        quote = sym[3:]
        if base != source:
            cur.add(base)
        if quote != source:
            cur.add(quote)
    return sorted(cur)


def _pull_dukascopy_history(
    symbols_path: str | Path,
    start_date: str,
    end_date: str,
    target_timeframe: str = "H1",
    raw_dir: str = "data/dukascopy_raw",
    out_dir: str = "data/real",
    mode: str = "BID",
    threads: int = 20,
    semaphore: int = 50,
) -> None:
    symbols = _configured_symbols(symbols_path)
    if not symbols:
        raise ValueError("No valid symbols in symbols config.")
    rows = pull_dukascopy_history_to_canonical(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        target_timeframe=target_timeframe,
        raw_output_folder=raw_dir,
        canonical_output_folder=out_dir,
        mode=mode,
        threads=threads,
        semaphore=semaphore,
    )
    print("Pulled Dukascopy history and wrote canonical files:")
    for r in rows:
        print(
            f"- {r.symbol}: raw_rows={r.raw_rows}, canonical_rows={r.canonical_rows}, "
            f"file={r.canonical_file}"
        )


def _pull_apilayer_live_snapshot(
    data_sources_path: str | Path,
    symbols_path: str | Path,
    explicit_access_key: str | None = None,
) -> None:
    sources = _load_yaml(data_sources_path)
    api_cfg = sources.get("api_sources", {}).get("apilayer_live", {})
    endpoint = str(api_cfg.get("endpoint", "http://apilayer.net/api/live"))
    source = str(api_cfg.get("source", "USD")).upper()
    key_env = str(api_cfg.get("access_key_env", "APILAYER_ACCESS_KEY"))
    access_key = explicit_access_key or os.getenv(key_env, "")
    if not access_key:
        raise ValueError(f"Missing apilayer key. Set {key_env} or pass --apilayer-access-key.")

    symbols = _configured_symbols(symbols_path)
    if not symbols:
        raise ValueError("No valid symbols in symbols config.")
    currencies = _required_currencies_from_symbols(symbols, source=source)
    live = fetch_apilayer_live_quotes(
        access_key=access_key,
        currencies=currencies,
        source=source,
        endpoint=endpoint,
    )
    if not live.raw_success:
        raise RuntimeError(f"apilayer live request failed: {live.raw_error}")
    snapshot = build_canonical_snapshot(live, symbols=symbols)
    append_snapshot_rows_to_symbol_csv(snapshot, output_dir="data/real")
    print(f"Pulled apilayer snapshot at {live.timestamp} for {len(snapshot)} symbol rows.")


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
    parser.add_argument(
        "--pull-apilayer-live",
        action="store_true",
        help="Fetch one live apilayer snapshot and append to data/real CSVs before evaluation.",
    )
    parser.add_argument(
        "--apilayer-access-key",
        default=None,
        help="Optional apilayer key (defaults to configured env var).",
    )
    parser.add_argument(
        "--pull-duka-history",
        action="store_true",
        help="Fetch Dukascopy history (via duka-dl) and write canonical data/real files.",
    )
    parser.add_argument(
        "--duka-start",
        default=None,
        help="Dukascopy start date DD-MM-YYYY (required with --pull-duka-history).",
    )
    parser.add_argument(
        "--duka-end",
        default=None,
        help="Dukascopy end date DD-MM-YYYY (required with --pull-duka-history).",
    )
    parser.add_argument(
        "--duka-mode",
        default="BID",
        help="Dukascopy price mode: BID or ASK (default BID).",
    )
    parser.add_argument(
        "--duka-threads",
        type=int,
        default=20,
        help="duka-dl thread count (default 20).",
    )
    parser.add_argument(
        "--duka-semaphore",
        type=int,
        default=50,
        help="duka-dl async semaphore (default 50).",
    )
    args = parser.parse_args()

    if args.create_demo_if_missing:
        _ensure_demo_real_csvs(args.symbols)
    if args.pull_duka_history:
        if not args.duka_start or not args.duka_end:
            raise ValueError("--pull-duka-history requires --duka-start and --duka-end.")
        _pull_dukascopy_history(
            symbols_path=args.symbols,
            start_date=args.duka_start,
            end_date=args.duka_end,
            target_timeframe="H1",
            raw_dir="data/dukascopy_raw",
            out_dir="data/real",
            mode=args.duka_mode,
            threads=args.duka_threads,
            semaphore=args.duka_semaphore,
        )
    if args.pull_apilayer_live:
        _pull_apilayer_live_snapshot(
            data_sources_path=args.data_sources,
            symbols_path=args.symbols,
            explicit_access_key=args.apilayer_access_key,
        )

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
