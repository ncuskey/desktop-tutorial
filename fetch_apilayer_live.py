from __future__ import annotations

import argparse
import os
from pathlib import Path

from data.apilayer_loader import (
    append_snapshot_rows_to_symbol_csv,
    build_canonical_snapshot,
    fetch_apilayer_live_quotes,
)


def _required_currencies_from_symbols(symbols: list[str], source: str) -> list[str]:
    source = source.upper()
    cur: set[str] = set()
    for sym in symbols:
        s = sym.upper()
        if len(s) != 6:
            continue
        base = s[:3]
        quote = s[3:]
        if base != source:
            cur.add(base)
        if quote != source:
            cur.add(quote)
    return sorted(cur)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch live FX quotes from apilayer and append canonical snapshot CSV rows."
    )
    parser.add_argument(
        "--access-key",
        default=os.getenv("APILAYER_ACCESS_KEY", ""),
        help="apilayer access key (or set APILAYER_ACCESS_KEY env var).",
    )
    parser.add_argument(
        "--endpoint",
        default="http://apilayer.net/api/live",
        help="apilayer live endpoint URL.",
    )
    parser.add_argument(
        "--source",
        default="USD",
        help="Source currency expected by API (default: USD).",
    )
    parser.add_argument(
        "--symbols",
        default="EURUSD,GBPUSD,USDJPY,AUDUSD",
        help="Comma-separated target symbols to materialize.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/real",
        help="Directory to append per-symbol CSVs.",
    )
    args = parser.parse_args()

    if not args.access_key:
        raise ValueError("Missing access key. Pass --access-key or set APILAYER_ACCESS_KEY.")

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    currencies = _required_currencies_from_symbols(symbols, source=args.source)
    if not currencies:
        raise ValueError("No valid 6-character symbols provided.")

    live = fetch_apilayer_live_quotes(
        access_key=args.access_key,
        currencies=currencies,
        source=args.source.upper(),
        endpoint=args.endpoint,
    )
    if not live.raw_success:
        raise RuntimeError(f"apilayer API call failed: {live.raw_error}")

    snapshot = build_canonical_snapshot(live, symbols=symbols)
    written = append_snapshot_rows_to_symbol_csv(snapshot_df=snapshot, output_dir=args.output_dir)

    print(f"Fetched timestamp: {live.timestamp}")
    print(f"Source: {live.source}")
    print(f"Symbols requested: {len(symbols)} | rows materialized: {len(snapshot)}")
    if written:
        print("Updated files:")
        for p in written:
            print(f"- {Path(p)}")
    else:
        print("No rows written (likely missing required quotes for requested symbols).")


if __name__ == "__main__":
    main()
