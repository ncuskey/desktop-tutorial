from __future__ import annotations

import argparse

from data.dukascopy_loader import pull_dukascopy_history_to_canonical


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Dukascopy historical data via duka-dl and materialize canonical CSVs."
    )
    parser.add_argument(
        "--symbols",
        default="EURUSD,GBPUSD,USDJPY,AUDUSD",
        help="Comma-separated 6-char symbols (default: EURUSD,GBPUSD,USDJPY,AUDUSD).",
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Start date in DD-MM-YYYY format for duka-dl.",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="End date in DD-MM-YYYY format for duka-dl.",
    )
    parser.add_argument(
        "--timeframe",
        default="H1",
        help="Target canonical timeframe (default: H1).",
    )
    parser.add_argument(
        "--raw-dir",
        default="data/dukascopy_raw",
        help="Raw duka-dl output directory.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/real",
        help="Canonical output directory.",
    )
    parser.add_argument(
        "--mode",
        default="BID",
        help="BID or ASK mode (default: BID).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=20,
        help="duka-dl thread count (default: 20).",
    )
    parser.add_argument(
        "--semaphore",
        type=int,
        default=50,
        help="duka-dl async semaphore (default: 50).",
    )
    parser.add_argument(
        "--parquet",
        action="store_true",
        help="Use parquet output from duka-dl as intermediate files.",
    )
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    results = pull_dukascopy_history_to_canonical(
        symbols=symbols,
        start_date=args.start,
        end_date=args.end,
        target_timeframe=args.timeframe,
        raw_output_folder=args.raw_dir,
        canonical_output_folder=args.out_dir,
        mode=args.mode,
        threads=args.threads,
        semaphore=args.semaphore,
        parquet=args.parquet,
    )

    if not results:
        print("No symbols processed.")
        return

    print("Dukascopy download + canonicalization completed:")
    for row in results:
        print(
            f"- {row.symbol}: raw_rows={row.raw_rows}, "
            f"canonical_rows={row.canonical_rows}, "
            f"raw={row.raw_file}, canonical={row.canonical_file}"
        )


if __name__ == "__main__":
    main()
