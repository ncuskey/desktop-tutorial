from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import shutil

import numpy as np
import pandas as pd

from data.real_loader import normalize_fx_dataframe


@dataclass
class DukascopyFetchResult:
    symbol: str
    raw_file: Path
    canonical_file: Path
    raw_rows: int
    canonical_rows: int
    detected_source_timeframe: str
    target_timeframe: str


def _symbol_to_duka_pair(symbol: str) -> str:
    s = symbol.upper()
    if len(s) != 6:
        raise ValueError(f"Expected 6-character FX symbol like EURUSD, got: {symbol}")
    return f"{s[:3]}-{s[3:]}"


def _newest_matching_file(folder: Path, pair: str, extension: str = "csv") -> Path:
    candidates = sorted(
        folder.glob(f"{pair}(*)*.{extension}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No downloaded duka-dl {extension} output found for pair {pair}")
    return candidates[0]


def run_duka_dl_download(
    symbol: str,
    start_date: str,
    end_date: str,
    output_folder: str | Path = "data/dukascopy_raw",
    mode: str = "BID",
    threads: int = 20,
    semaphore: int = 50,
    parquet: bool = False,
) -> Path:
    """
    Run duka-dl CLI and return downloaded raw file path.
    Date format must be DD-MM-YYYY to match duka-dl CLI.
    """
    pair = _symbol_to_duka_pair(symbol)
    out = Path(output_folder)
    out.mkdir(parents=True, exist_ok=True)

    executable = shutil.which("duka-dl")
    if executable is None:
        candidate = Path.home() / ".local" / "bin" / "duka-dl"
        if candidate.exists():
            executable = str(candidate)
        else:
            raise FileNotFoundError(
                "duka-dl executable not found on PATH. Install with `python3 -m pip install duka-dl`."
            )

    before = {p.resolve() for p in out.glob(f"{pair}(*)*.csv")} | {
        p.resolve() for p in out.glob(f"{pair}(*)*.parquet")
    }
    cmd = [
        executable,
        pair,
        "-s",
        start_date,
        "-e",
        end_date,
        "-f",
        str(out),
        "-m",
        mode.upper(),
        "-t",
        str(int(threads)),
        "-se",
        str(int(semaphore)),
    ]
    if parquet:
        cmd.append("--parquet")

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        raise RuntimeError(
            f"duka-dl failed for {symbol}.\nCommand: {' '.join(cmd)}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        )

    ext = "parquet" if parquet else "csv"
    after = {p.resolve() for p in out.glob(f"{pair}(*)*.{ext}")}
    new_files = sorted(after - before, key=lambda p: p.stat().st_mtime, reverse=True)
    if new_files:
        return new_files[0]
    return _newest_matching_file(out, pair, extension=ext)


def load_dukascopy_ohlc(file_path: str | Path, symbol: str) -> pd.DataFrame:
    """
    Load duka-dl CSV/Parquet output and map to canonical pre-normalization columns.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dukascopy file not found: {path}")
    if path.suffix.lower() == ".parquet":
        src = pd.read_parquet(path)
    else:
        src = pd.read_csv(path)
    if src.empty:
        raise ValueError(f"Dukascopy file is empty: {path}")

    required = ["timestamp", "open", "high", "low", "close"]
    missing = [c for c in required if c not in src.columns]
    if missing:
        raise ValueError(f"Dukascopy file missing required columns {missing}: {path}")

    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(src["timestamp"], errors="coerce", utc=True),
            "symbol": symbol.upper(),
            "open": pd.to_numeric(src["open"], errors="coerce"),
            "high": pd.to_numeric(src["high"], errors="coerce"),
            "low": pd.to_numeric(src["low"], errors="coerce"),
            "close": pd.to_numeric(src["close"], errors="coerce"),
            "volume": pd.to_numeric(src["volume"], errors="coerce")
            if "volume" in src.columns
            else np.nan,
            "spread_bps": np.nan,
        }
    )
    out = out.dropna(subset=["timestamp", "open", "high", "low", "close"]).copy()
    return out.sort_values("timestamp").reset_index(drop=True)


def pull_dukascopy_history_to_canonical(
    symbols: Iterable[str],
    start_date: str,
    end_date: str,
    target_timeframe: str = "H1",
    raw_output_folder: str | Path = "data/dukascopy_raw",
    canonical_output_folder: str | Path = "data/real",
    mode: str = "BID",
    threads: int = 20,
    semaphore: int = 50,
    parquet: bool = False,
) -> list[DukascopyFetchResult]:
    """
    Download minute Dukascopy history per symbol and save canonical H1/H4/D1 CSV files.
    """
    canonical_dir = Path(canonical_output_folder)
    canonical_dir.mkdir(parents=True, exist_ok=True)

    results: list[DukascopyFetchResult] = []
    for symbol in symbols:
        raw_path = run_duka_dl_download(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            output_folder=raw_output_folder,
            mode=mode,
            threads=threads,
            semaphore=semaphore,
            parquet=parquet,
        )
        raw_df = load_dukascopy_ohlc(raw_path, symbol=symbol)
        detected = "1m" if len(raw_df) > 2 else "unknown"
        canonical = normalize_fx_dataframe(raw_df, symbol=symbol, timeframe=target_timeframe)
        out_file = canonical_dir / f"{symbol.upper()}_{target_timeframe.upper()}.csv"
        canonical.to_csv(out_file, index=False)
        results.append(
            DukascopyFetchResult(
                symbol=symbol.upper(),
                raw_file=raw_path,
                canonical_file=out_file,
                raw_rows=int(len(raw_df)),
                canonical_rows=int(len(canonical)),
                detected_source_timeframe=detected,
                target_timeframe=target_timeframe.upper(),
            )
        )
    return results
