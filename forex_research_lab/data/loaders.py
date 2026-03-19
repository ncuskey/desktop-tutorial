"""Data loading utilities for OHLCV Forex datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import pandas as pd


REQUIRED_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def _validate_ohlcv_columns(df: pd.DataFrame, path: Path) -> None:
    missing = [col for col in REQUIRED_OHLCV_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in {path}")


def load_ohlcv_csv(
    path: str | Path,
    datetime_col: str = "timestamp",
    tz: str | None = "UTC",
) -> pd.DataFrame:
    """
    Load OHLCV data from a CSV file.

    Expected columns:
    - timestamp (or custom datetime_col)
    - open, high, low, close, volume
    - optional spread_pips
    """

    file_path = Path(path)
    df = pd.read_csv(file_path, parse_dates=[datetime_col])
    if datetime_col != "timestamp":
        df = df.rename(columns={datetime_col: "timestamp"})

    if "timestamp" not in df.columns:
        raise ValueError(f"'timestamp' column not found in {file_path}")

    df = df.set_index("timestamp").sort_index()
    if tz is not None:
        # If timestamps are naive, localize; otherwise convert.
        if df.index.tz is None:
            df.index = df.index.tz_localize(tz)
        else:
            df.index = df.index.tz_convert(tz)

    _validate_ohlcv_columns(df, file_path)

    # Guarantee numeric dtypes for the core fields.
    for col in REQUIRED_OHLCV_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "spread_pips" in df.columns:
        df["spread_pips"] = pd.to_numeric(df["spread_pips"], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"])
    return df


def load_multi_symbol_data(
    data_dir: str | Path,
    symbols: Iterable[str],
    timeframe: str = "H1",
) -> Dict[str, pd.DataFrame]:
    """
    Load multiple symbols from `<symbol>_<timeframe>.csv` files.
    """

    base_dir = Path(data_dir)
    symbol_data: Dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        file_path = base_dir / f"{symbol}_{timeframe}.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        df = load_ohlcv_csv(file_path)
        df["symbol"] = symbol
        symbol_data[symbol] = df
    return symbol_data
