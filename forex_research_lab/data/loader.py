"""Utilities for loading and resampling OHLCV data."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd


SUPPORTED_TIMEFRAMES: Mapping[str, str] = {
    "H1": "1h",
    "H4": "4h",
    "D1": "1D",
}


def normalize_timeframe(timeframe: str) -> str:
    """Normalize a user-supplied timeframe into a pandas resample rule."""
    key = timeframe.upper()
    return SUPPORTED_TIMEFRAMES.get(key, timeframe)


def load_ohlcv_csv(path: str | Path, symbol: str | None = None) -> pd.DataFrame:
    """Load a single OHLCV CSV file into a timestamp-indexed dataframe."""
    csv_path = Path(path)
    dataframe = pd.read_csv(csv_path)

    required_columns = {"timestamp", "open", "high", "low", "close"}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns {sorted(missing_columns)} in {csv_path}")

    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"], utc=False)
    dataframe = dataframe.sort_values("timestamp").set_index("timestamp")

    if symbol is None:
        if "symbol" in dataframe.columns and not dataframe["symbol"].dropna().empty:
            symbol = str(dataframe["symbol"].dropna().iloc[0]).upper()
        else:
            symbol = csv_path.stem.upper()

    dataframe["symbol"] = symbol
    if "volume" not in dataframe.columns:
        dataframe["volume"] = 0.0

    ordered_columns = ["symbol", "open", "high", "low", "close", "volume"]
    optional_columns = [column for column in ("carry_proxy",) if column in dataframe.columns]
    return dataframe[ordered_columns + optional_columns]


def load_ohlcv_directory(directory: str | Path, pattern: str = "*.csv") -> dict[str, pd.DataFrame]:
    """Load all matching CSV files from a directory into a symbol-keyed dictionary."""
    directory_path = Path(directory)
    if not directory_path.exists():
        raise FileNotFoundError(f"Data directory does not exist: {directory_path}")

    datasets: dict[str, pd.DataFrame] = {}
    for path in sorted(directory_path.glob(pattern)):
        dataframe = load_ohlcv_csv(path)
        symbol = str(dataframe["symbol"].iloc[0]).upper()
        datasets[symbol] = dataframe

    if not datasets:
        raise FileNotFoundError(f"No OHLCV CSV files found in {directory_path}")
    return datasets


def resample_ohlcv(dataframe: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample OHLCV data to a new timeframe without introducing lookahead."""
    rule = normalize_timeframe(timeframe)
    result = dataframe.copy()

    symbol = str(result["symbol"].dropna().iloc[0]).upper() if "symbol" in result.columns else None
    aggregation = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    for optional_column in ("carry_proxy", "spread", "half_spread", "slippage_bps", "commission_bps"):
        if optional_column in result.columns:
            aggregation[optional_column] = "last"

    resampled = result.drop(columns=["symbol"], errors="ignore").resample(rule).agg(aggregation)
    resampled = resampled.dropna(subset=["open", "high", "low", "close"])

    if symbol is not None:
        resampled["symbol"] = symbol

    ordered_columns = ["symbol", "open", "high", "low", "close", "volume"]
    optional_columns = [column for column in resampled.columns if column not in ordered_columns]
    available_columns = [column for column in ordered_columns if column in resampled.columns]
    return resampled[available_columns + optional_columns]


def prepare_multi_timeframe(
    datasets: Mapping[str, pd.DataFrame],
    timeframes: tuple[str, ...] = ("H1", "H4", "D1"),
) -> dict[str, dict[str, pd.DataFrame]]:
    """Build a symbol -> timeframe -> dataframe mapping."""
    multi_timeframe: dict[str, dict[str, pd.DataFrame]] = {}
    for symbol, dataframe in datasets.items():
        multi_timeframe[symbol] = {}
        for timeframe in timeframes:
            multi_timeframe[symbol][timeframe] = resample_ohlcv(dataframe, timeframe)
    return multi_timeframe
