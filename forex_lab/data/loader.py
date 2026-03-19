"""OHLCV data loading and mock data generation."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def load_ohlcv(
    path: str | Path,
    symbol: Optional[str] = None,
    date_column: str = "date",
) -> pd.DataFrame:
    """
    Load OHLCV data from CSV.

    Expected columns: open, high, low, close, volume (or date/datetime)
    """
    df = pd.read_csv(path)

    # Normalize column names
    col_map = {c.lower(): c for c in df.columns}
    if "datetime" in col_map:
        df = df.rename(columns={col_map["datetime"]: "date"})
    elif "timestamp" in col_map:
        df = df.rename(columns={col_map["timestamp"]: "date"})

    if date_column in df.columns:
        df["date"] = pd.to_datetime(df[date_column])
        df = df.drop(columns=[date_column], errors="ignore")
    elif "date" not in df.columns and len(df.columns) > 0:
        # Assume first column is datetime
        df["date"] = pd.to_datetime(df.iloc[:, 0])
        df = df.drop(columns=df.columns[0], errors="ignore")

    df = df.set_index("date").sort_index()
    df.index = pd.DatetimeIndex(df.index)

    required = ["open", "high", "low", "close"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    if "volume" not in df.columns:
        df["volume"] = 0.0

    if symbol:
        df["symbol"] = symbol

    return df


def generate_mock_data(
    n_bars: int = 10000,
    symbol: str = "EURUSD",
    start_date: Optional[str] = None,
    freq: str = "1h",
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data for testing.

    Uses geometric Brownian motion for price simulation.
    """
    if seed is not None:
        np.random.seed(seed)

    if start_date is None:
        start_date = "2020-01-01"

    dates = pd.date_range(start=start_date, periods=n_bars, freq=freq)

    # Geometric Brownian motion parameters
    mu = 0.00002  # drift
    sigma = 0.005  # volatility
    s0 = 1.1000  # initial price (e.g. EURUSD)

    returns = np.random.normal(mu, sigma, n_bars)
    log_prices = np.log(s0) + np.cumsum(returns)
    close = np.exp(log_prices)

    # Build OHLC from close
    high = close * (1 + np.abs(np.random.normal(0, 0.002, n_bars)))
    low = close * (1 - np.abs(np.random.normal(0, 0.002, n_bars)))
    open_ = np.roll(close, 1)
    open_[0] = s0

    # Ensure OHLC consistency
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    volume = np.random.lognormal(10, 1, n_bars).astype(int)

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )
    df.index.name = "date"
    df["symbol"] = symbol

    return df
