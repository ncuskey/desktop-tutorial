"""Data loading utilities — CSV files and synthetic sample data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_ohlcv(path: str, symbol: str | None = None) -> pd.DataFrame:
    """Load OHLCV data from a CSV file.

    Expected columns: datetime (or as index), open, high, low, close, volume.
    A ``symbol`` column is added when *symbol* is supplied.
    """
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    df.columns = [c.strip().lower() for c in df.columns]
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    if symbol:
        df["symbol"] = symbol
    df.sort_index(inplace=True)
    return df


def generate_sample_data(
    symbol: str = "EURUSD",
    periods: int = 5000,
    freq: str = "h",
    start: str = "2020-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data resembling a forex pair."""
    rng = np.random.default_rng(seed)
    base_prices = {
        "EURUSD": 1.10,
        "GBPUSD": 1.30,
        "USDJPY": 110.0,
        "AUDUSD": 0.72,
    }
    base = base_prices.get(symbol, 1.0)
    vol = 0.0003 if "JPY" not in symbol else 0.03

    returns = rng.normal(0, vol, periods)
    close = base * np.exp(np.cumsum(returns))

    high = close * (1 + np.abs(rng.normal(0, vol * 0.5, periods)))
    low = close * (1 - np.abs(rng.normal(0, vol * 0.5, periods)))
    open_ = low + (high - low) * rng.uniform(0.2, 0.8, periods)
    volume = rng.integers(100, 10000, periods).astype(float)

    idx = pd.date_range(start=start, periods=periods, freq=freq)
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
    df.index.name = "datetime"
    df["symbol"] = symbol
    return df
