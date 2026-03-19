"""OHLCV data loading and sample data generation."""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


def load_ohlcv(
    path: str | Path,
    symbol: Optional[str] = None,
    date_col: str = "datetime",
    parse_dates: bool = True,
) -> pd.DataFrame:
    """Load OHLCV data from a CSV file.

    Expected columns: datetime, open, high, low, close, volume
    Returns a DataFrame indexed by datetime with lowercase column names.
    """
    df = pd.read_csv(path, parse_dates=[date_col] if parse_dates else False)
    df.columns = df.columns.str.lower().str.strip()
    if date_col.lower() in df.columns:
        df = df.set_index(date_col.lower())
    df = df.sort_index()
    if symbol:
        df["symbol"] = symbol
    return df


def resample_timeframe(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample OHLCV data to a different timeframe.

    timeframe: pandas offset alias, e.g. '1h', '4h', '1D'
    """
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    cols_present = {c: agg[c] for c in agg if c in df.columns}
    resampled = df.resample(timeframe).agg(cols_present).dropna(subset=["close"])

    for col in df.columns:
        if col not in cols_present and col != "symbol":
            resampled[col] = df[col].resample(timeframe).last()

    if "symbol" in df.columns:
        resampled["symbol"] = df["symbol"].iloc[0]
    return resampled


def generate_sample_data(
    symbol: str = "EURUSD",
    periods: int = 5000,
    freq: str = "1h",
    start: str = "2020-01-01",
    base_price: float = 1.10,
    volatility: float = 0.0005,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic synthetic OHLCV data for testing.

    Uses geometric Brownian motion with intrabar noise.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range(start=start, periods=periods, freq=freq)

    returns = rng.normal(0, volatility, size=periods)
    close = base_price * np.exp(np.cumsum(returns))

    intra_noise = rng.uniform(0.0001, 0.0008, size=periods)
    high = close * (1 + intra_noise)
    low = close * (1 - intra_noise)

    open_shift = rng.normal(0, volatility * 0.3, size=periods)
    open_prices = np.roll(close, 1) * (1 + open_shift)
    open_prices[0] = base_price

    open_prices = np.clip(open_prices, low, high)

    volume = rng.integers(100, 10000, size=periods).astype(float)

    df = pd.DataFrame(
        {
            "open": open_prices,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )
    df.index.name = "datetime"
    df["symbol"] = symbol
    return df


SYMBOL_CONFIGS = {
    "EURUSD": {"base_price": 1.10, "volatility": 0.0004, "seed": 42},
    "GBPUSD": {"base_price": 1.27, "volatility": 0.0005, "seed": 43},
    "USDJPY": {"base_price": 148.0, "volatility": 0.0006, "seed": 44},
    "AUDUSD": {"base_price": 0.65, "volatility": 0.0005, "seed": 45},
}


def generate_multi_symbol_data(
    symbols: Optional[list[str]] = None,
    periods: int = 5000,
    freq: str = "1h",
) -> dict[str, pd.DataFrame]:
    """Generate sample data for multiple currency pairs."""
    if symbols is None:
        symbols = list(SYMBOL_CONFIGS.keys())
    result = {}
    for sym in symbols:
        cfg = SYMBOL_CONFIGS.get(sym, {"base_price": 1.0, "volatility": 0.0005, "seed": 99})
        result[sym] = generate_sample_data(symbol=sym, periods=periods, freq=freq, **cfg)
    return result
