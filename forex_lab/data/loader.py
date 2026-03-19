"""
OHLCV data loading and synthetic generation.

Real CSV files should have columns: datetime, open, high, low, close, volume
with datetime as the index or first column (UTC, ISO format).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


def load_ohlcv(path: str | Path, symbol: str = "") -> pd.DataFrame:
    """Load OHLCV data from a CSV file.

    Expected columns: open, high, low, close, volume (case-insensitive).
    Index must be parseable as datetime.
    """
    df = pd.read_csv(
        path,
        index_col=0,
        parse_dates=True,
        infer_datetime_format=True,
    )
    df.columns = [c.lower() for c in df.columns]
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df.index.name = "datetime"
    if symbol:
        df["symbol"] = symbol
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df[["open", "high", "low", "close", "volume"]]


def generate_synthetic_ohlcv(
    symbol: str = "EURUSD",
    start: str = "2018-01-01",
    end: str = "2023-12-31",
    freq: str = "H",
    seed: int = 42,
    drift: float = 0.0,
    vol: float = 0.0008,
    spread_pips: float = 1.0,
    pip_size: float = 0.0001,
) -> pd.DataFrame:
    """Generate synthetic Forex OHLCV data via geometric Brownian motion.

    Parameters
    ----------
    drift:
        Per-bar drift (annualised drift / bars_per_year).
    vol:
        Per-bar volatility (annualised vol / sqrt(bars_per_year)).
    spread_pips:
        Half-spread added to high, subtracted from low (cosmetic).
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range(start=start, end=end, freq=freq)
    n = len(index)

    log_returns = rng.normal(drift, vol, n)
    log_price = np.cumsum(log_returns)

    # Start around a realistic FX mid-price
    base_prices = {
        "EURUSD": 1.10, "GBPUSD": 1.27, "USDJPY": 110.0, "AUDUSD": 0.72,
    }
    start_price = base_prices.get(symbol, 1.0)
    close = start_price * np.exp(log_price)

    # Intra-bar range via independent noise
    bar_range = np.abs(rng.normal(0, vol * 0.6, n)) * close
    high = close + bar_range * rng.uniform(0.3, 0.7, n)
    low = close - bar_range * rng.uniform(0.3, 0.7, n)
    open_ = np.roll(close, 1)
    open_[0] = start_price

    # Ensure OHLC consistency
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    half_spread = spread_pips * pip_size / 2
    high += half_spread
    low -= half_spread

    volume = rng.integers(500, 5000, n).astype(float)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )
    df.index.name = "datetime"
    return df
