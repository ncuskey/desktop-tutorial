"""Deterministic synthetic Forex OHLCV dataset generator."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


START_PRICE_BY_SYMBOL = {
    "EURUSD": 1.10,
    "GBPUSD": 1.28,
    "USDJPY": 145.0,
    "AUDUSD": 0.72,
}

SPREAD_BY_SYMBOL = {
    "EURUSD": 0.8,
    "GBPUSD": 1.2,
    "USDJPY": 0.9,
    "AUDUSD": 1.3,
}


def generate_synthetic_ohlcv(
    symbol: str,
    start: str = "2019-01-01",
    periods: int = 5000,
    freq: str = "1h",
) -> pd.DataFrame:
    """Generate pseudo-realistic OHLCV data with trend and mean-reverting episodes."""

    symbol_u = symbol.upper()
    seed = abs(hash(symbol_u)) % (2**32)
    rng = np.random.default_rng(seed)

    index = pd.date_range(start=start, periods=periods, freq=freq, tz="UTC")
    start_price = START_PRICE_BY_SYMBOL.get(symbol_u, 1.0)

    # Regime component (slow trend) + cyclical component + noise.
    t = np.arange(periods)
    trend_component = 0.00002 * np.sin(t / 400.0)
    cyclical_component = 0.00008 * np.sin(t / 45.0)
    noise_component = rng.normal(0.0, 0.0007, size=periods)
    log_returns = trend_component + cyclical_component + noise_component

    close = start_price * np.exp(np.cumsum(log_returns))
    open_ = np.concatenate(([close[0]], close[:-1]))

    intrabar_span = np.abs(rng.normal(0.0, 0.0008, size=periods))
    high = np.maximum(open_, close) * (1 + intrabar_span)
    low = np.minimum(open_, close) * (1 - intrabar_span)
    low = np.clip(low, 1e-6, None)

    volume = rng.integers(700, 3500, size=periods)
    spread_base = SPREAD_BY_SYMBOL.get(symbol_u, 1.0)
    spread_noise = rng.normal(0.0, 0.10, size=periods)
    spread_pips = np.clip(spread_base * (1 + spread_noise), 0.2, None)

    df = pd.DataFrame(
        {
            "timestamp": index,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "spread_pips": spread_pips,
        }
    )
    return df


def ensure_sample_data(
    data_dir: str | Path,
    symbols: Iterable[str] = ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD"),
    timeframe: str = "H1",
    periods: int = 5000,
) -> list[Path]:
    """
    Ensure `<symbol>_<timeframe>.csv` files exist in data_dir.

    Existing files are preserved.
    """

    out_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written_files: list[Path] = []

    for symbol in symbols:
        file_path = out_dir / f"{symbol}_{timeframe}.csv"
        if not file_path.exists():
            df = generate_synthetic_ohlcv(symbol=symbol, periods=periods)
            df.to_csv(file_path, index=False)
        written_files.append(file_path)

    return written_files
