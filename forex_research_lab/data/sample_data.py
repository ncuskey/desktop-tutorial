"""Synthetic FX OHLCV generation for smoke tests and demos."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SYMBOL_CONFIG = {
    "EURUSD": {"start_price": 1.10, "drift": 0.00001, "volatility": 0.0012, "carry_proxy": -0.0002, "phase": 0.2},
    "GBPUSD": {"start_price": 1.28, "drift": 0.000015, "volatility": 0.0015, "carry_proxy": 0.0001, "phase": 0.7},
    "USDJPY": {"start_price": 145.0, "drift": 0.00002, "volatility": 0.0011, "carry_proxy": 0.00035, "phase": 1.1},
    "AUDUSD": {"start_price": 0.72, "drift": 0.000012, "volatility": 0.00135, "carry_proxy": 0.0002, "phase": 1.6},
}


def generate_sample_ohlcv(
    symbol: str,
    start: str = "2022-01-01",
    periods: int = 24 * 365,
    frequency: str = "1h",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate deterministic synthetic OHLCV data with trend and mean-reverting pockets."""
    config = SYMBOL_CONFIG[symbol.upper()]
    rng = np.random.default_rng(seed)

    timestamps = pd.date_range(start=start, periods=periods, freq=frequency)
    time_index = np.arange(periods)

    base_drift = config["drift"]
    trend_wave = 0.00018 * np.sin(time_index / 180.0 + config["phase"])
    cycle_wave = 0.00012 * np.sin(time_index / 32.0 + (config["phase"] * 2.0))
    volatility_wave = config["volatility"] * (1.0 + 0.5 * np.sin(time_index / 240.0 + config["phase"]) ** 2)
    noise = rng.normal(loc=0.0, scale=volatility_wave, size=periods)
    returns = base_drift + trend_wave + cycle_wave + noise

    close = config["start_price"] * np.exp(np.cumsum(returns))
    open_ = np.empty_like(close)
    open_[0] = close[0] / np.exp(returns[0])
    open_[1:] = close[:-1]

    range_scale = close * (volatility_wave * 1.6)
    high = np.maximum(open_, close) + rng.uniform(0.2, 1.1, size=periods) * range_scale
    low = np.minimum(open_, close) - rng.uniform(0.2, 1.1, size=periods) * range_scale
    low = np.maximum(low, close * 0.7)

    volume = rng.integers(700, 4000, size=periods)
    carry_proxy = config["carry_proxy"] + 0.00005 * np.sin(time_index / 360.0 + config["phase"])

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": symbol.upper(),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "carry_proxy": carry_proxy,
        }
    )


def ensure_sample_data(
    directory: str | Path,
    symbols: tuple[str, ...] = ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD"),
    periods: int = 24 * 365,
) -> dict[str, Path]:
    """Write sample CSVs if they do not already exist and return their paths."""
    output_directory = Path(directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    written_files: dict[str, Path] = {}
    for offset, symbol in enumerate(symbols):
        path = output_directory / f"{symbol.lower()}.csv"
        if not path.exists():
            dataframe = generate_sample_ohlcv(symbol=symbol, periods=periods, seed=42 + offset)
            dataframe.to_csv(path, index=False)
        written_files[symbol] = path
    return written_files
