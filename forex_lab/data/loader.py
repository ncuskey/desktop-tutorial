from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

REQUIRED_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
SYMBOLS = ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD")


def _pip_size_for_symbol(symbol: str) -> float:
    return 0.01 if symbol.endswith("JPY") else 0.0001


def ensure_sample_data(
    data_dir: str | Path,
    symbols: Iterable[str] = SYMBOLS,
    periods: int = 24 * 365 * 2,
    freq: str = "H",
    seed: int = 42,
) -> None:
    """Create reproducible synthetic OHLCV CSVs if they do not exist."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    index = pd.date_range("2022-01-01", periods=periods, freq=freq, tz="UTC")

    for symbol in symbols:
        file_path = data_path / f"{symbol}.csv"
        if file_path.exists():
            continue

        pip_size = _pip_size_for_symbol(symbol)
        base_price = 130.0 if symbol.endswith("JPY") else 1.0 + rng.uniform(-0.1, 0.1)
        drift = rng.uniform(-0.00001, 0.00002)
        vol = rng.uniform(0.0004, 0.0012)

        noise = rng.normal(loc=drift, scale=vol, size=len(index))
        close = base_price * np.exp(np.cumsum(noise))
        open_ = np.r_[close[0], close[:-1]]
        high = np.maximum(open_, close) + np.abs(rng.normal(0, vol * base_price * 0.2, len(index)))
        low = np.minimum(open_, close) - np.abs(rng.normal(0, vol * base_price * 0.2, len(index)))
        volume = rng.integers(100, 4000, len(index))

        df = pd.DataFrame(
            {
                "timestamp": index,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "spread": rng.uniform(0.6, 2.2, len(index)) * pip_size,
            }
        )
        df.to_csv(file_path, index=False)


def load_ohlcv_csv(path: str | Path, symbol: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if "timestamp" not in df.columns:
        raise ValueError(f"'timestamp' column is required in {path}")
    df = df.set_index("timestamp").sort_index()

    missing = [c for c in REQUIRED_OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in {path}")

    if symbol:
        df["symbol"] = symbol
    elif "symbol" not in df.columns:
        inferred = Path(path).stem.upper()
        df["symbol"] = inferred
    return df


def load_symbol_data(data_dir: str | Path, symbols: Iterable[str]) -> Dict[str, pd.DataFrame]:
    data = {}
    for symbol in symbols:
        path = Path(data_dir) / f"{symbol}.csv"
        data[symbol] = load_ohlcv_csv(path, symbol=symbol)
    return data


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample OHLCV bars and carry spread using average."""
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "spread": "mean",
        "symbol": "last",
    }
    existing_agg = {k: v for k, v in agg.items() if k in df.columns}
    out = df.resample(timeframe).agg(existing_agg).dropna(subset=["open", "high", "low", "close"])
    return out
