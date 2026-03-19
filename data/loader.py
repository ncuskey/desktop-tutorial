from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_SYMBOLS = ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD")


def ensure_mock_ohlcv_csv(
    path: str | Path,
    symbols: Iterable[str] = DEFAULT_SYMBOLS,
    periods: int = 2_400,
    freq: str = "1h",
    seed: int = 7,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return output_path

    rng = np.random.default_rng(seed)
    normalized_freq = freq.replace("H", "h").replace("D", "d")
    idx = pd.date_range("2020-01-01", periods=periods, freq=normalized_freq, tz="UTC")
    frames: list[pd.DataFrame] = []

    for i, symbol in enumerate(symbols):
        drift = 0.00001 + i * 0.000005
        vol = 0.0007 + i * 0.0001
        jump = rng.normal(loc=drift, scale=vol, size=periods)
        close = 1.10 + np.cumsum(jump)
        if symbol == "USDJPY":
            close = 110.0 + np.cumsum(rng.normal(loc=0.001, scale=0.06, size=periods))

        open_ = np.roll(close, 1)
        open_[0] = close[0]
        spread = np.abs(rng.normal(loc=vol * 0.35, scale=vol * 0.1, size=periods))
        high = np.maximum(open_, close) + spread
        low = np.minimum(open_, close) - spread
        volume = rng.integers(1_000, 20_000, size=periods)

        frames.append(
            pd.DataFrame(
                {
                    "timestamp": idx,
                    "symbol": symbol,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )
        )

    data = pd.concat(frames, ignore_index=True)
    data.to_csv(output_path, index=False)
    return output_path


def load_ohlcv_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if "symbol" not in df.columns:
        df["symbol"] = "UNKNOWN"
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
    return df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    required = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out_frames: list[pd.DataFrame] = []
    for symbol, g in df.groupby("symbol", sort=False):
        gi = g.set_index("timestamp").sort_index()
        resampled = (
            gi.resample(timeframe)
            .agg(
                open=("open", "first"),
                high=("high", "max"),
                low=("low", "min"),
                close=("close", "last"),
                volume=("volume", "sum"),
            )
            .dropna()
        )
        resampled["symbol"] = symbol
        out_frames.append(resampled.reset_index())
    return pd.concat(out_frames, ignore_index=True).sort_values(
        ["symbol", "timestamp"]
    )


def load_symbol_data(df: pd.DataFrame, symbol: str, timeframe: str = "1H") -> pd.DataFrame:
    if timeframe.upper() not in {"1H", "4H", "1D", "H1", "H4", "D1"}:
        raise ValueError("Unsupported timeframe. Use H1/H4/D1 equivalents.")

    normalized = timeframe.upper()
    pandas_tf = {"H1": "1h", "H4": "4h", "D1": "1d", "1H": "1h", "4H": "4h", "1D": "1d"}[
        normalized
    ]

    filtered = df[df["symbol"] == symbol].copy()
    if filtered.empty:
        raise ValueError(f"Symbol {symbol} not found in data.")
    if pandas_tf != "1h":
        filtered = resample_ohlcv(filtered, pandas_tf)
    return filtered.sort_values("timestamp").reset_index(drop=True)
