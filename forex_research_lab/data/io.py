"""Market data loading and resampling helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def ensure_sample_ohlcv_csv(
    path: str | Path,
    symbols: list[str] | None = None,
    periods: int = 2400,
    seed: int = 42,
) -> Path:
    """Create a deterministic multi-symbol sample dataset if one does not exist."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return output_path

    symbols = symbols or ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
    rng = np.random.default_rng(seed)

    raw_index = pd.date_range("2022-01-03", periods=periods * 2, freq="h", tz="UTC")
    trading_hours = raw_index[raw_index.dayofweek < 5][:periods]
    frames: list[pd.DataFrame] = []

    price_bases = {
        "EURUSD": 1.10,
        "GBPUSD": 1.28,
        "USDJPY": 138.0,
        "AUDUSD": 0.72,
    }
    drift_bps = {
        "EURUSD": 0.2,
        "GBPUSD": 0.1,
        "USDJPY": -0.05,
        "AUDUSD": 0.08,
    }

    for idx, symbol in enumerate(symbols):
        base_price = price_bases.get(symbol, 1.0 + idx * 0.1)
        base_drift = drift_bps.get(symbol, 0.05)

        regime = np.sin(np.linspace(0, 10 * np.pi, len(trading_hours)) + idx) * 0.00035
        trend = base_drift / 10_000
        noise = rng.normal(0.0, 0.0012 + idx * 0.00015, len(trading_hours))
        returns = regime + trend + noise

        close = base_price * np.exp(np.cumsum(returns))
        open_ = np.roll(close, 1)
        open_[0] = base_price

        intrabar_noise = np.abs(rng.normal(0.0, 0.0009, len(trading_hours)))
        high = np.maximum(open_, close) * (1 + intrabar_noise)
        low = np.minimum(open_, close) * (1 - intrabar_noise)
        volume = rng.integers(500, 3_000, len(trading_hours))

        frames.append(
            pd.DataFrame(
                {
                    "timestamp": trading_hours,
                    "symbol": symbol,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )
        )

    sample = pd.concat(frames, ignore_index=True).sort_values(["symbol", "timestamp"])
    sample.to_csv(output_path, index=False)
    return output_path


def load_ohlcv_csv(path: str | Path) -> pd.DataFrame:
    """Load long-form OHLCV data with one row per symbol/timestamp."""

    df = pd.read_csv(path)
    required = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample long-form hourly data into larger bars per symbol."""

    rule_mapping = {
        "H1": "1h",
        "H4": "4h",
        "D1": "1D",
    }
    rule = rule_mapping.get(timeframe.upper(), timeframe)
    frames: list[pd.DataFrame] = []
    for symbol, group in df.groupby("symbol", sort=True):
        resampled = (
            group.set_index("timestamp")
            .resample(rule)
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
            .reset_index()
        )
        resampled["symbol"] = symbol
        frames.append(resampled)

    if not frames:
        return pd.DataFrame(columns=df.columns)

    ordered = pd.concat(frames, ignore_index=True)
    return ordered[["timestamp", "symbol", "open", "high", "low", "close", "volume"]]
