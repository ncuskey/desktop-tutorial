"""Data loading, resampling, indicators, and sample data generation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}

DEFAULT_SPREAD_BPS = {
    "EURUSD": 0.8,
    "GBPUSD": 1.0,
    "USDJPY": 0.9,
    "AUDUSD": 1.1,
}

SYMBOL_BASE_PRICE = {
    "EURUSD": 1.10,
    "GBPUSD": 1.28,
    "USDJPY": 145.00,
    "AUDUSD": 0.68,
}


def _split_by_symbol(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    if "symbol" not in df.columns:
        return [("UNKNOWN", df.copy())]
    return [(str(symbol), frame.copy()) for symbol, frame in df.groupby("symbol", sort=False)]


def _validate_ohlcv_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"OHLCV data is missing columns: {sorted(missing)}")


def load_ohlcv_csv(path: str | Path) -> pd.DataFrame:
    """Load multi-symbol OHLCV data from CSV."""

    data = pd.read_csv(path, parse_dates=["timestamp"])
    _validate_ohlcv_columns(data)
    return data.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample OHLCV data by symbol into a higher timeframe."""

    _validate_ohlcv_columns(df)
    frames: list[pd.DataFrame] = []

    for symbol, frame in _split_by_symbol(df):
        indexed = frame.sort_values("timestamp").set_index("timestamp")
        aggregations = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        for optional_column in ("spread_bps", "commission_bps", "slippage_bps"):
            if optional_column in indexed.columns:
                aggregations[optional_column] = "mean"

        resampled = indexed.resample(timeframe).agg(aggregations)
        resampled["symbol"] = symbol
        frames.append(resampled.dropna(subset=["open", "high", "low", "close"]).reset_index())

    result = pd.concat(frames, ignore_index=True)
    return result.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def _compute_rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    relative_strength = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100 - (100 / (1 + relative_strength))


def _compute_atr(frame: pd.DataFrame, window: int) -> pd.Series:
    high_low = frame["high"] - frame["low"]
    high_close = (frame["high"] - frame["close"].shift(1)).abs()
    low_close = (frame["low"] - frame["close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def _compute_adx(frame: pd.DataFrame, window: int) -> pd.Series:
    up_move = frame["high"].diff()
    down_move = -frame["low"].diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=frame.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=frame.index,
    )
    atr = _compute_atr(frame, window)
    plus_di = 100 * plus_dm.ewm(alpha=1 / window, min_periods=window, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / window, min_periods=window, adjust=False).mean() / atr
    directional_index = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return 100 * directional_index.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def compute_basic_indicators(
    df: pd.DataFrame,
    *,
    ma_windows: Iterable[int] = (10, 20, 50),
    rsi_window: int = 14,
    atr_window: int = 14,
    bollinger_window: int = 20,
    bollinger_std: float = 2.0,
    adx_window: int = 14,
) -> pd.DataFrame:
    """Attach the required basic indicators to the dataset."""

    _validate_ohlcv_columns(df)
    indicator_frames: list[pd.DataFrame] = []

    for _, frame in _split_by_symbol(df):
        local = frame.sort_values("timestamp").copy()
        close = local["close"]

        for window in ma_windows:
            local[f"ma_{window}"] = close.rolling(window).mean()

        local["rsi"] = _compute_rsi(close, rsi_window)
        local["atr"] = _compute_atr(local, atr_window)
        rolling_mean = close.rolling(bollinger_window).mean()
        rolling_std = close.rolling(bollinger_window).std(ddof=0)
        local["bb_mid"] = rolling_mean
        local["bb_upper"] = rolling_mean + bollinger_std * rolling_std
        local["bb_lower"] = rolling_mean - bollinger_std * rolling_std
        local["adx"] = _compute_adx(local, adx_window)
        indicator_frames.append(local)

    return pd.concat(indicator_frames, ignore_index=True).sort_values(
        ["symbol", "timestamp"]
    ).reset_index(drop=True)


def attach_cost_model(
    df: pd.DataFrame,
    *,
    spread_map: dict[str, float] | None = None,
    default_spread_bps: float = 1.0,
    commission_bps: float = 0.1,
    slippage_bps: float = 0.5,
) -> pd.DataFrame:
    """Attach per-row spread, commission, and slippage assumptions."""

    if "symbol" not in df.columns:
        raise ValueError("Expected a 'symbol' column to attach the cost model.")

    result = df.copy()
    spread_lookup = spread_map or DEFAULT_SPREAD_BPS

    if "spread_bps" not in result.columns:
        result["spread_bps"] = result["symbol"].map(spread_lookup).fillna(default_spread_bps)
    else:
        result["spread_bps"] = result["spread_bps"].fillna(
            result["symbol"].map(spread_lookup).fillna(default_spread_bps)
        )

    result["commission_bps"] = commission_bps
    result["slippage_bps"] = slippage_bps
    return result


def generate_mock_ohlcv(
    *,
    symbols: Iterable[str] = ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD"),
    start: str = "2020-01-01",
    periods: int = 24 * 365,
    freq: str = "h",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate deterministic multi-symbol mock Forex OHLCV data."""

    rng = np.random.default_rng(seed)
    index = pd.date_range(start=start, periods=periods, freq=freq)
    records: list[pd.DataFrame] = []

    for symbol_idx, symbol in enumerate(symbols):
        base_price = SYMBOL_BASE_PRICE.get(symbol, 1.0)
        regime_cycle = np.sin(np.linspace(0, 12 * np.pi, periods) + symbol_idx)
        trend_component = 0.00015 * regime_cycle + 0.00003 * np.sign(regime_cycle)
        intraday_cycle = 0.00005 * np.sin(np.arange(periods) * 2 * np.pi / 24)
        noise = rng.normal(0.0, 0.0009 + 0.00015 * symbol_idx, periods)

        raw_returns = trend_component + intraday_cycle + noise
        close = base_price * np.exp(np.cumsum(raw_returns))
        open_ = np.roll(close, 1)
        open_[0] = close[0] * (1 - raw_returns[0])

        candle_noise = np.abs(rng.normal(0.0, base_price * 0.0015, periods))
        high = np.maximum(open_, close) + candle_noise
        low = np.minimum(open_, close) - candle_noise
        volume = rng.integers(800, 5000, periods)

        frame = pd.DataFrame(
            {
                "timestamp": index,
                "symbol": symbol,
                "open": open_,
                "high": high,
                "low": np.clip(low, 0.0001, None),
                "close": close,
                "volume": volume,
                "spread_bps": DEFAULT_SPREAD_BPS.get(symbol, 1.0),
            }
        )
        records.append(frame)

    return pd.concat(records, ignore_index=True).sort_values(["symbol", "timestamp"]).reset_index(
        drop=True
    )
