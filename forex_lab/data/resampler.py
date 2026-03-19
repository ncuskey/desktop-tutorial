"""Resample OHLCV data to higher timeframes."""

from __future__ import annotations

import pandas as pd

_FREQ_MAP = {
    "H1": "1h",
    "H4": "4h",
    "D1": "1D",
}


def resample_timeframe(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample an OHLCV DataFrame to *timeframe* (H1, H4, D1)."""
    freq = _FREQ_MAP.get(timeframe, timeframe)
    ohlcv = df[["open", "high", "low", "close", "volume"]].resample(freq).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    ohlcv.dropna(subset=["close"], inplace=True)

    extra_cols = [c for c in df.columns if c not in ("open", "high", "low", "close", "volume")]
    for col in extra_cols:
        ohlcv[col] = df[col].resample(freq).last()

    return ohlcv
