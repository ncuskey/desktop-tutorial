"""
Timeframe resampling for OHLCV data.

Supported aliases follow pandas offset strings:
  H1  -> '1h'   (hourly)
  H4  -> '4h'
  D1  -> '1D'   (daily)
"""

from __future__ import annotations

import pandas as pd


_ALIAS_MAP: dict[str, str] = {
    "H1": "1h",
    "H4": "4h",
    "D1": "1D",
    "W1": "1W",
    "M1": "1ME",
    "1h": "1h",
    "4h": "4h",
    "1D": "1D",
    "1W": "1W",
}


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample an OHLCV DataFrame to a coarser timeframe.

    Parameters
    ----------
    df:
        Source OHLCV frame (datetime index, sorted ascending).
    timeframe:
        Target timeframe string ('H1', 'H4', 'D1', or pandas offset alias).

    Returns
    -------
    Resampled OHLCV frame; rows with NaN close are dropped.
    """
    freq = _ALIAS_MAP.get(timeframe, timeframe)

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    existing_cols = {k: v for k, v in agg.items() if k in df.columns}
    resampled = df.resample(freq).agg(existing_cols)
    resampled = resampled.dropna(subset=["close"])
    return resampled
