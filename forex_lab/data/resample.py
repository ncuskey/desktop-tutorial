"""OHLCV resampling to different timeframes."""

from typing import Literal

import pandas as pd

TIMEFRAME_MAP = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1D",
    "W1": "1W",
}


def resample_ohlcv(
    df: pd.DataFrame,
    timeframe: str,
    agg: Literal["first", "last", "ohlc"] = "ohlc",
) -> pd.DataFrame:
    """
    Resample OHLCV to target timeframe.

    Args:
        df: DataFrame with open, high, low, close, volume
        timeframe: Target (e.g. H1, H4, D1) or pandas offset (e.g. 4h, 1D)
        agg: 'ohlc' for proper OHLC aggregation, 'first'/'last' for simple
    """
    rule = TIMEFRAME_MAP.get(timeframe.upper(), timeframe)

    if agg == "ohlc":
        resampled = df.resample(rule).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
    else:
        method = "first" if agg == "first" else "last"
        resampled = df.resample(rule).agg(
            {
                "open": method,
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )

    resampled = resampled.dropna(how="all")
    if "symbol" in df.columns:
        resampled["symbol"] = df["symbol"].iloc[0]

    return resampled
