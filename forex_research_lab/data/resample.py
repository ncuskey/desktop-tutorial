"""Timeframe resampling helpers."""

from __future__ import annotations

from typing import Dict, Iterable

import pandas as pd


TIMEFRAME_TO_RULE = {
    "H1": "1h",
    "H4": "4h",
    "D1": "1D",
}


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample OHLCV dataframe into a target timeframe."""

    if timeframe not in TIMEFRAME_TO_RULE:
        raise ValueError(f"Unsupported timeframe: {timeframe}. Use one of {list(TIMEFRAME_TO_RULE)}")

    rule = TIMEFRAME_TO_RULE[timeframe]
    aggregations = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    if "spread_pips" in df.columns:
        aggregations["spread_pips"] = "mean"
    if "symbol" in df.columns:
        aggregations["symbol"] = "last"

    out = df.resample(rule).agg(aggregations)
    out = out.dropna(subset=["open", "high", "low", "close"])
    return out


def resample_symbol_map(
    data_by_symbol: Dict[str, pd.DataFrame],
    timeframes: Iterable[str],
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Resample every symbol dataframe to multiple timeframes."""

    result: Dict[str, Dict[str, pd.DataFrame]] = {}
    for symbol, df in data_by_symbol.items():
        result[symbol] = {}
        for timeframe in timeframes:
            if timeframe == "H1":
                result[symbol][timeframe] = df.copy()
            else:
                result[symbol][timeframe] = resample_ohlcv(df, timeframe)
    return result
