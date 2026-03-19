from __future__ import annotations

from typing import Dict

import pandas as pd


def carry_proxy_signals(df: pd.DataFrame, params: Dict) -> pd.Series:
    """
    Simplified placeholder carry model.
    Expects params["symbol_carry"] dict mapping symbol -> signed carry score.
    """
    symbol = params.get("symbol")
    carry_map = params.get("symbol_carry", {})
    if symbol is None and "symbol" in df.columns:
        symbol = str(df["symbol"].iloc[-1])
    carry_score = float(carry_map.get(symbol, 0.0))
    signal = 1 if carry_score > 0 else -1 if carry_score < 0 else 0
    return pd.Series(signal, index=df.index, dtype=int)
