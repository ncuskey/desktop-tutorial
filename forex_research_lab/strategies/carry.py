"""Simplified carry strategy placeholder."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


DEFAULT_CARRY_BIAS = {
    "EURUSD": -1.0,
    "GBPUSD": -1.0,
    "USDJPY": 1.0,
    "AUDUSD": 1.0,
}


def carry_proxy_generate_signals(df: pd.DataFrame, params: Mapping[str, float]) -> pd.Series:
    """
    Simplified carry placeholder.

    The strategy applies a static directional bias based on symbol
    (or optional `rate_differential` parameter if provided).
    """

    symbol = str(params.get("symbol", df.get("symbol", pd.Series(["EURUSD"])).iloc[0])).upper()
    rate_differential = params.get("rate_differential")

    if rate_differential is None:
        base_signal = DEFAULT_CARRY_BIAS.get(symbol, 0.0)
    else:
        base_signal = float(np.sign(float(rate_differential)))

    out = pd.Series(base_signal, index=df.index, name="carry_signal", dtype=float)
    return out
