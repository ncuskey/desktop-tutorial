"""Spread and transaction cost model."""

from __future__ import annotations

import pandas as pd
import numpy as np

DEFAULT_SPREADS = {
    "EURUSD": 0.00010,
    "GBPUSD": 0.00015,
    "USDJPY": 0.015,
    "AUDUSD": 0.00012,
}

DEFAULT_COMMISSION_BPS = 0.5
DEFAULT_SLIPPAGE_BPS = 1.0


def attach_costs(
    df: pd.DataFrame,
    spread: float | None = None,
    commission_bps: float = DEFAULT_COMMISSION_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> pd.DataFrame:
    """Attach cost columns to the DataFrame.

    Adds: spread, commission_cost, slippage_cost, total_cost_per_trade
    All costs are in price units per round-trip trade.
    """
    df = df.copy()
    symbol = df.get("symbol", pd.Series(dtype=str))
    if spread is None:
        if "symbol" in df.columns:
            sym = df["symbol"].iloc[0]
            spread = DEFAULT_SPREADS.get(sym, 0.0002)
        else:
            spread = 0.0002

    df["spread"] = spread
    df["commission_cost"] = df["close"] * commission_bps / 10000
    df["slippage_cost"] = df["close"] * slippage_bps / 10000
    df["total_cost_per_trade"] = df["spread"] + df["commission_cost"] + df["slippage_cost"]
    return df
