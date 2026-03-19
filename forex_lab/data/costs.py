"""Transaction cost model — spread, slippage, commission."""

from __future__ import annotations

import pandas as pd

_DEFAULT_SPREADS = {
    "EURUSD": 0.00012,
    "GBPUSD": 0.00015,
    "USDJPY": 0.015,
    "AUDUSD": 0.00014,
}


def attach_cost_model(
    df: pd.DataFrame,
    spread: float | None = None,
    slippage_bps: float = 0.5,
    commission_per_lot: float = 3.5,
    lot_size: float = 100_000,
) -> pd.DataFrame:
    """Add cost-model columns to an OHLCV DataFrame.

    ``half_spread`` — half the bid/ask spread (applied on entry *and* exit).
    ``slippage`` — estimated slippage in price units.
    ``commission`` — per-unit commission cost.
    ``total_cost`` — round-trip cost per unit.
    """
    df = df.copy()
    symbol = df["symbol"].iloc[0] if "symbol" in df.columns else "EURUSD"

    if spread is None:
        spread = _DEFAULT_SPREADS.get(symbol, 0.0002)

    price = df["close"]
    half_spread = spread / 2
    slippage = price * slippage_bps / 10_000
    commission = commission_per_lot / lot_size

    df["half_spread"] = half_spread
    df["slippage"] = slippage
    df["commission"] = commission
    df["total_cost"] = spread + 2 * slippage + 2 * commission

    return df
