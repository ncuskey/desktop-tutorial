"""
Carry trade strategy — simplified interest rate differential proxy.

In live trading this would use actual overnight swap rates.
Here we simulate the carry signal using a configurable rate differential
table that reflects realistic central bank rate environments.

The strategy is always long the higher-yielding currency.
Position is sized proportionally to the differential magnitude.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .base import BaseStrategy, SignalSeries

# Approximate annualised policy rates (basis points) by currency, 2020-2023 proxy
_DEFAULT_RATES: dict[str, float] = {
    "USD": 5.25,
    "EUR": 4.00,
    "GBP": 5.25,
    "JPY": -0.10,
    "AUD": 4.35,
    "NZD": 5.50,
    "CAD": 5.00,
    "CHF": 1.75,
}

# Map FX pair to (base_ccy, quote_ccy)
_PAIR_CCY: dict[str, tuple[str, str]] = {
    "EURUSD": ("EUR", "USD"),
    "GBPUSD": ("GBP", "USD"),
    "USDJPY": ("USD", "JPY"),
    "AUDUSD": ("AUD", "USD"),
    "NZDUSD": ("NZD", "USD"),
    "USDCAD": ("USD", "CAD"),
    "USDCHF": ("USD", "CHF"),
    "EURGBP": ("EUR", "GBP"),
}


class CarryProxy(BaseStrategy):
    """Simplified carry trade strategy.

    Uses a static rate differential table to determine carry direction.
    The signal is constant over time (structural position) unless the
    differential flips sign.

    Params
    ------
    symbol      : str   FX pair (default 'EURUSD')
    min_diff    : float minimum |differential| to take a position (default 0.5)
    rates       : dict  override rate table {currency: annualised_rate_%}
    """

    name = "carry_proxy"

    def default_params(self) -> dict[str, Any]:
        return {"symbol": "EURUSD", "min_diff": 0.5, "rates": {}}

    def generate_signals(self, df: pd.DataFrame, params: dict[str, Any]) -> SignalSeries:
        symbol = params.get("symbol", "EURUSD").upper()
        min_diff = float(params.get("min_diff", 0.5))
        rate_override = params.get("rates", {})

        rates = {**_DEFAULT_RATES, **rate_override}

        ccy_pair = _PAIR_CCY.get(symbol)
        if ccy_pair is None:
            # Unknown pair — flat
            return self._clip_signals(pd.Series(0, index=df.index, dtype="int8"))

        base_ccy, quote_ccy = ccy_pair
        base_rate = rates.get(base_ccy, 0.0)
        quote_rate = rates.get(quote_ccy, 0.0)
        differential = base_rate - quote_rate

        if differential > min_diff:
            pos = 1    # long base currency (buy the pair)
        elif differential < -min_diff:
            pos = -1   # short base currency (sell the pair)
        else:
            pos = 0

        signal = pd.Series(pos, index=df.index, dtype="int8")
        return self._clip_signals(signal)
