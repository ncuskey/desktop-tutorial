"""Spread, slippage, and commission modeling."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class CostModel:
    """
    Realistic execution cost model.

    - spread_bps: bid-ask spread in basis points
    - slippage_bps: slippage per trade in basis points
    - commission_per_lot: commission per standard lot
    - commission_per_trade: flat commission per trade (alternative)
    """

    spread_bps: float = 10.0  # 1 pip for major pairs
    slippage_bps: float = 5.0
    commission_per_lot: float = 7.0  # USD per lot round-trip
    commission_per_trade: float = 0.0  # flat per trade
    lot_size: float = 100_000.0  # standard lot

    def cost_per_trade_bps(self, notional: Optional[float] = None) -> float:
        """Total cost in bps for a round-trip trade."""
        total_bps = self.spread_bps + 2 * self.slippage_bps  # entry + exit
        if self.commission_per_trade > 0 and notional and notional > 0:
            commission_bps = (self.commission_per_trade * 2 / notional) * 1e4
            total_bps += commission_bps
        if self.commission_per_lot > 0 and notional and notional > 0:
            lots = notional / self.lot_size
            commission_bps = (self.commission_per_lot * lots * 2 / notional) * 1e4
            total_bps += commission_bps
        return total_bps

    def apply_costs(
        self,
        price: float,
        direction: int,
        notional: Optional[float] = None,
    ) -> float:
        """
        Apply costs to execution price.

        direction: +1 long, -1 short
        Returns adjusted execution price (what you actually get).
        """
        total_bps = self.cost_per_trade_bps(notional) / 2  # half for entry
        cost_pct = total_bps / 1e4
        if direction > 0:  # long: pay more
            return price * (1 + cost_pct)
        else:  # short: receive less
            return price * (1 - cost_pct)


def attach_costs(
    df: pd.DataFrame,
    cost_model: CostModel,
    symbol: str = "EURUSD",
) -> pd.DataFrame:
    """
    Attach cost estimates to OHLC data.

    Adds columns: spread_bps, cost_per_trade_bps (for reference).
    """
    df = df.copy()
    df["spread_bps"] = cost_model.spread_bps
    df["cost_per_trade_bps"] = cost_model.cost_per_trade_bps()
    return df
