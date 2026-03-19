"""
Transaction cost model for Forex simulation.

Costs are applied per round-trip trade:
  - spread   : half-spread added on entry, half on exit
  - slippage : in basis-points of price, market-impact model
  - commission: fixed per-lot (in price units, e.g. 0.00005 for $5/100k)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import pandas as pd


@dataclass
class CostModel:
    """Per-symbol transaction cost parameters."""

    symbol: str = "EURUSD"
    spread_pips: float = 1.5          # round-trip spread in pips
    slippage_bps: float = 0.5         # one-way slippage in basis points
    commission_per_lot: float = 7.0   # USD per 100k round-trip
    pip_size: float = 0.0001          # pip value in price units (JPY pairs: 0.01)
    lot_size: float = 100_000.0       # standard lot

    @property
    def spread_cost(self) -> float:
        """Round-trip spread cost in price units."""
        return self.spread_pips * self.pip_size

    def slippage_cost(self, price: float) -> float:
        """One-way slippage cost in price units."""
        return price * self.slippage_bps / 10_000

    def commission_cost(self, price: float) -> float:
        """Round-trip commission in price units (normalised to 1 lot)."""
        return self.commission_per_lot / self.lot_size

    def total_round_trip_cost(self, price: float) -> float:
        """Total round-trip cost in price units."""
        return (
            self.spread_cost
            + 2 * self.slippage_cost(price)
            + self.commission_cost(price)
        )

    def cost_as_return(self, price: float) -> float:
        """Total round-trip cost expressed as a fraction of price."""
        return self.total_round_trip_cost(price) / price


# Default cost assumptions per symbol
DEFAULT_COSTS: dict[str, CostModel] = {
    "EURUSD": CostModel("EURUSD", spread_pips=1.0, slippage_bps=0.3, pip_size=0.0001),
    "GBPUSD": CostModel("GBPUSD", spread_pips=1.5, slippage_bps=0.5, pip_size=0.0001),
    "USDJPY": CostModel("USDJPY", spread_pips=1.2, slippage_bps=0.3, pip_size=0.01),
    "AUDUSD": CostModel("AUDUSD", spread_pips=1.8, slippage_bps=0.6, pip_size=0.0001),
}


def apply_costs_to_returns(
    returns: pd.Series,
    signals: pd.Series,
    cost_model: CostModel,
    prices: pd.Series,
) -> pd.Series:
    """Subtract transaction costs from a return series.

    Costs are incurred on every bar where the position changes.

    Parameters
    ----------
    returns:
        Bar-level strategy returns (before costs).
    signals:
        Position series (-1, 0, +1).
    cost_model:
        CostModel for the symbol being traded.
    prices:
        Close price series (aligned with returns).

    Returns
    -------
    Net returns after costs.
    """
    trades = signals.diff().abs().fillna(0) > 0
    cost_fraction = prices.map(lambda p: cost_model.cost_as_return(p))
    cost_series = (trades * cost_fraction * 0.5)  # half cost on open, half on close
    return returns - cost_series
