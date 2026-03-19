"""Transaction cost model helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


DEFAULT_SPREAD_BPS = {
    "EURUSD": 0.8,
    "GBPUSD": 1.0,
    "USDJPY": 0.9,
    "AUDUSD": 1.1,
}


@dataclass(slots=True)
class CostModel:
    spread_bps_by_symbol: dict[str, float] = field(default_factory=lambda: DEFAULT_SPREAD_BPS.copy())
    slippage_bps: float = 0.5
    commission_bps: float = 0.2


def attach_cost_model(df: pd.DataFrame, cost_model: CostModel | None = None) -> pd.DataFrame:
    """Attach spread and cost assumptions to an OHLCV frame."""

    model = cost_model or CostModel()
    enriched = df.copy()
    enriched["spread_bps"] = enriched["symbol"].map(model.spread_bps_by_symbol).fillna(1.0)
    enriched["slippage_bps"] = model.slippage_bps
    enriched["commission_bps"] = model.commission_bps
    return enriched
