from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CostModel:
    spread_bps: float = 0.8
    slippage_bps: float = 0.4
    commission_bps: float = 0.2

    @property
    def total_bps(self) -> float:
        return self.spread_bps + self.slippage_bps + self.commission_bps

    @property
    def total_fraction(self) -> float:
        return self.total_bps / 10_000.0


def attach_costs(df: pd.DataFrame, cost_model: CostModel) -> pd.DataFrame:
    out = df.copy()
    out["spread_bps"] = float(cost_model.spread_bps)
    out["slippage_bps"] = float(cost_model.slippage_bps)
    out["commission_bps"] = float(cost_model.commission_bps)
    out["roundtrip_cost_fraction"] = float(cost_model.total_fraction)
    return out
