from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

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
    if "spread_bps" not in out.columns:
        out["spread_bps"] = float(cost_model.spread_bps)
    else:
        out["spread_bps"] = pd.to_numeric(out["spread_bps"], errors="coerce")
    out["slippage_bps"] = float(cost_model.slippage_bps)
    out["commission_bps"] = float(cost_model.commission_bps)
    effective_spread = out["spread_bps"].fillna(float(cost_model.spread_bps))
    out["roundtrip_cost_fraction"] = (
        (effective_spread + float(cost_model.slippage_bps) + float(cost_model.commission_bps))
        / 10_000.0
    )
    return out


def resolve_symbol_cost_model(
    default_model: CostModel,
    symbol: str,
    overrides: Mapping[str, Mapping[str, float]] | None = None,
) -> CostModel:
    """Return symbol-specific cost model override if configured."""
    if overrides is None or symbol not in overrides:
        return default_model
    override = overrides[symbol]
    return CostModel(
        spread_bps=float(override.get("spread_bps", default_model.spread_bps)),
        slippage_bps=float(override.get("slippage_bps", default_model.slippage_bps)),
        commission_bps=float(override.get("commission_bps", default_model.commission_bps)),
    )
