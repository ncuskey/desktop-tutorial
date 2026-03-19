"""Execution cost helpers for FX research."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CostModel:
    """Transaction cost assumptions for a symbol or experiment."""

    spread_pips: float = 1.0
    slippage_bps: float = 0.5
    commission_bps: float = 0.2


DEFAULT_COST_MODELS: dict[str, CostModel] = {
    "EURUSD": CostModel(spread_pips=0.8, slippage_bps=0.4, commission_bps=0.15),
    "GBPUSD": CostModel(spread_pips=1.0, slippage_bps=0.5, commission_bps=0.15),
    "USDJPY": CostModel(spread_pips=0.9, slippage_bps=0.45, commission_bps=0.15),
    "AUDUSD": CostModel(spread_pips=1.1, slippage_bps=0.55, commission_bps=0.15),
}


def infer_pip_size(symbol: str) -> float:
    """Infer a conventional pip size from an FX symbol."""
    return 0.01 if symbol.upper().endswith("JPY") else 0.0001


def get_default_cost_model(symbol: str) -> CostModel:
    """Return a default cost model for a known symbol."""
    return DEFAULT_COST_MODELS.get(symbol.upper(), CostModel())


def attach_cost_model(
    dataframe: pd.DataFrame,
    symbol: str | None = None,
    cost_model: CostModel | None = None,
    carry_proxy: float | None = None,
) -> pd.DataFrame:
    """Attach spread, slippage, commissions, and optional carry proxy columns."""
    enriched = dataframe.copy()

    if symbol is None:
        if "symbol" in enriched.columns and not enriched["symbol"].dropna().empty:
            symbol = str(enriched["symbol"].dropna().iloc[0]).upper()
        else:
            symbol = "UNKNOWN"

    model = cost_model or get_default_cost_model(symbol)
    pip_size = infer_pip_size(symbol)

    enriched["spread"] = model.spread_pips * pip_size
    enriched["half_spread"] = enriched["spread"] / 2.0
    enriched["slippage_bps"] = model.slippage_bps
    enriched["commission_bps"] = model.commission_bps

    if "carry_proxy" not in enriched.columns:
        enriched["carry_proxy"] = 0.0 if carry_proxy is None else carry_proxy
    return enriched
