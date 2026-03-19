"""Execution cost model and helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


PIP_SIZE_BY_SYMBOL = {
    "USDJPY": 0.01,
}


def pip_size_for_symbol(symbol: str | None) -> float:
    if symbol is None:
        return 0.0001
    return PIP_SIZE_BY_SYMBOL.get(symbol.upper(), 0.0001)


@dataclass(frozen=True)
class ExecutionCostModel:
    """
    Cost model measured per full notional turnover (|delta position| == 1.0).

    - spread_pips: round-trip spread in pips for a unit turnover.
    - slippage_bps: slippage in basis points per turnover.
    - commission_bps: broker commission in basis points per turnover.
    """

    spread_pips: float = 1.0
    slippage_bps: float = 0.2
    commission_bps: float = 0.1


def attach_cost_columns(
    df: pd.DataFrame,
    spread_pips: float = 1.0,
    symbol: str | None = None,
) -> pd.DataFrame:
    """
    Attach per-bar spread and pip size columns used by the execution engine.
    """

    out = df.copy()
    if "spread_pips" not in out.columns:
        out["spread_pips"] = float(spread_pips)
    out["pip_size"] = pip_size_for_symbol(symbol)
    return out
