from __future__ import annotations

import pandas as pd


def attach_cost_model(
    df: pd.DataFrame,
    default_spread_bps: float = 1.5,
    commission_bps: float = 0.2,
    slippage_bps: float = 0.5,
) -> pd.DataFrame:
    """
    Attach per-bar transaction cost assumptions.

    spread_bps can be directly inferred from 'spread' and 'close' when available.
    """
    out = df.copy()
    if "spread" in out.columns:
        out["spread_bps"] = (out["spread"] / out["close"]) * 10000
        out["spread_bps"] = out["spread_bps"].replace([float("inf"), -float("inf")], pd.NA).fillna(
            default_spread_bps
        )
    else:
        out["spread_bps"] = default_spread_bps

    out["commission_bps"] = commission_bps
    out["slippage_bps"] = slippage_bps
    out["total_cost_bps"] = out["spread_bps"] + out["commission_bps"] + out["slippage_bps"]
    return out
