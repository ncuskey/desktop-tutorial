"""Carry-style placeholder strategies."""

from __future__ import annotations

import pandas as pd

from .base import BaseStrategy


class CarryProxyStrategy(BaseStrategy):
    name = "carry_proxy"

    def generate_signals(self, df: pd.DataFrame, params: dict) -> pd.Series:
        proxy = float(params.get("interest_rate_differential", 0.0))
        if "carry_proxy" in df.columns:
            return df["carry_proxy"].fillna(0).clip(-1, 1).astype(int)

        if proxy > 0:
            return pd.Series(1, index=df.index, dtype=int)
        if proxy < 0:
            return pd.Series(-1, index=df.index, dtype=int)
        return pd.Series(0, index=df.index, dtype=int)
