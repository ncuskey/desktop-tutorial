"""Carry trade strategy (simplified interest rate differential proxy)."""

from typing import Any

import pandas as pd

from .base import BaseStrategy


class CarryStrategy(BaseStrategy):
    """
    Simplified carry: long high-yield, short low-yield.
    Uses a placeholder 'carry_signal' column: +1 = long carry, -1 = short carry.
    In production, this would use actual interest rate differentials.
    """

    @property
    def default_params(self) -> dict[str, Any]:
        return {}

    def generate_signals(self, df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
        if "carry_signal" in df.columns:
            return df["carry_signal"].fillna(0).astype(int)
        # Placeholder: no signal if no carry data
        return pd.Series(0, index=df.index)
