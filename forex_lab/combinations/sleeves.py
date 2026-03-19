"""
Specialist sleeves — activate different strategies conditionally based on
market state indicators (e.g. ADX, ATR-based volatility regime).

Each sleeve is a (strategy, params, filter_fn) triple.
filter_fn receives the current row and returns True if this sleeve is active.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from data.indicators import add_indicators


@dataclass
class Sleeve:
    """A conditional strategy sleeve."""

    strategy: Any                    # BaseStrategy instance
    params: dict[str, Any]
    filter_fn: Callable[[pd.Series], bool]  # receives a single row
    weight: float = 1.0
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = getattr(self.strategy, "name", "unknown")


class SpecialistSleeves:
    """Activate specialist strategies based on per-bar market filters.

    If multiple sleeves are active on a bar, their signals are averaged
    (weighted by sleeve.weight).  If no sleeve is active, signal = 0.

    Parameters
    ----------
    sleeves:
        List of Sleeve objects.
    """

    name = "specialist_sleeves"

    def __init__(self, sleeves: list[Sleeve]) -> None:
        self.sleeves = sleeves

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate combined signal using only active sleeves per bar.

        Returns
        -------
        pd.Series of float (or int8) signals {-1, 0, +1}.
        """
        df = add_indicators(df)

        # Pre-compute all strategy signals once
        strat_signals: dict[str, pd.Series] = {}
        for sleeve in self.sleeves:
            key = f"{sleeve.name}_{id(sleeve)}"
            strat_signals[key] = sleeve.strategy.generate_signals(df, sleeve.params)

        combined = pd.Series(0.0, index=df.index)

        for i in range(len(df)):
            row = df.iloc[i]
            total_weight = 0.0
            weighted_signal = 0.0

            for sleeve in self.sleeves:
                if sleeve.filter_fn(row):
                    key = f"{sleeve.name}_{id(sleeve)}"
                    sig = float(strat_signals[key].iloc[i])
                    weighted_signal += sleeve.weight * sig
                    total_weight += sleeve.weight

            if total_weight > 0:
                avg = weighted_signal / total_weight
                # Threshold to discrete signal
                if avg >= 0.3:
                    combined.iloc[i] = 1.0
                elif avg <= -0.3:
                    combined.iloc[i] = -1.0
                else:
                    combined.iloc[i] = 0.0

        return combined.astype("int8")


# ------------------------------------------------------------------
# Pre-built filter functions
# ------------------------------------------------------------------

def adx_trend_filter(threshold: float = 25.0) -> Callable[[pd.Series], bool]:
    """Active when ADX > threshold (trending market)."""
    def _fn(row: pd.Series) -> bool:
        adx_val = row.get("adx", 0)
        return float(adx_val) > threshold if pd.notna(adx_val) else False
    return _fn


def adx_range_filter(threshold: float = 25.0) -> Callable[[pd.Series], bool]:
    """Active when ADX < threshold (range/mean-reversion market)."""
    def _fn(row: pd.Series) -> bool:
        adx_val = row.get("adx", 100)
        return float(adx_val) < threshold if pd.notna(adx_val) else False
    return _fn


def high_volatility_filter(atr_pct_threshold: float = 0.005) -> Callable[[pd.Series], bool]:
    """Active when ATR% > threshold (high volatility)."""
    def _fn(row: pd.Series) -> bool:
        atr_pct = row.get("atr_pct", 0)
        return float(atr_pct) > atr_pct_threshold if pd.notna(atr_pct) else False
    return _fn


def low_volatility_filter(atr_pct_threshold: float = 0.005) -> Callable[[pd.Series], bool]:
    """Active when ATR% < threshold (low volatility)."""
    def _fn(row: pd.Series) -> bool:
        atr_pct = row.get("atr_pct", 1)
        return float(atr_pct) < atr_pct_threshold if pd.notna(atr_pct) else False
    return _fn


from typing import Callable
