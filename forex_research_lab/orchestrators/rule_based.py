"""Rule-based orchestration policies."""

from __future__ import annotations

import pandas as pd


def adx_rule_switch(
    df: pd.DataFrame,
    trend_signal: pd.Series,
    mean_reversion_signal: pd.Series,
    adx_column: str = "adx_14",
    threshold: float = 25.0,
) -> pd.Series:
    regime_mask = df[adx_column] >= threshold
    return trend_signal.where(regime_mask, mean_reversion_signal).fillna(0).astype(int)
