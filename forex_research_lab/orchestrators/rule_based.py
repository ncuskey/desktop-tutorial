"""Rule-based orchestration layer."""

from __future__ import annotations

import pandas as pd


def adx_rule_switch(
    df: pd.DataFrame,
    trend_signal: pd.Series,
    mean_reversion_signal: pd.Series,
    adx_col: str = "adx_14",
    adx_threshold: float = 25.0,
) -> pd.Series:
    """
    If ADX is high -> trend strategy, else mean-reversion strategy.
    """

    if adx_col not in df.columns:
        raise ValueError(f"ADX column '{adx_col}' missing from dataframe")

    adx_high = df[adx_col] >= adx_threshold
    trend_aligned = trend_signal.reindex(df.index).fillna(0.0)
    mr_aligned = mean_reversion_signal.reindex(df.index).fillna(0.0)
    out = mr_aligned.where(~adx_high, trend_aligned)
    return out.rename("rule_based_orchestrated_signal")
