from __future__ import annotations

import pandas as pd


def adx_rule_signal(
    df: pd.DataFrame,
    trend_signal: pd.Series,
    mean_reversion_signal: pd.Series,
    adx_col: str = "adx_14",
    threshold: float = 25.0,
) -> pd.Series:
    if adx_col not in df.columns:
        raise ValueError(f"{adx_col} not found in dataframe.")
    adx = df[adx_col]
    out = mean_reversion_signal.reindex(df.index).fillna(0).astype(int)
    use_trend = adx >= threshold
    out[use_trend] = trend_signal.reindex(df.index).fillna(0).astype(int)[use_trend]
    return out
