from __future__ import annotations

import pandas as pd


def rule_based_orchestration(
    df: pd.DataFrame,
    trend_signal: pd.Series,
    mean_reversion_signal: pd.Series,
    adx_threshold: float = 25.0,
) -> pd.Series:
    """
    If ADX is high use trend, else use mean-reversion.
    """
    adx = df["adx"].reindex(df.index)
    trend = trend_signal.reindex(df.index).fillna(0)
    mr = mean_reversion_signal.reindex(df.index).fillna(0)
    out = pd.Series(0, index=df.index, dtype=float)
    out[adx >= adx_threshold] = trend[adx >= adx_threshold]
    out[adx < adx_threshold] = mr[adx < adx_threshold]
    return out.astype(int)
