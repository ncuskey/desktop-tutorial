from __future__ import annotations

from typing import Dict

import pandas as pd


def classify_regime(
    df: pd.DataFrame,
    atr_pct_threshold: float = 0.003,
    adx_threshold: float = 25.0,
) -> pd.Series:
    """
    Simple regime labels from volatility and trend strength:
    - trending_volatile
    - trending_calm
    - ranging_volatile
    - ranging_calm
    """
    atr_pct = df["atr_pct"]
    adx = df["adx"]
    labels = pd.Series("ranging_calm", index=df.index, dtype=object)

    labels[(adx >= adx_threshold) & (atr_pct >= atr_pct_threshold)] = "trending_volatile"
    labels[(adx >= adx_threshold) & (atr_pct < atr_pct_threshold)] = "trending_calm"
    labels[(adx < adx_threshold) & (atr_pct >= atr_pct_threshold)] = "ranging_volatile"
    return labels


def regime_switched_signals(
    regime: pd.Series,
    regime_to_signal: Dict[str, pd.Series],
    fallback: int = 0,
) -> pd.Series:
    out = pd.Series(fallback, index=regime.index, dtype=float)
    for label, signal in regime_to_signal.items():
        mask = regime == label
        out.loc[mask] = signal.reindex(regime.index).loc[mask].fillna(fallback)
    return out.astype(int)
