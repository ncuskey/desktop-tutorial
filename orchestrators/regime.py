from __future__ import annotations

import pandas as pd


def classify_regime(
    df: pd.DataFrame,
    atr_col: str = "atr_14",
    adx_col: str = "adx_14",
) -> pd.Series:
    if atr_col not in df.columns or adx_col not in df.columns:
        raise ValueError("Required ATR/ADX columns are missing.")

    atr_threshold = df[atr_col].rolling(250, min_periods=50).median()
    adx_threshold = 25.0
    high_vol = df[atr_col] >= atr_threshold
    strong_trend = df[adx_col] >= adx_threshold

    regime = pd.Series("ranging_calm", index=df.index, dtype=object)
    regime[strong_trend & high_vol] = "trending_volatile"
    regime[strong_trend & ~high_vol] = "trending_calm"
    regime[~strong_trend & high_vol] = "ranging_volatile"
    return regime


def regime_switched_signal(
    regime: pd.Series,
    trend_signal: pd.Series,
    mean_reversion_signal: pd.Series,
) -> pd.Series:
    out = pd.Series(0, index=regime.index, dtype=int)
    trend_regimes = {"trending_volatile", "trending_calm"}
    use_trend = regime.isin(trend_regimes)

    out[use_trend] = trend_signal.reindex(regime.index).fillna(0).astype(int)[use_trend]
    out[~use_trend] = (
        mean_reversion_signal.reindex(regime.index).fillna(0).astype(int)[~use_trend]
    )
    return out
