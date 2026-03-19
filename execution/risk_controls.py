from __future__ import annotations

import numpy as np
import pandas as pd


def apply_no_trade_filter_high_vol(
    signal: pd.Series,
    vol_regime: pd.Series,
    allow_high_vol: bool = False,
) -> pd.Series:
    """Block new entries when volatility regime is HIGH_VOL."""
    if allow_high_vol:
        return signal.copy()

    out = signal.copy().astype(float)
    aligned_vol = vol_regime.reindex(out.index).fillna("UNKNOWN").astype(str)
    prev = out.shift(1).fillna(0.0)
    entries = (out.abs() > 1e-12) & (prev.abs() <= 1e-12)
    block = entries & (aligned_vol == "HIGH_VOL")
    out[block] = 0.0
    return out


def apply_volatility_targeting(
    signal: pd.Series,
    atr_normalized: pd.Series,
    target_atr_norm: float = 0.001,
    max_leverage: float = 1.0,
    min_leverage: float = 0.0,
) -> pd.Series:
    """Scale exposure inversely with ATR-normalized volatility."""
    if max_leverage <= 0:
        raise ValueError("max_leverage must be > 0")
    if min_leverage < 0 or min_leverage > max_leverage:
        raise ValueError("min_leverage must be within [0, max_leverage]")

    out = signal.copy().astype(float)
    atr = atr_normalized.reindex(out.index).replace(0.0, np.nan)
    scale = (target_atr_norm / atr).replace([np.inf, -np.inf], np.nan)
    scale = scale.fillna(0.0).clip(lower=min_leverage, upper=max_leverage)
    return (out * scale).clip(lower=-max_leverage, upper=max_leverage)
