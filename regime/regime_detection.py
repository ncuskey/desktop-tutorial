from __future__ import annotations

import numpy as np
import pandas as pd


def _rolling_percentile_rank(series: pd.Series, window: int) -> pd.Series:
    min_periods = max(30, window // 3)
    return series.rolling(window=window, min_periods=min_periods).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )


def attach_regime_labels(
    df: pd.DataFrame,
    adx_col: str = "adx_14",
    atr_col: str = "atr_14",
    close_col: str = "close",
    adx_threshold: float = 25.0,
    vol_window: int = 252,
    high_vol_pct: float = 0.70,
    low_vol_pct: float = 0.30,
    slope_window: int = 20,
) -> pd.DataFrame:
    """Attach trend/volatility regime labels.

    Regimes:
      - trend regime: TRENDING / RANGING based on ADX threshold
      - volatility regime: HIGH_VOL / MID_VOL / LOW_VOL via ATR-normalized percentile
      - combined regime: "<trend>_<vol>"
    """
    required = {adx_col, atr_col, close_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required regime columns: {sorted(missing)}")

    out = df.copy()
    out["atr_norm"] = out[atr_col] / out[close_col].abs().replace(0.0, np.nan)
    out["atr_norm_pct_rank"] = _rolling_percentile_rank(out["atr_norm"], window=vol_window)

    out["trend_regime"] = np.where(out[adx_col] > adx_threshold, "TRENDING", "RANGING")

    out["vol_regime"] = "MID_VOL"
    out.loc[out["atr_norm_pct_rank"] >= high_vol_pct, "vol_regime"] = "HIGH_VOL"
    out.loc[out["atr_norm_pct_rank"] <= low_vol_pct, "vol_regime"] = "LOW_VOL"

    # Optional trend/range proxy based on directional slope vs realized variance.
    slope = out[close_col].pct_change(slope_window).abs()
    vol = out[close_col].pct_change().rolling(slope_window, min_periods=slope_window).std()
    out["trend_variance_ratio"] = slope / vol.replace(0.0, np.nan)

    out["regime_label"] = out["trend_regime"] + "_" + out["vol_regime"]
    return out
