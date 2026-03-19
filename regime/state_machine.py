from __future__ import annotations

import json

import numpy as np
import pandas as pd


def _rolling_percentile_rank(series: pd.Series, window: int = 252) -> pd.Series:
    min_periods = max(30, window // 3)
    return series.rolling(window=window, min_periods=min_periods).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )


def _trend_state_with_hysteresis(
    adx: pd.Series,
    enter_trending: float = 28.0,
    exit_trending: float = 22.0,
) -> pd.Series:
    if enter_trending <= exit_trending:
        raise ValueError("enter_trending must be greater than exit_trending for hysteresis.")

    out: list[str] = []
    current = "TRENDING" if float(adx.iloc[0]) >= enter_trending else "RANGING"
    for value in adx.fillna(method="ffill").fillna(0.0):
        if current == "RANGING" and value > enter_trending:
            current = "TRENDING"
        elif current == "TRENDING" and value < exit_trending:
            current = "RANGING"
        out.append(current)
    return pd.Series(out, index=adx.index, name="raw_trend_state")


def _vol_state_with_hysteresis(
    atr_pct_rank: pd.Series,
    high_enter_pct: float = 0.70,
    high_exit_pct: float = 0.60,
    low_enter_pct: float = 0.30,
    low_exit_pct: float = 0.40,
) -> pd.Series:
    if not (0.0 <= low_enter_pct < low_exit_pct < high_exit_pct < high_enter_pct <= 1.0):
        raise ValueError("Volatility hysteresis percentiles must satisfy low_enter < low_exit < high_exit < high_enter.")

    ranks = atr_pct_rank.fillna(0.5)
    out: list[str] = []
    current = "MID_VOL"
    for value in ranks:
        if current == "HIGH_VOL":
            if value <= low_enter_pct:
                current = "LOW_VOL"
            elif value < high_exit_pct:
                current = "MID_VOL"
        elif current == "LOW_VOL":
            if value >= high_enter_pct:
                current = "HIGH_VOL"
            elif value > low_exit_pct:
                current = "MID_VOL"
        else:  # MID_VOL
            if value >= high_enter_pct:
                current = "HIGH_VOL"
            elif value <= low_enter_pct:
                current = "LOW_VOL"
        out.append(current)
    return pd.Series(out, index=atr_pct_rank.index, name="raw_vol_state")


def _stabilize_labels(
    raw_labels: pd.Series,
    min_regime_bars: int = 12,
    confirm_bars: int = 6,
) -> pd.Series:
    if min_regime_bars < 1 or confirm_bars < 1:
        raise ValueError("min_regime_bars and confirm_bars must be >= 1")

    labels = raw_labels.fillna(method="ffill").fillna("UNKNOWN").astype(str)
    stable = []

    current = labels.iloc[0]
    bars_in_current = 1
    candidate = None
    candidate_count = 0
    stable.append(current)

    for label in labels.iloc[1:]:
        if label == current:
            bars_in_current += 1
            candidate = None
            candidate_count = 0
            stable.append(current)
            continue

        if bars_in_current < min_regime_bars:
            bars_in_current += 1
            stable.append(current)
            continue

        if candidate == label:
            candidate_count += 1
        else:
            candidate = label
            candidate_count = 1

        if candidate_count >= confirm_bars:
            current = candidate
            bars_in_current = 1
            candidate = None
            candidate_count = 0
            stable.append(current)
        else:
            bars_in_current += 1
            stable.append(current)

    return pd.Series(stable, index=labels.index, name="stable_regime_label")


def build_stable_regime_series(
    raw_regime_labels: pd.Series,
    adx: pd.Series,
    atr_normalized: pd.Series,
    atr_norm_pct_rank: pd.Series | None = None,
    enter_trending: float = 28.0,
    exit_trending: float = 22.0,
    min_regime_bars: int = 12,
    confirm_bars: int = 6,
    vol_window: int = 252,
) -> pd.Series:
    """Create a stable regime series using hysteresis and persistence rules."""
    idx = raw_regime_labels.index
    adx = adx.reindex(idx)
    atr_norm = atr_normalized.reindex(idx)
    if atr_norm_pct_rank is None:
        atr_norm_pct_rank = _rolling_percentile_rank(atr_norm, window=vol_window)
    else:
        atr_norm_pct_rank = atr_norm_pct_rank.reindex(idx)

    raw_trend = _trend_state_with_hysteresis(
        adx,
        enter_trending=enter_trending,
        exit_trending=exit_trending,
    )
    raw_vol = _vol_state_with_hysteresis(atr_norm_pct_rank)
    raw_hysteresis = raw_trend + "_" + raw_vol

    # Preserve unknown labels if provided by upstream logic.
    raw = raw_regime_labels.reindex(idx).fillna(raw_hysteresis)
    raw = raw.astype(str)
    raw = raw.where(raw.str.contains("_"), raw_hysteresis)

    return _stabilize_labels(
        raw,
        min_regime_bars=min_regime_bars,
        confirm_bars=confirm_bars,
    )


def attach_stable_regime_state(
    df: pd.DataFrame,
    raw_regime_col: str = "regime_label",
    adx_col: str = "adx_14",
    atr_norm_col: str = "atr_norm",
    atr_norm_pct_col: str = "atr_norm_pct_rank",
    enter_trending: float = 28.0,
    exit_trending: float = 22.0,
    min_regime_bars: int = 12,
    confirm_bars: int = 6,
    vol_window: int = 252,
) -> pd.DataFrame:
    required = {raw_regime_col, adx_col, atr_norm_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for stable regime state: {sorted(missing)}")

    out = df.copy()
    atr_rank = out[atr_norm_pct_col] if atr_norm_pct_col in out.columns else None
    out["stable_regime_label"] = build_stable_regime_series(
        raw_regime_labels=out[raw_regime_col],
        adx=out[adx_col],
        atr_normalized=out[atr_norm_col],
        atr_norm_pct_rank=atr_rank,
        enter_trending=enter_trending,
        exit_trending=exit_trending,
        min_regime_bars=min_regime_bars,
        confirm_bars=confirm_bars,
        vol_window=vol_window,
    )
    out["stable_trend_regime"] = out["stable_regime_label"].str.split("_").str[0]
    out["stable_vol_regime"] = out["stable_regime_label"].str.split("_").str[1:].str.join("_")
    return out


def regime_duration_distribution(regime_series: pd.Series) -> pd.DataFrame:
    """Return contiguous regime run lengths with summary-friendly fields."""
    labels = regime_series.fillna("UNKNOWN").astype(str)
    if labels.empty:
        return pd.DataFrame(columns=["Regime", "RunLength"])

    run_id = (labels != labels.shift(1)).cumsum()
    runs = (
        pd.DataFrame({"Regime": labels, "RunId": run_id})
        .groupby(["RunId", "Regime"], as_index=False)
        .size()
        .rename(columns={"size": "RunLength"})
    )
    return runs[["Regime", "RunLength"]]


def duration_distribution_json(regime_series: pd.Series) -> str:
    runs = regime_duration_distribution(regime_series)
    payload = (
        runs.groupby("Regime")["RunLength"]
        .apply(lambda s: sorted([int(x) for x in s.tolist()]))
        .to_dict()
    )
    return json.dumps(payload, sort_keys=True)
