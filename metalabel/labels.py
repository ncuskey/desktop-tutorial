from __future__ import annotations

import numpy as np
import pandas as pd


def entry_mask_from_signal(signal: pd.Series) -> pd.Series:
    s = signal.fillna(0.0).astype(float)
    change = (s != s.shift(1)).fillna(True)
    return (change & (s != 0.0)).astype(bool)


def resolve_signal(df: pd.DataFrame, signal_col: str | pd.Series) -> pd.Series:
    if isinstance(signal_col, pd.Series):
        return signal_col.reindex(df.index).fillna(0.0).astype(float)
    if signal_col not in df.columns:
        raise ValueError(f"Signal column not found: {signal_col}")
    return pd.to_numeric(df[signal_col], errors="coerce").fillna(0.0).astype(float)


def infer_filter_type_from_regime(df: pd.DataFrame) -> pd.Series:
    if "stable_trend_regime" in df.columns:
        trend_reg = df["stable_trend_regime"].astype(str).str.upper()
        return pd.Series(
            np.where(trend_reg.str.contains("TREND"), "trend", "mean_reversion"),
            index=df.index,
            dtype=str,
        )
    if "stable_regime_label" in df.columns:
        label = df["stable_regime_label"].astype(str).str.upper()
        return pd.Series(
            np.where(label.str.contains("TREND"), "trend", "mean_reversion"),
            index=df.index,
            dtype=str,
        )
    return pd.Series("global", index=df.index, dtype=str)


def compute_forward_trade_returns(
    df: pd.DataFrame,
    signal: pd.Series,
    forward_horizon: int,
) -> pd.Series:
    close = pd.to_numeric(df["close"], errors="coerce")
    side = np.sign(signal.reindex(df.index).fillna(0.0).astype(float))
    forward_price_return = close.shift(-forward_horizon) / close - 1.0
    return side * forward_price_return


def create_trade_success_labels(
    df: pd.DataFrame,
    signal_col: str | pd.Series,
    entry_mask: pd.Series | None = None,
    forward_horizon: int = 24,
    method: str = "top_quantile",
    quantile: float = 0.3,
    cost_threshold: float | None = None,
    horizon_bars: int | None = None,  # backward-compatible alias
    success_threshold: float | None = None,  # backward-compatible alias for cost threshold
) -> pd.Series:
    """
    Create trade-success labels aligned to df.index.

    Modes:
      - directional_accuracy
      - cost_adjusted_return
      - top_quantile (default)
    """
    if horizon_bars is not None:
        forward_horizon = int(horizon_bars)
    if success_threshold is not None and cost_threshold is None:
        cost_threshold = float(success_threshold)
    if forward_horizon <= 0:
        raise ValueError("forward_horizon must be > 0")

    signal = resolve_signal(df, signal_col)
    side = np.sign(signal)
    if entry_mask is None:
        entry_mask = entry_mask_from_signal(signal)
    entry_mask = entry_mask.reindex(df.index).fillna(False).astype(bool)

    close = pd.to_numeric(df["close"], errors="coerce")
    forward_price_return = close.shift(-forward_horizon) / close - 1.0
    forward_trade_return = side * forward_price_return
    valid = entry_mask & (side != 0.0) & forward_trade_return.notna()

    labels = pd.Series(np.nan, index=df.index, dtype=float)
    method_n = str(method).lower()

    if method_n == "directional_accuracy":
        direction_ok = np.sign(forward_price_return) == np.sign(side)
        labels.loc[valid] = direction_ok.loc[valid].astype(int).astype(float)
    elif method_n == "cost_adjusted_return":
        if cost_threshold is None:
            spread_bps = pd.to_numeric(df.get("spread_bps", 0.0), errors="coerce").fillna(0.0)
            slippage_bps = pd.to_numeric(df.get("slippage_bps", 0.0), errors="coerce").fillna(0.0)
            commission_bps = pd.to_numeric(df.get("commission_bps", 0.0), errors="coerce").fillna(0.0)
            threshold_series = 2.0 * (spread_bps + slippage_bps + commission_bps) / 10_000.0
        else:
            threshold_series = pd.Series(float(cost_threshold), index=df.index)
        labels.loc[valid] = (
            forward_trade_return.loc[valid] > threshold_series.loc[valid]
        ).astype(int).astype(float)
    elif method_n == "top_quantile":
        q = float(np.clip(quantile, 0.05, 0.95))
        returns = forward_trade_return.loc[valid]
        if returns.empty:
            return labels
        filter_type = infer_filter_type_from_regime(df).astype(str)
        grouped_idx = filter_type.loc[valid].groupby(filter_type.loc[valid]).groups
        for _, idx in grouped_idx.items():
            grp_ret = returns.loc[idx]
            if len(grp_ret) < 8:
                continue
            cutoff = float(grp_ret.quantile(1.0 - q))
            labels.loc[idx] = (grp_ret >= cutoff).astype(int).astype(float)
        remaining = valid & labels.isna()
        if remaining.any():
            rem_ret = returns.loc[remaining]
            cutoff = float(rem_ret.quantile(1.0 - q))
            labels.loc[remaining] = (rem_ret >= cutoff).astype(int).astype(float)
    else:
        raise ValueError(
            f"Unsupported label method: {method}. Use directional_accuracy, "
            "cost_adjusted_return, or top_quantile."
        )

    return labels
