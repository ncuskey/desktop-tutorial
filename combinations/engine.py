from __future__ import annotations

import numpy as np
import pandas as pd


def confirmation_signals(signals: dict[str, pd.Series], min_agree: int = 2) -> pd.Series:
    if len(signals) < min_agree:
        raise ValueError("Number of signals must be >= min_agree")
    stacked = pd.concat(signals, axis=1).fillna(0)
    longs = (stacked > 0).sum(axis=1)
    shorts = (stacked < 0).sum(axis=1)
    out = pd.Series(0, index=stacked.index, dtype=int)
    out[longs >= min_agree] = 1
    out[shorts >= min_agree] = -1
    return out


def weighted_ensemble_signals(
    signals: dict[str, pd.Series],
    weights: dict[str, float] | None = None,
    threshold: float = 0.0,
) -> pd.Series:
    if not signals:
        raise ValueError("signals cannot be empty")
    if weights is None:
        equal = 1.0 / len(signals)
        weights = {k: equal for k in signals}

    stacked = pd.concat(signals, axis=1).fillna(0)
    weight_vec = np.array([weights.get(c, 0.0) for c in stacked.columns])
    combined = stacked.to_numpy() @ weight_vec
    out = pd.Series(0, index=stacked.index, dtype=int)
    out[combined > threshold] = 1
    out[combined < -threshold] = -1
    return out


def specialist_sleeve(signal: pd.Series, activation_mask: pd.Series) -> pd.Series:
    aligned_mask = activation_mask.reindex(signal.index).fillna(False).astype(bool)
    out = signal.copy().astype(int)
    out[~aligned_mask] = 0
    return out


def combine_specialist_sleeves(
    strategy_outputs: dict[str, pd.Series],
    regime_series: pd.Series,
    regime_to_sleeve: dict[str, str],
    fallback: str = "flat",
    default_strategy: str | None = None,
) -> pd.Series:
    """Route specialist sleeve signals based on active regime.

    Supported fallback modes:
      - flat: hold 0 where regime has no sleeve mapping
      - previous_position: carry prior combined position
      - default_strategy: use `default_strategy` signal when unmapped
    """
    if not strategy_outputs:
        raise ValueError("strategy_outputs cannot be empty")
    if fallback not in {"flat", "previous_position", "default_strategy"}:
        raise ValueError("fallback must be flat, previous_position, or default_strategy")

    index = regime_series.index
    sleeve_df = pd.DataFrame(
        {name: sig.reindex(index).fillna(0.0) for name, sig in strategy_outputs.items()},
        index=index,
    )
    selected = regime_series.reindex(index).map(regime_to_sleeve)
    out = pd.Series(np.nan, index=index, dtype=float)

    for sleeve_name in selected.dropna().unique():
        if sleeve_name not in sleeve_df.columns:
            continue
        mask = selected == sleeve_name
        out.loc[mask] = sleeve_df.loc[mask, sleeve_name]

    unresolved = out.isna()
    if fallback == "flat":
        out.loc[unresolved] = 0.0
    elif fallback == "default_strategy":
        if default_strategy is None or default_strategy not in sleeve_df.columns:
            raise ValueError(
                "default_strategy fallback requires default_strategy present in strategy_outputs"
            )
        out.loc[unresolved] = sleeve_df.loc[unresolved, default_strategy]
    else:  # previous_position
        out.loc[unresolved] = np.nan
        out = out.ffill().fillna(0.0)

    return out.fillna(0.0).astype(float)
